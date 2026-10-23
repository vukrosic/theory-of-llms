"""Does a cheap probe on a data mix predict Muon's best lr for it?
usage: B=8 D=16 STEPS=800 python3 mixprobe.py sweep MIX SEED   -> mixprobe_runs.jsonl (final eval loss per lr, scale.train, decay on)
       B=8 D=16 python3 mixprobe.py probe MIX SEED              -> mixprobe_probes.jsonl (probes at init and after 50 steps)
       python3 mixprobe.py fit                                  -> best lr per mix, probes, spearman, log-log fit, leave-one-out error
Probes: gradient noise scale B_simple = B * tr(Sigma_batch) / |G|^2 and mean pairwise cosine of 16 batch gradients
(Muon matrices only), at init and after 50 steps at lr 0.014 (constant after warmup 20); eval-loss drop over those 50 steps.
Mixes use endpoint.py's generators (same seeds per type); eval = mean over the mix's types (answer tokens for algorithmic types).
"""
import sys, os, json, math, itertools, torch
import endpoint as E
import scale as S
import ladder as Ld
L = E.L; HERE = L.HERE
MIXES = {"chain1": ["base"], "chain4": ["base", "s2", "beta8", "d0.25"], "mix20": E.TYPES, "sort": ["sort"],
         "highent": ["base", "s0", "p0.5"]}
NB, PSTEPS, PLR = 16, 50, 0.014


def data(types, steps):                                                 # endpoint.data restricted to `types`
    ds, ntr = L.datasets(), steps * L.B // len(types) + 1
    tr, ev, mask = [], {}, {}
    for k in types:
        i = E.TYPES.index(k)
        if k in E.ALGO:
            f, mask[k] = E.ALGO[k]; tr.append(f(ntr, torch.Generator().manual_seed(100 + i))); ev[k] = f(E.NEV, torch.Generator().manual_seed(200 + i))
        else:
            tr.append(L.make(ds[k], ntr, 1)); ev[k] = L.make(ds[k], E.NEV, 2); mask[k] = torch.ones(L.T, dtype=torch.bool)
    tr = torch.cat(tr); return tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(0))], ev, mask


def grads(m, mats, xs):                                                 # flattened Muon-matrix gradient per batch
    out = []
    for x in xs:
        m.zero_grad(set_to_none=True); m.loss(x).backward(); out.append(torch.cat([p.grad.flatten() for p in mats]))
    return torch.stack(out)


def noise(g):
    n = len(g); G = g.mean(0); tr_s = ((g - G) ** 2).sum() / (n - 1); G2 = (G ** 2).sum() - tr_s / n   # unbiased |G|^2
    u = torch.nn.functional.normalize(g, dim=1); c = u @ u.T
    return dict(bnoise=(L.B * tr_s / G2).item() if G2 > 0 else float("inf"), cos=((c.sum() - n) / (n * (n - 1))).item())


def probe(tr, ev, mask, seed):
    torch.manual_seed(seed); m = L.LM(); mats, rest = L.split(m); g = torch.Generator().manual_seed(1000 + seed)
    batch = lambda: tr[torch.randint(len(tr), (L.B,), generator=g)]
    ev_loss = lambda: (lambda ls: sum(ls.values()) / len(ls))(E.evaluate(m, ev, mask))
    out = {f"{k}0": v for k, v in noise(grads(m, mats, [batch() for _ in range(NB)])).items()}; l0 = ev_loss()
    buf = [torch.zeros_like(p) for p in mats]; ex = [torch.zeros_like(p) for p in rest]; ex2 = [torch.zeros_like(p) for p in rest]
    for i in range(PSTEPS):                                             # scale.train's update, constant lr after warmup 20
        f = min(1.0, (i + 1) / 20); loss = m.loss(batch()); m.zero_grad(set_to_none=True); loss.backward()
        with torch.no_grad():
            u = []
            for p, b in zip(mats, buf): b.mul_(0.95).add_(p.grad); g_ = p.grad + 0.95 * b; u.append(g_.T if g_.shape[0] > g_.shape[1] else g_)
            for shp in {x.shape for x in u}:
                ix = [j for j, x in enumerate(u) if x.shape == shp]; O = L.ns(torch.stack([u[j] for j in ix]))
                for j, o in zip(ix, O):
                    p = mats[j]; p.add_(o.T if p.shape[0] > p.shape[1] else o, alpha=-PLR * f * max(1, p.shape[0] / p.shape[1]) ** 0.5)
            for p, e, e2 in zip(rest, ex, ex2):
                e.mul_(0.9).add_(p.grad, alpha=0.1); e2.mul_(0.98).add_(p.grad ** 2, alpha=0.02)
                p.add_(e / (1 - 0.9 ** (i + 1)) / ((e2 / (1 - 0.98 ** (i + 1))).sqrt() + 1e-8), alpha=-L.OTHER_LR * f)
    l50 = ev_loss(); out.update({f"{k}50": v for k, v in noise(grads(m, mats, [batch() for _ in range(NB)])).items()})
    out.update(drop=l0 - l50, reldrop=(l0 - l50) / l0); return out


def rank(v): o = sorted(range(len(v)), key=v.__getitem__); r = [0] * len(v); [r.__setitem__(j, i) for i, j in enumerate(o)]; return r
def spearman(a, b): ra, rb = rank(a), rank(b); n = len(a); return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))


def best7(d):                                                           # parabola in log lr through the 7 lrs around the min
    import numpy as np
    lrs = np.array(sorted(d)); Ls = np.array([sum(d[x]) / len(d[x]) for x in lrs]); i = int(Ls.argmin())
    lo = max(0, min(i - 3, len(lrs) - 7)); a, b, _ = np.polyfit(np.log(lrs[lo:lo + 7]), Ls[lo:lo + 7], 2)
    return float(np.exp(-b / (2 * a))) if a > 0 else float(lrs[i])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd in ("sweep", "probe"):
        mix, seed = sys.argv[2], int(sys.argv[3]); tr, ev, mask = data(MIXES[mix], E.STEPS if cmd == "sweep" else 200)
        if cmd == "probe":
            r = dict(mix=mix, seed=seed, B=L.B, D=L.D, **probe(tr, ev, mask, seed)); print(r)
            with open(f"{HERE}/mixprobe_probes.jsonl", "a") as fo: fo.write(json.dumps(r) + "\n")
            sys.exit()
        for lr in Ld.LRS:
            loss = S.train(tr, ev, mask, lr, seed, E.STEPS)
            with open(f"{HERE}/mixprobe_runs.jsonl", "a") as fo:
                fo.write(json.dumps(dict(mix=mix, B=L.B, D=L.D, steps=E.STEPS, lr=lr, seed=seed, loss=round(loss, 5))) + "\n")
            print(f"{mix} lr {lr} seed {seed}: {loss:.4f}", flush=True)
        sys.exit()
    runs, pr = {}, {}
    for l in open(f"{HERE}/mixprobe_runs.jsonl"): r = json.loads(l); runs.setdefault(r["mix"], {}).setdefault(r["lr"], []).append(r["loss"])
    for l in open(f"{HERE}/mixprobe_probes.jsonl"):
        r = json.loads(l); pr.setdefault(r["mix"], []).append(r)
    mixes = [k for k in MIXES if k in runs and k in pr]; keys = ["bnoise0", "cos0", "bnoise50", "cos50", "drop", "reldrop"]
    P = {k: {mx: sum(r[k] for r in pr[mx]) / len(pr[mx]) for mx in mixes} for k in keys}
    best = {mx: best7(runs[mx]) for mx in mixes}; band = {mx: Ld.band(runs[mx]) for mx in mixes}
    print(f"{'mix':8} {'best lr':>8} {'band(+.005)':>14} {'seeds':>5} " + " ".join(f"{k:>9}" for k in keys))
    for mx in mixes:
        print(f"{mx:8} {best[mx]:8.4f} {band[mx][0]:>6}-{band[mx][1]:<7} {min(len(v) for v in runs[mx].values()):5} "
              + " ".join(f"{P[k][mx]:9.4g}" for k in keys))
    y = [math.log(best[mx]) for mx in mixes]; bw = sum(math.log(band[mx][1] / band[mx][0]) for mx in mixes) / len(mixes)
    loo_mean = [abs(y[j] - sum(y[:j] + y[j + 1:]) / (len(y) - 1)) for j in range(len(y))]
    print(f"mean band width {math.exp(bw):.2f}x; LOO error of 'predict the mean': {math.exp(sum(loo_mean) / len(y)):.2f}x")
    for k in keys:
        x = [math.log(abs(P[k][mx])) for mx in mixes]; sl = S.slope(x, y); err = []
        for j in range(len(y)):
            xs, ys = x[:j] + x[j + 1:], y[:j] + y[j + 1:]; s = S.slope(xs, ys); c = sum(ys) / len(ys) - s * sum(xs) / len(xs)
            err.append(abs(y[j] - (c + s * x[j])))
        print(f"{k:9} spearman {spearman([P[k][mx] for mx in mixes], [best[mx] for mx in mixes]):+.2f}  best lr ~ |probe|^{sl:+.2f}  "
              f"LOO error {math.exp(sum(err) / len(err)):.2f}x")
