"""Do Muon runs at different lrs end at the same loss, and learn the same predictions, per data type?
usage: STEPS=3000 python3 endpoint.py train LR [SEED] -> ep/STEPS_LR_sSEED.jsonl (eval loss per type every 100 steps)
                                                      + ep/STEPS_LR_sSEED.pt (final argmax predictions per eval set)
       STEPS=3000 python3 endpoint.py report          -> per-type table (seed 0; seed-1 run at lr 0.02 = same-lr noise floor)
One model per lr trains on the shuffled union of ~20 data types in equal shares (one pass, no repeats).
Model + Muon step copied from lr.py (imported, not edited). Algorithmic types are scored only on their answer positions.
"""
import sys, os, json, glob, time
os.environ.setdefault("T", "32")
import torch, torch.nn.functional as F
import lr as L
torch.set_num_threads(1)
T, B, V, HERE = L.T, L.B, L.V, L.HERE
STEPS = int(os.environ.get("STEPS", 3000)); NEV, EVERY = 256, 100
OUT = f"{HERE}/ep"; os.makedirs(OUT, exist_ok=True)
CHAIN = ["base", "s0", "s2", "beta0.5", "beta8", "p0.1", "p0.5", "d0.25", "d0.9", "c0.5", "c1.0",
         "mix0", "mix1", "mix2", "mix3", "mix4"]
ANS = torch.zeros(T, dtype=torch.bool); ANS[T // 2:] = True              # target positions 16..31 = answer half


def sort_(n, g):   # 16 random tokens, separator, the same tokens sorted
    a = torch.randint(200, (n, T // 2), generator=g); return torch.cat([a, torch.full((n, 1), 255), a.sort(1).values], 1)
def reverse_(n, g):  # 16 random tokens, separator, the same tokens reversed
    a = torch.randint(200, (n, T // 2), generator=g); return torch.cat([a, torch.full((n, 1), 254), a.flip(1)], 1)
def modadd_(n, g):  # T/4 problems "a b = (a+b)%97", then a pad token (8 at T=32)
    a, b = torch.randint(97, (n, T // 4), generator=g), torch.randint(97, (n, T // 4), generator=g)
    x = torch.stack([a, b, torch.full_like(a, 253), (a + b) % 97], 2).flatten(1); return torch.cat([x, torch.full((n, 1), 252)], 1)
def count_(n, g):  # start token, then T/2 x (bit, running count of 1-bits so far) (16 at T=32)
    bits = torch.randint(2, (n, T // 2), generator=g)
    x = torch.stack([200 + bits, 210 + bits.cumsum(1)], 2).flatten(1); return torch.cat([torch.full((n, 1), 230), x], 1)
ALGO = dict(sort=(sort_, ANS), reverse=(reverse_, ANS), modadd=(modadd_, torch.arange(T) % 4 == 2),
            count=(count_, torch.arange(T) % 2 == 1))
TYPES = CHAIN + list(ALGO)


def data():
    ds, ntr = L.datasets(), STEPS * B // len(TYPES) + 1
    tr, ev, mask = [], {}, {}
    for i, k in enumerate(TYPES):
        if k in ALGO:
            f, mask[k] = ALGO[k]; tr.append(f(ntr, torch.Generator().manual_seed(100 + i))); ev[k] = f(NEV, torch.Generator().manual_seed(200 + i))
        else:
            tr.append(L.make(ds[k], ntr, 1)); ev[k] = L.make(ds[k], NEV, 2); mask[k] = torch.ones(T, dtype=torch.bool)
    tr = torch.cat(tr); return tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(0))], ev, mask


@torch.no_grad()
def evaluate(m, ev, mask, preds=False):
    out = {}
    for k, x in ev.items():
        lg = m(x[:, :-1]); y = x[:, 1:]; mk = mask[k]
        l = F.cross_entropy(lg[:, mk].reshape(-1, V), y[:, mk].reshape(-1)).item()
        out[k] = lg[:, mk].argmax(-1).to(torch.uint8) if preds else round(l, 5)
    return out


def train(lr, seed=0):
    tr, ev, mask = data(); torch.manual_seed(seed); m = L.LM(); mats, rest = L.split(m)
    buf = [torch.zeros_like(p) for p in mats]; ex = [torch.zeros_like(p) for p in rest]; ex2 = [torch.zeros_like(p) for p in rest]
    f_out = open(f"{OUT}/{STEPS}_{lr}_s{seed}.jsonl", "w"); t0 = time.time()
    for i in range(STEPS):
        if i % EVERY == 0:
            f_out.write(json.dumps(dict(lr=lr, step=i, loss=evaluate(m, ev, mask))) + "\n"); f_out.flush()
        f = min(1.0, (i + 1) / 20) * min(1.0, (1 - i / STEPS) / 0.5)        # warmup 20, linear decay over last 50%
        loss = m.loss(tr[i * B:(i + 1) * B]); m.zero_grad(set_to_none=True); loss.backward()
        with torch.no_grad():                                               # same update as lr.train
            u = []
            for p, b in zip(mats, buf): b.mul_(0.95).add_(p.grad); g_ = p.grad + 0.95 * b; u.append(g_.T if g_.shape[0] > g_.shape[1] else g_)
            for shp in {x.shape for x in u}:
                ix = [j for j, x in enumerate(u) if x.shape == shp]; O = L.ns(torch.stack([u[j] for j in ix]))
                for j, o in zip(ix, O):
                    p = mats[j]; p.add_(o.T if p.shape[0] > p.shape[1] else o, alpha=-lr * f * max(1, p.shape[0] / p.shape[1]) ** 0.5)
            for p, e, e2 in zip(rest, ex, ex2):
                e.mul_(0.9).add_(p.grad, alpha=0.1); e2.mul_(0.98).add_(p.grad ** 2, alpha=0.02)
                p.add_(e / (1 - 0.9 ** (i + 1)) / ((e2 / (1 - 0.98 ** (i + 1))).sqrt() + 1e-8), alpha=-L.OTHER_LR * f)
    f_out.write(json.dumps(dict(lr=lr, step=STEPS, loss=evaluate(m, ev, mask))) + "\n"); f_out.close()
    torch.save(evaluate(m, ev, mask, preds=True), f"{OUT}/{STEPS}_{lr}_s{seed}.pt")
    print(f"lr {lr}: {STEPS} steps in {time.time() - t0:.1f}s")


def report():
    runs = {}
    for fn in glob.glob(f"{OUT}/{STEPS}_*_s0.jsonl"):
        rows = [json.loads(l) for l in open(fn)]; runs[rows[0]["lr"]] = {r["step"]: r["loss"] for r in rows}
    lrs = sorted(runs); steps = sorted(set.intersection(*(set(runs[x]) for x in lrs))); last = steps[-1]
    P = {x: torch.load(f"{OUT}/{STEPS}_{x}_s0.pt") for x in lrs}
    nf = f"{OUT}/{STEPS}_0.02_s1.pt"; P1 = torch.load(nf) if os.path.exists(nf) else None
    agree = lambda a, b, k: 100 * (P[a][k] == P[b][k]).float().mean().item()
    hdr = f"{'type':8} " + " ".join(f"{'L@' + str(x):>8}" for x in lrs) + f" {'gap':>6} {'gap<.02 from':>12} {'agr .005/.08':>12} {'agr .02/.08':>11} {'agr .02 s0/s1':>13}"
    print(f"final step {last}\n{hdr}"); tot = {x: 0.0 for x in lrs}; ag1 = ag2 = 0.0
    for k in TYPES:
        fin = [runs[x][last][k] for x in lrs]; gap = max(fin) - min(fin)
        gaps = [max(runs[x][s][k] for x in lrs) - min(runs[x][s][k] for x in lrs) for s in steps]
        first = next((s for j, s in enumerate(steps) if all(g < 0.02 for g in gaps[j:])), None)
        a1, a2 = agree(lrs[0], lrs[-1], k), agree(lrs[1], lrs[-1], k); ag1 += a1 / len(TYPES); ag2 += a2 / len(TYPES)
        for x, v in zip(lrs, fin): tot[x] += v / len(TYPES)
        print(f"{k:8} " + " ".join(f"{v:8.3f}" for v in fin) + f" {gap:6.3f} {str(first) if first is not None else 'never':>12} {a1:12.1f} {a2:11.1f} " + (f"{100 * (P[0.02][k] == P1[k]).float().mean().item():13.1f}" if P1 else ""))
    t = [tot[x] for x in lrs]
    print(f"{'MEAN':8} " + " ".join(f"{v:8.3f}" for v in t) + f" {max(t) - min(t):6.3f} {'':>12} {ag1:12.1f} {ag2:11.1f}")


if __name__ == "__main__":
    if sys.argv[1] == "train": train(float(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    else: report()
