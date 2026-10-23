"""Does Muon's best lr fall with compute like DeepSeek's lr = 0.3118 * C^-0.125 (and with steps like T^-1/3 or T^-1/2)?
usage: D=64 STEPS=300 [DECAY=0] python3 scale.py SEED  -> appends (D, steps, lr, seed, final loss) to scale.jsonl, one process per (D, steps, seed)
       python3 scale.py fit                  -> best lr per (D, steps) + fitted slopes vs steps, params, compute
data: endpoint.py's mix of 20 types (16 token chains + sort/reverse/modadd/count), one pass, no repeats.
"""
import sys, os, json, math, time, torch
import endpoint as E
L = E.L; HERE = L.HERE; DECAY = int(os.environ.get("DECAY", 1))         # DECAY=0: constant lr after warmup
OUT = f"{HERE}/scale.jsonl" if DECAY else f"{HERE}/scale_nodecay.jsonl"
MOM = float(os.environ.get("MOM", 0.95)); WD = float(os.environ.get("WD", 0))   # knobs: Muon momentum, decoupled weight decay on Muon matrices
WARM = int(os.environ.get("WARM", 20)); DFRAC = float(os.environ.get("DFRAC", 0.5))  # warmup steps, share of the run spent decaying to 0
LRS = [0.0025, 0.0035, 0.005, 0.007, 0.01, 0.014, 0.02, 0.028, 0.04, 0.057, 0.08, 0.113, 0.16]


def train(tr, ev, mask, lr, seed, steps):
    torch.manual_seed(seed); m = L.LM(); mats, rest = L.split(m)
    buf = [torch.zeros_like(p) for p in mats]; ex = [torch.zeros_like(p) for p in rest]; ex2 = [torch.zeros_like(p) for p in rest]
    order = torch.randperm(steps * L.B, generator=torch.Generator().manual_seed(seed))
    for i in range(steps):
        f = (min(1.0, (i + 1) / WARM) if WARM else 1.0) * (min(1.0, (1 - i / steps) / DFRAC) if DECAY and DFRAC else 1.0)   # default: warmup 20, linear decay over last 50%
        loss = m.loss(tr[order[i * L.B:(i + 1) * L.B]]); m.zero_grad(set_to_none=True); loss.backward()
        with torch.no_grad():                                               # same update as lr.train
            u = []
            for p, b in zip(mats, buf): b.mul_(MOM).add_(p.grad); g_ = p.grad + MOM * b; u.append(g_.T if g_.shape[0] > g_.shape[1] else g_)
            for shp in {x.shape for x in u}:
                ix = [j for j, x in enumerate(u) if x.shape == shp]; O = L.ns(torch.stack([u[j] for j in ix]))
                for j, o in zip(ix, O):
                    p = mats[j]
                    if WD: p.mul_(1 - lr * f * WD)
                    p.add_(o.T if p.shape[0] > p.shape[1] else o, alpha=-lr * f * max(1, p.shape[0] / p.shape[1]) ** 0.5)
            for p, e, e2 in zip(rest, ex, ex2):
                e.mul_(0.9).add_(p.grad, alpha=0.1); e2.mul_(0.98).add_(p.grad ** 2, alpha=0.02)
                p.add_(e / (1 - 0.9 ** (i + 1)) / ((e2 / (1 - 0.98 ** (i + 1))).sqrt() + 1e-8), alpha=-L.OTHER_LR * f)
    ls = E.evaluate(m, ev, mask); return sum(ls.values()) / len(ls)


def best(d):                                                            # parabola through the 3 points around the min (log lr)
    lrs = sorted(d); Ls = [sum(d[x]) / len(d[x]) for x in lrs]; i = min(range(len(Ls)), key=Ls.__getitem__)
    if not 0 < i < len(Ls) - 1: return lrs[i], True
    x0, x1, x2 = (math.log(lrs[j]) for j in (i - 1, i, i + 1)); y0, y1, y2 = Ls[i - 1], Ls[i], Ls[i + 1]
    den = (x0 - x1) * (x0 - x2) * (x1 - x2); A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
    Bq = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) / den
    return (math.exp(-Bq / (2 * A)) if A > 0 else lrs[i]), False


def slope(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)


if __name__ == "__main__":
    if sys.argv[1] != "fit":
        seed = int(sys.argv[1]); tr, ev, mask = E.data(); m = L.LM()
        nonemb = sum(p.numel() for p in L.split(m)[0])
        for lr in LRS:
            t0 = time.time(); loss = train(tr, ev, mask, lr, seed, E.STEPS)
            with open(OUT, "a") as fo: fo.write(json.dumps(dict(D=L.D, steps=E.STEPS, lr=lr, seed=seed, loss=round(loss, 5), nonemb=nonemb)) + "\n")
            print(f"D {L.D} steps {E.STEPS} lr {lr} seed {seed}: {loss:.4f} ({time.time() - t0:.1f}s)", flush=True)
        sys.exit()
    runs, P = {}, {}
    for l in open(OUT): r = json.loads(l); k = (r["D"], r["steps"]); runs.setdefault(k, {}).setdefault(r["lr"], []).append(r["loss"]); P[k] = r["nonemb"]
    rows = []
    for (D, st), d in sorted(runs.items()):
        b, edge = best(d); C = 6 * P[(D, st)] * st * L.B * L.T; rows.append((D, st, b, C, edge))
        ends = {lr: sum(v) / len(v) for lr, v in sorted(d.items())}
        print(f"D {D:3} steps {st:4}  C {C:.2e}  best lr {b:.4f}{' (edge)' if edge else ''}  DeepSeek {0.3118 * C ** -0.125:.4f}  "
              f"loss at best {min(ends.values()):.4f}  seeds {min(len(v) for v in d.values())}")
    ok = [r for r in rows if not r[4]]
    for D in sorted({r[0] for r in ok}):
        g = [r for r in ok if r[0] == D]
        if len(g) > 1: print(f"width {D}: best lr ~ steps^{slope([math.log(r[1]) for r in g], [math.log(r[2]) for r in g]):+.2f}")
    for st in sorted({r[1] for r in ok}):
        g = [r for r in ok if r[1] == st]
        if len(g) > 1: print(f"steps {st}: best lr ~ width^{slope([math.log(r[0]) for r in g], [math.log(r[2]) for r in g]):+.2f}")
    if len(ok) > 1: print(f"all: best lr ~ compute^{slope([math.log(r[3]) for r in ok], [math.log(r[2]) for r in ok]):+.3f}  (DeepSeek -0.125)")
