"""Best Muon weight decay vs number of epochs (data repeated E times, total steps fixed).
usage: B=8 D=16 STEPS=800 python3 run.py E LR SEED WD1,WD2,...   -> appends to runs.jsonl (heldout loss, pool loss)
Copy of what-changes-muon-lr/scale.py train loop; the only change: batches come from a pool of STEPS*B/E unique sequences (E passes, new shuffle each pass).
"""
import sys, os, json, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = HERE + "/../what-changes-muon-lr"
sys.path.insert(0, SRC); os.chdir(SRC)
os.environ.setdefault("B", "8"); os.environ.setdefault("D", "16"); os.environ.setdefault("STEPS", "800")
import endpoint as E_, mixprobe as M
L = E_.L; torch.set_num_threads(1)
WDALL = int(os.environ.get("WDALL", 0))   # 1: also decay embedding + head (2-D non-Muon params) with the same wd, step OTHER_LR
MOM, WARM, DFRAC = 0.95, 20, 0.5; STEPS = E_.STEPS; OUT = HERE + "/" + os.environ.get("OUT", "runs.jsonl")


def train(pool, ev, mask, lr, wd, seed, steps):
    torch.manual_seed(seed); m = L.LM(); mats, rest = L.split(m); n = len(pool)
    buf = [torch.zeros_like(p) for p in mats]; ex = [torch.zeros_like(p) for p in rest]; ex2 = [torch.zeros_like(p) for p in rest]
    g = torch.Generator().manual_seed(seed); order = torch.cat([torch.randperm(n, generator=g) for _ in range(steps * L.B // n + 1)])
    for i in range(steps):
        f = min(1.0, (i + 1) / WARM) * min(1.0, (1 - i / steps) / DFRAC)
        loss = m.loss(pool[order[i * L.B:(i + 1) * L.B]]); m.zero_grad(set_to_none=True); loss.backward()
        with torch.no_grad():
            u = []
            for p, b in zip(mats, buf): b.mul_(MOM).add_(p.grad); g_ = p.grad + MOM * b; u.append(g_.T if g_.shape[0] > g_.shape[1] else g_)
            for shp in {x.shape for x in u}:
                ix = [j for j, x in enumerate(u) if x.shape == shp]; O = L.ns(torch.stack([u[j] for j in ix]))
                for j, o in zip(ix, O):
                    p = mats[j]
                    if wd: p.mul_(1 - lr * f * wd)
                    p.add_(o.T if p.shape[0] > p.shape[1] else o, alpha=-lr * f * max(1, p.shape[0] / p.shape[1]) ** 0.5)
            for p, e, e2 in zip(rest, ex, ex2):
                if WDALL and wd and p.ndim == 2: p.mul_(1 - L.OTHER_LR * f * wd)
                e.mul_(0.9).add_(p.grad, alpha=0.1); e2.mul_(0.98).add_(p.grad ** 2, alpha=0.02)
                p.add_(e / (1 - 0.9 ** (i + 1)) / ((e2 / (1 - 0.98 ** (i + 1))).sqrt() + 1e-8), alpha=-L.OTHER_LR * f)
    with torch.no_grad(): trl = m.loss(pool[:512]).item()
    ls = E_.evaluate(m, ev, mask); return sum(ls.values()) / len(ls), trl


if __name__ == "__main__":
    ep, lr, seed, wds = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), [float(x) for x in sys.argv[4].split(",")]
    tr, ev, mask = M.data(E_.TYPES, STEPS); pool = tr[:max(1, STEPS * L.B // ep)]
    for wd in wds:
        t0 = time.time(); l, trl = train(pool, ev, mask, lr, wd, seed, STEPS)
        with open(OUT, "a") as fo: fo.write(json.dumps(dict(D=L.D, B=L.B, steps=STEPS, E=ep, lr=lr, wd=wd, seed=seed, loss=round(l, 5), pool=round(trl, 5))) + "\n")
        print(f"D {L.D} E {ep} lr {lr} wd {wd} seed {seed}: heldout {l:.4f} pool {trl:.4f} ({time.time() - t0:.1f}s)", flush=True)
