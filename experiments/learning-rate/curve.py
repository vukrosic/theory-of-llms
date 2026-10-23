"""Loss curve per lr on one dataset (one pass, no repeats), same model + Muon step as lr.train.
usage: STEPS=600 python3 curve.py LR SEED   -> appends to curves{STEPS}_T{T}.jsonl (eval loss every 30 steps)
       STEPS=600 python3 curve.py plot      -> lr-curves-{STEPS}_T{T}.png  (T=16 halves text length)
"""
import sys, json, torch
import lr as L
import os
torch.set_num_threads(1); EVERY = 30; OUT = f"{L.HERE}/curves{L.STEPS}_T{L.T}.jsonl"


def run(lr, seed):
    fn = f"{L.HERE}/data{L.SUF}/base.pt"
    if L.T == 32 and os.path.exists(fn): d = torch.load(fn); tr, ev = d["train"], d["eval"]
    else: b = L.datasets()["base"]; tr, ev = L.make(b, L.NTR, 1), L.make(b, L.NEV, 2)     # other lengths: generate fresh
    torch.manual_seed(seed); m = L.LM(); mats, rest = L.split(m)
    buf = [torch.zeros_like(p) for p in mats]; ex = [torch.zeros_like(p) for p in rest]; ex2 = [torch.zeros_like(p) for p in rest]
    order = torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed)); curve = []
    for i in range(L.STEPS + 1):
        if i % EVERY == 0 or i == L.STEPS:
            with torch.no_grad(): curve.append((i, m.loss(ev).item()))
        if i == L.STEPS: break
        f = min(1.0, (i + 1) / 20) * min(1.0, (1 - i / L.STEPS) / 0.5)
        loss = m.loss(tr[order[i * L.B:(i + 1) * L.B]]); m.zero_grad(set_to_none=True); loss.backward()
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
    with open(OUT, "a") as fo: fo.write(json.dumps(dict(lr=lr, seed=seed, curve=curve)) + "\n")
    print(f"lr {lr} seed {seed}: final {curve[-1][1]:.4f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] != "plot": run(float(sys.argv[1]), int(sys.argv[2])); sys.exit()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    runs = {}
    for l in open(OUT): r = json.loads(l); runs.setdefault(r["lr"], []).append(r["curve"])
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150); cm = plt.get_cmap("viridis")
    for k, lr in enumerate(sorted(runs)):
        cs = runs[lr]; s = [p[0] for p in cs[0]]; y = [sum(c[j][1] for c in cs) / len(cs) for j in range(len(s))]
        ax.plot(s, y, color=cm(k / (len(runs) - 1)), lw=2.2, label=f"learning rate {lr:g}  (ends {y[-1]:.3f})")
    ends = [sum(c[-1][1] for c in runs[lr]) / len(runs[lr]) for lr in runs]
    ax.set_ylim(min(ends) - 0.05, max(ends) + 0.4); ax.set_xlim(0, L.STEPS)
    ax.set_xlabel("training steps", fontsize=12); ax.set_ylabel("loss (lower is better)", fontsize=12)
    ax.set_title("Different learning rates, same end point", fontsize=14, weight="bold")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=10)
    fig.tight_layout(); fn = f"{L.HERE}/lr-curves-{L.STEPS}_T{L.T}.png"; fig.savefig(fn); print("saved", fn)
