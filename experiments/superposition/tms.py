"""Toy Models of Superposition (Elhage et al. 2022), section 2-4 setup, trained with Muon.
n features, each 0 with chance S else uniform(0,1); importance 0.9^i; h = W x (m dims); out = ReLU(W^T h + b).
W (m x n) is the Muon matrix. OPT: ns = Muon (Newton-Schulz on the momentum), norm = same momentum and step size
without Newton-Schulz (scaled to the same Frobenius size), sgd = plain momentum.  b always uses the usual Muon recipe for vectors.

usage: python3 tms.py OPT S SEED LR [N M STEPS WD]  -> one json line (appended to runs.jsonl)
"""
import json, os, sys, time, math
import torch

def msign(g):
    a, b, c = 3.4445, -4.7750, 2.0315; t = g.shape[0] > g.shape[1]
    X = g.T if t else g; X = X / (X.norm() + 1e-7)
    for _ in range(5): A = X @ X.T; X = a * X + (b * A + c * A @ A) @ X
    return X.T if t else X

def run(opt, S, seed, lr, n=20, m=5, steps=2000, wd=0.0, bs=1024, imp=0.9):
    torch.manual_seed(seed)
    W = (torch.randn(m, n) / math.sqrt(m)).requires_grad_(); b = torch.zeros(n, requires_grad=True)
    I = imp ** torch.arange(n).float()
    vec = torch.optim.Adam([b], lr=0.01)
    buf = torch.zeros_like(W); g = torch.Generator().manual_seed(seed)
    def data(k, gg):
        x = torch.rand(k, n, generator=gg); return x * (torch.rand(k, n, generator=gg) >= S)
    for i in range(steps):
        f = min(1, (i + 1) / (0.05 * steps)) * min(1, 2 * (steps - i) / steps)
        x = data(bs, g); out = torch.relu(x @ W.T @ W + b)
        loss = (I * (x - out) ** 2).sum(1).mean()
        vec.zero_grad(); W.grad = None; loss.backward()
        for gr in vec.param_groups: gr["lr"] = 0.01 * f
        vec.step()
        with torch.no_grad():
            buf.mul_(0.95).add_(W.grad); u = W.grad + 0.95 * buf
            if opt == "ns": st = msign(u)
            elif opt == "norm": st = u / (u.norm() + 1e-7) * math.sqrt(min(m, n))
            else: st = u
            W.mul_(1 - lr * f * wd); W.add_(st, alpha=-lr * f)
    with torch.no_grad():
        x = data(20000, torch.Generator().manual_seed(999)); out = torch.relu(x @ W.T @ W + b)
        loss = (I * (x - out) ** 2).sum(1).mean().item()
        nrm = W.norm(dim=0); Wh = W / nrm.clamp_min(1e-6)
        inter = ((Wh.T @ W) ** 2).sum(1) - nrm ** 2              # how much other features leak onto feature i's direction
        dim = nrm ** 2 / (nrm ** 2 + inter).clamp_min(1e-9)      # paper's feature dimensionality D_i
        sv = torch.linalg.svdvals(W)
    return dict(opt=opt, S=S, seed=seed, lr=lr, n=n, m=m, steps=steps, wd=wd, loss=round(loss, 5),
                feats=int((nrm > 0.5).sum()), fro2=round((nrm ** 2).sum().item(), 3),
                dims=[round(v, 2) for v in dim.tolist()], norms=[round(v, 2) for v in nrm.tolist()],
                sv=[round(v, 3) for v in sv.tolist()], bias=[round(v, 2) for v in b.tolist()])

if __name__ == "__main__":
    a = sys.argv[1:]
    opt, S, seed, lr = a[0], float(a[1]), int(a[2]), float(a[3])
    n, m, steps = (int(a[4]), int(a[5]), int(a[6])) if len(a) > 6 else (20, 5, 2000)
    wd = float(a[7]) if len(a) > 7 else 0.0; imp = float(os.environ.get("IMP", 0.9))
    t0 = time.time(); r = run(opt, S, seed, lr, n, m, steps, wd, imp=imp); r["imp"] = imp; r["sec"] = round(time.time() - t0, 2)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.jsonl"), "a") as f: f.write(json.dumps(r) + "\n")
    print(json.dumps({k: r[k] for k in ("opt", "S", "seed", "lr", "loss", "feats", "fro2", "sec")}))
