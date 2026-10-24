"""Linear associative memory (Wang et al. 2509.26030 setup) pushed into superposition: K facts, width d, K can exceed d.
Fact i: key e_i (random unit vector in R^d), answer = class i; logits = U W e_i with U (K x d) fixed random unit rows.
Fact i is asked with chance ~ 1/(i+1)^Z (Zipf). W (d x d) trained. OPT ns = Muon, norm = same momentum without Newton-Schulz
(same Frobenius step size), sgd = plain momentum.
usage: python3 assoc.py OPT K D SEED LR [STEPS Z] -> json line appended to assoc_runs.jsonl
"""
import json, os, sys, time, math
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tms import msign

def run(opt, K, d, seed, lr, steps=300, Z=1.0, bs=256):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    E = F.normalize(torch.randn(K, d, generator=g), dim=1); U = F.normalize(torch.randn(K, d, generator=g), dim=1)
    p = 1 / torch.arange(1, K + 1).float() ** Z; p /= p.sum()
    W = torch.zeros(d, d, requires_grad=True); buf = torch.zeros(d, d)
    for i in range(steps):
        f = min(1, (i + 1) / 15) * min(1, 2 * (steps - i) / steps)
        j = torch.multinomial(p, bs, replacement=True, generator=g)
        loss = F.cross_entropy(E[j] @ W.T @ U.T, j); W.grad = None; loss.backward()
        with torch.no_grad():
            buf.mul_(0.95).add_(W.grad); u = W.grad + 0.95 * buf
            st = msign(u) if opt == "ns" else u / (u.norm() + 1e-9) * math.sqrt(d) if opt == "norm" else u
            W.add_(st, alpha=-lr * f)
    with torch.no_grad():
        acc = ((E @ W.T @ U.T).argmax(1) == torch.arange(K)).float()
        sv = torch.linalg.svdvals(W)
    q = lambda a, b: round(acc[int(a * K):max(int(a * K) + 1, int(b * K))].mean().item(), 3)
    mass = p.cumsum(0)
    return dict(opt=opt, K=K, d=d, seed=seed, lr=lr, steps=steps, Z=Z, acc=round(acc.mean().item(), 3),
                wacc=round((acc * p).sum().item(), 3), head=q(0, 0.1), mid=q(0.1, 0.5), tail=q(0.5, 1),
                sv_ratio=round((sv[0] / sv[-1].clamp_min(1e-9)).item(), 1), erank=round((sv.sum() ** 2 / (sv ** 2).sum()).item(), 1))

if __name__ == "__main__":
    a = sys.argv[1:]
    opt, K, d, seed, lr = a[0], int(a[1]), int(a[2]), int(a[3]), float(a[4])
    steps = int(a[5]) if len(a) > 5 else 300; Z = float(a[6]) if len(a) > 6 else 1.0
    t0 = time.time(); r = run(opt, K, d, seed, lr, steps, Z); r["sec"] = round(time.time() - t0, 2)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assoc_runs.jsonl"), "a") as f: f.write(json.dumps(r) + "\n")
    print(json.dumps(r))
