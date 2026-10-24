"""n=2 features, m=1 dim (paper's phase diagram), Muon on W (1x2), 3 seeds; prints state per (r, density)
state: 0 = only feature 1 stored, 1 = only feature 2 stored, 2 = both, superposed antipodally (signs opposite)"""
import json, os, sys, math
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tms import msign
def run(r, dens, seed, lr=0.03, steps=2000, bs=1024):
    torch.manual_seed(seed); W = (torch.randn(1, 2)).requires_grad_(); b = torch.zeros(2, requires_grad=True)
    I = torch.tensor([1.0, r]); vec = torch.optim.Adam([b], lr=0.01); buf = torch.zeros_like(W); g = torch.Generator().manual_seed(seed)
    for i in range(steps):
        f = min(1, (i + 1) / 100) * min(1, 2 * (steps - i) / steps)
        x = torch.rand(bs, 2, generator=g) * (torch.rand(bs, 2, generator=g) < dens)
        loss = (I * (x - torch.relu(x @ W.T @ W + b)) ** 2).sum(1).mean()
        vec.zero_grad(); W.grad = None; loss.backward()
        for gr in vec.param_groups: gr["lr"] = 0.01 * f
        vec.step()
        with torch.no_grad(): buf.mul_(0.95).add_(W.grad); W.add_(msign(W.grad + 0.95 * buf), alpha=-lr * f)
    w = W.detach()[0]; st = 2 if (w.abs() > 0.5).all() and w[0] * w[1] < 0 else (0 if w[0].abs() > w[1].abs() else 1)
    return st, w.tolist()
rs, dens = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 3.0], [1.0, 0.5, 0.3, 0.1, 0.03, 0.01]
out = {}
for r in rs:
    for d in dens:
        out[f"{r},{d}"] = [run(r, d, s, steps=1500)[0] for s in range(3)]
        print(r, d, out[f"{r},{d}"], flush=True)
json.dump(dict(rs=rs, dens=dens, states=out), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase.json"), "w"))
