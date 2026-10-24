"""Paper section 'superposition in computation': y = |x| for n sparse features in [-1,1], through m ReLU neurons.
out = ReLU(W2 ReLU(W1 x + b1) + b2); importance 0.8^i... (here IMP, default 0.97 over n=100). W1 (m x n), W2 (n x m) Muon matrices.
OPT ns = Muon, norm = same momentum + step size without Newton-Schulz.
Neuron metrics on W1: features served per neuron = participation ratio of W1[j,:]^2 over features; monosemantic = top feature > 90% of the neuron's weight.
usage: python3 absval.py OPT S SEED LR [N M STEPS] -> json line appended to abs_runs.jsonl"""
import json, os, sys, time, math
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tms import msign

def run(opt, S, seed, lr, n=100, m=40, steps=3000, imp=0.97, bs=1024):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    W1 = (torch.randn(m, n) / math.sqrt(n)).requires_grad_(); W2 = (torch.randn(n, m) / math.sqrt(m)).requires_grad_()
    b1 = torch.zeros(m, requires_grad=True); b2 = torch.zeros(n, requires_grad=True)
    I = imp ** torch.arange(n).float(); vec = torch.optim.Adam([b1, b2], lr=0.01)
    mats = [W1, W2]; bufs = [torch.zeros_like(W) for W in mats]
    data = lambda k, gg: (torch.rand(k, n, generator=gg) * 2 - 1) * (torch.rand(k, n, generator=gg) >= S)
    for i in range(steps):
        f = min(1, (i + 1) / (0.05 * steps)) * min(1, 2 * (steps - i) / steps)
        x = data(bs, g); out = torch.relu(torch.relu(x @ W1.T + b1) @ W2.T + b2)
        loss = (I * (x.abs() - out) ** 2).sum(1).mean()
        vec.zero_grad(); [setattr(W, "grad", None) for W in mats]; loss.backward()
        for gr in vec.param_groups: gr["lr"] = 0.01 * f
        vec.step()
        with torch.no_grad():
            for W, b in zip(mats, bufs):
                b.mul_(0.95).add_(W.grad); u = W.grad + 0.95 * b
                st = msign(u) if opt == "ns" else u / (u.norm() + 1e-9) * math.sqrt(min(u.shape))
                W.add_(st, alpha=-lr * f)
    with torch.no_grad():
        x = data(20000, torch.Generator().manual_seed(999)); out = torch.relu(torch.relu(x @ W1.T + b1) @ W2.T + b2)
        loss = (I * (x.abs() - out) ** 2).sum(1).mean().item()
        # per-feature error when it is on alone at value 1 (stored = error < 0.25)
        e = torch.eye(n); one = torch.relu(torch.relu(e @ W1.T + b1) @ W2.T + b2)
        stored = ((one.diagonal() - 1).abs() < 0.5)
        A = W1[:, stored] ** 2; tot = A.sum(1); alive = tot > 0.01 * tot.max()
        pr = (A.sum(1) ** 2 / (A ** 2).sum(1).clamp_min(1e-12))[alive]
        mono = (A.max(1).values / tot.clamp_min(1e-12))[alive] > 0.9
        sv1 = torch.linalg.svdvals(W1)
    return dict(opt=opt, S=S, seed=seed, lr=lr, n=n, m=m, steps=steps, imp=imp, loss=round(loss, 5), stored=int(stored.sum()),
                alive=int(alive.sum()), feats_per_neuron=round(pr.mean().item(), 2), mono=round(mono.float().mean().item(), 3),
                erank1=round((sv1.sum() ** 2 / (sv1 ** 2).sum()).item(), 1))

if __name__ == "__main__":
    a = sys.argv[1:]; opt, S, seed, lr = a[0], float(a[1]), int(a[2]), float(a[3])
    n, m, steps = (int(a[4]), int(a[5]), int(a[6])) if len(a) > 6 else (100, 40, 3000)
    t0 = time.time(); r = run(opt, S, seed, lr, n, m, steps); r["sec"] = round(time.time() - t0, 2)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "abs_runs.jsonl"), "a") as f: f.write(json.dumps(r) + "\n")
    print(json.dumps(r))
