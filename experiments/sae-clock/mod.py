"""(a + b) mod P with a 1-layer transformer (RoPE, QK-norm, RMSNorm, SwiGLU, no biases), full batch.
Hidden matrices (q k v o gate up down) on Muon or AdamW; embedding, head, norm gains always on AdamW.

usage: OPT=muon|adam python3 mod.py SEED STEPS LR [WD]      saves the model to runs/OPT_SEED.pt, appends to runs.jsonl
"""
import json, math, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F

P, D, H, FF = int(os.environ.get("P", 59)), int(os.environ.get("D", 64)), 4, int(os.environ.get("FF", 128))
torch.set_num_threads(int(os.environ.get("THREADS", 1)))
FRAC = float(os.environ.get("FRAC", 0.4))
OPT = os.environ.get("OPT", "muon")
EMB_LR = float(os.environ.get("EMB_LR", 3e-3))


class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.n1, s.n2, s.qn, s.kn = nn.RMSNorm(D), nn.RMSNorm(D), nn.RMSNorm(D // H), nn.RMSNorm(D // H)
        s.q, s.k, s.v, s.o = (nn.Linear(D, D, bias=False) for _ in range(4))
        s.gate, s.up, s.down = nn.Linear(D, FF, bias=False), nn.Linear(D, FF, bias=False), nn.Linear(FF, D, bias=False)
        nn.init.zeros_(s.o.weight); nn.init.zeros_(s.down.weight)

    def attn(s, x):
        b, t, _ = x.shape; sh = lambda z: z.view(b, t, H, D // H).transpose(1, 2)
        q, k, v = s.qn(sh(s.q(x))), s.kn(sh(s.k(x))), sh(s.v(x))
        q, k = rope(q), rope(k)
        s.att = (q @ k.transpose(-1, -2) / math.sqrt(D // H)).masked_fill(torch.ones(t, t, dtype=torch.bool).triu(1), -1e9).softmax(-1)
        return s.o((s.att @ v).transpose(1, 2).reshape(b, t, D))

    def forward(s, x):
        x = x + s.attn(s.n1(x)); h = s.n2(x)
        s.hid = F.silu(s.gate(h)) * s.up(h)            # MLP neurons (kept for analysis)
        return x + s.down(s.hid)


def rope(x):
    t, d = x.shape[-2], x.shape[-1]
    f = 1 / 10000 ** (torch.arange(0, d, 2) / d); a = torch.arange(t)[:, None] * f
    c, s_ = a.cos(), a.sin(); x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s_, x1 * s_ + x2 * c], -1).flatten(-2)


class Model(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(P + 1, D); s.blk = Block(); s.norm = nn.RMSNorm(D); s.head = nn.Linear(D, P, bias=False)

    def forward(s, x):
        s.res = s.blk(s.emb(x)); return s.head(s.norm(s.res))[:, -1]


def msign(g):                                          # Muon's 5-step quintic Newton-Schulz
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g / (g.norm() + 1e-7); tr = x.shape[0] > x.shape[1]
    if tr: x = x.T
    for _ in range(5): A = x @ x.T; x = a * x + (b * A + c * A @ A) @ x
    return x.T if tr else x


def data(seed):
    a, b = torch.meshgrid(torch.arange(P), torch.arange(P), indexing="ij"); a, b = a.flatten(), b.flatten()
    x = torch.stack([a, b, torch.full_like(a, P)], 1); y = (a + b) % P
    perm = torch.randperm(P * P, generator=torch.Generator().manual_seed(100 + seed)); n = int(FRAC * P * P)
    return x, y, perm[:n], perm[n:]


def run(seed, steps, lr, wd):
    torch.manual_seed(seed); m = Model(); x, y, tr, te = data(seed)
    mats = [p for n, p in m.named_parameters() if n.startswith("blk.") and p.ndim == 2]; ids = {id(p) for p in mats}
    rest = [p for p in m.parameters() if id(p) not in ids]
    groups = [dict(params=rest, lr=EMB_LR, weight_decay=float(os.environ.get("EMB_WD", 0)))]
    if OPT == "adam": groups.append(dict(params=mats, lr=lr, weight_decay=wd))
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98)); buf = [torch.zeros_like(p) for p in mats]
    log, t0, grok = [], time.time(), None
    for i in range(steps + 1):
        f = min(1, (i + 1) / 50)                        # 50-step warmup, then constant
        if i % 50 == 0 or i == steps:
            with torch.no_grad():
                out = m(x); acc = (out.argmax(-1) == y).float()
                tra, tea = acc[tr].mean().item(), acc[te].mean().item(); log.append((i, round(tra, 3), round(tea, 3)))
                if grok is None and tea >= 0.95: grok = i
        if i == steps: break
        loss = F.cross_entropy(m(x[tr]), y[tr])
        for g in opt.param_groups: g["lr"] = g.setdefault("base", g["lr"]) * f
        opt.zero_grad(); [setattr(p, "grad", None) for p in mats]; loss.backward(); opt.step()
        if OPT == "muon":
            with torch.no_grad():
                for p, bf in zip(mats, buf):
                    bf.mul_(0.95).add_(p.grad); u = msign(p.grad + 0.95 * bf)
                    p.mul_(1 - lr * f * wd); p.add_(u, alpha=-lr * f * max(1, p.shape[0] / p.shape[1]) ** 0.5)
    return m, log, grok, time.time() - t0


if __name__ == "__main__":
    seed, steps, lr = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]); wd = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
    m, log, grok, dt = run(seed, steps, lr, wd)
    print(f"{OPT} seed {seed} lr {lr} wd {wd} {dt:.1f}s  grok(test>=0.95) at {grok}  final train {log[-1][1]} test {log[-1][2]}")
    print("  " + " ".join(f"{i}:{a}/{b}" for i, a, b in log[:: max(1, len(log) // 12)]))
    os.makedirs(os.path.join(os.path.dirname(__file__) or ".", "runs"), exist_ok=True)
    torch.save(m.state_dict(), os.path.join(os.path.dirname(__file__) or ".", "runs", f"{OPT}_{seed}_{lr}_{wd}.pt"))
    with open(os.path.join(os.path.dirname(__file__) or ".", "runs.jsonl"), "a") as fh:
        fh.write(json.dumps(dict(opt=OPT, P=P, frac=FRAC, seed=seed, steps=steps, lr=lr, wd=wd, grok=grok, log=log, sec=round(dt, 1))) + "\n")
