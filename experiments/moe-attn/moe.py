"""How mixture-of-experts layers and attention interact, on a toy where both are needed.
Task 'facts': [subject, rel, ans, rel, ans, ...]: the answer depends on the subject (fetched by attention from position 0)
and the relation (previous token); facts are stored in the MLP/experts. Task 'copy': a random word segment repeats (attention only).
model: NL layers, width D, 4 heads, RoPE, QK-norm, SwiGLU, RMSNorm, no biases. MOE=E: E SwiGLU experts, top-K, expert hidden FF/K
(same active size as dense), softmax router, gates renormalized over the chosen ones, DeepSeek-V3 bias balancing (no aux loss).
Muon (5-step Newton-Schulz, momentum 0.95 Nesterov) on block matrices incl. each expert slice; Adam on embedding, head, norms, router.
usage: [MOE=8 K=2 D=32 NL=2 STEPS=400 NS=512 ...] python3 moe.py LR SEED   -> runs.jsonl
"""
import sys, os, json, time, math
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
env = lambda k, d: type(d)(os.environ.get(k, d))
D, NL, H, T, B = env("D", 32), env("NL", 2), 4, 16, env("B", 32)
STEPS, WARM = env("STEPS", 400), env("WARM", 20)
MOE, K = env("MOE", 0), env("K", 2)
FF = env("FF", 88 * D // 32)                       # dense hidden; experts get FF // K each (same active size)
EF = env("EF", FF // max(1, K))
NS, NR, NA = env("NS", 512), env("NR", 8), env("NA", 256)   # subjects, relations, answer words
COPYW = env("COPYW", 0.5)                          # share of copy sequences
ADAM_LR, EMB_MULT = env("ADAM_LR", 0.01), env("EMB_MULT", 8.0)
DENSE0 = env("DENSE0", 0)                          # first DENSE0 layers dense (DeepSeek first_k_dense_replace)
MUON_EXP = os.environ.get("MUON_EXP", "each")       # each: orthogonalize every expert's matrix alone; stack: one orthogonalization of all experts side by side
ROUTER_MUON = env("ROUTER_MUON", 0)                # 1: router weight on Muon (as a plain 2D matrix) instead of Adam
LOADPOW = env("LOADPOW", 0.0)                      # per-expert Muon step x (expert's token share / fair share)^LOADPOW; 0 = off
ROUTE = os.environ.get("ROUTE", "h")               # router input: h (normal: after attention), x0 (token embedding)
EXP_MULT, ATT_MULT, ROUTER_MULT = env("EXP_MULT", 1.0), env("ATT_MULT", 1.0), env("ROUTER_MULT", 1.0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("FOUT", "runs.jsonl"))
S0, R0, A0 = 0, NS, NS + NR; V = NS + NR + NA + 1  # token ranges: subjects, relations, answer words, one 'is' token (IS)
IS = V - 1
GENERIC = env("GENERIC", 0)                        # 1: facts as [s, rel, is, ans, rel, is, ans, ...]: the token before every answer is the same 'is' token
FACT = torch.randint(NA, (NS, NR), generator=torch.Generator().manual_seed(7)) + A0   # the fact table


def gen(n, g):
    """n sequences of length T+1 and a loss mask over the T targets"""
    x = torch.empty(n, T + 1, dtype=torch.long); m = torch.zeros(n, T, dtype=torch.bool)
    task = torch.rand(n, generator=g) < COPYW                                  # True = copy
    nf = int((~task).sum()); s = torch.randint(NS, (nf,), generator=g)
    r = torch.randint(NR, (nf, T // 2), generator=g)
    f = torch.empty(nf, T + 1, dtype=torch.long); f[:, 0] = s
    f[:, 1::2] = (r + R0)[:, :T // 2 + (T + 1) % 2 * 0][:, :f[:, 1::2].shape[1]]
    f[:, 2::2] = FACT[s[:, None], r][:, :f[:, 2::2].shape[1]]
    mf = torch.zeros(nf, T, dtype=torch.bool); mf[:, 1::2] = True
    if GENERIC:
        f = torch.full((nf, T + 1), IS, dtype=torch.long); f[:, 0] = s; nq = T // 3
        f[:, 1:3 * nq + 1:3] = r[:, :nq] + R0; f[:, 3:3 * nq + 1:3] = FACT[s[:, None], r[:, :nq]]
        mf = torch.zeros(nf, T, dtype=torch.bool); mf[:, 2:3 * nq:3] = True
    x[~task] = f; m[~task] = mf
    nc = n - nf; L = 6
    w = torch.randint(NA, (nc, L), generator=g) + A0; c = torch.randint(NA, (nc, T + 1), generator=g) + A0
    c[:, 1:L + 1] = w; c[:, L + 3:2 * L + 3] = w; x[task] = c
    mc = torch.zeros(nc, T, dtype=torch.bool); mc[:, L + 3:2 * L + 2] = True; m[task] = mc   # targets: copied words 2..L
    return x, m, task


class RMS(nn.Module):
    def __init__(s, n): super().__init__(); s.w = nn.Parameter(torch.ones(n))
    def forward(s, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * s.w


_f = 1.0 / 10000 ** (torch.arange(0, D // H, 2) / (D // H)); _a = torch.arange(T)[:, None] * _f[None]
COS, SIN = _a.cos(), _a.sin()
def rope(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]; c, s = COS[:x.shape[-2]], SIN[:x.shape[-2]]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class Experts(nn.Module):
    def __init__(s, e, out, inp): super().__init__(); s.weight = nn.Parameter(torch.empty(e, out, inp).uniform_(-inp ** -0.5, inp ** -0.5))


class MoE(nn.Module):
    def __init__(s):
        super().__init__(); s.router = nn.Linear(D, MOE, bias=False)
        s.gate, s.up, s.down = Experts(MOE, EF, D), Experts(MOE, EF, D), Experts(MOE, D, EF)
        s.register_buffer("bias", torch.zeros(MOE)); s.top = None
        if ROUTE == "both": s.router2 = nn.Linear(D, MOE, bias=False)   # second router on the token embedding, logits added
    def forward(s, h, r):
        b, t, _ = h.shape; x = h.reshape(-1, D)
        if ROUTE == "both": p = (s.router(x) + s.router2(r.reshape(-1, D))).softmax(-1)
        else: p = s.router(r.reshape(-1, D)).softmax(-1)
        top = (p + s.bias).topk(K, -1).indices; s.top = top
        g = torch.zeros_like(p).scatter(1, top, p.gather(1, top)); g = g / g.sum(-1, keepdim=True)
        if torch.is_grad_enabled():
            load = (g > 0).float().mean(0) * MOE / K; s.bias -= 0.01 * torch.sign(load - 1)
        a = F.silu(torch.einsum("nd,efd->enf", x, s.gate.weight)) * torch.einsum("nd,efd->enf", x, s.up.weight)
        y = torch.einsum("enf,edf->end", a, s.down.weight)               # all experts on all tokens (tiny sizes), then weight by gates
        return (y * g.T[:, :, None]).sum(0).view(b, t, D)


class Block(nn.Module):
    def __init__(s, moe):
        super().__init__(); L = lambda i, o: nn.Linear(i, o, bias=False)
        s.n1, s.n2, s.qn, s.kn = RMS(D), RMS(D), RMS(D // H), RMS(D // H)
        s.q, s.k, s.v, s.o = L(D, D), L(D, D), L(D, D), L(D, D)
        s.is_moe = moe
        if moe: s.moe = MoE()
        else: s.gate, s.up, s.down = L(D, FF), L(D, FF), L(FF, D)
    def forward(s, x, x0):
        b, t, _ = x.shape; h = s.n1(x); sp = lambda z: z.view(b, t, H, D // H).transpose(1, 2)
        a = F.scaled_dot_product_attention(rope(s.qn(sp(s.q(h)))), rope(s.kn(sp(s.k(h)))), sp(s.v(h)), is_causal=True)
        x = x + s.o(a.transpose(1, 2).reshape(b, t, D)); h = s.n2(x)
        if s.is_moe: return x + s.moe(h, h if ROUTE == "h" else x0)       # x0 / both: router (also) reads the token embedding
        return x + s.down(F.silu(s.gate(h)) * s.up(h))


class LM(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(V, D)
        s.blocks = nn.ModuleList(Block(bool(MOE) and i >= DENSE0) for i in range(NL))
        s.norm = RMS(D); s.head = nn.Linear(D, V, bias=False)
    def forward(s, x):
        h = x0 = s.emb(x)
        for b in s.blocks: h = b(h, x0)
        return s.head(s.norm(h))


def ns(G):
    a, b, c = 3.4445, -4.7750, 2.0315; X = G / (G.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    tr = X.shape[-2] > X.shape[-1]
    if tr: X = X.transpose(-2, -1)
    for _ in range(5): A = X @ X.transpose(-2, -1); X = a * X + (b * A + c * A @ A) @ X
    return X.transpose(-2, -1) if tr else X


@torch.no_grad()
def evaluate(m):
    x, msk, task = gen(1024, torch.Generator().manual_seed(999)); lo = m(x[:, :-1]); y = x[:, 1:]
    l = F.cross_entropy(lo.reshape(-1, V), y.reshape(-1), reduction="none").view(y.shape)
    acc = (lo.argmax(-1) == y).float()
    out = {}
    for name, sel in (("facts", ~task), ("copy", task)):
        mm = msk & sel[:, None]; out[name] = round(l[mm].mean().item(), 4); out[name + "_acc"] = round(acc[mm].mean().item(), 4)
    return out


def train(lr, seed):
    torch.manual_seed(seed); m = LM(); g = torch.Generator().manual_seed(seed + 100)
    mats, rest, mult = [], [], {}
    for n, p in m.named_parameters():
        if p.ndim >= 2 and "emb" not in n and "head" not in n and ("router" not in n or ROUTER_MUON):
            mats.append(p); mult[id(p)] = ROUTER_MULT if "router" in n else EXP_MULT if "moe" in n else ATT_MULT if any(f".{z}." in n for z in "qkvo") else 1.0
        else: rest.append(p); mult[id(p)] = EMB_MULT if ("emb" in n or "head" in n) else ROUTER_MULT if "router" in n else 1.0
    buf = [torch.zeros_like(p) for p in mats]; e1 = [torch.zeros_like(p) for p in rest]; e2 = [torch.zeros_like(p) for p in rest]
    for i in range(STEPS):
        f = min(1.0, (i + 1) / WARM) * min(1.0, 2 * (1 - i / STEPS))
        x, msk, _ = gen(B, g); lo = m(x[:, :-1]); loss = F.cross_entropy(lo[msk], x[:, 1:][msk])
        m.zero_grad(set_to_none=True); loss.backward()
        with torch.no_grad():
            ls = {}
            if LOADPOW:
                for blk in m.blocks:
                    if blk.is_moe:
                        c = torch.bincount(blk.moe.top.flatten(), minlength=MOE).float(); w = (c / c.mean()).pow(LOADPOW)[:, None, None]
                        for q in (blk.moe.gate.weight, blk.moe.up.weight, blk.moe.down.weight): ls[id(q)] = w
            for p, bf in zip(mats, buf):
                bf.mul_(0.95).add_(p.grad); u = p.grad + 0.95 * bf
                if p.ndim == 3 and MUON_EXP == "stack":
                    e, o_, i_ = u.shape
                    if o_ > i_: u = ns(u.reshape(e * o_, i_)).view(e, o_, i_); sc = max(1, e * o_ / i_) ** 0.5      # gate/up: experts stacked by rows
                    else: u = ns(u.permute(1, 0, 2).reshape(o_, e * i_)).view(o_, e, i_).permute(1, 0, 2); sc = 1.0   # down: side by side
                elif p.ndim == 3 and MUON_EXP == "fused":                    # fused 2D storage [E*rows, cols] treated as one plain matrix
                    e, o_, i_ = u.shape; u = ns(u.reshape(e * o_, i_)).view(e, o_, i_); sc = max(1, e * o_ / i_) ** 0.5
                elif p.ndim == 3 and MUON_EXP == "flat":                     # g.view(g.size(0), -1): each expert one row (Moonlight example code)
                    e, o_, i_ = u.shape; u = ns(u.reshape(e, o_ * i_)).view(e, o_, i_); sc = 1.0
                else: u = ns(u); sc = max(1, p.shape[-2] / p.shape[-1]) ** 0.5
                if id(p) in ls: u = u * ls[id(p)]
                p.add_(u, alpha=-lr * f * mult[id(p)] * sc)
            for p, a, b2 in zip(rest, e1, e2):
                a.mul_(0.9).add_(p.grad, alpha=0.1); b2.mul_(0.95).add_(p.grad ** 2, alpha=0.05)
                p.add_(a / (1 - 0.9 ** (i + 1)) / ((b2 / (1 - 0.95 ** (i + 1))).sqrt() + 1e-8), alpha=-ADAM_LR * f * mult[id(p)])
    return m


if __name__ == "__main__":
    lr, seed = float(sys.argv[1]), int(sys.argv[2]); t0 = time.time(); m = train(lr, seed); ev = evaluate(m)
    keys = ["MOE", "K", "D", "NL", "STEPS", "NS", "NR", "NA", "COPYW", "EF", "FF", "DENSE0", "ROUTER_MUON", "GENERIC", "LOADPOW", "MUON_EXP", "ROUTE", "EXP_MULT", "ATT_MULT", "ROUTER_MULT", "EMB_MULT", "B"]
    cfg = {k.lower(): globals()[k] for k in keys}; cfg.update({k.lower(): os.environ[k] for k in os.environ if k.startswith("X_")})
    row = dict(cfg, lr=lr, seed=seed, sec=round(time.time() - t0, 1), **ev)
    open(OUT, "a").write(json.dumps(row) + "\n")
    print(" ".join(f"{k}={v}" for k, v in cfg.items() if k in ("moe", "k", "d", "nl", "steps", "ns", "dense0", "muon_exp", "route", "exp_mult", "att_mult")), f"lr {lr} seed {seed}:",
          " ".join(f"{k} {v}" for k, v in ev.items()), f"({row['sec']}s)", flush=True)
