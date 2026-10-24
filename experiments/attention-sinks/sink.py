"""Attention sinks and huge activations in a small byte-level LM trained with Muon.

Model: pre-norm transformer, RoPE, QK-norm, SwiGLU, RMSNorm, no biases, untied head, vocabulary 256 bytes.
Data: local markdown text, windows of T bytes cut at random places (so the first byte is arbitrary).
Muon (Newton-Schulz, Nesterov momentum) on every attention and MLP matrix; embedding, head and norm gains use Adam.

Measured at the end on fixed validation windows:
  val      next-byte loss
  sink     attention on position 0 from queries at positions >= 8, per layer x head
  ratio    share of heads with sink > 0.3 (threshold of Gu et al. 2024)
  big      largest |activation| in the residual stream after each layer / median |activation| of that layer
  big0     share of windows whose largest activation (last layer) sits at position 0

env: WIDTH LAYERS HEADS T BS STEPS LR MOM WD ADAM_LR
     GATE=1  sigmoid gate on each head's attention output (Qwen gated attention)
     SM1=1   softmax + 1: an extra key with score 0 and value 0 every query can see
     SINKKV=1 an extra learned key and value per head every query can see
     SINKB=1 an extra key with a learned score per head and value 0 (gpt-oss attention sinks)
     BOS=1   every window starts with the same extra token 256
usage: python3 sink.py SEED  -> one json line appended to $OUT (default runs.jsonl)
"""
import json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(1)
E = lambda k, d: type(d)(os.environ.get(k, d))
D, L, H, T, BS, STEPS = E("WIDTH", 64), E("LAYERS", 4), E("HEADS", 4), E("T", 64), E("BS", 16), E("STEPS", 300)
LR, MOM, WD, ALR = E("LR", 0.02), E("MOM", 0.95), E("WD", 0.0), E("ADAM_LR", 0.01)
GATE, SM1, SINKKV, SINKB, BOS = (os.environ.get(k, "0") == "1" for k in ("GATE", "SM1", "SINKKV", "SINKB", "BOS"))
DATA = os.environ.get("DATA", "text")                     # text = bytes of markdown; bb = bigram-backcopy (Guo et al. 2024)
NORM_LR = E("NORM_LR", 1.0)
QKNORM = os.environ.get("QKNORM", "1") == "1"
OPT = os.environ.get("OPT", "muon")                        # muon | sgd (plain momentum SGD on the matrices, a check of the task)
SCHED = os.environ.get("SCHED", "decay")                   # decay = linear decay over the last half; const = flat after warmup                                # Adam lr multiplier for the norm gains (0 = gains frozen at 1)
CORPUS = os.environ.get("CORPUS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.npz"))
BLOCK0 = [False]
V, DK = (257 if DATA == "text" else 65), D // H
if DATA == "bb":                                           # 64 tokens, each has 6 likely next tokens; 3 triggers copy the token before them
    _g = torch.Generator().manual_seed(7)
    BIG = torch.zeros(64, 64)
    for i in range(64): BIG[i, torch.randperm(64, generator=_g)[:6]] = 1 / torch.arange(1, 7).float()
    BIG /= BIG.sum(1, keepdim=True); TRIG = torch.tensor([3, 17, 41])


def rope(x, base=10000):
    t, d = x.shape[-2], x.shape[-1]
    ang = torch.outer(torch.arange(t).float(), 1.0 / base ** (torch.arange(0, d, 2).float() / d))
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack([a * ang.cos() - b * ang.sin(), b * ang.cos() + a * ang.sin()], -1).flatten(-2)


class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.q, s.k, s.v, s.o = (nn.Linear(D, D, bias=False) for _ in range(4))
        F_ = 176 * D // 64
        s.gate, s.up, s.down = nn.Linear(D, F_, bias=False), nn.Linear(D, F_, bias=False), nn.Linear(F_, D, bias=False)
        s.n1, s.n2, s.qn, s.kn = nn.RMSNorm(D), nn.RMSNorm(D), *((nn.RMSNorm(DK), nn.RMSNorm(DK)) if QKNORM else (nn.Identity(), nn.Identity()))
        if GATE: s.g = nn.Linear(D, D, bias=False)
        if SINKKV: s.sk, s.sv = nn.Parameter(torch.randn(H, 1, DK)), nn.Parameter(torch.zeros(H, 1, DK))
        if SINKB: s.sb = nn.Parameter(torch.zeros(H))

    def forward(s, x, keep=None):
        b, t, _ = x.shape
        h = s.n1(x)
        q, k, v = (m(h).view(b, t, H, DK).transpose(1, 2) for m in (s.q, s.k, s.v))
        q, k = rope(s.qn(q)), rope(s.kn(k))
        mask = torch.ones(t, t, dtype=torch.bool).tril()
        if BLOCK0[0]: mask[1:, 0] = False                   # eval only: later queries cannot see token 0 (its KV evicted)
        if SM1 or SINKKV or SINKB:                          # one extra key at the front that every query sees
            k0 = s.kn(s.sk).expand(b, H, 1, DK) if SINKKV else torch.zeros(b, H, 1, DK)
            v0 = s.sv.expand(b, H, 1, DK) if SINKKV else torch.zeros(b, H, 1, DK)
            k, v = torch.cat([k0, k], 2), torch.cat([v0, v], 2)
            mask = torch.cat([torch.ones(t, 1, dtype=torch.bool), mask], 1)
            if SINKB:                                       # its score is a learned number per head instead of q . 0
                bias = torch.zeros(H, t, t + 1); bias[:, :, 0] = s.sb[:, None]
                mask = bias.masked_fill(~mask, float("-inf"))
        if keep is None: a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            z = (q @ k.transpose(-1, -2)) / math.sqrt(DK)
            p = (z + mask if mask.dtype != torch.bool else z.masked_fill(~mask, float("-inf"))).softmax(-1)
            keep.append(p); a = p @ v; s.vn = v[:, :, -t:].norm(dim=-1)
        a = a.transpose(1, 2).reshape(b, t, D)
        if GATE: a = a * torch.sigmoid(s.g(h))
        x = x + s.o(a)
        h = s.n2(x)
        return x + s.down(F.silu(s.gate(h)) * s.up(h))


class LM(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb, s.blocks, s.norm, s.head = nn.Embedding(V, D), nn.ModuleList(Block() for _ in range(L)), nn.RMSNorm(D), nn.Linear(D, V, bias=False)

    def forward(s, x, keep=None, resid=None):
        x = s.emb(x)
        for blk in s.blocks:
            x = blk(x, keep)
            if resid is not None: resid.append(x)
        return s.head(s.norm(x))


def msign(g):
    a, b, c = 3.4445, -4.7750, 2.0315; tr = g.shape[0] > g.shape[1]
    X = g.T if tr else g; X = X / (X.norm() + 1e-7)
    for _ in range(5): A = X @ X.T; X = a * X + (b * A + c * A @ A) @ X
    return X.T if tr else X


def bb(n, g):
    s = torch.zeros(n, T + 1, dtype=torch.long); s[:, 0] = 64
    s[:, 1] = torch.randint(64, (n,), generator=g)
    for t in range(1, T):
        nxt = torch.multinomial(BIG[s[:, t]], 1, generator=g)[:, 0]
        s[:, t + 1] = torch.where(torch.isin(s[:, t], TRIG) & (t > 1), s[:, t - 1], nxt)
    return s


def windows(arr, idx):
    w = torch.stack([torch.from_numpy(arr[i:i + T + 1].astype(np.int64)) for i in idx.tolist()])
    if BOS: w = torch.cat([torch.full((len(w), 1), 256), w[:, :-1]], 1)
    return w


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if DATA == "text":
        d = np.load(CORPUS); tr, va = d["train"], d["val"]
        val = windows(va, torch.randint(len(va) - T - 1, (128,), generator=torch.Generator().manual_seed(12345)))
        draw = lambda g: windows(tr, torch.randint(len(tr) - T - 1, (BS,), generator=g))
    else:
        val = bb(128, torch.Generator().manual_seed(12345)); pool = bb(65536, torch.Generator().manual_seed(seed + 100))
        draw = lambda g: pool[torch.randint(len(pool), (BS,), generator=g)]
    torch.manual_seed(seed); m = LM()
    mats = [p for n, p in m.named_parameters() if p.ndim == 2 and not n.startswith(("emb", "head"))]
    ids = {id(p) for p in mats}
    rest = [(n, p) for n, p in m.named_parameters() if id(p) not in ids]
    gains = [p for n, p in rest if p.ndim == 1 and not n.endswith(".sb")]
    if NORM_LR == 0: [p.requires_grad_(False) for p in gains]
    adam = torch.optim.Adam([dict(params=[p for n, p in rest if p.ndim != 1 or n.endswith(".sb")], lr=ALR, mult=1.0)]
                            + ([dict(params=gains, lr=ALR * NORM_LR, mult=NORM_LR)] if NORM_LR else []), betas=(0.9, 0.98))
    buf = [torch.zeros_like(p) for p in mats]
    g = torch.Generator().manual_seed(seed)
    lossf = lambda w: F.cross_entropy(m(w[:, :-1]).reshape(-1, V), w[:, 1:].reshape(-1))
    off = 1 if (SM1 or SINKKV or SINKB) else 0
    EVERY = E("EVERY", 0)

    @torch.no_grad()
    def measure():
        v = lossf(val).item()
        keep, resid = [], []
        m(val[:, :-1], keep, resid)
        q = torch.zeros(val.shape[0], val.shape[1] - 1, dtype=torch.bool); q[:, 8:] = True
        if DATA == "bb": q &= ~torch.isin(val[:, :-1], TRIG)                             # bb: only queries with nothing to copy
        w = q.float()[:, None, :] / q.sum()
        sink = torch.stack([(p[:, :, :, off] * w).sum((0, 2)) for p in keep])           # attention on position 0, layers x heads
        extra = torch.stack([(p[:, :, :, 0] * w).sum((0, 2)) for p in keep]).mean().item() if off else None   # on the extra key
        big = [(r.abs().amax() / r.abs().median()).item() for r in resid]
        big0 = (resid[-1].abs().amax(-1).argmax(-1) == 0).float().mean().item()
        r0 = [(x[:, 0] / x[:, 1:].median(-1).values).mean().item() for x in (r.norm(dim=-1) for r in resid)]  # residual norm token 0 / others
        v0 = [(blk.vn[:, :, 0].mean() / blk.vn[:, :, 1:].mean()).item() for blk in m.blocks]                  # value norm token 0 / others
        lt = lambda w: F.cross_entropy(m(w[:, :-1])[:, 8:].reshape(-1, V), w[:, 9:].reshape(-1)).item()
        base = lt(val); BLOCK0[0] = True; drop0 = lt(val) - base; BLOCK0[0] = False   # loss rise at positions >= 8 when token 0 is dropped
        en = m.emb.weight.norm(dim=-1); tok0 = 64 if DATA == "bb" else 256
        e0 = (en[tok0] / en[:tok0].median()).item()                                        # embedding size of the start token / others
        return dict(val=round(v, 4), drop0=round(drop0, 4), e0=round(e0, 2), ratio=round((sink > 0.3).float().mean().item(), 3), sink_mean=round(sink.mean().item(), 3),
                    sink=[[round(x, 3) for x in row] for row in sink.tolist()], extra=None if extra is None else round(extra, 3),
                    big=[round(x, 1) for x in big], big0=round(big0, 3), r0=[round(x, 2) for x in r0], v0=[round(x, 3) for x in v0])

    t0, curve = time.time(), []
    for i in range(STEPS):
        if EVERY and i % EVERY == 0: c = measure(); curve.append(dict(step=i, **{k: c[k] for k in ("val", "sink_mean", "ratio", "big", "r0", "v0", "e0")}))
        f = min(1, (i + 1) / 30) * (min(1, 2 * (1 - i / STEPS)) if SCHED == "decay" else 1)
        loss = lossf(draw(g))
        if not torch.isfinite(loss): break
        adam.zero_grad(); [setattr(p, "grad", None) for p in mats]; loss.backward()
        for gr in adam.param_groups: gr["lr"] = ALR * gr["mult"] * f
        adam.step()
        with torch.no_grad():
            for p, bf in zip(mats, buf):
                bf.mul_(MOM).add_(p.grad)
                p.mul_(1 - LR * f * WD)
                if OPT == "sgd": p.add_(p.grad + MOM * bf, alpha=-LR * f)
                else: p.add_(msign(p.grad + MOM * bf), alpha=-LR * f * max(1, p.shape[0] / p.shape[1]) ** 0.5)
    sec = time.time() - t0
    r = dict(opt=OPT, data=DATA, width=D, layers=L, heads=H, T=T, bs=BS, steps=STEPS, sched=SCHED, lr=LR, mom=MOM, wd=WD, adam_lr=ALR, norm_lr=NORM_LR,
             qknorm=int(QKNORM), gate=int(GATE), sm1=int(SM1), sinkkv=int(SINKKV), sinkb=int(SINKB), bos=int(BOS), seed=seed, **measure(), sec=round(sec, 1), curve=curve)
    with open(os.environ.get("OUT", "runs.jsonl"), "a") as fh: fh.write(json.dumps(r) + "\n")
    print(json.dumps({k: r[k] for k in ("opt", "width", "layers", "steps", "lr", "wd", "norm_lr", "gate", "sm1", "sinkkv", "sinkb", "seed", "val", "ratio",
                                        "sink_mean", "extra", "drop0", "big", "big0", "r0", "v0", "e0", "sec")}))
    return m, val


if __name__ == "__main__":
    main()
