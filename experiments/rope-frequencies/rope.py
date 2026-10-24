"""Which RoPE frequency pairs a small byte LM uses, and how that decides what happens past the training length.

Model: pre-norm transformer, RoPE, QK-norm, SwiGLU, RMSNorm, no biases, vocabulary 256 bytes.
Data: local markdown text (attention-sinks corpus); COPYF of the rows are random bytes repeated with period k in [4, T/2].
Muon (Newton-Schulz, Nesterov momentum 0.95) on attention and MLP matrices; Adam on embedding, head, norm gains.

Pairs are numbered 0 (fastest, angle 1 rad per token) to DK/2-1 (slowest).
env: WIDTH LAYERS HEADS T BS STEPS LR ADAM_LR BASE COPYF
     ROT=f    rotate only round(f * pairs) pairs, the rest get angle 0; 1 = full RoPE, 0 = NoPE
     SPAN=trunc  the rotated pairs keep their full-RoPE frequencies, fastest first (p-RoPE: the slow end is cut off)
     SPAN=sub    the rotated pairs span the whole range 1 .. 1/BASE, as partial_rotary_factor does in HF / GPT-NeoX code
     SPAN=slow   control: rotate the slowest pairs of the full spectrum
     QKNORM=0    no q, k norm
     EXT_STEPS=n  after pretraining, continue n steps at EXT_S x T (batch BS / EXT_S, same tokens per step) with EXT_METHOD frequencies,
                  peak lr x EXT_LR, warmup 10, linear decay to 0 (like a long-context continued-pretraining stage); then eval again ("ext")
At the end, for windows of 1x, 2x, 4x T:
  text   loss on text positions >= T/2 (1x) or >= T (longer), for each extension method:
         none, pi (positions / s), ntk (base x s^(d/(d-2))), yarn (ramp by rotations per window, alpha 1 beta 32, logits x (0.1 ln s + 1)^2)
  copy   loss on the repeats of a random segment of length n T / 2 (copy distance = n T / 2), only if COPYF > 0
  near / far  text targets right after a 4-byte string that already occurred <= T/2 back (near) or > T back (far, past the training length)
  drop   at 1x: loss change when one band of pairs is zeroed in q and k of every head (fast half / slow half)
  use    per layer, head, pair: share of |q_i| |k_i| (Barbero's norm measure)
usage: python3 rope.py SEED -> one json line appended to $OUT (default runs.jsonl)
"""
import json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(1)
E = lambda k, d: type(d)(os.environ.get(k, d))
D, L, H, T, BS, STEPS = E("WIDTH", 64), E("LAYERS", 2), E("HEADS", 4), E("T", 64), E("BS", 16), E("STEPS", 300)
LR, ALR, BASE, COPYF, ROT = E("LR", 0.02), E("ADAM_LR", 0.01), E("BASE", 10000.0), E("COPYF", 0.0), E("ROT", 1.0)
SPAN = os.environ.get("SPAN", "trunc")
QKN = E("QKNORM", 1)                                                # 0 = no RMSNorm on each head's q and k
HEADSTAT = E("HEADSTAT", 0)                                         # 1 = per-head type scores and per-head band drops
EXT_STEPS, EXT_METHOD, EXT_LR, EXT_S = E("EXT_STEPS", 0), os.environ.get("EXT_METHOD", "ntk"), E("EXT_LR", 0.3), E("EXT_S", 4)
CORPUS = os.environ.get("CORPUS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "attention-sinks", "corpus.npz"))
V, DK = 256, D // H
NP = DK // 2
NROT = round(ROT * NP)
ROTMASK = torch.zeros(NP)
if SPAN == "slow": ROTMASK[NP - NROT:] = 1
else: ROTMASK[:NROT] = 1
TH0 = 1.0 / BASE ** (torch.arange(0, DK, 2).float() / DK)          # full-RoPE pair frequencies, fastest first
if SPAN == "sub" and NROT: TH0[:NROT] = 1.0 / BASE ** (torch.arange(NROT).float() / NROT) if NROT > 1 else 1.0
TH = TH0 * ROTMASK


class Ctx:                     # eval-time settings shared by all blocks
    th, temp, zero, use, zl, zh, attn = TH, 1.0, None, None, None, None, None


def rope(x, th):
    t = x.shape[-2]
    ang = torch.outer(torch.arange(t).float(), th)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack([a * ang.cos() - b * ang.sin(), b * ang.cos() + a * ang.sin()], -1).flatten(-2)


class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.q, s.k, s.v, s.o = (nn.Linear(D, D, bias=False) for _ in range(4))
        F_ = 176 * D // 64
        s.gate, s.up, s.down = nn.Linear(D, F_, bias=False), nn.Linear(D, F_, bias=False), nn.Linear(F_, D, bias=False)
        s.n1, s.n2, s.qn, s.kn = nn.RMSNorm(D), nn.RMSNorm(D), *((nn.RMSNorm(DK), nn.RMSNorm(DK)) if QKN else (nn.Identity(), nn.Identity()))

    def forward(s, x):
        b, t, _ = x.shape
        h = s.n1(x)
        q, k, v = (m(h).view(b, t, H, DK).transpose(1, 2) for m in (s.q, s.k, s.v))
        q, k = s.qn(q), s.kn(k)
        if Ctx.use is not None:   # |q_i| |k_i| per head and pair, averaged over positions
            nq, nk = q.view(b, H, t, NP, 2).norm(dim=-1), k.view(b, H, t, NP, 2).norm(dim=-1)
            u = (nq.mean((0, 2)) * nk.mean((0, 2))); Ctx.use.append(u / u.sum(-1, keepdim=True))
        if Ctx.zero is not None and Ctx.zl in (None, s.li):
            z = torch.ones(H, 1, NP); z[slice(None) if Ctx.zh is None else Ctx.zh, :, Ctx.zero] = 0; z = z.repeat_interleave(2, -1)
            q, k = q * z, k * z
        q, k = rope(q, Ctx.th), rope(k, Ctx.th)
        if Ctx.attn is not None:
            sc = (q @ k.transpose(-1, -2)) * Ctx.temp / math.sqrt(DK)
            Ctx.attn.append(sc.masked_fill(torch.ones(t, t, dtype=torch.bool).triu(1), float("-inf")).softmax(-1))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=Ctx.temp / math.sqrt(DK))
        x = x + s.o(a.transpose(1, 2).reshape(b, t, D))
        h = s.n2(x)
        return x + s.down(F.silu(s.gate(h)) * s.up(h))


class LM(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb, s.blocks, s.norm, s.head = nn.Embedding(V, D), nn.ModuleList(Block() for _ in range(L)), nn.RMSNorm(D), nn.Linear(D, V, bias=False)
        for i, blk in enumerate(s.blocks): blk.li = i

    def forward(s, x):
        x = s.emb(x)
        for blk in s.blocks: x = blk(x)
        return s.head(s.norm(x))


def msign(g):
    a, b, c = 3.4445, -4.7750, 2.0315; tr = g.shape[0] > g.shape[1]
    X = g.T if tr else g; X = X / (X.norm() + 1e-7)
    for _ in range(5): A = X @ X.T; X = a * X + (b * A + c * A @ A) @ X
    return X.T if tr else X


def text(arr, n, t, g):
    idx = torch.randint(len(arr) - t - 1, (n,), generator=g)
    return torch.stack([torch.from_numpy(arr[i:i + t + 1].astype(np.int64)) for i in idx.tolist()])


def copy_rows(n, t, g, k=None):
    """random bytes repeated with period k; returns rows and the mask of predictable targets (positions >= k)"""
    ks = torch.full((n,), k) if k else torch.randint(4, t // 2 + 1, (n,), generator=g)
    r = torch.randint(V, (n, t + 1), generator=g); p = torch.arange(t + 1)
    w = r.gather(1, p % ks[:, None])
    return w, p[1:][None, :] >= ks[:, None]


def method_th(name, s):
    """frequencies and logit scale for a context-extension method at scale s (no fine-tuning)"""
    if name == "none" or s == 1 or not NROT: return TH, 1.0
    if name == "pi": return TH / s, 1.0
    if name == "ntk":   # fastest rotated pair unchanged, slowest rotated pair / s, geometric in between (= base x s^(d/(d-2)) for full RoPE)
        on = ROTMASK > 0; lo, hi = TH0[on].min().log(), TH0[on].max().log()
        e = ((hi - TH0.log()) / (hi - lo)) if hi > lo else torch.ones(NP)
        return TH0 * s ** (-e) * ROTMASK, 1.0
    if name == "yarn":
        r = T * TH0 / (2 * math.pi)                                   # full rotations of each pair inside the training window
        gam = ((r - 1) / (32 - 1)).clamp(0, 1)
        return ((1 - gam) * TH0 / s + gam * TH0) * ROTMASK, (0.1 * math.log(s) + 1) ** 2


def repeats(w, t):
    """masks over targets (positions 1..t of w): the 4 bytes before the target already occurred earlier, <= T/2 back (near) or > T back (far)"""
    a = w.numpy(); near = np.zeros((len(a), t), bool); far = np.zeros((len(a), t), bool)
    for r, row in enumerate(a):
        last = {}
        for j in range(3, t):          # context ends at position j, target is row[j + 1] = mask index j
            key = row[j - 3:j + 1].tobytes()
            if key in last:
                dist = j - last[key]
                near[r, j] = dist <= T // 2; far[r, j] = dist > T
            last[key] = j
    return torch.from_numpy(near), torch.from_numpy(far)


def ind_keys(w, t):
    """(row, query, key) for near repeats: query j ends a 4-byte string seen before ending at p; an induction head attends to key p + 1"""
    out = []
    for r, row in enumerate(w.numpy()):
        last = {}
        for j in range(3, t):
            key = row[j - 3:j + 1].tobytes()
            if key in last and j - last[key] <= T // 2 and j >= T // 2: out.append((r, j, last[key] + 1))
            last[key] = j
    return torch.tensor(out)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    d = np.load(CORPUS); tr, va = d["train"], d["val"]
    torch.manual_seed(seed); m = LM()
    mats = [p for n, p in m.named_parameters() if p.ndim == 2 and not n.startswith(("emb", "head"))]
    ids = {id(p) for p in mats}
    adam = torch.optim.Adam([p for p in m.parameters() if id(p) not in ids], lr=ALR, betas=(0.9, 0.98))
    buf = [torch.zeros_like(p) for p in mats]
    g = torch.Generator().manual_seed(seed)

    def train(steps, t, bs, peak, warm):
        nc = round(bs * COPYF)
        for i in range(steps):
            f = peak * min(1, (i + 1) / warm) * min(1, 2 * (1 - i / steps))
            w = text(tr, bs - nc, t, g); c, cm = copy_rows(nc, t, g)
            msk = torch.cat([torch.ones(bs - nc, t, dtype=torch.bool), cm])   # one forward pass for text and copy rows
            w = torch.cat([w, c]); loss = F.cross_entropy(m(w[:, :-1])[msk], w[:, 1:][msk])
            if not torch.isfinite(loss): return False
            adam.zero_grad(); [setattr(p, "grad", None) for p in mats]; loss.backward()
            for gr in adam.param_groups: gr["lr"] = ALR * f
            adam.step()
            with torch.no_grad():
                for p, bf in zip(mats, buf):
                    bf.mul_(0.95).add_(p.grad)
                    p.add_(msign(p.grad + 0.95 * bf), alpha=-LR * f * max(1, p.shape[0] / p.shape[1]) ** 0.5)
        return True

    t0 = time.time(); ok = train(STEPS, T, BS, 1.0, 30)
    sec = time.time() - t0
    res = dict(width=D, layers=L, heads=H, dk=DK, T=T, bs=BS, steps=STEPS, lr=LR, base=BASE, copyf=COPYF, rot=ROT, 
               span=SPAN, qknorm=QKN, seed=seed, sec=round(sec, 1), ok=ok, text={}, near={}, far={}, copy={}, drop={})
    ge = lambda: torch.Generator().manual_seed(12345)
    with torch.no_grad():
        for s in (1, 2, 4):
            t = T * s; w = text(va, 64, t, ge()); c, cm = copy_rows(64, t, ge(), k=t // 2)
            st = T // 2 if s == 1 else T; nr, fr = repeats(w, t); nr[:, :st] = False
            for name in ("none", "pi", "ntk", "yarn"):
                if s == 1 and name != "none": continue
                Ctx.th, Ctx.temp = method_th(name, s)
                o = m(w[:, :-1]); y = w[:, 1:]
                res["text"][f"{s}x_{name}"] = round(F.cross_entropy(o[:, st:].reshape(-1, V), y[:, st:].reshape(-1)).item(), 4)
                res["near"][f"{s}x_{name}"] = round(F.cross_entropy(o[nr], y[nr]).item(), 4)
                if s > 1: res["far"][f"{s}x_{name}"] = round(F.cross_entropy(o[fr], y[fr]).item(), 4)
                if COPYF:
                    oc = m(c[:, :-1])
                    res["copy"][f"{s}x_{name}"] = round(F.cross_entropy(oc[cm], c[:, 1:][cm]).item(), 4)
        Ctx.th, Ctx.temp = TH, 1.0
        w = text(va, 64, T, ge()); c, cm = copy_rows(64, T, ge(), k=T // 2); nr, _ = repeats(w, T); nr[:, :T // 2] = False
        Ctx.use = []; m(w[:, :-1]); res["use"] = [[[round(x, 3) for x in hd] for hd in u.tolist()] for u in Ctx.use]; Ctx.use = None
        for band, z in (("fast", list(range(NP // 2))), ("slow", list(range(NP // 2, NP)))):
            Ctx.zero = z
            o = m(w[:, :-1]); y = w[:, 1:]
            res["drop"][band] = [round(F.cross_entropy(o[:, T // 2:].reshape(-1, V), y[:, T // 2:].reshape(-1)).item() - res["text"]["1x_none"], 4),
                                 round(F.cross_entropy(o[nr], y[nr]).item() - res["near"]["1x_none"], 4)]
        Ctx.zero = None
        if HEADSTAT:   # per head: previous-token score, induction score, loss change when its fast / slow half is zeroed
            Ctx.attn = []; m(w[:, :-1]); A = Ctx.attn; Ctx.attn = None
            ik = ind_keys(w, T); y = w[:, 1:]; o = m(w[:, :-1])
            base_t = F.cross_entropy(o[:, T // 2:].reshape(-1, V), y[:, T // 2:].reshape(-1)).item(); base_n = F.cross_entropy(o[nr], y[nr]).item()
            hs = []
            for li in range(L):
                for h in range(H):
                    a = A[li][:, h]
                    row = dict(l=li, h=h, prev=round(a[:, 1:, :].diagonal(-1, 1, 2).mean().item(), 3),
                               ind=round(a[ik[:, 0], ik[:, 1], ik[:, 2]].mean().item(), 3))
                    Ctx.zl, Ctx.zh = li, h
                    for band, z in (("fast", list(range(NP // 2))), ("slow", list(range(NP // 2, NP)))):
                        Ctx.zero = z; o = m(w[:, :-1])
                        row[band] = [round(F.cross_entropy(o[:, T // 2:].reshape(-1, V), y[:, T // 2:].reshape(-1)).item() - base_t, 4),
                                     round(F.cross_entropy(o[nr], y[nr]).item() - base_n, 4)]
                    hs.append(row)
            Ctx.zero = Ctx.zl = Ctx.zh = None; res["heads"] = hs
    if EXT_STEPS:
        Ctx.th, Ctx.temp = method_th(EXT_METHOD, EXT_S)
        res.update(ext_steps=EXT_STEPS, ext_method=EXT_METHOD, ext_lr=EXT_LR, ext_s=EXT_S, ext_ok=train(EXT_STEPS, T * EXT_S, max(1, BS // EXT_S), EXT_LR, 10))
        with torch.no_grad():
            t = T * EXT_S; w = text(va, 64, t, ge()); nr, fr = repeats(w, t); nr[:, :T] = False
            o = m(w[:, :-1]); y = w[:, 1:]
            res["ext"] = dict(text=round(F.cross_entropy(o[:, T:].reshape(-1, V), y[:, T:].reshape(-1)).item(), 4),
                              near=round(F.cross_entropy(o[nr], y[nr]).item(), 4), far=round(F.cross_entropy(o[fr], y[fr]).item(), 4))
            w = text(va, 64, T, ge()); o = m(w[:, :-1]); y = w[:, 1:]
            res["ext"]["short"] = round(F.cross_entropy(o[:, T // 2:].reshape(-1, V), y[:, T // 2:].reshape(-1)).item(), 4)
        res["sec"] = round(time.time() - t0, 1)
    with open(os.environ.get("OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.jsonl")), "a") as fh: fh.write(json.dumps(res) + "\n")
    print(json.dumps({k: res[k] for k in ("width", "layers", "T", "steps", "base", "rot", "span", "qknorm", "seed", "sec", "text", "near", "far", "drop", "ext") if k in res}))


if __name__ == "__main__":
    main()
