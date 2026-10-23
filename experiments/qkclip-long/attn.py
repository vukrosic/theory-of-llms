"""Max attention logit under Muon without QK-norm: logit soft-capping (Gemma 2) and output gating (Qwen gated attention) as alternatives.
Knobs: SOFTCAP=c (scores -> c*tanh(scores/c) before softmax), GATE=1 (attention output *= sigmoid(W_g h), per head-dim, Muon-trained).
usage: [QKNORM=0] [D= B= STEPS= MOM= DECAY= LOGEVERY=] python3 attn.py LR1,LR2,... [SEED]  -> attn.jsonl (one row per run)
Model/loop copied from what-changes-muon-lr (lr.py, scale.py, endpoint.py imported, not edited).
Knobs (off by default): QKNORM=0 removes QK-norm; logged: max |q.k|/sqrt(d_head) (after RoPE) per layer on a fixed probe batch every LOGEVERY steps.
"""
import sys, os, json, time, math
sys.dont_write_bytecode = True
SRC = "/Users/vukrosic/my-life/openai-anthropic-goal/theory-of-llms/experiments/what-changes-muon-lr"
sys.path.insert(0, SRC)
os.environ.setdefault("B", "8"); os.environ.setdefault("D", "16"); os.environ.setdefault("STEPS", "800")
import torch, torch.nn as nn, torch.nn.functional as F
import endpoint as E
L = E.L
torch.set_num_threads(1)
QKNORM = int(os.environ.get("QKNORM", 1)); MOM = float(os.environ.get("MOM", 0.95)); WD = float(os.environ.get("WD", 0)); DECAY = int(os.environ.get("DECAY", 1))
SOFTCAP = float(os.environ.get("SOFTCAP", 0)); GATE = int(os.environ.get("GATE", 0))
QKCLIP = float(os.environ.get("QKCLIP", 0)); CLIPSTOP = int(os.environ.get("CLIPSTOP", 10**9))   # stop clipping after this step   # tau; 0 = off. Kimi K2 QK-Clip, simple per-head version (all q,k dims clipped, sqrt(gamma) each; no MLA shared-rope exception)
LOGEVERY = int(os.environ.get("LOGEVERY", 50)); WARM = int(os.environ.get("WARM", 20)); DFRAC = 0.5
D, H, FF, V = L.D, L.H, L.FF, L.V
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("FOUT", "long.jsonl"))


class Block(nn.Module):
    def __init__(s):
        super().__init__(); Li = lambda i, o: nn.Linear(i, o, bias=False)
        s.n1, s.n2 = L.RMS(D), L.RMS(D)
        if QKNORM == 1: s.qn, s.kn = L.RMS(D // H), L.RMS(D // H)
        if QKNORM == 2: s.qn, s.kn = L.RMS(D), L.RMS(D)   # layer-wise (OLMo-style): one norm over all heads
        s.q, s.k, s.v, s.o, s.gate, s.up, s.down = Li(D, D), Li(D, D), Li(D, D), Li(D, D), Li(D, FF), Li(D, FF), Li(FF, D)
        if GATE: s.g = Li(D, D)
        s.rec = False; s.maxlogit = 0.0
    def forward(s, x):
        b, t, _ = x.shape; h = s.n1(x); sp = lambda z: z.view(b, t, H, D // H).transpose(1, 2)
        if QKNORM == 2: q, k = sp(s.qn(s.q(h))), sp(s.kn(s.k(h)))
        else: q, k = sp(s.q(h)), sp(s.k(h))
        if QKNORM == 1: q, k = s.qn(q), s.kn(k)
        q, k = L.rope(q), L.rope(k)
        if s.rec:                                                        # explicit scores, only on probe passes
            with torch.no_grad():
                sc = (q @ k.transpose(-1, -2)) / math.sqrt(D // H); mk = torch.ones(t, t, dtype=torch.bool).tril()
                ph = sc.masked_fill(~mk, 0).abs().amax((0, 2, 3)); s.maxhead = ph.tolist(); s.maxlogit = ph.max().item(); s.phead = ph
                pa = sc.masked_fill(~mk, float('-inf')).softmax(-1); ent = -(pa * pa.clamp_min(1e-12).log()).sum(-1)[:, :, 4:].mean((0, 2)); s.ent = ent.tolist()   # mean attention entropy per head (positions >= 4)
        if SOFTCAP:
            sc = (q @ k.transpose(-1, -2)) / math.sqrt(D // H); sc = SOFTCAP * torch.tanh(sc / SOFTCAP)
            sc = sc.masked_fill(~torch.ones(t, t, dtype=torch.bool).tril(), float("-inf")); a = sc.softmax(-1) @ sp(s.v(h))
        else: a = F.scaled_dot_product_attention(q, k, sp(s.v(h)), is_causal=True)
        if GATE: a = a * torch.sigmoid(sp(s.g(h)))
        x = x + s.o(a.transpose(1, 2).reshape(b, t, D)); h = s.n2(x)
        return x + s.down(F.silu(s.gate(h)) * s.up(h))


class LM(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(V, D); s.blocks = nn.ModuleList(Block() for _ in range(L.NL))
        s.norm = L.RMS(D); s.head = nn.Linear(D, V, bias=False)
    def forward(s, x):
        h = s.emb(x)
        for b in s.blocks: h = b(h)
        return s.head(s.norm(h))
    def loss(s, x): return F.cross_entropy(s(x[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1))


@torch.no_grad()
def probe(m, x):
    for b in m.blocks: b.rec = True
    m(x[:, :-1])
    for b in m.blocks: b.rec = bool(QKCLIP)
    return [b.maxlogit for b in m.blocks]


def train(tr, ev, mask, lr, seed, steps):
    torch.manual_seed(seed); m = LM(); mats, rest = L.split(m)
    pb = torch.cat([v[:16] for v in ev.values()])                       # fixed probe batch: 16 sequences of each of the 20 types
    buf = [torch.zeros_like(p) for p in mats]; ex = [torch.zeros_like(p) for p in rest]; ex2 = [torch.zeros_like(p) for p in rest]
    for b in m.blocks: b.rec = bool(QKCLIP)
    nclip = 0; order = torch.randperm(steps * L.B, generator=torch.Generator().manual_seed(seed)); log = []
    for i in range(steps):
        if i % LOGEVERY == 0: log.append([i] + probe(m, pb))
        f = (min(1.0, (i + 1) / WARM)) * (min(1.0, (1 - i / steps) / DFRAC) if DECAY else 1.0)
        loss = m.loss(tr[order[i * L.B:(i + 1) * L.B]]); m.zero_grad(set_to_none=True); loss.backward()
        with torch.no_grad():
            u = []
            for p, b in zip(mats, buf): b.mul_(MOM).add_(p.grad); g_ = p.grad + MOM * b; u.append(g_.T if g_.shape[0] > g_.shape[1] else g_)
            for shp in {x.shape for x in u}:
                ix = [j for j, x in enumerate(u) if x.shape == shp]; O = L.ns(torch.stack([u[j] for j in ix]))
                for j, o in zip(ix, O):
                    p = mats[j]
                    if WD: p.mul_(1 - lr * f * WD)
                    p.add_(o.T if p.shape[0] > p.shape[1] else o, alpha=-lr * f * max(1, p.shape[0] / p.shape[1]) ** 0.5)
            if QKCLIP and i < CLIPSTOP:                                  # QK-Clip after the step, using this step's training-batch per-head max logit
                for b in m.blocks:
                    g = (QKCLIP / b.phead).clamp(max=1.0); nclip += int((g < 1).sum()); sg = g.sqrt().repeat_interleave(D // H)[:, None]
                    b.q.weight.mul_(sg); b.k.weight.mul_(sg)
            for p, e, e2 in zip(rest, ex, ex2):
                e.mul_(0.9).add_(p.grad, alpha=0.1); e2.mul_(0.98).add_(p.grad ** 2, alpha=0.02)
                p.add_(e / (1 - 0.9 ** (i + 1)) / ((e2 / (1 - 0.98 ** (i + 1))).sqrt() + 1e-8), alpha=-L.OTHER_LR * f)
    log.append([steps] + probe(m, pb)); ents = [[round(v, 3) for v in b.ent] for b in m.blocks]; heads = [[round(v, 2) for v in b.maxhead] for b in m.blocks]
    sv = [max(torch.linalg.matrix_norm(b.q.weight.detach(), 2).item(), 0) for b in m.blocks] + [torch.linalg.matrix_norm(b.k.weight.detach(), 2).item() for b in m.blocks]
    ls = E.evaluate(m, ev, mask); return sum(ls.values()) / len(ls), log, sv, heads, nclip, ents


if __name__ == "__main__":
    lrs = [float(x) for x in sys.argv[1].split(",")]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    tr, ev, mask = E.data()
    for lr in lrs:
        t0 = time.time(); loss, log, sv, heads, nclip, ents = train(tr, ev, mask, lr, seed, E.STEPS)
        row = dict(clipstop=CLIPSTOP, ents=ents, warm=WARM, softcap=SOFTCAP, gate=GATE, qkclip=QKCLIP, nclip=nclip, qknorm=QKNORM, D=D, B=L.B, steps=E.STEPS, mom=MOM, wd=WD, decay=DECAY, heads=heads, lr=lr, seed=seed, loss=round(loss, 5),
                   maxlogit_final=round(max(log[-1][1:]), 3), maxlogit_peak=round(max(max(r[1:]) for r in log), 3),
                   log=[[r[0]] + [round(v, 3) for v in r[1:]] for r in log], qk_specnorm=[round(v, 3) for v in sv], sec=round(time.time() - t0, 2))
        open(OUT, "a").write(json.dumps(row) + "\n")
        print(f"SOFTCAP={SOFTCAP} GATE={GATE} QKCLIP={QKCLIP} nclip={nclip} QKNORM={QKNORM} D={D} steps={E.STEPS} mom={MOM} lr={lr}: loss {loss:.4f} peak logit {row['maxlogit_peak']} final {row['maxlogit_final']} minent {min(min(e) for e in ents):.3f} ({row['sec']}s)", flush=True)
