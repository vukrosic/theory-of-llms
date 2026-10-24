"""Attention sinks and huge activations in the 8-skill toy (toy-suite/sampled.py, default MoE command).

BOS=1 (default) puts one fixed start token before every sequence (the skill token moves to position 1), so the
start token carries no information and attention on it is a pure sink. Trains one model, then on 128 sequences
per skill measures, from real (non-padding) queries at positions >= 2:
  sink     attention on the start token, per layer x head (mean over skills, weighted by the eval mix)
  ratio    share of heads with sink > 0.3 (Gu et al. 2024)
  v0       value size at the start token / other tokens, per layer (small = "value drain")
  r0       residual size at the start token / median of other tokens, after the last layer
  big      largest |activation| / median |activation| after each layer
  drop0    loss rise when no query may look at the start token (its KV evicted, as in StreamingLLM)
  q8, q4   loss rise when every linear layer's input is rounded to 8 / 4 bits (one scale per tensor)
  cdim     after the last layer, mean |x| on the largest dimension / median |x|; cshare = share of tokens where it is the largest
SM1=1 adds an always-visible extra key with score 0 and value 0 (softmax + 1); ATTN_GATE=1 is suite's Qwen gate.
usage: MOE=8 MSIGN=ns EMB_MULT=8 python3 toyprobe.py SEED [STEPS LR]  -> json line to $OUT (default toy_runs.jsonl)
"""
import json, math, os, sys, time
sys.path.insert(0, os.path.expanduser("~/my-life/openai-anthropic-goal/muon/toy-suite"))
import torch, torch.nn.functional as F
import sampled, mix, suite

BOS, SM1 = os.environ.get("BOS", "1") == "1", os.environ.get("SM1", "0") == "1"
st = dict(rec=False, block0=False, keep=[])


def sdpa(q, k, v, is_causal=True, scale=None, attn_mask=None):
    t = q.shape[2]; z = (q @ k.transpose(-1, -2)) * (scale or 1 / math.sqrt(q.shape[-1]))
    mk = torch.ones(t, t, dtype=torch.bool).tril()
    if st["block0"]: mk[1:, 0] = False                                   # later queries cannot see token 0
    z = z.masked_fill(~mk, float("-inf"))
    if SM1: z = torch.cat([torch.zeros_like(z[..., :1]), z], -1)          # extra key: score 0, value 0
    p = z.softmax(-1)
    if st["rec"]: st["keep"].append(p)
    return (p[..., 1:] if SM1 else p) @ v


suite.F = type(F)("Fshim"); suite.F.__dict__.update(F.__dict__); suite.F.scaled_dot_product_attention = sdpa
if BOS:
    BOS_ID = suite.V; suite.V += 1; _gen = sampled.gen

    def gen(skill, n, g):
        s, m = _gen(skill, n, g)
        return (torch.cat([torch.full((n, 1), BOS_ID), s[:, :-1]], 1), torch.cat([torch.zeros(n, 1, dtype=torch.bool), m[:, :-1]], 1))
    sampled.gen = mix.gen = gen

if suite.HBIAS in ("log", "loginit"):                                   # same log-frequency logit bias as sampled.py's main
    gc, cnt = torch.Generator().manual_seed(777), torch.ones(suite.V)
    for _ in range(200): _, yb, mb = sampled.batch(32, gc); cnt += torch.bincount(yb[mb], minlength=suite.V)[: suite.V]
    lp = (cnt / cnt.sum()).log(); suite.HB_INIT = lp - lp.mean()
seed = int(sys.argv[1]); steps = int(sys.argv[2]) if len(sys.argv) > 2 else 260; lr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.014
t0 = time.time(); rows = mix.run("whole", seed, steps, lr, every=steps, bs=32); sec = time.time() - t0
m = mix.last_model; last = rows[-1][1]; W = sampled.W
wl = sum(w.item() * last[sk][0] for w, sk in zip(W, sampled.SKILLS))
res = []
for l in m.layers: l.register_forward_hook(lambda mod, i_, o: res.append(o))
sink, sk1, vr, big, rr, wsum, drop0 = 0, 0, 0, [], 0, 0, 0.0
o0 = 1 if SM1 else 0
with torch.no_grad():
    for w, sk in zip(W, sampled.SKILLS):
        st["keep"].clear(); res.clear(); st["rec"] = True
        s, msk = sampled.gen(sk, 128, torch.Generator().manual_seed(999)); x = s[:, :-1]; y = s[:, 1:]; o = m(x); st["rec"] = False
        st["block0"] = True; drop0 += w.item() * (F.cross_entropy(m(x)[msk], y[msk]).item() - F.cross_entropy(o[msk], y[msk]).item()); st["block0"] = False
        big.append([(r.abs().amax() / r.abs().median()).item() for r in res])
        ok = (x != sampled.PAD); ok[:, :2] = False
        if ok.sum() == 0: continue                                       # skills that end early still count for drop0 and big
        wq = ok.float() / ok.sum()
        sink = sink + w * torch.stack([(p[..., o0] * wq[:, None]).sum((0, 2)) for p in st["keep"]])        # layers x heads
        sk1 = sk1 + w * torch.stack([(p[..., o0 + 1] * wq[:, None]).sum((0, 2)) for p in st["keep"]])
        if SM1: vr = vr + w * torch.stack([(p[..., 0] * wq[:, None]).sum((0, 2)) for p in st["keep"]])     # on the extra key
        rn = res[-1].norm(dim=-1); rr += w.item() * (rn[:, 0] / rn[:, 1:].median(-1).values).mean().item()
        wsum += w.item()
sink, sk1, rr = sink / wsum, sk1 / wsum, rr / wsum
qb = [0]                                                                 # fake-quantize every linear layer's input per tensor (absmax) to qb bits


def fq(mod, inp):
    if not qb[0]: return None
    x = inp[0]; sc = x.abs().amax().clamp_min(1e-8) / (2 ** (qb[0] - 1) - 1); return ((x / sc).round() * sc,)


for mod in m.modules():
    if isinstance(mod, torch.nn.Linear): mod.register_forward_pre_hook(fq)
qd, cd = {}, []
with torch.no_grad():
    for bits in (8, 4):
        tot = 0.0
        for w, sk in zip(W, sampled.SKILLS):
            s, msk = sampled.gen(sk, 128, torch.Generator().manual_seed(999)); x = s[:, :-1]; y = s[:, 1:]
            qb[0] = 0; l0 = F.cross_entropy(m(x)[msk], y[msk]).item(); qb[0] = bits; tot += w.item() * (F.cross_entropy(m(x)[msk], y[msk]).item() - l0)
        qd[bits] = round(tot, 4)
    qb[0] = 0; res.clear(); s, _ = sampled.gen("chain", 128, torch.Generator().manual_seed(5)); m(s[:, :-1])
    a = res[-1].abs(); d = a.amax((0, 1)).argmax()                       # constant outlier dimension after the last layer
    cd = [round((a[..., d].mean() / a.median()).item(), 1), round((a.argmax(-1) == d).float().mean().item(), 3)]
v0 = [(l.vout[:, :, 0].norm(dim=-1).mean() / l.vout[:, :, 1:].norm(dim=-1).mean()).item() for l in m.layers]  # last skill's batch
r = dict(env={k: os.environ[k] for k in ("MOE", "MSIGN", "EMB_MULT", "WIDTH", "ATTN_GATE", "QKNORM", "WD", "ATT_MOM", "HEADS", "BOS", "SM1", "HBIAS", "NORM_MULT") if k in os.environ},
         seed=seed, steps=steps, lr=lr, wloss=round(wl, 4), sink_mean=round(sink.mean().item(), 3),
         sink_lh=[[round(x, 3) for x in row] for row in sink.tolist()], ratio=round((sink > 0.3).float().mean().item(), 3),
         skill_tok=round(sk1.mean().item(), 3), extra=round((vr / wsum).mean().item(), 3) if SM1 else None,
         big=[round(x, 1) for x in torch.tensor(big).amax(0).tolist()], v0=[round(x, 3) for x in v0], r0=round(rr, 2),
         drop0=round(drop0, 4), q8=qd[8], q4=qd[4], cdim=cd[0], cshare=cd[1], sec=round(sec, 1))
with open(os.environ.get("OUT", "toy_runs.jsonl"), "a") as f: f.write(json.dumps(r) + "\n")
print(json.dumps(r))
