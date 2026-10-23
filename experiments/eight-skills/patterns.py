"""Which patterns does the 8-skill toy build inside? Trains the default toy (Muon), then looks inside.
usage: MOE=8 MSIGN=ns EMB_MULT=8 python3 patterns.py SEED [STEPS] [LR]   -> patterns_attn.png, patterns_experts.png, patterns_numbers.png
1. attention: per skill, how much each head looks 1 token back (previous-token head) and at the token after the last
   earlier copy of the current token (induction head)
2. experts: share of each skill's scored tokens sent to each expert, per layer
3. number lines: is word order (sort) a straight line in the embeddings, and is the count (0-6) a line in the hidden state
"""
import sys, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import suite, mix, sampled
from sampled import SKILLS, gen, NV, SEP

seed = int(sys.argv[1]); steps = int(sys.argv[2]) if len(sys.argv) > 2 else 260; lr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.014
rows = mix.run("whole", seed, steps, lr, every=steps, bs=32); m = mix.last_model; m.eval()
print("acc " + " ".join(f"{sk}:{rows[-1][1][sk][1]:.2f}" for sk in SKILLS))

ATT = []
sdpa = F.scaled_dot_product_attention
def keep(q, k, v, is_causal=True, scale=None, **kw):
    t = q.shape[-2]; p = ((q @ k.transpose(-1, -2)) * scale).masked_fill(~torch.ones(t, t, dtype=torch.bool).tril(), float("-inf")).softmax(-1)
    ATT.append(p); return p @ v
F.scaled_dot_product_attention = keep
ROUTE = []
for l in m.layers: l.moe.router.register_forward_hook(lambda mod_, i, o: ROUTE.append((o.softmax(-1) + mod_.bias_ref).topk(suite.MOE_K, -1).indices))
for l in m.layers: l.moe.router.bias_ref = l.moe.bias

g = torch.Generator().manual_seed(999); data = {sk: gen(sk, 256, g) for sk in SKILLS}
prev, induct, experts, hid = {}, {}, {}, {}
with torch.no_grad():
    for sk, (s, msk) in data.items():
        ATT.clear(); ROUTE.clear(); x = s[:, :-1]; b, t = x.shape
        h = m.emb(x); res = []
        for l in m.layers: h = l(h); res.append(h)
        hid[sk] = res
        A = torch.stack(ATT, 1)                                     # b, layer, head, t, t
        i = torch.arange(t)
        prev[sk] = A[:, :, :, i[1:], i[:-1]].permute(0, 3, 1, 2)[msk[:, 1:]].reshape(-1, 2 * suite.H).mean(0)
        tgt = torch.full((b, t), -1)                                # position after the last earlier copy of the current token
        for q in range(1, t):
            same = (x[:, :q] == x[:, q:q + 1]).float() * torch.arange(1, q + 1)
            j = same.argmax(1); ok = same.max(1).values > 0
            tgt[ok, q] = j[ok] + 1
        ok = msk & (tgt >= 0) & (tgt < torch.arange(t)[None] + 1)
        induct[sk] = A.permute(0, 3, 1, 2, 4)[ok].gather(-1, tgt[ok].view(-1, 1, 1, 1).expand(-1, 2, suite.H, 1))[..., 0].reshape(-1, 2 * suite.H).mean(0) if ok.any() else torch.full((2 * suite.H,), float("nan"))
        experts[sk] = torch.stack([F.one_hot(r.view(b, t, -1)[msk], suite.MOE).sum(1).float().mean(0) / suite.MOE_K for r in ROUTE])   # layer x expert

lab = [f"L{l} h{h}" for l in range(2) for h in range(suite.H)]
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
for a_, d_, ttl in ((ax[0], prev, "looks 1 token back (previous-token head)"), (ax[1], induct, "looks at the token after the earlier copy (induction head)")):
    M = torch.stack([d_[sk] for sk in SKILLS]); im = a_.imshow(M, cmap="viridis", vmin=0, vmax=1)
    a_.set_yticks(range(8), SKILLS); a_.set_xticks(range(len(lab)), lab); a_.set_title(ttl)
    for r in range(8):
        for c in range(len(lab)):
            if not torch.isnan(M[r, c]): a_.text(c, r, f"{M[r, c]:.2f}", ha="center", va="center", fontsize=7, color="w" if M[r, c] < 0.6 else "k")
plt.colorbar(im, ax=ax); plt.savefig("patterns_attn.png", dpi=110, bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for l in range(2):
    M = torch.stack([experts[sk][l] for sk in SKILLS]); ax[l].imshow(M, cmap="magma", vmin=0, vmax=1)
    ax[l].set_yticks(range(8), SKILLS); ax[l].set_xticks(range(suite.MOE), [f"e{e}" for e in range(suite.MOE)]); ax[l].set_title(f"layer {l}: share of each skill's tokens sent to each expert")
    for r in range(8):
        for c in range(suite.MOE): ax[l].text(c, r, f"{M[r, c]:.2f}", ha="center", va="center", fontsize=7, color="w" if M[r, c] < 0.6 else "k")
plt.tight_layout(); plt.savefig("patterns_experts.png", dpi=110); plt.close()

# 3. number lines
E = m.emb.weight.detach()[:NV]; ids = torch.arange(NV).float()
def line_r2(X, yv):                                                 # share of yv explained by a straight-line readout of X (half fit, half test)
    n = len(yv); p = torch.randperm(n, generator=torch.Generator().manual_seed(0)); a, b = p[: n // 2], p[n // 2:]
    X1 = torch.cat([X, torch.ones(n, 1)], 1); w = torch.linalg.lstsq(X1[a], yv[a, None]).solution
    return 1 - ((X1[b] @ w)[:, 0] - yv[b]).pow(2).sum().item() / (yv[b] - yv[b].mean()).pow(2).sum().item(), X1 @ w
r2_id, fit_id = line_r2(E, ids)
s, msk = data["count"]; pos = msk.float().argmax(1); cnt = s[torch.arange(len(s)), pos + 1].float()
Hc = hid["count"][1][torch.arange(len(s)), pos]; r2_c, fit_c = line_r2(Hc, cnt)
U, S, Vh = torch.linalg.svd(Hc - Hc.mean(0), full_matrices=False)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].scatter(ids, fit_id[:, 0], s=3, alpha=0.5); ax[0].plot([0, NV], [0, NV], "k--", lw=1)
ax[0].set_xlabel("word id (the sort order)"); ax[0].set_ylabel("word id read from its embedding by one straight line"); ax[0].set_title(f"word order is a line in the embeddings: {r2_id:.0%} explained")
sc = ax[1].scatter(U[:, 0] * S[0], U[:, 1] * S[1], c=cnt, cmap="viridis", s=12); plt.colorbar(sc, ax=ax[1], label="how many 0s")
ax[1].set_title(f"count task, hidden state at the answer: count read by a line {r2_c:.0%}"); ax[1].set_xlabel("main direction 1"); ax[1].set_ylabel("main direction 2")
plt.tight_layout(); plt.savefig("patterns_numbers.png", dpi=110); plt.close()
print("prev", {sk: round(v.max().item(), 2) for sk, v in prev.items()})
print("induct", {sk: round(v.max().item(), 2) for sk, v in induct.items()})
print(f"word-order line {r2_id:.2f}, count line {r2_c:.2f}")
