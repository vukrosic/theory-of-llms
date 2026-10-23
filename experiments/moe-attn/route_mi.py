"""What does each MoE layer's router key on at fact-answer positions: the subject (context, via attention) or the relation?
Mutual information (bits) between the top-1 expert and the subject / the relation, on 2048 fact sequences.
usage: [same env as moe.py] python3 route_mi.py LR SEED"""
import sys, torch, math
import moe


def mi(a, b):
    j = torch.zeros(int(a.max()) + 1, int(b.max()) + 1); j.index_put_((a, b), torch.ones(len(a)), accumulate=True); j /= j.sum()
    pa, pb = j.sum(1, keepdim=True), j.sum(0, keepdim=True); nz = j > 0
    return (j[nz] * (j[nz] / (pa * pb).expand_as(j)[nz]).log2()).sum().item()


lr, seed = float(sys.argv[1]), int(sys.argv[2]); m = moe.train(lr, seed); ev = moe.evaluate(m)
old = moe.COPYW; moe.COPYW = 0.0
x, msk, _ = moe.gen(2048, torch.Generator().manual_seed(5)); moe.COPYW = old
with torch.no_grad(): m(x[:, :-1])
subj = x[:, :1].expand(-1, moe.T)[msk]
rel = (x[:, :-1] if moe.GENERIC == 0 else torch.cat([x[:, :1], x[:, :-2]], 1))[msk] - moe.R0   # relation token: previous token (or 2 back with GENERIC)
print({k: v for k, v in ev.items() if "facts" in k}, f"H(subject) {math.log2(moe.NS):.1f} bits, H(relation) {math.log2(moe.NR):.1f}")
for i, b in enumerate(m.blocks):
    if not b.is_moe: continue
    top1 = b.moe.top[:, 0].view(x.shape[0], moe.T)[msk]
    print(f"layer {i}: I(expert; subject) {mi(top1, subj):.2f}  I(expert; relation) {mi(top1, rel):.2f}  bits (max {math.log2(moe.MOE):.0f})")
