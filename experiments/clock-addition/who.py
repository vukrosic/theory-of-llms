"""Which neurons build the wave-K output? Each neuron's linear share of the scores, projected on cos/sin(2 pi K (a+b-c) / P).
usage: P=47 python3 who.py runs/X.pt K"""
import sys, math, torch, torch.nn.functional as F
import mod
P = mod.P; K = int(sys.argv[2]); m = mod.Model(); m.load_state_dict(torch.load(sys.argv[1])); B = m.blk
x, y, _, _ = mod.data(0)
with torch.no_grad():
    h = m.emb(x); r1 = h + B.attn(B.n1(h)); z = B.n2(r1); hf = (F.silu(B.gate(z)) * B.up(z))[:, -1]; r2 = (r1 + B.down(hf.unsqueeze(1)).squeeze(1)[:, None])[:, -1] if False else (r1[:, -1] + B.down(hf))
    sc = m.norm.weight / r2.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    out = (B.down.weight * 1)                                         # D x N
    W = (sc[:, :, None] * out[None]).transpose(1, 2) @ m.head.weight.T  # rows x N x P: per-neuron score change per unit activation
    contrib = hf[:, :, None] * W                                      # rows x N x P
c = torch.arange(P); d = (x[:, 0, None] + x[:, 1, None] - c[None]) % P
cs, sn = torch.cos(2 * math.pi * K * d / P), torch.sin(2 * math.pi * K * d / P)
amp = ((contrib * cs[:, None]).mean((0, 2)) ** 2 + (contrib * sn[:, None]).mean((0, 2)) ** 2).sqrt()   # wave-K amplitude per neuron
proj = (contrib * cs[:, None]).mean((0, 2)); tot = proj.sum()          # cos part = the part that peaks at the right answer
o = proj.argsort(descending=True); cum = proj[o].cumsum(0) / tot
need = lambda f: int((cum >= f).nonzero()[0]) + 1 if (cum >= f).any() else None
print(f"wave {K}: total peak height {tot * 2:.2f}; neurons needed for 50% / 80% / 100%: {need(0.5)} / {need(0.8)} / {need(1.0)} of {len(proj)};"
      f" top 5 neurons give {proj[o[:5]].sum() / tot:.2f}; neurons pushing against it: {(proj < 0).sum().item()}")
H = hf.view(P, P, -1); H = H - H.mean((0, 1)); H2 = torch.fft.fft2(H, dim=(0, 1)).abs() ** 2; T = H2.sum((0, 1))
q = lambda i, j: H2[i % P, j % P] + H2[-i % P, -j % P]
for n in o[:3].tolist():
    best = max(range(1, P // 2 + 1), key=lambda k: (q(k, 0) + q(0, k) + q(k, k) + q(k, -k))[n])
    print(f"   neuron {n}: {proj[n] / tot:.0%} of wave {K}; its own main wave {best}: a alone {q(best, 0)[n] / T[n]:.2f}, b alone {q(0, best)[n] / T[n]:.2f}, "
          f"a+b {q(best, best)[n] / T[n]:.2f}, a-b {q(best, -best)[n] / T[n]:.2f}")
