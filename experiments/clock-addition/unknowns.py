"""Answer the open questions about a trained adder. usage: P=47 python3 unknowns.py runs/X.pt
1. clock or pizza: correct answer's margin vs (a-b). clock -> flat; pizza -> follows |cos(pi k (a-b)/P)| of the main waves
2. the 23% of scores that are not a function of (a+b-c): how much is a function of (a, c) + (b, c) (each number pushing answers by itself)
3. knockout gap: wave-22 knockout drop with the final norm scale frozen (pure linear) vs real
"""
import sys, math, torch, torch.nn.functional as F
import mod
P = mod.P; K = 22; NS = [84, 100, 76]
m = mod.Model(); m.load_state_dict(torch.load(sys.argv[1])); B = m.blk
x, y, _, _ = mod.data(0); n = torch.arange(P)


@torch.no_grad()
def forward(kill=None, sc=None):
    h = m.emb(x); r1 = h + B.attn(B.n1(h)); z = B.n2(r1); hid = F.silu(B.gate(z)) * B.up(z)
    if kill is not None: hid[:, :, kill] = 0
    r2 = (r1 + B.down(hid))[:, -1]
    s = sc if sc is not None else m.norm.weight / r2.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    return (r2 * s) @ m.head.weight.T, s


L, sc = forward(); Lc = (L - L.mean(1, keepdim=True)).view(P, P, P)
a, b, c = torch.meshgrid(n, n, n, indexing="ij"); d = (a + b - c) % P
margin = lambda L_: L_.gather(1, y[:, None])[:, 0] - L_.mean(1)

# 1. clock vs pizza
mg = margin(L).view(P, P); diff = ((a[:, :, 0] - b[:, :, 0]) % P)
by_diff = torch.zeros(P).index_add_(0, diff.flatten(), mg.flatten()) / P
g = torch.zeros(P).index_add_(0, d.flatten(), Lc.flatten()) / (P * P); G = torch.fft.rfft(g)
top = (G.abs()[1:].argsort(descending=True)[:5] + 1).tolist(); w = G.abs()[top]
pizza = sum(wi * torch.cos(math.pi * k * n / P).abs() for k, wi in zip(top, w))
r = torch.corrcoef(torch.stack([by_diff, pizza]))[0, 1].item()
print(f"1. margin by (a-b): mean {by_diff.mean():.2f}, spread (std/mean) {by_diff.std() / by_diff.mean():.2f}, corr with pizza pattern {r:.2f}")

# 2. leftover
fit = g[d]; R = Lc - fit
ac = R.mean(1, keepdim=True); ac = ac - ac.mean(2, keepdim=True)             # f(a, c)
bc = (R - ac).mean(0, keepdim=True)                                           # g(b, c)
tot = (Lc ** 2).sum()
print(f"2. scores: wave part {1 - (R ** 2).sum() / tot:.2f}; + f(a,c) + g(b,c) {1 - ((R - ac - bc) ** 2).sum() / tot:.2f}")
R2 = R - ac - bc; s2 = R2.pow(2).sum()
gd = torch.zeros(P).index_add_(0, ((a - b - c) % P).flatten(), R2.flatten()) / (P * P)
print(f"   rest as function of (a-b-c): {1 - ((R2 - gd[(a - b - c) % P]) ** 2).sum() / s2:.2f} of the rest; rest margin share {(R2.view(-1, P).gather(1, y[:, None]).abs().mean() / margin(L).mean()):.2f}")

# 3. knockout gap
pred_drop = 2 * G[K].abs().item() / P * (1 + 1 / (P - 1))
Lr, _ = forward(NS); Lf, _ = forward(NS, sc)
print(f"3. wave-{K} wave size {pred_drop:.2f}; drop with norm frozen {(margin(L) - margin(Lf)).mean():.2f}, real drop {(margin(L) - margin(Lr)).mean():.2f}")
Lk = L - Lf                                                                   # what the 3 neurons add (linear)
Lkc = (Lk - Lk.mean(1, keepdim=True)).view(P, P, P)
gk = torch.zeros(P).index_add_(0, d.flatten(), Lkc.flatten()) / (P * P); Gk = torch.fft.rfft(gk).abs()
print(f"   3 neurons' push: in wave {K} {Gk[K] ** 2 / (Gk[1:] ** 2).sum():.2f} of its (a+b-c) part; (a+b-c) part is {1 - ((Lkc - gk[d]) ** 2).sum() / (Lkc ** 2).sum():.2f} of the push")
