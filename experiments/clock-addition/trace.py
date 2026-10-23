"""Follow one wave through the trained model. usage: P=47 python3 trace.py runs/X.pt [K]
1. attention: is the '=' position's input to the MLP = f(a) + g(b)? (share explained), and how much of f is wave K
2. neurons: each neuron's (a, b) map split into wave terms: a, b alone / a+b / a-b, per frequency
3. output: the wave-K neurons' contribution to the answer scores vs cos(2 pi K (a+b-c) / P)
4. knock out the wave-K neurons: measured drop of the correct answer's score vs the wave-K size in the scores
"""
import sys, math, torch, torch.nn.functional as F
import mod
P = mod.P; K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
m = mod.Model(); m.load_state_dict(torch.load(sys.argv[1])); B = m.blk
x, y, _, _ = mod.data(0); A_, B_ = x[:, 0].view(P, P), x[:, 1].view(P, P)


@torch.no_grad()
def forward(kill=None):
    h = m.emb(x); r1 = h + B.attn(B.n1(h)); z = B.n2(r1); hid = F.silu(B.gate(z)) * B.up(z)
    if kill is not None: hid[:, :, kill] = 0
    r2 = r1 + B.down(hid); return r1[:, -1], hid[:, -1], m.head(m.norm(r2))[:, -1]


def additive_share(v):                                     # v: P x P x D; share of variance in f(a) + g(b)
    v = v - v.mean((0, 1)); fa, gb = v.mean(1, keepdim=True), v.mean(0, keepdim=True)
    return 1 - ((v - fa - gb) ** 2).sum().item() / (v ** 2).sum().item(), fa[:, 0], gb[0]


def wave_share(f, k):                                      # share of f (P x D, mean removed) in frequency k
    pw = (torch.fft.rfft(f - f.mean(0), dim=0).abs() ** 2).sum(1); return (pw[k] / pw[1:].sum()).item()


r1, hid, L = forward()
s, fa, gb = additive_share(r1.view(P, P, -1))
print(f"1. MLP input at '=' = f(a) + g(b): {s:.2f} of its variance;  wave {K} share of f: {wave_share(fa, K):.2f}, of g: {wave_share(gb, K):.2f}")

H = hid.view(P, P, -1); H = H - H.mean((0, 1)); H2 = torch.fft.fft2(H, dim=(0, 1)).abs() ** 2; tot = H2.sum((0, 1))
def terms(k):                                              # per neuron: power on a alone, b alone, a+b, a-b at frequency k
    q = lambda i, j: H2[i % P, j % P] + H2[-i % P, -j % P]
    return q(k, 0) + q(0, k), q(k, k), q(k, -k)
per = torch.stack([torch.stack(terms(k)).sum(0) for k in range(1, P // 2 + 1)]) / tot      # freq x neuron
dom = per.argmax(0) + 1; share = per.max(0).values
grp = ((dom == K) & (share > 0.5)).nonzero()[:, 0]
single, plus, minus = (t[grp].sum() / tot[grp].sum() for t in terms(K))
print(f"2. neurons whose main wave is {K} (>50%): {len(grp)} of {H.shape[-1]};  their power: a or b alone {single:.2f}, a+b {plus:.2f}, a-b {minus:.2f}")
print(f"   main wave of every neuron with >50% in one wave: {torch.bincount(dom[share > 0.5], minlength=P // 2 + 1)[1:].tolist()}")

with torch.no_grad():                                      # wave-K neurons' share of the scores, linear: down -> final norm (its scale from the full model) -> head
    h = m.emb(x); r1f = h + B.attn(B.n1(h)); z = B.n2(r1f); hf = F.silu(B.gate(z)) * B.up(z); r2 = (r1f + B.down(hf))[:, -1]
    sc = m.norm.weight / r2.pow(2).mean(-1, keepdim=True).add(m.norm.eps or 1e-6).sqrt()
    keep = torch.zeros(H.shape[-1]); keep[grp] = 1
    Lk = (B.down(hf[:, -1] * keep) * sc) @ m.head.weight.T; L0 = torch.zeros_like(Lk)
C = (Lk - L0).view(P, P, P); C = C - C.mean(2, keepdim=True)
a, b, c = torch.meshgrid(torch.arange(P), torch.arange(P), torch.arange(P), indexing="ij"); d = (a + b - c) % P
basis = torch.stack([torch.cos(2 * math.pi * K * d / P), torch.sin(2 * math.pi * K * d / P)], -1).view(-1, 2)
coef = torch.linalg.lstsq(basis, C.flatten()[:, None]).solution; fit = (basis @ coef).view(P, P, P)
print(f"3. wave-{K} neurons alone -> scores: {1 - ((C - fit) ** 2).sum() / (C ** 2).sum():.2f} explained by one wave cos(2pi {K}(a+b-c)/{P})")

Lc = L.view(P, P, P) - L.view(P, P, P).mean(2, keepdim=True)
g = torch.zeros(P).index_add_(0, d.flatten(), Lc.flatten()) / (P * P); G = torch.fft.rfft(g)
pred = 2 * G[K].abs().item() / P * (1 + 1 / (P - 1))       # wave-K height at the correct answer above the average answer
_, _, Lx = forward(kill=grp.tolist())
corr = lambda L_: (L_.gather(1, y[:, None])[:, 0] - L_.mean(1))
acc = lambda L_: (L_.argmax(1) == y).float().mean().item()
print(f"4. knock out the {len(grp)} wave-{K} neurons: correct answer's score above average drops {corr(L).mean():.2f} -> {corr(Lx).mean():.2f} "
      f"(drop {(corr(L) - corr(Lx)).mean():.2f}; wave-{K} size in the scores {pred:.2f});  accuracy {acc(L):.3f} -> {acc(Lx):.3f}")
