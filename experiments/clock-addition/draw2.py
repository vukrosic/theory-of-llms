"""More pictures of the trained adder. usage: P=47 python3 draw2.py runs/X.pt -> path22.png, reorder.png, neuron_waves.png, knockout.png"""
import sys, math, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import mod
P = mod.P; K = 22; NS = [84, 100, 76]
m = mod.Model(); m.load_state_dict(torch.load(sys.argv[1])); B = m.blk
x, y, _, _ = mod.data(0); n = torch.arange(P)


@torch.no_grad()
def forward(kill=None):
    h = m.emb(x); r1 = h + B.attn(B.n1(h)); z = B.n2(r1); hid = F.silu(B.gate(z)) * B.up(z)
    if kill is not None: hid[:, :, kill] = 0
    r2 = r1 + B.down(hid); return r2[:, -1], hid[:, -1], m.head(m.norm(r2))[:, -1]


r2, hid, L = forward(); H = hid.view(P, P, -1)

# 1. one path: wave 22 of the numbers -> 3 neurons -> wave in the answer scores
fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
E = m.emb.weight.detach()[:P]; E = E - E.mean(0); Fk = torch.zeros(P // 2 + 1, E.shape[1], dtype=torch.cfloat); Fk[K] = torch.fft.rfft(E, dim=0)[K]
Ek = torch.fft.irfft(Fk, n=P, dim=0); U, S, _ = torch.linalg.svd(Ek, full_matrices=False); pos = (K * n) % P; o = pos.argsort()
ax[0].plot(pos[o], (U[:, 0] * S[0])[o], "o-", label="direction 1"); ax[0].plot(pos[o], (U[:, 1] * S[1])[o], "o-", label="direction 2")
ax[0].set_title(f"1. each number's wave-{K} part\n(numbers sorted by {K}*n mod {P})"); ax[0].set_xlabel(f"{K} * number mod {P}"); ax[0].legend(fontsize=8)
s = ((n[:, None] + n[None]) % P).flatten()
for nn in NS: ax[1].scatter(((K * s) % P), H[:, :, nn].flatten(), s=3, label=f"neuron {nn}")
ax[1].set_title(f"2. the 3 neurons vs (a + b)\n(x = {K}*(a+b) mod {P})"); ax[1].set_xlabel(f"{K} * (a + b) mod {P}"); ax[1].legend(fontsize=8, markerscale=4)
with torch.no_grad():
    sc = m.norm.weight / r2.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt(); keep = torch.zeros(H.shape[-1]); keep[NS] = 1
    C = (B.down(hid * keep) * sc) @ m.head.weight.T
C = C - C.mean(1, keepdim=True); d = (x[:, 0, None] + x[:, 1, None] - n[None]) % P
g = torch.zeros(P).index_add_(0, d.flatten(), C.flatten()) / (P * P)
ax[2].plot(n, g, "o-", color="#222", label="3 neurons' push on answer c"); ax[2].plot(n, g[0] * torch.cos(2 * math.pi * K * n / P) * 0 + (g @ torch.cos(2 * math.pi * K * n / P)) * 2 / P * torch.cos(2 * math.pi * K * n / P), color="#e4572e", label=f"cos(2pi*{K}(a+b-c)/{P})")
ax[2].set_title("3. what they add to each answer's score"); ax[2].set_xlabel("a + b - c  (0 = correct answer)"); ax[2].legend(fontsize=8)
accs = []
for kill in [None, NS, [5, 50, 111]]:
    _, _, Lx = forward(kill); accs.append((Lx.argmax(1) == y).float().mean().item())
ax[3].bar(["full model", f"remove neurons\n{NS}", "remove 3 random\nneurons"], accs, color=["#888", "#e4572e", "#4c9be8"])
for i, v in enumerate(accs): ax[3].text(i, v + 0.01, f"{v:.0%}", ha="center")
ax[3].set_ylim(0, 1.1); ax[3].set_title("4. accuracy on all a + b pairs")
plt.tight_layout(); plt.savefig("path22.png", dpi=110); plt.close()

# 2. reorder: the same neuron map looks like noise in number order, smooth stripes in wave order
fig, ax = plt.subplots(1, 2, figsize=(11, 5)); nn = NS[0]; o = ((K * n) % P).argsort()
ax[0].imshow(H[:, :, nn], cmap="RdBu"); ax[0].set_title(f"neuron {nn} over all (a, b), numbers in normal order"); ax[0].set_xlabel("b"); ax[0].set_ylabel("a")
ax[1].imshow(H[o][:, o, nn], cmap="RdBu"); ax[1].set_title(f"same neuron, numbers sorted by {K}*n mod {P}"); ax[1].set_xlabel("b (re-sorted)"); ax[1].set_ylabel("a (re-sorted)")
plt.tight_layout(); plt.savefig("reorder.png", dpi=110); plt.close()

# 3. which wave every neuron uses
Hc = H - H.mean((0, 1)); H2 = torch.fft.fft2(Hc, dim=(0, 1)).abs() ** 2; T = H2.sum((0, 1)).clamp_min(1e-12)
q = lambda i, j: H2[i % P, j % P] + H2[-i % P, -j % P]
M = torch.stack([(q(k, 0) + q(0, k) + q(k, k) + q(k, -k)) / T for k in range(1, P // 2 + 1)])   # freq x neuron
o = (M.argmax(0) * 10 - M.max(0).values).argsort()
fig, ax = plt.subplots(figsize=(14, 4.5)); im = ax.imshow(M[:, o], aspect="auto", cmap="magma", extent=[0, M.shape[1], P // 2 + 0.5, 0.5])
ax.set_xlabel("128 neurons (grouped by their main wave)"); ax.set_ylabel("wave frequency"); ax.set_title("share of each neuron's activity in each wave (bright = one clean wave)")
plt.colorbar(im); plt.tight_layout(); plt.savefig("neuron_waves.png", dpi=110); plt.close()

# 4. knock out each wave's top neurons: how much each wave matters
ks = [13, 3, 22, 20, 11]; res = []
for k in ks:
    grp = ((M.argmax(0) + 1 == k) & (M.max(0).values > 0.5)).nonzero()[:, 0].tolist()
    _, _, Lx = forward(grp); res.append(((Lx.argmax(1) == y).float().mean().item(), len(grp)))
fig, ax = plt.subplots(figsize=(8, 4)); ax.bar([f"wave {k}\n({c} neurons)" for k, (a, c) in zip(ks, res)], [a for a, c in res], color="#e4572e")
ax.axhline(accs[0], color="#888", ls="--", label=f"full model {accs[0]:.0%}")
for i, (a, c) in enumerate(res): ax.text(i, a + 0.01, f"{a:.0%}", ha="center")
ax.set_ylim(0, 1.1); ax.legend(); ax.set_title("accuracy after removing the neurons that mainly use one wave")
plt.tight_layout(); plt.savefig("knockout.png", dpi=110); print("done", accs, res)
