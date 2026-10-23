"""Pictures of the waves in a trained model. usage: P=47 python3 draw.py runs/X.pt  -> waves.png"""
import sys, math, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import mod
P = mod.P
m = mod.Model(); m.load_state_dict(torch.load(sys.argv[1]))
with torch.no_grad():
    x, *_ = mod.data(0); L = m(x).view(P, P, P); hid = m.blk.hid[:, -1].view(P, P, -1)
a, b, c = torch.meshgrid(torch.arange(P), torch.arange(P), torch.arange(P), indexing="ij")
Lc = L - L.mean(2, keepdim=True)
g = torch.zeros(P).index_add_(0, ((a + b - c) % P).flatten(), Lc.flatten()) / (P * P)
G = torch.fft.rfft(g); top = (G.abs()[1:].argsort(descending=True)[:5] + 1).tolist()
G5 = torch.zeros_like(G); G5[top] = G[top]; g5 = torch.fft.irfft(G5, n=P)

fig = plt.figure(figsize=(15, 9))
ax = fig.add_subplot(2, 3, (1, 2)); n = torch.arange(P)
ax.plot(n, g, "o-", color="#222", label="model's score for answer c")
ax.plot(n, g5, color="#e4572e", lw=2, label=f"sum of 5 waves (frequencies {sorted(top)})")
ax.set_xlabel("a + b - c  (mod 47);  0 = the correct answer"); ax.set_ylabel("score"); ax.legend(); ax.set_title("The answer score is a sum of 5 sine waves that all peak at the correct answer")
ax = fig.add_subplot(2, 3, 3)
for k in top: ax.plot(n, torch.fft.irfft(torch.where(torch.arange(len(G)) == k, G, torch.zeros_like(G)), n=P), label=f"k={k}")
ax.set_title("the 5 waves one by one"); ax.set_xlabel("a + b - c"); ax.legend(fontsize=8)
E = m.emb.weight.detach()[:P]; E = E - E.mean(0); F = torch.fft.rfft(E, dim=0)
for j, k in enumerate(top[:2]):                                     # numbers placed on the plane of one frequency
    ax = fig.add_subplot(2, 3, 4 + j); Fk = torch.zeros_like(F); Fk[k] = F[k]; Ek = torch.fft.irfft(Fk, n=P, dim=0)   # the frequency-k part of the embeddings
    U, S, Vh = torch.linalg.svd(Ek, full_matrices=False); px, py = U[:, 0] * S[0], U[:, 1] * S[1]
    ax.scatter(px, py, c=(n * k) % P, cmap="hsv", s=60)
    for i in range(P): ax.annotate(str(i), (px[i], py[i]), fontsize=7)
    ax.set_aspect("equal"); ax.set_title(f"embeddings, wave {k}: numbers on a circle", fontsize=10)
ax = fig.add_subplot(2, 3, 6)                                       # the most single-frequency MLP neuron
H2 = torch.fft.fft2(hid - hid.mean((0, 1)), dim=(0, 1)).abs() ** 2; tot = H2.sum((0, 1)); share = torch.zeros(hid.shape[-1])
for k in range(1, P // 2 + 1): share = torch.maximum(share, sum(H2[i, j] for i, j in [(k, 0), (P - k, 0), (0, k), (0, P - k), (k, k), (P - k, P - k), (k, P - k), (P - k, k)]) / tot)
ax.imshow(hid[:, :, share.argmax()], cmap="RdBu"); ax.set_xlabel("b"); ax.set_ylabel("a"); ax.set_title(f"one MLP neuron over all (a, b): {share.max():.0%} in one wave", fontsize=10)
plt.tight_layout(); plt.savefig("waves.png", dpi=110); print("top", top)
