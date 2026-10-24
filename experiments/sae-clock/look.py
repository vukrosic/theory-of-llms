"""Print where each latent of a saved SAE fires: its mean activation per answer s = (a+b) mod P.
usage: P=47 python3 look.py MODEL.pt SAE.pt SITE"""
import sys, torch
import mod, sae
P = mod.P
m = mod.Model(); m.load_state_dict(torch.load(sys.argv[1])); S = torch.load(sys.argv[2]); A, x, y = sae.acts(m)
Z = sae.encode(A[sys.argv[3]], S["We"], S["be"], S["bd"], S["k"])
s = (x[:, 0] + x[:, 1]) % P
g = torch.zeros(P, Z.shape[1]).index_add_(0, s, Z) / P               # mean activation per answer
frac = torch.zeros(P, Z.shape[1]).index_add_(0, s, (Z > 0).float()) / P  # share of that answer's pairs where it fires
sup = (g > 0.3 * g.max(0).values).sum(0)
print("answers each latent fires on (>30% of its max):", torch.bincount(sup).tolist())
print("share of an answer's 47 pairs where its top latent fires:", f"{frac.max(1).values.mean():.2f}")
for j in range(min(12, Z.shape[1])):
    top = g[:, j].argsort(descending=True)[:4]
    print(f"latent {j:2}: answers {top.tolist()} mean act {[round(v, 2) for v in g[top, j].tolist()]}, fires on {frac[top[0], j]:.2f} of answer {top[0].item()}'s pairs")
# which clock bunches each latent's answers together? weighted resultant length |mean exp(2 pi i k s / P)| for k = 1..P//2
ks = torch.arange(1, P // 2 + 1).float(); ang = 2 * torch.pi * ks[:, None] * torch.arange(P)[None] / P     # k x s
w = g / g.sum(0).clamp_min(1e-12)                                                                           # s x latent
Rk = ((torch.cos(ang) @ w) ** 2 + (torch.sin(ang) @ w) ** 2).sqrt()                                         # k x latent
best = Rk.argmax(0) + 1; Rb = Rk.max(0).values
multi = sup > 1
print("latents on 2+ answers:", multi.sum().item(), " their best clock (count):",
      sorted(torch.bincount(best[multi], minlength=P // 2 + 1).nonzero()[:, 0].tolist(), key=lambda k: -torch.bincount(best[multi], minlength=P // 2 + 1)[k].item()),
      torch.bincount(best[multi], minlength=P // 2 + 1)[torch.bincount(best[multi], minlength=P // 2 + 1) > 0].tolist(),
      f" bunching {Rb[multi].mean():.2f} (random sets of the same size: {torch.stack([((torch.cos(ang[:, torch.randperm(P)[:max(2,int(c))]]).mean(1))**2 + (torch.sin(ang[:, torch.randperm(P)[:max(2,int(c))]]).mean(1))**2).sqrt().max() for c in sup[multi]]).mean():.2f})")
