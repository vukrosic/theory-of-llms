"""Are the clocks hidden in how the SAE's answer latents are arranged? Takes the latents that fire on one answer only,
puts their decoder vectors in answer order (mean over duplicates), and Fourier-transforms along the answer axis.
usage: P=47 python3 geometry.py MODEL.pt SAE.pt SITE"""
import sys, torch
import mod, sae
P = mod.P
m = mod.Model(); m.load_state_dict(torch.load(sys.argv[1])); S = torch.load(sys.argv[2]); A, x, y = sae.acts(m)
Z = sae.encode(A[sys.argv[3]], S["We"], S["be"], S["bd"], S["k"]); s = (x[:, 0] + x[:, 1]) % P
g = torch.zeros(P, Z.shape[1]).index_add_(0, s, Z) / P
one = ((g > 0.3 * g.max(0).values.clamp_min(1e-12)).sum(0) == 1) & (Z > 0).any(0); ans = g.argmax(0)
D = torch.zeros(P, S["Wd"].shape[0]); cnt = torch.zeros(P)
for j in one.nonzero()[:, 0].tolist(): D[ans[j]] += S["Wd"][:, j]; cnt[ans[j]] += 1
have = cnt > 0; print(f"{one.sum().item()} one-answer latents covering {have.sum().item()} of {P} answers")
D = D[have] / cnt[have, None]; D = D - D.mean(0); sv = have.nonzero()[:, 0]
ks = torch.arange(1, P // 2 + 1)                              # least-squares power per clock speed (answers may be missing)
pw = torch.stack([(torch.stack([torch.cos(2 * torch.pi * k * sv / P), torch.sin(2 * torch.pi * k * sv / P)], 1).pinverse() @ D).pow(2).sum() for k in ks])
o = pw.argsort(descending=True)
print("clock speeds in the decoder vectors, strongest first:", (ks[o[:6]]).tolist(), " share of top 5:", f"{pw[o[:5]].sum() / pw.sum():.2f}")
M = A["res"]; Ms = torch.zeros(P, M.shape[1]).index_add_(0, s, M) / P; Ms = Ms - Ms.mean(0)   # the model's own mean residual per answer
pm = torch.fft.rfft(Ms, dim=0).abs().pow(2).sum(1)[1:]; om = pm.argsort(descending=True)
print("clock speeds in the model's own residual per answer:", (om[:6] + 1).tolist(), f" share of top 5: {pm[om[:5]].sum() / pm.sum():.2f}")
