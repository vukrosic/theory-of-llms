"""Scaling check: both findings on one model. Residual SAE (64 latents, k=4) and transcoder (64 latents, k=4).
usage: P=47 D=.. FF=.. python3 scale.py MODEL.pt SEED   -> appends scale_runs.jsonl (activations scaled to unit mean norm first)
  one = latents on one answer only, clocks = latents with >50% of their (a, b) map on one clock,
  geo = how many of the model's 5 strongest clock speeds are in the decoder layout's top 5,
  own / other = break rate of the latent's own answer / of other answers when it is removed,
  tc_sum / tc_alone = share of the MLP output's sum part / each-number-alone part the transcoder rebuilds, tc_lost = accuracy lost"""
import json, os, sys, time, torch
import mod, sae
P = mod.P; path, seed = sys.argv[1], int(sys.argv[2]); t0 = time.time()
m = mod.Model(); m.load_state_dict(torch.load(path)); A, x, y = sae.acts(m)
u = lambda V: V / V.norm(dim=1).mean()            # unit mean norm (long weight decay shrinks activations ~50x; the head and next norm ignore scale)
X = u(A["res"]); A["r1"], A["out"] = A["r1"] / A["out"].norm(dim=1).mean(), u(A["out"]); A["z"] = u(A["z"])
We, be, Wd, bd = sae.train(X, 64, 4, seed, 3000); Z = sae.encode(X, We, be, bd, 4)
g = torch.zeros(P, 64).index_add_(0, y, Z) / P; alive = (Z > 0).any(0)
sup = (g > 0.3 * g.max(0).values.clamp_min(1e-12)).sum(0) * alive; one = sup == 1; ans = g.argmax(0)
clocks = ((sae.label(Z)["share2d"] > 0.5) & alive).sum().item()
Dm = torch.zeros(P, X.shape[1]); cnt = torch.zeros(P)
for j in one.nonzero()[:, 0].tolist(): Dm[ans[j]] += Wd[:, j]; cnt[ans[j]] += 1
sv = (cnt > 0).nonzero()[:, 0]; Dm = Dm[sv] / cnt[sv, None]; Dm = Dm - Dm.mean(0); ks = torch.arange(1, P // 2 + 1)
pw = torch.stack([(torch.stack([torch.cos(2 * torch.pi * k * sv / P), torch.sin(2 * torch.pi * k * sv / P)], 1).pinverse() @ Dm).pow(2).sum() for k in ks])
Ms = torch.zeros(P, X.shape[1]).index_add_(0, y, X) / P; pm = torch.fft.rfft(Ms - Ms.mean(0), dim=0).abs().pow(2).sum(1)[1:]
geo = len(set(pw.argsort(descending=True)[:5].tolist()) & set(pm.argsort(descending=True)[:5].tolist()))
with torch.no_grad():
    base = sae.head(m, X).argmax(-1) == y; own, oth = [], []
    for j in one.nonzero()[:, 0].tolist():
        ok = sae.head(m, X - Z[:, j:j + 1] * Wd[:, j]).argmax(-1) == y; on = (g[:, j] > 0.3 * g[:, j].max())[y]
        own.append((base & on & ~ok).sum().item() / max(1, (base & on).sum().item())); oth.append((base & ~on & ~ok).sum().item() / max(1, (base & ~on).sum().item()))
TN, TK = int(os.environ.get("TC_N", 64)), int(os.environ.get("TC_K", 4))   # transcoder latents / active latents
SK = os.environ.get("SKIP") == "1"                 # skip transcoder: + a linear path from MLP input to output
T = sae.train(A["z"], TN, TK, seed, 2000, Y=A["out"], skip=SK); We2, be2, Wd2, bd2, bi = T[:5]
R = sae.encode(A["z"], We2, be2, bi, TK) @ Wd2.T + bd2 + ((A["z"] - bi) @ T[5].T if SK else 0)
sv2 = torch.linalg.svdvals(sae.split(A["out"])[1].reshape(P * P, -1)) ** 2; srank = ((sv2.cumsum(0) / sv2.sum()) < 0.9).sum().item() + 1
acc = lambda r: (sae.head(m, A["r1"] + r).argmax(-1) == y).float().mean().item(); kp = sae.kept(A["out"], R)
row = dict(model=os.path.basename(path), D=mod.D, seed=seed, tc_n=TN, tc_k=TK, skip=SK, sum_rank90=srank, model_acc=round(acc(A["out"]), 4), one=one.sum().item(), alive=alive.sum().item(), clocks=clocks, geo=geo,
           own=round(sum(own) / max(1, len(own)), 3), other=round(sum(oth) / max(1, len(oth)), 4), tc_alone=kp[0], tc_sum=kp[1], tc_lost=round(acc(A["out"]) - acc(R), 3))
print(json.dumps(row), f"{time.time() - t0:.1f}s")
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scale_runs.jsonl"), "a") as fh: fh.write(json.dumps(row) + "\n")
