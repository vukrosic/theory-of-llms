"""Figures for the two pages. usage: P=47 python3 fig.py RUNS_DIR   (RUNS_DIR holds muon_{0,1,2}_0.03_1.0.pt)
-> ../../figures/sae-answers-not-clocks.png, ../../figures/transcoder-drops-the-sum.png"""
import json, os, sys, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import mod, sae
P = mod.P; R = sys.argv[1]; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures")
BLUE, ORANGE, INK, INK2, GRID, BG = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": BG, "axes.facecolor": BG, "text.color": INK})


def load(ms):
    m = mod.Model(); m.load_state_dict(torch.load(f"{R}/muon_{ms}_0.03_1.0.pt")); return m, *sae.acts(m)


def per_answer(S, X, y):
    Z = sae.encode(X, S["We"], S["be"], S["bd"], S["k"]); g = torch.zeros(P, Z.shape[1]).index_add_(0, y, Z) / P
    one = ((g > 0.3 * g.max(0).values.clamp_min(1e-12)).sum(0) == 1) & (Z > 0).any(0); return Z, g, one


m, A, x, y = load(0)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw=dict(width_ratios=[1.3, 0.9, 1.3]))
S = torch.load("sae_runs/muon_0_0.03_1.0_res_64_4_0.pt"); Z, g, one = per_answer(S, A["res"], y)
ax[0].bar(range(P), g[:, 2], color=BLUE, width=0.8)
ax[0].set_xlabel("the right answer (a + b) mod 47"); ax[0].set_ylabel("latent 2's mean activation")
ax[0].set_title("A typical SAE latent fires on one answer only\n(here: answer 29; removing it breaks 29 on 100% of its pairs)", fontsize=10)
sizes = [16, 32, 47, 64, 128]; ones, clocks = [], []
for n in sizes:
    S_ = torch.load(f"sae_runs/muon_0_0.03_1.0_res_{n}_4_0.pt"); Z_, g_, one_ = per_answer(S_, A["res"], y); L = sae.label(Z_)
    ones.append(one_.sum().item()); clocks.append(((L["share2d"] > 0.5) & (Z_ > 0).any(0)).sum().item())
ax[1].bar(range(len(sizes)), ones, color=BLUE, width=0.6, label="latents on one answer")
for i, (o, c) in enumerate(zip(ones, clocks)): ax[1].text(i, o + 1, f"{o}", ha="center", fontsize=9, color=INK2)
ax[1].set_xticks(range(len(sizes)), sizes); ax[1].set_xlabel("SAE size (latents)"); ax[1].set_ylabel("latents")
ax[1].set_title("Latents that fire on one answer only, by SAE size\n(latents that are a clock: " + ", ".join(map(str, clocks)) + ")", fontsize=10)
D = torch.zeros(P, S["Wd"].shape[0]); cnt = torch.zeros(P); ans = g.argmax(0)
for j in one.nonzero()[:, 0].tolist(): D[ans[j]] += S["Wd"][:, j]; cnt[ans[j]] += 1
sv = (cnt > 0).nonzero()[:, 0]; D = D[sv] / cnt[sv, None]; D = D - D.mean(0); ks = torch.arange(1, P // 2 + 1)
pw = torch.stack([(torch.stack([torch.cos(2 * torch.pi * k * sv / P), torch.sin(2 * torch.pi * k * sv / P)], 1).pinverse() @ D).pow(2).sum() for k in ks])
Ms = torch.zeros(P, A["res"].shape[1]).index_add_(0, y, A["res"]) / P; pm = torch.fft.rfft(Ms - Ms.mean(0), dim=0).abs().pow(2).sum(1)[1:]
w = 0.4
ax[2].bar(ks - w / 2, pm / pm.sum(), w, color=ORANGE, label="the model's own residual stream")
ax[2].bar(ks + w / 2, pw / pw.sum(), w, color=BLUE, label="SAE answer latents, put in answer order")
ax[2].set_xticks(ks[::2]); ax[2].set_xlabel("clock speed"); ax[2].set_ylabel("share of power")
ax[2].set_title("The clocks come back from how the latents are laid out\n(same 5 speeds: 13, 3, 20, 22, 11)", fontsize=10); ax[2].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "sae-answers-not-clocks.png"), dpi=120); plt.close()

rows = [json.loads(l) for l in open("tc_runs.jsonl")]; rows = [r for r in rows if r["seed"] == 0 and r["n"] == 64]
last = {}
for r in rows: last[(r["model"], r["k"])] = r
kk = sorted({k for _, k in last}); models = sorted({mm for mm, _ in last})
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for idx, lab, col in ((0, "each number alone (a, b)", BLUE), (1, "the sum a + b", ORANGE)):
    v = torch.tensor([[last[(mm, k)]["kept"][idx] for k in kk] for mm in models])
    for row in v: ax[0].plot(range(len(kk)), row, color=col, lw=1, alpha=0.35)
    ax[0].plot(range(len(kk)), v.mean(0), color=col, lw=2, marker="o", ms=6, label=lab)
ax[0].set_xticks(range(len(kk)), kk); ax[0].set_xlabel("active latents per input (k of 64)"); ax[0].set_ylabel("share of that part rebuilt")
ax[0].set_ylim(0.3, 1.02); ax[0].legend(frameon=False, fontsize=9, loc="lower right"); ax[0].grid(axis="y", color=GRID, lw=0.6)
ax[0].set_title("A transcoder keeps the inputs and loses the sum", fontsize=10)
v = torch.tensor([[last[(mm, k)]["model_acc"] - last[(mm, k)]["acc"] for k in kk] for mm in models])
for row in v: ax[1].plot(range(len(kk)), row, color=INK2, lw=1, alpha=0.35)
ax[1].plot(range(len(kk)), v.mean(0), color=INK, lw=2, marker="o", ms=6)
ax[1].set_xticks(range(len(kk)), kk); ax[1].set_xlabel("active latents per input (k of 64)"); ax[1].set_ylabel("accuracy lost (transcoder in place of MLP)")
ax[1].grid(axis="y", color=GRID, lw=0.6); ax[1].set_title("Accuracy comes back only when the sum does (thin: 3 models)", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "transcoder-drops-the-sum.png"), dpi=120); plt.close()
print("ones", ones, "clocks", clocks, "decoder top5", (pw.argsort(descending=True)[:5] + 1).tolist())
