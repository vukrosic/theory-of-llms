"""Scaling figure for the transcoder page. usage: python3 fig_scale.py  (reads scale_runs.jsonl) -> ../../figures/transcoder-sum-gap-by-width.png"""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures")
BLUE, ORANGE, INK2, GRID, BG = "#2a78d6", "#eb6834", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": BG, "axes.facecolor": BG})
last = {}
for l in open("scale_runs.jsonl"):                       # old rows lack skip / tc_n / tc_k: plain, 64, 4; keep the last row per setting
    r = {"skip": False, "tc_n": 64, "tc_k": 4, **json.loads(l)}; last[(r["model"], r["skip"], r["tc_n"], r["tc_k"])] = r
rows = list(last.values())
W = [32, 64, 128]; width_model = lambda r: "_s" not in r["model"]
pick = lambda sk, D: [r for r in rows if r["skip"] == sk and r["tc_k"] == 4 and r["D"] == D and r["tc_n"] == D and width_model(r)]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i, (key, lab) in enumerate((("tc_sum", "share of the sum part rebuilt"), ("tc_lost", "accuracy lost (transcoder in place of MLP)"))):
    for sk, col, name, dx in ((False, ORANGE, "plain transcoder", -0.17), (True, BLUE, "with a skip connection", 0.17)):
        for j, D in enumerate(W):
            v = [r[key] for r in pick(sk, D)]
            ax[i].bar(j + dx, sum(v) / len(v), 0.32, color=col, label=name if j == 0 else None)
            ax[i].scatter([j + dx] * len(v), v, s=12, color=INK2, zorder=3)
    ax[i].set_xticks(range(3), [f"width {D}" for D in W]); ax[i].set_ylabel(lab); ax[i].grid(axis="y", color=GRID, lw=0.6)
ax[0].set_ylim(0, 1); ax[0].legend(frameon=False, fontsize=9, loc="lower left")
ax[0].set_title("Wider model: the plain transcoder loses more of the sum\n(latents = width, 4 active; dots: 3 models)", fontsize=10)
ax[1].set_title("A skip connection gives the accuracy back", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "transcoder-sum-gap-by-width.png"), dpi=120)
