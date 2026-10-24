"""figure for findings/fine-tuning-stops-prompt-reading.md: copy accuracy after fine-tuning on 8 fixed tasks, per size, with and without 5% made-up-rule sequences"""
import json, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = [json.loads(l) for l in open("runs.jsonl")]
ft = [r for r in R if r.get("PRE") and r["T"] == 8 and r["steps"] == 1200 and r["lr"] == 0.02 and r.get("MIXB", 6) == 6
      and (r["PRE"] == 1500 if r["D"] == 32 else r["PRE"] == 600)]
agg = collections.defaultdict(list)
for r in ft:
    if r["MIXF"] in (0.0, 0.05): agg[(r["D"], r["MIXF"])].append((r["copy"], r["recog"]))
par = lambda d: 24.5 * d * d + 2 * 64 * d
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); plt.rcParams["font.size"] = 10
for f, c, lab in ((0.0, ORANGE, "fine-tune on the 8 tasks only"), (0.05, BLUE, "+ 5% sequences with a made-up rule")):
    ds = sorted(d for d, ff in agg if ff == f)
    y = [sum(a for a, b in agg[(d, f)]) / len(agg[(d, f)]) for d in ds]
    ax[0].plot([par(d) for d in ds], y, "-o", color=c, lw=2, ms=8, label=lab, mec="white", mew=1.5)
    for d in ds:
        for a, b in agg[(d, f)]: ax[0].plot(par(d), a, ".", color=c, alpha=0.4, ms=5)
ax[0].set_xscale("log"); ax[0].set_xticks([par(d) for d in (32, 64, 96, 128)], [f"{par(d)/1e3:.0f}k\nwidth {d}" for d in (32, 64, 96, 128)]); ax[0].minorticks_off(); ax[0].set_ylim(0, 1.05); ax[0].set_xlabel("model parameters"); ax[0].set_ylabel("new rule shown in the prompt: answered right")
ax[0].set_title("After fine-tuning: does the model still read the prompt?", loc="left", fontsize=11)
ax[0].legend(frameon=False, loc="center left"); ax[0].axhline(1 / 32, color=MUTED, lw=0.8, ls=":"); ax[0].text(par(96), 0.05, "chance", color=MUTED, fontsize=8)
cur = {r["MIXF"]: r for r in R if r.get("PRE") == 600 and r["D"] == 96 and r["steps"] == 1200 and r["T"] == 8 and r["seed"] == 0 and r["curve"]}
for f, c in ((0.0, ORANGE), (0.05, BLUE)):
    cv = cur[f]["curve"] + [(1200, cur[f]["recog"], cur[f]["copy"])]
    ax[1].plot([i for i, a, b in cv], [b for i, a, b in cv], "-o", color=c, lw=2, ms=7, mec="white", mew=1.5)
    ax[1].plot([i for i, a, b in cv], [a for i, a, b in cv], "--", color=c, lw=1.2, alpha=0.8)
ax[1].text(640, 0.17, "new rule in prompt, tasks only", color=INK, fontsize=9); ax[1].text(640, 0.6, "new rule in prompt, + 5%", color=INK, fontsize=9)
ax[1].text(330, 0.35, "dashed: the 8 fine-tuning tasks\n(both runs reach 1.0)", color=MUTED, fontsize=8)
ax[1].set_ylim(0, 1.1); ax[1].set_xlabel("fine-tuning steps (width 96)"); ax[1].set_title("The prompt-reading skill is gone within 600 steps", loc="left", fontsize=11)
for a in ax:
    a.spines[["top", "right"]].set_visible(False); a.grid(alpha=0.2)
plt.tight_layout(); plt.savefig("../../figures/fine-tuning-prompt-reading.png", dpi=130, bbox_inches="tight")
for k in sorted(agg): print(k, len(agg[k]), [round(x, 3) for x in map(lambda v: sum(v) / len(v), zip(*agg[k]))])
