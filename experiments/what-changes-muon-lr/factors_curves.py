"""Loss vs learning rate for the three settings that move Muon's best lr (factors.jsonl, mean over seeds). -> ../../figures/lr-valleys.png"""
import json, collections, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
BASE = dict(B=8, D=16, NL=2, T=32, MOM=0.95, WD=0.0, WARM=20, DFRAC=0.5, MIX="mix20")
rows = [json.loads(l) for l in open("factors.jsonl")]
def curve(**kw):
    want = {**BASE, **kw}; d = collections.defaultdict(list)
    for r in rows:
        if all(r[k] == v for k, v in want.items()): d[r["lr"]].append(r["loss"])
    lr = np.array(sorted(d)); m = np.array([np.mean(d[x]) for x in lr]); return lr, m
# best lr per setting from the parabola fit over 5 seeds (NOTES.md, factors.py fit); grid minimum alone is noisy
BEST = {("B", 4): .0104, ("B", 8): .0150, ("B", 16): .0209, ("B", 32): .0308,
        ("MOM", .9): .0252, ("MOM", .95): .0150, ("MOM", .98): .0097,
        ("DFRAC", 0.0): .0074, ("DFRAC", .25): .0127, ("DFRAC", .5): .0150, ("DFRAC", 1.0): .0219}
P = [("Batch size", "B", [4, 8, 16, 32], lambda v: f"batch {v}"),
     ("Momentum", "MOM", [0.9, 0.95, 0.98], lambda v: f"momentum {v}"),
     ("Learning rate decay", "DFRAC", [0.0, 0.25, 0.5, 1.0], lambda v: {0.0: "no decay", 0.25: "last quarter", 0.5: "last half", 1.0: "whole run"}[v])]
fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), dpi=150)
for ax, (title, k, vals, lab) in zip(axs, P):
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(vals)))
    for v, c in zip(vals, cols):
        lr, m = curve(**{k: v}); y = m - m.min(); b = BEST[(k, v)]
        ax.plot(lr, y, "-o", color=c, ms=3, lw=1.8, label=lab(v))
        ax.axvline(b, color=c, ls=":", lw=1.2); ax.plot(b, np.interp(np.log(b), np.log(lr), y), "*", color=c, ms=16, mec="k", mew=.6, zorder=5)
    ax.set_xscale("log"); ax.set_ylim(-0.005, 0.14); ax.set_title(title, fontsize=13)
    ax.set_xlabel("learning rate"); ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("loss above the best loss")
fig.suptitle("The best learning rate (star) moves when these three settings change", fontsize=14)
fig.text(.01, .01, "Tiny Muon language model, short training runs. Lower is better.", fontsize=9, color="#444")
fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig("../../figures/lr-valleys.png"); print("ok")
