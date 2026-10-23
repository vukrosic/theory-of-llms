"""Loss vs QK-Clip threshold at 8x and 1x the best lr, and loss for clip schedules at 8x (3-seed means from NOTES.md batches 2, 4, 5 and ../qkclip-long CLIPSTOP). -> ../../figures/muon-qkclip-rules.png"""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
t8 = [(1, 4.095), (3, 4.115), (5, 4.116), (10, 4.128), (30, 4.186), (100, 4.261)]; t1 = [(1, 4.044), (3, 4.024), (10, 4.022)]
a.plot(*zip(*t8), "-o", color="#d1495b", lw=2, ms=5, label="8x the best learning rate")
a.plot(*zip(*t1), "-o", color="#00798c", lw=2, ms=5, label="best learning rate")
a.axhline(4.358, color="#d1495b", ls=":", lw=1.5); a.text(1.05, 4.345, "8x, no clip", color="#d1495b", fontsize=9, va="top")
a.axhline(4.122, color="#888", ls="--", lw=1.5); a.text(140, 4.13, "8x, QK-norm", color="#666", fontsize=9, ha="right")
a.axhline(4.022, color="#00798c", ls=":", lw=1.5); a.text(140, 4.01, "best, no clip", color="#00798c", fontsize=9, ha="right", va="top")
a.set_xscale("log"); a.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"]); a.minorticks_off(); a.set_ylim(3.98, 4.4)
a.set_xlabel("QK-Clip threshold (healthy largest score here is about 10)"); a.set_ylabel("held-out loss (lower is better)")
a.set_title("A threshold near the healthy scores fixes it; 100 does not", fontsize=11); a.legend(frameon=False, fontsize=9, loc="center left")
s = [("every step", 4.128), ("every 10 steps", 4.144), ("every 50 steps", 4.196), ("start at step 100", 4.158), ("start at step 300", 4.216), ("stop at step 400", 4.182), ("stop at step 200", 4.264), ("no clip", 4.358)]
cols = ["#00798c"] * 2 + ["#bc8034"] + ["#00798c"] + ["#bc8034"] * 3 + ["#d1495b"]
b.barh(range(len(s))[::-1], [v - 4.0 for _, v in s], left=4.0, color=cols, height=.6)
b.set_yticks(range(len(s))[::-1], [n for n, _ in s]); b.axvline(4.122, color="#888", ls="--", lw=1.5); b.text(4.124, 7.35, "QK-norm", color="#666", fontsize=9)
for i, (_, v) in enumerate(s): b.text(v + .004, len(s) - 1 - i, f"{v:.3f}", va="center", fontsize=9)
b.set_xlim(4.0, 4.4); b.set_xlabel("held-out loss at 8x the best learning rate, threshold 10")
b.set_title("Check every 10 steps, start early, keep it on", fontsize=11)
for ax in (a, b): ax.spines[["top", "right"]].set_visible(False)
fig.text(.01, .01, "2-layer Muon language model without QK-norm, width 16, 4 heads, 800 steps, best lr 0.014, mean of 3 seeds.", fontsize=8.5, color="#444")
fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig("../../figures/muon-qkclip-rules.png"); print("ok")
