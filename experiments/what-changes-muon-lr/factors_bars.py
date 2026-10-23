"""Simple chart: how many times Muon's best lr changes when each setting changes (NOTES.md tables). -> ../../figures/lr-what-changes.png
'yes' = the 95% ranges of the two end values do not overlap (5 seeds for the yes rows, 3-5 for the rest)."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RED, ORA, GRY = "#d1495b", "#f0a04b", "#bdbdbd"
R = [("batch size 4 → 32", 2.96, RED, "learning rate 0.010 → 0.031"),
     ("learning rate decay: none → whole run", 2.96, RED, "learning rate 0.007 → 0.022"),
     ("momentum 0.9 → 0.98", 2.60, RED, "learning rate 0.025 → 0.010"),
     ("training length 100 → 1600 steps", 1.48, GRY, ""),
     ("depth 1 → 4 layers", 1.38, GRY, ""),
     ("sequence length 16 → 64", 1.22, GRY, ""),
     ("warmup 0 → 100 steps", 1.18, GRY, ""),
     ("width 16 → 64", 1.15, GRY, ""),
     ("weight decay 0 → 0.1", 1.05, GRY, ""),
     ("data: real text vs toy task mix", 1.04, GRY, "")]
fig, ax = plt.subplots(figsize=(10, 6.2), dpi=150)
for i, (n, r, c, note) in enumerate(R[::-1]):
    ax.barh(i, r - 1, left=1, color=c); ax.text(r + .03, i, f"{r:.2f}x   {note}", va="center", fontsize=9.5)
ax.set_yticks(range(len(R))); ax.set_yticklabels([n for n, _, _, _ in R[::-1]], fontsize=10.5)
ax.axvline(1, color="k", lw=.8); ax.set_xlim(1, 4.6); ax.set_xticks([1, 1.5, 2, 2.5, 3])
ax.set_xlabel("how many times the best learning rate changed")
ax.set_title("What changes Muon's best learning rate", fontsize=13)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.text(.01, .01,
    "Red = changes the best learning rate (same result with every random seed). Grey = barely changes it.\n"
    "Tiny Muon language model, short training runs.",
    fontsize=9, color="#444", va="bottom")
fig.tight_layout(rect=(0, .06, 1, 1)); fig.savefig("../../figures/lr-what-changes.png")
