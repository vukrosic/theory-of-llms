"""Facts answered right by Muon expert layout (numbers from findings/muon-moe-expert-layout.md, means over seeds). -> ../../figures/muon-moe-expert-layout.png"""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150, gridspec_kw=dict(width_ratios=[1, 1.3]))
C = dict(each="#00798c", fused="#d1495b", dense="#888888")
labs = ["each expert\nits own matrix", "fused\n[E·rows, cols]", "stacked", "flattened\ng.view(E, -1)", "dense MLP\n(no experts)"]
vals = [82, 66, 64, 61, 63]; cols = [C["each"], C["fused"], C["fused"], C["fused"], C["dense"]]
a.bar(range(5), vals, color=cols, width=.65)
for i, v in enumerate(vals): a.text(i, v + 1, f"{v}%", ha="center", fontsize=10)
a.set_xticks(range(5), labs, fontsize=8.5); a.set_ylim(0, 100); a.set_ylabel("facts answered right")
a.set_title("How Muon sees the expert weights", fontsize=11)
groups = ["8 experts", "16 experts", "32 narrow experts\n(hidden 11 < width 32)", "8 experts,\n4x tokens per expert"]
each, fused, dense = [82, 89, 83, 89], [66, 69, 65, 78], [63, 63, 63, 73]
w = .26
for j, (v, n, c) in enumerate([(each, "each expert alone", C["each"]), (fused, "fused", C["fused"]), (dense, "dense, same active size", C["dense"])]):
    b.bar([i + (j - 1) * w for i in range(4)], v, w, color=c, label=n)
b.set_xticks(range(4), groups, fontsize=9); b.set_ylim(0, 108); b.set_ylabel("facts answered right")
b.set_title("Fused stays near dense in every setting", fontsize=11); b.legend(frameon=False, fontsize=9, loc="upper left", ncol=3)
for ax in (a, b): ax.spines[["top", "right"]].set_visible(False)
fig.text(.01, .01, "2-layer MoE, width 32, top-2 (top-8 for 32 experts), 2,048 facts, Muon, each layout at its best expert learning rate, mean of 2-4 seeds.", fontsize=8.5, color="#444")
fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig("../../figures/muon-moe-expert-layout.png"); print("ok")
