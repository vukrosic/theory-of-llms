"""results.json (printed by facts.py trace / edit, 3 seeds) -> ../../figures/mlp-edit-without-storage.png, ../../figures/edit-the-last-token.png"""
import json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = json.load(open("results.json")); S = ["0", "1", "2"]; N = len(R["spots"])
lab = ["layer 0\nfirst subject\ntoken", "layer 0\nlast subject token\n(ROME's spot)", "layer 0\nrelation token", "layer 1\nrelation token\n(last token)"]
BLUE, ORANGE, AQUA, INK = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"


def bars(ax, series, w=0.26):
    for j, (name, key, f, c) in enumerate(series):
        for i in range(N):
            v = [f(R[key][s][i]) for s in S if R[key][s][i] is not None]
            x = i + (j - (len(series) - 1) / 2) * w
            ax.bar(x, sum(v) / len(v), w * 0.9, color=c, label=name if i == 0 else None)
            ax.plot([x] * len(v), v, "o", ms=3.5, color="#0b0b0b", alpha=.55)
    ax.set_xticks(range(N), lab, fontsize=8.5); ax.spines[["top", "right"]].set_visible(False)


fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
bars(ax, [("causal tracing effect of the MLP output", "trace_mlp", lambda v: v, BLUE),
          ("share of facts lost when the MLP output is knocked out", "knockout_mlp_kept", lambda v: 1 - v, ORANGE),
          ("share of ROME edits that work", "rome_works", lambda v: v, AQUA)])
ax.set_ylim(0, 1.32); ax.set_yticks([0, .2, .4, .6, .8, 1]); ax.set_ylabel("share"); ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
ax.set_title("The subject-token MLPs hold nothing (tracing 0, knockout costs 0), yet editing them works 90% of the time", fontsize=10.5)
fig.text(.01, .01, "2-layer MoE trained with Muon, 128 two-token subjects x 2 relations. Bars: mean of 3 seeds, dots: each seed. 32 edits per spot per seed.", fontsize=8, color=INK)
fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig("../../figures/mlp-edit-without-storage.png")

fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
ser = lambda k: [("ROME (key whitened by C$^{-1}$)", f"rome_{k}", lambda v: v, BLUE), ("raw key, the shape of one Muon step", f"raw_{k}", lambda v: v, ORANGE)]
bars(a, ser("changed"), 0.36); a.set_ylabel("other facts whose answer changed (of 255)"); a.legend(frameon=False, fontsize=9)
a.set_title("Edits at the last token break 6x fewer other facts", fontsize=11)
bars(b, ser("sibling"), 0.36); b.set_ylim(0, 1); b.set_ylabel("share of edits that also change the subject's other fact")
b.set_title("Edits at the subject leak into its other relation", fontsize=11)
fig.text(.01, .01, "2-layer MoE trained with Muon, 128 two-token subjects x 2 relations, 32 edits per spot per seed. Bars: mean of 3 seeds, dots: each seed (raw key: seed 0 only at the first and third spot).", fontsize=8, color=INK)
fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig("../../figures/edit-the-last-token.png"); print("ok")
