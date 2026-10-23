"""Held-out loss vs weight decay for each number of epochs (runs_all.jsonl: decay on all matrices) and best loss per epochs, all vs Muon-matrices-only (runs.jsonl). lr 0.014. -> ../../figures/muon-wd-epochs.png"""
import json, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def load(f):
    d = collections.defaultdict(list)
    for l in open(f):
        r = json.loads(l)
        if r["lr"] == 0.014 and r["loss"] == r["loss"]: d[(r["E"], r["wd"])].append(r["loss"])
    return {k: sum(v) / len(v) for k, v in d.items()}
A, M = load("runs_all.jsonl"), load("runs.jsonl")
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
cols = {1: "#00798c", 4: "#30638e", 8: "#6a4c93", 16: "#bc8034", 32: "#d1495b", 64: "#8c1c13"}
for E, c in cols.items():
    pts = sorted((wd, l) for (e, wd), l in A.items() if e == E and wd > 0); x, y = zip(*pts)
    a.plot(x, y, "-o", color=c, ms=4, lw=2, label=f"{E} epoch" + ("s" if E > 1 else "")); i = y.index(min(y)); a.plot(x[i], y[i], "*", color=c, ms=13)
a.set_xscale("log"); a.set_ylim(3.95, 5.2); a.set_xlabel("weight decay"); a.set_ylabel("held-out loss (lower is better)")
a.set_title("The more you repeat the data, the higher the best weight decay", fontsize=11); a.legend(frameon=False, fontsize=9, ncol=2)
Es = sorted(cols)
for D_, lab, c in [(A, "decay on all weights", "#00798c"), (M, "decay on Muon matrices only (embedding, head not decayed)", "#d1495b")]:
    b.plot(Es, [min(l for (e, wd), l in D_.items() if e == E) for E in Es], "-o", color=c, ms=5, lw=2, label=lab)
b.plot(Es, [D_ for D_ in [A[(E, 0.1)] for E in Es]], "--", color="#888", lw=1.5, label="weight decay 0.1 on all weights")
b.set_xscale("log", base=2); b.set_xticks(Es, [str(e) for e in Es]); b.minorticks_off(); b.set_ylim(3.95, 8.3)
b.set_xlabel("epochs (same total steps, smaller dataset)"); b.set_ylabel("held-out loss at the best weight decay")
b.set_title("Embedding and head must be decayed too", fontsize=11); b.legend(frameon=False, fontsize=9)
for ax in (a, b): ax.spines[["top", "right"]].set_visible(False)
fig.text(.01, .01, "2-layer Muon language model, width 16, 800 steps, lr 0.014, mean of 2 seeds (1 seed for 1 epoch). Weight decay: p *= 1 - lr*wd each step.", fontsize=8.5, color="#444")
fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig("../../figures/muon-wd-epochs.png"); print("ok")
