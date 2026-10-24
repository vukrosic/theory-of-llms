"""figure for findings/ordered-values-bend.md from runs.jsonl"""
import json, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = [json.loads(l) for l in open("runs.jsonl") if '"drift"' in l and json.loads(l).get("mix") == 0]
R = list({(r["width"], r["steps"], r["seed"]): r for r in R}.values())
avg = lambda v: sum(v) / len(v); md = lambda d: avg(list(d.values()))
g = collections.defaultdict(list)
for r in R: g[(r["width"], r["steps"])].append(r)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
col = {32: "#9ecae1", 64: "#4292c6", 128: "#08519c"}
for w in (32, 64, 128):
    st = sorted(s for (ww, s) in g if ww == w and s >= 1040)
    for site, ls, lab in (("1", "-", "middle layer"), ("2", "--", "last layer")):
        ax[0].plot(st, [avg([md(r["concepts"]["count"]["sites"][site]["steer"]["line"]["best"]) for r in g[(w, s)]]) for s in st], ls, color=col[w], marker="o", label=f"width {w}, {lab}")
ax[0].axhline(1.0, color="k", lw=1, ls=":"); ax[0].text(1100, 0.92, "adding mean(target count) - mean(source count): 99-100% everywhere", fontsize=8)
ax[0].set_xscale("log"); ax[0].minorticks_off(); ax[0].set_xticks([1040, 4160], ["1,040", "4,160"]); ax[0].set_ylim(0, 1.05); ax[0].set_xlabel("training steps")
ax[0].set_ylabel("count moved by the asked +-1 or +-2"); ax[0].set_title("one scaled direction, best strength"); ax[0].legend(fontsize=7, loc="lower left", ncol=2)
pts = [(sd["turn"], md(sd["steer"]["line"]["s1"]), r["width"]) for r in R if r["concepts"]["count"]["base_acc"] >= 0.9 for sd in r["concepts"]["count"]["sites"].values()]
ax[1].scatter([p[0] for p in pts], [p[1] for p in pts], c=[col[p[2]] for p in pts], s=25)
ax[1].set_xlabel("cosine between steps 0->1, 1->2, ... of the count means (1 = straight line)"); ax[1].set_ylabel("count moved, one direction at strength 1")
[ax[1].scatter([], [], c=col[w], label=f"width {w}") for w in col]; ax[1].legend(fontsize=7, loc="upper left"); ax[1].set_title("the straightness check predicts it (36 model-layer pairs)")
plt.tight_layout(); plt.savefig("../../figures/ordered-values-bend.png", dpi=120)
