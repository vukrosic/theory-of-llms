"""figures/muon-superposition.png: (1) 2-feature phase diagram, (2) features stored + feature dimensionality vs sparsity,
(3) Muon weight decay caps how many features a 5-dim layer holds"""
import json, os, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
here = os.path.dirname(os.path.abspath(__file__)); out = os.path.join(here, "../../figures/muon-superposition.png")
rs = [json.loads(l) for l in open(os.path.join(here, "runs.jsonl"))]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

ph = json.load(open(os.path.join(here, "phase.json")))
cols = {0: "#9aa5b1", 1: "#e9a23b", 2: "#3b6fb6"}
for i, r in enumerate(ph["rs"]):
    for j, d in enumerate(ph["dens"]):
        st = ph["states"][f"{r},{d}"]; maj = max(set(st), key=st.count)
        ax[0].add_patch(plt.Rectangle((j, i), 1, 1, color=cols[maj], alpha=0.35 + 0.65 * st.count(maj) / 3))
ax[0].set_xlim(0, len(ph["dens"])); ax[0].set_ylim(0, len(ph["rs"]))
ax[0].set_xticks(np.arange(len(ph["dens"])) + 0.5, [str(d) for d in ph["dens"]]); ax[0].set_yticks(np.arange(len(ph["rs"])) + 0.5, [str(r) for r in ph["rs"]])
ax[0].set_xlabel("how often each feature is on"); ax[0].set_ylabel("importance of feature 2 (feature 1 = 1)")
ax[0].set_title("2 features, 1 dimension (Muon, 3 seeds)")
for k, lab in ((0, "only feature 1"), (1, "only feature 2"), (2, "both, opposite signs")): ax[0].scatter([], [], color=cols[k], marker="s", s=80, label=lab)
ax[0].legend(loc="upper left", fontsize=8, framealpha=0.9)

sw = [r for r in rs if r["opt"] == "ns" and r["steps"] == 3000 and r["lr"] == 0.04 and r["wd"] == 0 and r["seed"] < 3 and r.get("imp", 0.9) == 0.9]
byS = collections.defaultdict(list)
for r in sw: byS[r["S"]].append(r)
Ss = sorted(byS); x = [1 - S for S in Ss]
ax[1].plot(x, [np.mean([r["feats"] for r in byS[S]]) for S in Ss], "o-", color="#3b6fb6", label="features stored (of 20)")
ax[1].axhline(5, color="#9aa5b1", ls="--", lw=1); ax[1].text(0.6, 5.4, "5 = number of dimensions", fontsize=8, color="#666")
ax[1].set_xscale("log"); ax[1].invert_xaxis(); ax[1].set_xlabel("how often each feature is on"); ax[1].set_ylabel("features stored")
a2 = ax[1].twinx()
for S in Ss:
    for r in byS[S]:
        d = [v for v in r["dims"] if v > 0.05]; a2.scatter([1 - S] * len(d), d, s=8, color="#e9a23b", alpha=0.5)
for v, lab in ((1, "1"), (0.5, "1/2 pair"), (1 / 3, "1/3 triangle"), (0.4, "2/5 pentagon")): a2.axhline(v, color="#e9a23b", lw=0.5, ls=":")
a2.set_ylabel("dimensions per feature (orange dots)", color="#b5781a"); a2.set_ylim(0, 1.1)
ax[1].set_title("20 features, 5 dimensions (Muon)")

wdr = sorted([r for r in rs if r["opt"] == "ns" and r["S"] == 0.97 and r["seed"] == 0 and r["steps"] == 3000 and r["lr"] == 0.04], key=lambda r: r["wd"])
wds = [r["wd"] for r in wdr]
ax[2].plot(wds, [r["feats"] for r in wdr], "o-", color="#3b6fb6", label="features stored")
ax[2].plot(wds, [r["fro2"] for r in wdr], "s-", color="#e9a23b", label="total feature size (||W||²)")
g = np.linspace(0.45, 2.05, 50); ax[2].plot(g, 5 / g ** 2, "--", color="#555", label="limit 5 / weight decay²")
ax[2].set_ylim(0, 22); ax[2].set_xlabel("Muon weight decay"); ax[2].set_ylabel("features"); ax[2].legend(fontsize=8)
ax[2].set_title("weight decay caps the packing (96% sparse)")
plt.tight_layout(); plt.savefig(out, dpi=130); print(out)
