"""Peak attention logit vs lr (attn.jsonl) and loss vs lr for QK-Clip thresholds (../muon-qkclip/qkclip.jsonl), seed 0. -> ../../figures/muon-attn-blowup.png"""
import json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A = [json.loads(l) for l in open("attn.jsonl")]
Q = [json.loads(l) for l in open("../muon-qkclip/qkclip.jsonl")]
base = lambda r: r["D"] == 16 and r["B"] == 8 and r["steps"] == 800 and r["mom"] == 0.95 and r["seed"] == 0
def pts(rows, key, **kw):
    d = {r["lr"]: r[key] for r in rows if base(r) and all(r.get(k, 0) == v for k, v in kw.items())}
    return sorted(d), [d[x] for x in sorted(d)]
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
for kw, lab, c in [(dict(qknorm=0, wd=0), "no QK-norm", "#d1495b"), (dict(qknorm=0, wd=0.1), "no QK-norm, weight decay 0.1", "#edae49"),
                   (dict(qknorm=1, wd=0), "QK-norm", "#00798c")]:
    x, y = pts(A, "maxlogit_peak", **kw); a.plot(x, y, "-o", color=c, ms=4, lw=2, label=lab)
a.axvline(0.014, color="#888", ls=":", lw=1.2); a.text(0.0145, 3000, "best lr", color="#555", fontsize=9)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("learning rate"); a.set_ylabel("largest attention score during training")
a.set_title("Without QK-norm, attention scores blow up above ~3x the best lr", fontsize=11); a.legend(frameon=False, fontsize=9)
for t, lab, c in [(0, "no QK-norm", "#d1495b"), (100, "QK-Clip 100", "#f4a259"), (30, "QK-Clip 30", "#bc8034"), (5, "QK-Clip 5", "#6a4c93")]:
    x, y = pts(Q, "loss", qknorm=0, qkclip=float(t)); b.plot(x, y, "-o", color=c, ms=4, lw=2, label=lab)
x, y = pts(Q, "loss", qknorm=1, qkclip=0.0); b.plot(x, y, "--s", color="#00798c", ms=4, lw=2, label="QK-norm")
b.set_xscale("log"); b.set_xticks([0.014, 0.028, 0.057, 0.113], ["0.014", "0.028", "0.057", "0.113"]); b.minorticks_off(); b.set_xlabel("learning rate"); b.set_ylabel("final loss (lower is better)")
b.set_title("A tight QK-Clip threshold matches QK-norm; 100 does not", fontsize=11); b.legend(frameon=False, fontsize=9)
for ax in (a, b): ax.spines[["top", "right"]].set_visible(False)
fig.text(.01, .01, "2-layer Muon language model, width 16, 800 steps, one seed. Attention score = largest q.k/sqrt(d) on a fixed batch.", fontsize=8.5, color="#444")
fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig("../../figures/muon-attn-blowup.png"); print("ok")
