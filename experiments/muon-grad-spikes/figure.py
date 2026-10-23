"""Loss vs gradient-spike size for Muon clipping variants (numbers from runs.jsonl, seed 0) and the clip threshold. -> ../../figures/muon-grad-clipping.png"""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150, gridspec_kw=dict(width_ratios=[1.5, 1]))
x = [4.6, 15, 46, 170, 457, 1520]                      # spike gradient norm / median gradient norm, for loss scale 3, 10, 30, 100, 300, 1000
S = dict(none=[4.034, 4.109, 4.232, 4.489, 4.859, 5.484], adam=[4.030, 4.050, 4.108, 4.157, 4.186, 4.241],
         all=[4.019, 4.024, 4.024, 4.024, 4.024, 4.024], pre=[4.022, 4.025, 4.025, 4.025, 4.025, 4.025])
for k, n, c, m in [("none", "no clipping", "#888888", "o"), ("adam", "clip only the Adam group (Megatron PR #5395)", "#d1495b", "o"),
                   ("all", "clip everything at 1.0", "#00798c", "s"), ("pre", "Adam group clipped, each Muon gradient normalized before momentum", "#edae49", "^")]:
    a.plot(x, S[k], marker=m, color=c, label=n, lw=2, ms=5)
a.axhline(4.017, color="#444", ls=":", lw=1); a.text(1000, 4.003, "no spikes", fontsize=8.5, color="#444")
a.set_xscale("log"); a.set_ylim(3.98, 4.6); a.set_xlabel("spike size (gradient norm / usual gradient norm)"); a.set_ylabel("held-out loss")
a.set_title("A bad batch every 50 steps", fontsize=11); a.legend(frameon=False, fontsize=8.5, loc="upper left")
a.text(1500, 4.585, "no clipping: 4.86, 5.48 →", ha="right", fontsize=8, color="#888888")
th = ["0.2\n(0.5x usual)", "1.0\n(2.4x)", "3\n(7x)", "10\n(25x)", "none"]; v = [4.015, 4.024, 4.060, 4.227, 4.489]
b.bar(range(5), [t - 3.98 for t in v], bottom=3.98, color=["#00798c"] * 4 + ["#888888"], width=.6)
for i, t in enumerate(v): b.text(i, t + .01, f"{t:.3f}", ha="center", fontsize=9.5)
b.set_xticks(range(5), th, fontsize=8.5); b.set_ylim(3.98, 4.6); b.set_ylabel("held-out loss"); b.set_xlabel("global clip threshold")
b.set_title("Lower clip threshold is better (spike 170x)", fontsize=11)
for ax in (a, b): ax.spines[["top", "right"]].set_visible(False)
fig.text(.01, .01, "2-layer model, width 16, 20-task mix, 800 steps, Muon (momentum 0.95) on matrices, Adam on embedding, head and norms; bad batch = random tokens with its loss scaled up; seed 0.", fontsize=8.5, color="#444")
fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig("../../figures/muon-grad-clipping.png"); print("ok")
