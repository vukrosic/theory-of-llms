# Do attention scores grow at the NORMAL lr in long runs (Kimi K2's case), and does QK-Clip τ=100 vs tight matter there?

Lab-change sentence: a lab would know whether QK-Clip is needed at its normal lr for long runs, and whether τ=100 is a safe default.
Setup: muon-attn-fixes/attn.py copy, QKNORM=0, seed 0, STEPS = 800 / 3200 / 8000 (more steps = more fresh data, one pass). Decay over last half scales with STEPS.

Prediction: at lr 0.014 the peak score grows slowly (9.8 at 800 → ~15 at 8000), no loss damage vs QK-norm; at 0.028 (2× best) it crosses into blow-up with length (12 → 100+ at 8000).

## Results (seed 0, loss / peak score)
| steps | lr | no QK-norm | QK-Clip 100 | QK-Clip 30 | QK-Clip 10 | QK-norm |
|---|---|---|---|---|---|---|
| 800 | 0.014 | 4.002 / 9.8 | | | | 4.017 / 3.6 |
| 3200 | 0.014 | 3.869 / 27 | | | | 3.866 / 4.4 |
| 8000 | 0.014 | 3.826 / 45 | | 3.826 / 32 | 3.824 / 11 | 3.804 / 6.7 |
| 3200 | 0.028 | 3.880 / 27 | | | | 3.894 / 4.3 |
| 3200 | 0.04 | 3.901 / 386 | 3.895 / 113 | | | 3.898 / 3.2 |
| 3200 | 0.057 | 3.956 / 4333 | 3.912 / 137 | | | 3.898 / 6.6 |
(800 steps reference: 0.04 → 48, 0.057 → 653, loss gap +0.024 at 0.057)

- Prediction half right: at the normal lr the score grows with length (9.8 → 27 → 45, ~steps^0.66), more than predicted, but costs no loss; clipping to 10 or 30 changes nothing (3.824-3.826). The 0.02 QK-norm edge at 8000 is not from the scores (clipping doesn't recover it); one seed.
- The safe lr range doesn't shrink with length (0.028 still healthy at 3200), but above it scores grow much faster with length (0.04: 48 → 386; 0.057: 653 → 4333).
- Big scores are not automatically harmful: 386 at lr 0.04 costs nothing (3.901 vs 3.898). Damage appears in the thousands (0.057: +0.058; τ=100 recovers most: 3.912).

## Literature (3 extended searches, 2026-10-23)
- Kimi K2 (2507.20534): τ=100 per head; max logits sit at the cap then decay after ~30% of training; 12.7% of heads clipped at least once in the first 70k steps; mid-scale run without clip > 1000.
- Anson 2511.21377 "Controlling changes to attention logits": per-parameter lr for q,k as a QK-norm substitute for MLA; reports QK-Clip unstable at high lr (matches our τ=100 failing at 8× best lr). No τ sweep.
- 2606.16310 "QK-Normed MLA": QK-norm made possible for MLA, keeps logits lower than QK-Clip.
- NCP-ArchPreview 2609.10715: layer-wise QK-norm (OLMo-3 style, one norm over all heads) still lets logits grow under Muon; per-head norm stops it.
- Found nowhere: a τ sweep, a rule for setting τ, how data or shape move the need for clipping.

## Layer-wise vs per-head QK-norm (QKNORM=2: RMSNorm over the full width before splitting heads)
Lab-change: labs copying OLMo-style layer-wise QK-norm with Muon would switch to per-head.
Prediction: layer-wise keeps scores ≤ ~4x per-head and loss within 0.02 at best lr; at 0.113 slightly worse than per-head (~4.15 vs 4.10).
Result (loss / peak): 800 steps layer-wise 4.022/8.4, 4.059/9.9, 4.137/18.6 at lr 0.014/0.057/0.113 (per-head 4.017/4.073/4.099). 8000 steps lr 0.014: seed 0 none 3.826 / layer 3.826 / per-head 3.804; seed 1 none 3.802 / layer 3.794 / per-head 3.801.
Prediction right. Layer-wise lets scores reach ~20 (vs ~6 per-head, ~45 none) but loss is the same within seed noise, except +0.04 at 8x best lr. The per-head edge at 8000 steps was seed noise. Dropped as a finding (toy can't show NCP's claim matters).

## Does attention entropy predict loss damage better than max score? (ent.jsonl, seed 0)
Layer-1 heads all collapse to near one-hot attention (entropy 0.01-0.2 vs 2.2-2.6 healthy) in every blown-up run, harmless (3200/0.04: 0.09-0.2, +0.003) and harmful (800/0.057: 0.03-0.05, +0.024; 3200/0.057: 0.008, +0.058) alike. The big damage (800/0.113: +0.33) is the only case where layer 0 also collapses (entropy 0.04-0.08, scores 330-640).
Conclusion: entropy separates no better than max score in this 2-layer toy; the large damage comes when the first layer also collapses. Dropped as a monitoring rule. Corrected "first layer never blows up" in the finding page: it does at 8x best lr.

## τ at 3200 steps, lr 0.057 (seed 0)
none 3.956 / τ100 3.912 / τ30 3.908 / τ10 3.908 / QK-norm 3.898. At 4x best lr in a longer run, τ=100 is as good as tight τ. The big τ=100 failure (800 steps) was only at 8x best lr. Combined with qkclip-data (τ=5 costs +0.33 on copy data at the best lr), the rule "always use tight τ" is wrong: τ=100 is a safe default, tight τ only pays at extreme lr and can hurt sharp-attention data.

## Clip only early (CLIPSTOP=N: τ=10 for steps < N, then off), lr 0.113, seeds 0-2, 800 steps
Lab-change: if clipping only the first ~25% of training is enough, labs can stop paying for the max-score check later (Kimi saw it go inactive after 70k steps).
Prediction: CLIPSTOP=200 gives ~4.15 (close to always-on 4.128); scores regrow after the stop but the damage is done early.
Result: CLIPSTOP=200 4.233/4.157/4.401 (mean 4.264, scores regrow to 1900-2800); CLIPSTOP=400 4.184/4.127/4.235 (4.182, regrow to 620-950); always-on τ=10 4.128; none 4.358. Prediction wrong: stopping the clip lets scores regrow and the loss loses 0.05-0.14. Keep it on while lr is high. Dropped.
