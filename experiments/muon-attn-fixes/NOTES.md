# Without QK-norm under Muon: does logit soft-capping or attention output gating replace it?

**Lab-change sentence:** a lab that can't use QK-norm (MLA) and trains with Muon would pick soft-capping / gating / QK-Clip based on which keeps the loss at high lr.
**Setup:** muon-qkclip toy (2 layers, width 16, 800 steps), QKNORM=0, lr 0.014 / 0.057 / 0.113, seed 0. Logged score = raw q.k/sqrt(d) before the cap.
Reference (from ../muon-qkclip): no fix 4.002 / 4.097 / 4.428; QK-Clip 5: 4.004 / 4.059 / 4.071; QK-norm 4.017 / 4.073 / 4.099.

**Prediction (before running):** soft-cap 30 limits the loss damage partly (raw scores still grow to thousands; tanh saturates, gradients die) -> ~4.2 at 0.113. Gating does not stop the raw score growth and helps the loss little -> ~4.35 at 0.113.

## Results (seed 0, loss / peak raw score)
| fix | 0.014 | 0.057 | 0.113 |
|---|---|---|---|
| soft-cap 30 | 4.003 / 9.8 | 4.065 / 690 | 4.221 / 4096 |
| soft-cap 5 | 4.005 / 9.4 | | 4.264 / 4149 |
| output gate | 4.015 / 25 | 4.105 / 599 | 4.432 / 5669 |
Soft-cap prediction right (4.22); gate prediction right (no help). Only tight QK-Clip matches QK-norm. Added as claim 5 in findings/muon-attention-blowup.md.

## Warmup vs the blow-up (WARM env, QKNORM=0, seed 0)
Lab-change: if longer warmup raises the threshold a lot, labs without QK-norm get a free fix.
Prediction: small effect (threshold ×1.3 at most); peak score at lr 0.057 drops from 653 to ~300 with 200 warmup steps.
Result (WARM=200 vs 20): peak 20 vs 48 at 0.04, 322 vs 653 at 0.057, 973 vs 1441 at 0.08; loss at 0.057 4.089 vs 4.097. Prediction right: ~2x smaller scores, threshold up ~1.2x, no loss rescue. Dropped.

## 3-seed check at lr 0.113 (seed noise here ±0.1)
no fix 4.428 / 4.206 / 4.441 (mean 4.358); soft-cap 30 4.221 / 4.148 / 4.246 (4.205); gate 4.432 / 4.149 / 4.428 (4.336); QK-norm mean 4.122 (qkclip-variants). Soft-cap recovers ~2/3 of the gap, gate nothing. Holds.
