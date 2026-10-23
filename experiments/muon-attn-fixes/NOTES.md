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
