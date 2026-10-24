# muon-grad-spikes notes (2026-10-24)

A lab researcher who reads this would change: whether they keep gradient clipping for Muon (Megatron PR #5395, open, removes it because "Newton-Schulz discards magnitude").
Literature (3 extended searches): Megatron PR #5395 / issue #5394 (clip stall via NS eps floor, Qwen3.5-35B-A3B SFT); MuonQ 2605.11396 normalizes grad+momentum, no change on clean data; MVN-Grad 2602.10204 theory: normalizing before momentum bounds spike response (Adam-family); SPAM 2501.06842 (Adam). Nobody measures what a spike does to Muon with vs without clipping.
Mechanism: NS ignores the size of the momentum, not the size of one gradient inside it. A gradient S x larger than usual outweighs the rest of the momentum for about ln(S)/(1-beta) steps (S=100, beta=0.95: ~90 steps), all at full step size.
Prediction: on clean data clip/prenorm change nothing (<0.005). With a bad batch x100 every 50 steps: no clip clearly worse (+0.05 or more); CLIP=1 (2.5x the median grad norm 0.41) recovers most; PRENORM recovers all.

Toy: qkclip-batch model (2 layers, width 16, 20-task mix, 800 steps, Muon 0.95 Nesterov, lr 0.014 = best), ~3 s/run.
- Prediction single spike (x100 at step 400): Adam-only clip +0.02 vs clip-all.
- Prediction MOM=0: Adam-only clip = clip-all (no momentum to carry the spike).
- 2026-10-24 clip threshold far below usual norm (0.001, 0.1) = same as 0.2/0.5 within 0.01 in both toys (3 seeds); added to finding point 6.
