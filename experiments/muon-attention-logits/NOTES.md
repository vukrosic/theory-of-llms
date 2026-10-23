# Muon max attention logit, with and without QK-norm

Question: how does the max |q.k|/sqrt(d) logit grow under Muon without QK-norm (vs lr, steps, width, momentum), and when does it hurt loss?
Lab-change sentence: "Without QK-norm, Muon's max attention logit grows like lr x (steps/width/momentum), so above lr X or after N steps you must add QK-norm / QK-clip."

## Literature (6 extended searches, 2026-10-23)
- Kimi K2 (arXiv 2507.20534): Muon makes max logits exceed 1000 at scale -> QK-Clip (rescale Wq,Wk per head when max logit > tau=100). Only a few heads explode. No lr/steps law given.
- Wortsman et al. 2023 (2309.14322): logit growth instability appears at high lr in small models (2.4M-1.2B), QK-layernorm removes lr sensitivity over 3 orders of magnitude. Optimizer there: Adam, not Muon.
- Zhai et al. 2023 (2303.06296): low attention entropy <-> instability; entropy lower bound shrinks exponentially with spectral norm of logits (sigma-Reparam).
- QuacK (2511.21377): per-parameter lr on Wq,Wk controls logit change; allows higher base lr. Not a growth law.
- Musec (2609.11655) / Muown (2605.10797) / blog posts: Muon inflates spectral norm of Wq,Wk (flat singular values); no validated formula vs lr, steps. A Substack post claims exponential growth in steps (unvalidated). Muon associative-memory paper (2509.26030): at 160M no max-logit explosion difference.
- Known: Muon logit explosion at 1T scale, QK-Clip fix, QK-norm fix. NOT known (found): a measured law max logit ~ lr^a steps^b width^c / momentum for Muon, or the lr where no-QK-norm Muon starts to lose to QK-norm.

## Prediction (written before runs)
Toy: D=16, B=8, 800 steps, 2 layers, 4 heads (head dim 4), 20-task mix, warmup 20 + linear decay last 50%.
- With QK-norm: max logit stays <~5 at all lrs (bounded by the norm gains).
- Without: max logit ~ lr^1 (Muon step size is fixed per step, so weight norm change ~ lr x steps; logit is bilinear in Wq,Wk so ~ (lr x steps)^2 at large lr). At lr 0.014 ~5-10, at 0.1 > 50.
- Loss: no-QK-norm worse than QK-norm only at lr >= 0.04; best lr and best loss about equal.

## Results (1 seed; toy = D16, B8, 800 steps, mom 0.95 unless noted; `python3 fit.py` reproduces; raw: attn.jsonl)
Peak max logit over training (probe batch) / final loss, no-QK-norm vs QK-norm, D=16:
| lr | 0.0025 | 0.007 | 0.014 | 0.028 | 0.04 | 0.057 | 0.08 | 0.113 | 0.16 |
|---|---|---|---|---|---|---|---|---|---|
| logit, no QK-norm | 8.8 | 9.4 | 9.8 | 12 | 48 | 653 | 1441 | 3160 | 8662 |
| logit, QK-norm | 4.3 | 4.6 | 3.6 | 3.4 | 3.2 | 3.3 | 5.1 | 6.2 | 7.9 |
| loss gap (no QK - QK) | -0.006 | -0.001 | -0.014 | -0.005 | 0.000 | +0.024 | +0.063 | +0.329 | +0.346 |
(logit at init = 2.1; QK-norm logit stays <= 14 in all 40 runs incl. D=64)

Blow-up lr* (peak logit = 5x its lr-0.014 value) and loss cost:
| config | lr* | best lr no-QK / QK | gap at lr 0.057 / 0.113 |
|---|---|---|---|
| 300 steps | 0.050 | 0.028 / 0.014 | +0.02 / +0.03 |
| 800 steps | 0.040 | 0.014 / 0.007 | +0.02 / +0.33 |
| 1200 steps | 0.042 | 0.014 / 0.014 | +0.06 / +0.05 |
| D=32 | 0.043 | 0.028 / 0.014 | +0.05 / +0.40 |
| D=64 | 0.043 | 0.028 / 0.014 | +0.04 / +0.48 |
| mom 0.90 | 0.055 | 0.014 / 0.014 | +0.00 / +0.07 |
| mom 0.98 | 0.029 | 0.014 / 0.014 | +0.04 / +0.29 |

Law (from these runs): below lr* the max logit is flat in lr (about 8-12 at D=16, 20-33 at D=32/64) and grows only slowly with steps (9 -> 8.5 -> 11 at 300/800/1200 steps). At lr* it jumps; above it, peak logit ~ lr^2.6 x steps^1.8 (D=16, 11 points, resid 0.12 ln) = roughly (lr x steps)^2, matching spectral norm of Wq,Wk growing linearly in lr x steps (final ||Wq||_2 = 1.4 at lr 0.005, 2.5 at 0.014, 6.5 at 0.04, 19 at 0.1). lr* is about 3x the best lr, independent of width (0.043 at D=16/32/64) and of steps (0.05/0.04/0.042); it falls with momentum (0.055 / 0.040 / 0.029 at 0.90 / 0.95 / 0.98).
Loss: best lr and best loss with and without QK-norm are equal within noise (<= 0.02); no-QK-norm loses only above lr* (+0.02..0.06 at 1.4x lr*, +0.3..0.5 at 2.8x lr*), the same shape Wortsman et al. report for Adam.
Caveats: 1 seed, loss noise ~0.01-0.02, toy of 800 steps (logit saturation from QK-Clip/weight-decay not tested; no weight decay here).

## Conclusion
In this toy, Muon without QK-norm is fine up to about 3x its best lr and blows up (logit 30-8000, loss +0.05..+0.5) above that, with threshold independent of width and steps and lower at higher momentum; QK-norm costs nothing at the best lr and removes the blow-up.

## Follow-up (1 seed; `python3 fit2.py`; best lr = parabola in log lr through 3 points, loss minimum is flat so +-30%)
1. lr*/best lr per config (best lr from QK-norm runs where the QK grid is full, else from no-QK loss):
| config | lr* | best lr | lr*/best |
|---|---|---|---|
| B=4, 1600 steps | 0.031 | 0.0073 (QK) | 4.3 |
| B=8, 800 steps | 0.040 | 0.0094 (QK) | 4.3 |
| B=32, 200 steps | 0.074 | 0.027 (QK) | 2.7 |
| mom 0.90 / 0.95 / 0.98 | 0.055 / 0.040 / 0.029 | 0.0094 / 0.0073 / 0.0133 (no-QK) | 5.8 / 5.5 / 2.2 |
| D=32 / D=64 (no-QK best) | 0.043 / 0.043 | 0.020 / 0.022 | 2.2 / 2.0 |
Same tokens, 8x batch: best lr moves 3.7x, lr* only 2.4x (lr* ~ B^0.4 vs best lr ~ B^0.6). So "lr* = k x best lr" holds only as a band: 2-6x, lr* > 2x best lr in all 9 configs, and the ratio shrinks as batch grows.
2. Weight decay 0.1 (decoupled, Muon matrices, D16 800 steps): lr* 0.040 -> 0.061; above lr* the logit is capped: final logit 10 / 51 / 55 / 100 at lr 0.057 / 0.08 / 0.113 / 0.16 vs 653 / 1393 / 2963 / 8570 without. Loss gap to no-WD: -0.037 / -0.033 / -0.30 / -0.17 at lr 0.057 / 0.08 / 0.113 / 0.16 (WD run at 0.08: 4.102 vs QK-norm 4.073 without WD). Best loss at low lr unchanged (4.002 at 0.014).
3. Heads above lr* (D16 800 steps, final probe, per-head max logit, no-QK): all 4 heads explode together, layer 1 first. lr 0.057: layer 1 = 564 / 602 / 653 / 354, layer 0 = 2-8. lr 0.113: layer 0 = 326 / 641 / 479 / 604, layer 1 = 2500-2960. Not a few heads (toy has only 4 heads/layer; Kimi saw a few of many).

## Conclusion (follow-up)
lr* grows more slowly with batch than the best lr (ratio 2.7 at B=32, 4.3 at B=4/8), so the safe statement is lr* >= 2x best lr, and weight decay 0.1 raises lr* ~1.5x and caps the logit above it.

## Depth (coordinator, 2026-10-23; depth.jsonl / depth400.jsonl, NL = number of entries in log rows, QKNORM=0, seed 0)
Peak logit at 400 steps, lr 0.01 / 0.02 / 0.04 / 0.057:
| layers | 0.01 | 0.02 | 0.04 | 0.057 |
|---|---|---|---|---|
| 1 | 7.9 | 7.3 | 8.5 | 9.1 |
| 2 | 10.0 | 12.4 | 33.7 | 121 |
| 4 | 10.8 | 14.3 | 73 | 210 |
| 8 | 9.2 | 18.9 | 94 | 180 |
4 layers, 800 steps, final per-layer max logit at lr 0.028: 3.8 / 17 / 19 / 55; at 0.057: 1.3 / 400 / 513 / 562. Layer 0 never blows up (also in the 1-layer model); later layers blow up first and highest. Deeper models start blowing up at a somewhat lower lr (lr 0.04: 34 → 73 → 94 for 2 → 4 → 8 layers).
