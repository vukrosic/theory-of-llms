# Muon best weight decay vs epochs (repeated data)

**Question:** with total steps fixed and data repeated E times, how does Muon's best decoupled weight decay (on Muon matrices, p *= 1 - lr*f*wd) change?
**Lab-change sentence:** "When you repeat your data N times with Muon, raise weight decay to X (rule: best wd ∝ epochs^a)."
**Setup:** what-changes-muon-lr toy (mix20, D=16, B=8, 800 steps, lr 0.014, decay schedule), pool = 6400/E unique sequences, held-out = the existing eval set. Code: run.py, results: runs.jsonl.

## Literature (5 searches/fetches, 2026-10-23)
- Kim et al. 2025 (2509.14786): jointly tuned wd/lr/epochs for data-constrained pretraining, best wd ~30x the 0.1 default, larger for larger models; Adam-type optimizer, no Muon.
- 2606.06888 (data-constrained AR/diffusion LMs): table of best (epochs, wd, lr) per (model, unique tokens); wd 0.1-3.2; Adam-type; wd falls as unique tokens grow; no lr·wd rule.
- 2605.01640: wd 1.0 vs 0.1 cuts overfitting coefficient ~70%, costs loss at 1 epoch. 2607.25271: more repetition -> higher wd for small models.
- Muon papers (Moonlight 2502.16982, 2506.15054 spectral-norm constraint): wd helps in long/over-trained runs; nothing on epochs.
- Wang & Aitchison / Power Lines 2505.13738: wd timescale lr·wd·steps (Adam-type).
- Not found: a measurement of Muon's best wd as a function of epochs, or a fitted exponent.

## Prediction (E=2,8,32 and wd up to 300 written before running them; E=1,4,16,64 up to wd 10 already seen)
Seen so far: best wd ~0.03 (E=1), ~0.1 (E=4), 3 (E=16), >=10 at the grid edge (E=64).
Predict best wd ∝ E^a with a ~ 1: E=8 ~1, E=32 ~10, E=64 ~30; gain over wd=0 grows from ~0 at E<=4 to >1 nat at E>=16.

## Results (held-out loss, lr 0.014; WDALL=1 = wd on Muon matrices + embedding + head, the usual recipe; seeds 0,1 for E>=4, seed 0 for E=1)
| epochs E | best wd (grid / parabola) | loss at best | loss at wd=0 | gain |
|---|---|---|---|---|
| 1 | 0.03 (flat) | 4.012 | 4.017 | 0.005 |
| 4 | 0.1 / 0.09 | 4.051 | 4.064 | 0.012 |
| 8 | 1 / 0.65 | 4.145 | 4.249 | 0.10 |
| 16 | 3 / 2.3 | 4.252 | 4.974 | 0.72 |
| 32 | 10 / 7.8 | 4.393 | 8.059 | 3.67 |
| 64 | 30 / 18 (10-30 flat) | 4.610 | 17.63 | 13.0 |

Fit for E>=8: best wd ∝ E^1.5 (E^1.2 to E^1.75 between neighbours; 3-point parabola on a coarse grid). wd=0.1 is right at E<=4 and costs 0.7 nat at E=32, 4+ nats at E=64.
- wd on Muon matrices only (embedding/head undecayed): best wd moves the same way (E=16: 3, E=32: 3-10, E=64: >=30 plateau) but loses: E=64 stays at 8.0 vs 4.6, because embedding+head memorize the pool.
- lr 0.028 (WDALL=1, seed 0): best wd unchanged (E=16: 1-3, E=64: 10-30), so lr·wd is NOT the fixed quantity here; embedding/head decay uses the fixed Adam-type lr 0.02, which dominates.

**Conclusion:** with Muon, best weight decay rises about as E^1.5 (0.1 at <=4 epochs, ~1 at 8, ~3 at 16, ~8 at 32, ~20 at 64), and keeping 0.1 at 32-64 epochs loses 0.7-4+ nats of held-out loss.
Caveats: one toy (D=16, 800 steps, task mix, 1-2 seeds), wd normalized as p *= 1 - lr*f*wd; absolute values depend on this scaling, only the E-dependence transfers.
