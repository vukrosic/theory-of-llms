# QK-Clip implementation details (Muon, no QK-norm, toy D=16 H=4 B=8 800 steps)
Lab-change sentence: "A lab implementing QK-Clip would know which implementation details matter (per-head vs per-layer, q/k split, how often to check, probe batch vs training batch) and how weight decay / warmup change the needed tau."
Code: attn.py (copy of ../muon-attn-fixes/attn.py + knobs off by default: CLIPMODE=head|layer, QSPLIT, CLIPPOW, CLIPEVERY, CLIPSTART, CLIPPROBE; logs first=first clipped step, nclip). Output clip.jsonl.
Run: QKNORM=0 QKCLIP=10 nice -n 20 python3 attn.py 0.113 SEED
Sanity: new default tau=10 lr=0.113 seed0 -> 4.1177 vs original code 4.1572 (exp/log vs sqrt rounding only) => chaotic regime; measure seed noise first.

## Batch 0: seed noise at lr 0.113, tau=10 and tau=100, seeds 0-2
Prediction: seed spread ~0.03 at tau=10, ~0.06 at tau=100; differences below that are noise.
Result (lr 0.113, seeds 0/1/2, loss): none 4.43/4.21/4.44 (mean 4.358); tau=10 4.12/4.11/4.16 (4.128, peak 19.5); tau=100 spread 4.14-4.35. Seed spread is 0.05 (tau=10) to 0.23 (none, tau=100), so single-seed loss at lr 0.113 is NOT readable to better than ~0.1.
=> Decision: all points below use 3 seeds (run.sh, mean loss + per-seed + mean peak logit); peak/final logit are the cleaner signal. Summary lines also in summary.txt.

## Batch 1: head vs layer, q/k split, overshoot (lr 0.113, tau 10 and 100)
Prediction: per-layer clip (one gamma from max head) is ~same loss as per-head at tau=10 (<0.05 diff) but clips more heads (more nclip); all-on-q or all-on-k ~same as sqrt split (score is bilinear) except all-on-k slightly worse/better by <0.05; overshoot pow=2 hurts a bit (+0.05) since it shrinks the heads below tau.
Result batch 1 (lr 0.113, 3-seed mean loss; peak logit): 
| variant | tau=10 loss/peak | tau=100 loss/peak |
| head, sqrt split (default) | 4.128 / 19.5 | 4.261 / 145 |
| layer (one gamma from max head) | 4.124 / 19.3 | 4.255 / 130 |
| all gamma on q | 4.141 / 26.7 | 4.272 / 157 |
| all gamma on k | 4.132 / 24.6 | 4.283 / 171 |
| overshoot pow=2 | 4.147 / 20.7 | 4.256 / 138 |
(no clip: 4.358 / 3791). Seed noise ~0.05, so all variants are flat on loss (<=0.02-0.03 apart, within noise); only the q/k split shows in peak logit (all-on-q/k 25-35% higher peak than sqrt split, nclip 1.5x more, since q-only/k-only rescale hits the same heads' score but the other matrices' relative sizes drift).
Conclusion: per-head vs per-layer, q/k split, overshoot: no detectable effect on loss. tau is what matters (tau=10 vs 100: -0.13 loss, paired in all 3 seeds).

## Batch 2: frequency, start delay, probe batch (lr 0.113, tau=10)
Prediction: every 10: loss +0.03, peak ~2x; every 50: loss +0.1 and peak in the hundreds (scores can grow between checks); probe-batch scores: ~same loss; start delay 100/200: peak and loss worse by <0.1 (clipping rescales the weights so it can recover).
Result batch 2 (lr 0.113, tau=10, 3-seed mean; peak = max logit on fixed probe batch over the run):
| variant | loss | peak | nclip |
| every step, train-batch scores (default) | 4.128 | 19.5 | 2166 |
| every 10 steps | 4.144 | 44 | 375 |
| every 50 steps | 4.196 | 241 | 94 |
| probe-batch scores every step | 4.131 | 10.3 | 3727 |
| probe-batch scores every 10 | 4.127 | 33.7 | 408 |
| start at step 100 | 4.158 | 188 | 2003 |
| start at step 300 | 4.216 | 1393 | 1367 |
Conclusions: checking every 10 steps costs nothing (+0.016, within noise) and cuts clip count 6x; every 50 costs +0.07. Probe vs train batch: same loss (probe batch pins the probe max to tau exactly, train-batch clip leaves probe max at ~2tau). Delayed start is mostly rescued: starting at step 300 after scores reached 1400 still gives 4.216 vs 4.358 unclipped; early start is worth 0.09 over late. Clipping works as a rescue, not only prevention.

## Batch 3: weight decay and warmup interaction (lr 0.113 and 0.17)
Prediction: wd 0.1 lr 0.113 no clip: peak ~300, loss ~4.2; tau=100 adds <0.05; tau=10 still gives ~0.05-0.1. Warmup 200: peak halves, tau=100 no loss help.
Result batch 3a (wd 0.1, 3-seed mean loss / peak):
| lr | none | tau=100 | tau=30 | tau=10 |
| 0.113 | 4.152 / 166 | 4.158 / 122 (86 clips, first at step ~175) | 4.142 / 43 | 4.124 / 16 |
| 0.17 | 4.381 / 353 | 4.324 / 137 | - | 4.187 / 18 |
(no wd, lr 0.113: none 4.358 / 3791, tau=100 4.261, tau=10 4.128.)
Conclusion: wd 0.1 alone at 0.113 gives the loss that tau=10 gives without wd (4.152 vs 4.128); tau=100 on top of wd does nothing (scores of ~160 rarely cross 100). At lr 0.17 wd isn't enough (4.38), tau=10 gives 4.19, tau=100 4.32.
Prediction batch 3b: warmup 200 (no wd) lr 0.113: tau=100 no better than WARM=20 (+-0.05); tau=10 same. wd+warmup: both together no extra gain. QK-norm ref ~4.10.
Result batch 3b (lr 0.113, 3-seed mean loss / peak): QK-norm ref 4.122 / 5.1 (wd0.1 lr0.17: 4.149).
| setting | none | tau=100 | tau=10 |
| warm20 (default) | 4.358 / 3791 | 4.261 / 145 | 4.128 / 19.5 |
| warm200 | 4.156 / 2177 | 4.113 / 136 | 4.085 / 16.7 |
| warm200 + wd0.1 | 4.134 / 133 | 4.138 / 101 (10 clips) | - |
| warm20 + wd0.1 | 4.152 / 166 | 4.158 / 122 | 4.124 / 16 |
Surprise (my earlier context said warmup 200 does not rescue the loss; that was lr 0.057): at lr 0.113 warmup 200 alone cuts loss 4.358 -> 4.156 and the seed spread (4.43/4.21/4.44 -> 4.16/4.14/4.16), though scores stay 2000. Warmup 200 + tau=10 gives 4.085, better than QK-norm 4.122 at warm20.
Conclusion: wd and warmup each rescue the loss at lr 0.113 and both make tau=100 useless (<=500 clips, loss within noise of no clip); tau=10 still adds ~-0.03 to -0.07 on top of either.

## Batch 4: tau sweep and lr dependence
Prediction: tau 3/5 at lr 0.113 ~4.09-4.11 (no worse than tau=10; too-tight does not hurt in this toy); tau=30 ~4.17; at lr 0.057 clip gaps are ~0.02 (all within noise); at lr 0.17 wd0.1, tau=3/5 beat QK-norm 4.149.
Result batch 4 (3-seed mean loss / peak):
lr 0.113 no wd: tau 3: 4.115 / 8.5; tau 5: 4.116 / 10.5; tau 10: 4.128 / 19.5; tau 30: 4.186 / 47; tau 100: 4.261 / 145; none 4.358 / 3791; QK-norm 4.122 (warm200: 4.068; warm200+tau10 4.085).
lr 0.17 wd0.1: none 4.381 / 353; tau100 4.324; tau10 4.187; tau5 4.146; tau3 4.144; QK-norm 4.149 / 3.6.
lr 0.057 no wd: none 4.093 / 459; tau100 4.083; tau10 4.063; QK-norm 4.057 (gap none vs QK-norm only 0.036 despite score 459).
Conclusion: loss is monotone in tau (lower is better down to 3, no U-shape) at the lrs where scores blow up; tau<=5 matches QK-norm at lr 0.113 and 0.17+wd; tau=100 recovers only ~1/3 of the gap. Needed tau does not grow with lr: it is "healthy score level" (~5-10), not "just below the blow-up level".

## Batch 5: cost of too-tight tau where no clip is needed (lr 0.014 best lr, scores ~10)
Prediction: tau=10 ~ +0.00 (clips rarely), tau=3 +0.01-0.03, tau=1 +0.05 or more (clipping fights the model). At lr 0.113 tau=1 ~4.12 (still near QK-norm).
Result batch 5 (3-seed mean loss / peak): lr 0.014: none 4.022 / 9.8; tau10 4.022 (3 clips); tau3 4.024 (845 clips); tau1 4.044 (+0.022); QK-norm 4.029. lr 0.113 tau1: 4.095 (better than QK-norm 4.122).
Conclusion: tau=10 is free at the best lr (almost never fires); tau>=3 costs <=0.002; tau=1 costs +0.022 at the best lr. Recommended tau ~3-10 for this model (typical healthy max score ~10 at the best lr).

## Batch 6: recipe check, tau=5 every 10 steps and per-layer (lr 0.113)
Prediction: tau=5 every10 ~4.13 (every10 cost +0.016 at tau=10); tau=5 layer ~4.12 (same as head).
Result batch 6 (lr 0.113, 3 seeds): tau5 head every step 4.116; tau5 every10 4.119 (peak 43, 419 clips); tau5 layer 4.108; tau5 every10 probe 4.120. Predictions right. At tight tau the check frequency costs nothing (+0.003).
Dropped (one line each): per-layer vs per-head (flat), q/k split (flat on loss), overshoot pow2 (flat), probe vs training batch (flat). Not tested: data dependence, MLA-style shared rope exception, larger models.
Note: single-seed loss at lr 0.113 has +-0.1 noise (unclipped 4.21-4.44), so every point above is a 3-seed mean (deviation from the one-seed rule). ls: summary.txt has the raw lines.
