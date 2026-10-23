# QK-Clip tau across model shapes

**Lab-change sentence:** A lab using QK-Clip would set tau from a short healthy run (tau ~ k x max score) instead of a fixed 100, and know which shapes need clipping.
**Code:** attn.py (copy of ../muon-attn-fixes/attn.py + env HEADS = number of heads, head dim = D/HEADS; log in runs.jsonl). Toy: 2 layers, 800 steps, B8, T32, Muon mom 0.95, seed 0 unless stated. "healthy max" = peak probe score (every 50 steps, 2 layers' max over heads) without QK-norm at lr 0.014.
Reference at D16 H4 (hd 4): healthy max 9.8; QK-clip tau=5 matches QK-norm at 0.113.

## Batch 1: healthy max + blow-up + QK-norm per shape (QKNORM=0 and 1, lr 0.014/0.057/0.113)
Prediction: healthy max grows with head dim and width (D32 H8 already gave 24.9 vs 9.8 at D16 H4); blow-up at 0.113 (>1000) in every shape; QK-norm loss at 0.113 beats no-QK-norm in every shape.
### Batch 1 result (no tau; loss / peak score; 1 seed). Raw runs.jsonl, table via `python3 show.py`
| shape (hd) | healthy max @0.014 | 0.057 | 0.113 | loss no-QKnorm 0.014/0.057/0.113 | QK-norm 0.014/0.057/0.113 | QK-norm peak score |
|---|---|---|---|---|---|---|
| D16 H1 (16) | 8 | 479 | 4390 | 4.003/4.107/4.607 | 4.023/4.065/4.075 | 5-6 |
| D16 H2 (8) | 11 | 300 | 4075 | 3.998/4.093/4.488 | 4.001/4.072/4.099 | 3-6 |
| D16 H4 (4) | 10 | 653 | 3160 | 4.002/4.097/4.428 | 4.017/4.073/4.099 | 3-6 |
| D32 H1 (32) | 12 | 693 | 4490 | 3.881/3.937/4.256 | 3.878/3.905/3.974 | 6-7 |
| D32 H2 (16) | 23 | 921 | 6001 | 3.865/3.938/4.405 | 3.872/3.894/3.954 | 5-9 |
| D32 H4 (8) | 25 | 1740 | 3715 | 3.874/3.955/4.361 | 3.878/3.907/3.964 | 6-8 |
| D32 H8 (4) | 25 | 543 | 4808 | 3.882/3.958/4.282 | 3.880/3.915/3.967 | 3-7 |
| D64 H4 (16) | 31 | 605 | 4274 | 3.802/3.852/4.333 | 3.805/3.809/3.850 | 10-14 |
| D64 H8 (8) | 62 | 895 | 8889 | 3.808/3.861/4.258 | 3.811/3.811/3.874 | 6-15 |
Prediction check: right (every shape blows up at 0.113, QK-norm wins by 0.2-0.5 there). Healthy max grows with WIDTH (8-12 at D16, 12-25 at D32, 31-62 at D64) not head dim (D32: hd 32 gives 12, hd 4-16 give 23-25).

## Batch 2: tau sweep at 0.113 and 0.014 per shape, tau in {100,30,10,5,2.5}
Prediction: best tau scales with healthy max (D16 ~5, D32 ~10-12, D64 ~15-30), so a fixed tau=5 is right at D16 but over-clips (costs >0.01 at 0.014) at D64; tau=100 leaves a gap >0.1 at 0.113 in every shape.
### Batch 2 result (gap = loss - QK-norm loss at 0.113; cost = loss - unclipped loss at 0.014; 1 seed, noise ~0.01-0.02)
| shape (healthy max / median head) | tau=100 gap | 30 | 10 | 5 | 2.5 | cost at 0.014 for tau 5 / 2.5 |
|---|---|---|---|---|---|---|
| D16 H1 (8 / 5.8) | +.31 | +.26 | +.05 | +.01 | -.02 | +.003 / +.014 |
| D16 H2 (11 / 5.1) | +.25 | +.14 | +.07 | -.01 | -.03 | +.001 / +.011 |
| D16 H4 (10 / 5.0) | +.20 | +.09 | +.06 | -.03 | -.01 | +.002 / +.015 |
| D32 H1 (12 / 8.0) | +.14 | +.09 | +.04 | -.03 | -.04 | -.004 / .000 |
| D32 H2 (23 / 5.6) | +.24 | +.12 | +.04 | -.03 | -.02 | +.005 / +.012 |
| D32 H4 (25 / 6.7) | +.19 | +.11 | +.02 | -.03 | -.03 | .000 / +.007 |
| D32 H8 (25 / 7.0) | +.19 | +.09 | +.03 | +.02 | -.02 | +.001 / +.002 |
| D64 H4 (31 / 8.6) | +.23 | +.11 | +.02 | .00 | +.10 | +.012 / +.020 |
| D64 H8 (62 / 7.9) | +.22 | +.09 | +.02 | +.05 | +.11 | +.001 / +.016 |
Prediction check: wrong. Best tau does NOT scale with healthy max (8 -> 62 is 8x; best tau stays ~5, 5-10 at D64). tau=100 leaves +0.14 to +0.31 in all 9 shapes (even where 100 is only 1.6x the healthy max). The healthy MAX is set by one outlier head (D32H4: heads 5.9,4.6,5.0,7.5 | 7.4,24.6,7.7,6.0; D64H8 one head 52 vs others 6-18); the MEDIAN head score is 5-9 in every shape and matches best tau (~0.6-1x median).
New hypothesis: tau ~ 0.6-1x MEDIAN head score of a healthy run (not max).

## Batch 3: finer tau {20,15,7,3.5} at 0.113 and 0.014 per shape
Prediction: loss at 0.113 is flat within +-0.02 of QK-norm for tau between ~0.5x and ~1.2x median head score (D16: 3-6, D32: 3.5-8, D64: 5-10), and degrades below 0.4x median (D64 tau 2.5, 3.5) -> a lab setting tau = 0.8 x median head score is in the flat zone for every shape.
### Batch 3 result (tau.py; units = tau / median head score of healthy run; gap@0.113 [cost@0.014]; 1 seed)
tau ~ 0.7-1.0 x median head score: gap at 0.113 is -0.03..+0.03 in all 9 shapes (D64 H4 0.81x +0.031, D64 H8 0.88x +0.030, rest <= +0.020) and cost at 0.014 <= +0.006. At 1.2-1.5x gap +0.00..+0.03 (D16 H4 1.4x: -0.02); at 1.7-2x +0.02..+0.07; at 3x +0.05..+0.09; tau 100 (12-20x median) +0.14..+0.31. Below 0.45x median D64 degrades (+0.08..+0.11, cost +0.02) while D16/D32 still fine. Prediction right.
In units of healthy MAX the same good tau is 0.2-0.6x max (D16: 0.5, D32 H4: 0.28, D64 H8: 0.11-0.14), so "k x max" is a 3-4x looser rule than "k x median"; max is set by one outlier head.

## Batch 4: depth 4 and momentum 0.98 (D16 H4 and D32 H4), lr 0.007/0.014/0.057/0.113, no clip + QK-norm, then tau sweep at 0.113/0.014
Prediction: same rule holds: tau ~0.8x median head score is within +0.03 of QK-norm at 0.113 in both; healthy lr 0.014 may already be near the blow-up threshold for NL4/mom 0.98 (threshold lr lower), so median head score will be somewhat higher (8-12) and best tau ~ 6-10.
### Batch 4 results (no tau; 1 seed; loss / peak score; median head = median over layers x heads of final-probe per-head max at lr 0.014)
| config | no-QKnorm 0.007/0.014/0.057/0.113 | QK-norm 0.014/0.057/0.113 | healthy max / median head @0.014 |
|---|---|---|---|
| D16 H4 NL4 | 3.983/3.990/4.120/4.498 | 3.989/4.032/4.172 | 14 / 6.1 |
| D32 H4 NL4 | 3.867/3.864/3.970/4.352 | 3.852/3.890/3.945 | 29 / 6.4 |
| D16 H4 mom .98 | 4.008/4.017/4.130/4.460 | 4.020/4.089/4.168 | 11 / 5.1 |
| D32 H4 mom .98 | 3.893/3.877/3.989/4.384 | 3.884/3.926/3.993 | 31 / 5.5 |
Prediction wrong on the number: median head score at 0.014 is again 5-6.4 (not 8-12); blow-up (500-13000) at 0.057/0.113 in all four. Caution: median head score is 5-9 in all 13 configs so far, so "median rule" and "constant tau ~5" are not yet separable -> need configs where the median head score moves (longer training, lower lr, weight decay, batch).

Batch 5 (next): tau sweep {30,15,10,7,5,3.5,2.5} for the four Batch-4 configs.
Prediction: best tau 5-7 in all four, gap at tau 5-7 <= +0.03; NL4 needs no per-layer tau.
### Batch 5 result (tau sweep for Batch-4 configs; gap@0.113 [cost@0.014]; 1 seed; tau.py)
| config (median head) | 30 | 15 | 10 | 7 | 5 | 3.5 | 2.5 |
|---|---|---|---|---|---|---|---|
| D16 NL4 (6.1) | +.020 | -.008 | -.001 | -.004 | -.039 | -.011 | +.015 [+.005] |
| D32 NL4 (6.4) | +.063 | +.037 | +.033 | +.036 | .000 [+.004] | +.065 [+.008] | +.002 [+.014] |
| D16 mom.98 (5.1) | +.061 | +.027 | -.010 | -.001 | .000 | +.016 | +.023 [+.003] |
| D32 mom.98 (5.5) | +.127 | +.044 | +.015 | +.014 | +.031 [+.004] | +.055 [+.006] | +.082 [+.012] |
Prediction right: best tau 5-10 (0.8-1.8x median), gap <= ~+0.03; cost at 0.014 <= +0.006 for tau >= 3.5. Depth 4 / mom .98 need no different rule; tau=30 leaves +0.02..+0.13 (more than 5-10), tau 100 not run here.

## Batch 6 (coordinator request): training length STEPS=3200 (healthy max 27 vs 9.8 at 800), D16 H4 NL2
Prediction: median head score at lr 0.014 rises from 5 to ~8-12; best tau at 0.113 moves from ~5 to ~10-20 (if the median rule holds); if tau 5 still wins it is a constant.
### Batch 6 result (D16 H4, STEPS=3200, loss; gap vs QK-norm at lr 0.113; cost = clipped - unclipped at 0.014; long.jsonl, `python3 tau.py long.jsonl`)
healthy lr 0.014: max 27, median head 8.4 (5.0 at 800 steps); QK-norm 3.866 (seed 0) / 3.933@0.113 (seed 1).
| tau (x median) | gap@0.113 s0 | gap@0.113 s1 | cost@0.014 s0 | cost@0.014 s1 |
|---|---|---|---|---|
| none | +.103 | +.441 | | |
| 100 (12x) | +.022 | +.150 | 0 | |
| 40 (4.8x) | +.020 | | 0 | |
| 20 (2.4x) | +.009 | -.001 | 0.000 | |
| 10 (1.2x) | +.007 | -.012 | +.004 | |
| 5 (0.6x) | +.007 | -.019 | +.007 | +.017 |
Prediction half-right: median rises 5.0 -> 8.4 as predicted, but at 3200 steps every tau <= 40 is within +0.02 of QK-norm at 0.113 (blow-up is milder: no-clip gap +0.10 s0 but +0.44 s1), so tau barely matters except that tau=100 is unreliable (+0.02 s0, +0.15 s1) and tau=5 now costs +0.007..+0.017 at the best lr. Consistent with tau ~ 1-2x median (10-20 here, 5-7 at 800 steps): tau must rise with the median, a constant 5 would cost at long training.

## Batch 7: per-head vs per-layer clip (QKCLIPL=1: one gamma from the layer's max head), D32 H8 (outlier head 25 vs median 7), tau 5/10 at 0.113/0.014
Prediction: per-layer costs more at 0.014 (outlier head drags the 7 normal heads down): +0.01..+0.03 at tau=10 vs ~0 per-head; at 0.113 similar to per-head.
### Batch 7 result (D32 H8, 800 steps, seed 0; loss at 0.113 / at 0.014; per-head numbers from Batch 2-3)
| tau | per-head | per-layer (QKCLIPL=1) |
|---|---|---|
| 10 | 3.992 / cost -.001 | 3.987 / cost +.002 |
| 5 | 3.987 / cost +.001 | 3.947 / cost +.005 |
Prediction wrong: per-layer clipping costs no more than per-head here (differences <= 0.04 at 0.113 in per-layer's favour, <= 0.005 at 0.014, within noise). Per-head clipping is not important in this toy.
Number of heads ever clipped at the best lr (nclip counts head-steps, 800 steps x layers x heads): tau 10 -> 2-281 head-steps (1-2 outlier heads); tau 5 -> 77-1256; tau >= 1.5x healthy max -> 0.
Dropped (one line each): tau as k x healthy max (3-4x spread of k across shapes, outlier-head driven); constant tau 5 (costs +0.007..+0.017 at 3200 steps).
ANSWER: set tau = 1 to 1.5 x the MEDIAN head score of a short healthy run (median over layers x heads of the per-head max |q.k|/sqrt(d) at your best lr; 5-9 for all 13 shapes at 800 steps, 8.4 at 3200 steps) -> within +0.03 of QK-norm at 8x the best lr and <= +0.006 cost at the best lr (800 steps, 1 seed); k x healthy MAX is a worse rule (best tau is 0.1-0.5x max across shapes because one outlier head sets the max) and Kimi's fixed 100 leaves +0.14..+0.31 (800 steps) / +0.02..+0.15 (3200 steps) at 8x best lr.

## Batch 8 (Opus): STEPS=8000, D16 H4 (healthy max 45 at 800->8000 grows): does the needed tau keep rising with the median?
Prediction: median head at 0.014 ~12-15; tau 5 costs +0.02..+0.04 at 0.014, tau 20 costs ~0; at 0.113 tau 10-20 within +0.02 of QK-norm, tau 100 worse.
Result (seed 0, long8k.jsonl): healthy max 45 but median head 7.2 (one outlier head 36.9, rest 3-8; the median did NOT keep rising). lr 0.113 loss: none 3.912 (+0.070 vs QK-norm 3.842, layer-2 scores ~1e5), tau 5/10/20/100 3.859/3.858/3.861/3.861 (+0.016..+0.019, all equal). Cost at 0.014: tau 5 +0.005, tau 10 -0.002, tau 20 -0.003. Prediction wrong: in long runs the blow-up is mild and every tau 5-100 gives the same loss; tau only matters in short high-lr runs. Tau below the median head costs a little at the best lr (+0.005 here, +0.007..+0.017 at 3200).
