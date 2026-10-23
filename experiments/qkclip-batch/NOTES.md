# Does the blow-up margin shrink with batch size? (Muon, no QK-norm)
Lab-change sentence: a lab training with large batches would know that the safe gap between its best lr and the blow-up lr shrinks with batch, so at lab batch sizes QK-norm/QK-Clip is needed at the normal lr.
Earlier (muon-attention-logits): ratio blow-up lr / best lr = 4.3 at B=4 and 8, 2.7 at B=32 (same tokens); lr* ~ B^0.4, best lr ~ B^0.6.
Setup: attn.py copy of qkclip-shape (B env), STEPS=200 for all batches (more tokens at bigger batch), D16 H4, seed 0, lr grid 0.01..0.32 (sqrt 2 steps), QKNORM 0 and 1.
Blow-up lr = first lr where no-QK-norm peak score >= 5x its value at the best lr (best lr = argmin QK-norm loss).
Prediction: ratio ~4 (B8), ~2.7 (B32), ~1.7-2 (B128); by B^-0.2, ~1.2 at B=4096.
Result (seed 0, runs.jsonl, show.py): blow-up lr / best lr = 2.9 / 4.0 / 2.8 at B=8/32/128 (200 steps); damage (gap > 0.03) starts at 2.9x / 11x / 5.6x. Prediction wrong: the margin does not shrink with batch here; bigger batches tolerate large scores better. Dropped.

## Weight decay on W_q, W_k only (WDQK) instead of QK-Clip
Lab-change: if decaying only the query/key weights fixes the blow-up, labs get QK-Clip's effect with a one-line optimizer setting and no max-logit tracking in the attention kernel.
Setup: B=8, 800 steps (default), lr 0.113 (8x best) seeds 0-2, and cost at lr 0.014 seed 0. Refs (qkclip-variants, 3 seeds): none 4.358, wd 0.1 all 4.152, tau 10 4.128, QK-norm 4.122.
Prediction: WDQK 1 -> ~4.15, WDQK 3 -> ~4.13 (matches tau 10), cost at 0.014 <= +0.01 for WDQK <= 1, +0.02 at 3.
Result (wdqk.jsonl, 3 seeds at 0.113 / 3 seeds at 0.014): WDQK 0.3: 4.116 (peak 47) / cost 0.000; WDQK 1: 4.101 (peak 10) / +0.016; WDQK 3: 4.095 / +0.024 (seed 0); WDQK 10: 4.096 / +0.024 (s0). WD 0.1 all: 4.152 / +0.004; WD 0.1 + WDQK 0.3: 4.129 / +0.004. Prediction beaten: WDQK 0.3 alone matches QK-norm (4.122) and tau 10 (4.128) at 8x and is free at the best lr.
Literature (3 extended searches + read 2511.21377): no paper tests q/k-only weight decay as a QK-Clip substitute. Anson & Aitchison 2511.21377 (Muon) lower the q/k LEARNING RATE instead (norm-dependent), never mention weight decay. Kimi K2 used weight decay on everything and still saw blow-up. Theory guess: Muon steps have fixed size, so with decay lr*wd the q/k norms settle at ~sqrt(lr/wd) and scores at ~lr/wd: the bound tightens exactly when needed.
Batch 2: lr 0.17 (wd 0.1 all failed there: 4.381; tau 5 4.146; QK-norm+wd 4.149), and D32 H8.
Prediction: WDQK 1 at 0.17 ~4.15; WDQK 0.3 ~4.2. D32 H8 WDQK 0.3 within 0.02 of QK-norm at 0.113, free at 0.014.
Result batch 2: lr 0.17 (12x best, 3 seeds): none 4.523 (peak ~11000), WDQK 0.3 4.212, WDQK 1 4.187, QK-norm 4.153 (s0). D32 H8 seed 0: 0.113 none 4.282 / WDQK 0.3 3.956 / QK-norm 3.967; 0.014 3.882 / 3.878 / 3.880. Prediction right.
Batch 3: sharp-attention data (copy only c1.0; algo only) and 8000-step cost at the best lr.
Prediction: on copy data q/k decay hurts like tight QK-Clip (+0.1..+0.3 at 0.014) because copying needs scores ~50+; at 8000 steps WDQK 0.3 costs <= +0.01.
Result batch 3 (seed 0, wdqk_data.jsonl): copy only at 0.007 / 0.014: none 3.526 / 3.251, WDQK 0.3 3.531 / 3.250, WDQK 1 3.617 / 3.632 (+0.38), QK-norm 3.190 / 3.052. Algo only at 0.014 / 0.04: none 3.456 / 3.762, WDQK 0.3 3.455 / 3.646, WDQK 1 3.451 / 3.547, QK-norm 3.437 / 3.597. 8000 steps at 0.014: WDQK 0.3 3.835 (+0.009 vs none 3.826), WDQK 1 3.843 (+0.017). Prediction half right: 0.3 is free on copy data (1 hurts +0.38, like tight clip); long runs make q/k decay cost a little.
Batch 4: keep the decay timescale 1/(lr*wd) a fixed fraction of training: at 8000 steps use WDQK 0.03 (same lr*wd*steps as 0.3 at 800).
Prediction: 8000 steps WDQK 0.03 costs <= +0.003 at 0.014 and recovers most of the +0.070 gap at 0.113 (none 3.912, QK-norm 3.842).
Result batch 4 (seed 0, 8000 steps, lr 0.113): WDQK 0.03 3.893 (peak 5087), 0.1 3.879 (1483), 0.3 3.874 (170); none 3.912, QK-Clip any tau 3.858-3.861, QK-norm 3.842. At 0.014 WDQK 0.03 3.824 (free). Prediction wrong: scaling wd by 1/steps is too weak; in long runs q/k decay recovers ~half the gap vs ~3/4 for QK-Clip (1 seed). Checking seeds 1-2.
Result seeds 0-2, 8000 steps: lr 0.113 WDQK 0.3 3.874/3.874/3.874 (mean 3.874) vs QK-Clip tau 10 3.858/3.863/3.888 (3.870); cost at 0.014 WDQK 0.3 +0.009/+0.007/+0.013 (+0.010). So in long runs q/k decay equals QK-Clip at 8x but costs 0.01 at the best lr; QK-Clip is free there.
Batch 5: lab default wd 0.1 on everything, 8000 steps, lr 0.113, seeds 0-2, with and without tau 10.
Prediction: WD 0.1 alone ~3.87 (like clip); tau 10 on top adds <0.01.
Result batch 5 (8000 steps, 0.113): WD 0.1 all 3.911/3.912/3.918 (3.914, peaks 300-835), + tau 10 3.895/3.915/3.901 (3.904); no wd: none 3.912 / 4.293 (seed 1, peak 153k), tau 10 3.870, WDQK 0.3 3.874. WD 0.1 on everything stops the catastrophic seed but is 0.04 worse than q/k-only decay or the clip at this lr (decaying all weights at 8x lr costs). Dropped as a separate point.

## Batch 6: WDQK 0.1 at 8000 steps (is there a value free at the best lr that still fixes 8x?)
Prediction: cost at 0.014 ~+0.003 (3 seeds); at 0.113 ~3.88 (between 0.3's 3.874 and none).
Result (8000 steps): WDQK 0.1 cost at 0.014: -0.017/0.000/+0.005 (mean -0.004, free); at 0.113: 3.879 (s0)/3.878/3.900 (3.886; peaks 1400-2200) vs QK-Clip tau 10 3.870, WDQK 0.3 3.874. So 0.1 is free in long runs and gets most of the clip's benefit; 0.3 matches the clip but costs 0.01.
## Batch 7: lower learning rate on W_q, W_k only (LRQK multiplier; Anson & Aitchison 2511.21377 ablation form), 800 steps, 3 seeds
Prediction: LRQK 0.25 at 0.113 ~4.15 (worse than WDQK 0.3's 4.116, since scores can still grow slowly without a pull-back) and costs +0.01 at 0.014.
Result batch 7 (3 seeds unless noted): LRQK 0.5: 4.130 at 0.113 (peaks 576-1050), cost +0.002; LRQK 0.25: 4.105 (peaks ~200), cost +0.008; LRQK 0.125 (s0): 4.084, cost +0.023. Prediction wrong: lowering the q/k lr 4x works as well as WDQK 0.3 (4.116, cost 0.000) in short runs. q/k decay is not better than the published fix, just another one-line option.
