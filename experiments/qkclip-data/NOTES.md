# QK-Clip vs data (Muon, no QK-norm)
Lab-change sentence: A lab training with Muon without QK-norm would know which data mixtures (heavy copying/duplication, low-entropy, long sequences) need QK-Clip and at what threshold.
Code: attn.py (copy of muon-attn-fixes/attn.py + TASKS knob, groups chains/algo; rows in runs.jsonl). All runs one seed (0) unless marked.

## Batch 1: data family x lr, QKNORM=0 (peak max score | loss); one seed
Prediction: copy-heavy (c1.0) and duplicate-heavy (d0.9) blow up more than the full mix at the same lr (induction/lookup heads need sharp attention); low entropy (beta8) and noise (p0.5) do not.
Result (b1a.txt noQKN, b1b.txt QKN). peak score / loss, lr .014 | .04 | .057:
| mix | noQKN peak | noQKN loss | QKN loss (QKN peak) |
|---|---|---|---|
| full 20 | 9.8 / 48 / 653 | 4.002/4.051/4.097 | 4.017/4.051/4.073 (3.6/3.2/3.3) |
| c1.0 only | 349 / 219 / 409 | 3.251/3.574/3.648 | 3.052/3.399/3.119 (69/55/79) |
| d0.9 only | 19 / 15 / 21 | 1.061/1.071/1.094 | 1.080/1.059/1.114 |
| beta8 only | 2.5 / 3.6 / 5.0 | .652/.655/.651 | same |
| p0.5 only | 6.7 / 234 / 457 | 5.149/5.149/5.156 | 5.150/5.156/5.157 |
| algo only (4 tasks) | 115 / 534 / 1920 | 3.456/3.762/3.824 | 3.437/3.597/3.651 |
| chains only (16) | 6.1 / 23 / 304 | 3.933/3.947/3.960 | 3.936/3.945/3.937 |
Conclusions (1 seed): prediction half wrong. d0.9 does NOT blow up; copy (c1.0) and algorithmic tasks blow up already at lr .014 (349, 115) and lose 0.02-0.2 loss vs QK-norm; p0.5 noise blows up at .04+ (234) with ZERO loss cost (no signal to hurt). High score != harm. Full mix at best lr is safe (9.8) because copy/algo are only 5/20 of data.

## Batch 2: each algorithmic task alone, lr .014/.04, QKN 0 vs 1
Prediction: sort/reverse (far pointer attention) blow up most; modadd/count (local) stay < 30.
Result b2 (peak / loss; noQKN .014|.04; QKN loss .014|.04): sort 40/464, loss 2.72/3.55 vs 2.82/3.03; reverse 99/776, 5.30/5.32 vs 5.30/5.30; modadd 90/658, 4.577/4.577 vs same (unlearnable, ln97, no loss cost); count 90/957, loss .0023/.0229 vs .0006/.0008. Prediction wrong: ALL 4 algorithmic tasks blow up alone (even local count/modadd). Loss harm only where task is learned (count, sort).
Hypothesis: blow-up depends on homogeneous data (narrow mix), not specific task; diluting with diverse chains lowers peak.

## Batch 3: dilution. count + first n chains (CHAIN order), lr .04, QKN 0/1 (+ .014 noQKN)
Prediction: count share 50% -> peak ~300, 25% -> ~100, 12.5% -> <50, 6% -> <20.
Result b3 (count + first n chains; noQKN peak at .014|.04; loss at .04 noQKN vs QKN(peak)): 
| n chains (count share) | peak .014 | peak .04 | loss .04 | QKN loss .04 (peak) |
|---|---|---|---|---|
| 1 (50%) | 17 | 188 | 1.833 | 1.831 (9.6) |
| 3 (25%) | 12.7 | 18 | 2.658 | 2.660 (7.0) |
| 7 (12.5%) | 12.9 | 20 | 3.322 | 3.330 (4.7) |
| 15 (6%) | 8.7 | 30 | 3.718 | 3.721 (4.2) |
Conclusion (1 seed): prediction ~right (50% -> 188; <=25% -> ~18-30). Dilution by diverse data kills the blow-up; no loss cost at any share at .04.

## Batch 4: is it narrowness in general? chains only, n = 1, 2, 4 distinct chains (no count), noQKN lr .04/.057, QKN lr .057
Prediction: 1 chain alone blows up (>100 at .057), 4 chains ~50, 16 chains 304 already seen (so NOT monotone in diversity: chains-only 16 @.057 = 304).
Result b4 (chains only, noQKN peak .04|.057; loss; QKN .057 peak): n=1: 6/46, loss 3.553; n=2: 11/58; n=4: 43/320 (QKN 3.7); 16 chains: 23/304. Conclusion: NOT narrowness. Chain-only data blows up only at lr >= .05 regardless of n; QKN loss equal to noQKN loss within .004. Prediction (1 chain blows up) wrong; dropped narrowness idea.
Rule so far: attention-heavy data (copy, sort, reverse, count, modadd) blows up at lr .014 already (40-350); diluting it with chains to <=25% share removes it at .04.

## Batch 5: sequence length. mix = chains + sort,reverse,modadd (19 types; count excluded since its labels exceed V=256 at T=128), noQKN/QKN, lr .014/.04, T=16/32/64/128, STEPS=800
Prediction: peak score grows with T (more positions to discriminate): .04 peak 20 (T16) -> 50 (T32) -> 200 (T128).
Result b5 (19-type mix, noQKN peak .014|.04 ; loss; QKN loss): T16 12.5|74 (loss 4.291|4.328 vs QKN 4.292|4.302); T32 9.9|30 (4.205|4.201 vs 4.207|4.201); T64 11.9|14.7 (4.105|4.100 vs 4.109|4.099); T128 .04 only: 33 (4.017 vs 4.020; 18 s/run). Prediction wrong (opposite): SHORTER sequences give larger peaks; non-monotone at T128; loss gap only at T16 .04 (.026, 1 seed). T is not a driver; flat.

## Batch 6: algo-only (4 tasks, blows up most with c1.0): lr sweep QKN0 vs QKN1 to find best lr, then QK-Clip tau.
Prediction: best lr in QKN ~.014-.02; noQKN loses ~.02 at .014; tau=10 matches QKN at .014 (peak there 115, QKN peak 10), tau=100 does not.
Result b6 (algo-only; noQKN peak|loss; QKN peak|loss): lr .007 39|3.440 ; 17|3.459. lr .014 115|3.456 ; 10|3.437. lr .028 147|3.605 ; 7|3.543. lr .04 534|3.762 ; 9|3.597. Best lr ~.007-.014 where noQKN loss is within .003-.02 of QKN. Gap opens above best lr (+.06 at .028, +.17 at .04).
Batch 7: tau sweep on algo-only at lr .028 and .04 (QKN loss 3.543/3.597). Prediction: tau=10 and 5 recover to ~QKN loss; 30 partly; 100 nothing.
Result b7 (algo-only noQKN, QK-Clip tau; loss at .028 | .04; peak). noclip 3.605|3.762 (147|534); QKN 3.543|3.597. tau100 3.608|3.729 (103|107); tau30 3.614|3.655 (36|35); tau10 3.555|3.603 (12|13); tau5 3.556|3.591 (6.8|7.4). Conclusion (1 seed): tau must be ~ QKN's own peak score on that data (7-10 here); 100 and 30 leave most of the gap.
Batch 8: does the needed tau track the data? c1.0 alone at lr .04 (QKN loss 3.399, QKN peak 55; noQKN 3.574, peak 219): tau 100/50/30/10/5.
Prediction: tau 50-100 recover, tau 10 and 5 HURT (copy needs score ~55), i.e. needed tau tracks the QKN peak, not "lower is better".
Result b8 (c1.0 alone noQKN; loss at lr .014 | .04; peak): noclip 3.251|3.574 (349|219); QKN 3.052|3.399 (69|55); tau100 3.313|3.563; tau50 3.375|3.561; tau30 3.417|3.569; tau10 3.522|3.610; tau5 3.585|3.634. Prediction wrong: on copy data QK-Clip does NOT recover the QKN loss at any tau, and at the best lr (.014) it HURTS monotonically as tau falls (tau5 +0.33). Reason guess: copy needs scores >50; clipping scales q,k weights globally. Single seed -> confirm with seed 1.
Batch 9 (seed 1 confirm): c1.0 @.014 noclip/QKN/tau100/tau30; algo @.04 noclip/QKN/tau10/tau30.
Result b9 (seed 1 confirm). c1.0 @.014: noclip 3.345 (197), QKN 3.165 (58), tau100 3.357, tau30 3.417 -> seed-to-seed noise ~.1 but order holds: tau100 = noclip, lower tau worse, QKN best by .18 (clip cannot recover). algo @.04: noclip 3.796 (843), QKN 3.572 (10.7), tau10 3.624, tau30 3.629 -> clip recovers ~75%.
Batch 10: dilution of copy data: c1.0 + first n chains (n=1,3,7,9; share 50/25/12.5/10%), noQKN peak at .014 and .04 + QKN loss at .04.
Prediction: like count: >100 at 50% share, <30 at <=25%.
Result b10 (c1.0 + n chains; noQKN peak .014|.04; loss .04 noQKN vs QKN): n1 (50%) 7|70, 3.615 vs 3.616; n3 6|16, 3.553 vs 3.554; n7 7|15, 3.780 vs 3.779; n9 9|23, 3.778 vs 3.761. Prediction partly wrong: even 50% copy + 50% chains is safe at .014 (7 vs 349 alone); blow-up needs a mix that is ~100% attention-task data.

## Summary rules (all one seed unless noted)
1. Data that blows up scores at the BEST lr (.007-.014): pure copy (peak 349), pure algorithmic sort/reverse/modadd/count (40-115, count 90), i.e. mixes with no ordinary text-like data. Chain-only data: 6 at .014, blows up only at lr >= .05 (304 at .057).
2. Adding ordinary data fixes it: count 50% + chains -> 17 at .014; copy 50% + chains -> 7. Full 20-task mix 9.8 at .014.
3. High score != harm. Noise (p0.5) hit 457 at .057 with 0 loss cost; modadd (unlearnable) 658 at .04 no loss cost. Loss gap vs QKN appears only when the task needs sharp attention AND lr > best (algo: +.06 at .028, +.17 at .04; copy +.2 at .014; count .023 vs .0008).
4. Seq length T=16..128: not a driver (shorter was higher: 74 at T16 vs 15 at T64 at .04; flat loss).
5. tau: needed tau ~ QKN's own peak on that data. algo (QKN peak 7-10): tau 10/5 recover ~all, tau30 ~55-75%, tau100 ~20% (seed0 .04: 3.762 -> 3.729). Copy (QKN peak 55-70): clip never recovers (tau100 = noclip, tau<=30 worse, tau5 +.33 at .014).
Dropped: narrowness (b4), T (b5), d0.9 duplicates (b1: peak 19, no blow-up).

## Lead check (Opus): copy-only at lower lr, seed 0 (mine.jsonl)
lr 0.0035 / 0.007: no QK-norm 3.605 (peak 22) / 3.526 (58); QK-norm 3.456 (41) / 3.190 (73).
On copy data QK-norm wins by 0.15-0.34 even when no-QK-norm scores are small (22-58, no blow-up). So the copy gap is not caused by large scores, which is why no QK-Clip τ recovers it. QK-norm helps copy learning for a separate reason.
