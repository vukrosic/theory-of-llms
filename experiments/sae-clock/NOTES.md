# SAEs on the clock adder (2026-10-24)

Lab sentence: a lab interpretability researcher who reads this would change how they read and compare SAE features
(e.g. group latents into circles before calling one latent "a feature", compare SAE seeds at the group level).

Ground truth: the grokked Muon adder ((a+b) mod 47, models from ../../../muon/fourier-muon/runs/muon_{0,1,2}_0.03_1.0.pt).
Known: 5 clocks per seed (seed 0: 13, 3, 22, 20, 11), each built by 2-4 MLP neurons, output = cos(2 pi k (a+b-c)/P) waves.

Literature: Engels et al. 2024 (circular day/month features found by clustering SAE latents in GPT-2), Paulo & Belrose 2025
(SAE seeds share ~30% of latents; TopK less stable than L1), 2509.02565 (feature manifolds eat many latents), Chanin 2605.18229
(synthetic ground truth). No SAE study on a grokked adder with a known clock found.

## Runs
Prediction 1 (res, 64 latents, k=4, model seed 0): unexplained var < 0.1; most alive latents are one function of a, b or a+b
with one clock; the clocks match 13, 3, 22, 20, 11; each a+b clock is split into 3-6 latents at different phases.
Run 1: unexplained 0.17, SAE output keeps accuracy 0.99; all 64 latents are functions of a+b, but 46 fire on ONE answer only
(latent 2 = "answer is 29", fires on 98% of its pairs, ~0 elsewhere). No latent is a clock (purity ~0.07 = flat spectrum = a spike).
Prediction 2 (sweep N = 16, 32, 47, 64, 128 at k=4): one-answer latents grow until N ~ 47, then stay ~47 (+ duplicates);
at N=16 latents fire on 2-4 answers each.
Run 2 (sweep): one-answer latents N=16: 0, 32: 3, 47: 20, 64: 46, 128: 62 (all 47 answers covered). At N=16 each latent fires on
4-15 answers, bunched on one known clock (14 of 16 latents' best clock is 13/11/22/20/3); e.g. latent 3 = answers 14, 16, 44, 42
= positions 26, 23, 28, 31 on the speed-22 clock. No latent is a whole clock at any size.
Prediction 3 (MLP site, 64 latents, k=4): latents are mixes of a and b; < 30% have >50% of their map on one clock.
Run 3 (MLP site, 64 latents, k=4): unexplained var 0.15 looks good, but the model with the SAE's output in place of its MLP
neurons gets 0.66 (model 0.99). parts.py: MLP activations = 75% "a alone + b alone", 9% "the answer (a+b)", 16% rest.
The SAE rebuilds 99% of the first, 65% of the answer part, 32% of the rest -> it keeps the inputs, drops the computation.
Residual SAE (64, k=4) rebuilds the answer part 100% (it is 71% of the residual), acc 0.99.
Prediction 4 (MLP, N = 128, 256): answer part rebuilt rises to ~0.8-0.9, accuracy stays < 0.9.
Run 4: MLP N=128: acc 0.78, answer part 0.76; N=256: acc 0.90, answer part 0.87; "a alone + b alone" 0.99-1.00 at every size.
Prediction 5 (MLP, N=64, k = 1, 2, 8, 16, 32): k=1-2 keep a/b alone, answer part < 0.5; higher k rebuilds the answer part better
(k=32 > 0.9) and accuracy recovers.
Run 5 (MLP, N=64, k sweep): k=1 acc 0.29 / answer part 0.43; k=2 0.41 / 0.50; k=4 0.66 / 0.65; k=8 0.96 / 0.90; k=16 0.99 / 0.98.
Baseline (neurons.py): 24 of 128 raw MLP neurons have >50% of their map on one clock (median 0.11); MLP SAE k=4: 1 of 63
(median 0.07); k=16: 22 of 64 (0.25). Residual SAE latents: 0 at any k (bumps, not waves).
Run 6 (geometry.py): one-answer latents' decoder vectors in answer order -> clocks 13, 3, 20, 22, 11 = the model's own top 5
(N=128: 47/47 answers, top-5 share 0.76 vs model 0.80; N=64: 42/47 answers, 0.68).
Prediction 6 (3 model seeds x 3 SAE seeds, res N=64 k=4): every run has 35-50 one-answer latents and 0 clock latents;
geometry recovers 5/5 of each model's own top clocks; one-answer latents match across SAE seeds (cos > 0.9) >= 90%,
multi-answer latents < 50%.
Run 7: 9 of 9 runs: 0 clock latents, 23-49 one-answer latents. geometry.py recovers each model's own top-5 clocks 5/5 in 8 runs,
4/5 in 1 (model 2 SAE 0, only 23 answers covered). Model 1: [17, 2, 9, 1, 3] vs model [17, 3, 2, 9, 1]; model 2: [16, 5, 23, 1, 4].
Run 8 (stability.py, res N=64 k=4, 3 SAE seeds): one-answer latents found again at cos > 0.9 in 34-57%; multi-answer 6-14%.
The same answer has its own latent in the other seed 61-93% of the time, cos median 0.82-0.95. Moderate; prediction (>= 90%) wrong.
Not a clean result, dropped.
Prediction 9 (ablate.py, res N=64 k=4): removing the "answer = s" latent from the residual breaks answer s on most of its
pairs (> 80% wrong); the wrong answers it gives instead are spread, not one neighbour; other answers unchanged.
Run 9 (ablate.py): matches. One-answer latents: removing one breaks its answer on 0.82 of its pairs (median 0.98), other answers 0.0000.
Latent 2 "answer 29": breaks 100%. So these latents pass the usual tests (one meaning, causal) and none is a clock.
Literature: Anthropic's 2025 MOLT update says transcoders split addition into lookup-table features ("6 + 9 = 5 mod 10") and
suspects that misses geometric structure; no known-mechanism test there.
Prediction 10 (transcoder, N=64, k=4): spliced acc < 0.8; latents mostly answer-like (function of a+b) lookups on 1-3 answers.
Run 10 (transcoder 64, k=4, model 0): unexplained 0.18, spliced acc 0.72 (model 0.99); rebuilt: a/b alone 0.98, answer 0.71,
rest 0.00. Latents: 0 one-answer, 0 pure a+b, 5 of 60 on one clock. Same pattern as the MLP SAE.
Run 11 (ablate, all 9 res SAEs): own answer breaks on 0.66-0.85 of its pairs (median 0.81-0.98), other answers <= 0.0008.
Prediction 12 (transcoder 3 x 3, k=4): every run acc < 0.8, a/b alone kept > 0.95, answer part < 0.8.
Run 12 (transcoder 3 x 3, k=4): direction holds 9/9 (a/b alone 0.96-0.98 > answer 0.67-0.89), acc loss varies by model:
model 0: 0.71-0.72 (model 0.99), model 1: 0.93 (0.997), model 2: 0.76-0.78 (0.93). Prediction partly wrong (model 1 acc 0.93).
Prediction 13 (transcoder k = 1, 2, 4, 8, 16, 3 models, seed 0): answer part kept rises with k, reaches > 0.95 by k = 16;
spliced acc within 0.02 of the model at k = 16.
Run 13 (transcoder k sweep, 3 models): answer part kept k=1: 0.38-0.40, k=2: 0.44-0.67, k=4: 0.71-0.89, k=8: 0.94-0.97, k=16: 0.98-0.99;
"each number alone" k=1: 0.69-0.71, k=2: 0.89-0.91, k=4+: 0.96-1.00. Spliced acc at k=16 within 0.001-0.009 of the model. Matches.

## Results -> pages
- findings/sae-sees-answers-not-clocks.md (runs 1, 2, 6, 7, 9, 11)
- findings/transcoders-drop-the-sum.md (runs 3, 4, 5, 10, 12, 13)

## Reproduce
Models: `cd ../clock-addition && P=47 FRAC=0.6 EMB_WD=1 python3 mod.py SEED 300 0.03 1.0` for SEED 0, 1, 2 (runs/muon_SEED_0.03_1.0.pt).
Then `P=47 python3 sae.py runs/muon_0_0.03_1.0.pt res 64 4 0`, `ablate.py`, `geometry.py`, `stability.py`, `transcoder.py MODEL 64 K 0`,
`fig.py RUNS_DIR`. sae_runs/ holds the trained SAEs (not committed).

## Scaling check (2026-10-24, new rule in muon/AGENTS.md)
Sizes: width D = 32 / 64 / 128 with FF = 2D (hidden matrices 16x from smallest to largest); lengths 300 / 1200 / 4800 steps at D=64.
Prediction S1: all grok at lr 0.03 (test > 0.9) by 300 steps except maybe D=32; both effects keep their size at every width and length.
Training result: D=32 needed 500 steps (300 left seed 2 at 0.77), D=128 needed 600 steps at lr 0.03 (lr 0.05 / 300 steps left seed 1 at 0.89); all 15 models end at test 0.98-1.00.
Prediction S2 (before scale.py): one-answer latents stay ~30-49 of 64 and clock latents 0 at every width and length; geometry finds >=4/5 clocks;
own-answer break >0.6, other <0.01; transcoder k=4 rebuilds the sum part <0.9 while the alone part >0.95. Longer training may sharpen clocks, so the sum part may be easier (risk for B).
Run S2a (raw activations): fine at 300 steps, but 2 of 3 models at 4800 steps gave a dead SAE (23-27 of 64 alive, 0 one-answer latents, transcoder rebuilds 7-13%).
Cause: weight decay shrinks the residual ~50x over long training (mean norm 0.6 -> 0.01) and the SAE's fixed Adam lr can't follow. Fix in scale.py: scale activations to unit mean norm (Anthropic does this too); rerun all 15.
Run S2b (normalised, 15 models): A holds everywhere; B's gap grows with width (D=32 0.08, D=64 0.19, D=128 0.58) but every run used 64 transcoder latents.
Prediction S3: with latents scaled to width (32/64/128 for D=32/64/128) the D=128 gap shrinks but stays > the D=32 gap; the sum part needs more directions in wider models (rank90 grows with D).
Run S3: prediction right. Latents scaled to width, k=4: gap (alone - sum rebuilt) 0.09 / 0.19 / 0.39 at D=32/64/128; sum part rank90 7-11 / 11-12 / 24; accuracy lost 0.02-0.16 / 0.07-0.29 / 0.18-0.26. k=8 at D=128: gap 0.15, lost 0.01-0.04.
Prediction S4: a skip transcoder (2501.18823, linear path in parallel) frees latents from the each-alone part, so at D=128 k=4 the sum part rises from ~0.6 to >0.8 and lost falls below 0.1.
Run S4: prediction right. Skip transcoder at k=4, latents = width: sum part 0.92-0.98 / 0.93-0.97 / 0.79-0.85 at D=32/64/128 (plain 0.83-0.91 / 0.66-0.87 / 0.57-0.65);
accuracy lost 0.005-0.05 / 0.007-0.03 / 0.03-0.05 (plain 0.02-0.16 / 0.07-0.29 / 0.18-0.26). D=64 lengths with skip: lost 0.03-0.04 at 1200, 0.045-0.063 at 4800 (plain 0.07-0.10 at both).

### Per-size and per-length table (3 models each, SAE/transcoder seed 0, activations normalised; scale_runs.jsonl)
| model | params ratio | one-answer latents (of 64) | clock latents | clocks from decoder layout | own answer / other breaks | plain tc: alone - sum gap | plain tc lost | skip tc lost |
|---|---|---|---|---|---|---|---|---|
| D=32, 500 steps | 1x | 23, 31, 44 | 0, 0, 1 | 4, 5, 4 /5 | 0.50-0.86 / 0.0000 | 0.07-0.10 | 0.02-0.16 | 0.005-0.05 |
| D=64, 300 steps | 4x | 49, 27, 22 | 0, 0, 0 | 5, 5, 5 /5 | 0.68-0.80 / <=0.0005 | 0.10-0.33 | 0.07-0.29 | 0.007-0.03 |
| D=128, 600 steps | 16x | 33, 35, 54 | 0, 0, 0 | 4, 4, 4 /5 | 0.66-0.76 / <=0.0001 | 0.33-0.43 | 0.18-0.26 | 0.03-0.05 |
| D=64, 1200 steps | 4x length | 33, 42, 37 | 0, 0, 0 | 4, 3, 5 /5 | 0.66-0.68 / 0.0000 | 0.13-0.18 | 0.07-0.10 | 0.03-0.04 |
| D=64, 4800 steps | 16x length | 41, 43, 43 | 0, 0, 0 | 5, 4, 4 /5 | 0.66-0.87 / 0.0000 | 0.08-0.20 | 0.07-0.10 | 0.045-0.063 |
(D=128 transcoder: 128 latents; D=32: 32 latents. Hidden-matrix params 16x from D=32 to D=128.)

Check 1 (size): A holds at all 3 widths. B grows with width: gap 0.09 -> 0.19 -> 0.39 mean, accuracy lost 0.09 -> 0.18 -> 0.21 mean; the skip fix holds at all widths.
Check 2 (length): A holds 1x/4x/16x. B's gap stays ~0.15 from 1200 to 4800 steps (0.19 at 300); accuracy lost drops 0.18 -> 0.09 -> 0.08 from 300 to 1200 then holds.
Check 3 (cause, not toy-only): A: latents are nonnegative and sparse, so a circular feature is tiled by one latent per point (Gurnee et al. 2026 see the same tiling of a helix in Claude 3.5 Haiku). B: the computed part (a+b) spans more directions as the model widens (rank90 7-11 -> 11-12 -> 24) while k stays fixed, and the plain transcoder spends latents on the easy linear pass-through; a skip path takes the linear part and frees the latents.
Big-model evidence: Anthropic's CLTs: 11.5% reconstruction error at L0 88 on an 18-layer model, 21.7% at L0 235 on Haiku (error grows with size even at higher L0); skip transcoders (2501.18823) lower reconstruction loss on Pythia/Llama; Biology 2025 reads Haiku's addition as lookup-table features; Engels 2024 finds GPT-2/Mistral day/month circles only after clustering SAE latents.
1B+ guess (a guess, untested): A gets stronger (more latents per manifold, each a single point); B: a plain transcoder at fixed L0 loses a larger share of the MLP's computed part as width grows, and a skip connection recovers most of it.
