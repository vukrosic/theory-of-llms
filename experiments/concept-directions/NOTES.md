# Concept directions (linear representation hypothesis + steering) — notes

Lab-use sentence (draft, revise after first results): a lab researcher who reads this would change how they
build and place a steering vector (which direction recipe, which layer, which strength) and whether they trust
a probe direction from one checkpoint or model size on another.

## Literature (2026-10-24, 3 extended searches + 2 papers read)
- Park 2023 (LRH, causal inner product), Marks & Tegmark 2023 (truth is linear; mass-mean probe directions are more causal than logistic-probe ones), Engels 2024 (days/months are circles), Turner 2023 ActAdd, Zou 2023 RepE.
- Tan 2024 (NeurIPS, 2407.12404): steering unreliable per input, many anti-steerable examples; steerability is a property of the dataset more than the model.
- Braun 2025 (2505.22637): higher cosine between training activation differences predicts steering success.
- Billa 2026 (2604.15557, read): logit-lens "linear accessibility" per layer predicts steering success (rho 0.86-0.91); middle layer gives no effect for facts; across Pythia 160M -> 6.9B steerable concepts 5/23 -> 9/23, dP x3.4; no training-checkpoint analysis.
- Cylindrical RH 2026 (2605.01844, read): concept overlap makes steering sample-specific; removing the off-axis component stabilizes output.
- Probe-vs-steer gap: 2604.13068 (hallucination probes detect, steering fails 7/7 models 117M-7B), 2609.22135 (read-best layer != steer-best layer, 26x causal gap), 2609.18533 (rho ~0 between probe acc and steering benefit).
- Open (not found): how linearity / steerability of one concept changes along training length in one model; whether a steering vector from an early checkpoint still works later.

## Run 1 prediction (default MoE toy, width 64, 260 steps, seed 0)
count: linear probe ~1.0 at sites 1-2, class means bent (pc1 ~0.8, turn ~0.5); chord steering ~90% at site 2, line/first worse at +-2; random ~0.
rel: chord from 32 subjects moves held-out subjects to the other relation ~50% at site 1; random ~0.

## First results (width 64, MIX=0 equal skill mix, 520 steps, seed 0)
MIX=0 because count is only 42% at the natural mix at 260 steps; slot (sort) is learned to ~47% only, so it is off by default.
count: linear probe 1.0 at both layers, but the class means bend (pc1 0.83 / 0.65, consecutive-step cosine 0.59 / 0.29).
 steering 0..4 by +-1,+-2: chord (target mean - source mean) 0.83-0.90, one best-line direction 0.16-0.41, reused 0->1 step 0.26-0.40, random 0.03-0.13.
rel (factc <-> factr): probe 1.0, vector from 32 subjects moves the other 32 only 0.29-0.54 at the best strength (random 0.05-0.09).
Width 32 at 260 steps: count barely learned (base 0.31), no chord/line gap; width 128 at 260: chord 0.76 vs line 0.28-0.42.

## Grid prediction (widths 32/64/128 x steps 260/1040/4160 x seeds 0-2, DRIFT=1, MIX=0)
count: chord success rises with width and length, line stays <0.5, so the chord-line gap grows; turn falls (more bent) with length.
rel: stays 0.3-0.6 at every size. drift: a probe from the half-way checkpoint loses more accuracy on the final model the longer the run.

## Grid result (3 seeds each, MIX=0; site 1 = after layer 0 (middle), site 2 = after layer 1 (last))
count, last layer, count learned (1040+ steps): consecutive-step cosine 0.47/0.21/0.29 (w32/64/128, 1040) -> 0.11/-0.09/-0.04 (4160).
 one scaled direction: 0.10/0.02/0.05 at strength 1, 0.30/0.22/0.24 best strength (1040) -> 0.03/0.00/0.05, 0.18/0.18/0.19 (4160); target mean difference 0.99-1.00 everywhere.
count, middle layer: cosine 0.88/0.78/0.82 -> 0.73/0.65/0.58, one direction best 0.56/0.47/0.50 -> 0.44/0.40/0.36.
Holds at all 3 widths (16x params), grows with length. Turn predicts one-direction success: Spearman 0.91 over 36 (run, layer) points; turn >0.8 -> 0.44 at strength 1, turn <0.3 -> 0.03.
rel vector on held-out subjects falls with length (last layer 0.49 -> 0.17 at w32) and width; dropped: Todd 2023 function vectors transfer to held-out inputs in GPT-J, and our 2-layer toy has at most 1 layer after the edit.
drift: half-way probe on the final model gets 0.44 -> 0.65 (w32 -> 128) at the middle layer; shrinks with width, dropped.
Big-model evidence: Heinzerling & Inui 2024 numeric directions + edits work best at relative depth 0.2-0.4 (early-middle); Wu & Papyan 2024 last-layer collapse grows with scale and training.

## Cause check prediction
last-layer count means point at their own output rows (head_align >0.8 at 4160) and the head rows for 0-4 are not on a line (head_turn <0.3).
Cause result: head rows for 0-4 have step cosine -0.42 to -0.45 (simplex corners, -0.5 ideal) - predicted. Last-layer count means vs their own head row: cosine 0.69/0.68 (w32, 1040/4160), 0.67 (w64, 1040) - below the predicted 0.8, so they point toward their rows but not fully onto them.

## Answers to the brief
- Linear (readable by one line, probe 100%): count at both layers, fact relation at both layers.
- Curved: count's class means bend from a near-line in the middle layer (step cosine 0.6-0.9) to separate corners in the last layer (cosine near 0); no circular concept in this toy.
- Steering: target-mean minus source-mean works 99-100%; one scaled direction 18-56% at the best strength, worse at the last layer.
- Scaling: the bend and the one-direction failure hold at widths 32/64/128 and grow from 1040 to 4160 steps. Page: findings/ordered-values-bend.md.
