# Superposition with Muon — notes

Lab-use sentence: a lab researcher who reads this would change how they read and set up Muon runs for
rare features (which weights or settings decide how many rare features a layer keeps), and the dictionary
size they expect when they interpret a Muon-trained model.

Literature (2026-10-24, 5 extended searches): no paper on how Muon changes superposition / feature packing.
Closest: Wang et al. 2509.26030 (ICLR 2026) — Muon learns rare facts in a linear associative memory evenly
because every singular direction gets the same step; Muon weights have a flatter spectrum.
Muse 2607.14536 (representation geometry of Muon) — about the update geometry, not features.
TMS follow-ups (2310.06301 phase transitions, 2512.13568 lossy compression, 2608.22155 robustness,
2609.06862 survey) — none vary the optimizer.

## Toy 1: TMS autoencoder (tms.py), n=20 features, m=5 dims, importance 0.9^i, 2000 steps, ~1 s/run
OPT ns = Muon, norm = same momentum + step size without Newton-Schulz, sgd = plain momentum.

Prediction before lr sweep (S=0.9): ns stores more features than norm (it lifts weak singular directions,
which carry the less important features); the gap closes at S=0 (all store ~5, PCA).

S=0.9 lr sweep: all three converge to the same solution (12-13 features, loss 0.078-0.080); Muon changes nothing at convergence here. Prediction wrong.
Sparsity sweep prediction (paper): 5 features at S=0, rising with S, nearly all 20 at S=0.99.

Sparsity sweep (Muon lr 0.04, 3000 steps, 3 seeds): reproduces the paper. Features stored 5 (S=0) -> 8 (0.5) -> 10 (0.8) -> 13 (0.9) -> 16 (0.97) -> 19 (0.995).
Feature dimensionality sticks at 1, 1/2 (antipodal pairs), 1/3 (triangles), 2/5 (pentagons), ~2/7 — the paper's sticky points.
S=0 seed 2 ended at loss 0.58 vs 0.42 (stuck solution).
Next: same comparison over 5 seeds for ns / norm / sgd. Prediction: ns gets stuck less (all directions move equally).

S=0, 5 seeds: norm reaches the PCA optimum 0.390 in 4/5 seeds; Muon only 1/5 (0.42-0.58), same 5 features kept (seed 2 keeps feature 5 instead of 3). Small, not a lab result.
LLM (llm.py, seed 0): MoE expert neuron serves 5.0 / 4.4 skills (L0/L1) vs dense 7.5 / 6.8. Word embeddings: no frequency dependence of dimensionality (TMS predicts dense words get own dims; all words are >87% sparse so none do).
Next: short-budget TMS, n=100 m=10 S=0.99 imp 0.98: prediction Muon stores more features than norm/sgd at 300 steps (flat spectrum = tight frame).

Short-budget TMS (n=100, m=10, S=0.99, 300 steps): ns, norm, sgd all reach 60-67 features at loss 0.052-0.053 at their best lr. Muon does not change packing in the TMS autoencoder. Dropped.
MoE less polysemantic than dense is already published (Chaudhari et al. 2510.23671; Expert Strikes Back 2604.02178). Our 5.0 vs 7.5 skills/neuron just confirms it.

## Toy 2: assoc.py — Wang et al.'s associative memory with more facts than dims
Wang's theory needs orthonormal keys, K <= d (they use K = d = 999). Question: does Muon's even tail learning survive K > d?
Prediction: at K <= d, Muon tail accuracy >> norm (NS off); at K = 4-8 d the gap shrinks to ~0 because Muon can only even out d directions, not K facts.

assoc.py d=32, 200 steps, tuned lr: ns and norm equal at every K (K=32,128: both 1.0; K=512: 0.09 vs 0.09; K=2048: 0.017 vs 0.017). Muon's tail edge exists only at untuned low lr. Wang's claim does not show here vs NS-off. Dropped.

## Toy 3: real LM, MSIGN=fro (NS off, same step size) vs ns. Prediction: fro best lr differs, loses 0.05-0.1 wloss at its best; its MLP weights have lower effective rank; skills/neuron about the same.

fro lr sweep seed 0: 0.0035 2.942, 0.007 2.845, 0.014 2.986, 0.03 3.21 -> best 0.007, still +0.18 vs ns (2.668).
'alive' (relative to the biggest neuron) falls 206 -> 37 at fro 0.007 while participation ratio stays ~300: NS off grows a few outlier neurons, not fewer used neurons.
Next: NS_STEPS 1,2,3,5 + fro, metric L*_top = biggest neuron / median neuron. Prediction: top ~10 at 5 steps, ~100 for fro, 1 step in between.

NS_STEPS (seed 0, best lr each): 1 step 2.839, 2 steps 2.699, 3 steps 2.665, 5 steps 2.668, NS off 2.845. biggest/median neuron 10-37, noisy, no clear outlier trend. Not superposition; dropped.

## Toy 1b: n=2, m=1 phase diagram (paper fig. 'phase change'), Muon. feature 2 importance r, density 1-S.
Prediction: like the paper — feature 2 dropped at high density or low r, antipodal superposition (W = [w, -w]) when sparse and r near 1, feature 2 alone wins when r > 1.

Phase diagram (phase.json, 3 seeds, 1500 steps): reproduces the paper. Dense (density 1): only the more important feature. Sparse: antipodal pair.
Boundary: r=0.1 needs density <= 0.1, r=0.3-0.5 <= 0.3, r=0.7-1.5 <= 0.5. r>1: feature 2 alone when dense. Seed 1 stuck at feature 1 only for r=0.7-1.0, density 0.03-0.5 (paper: GD struggles to switch geometry).
Facts in this toy live in attention + embeddings, not MLP (fact-editing notes) -> MLP fact superposition is the wrong place to look.

## Toy 1c: Muon weight decay caps the spectral norm at ~1/wd (Chen et al. 2025) -> a 5-dim layer can hold at most 5/wd^2 unit features.
Prediction (S=0.97, 16 features at wd 0): feats ~ min(16, 5/wd^2): wd 0.5 -> 16, 0.7 -> ~10, 1 -> 5, 1.5 -> ~2.

Result 1c (seed 0): wd 0/0.3/0.5/0.7/1/1.5/2 -> feats 16/16/14/10/7/4/3, top singular value 2.00/1.97/1.74/1.29/1.07/0.75/0.55 (cap 1/wd: -/3.3/2/1.43/1/0.67/0.5),
||W||_F^2 19.6/19.0/14.1/7.6/4.3/2.4/1.2 vs 5/wd^2 -/56/20/10.2/5/2.2/1.25. Cap holds; features fall once 5/wd^2 < what the data wants.
Next: LM, WD on Muon matrices 0/0.1/1/3/10. Prediction: largest singular value ~ min(natural, 1/wd); loss starts rising when 1/wd < natural; rare-answer facts (factr) drop first.

LM wd (seed 0): wd 0/0.1/1/3/10 -> wloss 2.668/2.681/2.963/3.481/4.395; top singular values ~1.9/1.6/0.7/0.3/0.1 (cap 1/wd holds at 3, 10).
First skills lost: copy 0.29->0.21->0.12, back2 0.70->0.56->0.23 (attention skills), then factr 0.89->0.80->0.46. Confounded by scale vs the undecayed embedding (EMB_MULT 8); not a clean superposition test.
Next: S=0 stuck-solution rate, 10 seeds, ns lr 0.02/0.04 vs norm 0.04 (optimum loss 0.390). Prediction: ns stuck more often (5-seed hint: 4/5 vs 1/5).

S=0 seeds 10-19: optimum reached ns lr0.02 8/10, ns lr0.04 7/10, norm 10/10. Over 15 seeds ns 0.04: 8/15, norm 14/15. Muon ends off the dense optimum a bit more often (loss 0.42-0.58). Tiny toy effect, noted only.

## Toy 4: absval.py (paper's computation toy y=|x|, n=100, m=40 ReLU neurons, imp 0.97^i).
Prediction: Muon neurons serve more features each (more polysemantic) than NS-off at the same sparsity, since Muon spreads its step over all directions.
absval.py: one run 11.9 s at n=100 m=40 3000 steps (over the 5 s limit); seed 0 Muon stored 45/100, 2.74 features per neuron, 5% monosemantic. Dropped, not pursued.

Prediction (wd seeds 1,2): same features stored as seed 0 within +-1 at every wd; top singular value just under 1/wd.
Result: seeds 1,2 identical to seed 0 (feats 14/10/7/4/3 at wd 0.5/0.7/1/1.5/2). Prediction right. Note: 7 other heavy python procs were running from other sessions during these runs (some runs 9-16 s).
