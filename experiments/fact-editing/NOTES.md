# Fact storage and editing (Geva 2021, ROME 2022, Hase 2023) on the 8-skill toy

Lab sentence: a lab researcher who reads this would pick where to write new facts (which layer / token / which check before an edit) from a measured rule instead of from causal tracing.

Toy: default MoE toy (MOE=8 MSIGN=ns EMB_MULT=8, 260 steps, lr 0.014), 128 facts = 64 subjects x 2 relations (factc, factr).
Prompt [relation, subject, SEP] -> object. Code: facts.py.

## Predictions (written before running)
- Tracing (seed 0-2): like ROME, two hot spots: MLP layer 0 at the subject (pos 1) ~0.5 and attention layer 1 at SEP (pos 2) ~0.6.
- Edits work ~100% at every MLP spot (rank-one with exact key).
- Edit at the subject spot changes the sibling fact (same subject, other relation) in >30% of edits; edit at SEP layer 1 changes fewer siblings but more facts of the same relation.

## Results
- seed 0 default toy: tracing MLP effect 0.01-0.06 everywhere; attention L0 at SEP 0.50, L1 at SEP 0.44. Prediction wrong: no MLP hot spot.
- Knockout (mean-ablate one spot): any single MLP spot leaves 98-100% facts right; attention L0 SEP -> 36%, L1 SEP -> 52%. Whole MLP (all spots, both layers) -> 86%/87% (from 98/95).
- Direct answer score shares (seed 0 default): emb(SEP) 0.16, attn0 0.31, mlp0 0.08, attn1 0.38, mlp1 0.08.
- Edits (32 facts, seed 0 default): MLP rank-one edit works 97-100% at L0 subj, L0 SEP, L1 SEP. Others changed: L0 subject 5.6/127, L0 SEP 2.8, L1 SEP 0.4.
- More facts (512 subj x2, PACK 4): still MLP knockout ~1.00 (model only learns 44%/9%). 8 relations x 64: MLP knockout L1 0.90. EMB_MULT 1: MLP knockout L1 0.86. MLP never becomes the main store in 260 steps.
- ROME-like config C1 (2-token subjects from 16x16 pools, [subject, relation] -> object, 128 subjects x 2 relations, PACK 3), seed 0 edits:
  L0 first name: 15.3/255 others changed, same subject other relation 0.47; L0 last subject token (ROME spot): 8.1, sibling 0.62; L0 relation token: 8.0, sibling 0.00; L1 relation token: 3.8, sibling 0.03.

Prediction for lr tune of C1 (lr 0.007/0.014/0.028): 0.014 best (same as default toy).
- lr tune C1 seed 0: 0.007 0.94/0.73, 0.014 0.96/0.80, 0.028 0.98/0.82, 0.056 0.96/0.80; seed 1 0.014 0.98/0.79. Prediction roughly right (0.014-0.028 flat).

## Muon-shaped edits (prediction before running)
One Muon step on one fact at one token has gradient d k^T (rank one); msign makes it (d/|d|)(k/|k|)^T. That is ROME without the C^-1 whitening of the key.
- EDIT=raw (rank one, key not whitened) at the same spot changes >=2x more other facts than EDIT=rome.
- EDIT=ft (Muon steps on that one prompt): all matrices changes the most other facts; mlp1 only and attn1 only fewer; ft on all matrices changes >= 3x more than ROME at L1 relation token.
- C1 lr 0.028 chosen (seed 1: 1.00/0.83 vs 0.98/0.79 at 0.014). Seed 2 at 0.028: 0.98/0.79.
- C1 seed 0 (pos1,2 = subject tokens, pos3 = relation = answer spot), 32 edits, others changed of 255 / same-subject sibling changed:
  rome: L0 s1 24.8/0.50, L0 s2 (ROME spot) 11.7/0.53, L0 rel 3.0/0.03, L1 rel 1.9/0.06
  raw : L0 s1 35.5/0.50, L0 s2 14.8/0.53, L0 rel 7.1/0.00, L1 rel 6.5/0.16 -> raw 1.3-3.4x worse; prediction right at the relation spot, not at subject.
  ft Muon lr 0.005 (works / changed): all 1.00/16.7, mlp1 0.22/0.9 (lr too small), attn1 0.47/11.8, mlp 0.97/9.3, attn 1.00/23.0
Prediction for ft lr sweep: smallest lr that still reaches ~100% gives the least damage; fro (no orthogonalization) changes fewer facts than ns at equal success.
- ft lr sweep seed 0: Muon all-matrix ft breaks 14.3 (lr .002, works .97) / 16.7 (.005) / 18.4 (.02); mlp1-only ft at .02 works 1.00, breaks 3.3. fro (no orthogonalization) all: 18.2 (.005), 41.2 (.02). Orthogonalization is not the cause; touching all matrices is. Dropped as a Muon finding.
- C1 3 seeds (lr 0.028). Tracing MLP at subject tokens 0.00-0.01 (all seeds); MLP knockout there 0.99-1.00; attn L0 at relation token tracing 0.56/-/-, knockout 0.29/0.15/0.26.
  rome L0 last subject token: works .91/.81/.97, changed 11.7/10.7/19.4, sibling .53/.53/.75; L1 relation: works 1/1/1, changed 1.9/2.0/2.7, sibling .06/.06/.09
  raw  L0 last subject token: works .91/.72/.97, changed 14.8/16.2/28.7; L1 relation: changed 6.5/3.7/4.8 (rome x2.3 on average), other skills loss .0028/.0011/.0032 vs rome .0005/.0005/.0002
- Literature: relation-token editing = RETS (Liu et al. AAAI 2025, 2408.15091); C=I loosens neighbours only in unreviewed notes. Pages: mlp-edit-without-storage, edit-the-last-token.
