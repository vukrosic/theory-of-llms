# In-context learning beyond copying — notes

Lab-use sentence: a lab researcher who reads this would change how many distinct tasks (and how much repetition inside a
document) their data mix needs as the model grows, so a bigger model still reads the rule from the prompt instead of
memorizing a fixed set of tasks, and which layer they take a task vector from for steering.

## Literature (2026-10-24, 5 extended searches)
- Todd 2023 (function vectors, 2310.15213): a few mid-layer heads carry the task; mean of their outputs added in a zero-shot
  prompt triggers the task; consistent Llama 2 7B-70B, best at early-middle layers.
- Hendel 2023 (task vectors, 2310.15916): hidden state at the last "->" of a few-shot prompt, patched into a zero-shot prompt.
- Chan 2022 (2205.05055): burstiness + many rare classes -> in-context; Zipf 1 keeps both; Singh 2023 (2311.08360): ICL is
  transient, wider models peak higher and decay slower; depth no consistent effect.
- Raventós 2023 (2306.15063, linear regression): a task-diversity threshold; MORE layers / width RAISE the threshold.
- Yang 2501.09240: task vectors form in small from-scratch transformers, in the middle layers.
- Newest close paper: 2605.03780 (May 2026, task vector geometry, dual modes): Bayesian task retrieval (M1) vs extrapolative
  task learning (M2); diversity pushes M2; hidden states = convex mixtures of task vectors. No model-size sweep, no length sweep.
- Pan 2023 (2305.09731): in LLMs task recognition is flat with size, task learning grows with size.
- Gap: no size x length sweep of the recognition-vs-copy switch on discrete tasks, none with Muon.

## Toy (icl.py)
Pool of T random tables f_t: 32 inputs -> 32 outputs. Sequence = 12 pairs of one task. recog = seen task, new x (needs task
recognition + stored table); copy = never-seen task, repeated x (needs reading the prompt). Chance 1/32 = 0.03.

D=64 L=2 300 steps lr 0.02, T=16: recog 0.85, copy 0.05 (no copying). 4.8 s.

## Diversity threshold vs size (K=8 pairs, BURST=6 distinct x per prompt, 300 steps, lr 0.02, 3 seeds)
recog / copy per task-pool size T:
| D | T=8 | 12 | 16 | 24 | 32 | 128 |
|---|---|---|---|---|---|---|
| 32 | .52/.14 | .26/.20 | .16/.24 | .11/.28 | .08/.37 | .05/.44 |
| 64 | .97/.12 | .59/.23 | .26/.41 | .14/.68 | .11/.90 | .05/.91 |
| 96 | .99/.12 | .78/.21 | .36/.35 | .15/.65 | .10/.82 | .05/.86 |
Crossing (recog = copy) T* ~ 10 (D=32), ~14 (64), ~16 (96): small rise with size at 300 steps. A model that memorized the
tasks (T=8) ignores the prompt on a new task: copy 0.12 even when the answer is in the prompt.
Prediction for length (D=64, 1200/4800 steps): recog rises at T=16/32 and copy fades there (transience); T=128 keeps copying.
MIXF (share of fresh-task sequences) with T=8, 300 steps: 0.03/0.1 give no copying (0.13-0.19), 0.3 gives 0.25-0.67 and costs recognition. Not a cheap recipe; dropped.

## Length (1200 steps = 4x, seed 0)
| D | T=16 | 32 | 64 |
|---|---|---|---|
| 32 | .98/.25 | .16/.97 | .09/1.0 |
| 64 | 1.0/.23 | .25/.99 | .11/1.0 |
| 96 | 1.0/.23 | .42/.96 | .12/1.0 |
Switch stays between T=16 and 32 at every size; at 4x length T=16 flips fully to memorizing (copy .41 -> .23 at D=64).
Bigger models at T=32 do both (recog .16/.25/.42, copy ~.97). The threshold barely moves with size (Raventos: rises).
Prediction for weight decay (T=16, 1200 steps, D=64, Muon decoupled WD on matrices): WD 0.3-1 brings copying back (>0.8)
because it caps how much the weights can memorize, like it forces the clock in the adder.
WD on Muon matrices (T=16, 1200 steps, D=64): 0.1/0.3 no change (copy .24/.27), 1 kills memory (recog .26) but copy only .41, 3 hurts both. Prediction wrong; dropped.
Conflict test (pool task, 3 inputs relabeled in the prompt): follows the prompt exactly as often as copy (e.g. D=64 T=32: ctx .92 mem .01; T=8: ctx .09 mem .62). Nothing beyond copy.

## 16x length (4800 steps, D=64, seed 0) — transience
T=32: recog .96, copy .46 (was .99 at 1200 steps): the copy circuit was built, then faded once the 32 tables were memorized.
T=64: recog .15, copy 1.0. T=24 at 1200: .70/.72.
Crossing T* ~ 14 (300 steps) -> ~24 (1200) -> ~32 (4800): ~1.5x per 4x steps.
Prediction: MIXF=0.1 (10% fresh-rule sequences) at T=32, 4800 steps keeps copy >= .9 with recog ~.95; MIXF=0.02 partly.
MIXF at 16x length (T=32, D=64, 4800 steps, seed 0): 0.02 -> recog .96 copy .98 (no fade; conflict follows prompt .97);
0.1 -> recog .79 copy .99. So 2% fresh-rule sequences stop the fade at no memory cost. 0.03 at 300 steps did NOT create copying
(T=8), so this keeps a circuit alive rather than building it — check MIXF=0.02 at T=8, 4800 steps.
Literature: Singh 2025 (2503.05631) makes ICL persistent by making EVERY context exemplar match the query; no mixture fraction,
no size sweep of it. 2602.23197 (fine-tuning erodes ICL, linear attention): fix = update only W_v. No small-share-of-fresh-rules recipe found.
Next: scaling check of the 2% fix — sizes D=32/96, 3 seeds at D=64, curves.
Prediction (pretrain 600 steps on fresh rules, then fine-tune on T=8 tasks, D=64): MIXF=0 copy falls to ~.15 by 1200 FT steps; MIXF=0.02 keeps >=.9; recog ~1 both.

## Pretrain -> fine-tune (PRE=600 steps fresh rules, then 1200 steps on T=8 tasks, D=64, seed 0)
copy starts at 1.0 after pretraining. After fine-tuning (recog .995-.999 in every row):
| MIXF | lr .02 copy | lr .005 copy |
|---|---|---|
| 0 | .13 | .14 |
| .02 | .72 | .67 |
| .05 | .97 | |
| .10 | 1.0 | |
Prediction for sizes: same pattern at D=32 and D=96 (5% keeps >.9).
Sizes (PRE=600, FT 1200 on T=8): D=96 copy .14 -> .99 with MIXF .05 (recog .999 both). D=32: pretraining reached only copy .49/.38, FT erases it and .05 does not help -> rerun D=32 with PRE=1500 so it starts at ~1.0.
D=32 with PRE=1500 (copy 1.0 before FT): FT barely erases it (copy .84, recog only .58 — memorizes slowly); MIXF .05: copy .90, recog .80. The fade grows with size: -.16 (D=32), -.87 (64), -.86 (96).
D=128 (PRE 600, FT 1200): copy .15 -> .99 with MIXF .05, recog 1.0/.999. FT length 300 (D=64): copy .71 -> .99 (recog .57/.47, not done memorizing).
FT 16x (4800, D=64): copy .14 -> 1.0 with MIXF .05, recog 1.0/.999. Control prediction: MIXF .05 with MIXB=32 (fresh rules, no repeated x -> nothing to read) does not help (copy ~.14).
Seeds (D=64, FT 1200): MIXF 0 copy .126/.121/.138, MIXF .05 copy .971/.986/.948; recog .997-1.0 all.
Control MIXF .05 with MIXB=32 (fresh rules, nothing to read): copy .13 -> no help. The mix must reward reading the prompt.
Big-model evidence: symbol tuning (Wei 2023, 2305.08298, Flan-PaLM 8B-540B): "flipped-label following is lost during
instruction tuning but can be restored via symbol tuning"; a 16% symbol-data mix already changes results. They restore after,
as a separate stage or >=16% mix; here 5% during fine-tuning prevents the loss with no task cost, and the loss grows with size.
Seed 1 at D=96 / D=128 (FT 1200): MIXF 0 copy .14 / .14, MIXF .05 copy .99 / 1.0; recog .999 all.

## Scaling check — fine-tuning erases prompt reading; 5% made-up-rule sequences prevent it (PASSED -> findings/fine-tuning-stops-prompt-reading.md)
| width (params) | copy after FT, tasks only | + 5% | tasks right (only / +5%) | seeds |
|---|---|---|---|---|
| 32 (29k, PRE 1500) | .84 | .90 | .58 / .80 | 1 |
| 64 (109k) | .13 | .97 | .998 / .997 | 3 |
| 96 (238k) | .14 | .99 | .999 / .999 | 2 |
| 128 (418k) | .15 | .99 | 1.0 / .999 | 2 |
Length (D=64, FT 300/1200/4800): .71/.13/.14 without, .99/.97/1.0 with 5%.
Cause: once the fixed tasks are stored, the stored answer is as good as reading, so nothing rewards the reading circuit; a mix
whose answers exist only in the prompt keeps rewarding it (control without repeats does nothing). Not toy-only.
Guess at 1B+ (guess): the erasure gets no milder with size (bigger models memorize fine-tuning sets faster, Tirumala 2022);
a few % of arbitrary-label in-context data should still prevent it; symbol tuning sees the erasure in Flan-PaLM 8B-540B.

## Task vector (tv.py, T=16, D=48, L=4, 300 steps, seed 0) — reproduction, not scaled
Mean hidden state at the query per task, added (x2) to a lone input: right 0.60 at the middle layer (after block 1 of 4) vs
0.13 without and 0.96 with the full prompt; earlier/later layers 0.12-0.46. Swapping the task in a prompt of another task: 0.42.
Matches Todd/Hendel (middle layers). Not new; not scaled (D=96 L=4 runs exceed the time budget under current load).
