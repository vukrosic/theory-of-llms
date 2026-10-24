# RoPE frequencies and long context — notes

Lab-use sentence: a lab researcher who reads this would change how many RoPE pairs they rotate (leave the slowest ones
unrotated) or which context-extension trick they plan for, if the frequency split measured on a short model predicts
which one keeps loss flat at 4x the training length.

## Literature (2026-10-24, 4 extended searches + abstracts/papers read)
- Barbero 2024 (2410.06205, Gemma 7B): highest frequencies build positional heads (diagonal / previous-token); most norm sits in the
  lowest frequencies, which act as semantic channels; p-RoPE (drop the slowest pairs' rotation) improves loss.
- Urrutia 2026 (2605.31558, newest close paper, read): positional heads use mid-high frequencies, symbolic heads low; symbolic
  mechanisms extrapolate to longer inputs, positional ones fail; checked on GPT-5.x / Claude 3.7 by prompting only.
- 2606.21249 (OLMo-2, Qwen2.5, Gemma-2, Mistral 7-14B): zeroing the 32 slowest of 128 dims of retrieval heads drops NIAH recall 1.00 -> 0.18
  (random dims: 0.98). Retrieval heads need the slowest pairs.
- AdaRoPE (2607.19363): learned per-head frequencies + per-head scaling beat partial RoPE / NoPE / YaRN's uniform scaling.
- Partial RoPE (2603.11611): rotating ~10% of dims matches full RoPE loss; NoPE gives loss spikes. No extrapolation test.
- Qwen3-Next rotates 25% of dims and claims better extrapolation (no public ablation found).
- Scaling laws of RoPE extrapolation (2310.05209): "critical dimension" = pairs whose period fits in the training window; pairs past it
  cause extrapolation failure. LongRoPE/LongRoPE2: per-pair rescale factors found by search. YaRN: ramp by rotations-per-window r.
- Cracks in the Foundation (2608.10296, 26 x 7B OlmPool): norm, GQA, pretrain length, SWA compound to -47% long-context; not visible in short loss.
- Gap: nobody sweeps the rotated fraction during pretraining and measures extension (no trick / PI / NTK / YaRN), and nobody checks
  whether a short-model frequency measurement predicts which trick works.

## Toy
rope.py: byte LM on the attention-sinks corpus (7.4 MB local markdown, not in the repo), RoPE, QK-norm, SwiGLU, RMSNorm, no biases.
Muon NS5 momentum 0.95 on matrices; Adam lr 0.01 on embedding/head/norms; warmup 30, linear decay over the second half.
Eval at 1x, 2x, 4x the training length with no trick / PI / NTK / YaRN (no fine-tuning): text loss on positions past the training
length; near / far = text targets right after a 4-byte string seen <= T/2 back / > T back. EXT_STEPS: continued training at 4x length
(same tokens per step, peak lr x0.3) with the method's frequencies, then eval again. HEADSTAT: per-head previous-token score, induction
score, loss change when that head's fast or slow half of pairs is zeroed. COPYF (random-byte copy rows) never learned; off by default.
Runs: ./run.sh OUT "ENV=.." seeds; summary: python3 show.py OUT [columns].

## Runs
### 1. base 10000, width 64, 4 heads (head 16 = 8 pairs), 2 layers, T 64, 300 steps, 3 seeds (base.jsonl)
Prediction: PI worst, NTK best; dropping fast pairs hurts more than slow. Held.
| rotated | 1x | 4x none | 4x ntk | 4x pi | drop fast | drop slow |
|---|---|---|---|---|---|---|
| all 8 | 2.317 | 2.582 | 2.449 | 3.216 | +0.92 | +0.06 |
| fastest 4 (p-RoPE) | 2.314 | 2.591 | 2.585 | 3.200 | | |
| 4 spread (HF partial) | 2.356 | 2.848 | 2.787 | 3.152 | | |
| fastest 2 | 2.449 | 3.004 | 3.267 | 3.207 | | |
| 2 spread (HF partial) | 2.600 | 2.604 | 2.778 | 3.007 | | |
| none (NoPE) | 2.825 | 2.874 | | | | |
- Most used pair = pair 2, period 63 = the training window (Oka 2026 "frequency band"); pairs slower than the window carry ~3% each.
- Caveat: base 10000 at T 64 leaves 5/8 pairs below one turn per window; real models (Llama 3 at 8k, base 500k) ~45%. Recalibrate.

### 2. base 100, 2 heads (head 32 = 16 pairs), rest as run 1, 3 seeds (dk32.jsonl) -- ~50% of pairs turn once per window, like Llama 3
| rotated | 1x | 4x none | 4x ntk | drop fast | drop slow |
|---|---|---|---|---|---|
| all 16 | 2.288 | 2.463 | 2.395 | +0.91 | +0.27 |
| fastest 8 | 2.298 | 2.455 | 2.526 | | |
| 8 spread | 2.319 | 2.554 | | | |
| fastest 4 | 2.339 | 2.463 | 2.827 | | |
| 4 spread | 2.333 | 2.659 | | | |
| none | 2.838 | 2.877 | | | |
| all 16, base 10000 | 2.317 | 2.548 | | | |
- Fastest-k vs spread-k order flips between run 1 and run 2 (depends on base / head size): no stable rule. Dropped.
- NTK hurts truncated variants: it stretches the pairs they use. PI without fine-tuning is worst everywhere (known).

### 3. QK-norm and partial RoPE through a 4x extension stage (qk.jsonl, base 100, 2 heads, ext 60 steps NTK, 3 seeds)
Prediction (from Cracks 2608.10296 / Yang 2025b): QK-norm on = same 1x loss, worse 4x text / far after extension.
| variant | 1x | 4x no trick | after ext 4x text | after ext far | after ext 1x |
|---|---|---|---|---|---|
| QK-norm off | 2.300 | 2.819 | 2.274 | 2.440 | 2.271 |
| QK-norm on | 2.288 | 2.463 | 2.263 | 2.412 | 2.271 |
| on, 8 of 16 pairs rotated (HF partial) | 2.319 | 2.554 | 2.300 | 2.461 | 2.294 |
| on, 4 of 16 pairs rotated (HF partial) | 2.333 | 2.659 | 2.337 | 2.509 | 2.327 |
- Prediction failed: after extension QK-norm on = off (within 1 sd); the 7B harm is not reproduced at this size. QK-norm only helps
  no-trick extrapolation (-0.36), which labs do not rely on. Partial RoPE stays 0.03-0.07 behind full RoPE through extension. Dropped.

### 4. Per-head frequency use (HEADSTAT=1, 4 heads x 2 layers, base 100, seed 0, smoke.jsonl)
- Induction score <= 0.034 in every head: byte-level 2-layer toys at 300 steps form no copy / retrieval heads, so "which pairs do
  retrieval heads use" (2606.21249: the slowest) cannot be measured in 5 s toys. Previous-token heads (score 0.18-0.23) lose most from
  zeroing their fast half (+0.06 to +0.21 text) vs slow half (+0.03 to +0.06), matching Barbero.

## Status
Every stable toy result here reproduces a published one (Oka frequency band, PI worst, slow pairs light, fast pairs = previous-token);
nothing passed the lab-use test, so no scaling check was run. Retrieval-head frequency use needs a model that forms induction heads
(longer training or a larger vocabulary) -- outside the 5 s budget.
