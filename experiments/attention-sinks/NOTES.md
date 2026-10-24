# Attention sinks and huge activations with Muon — notes

Lab-use sentence: a lab researcher who reads this would change which Muon setting they use (or whether they add a gate / extra key)
to keep sinks and huge activations small, which matters for low-bit training, quantization and long context.

## Literature (2026-10-24, 3 extended searches)
- Xiao 2023 (StreamingLLM): keep the first 4 tokens' KV or the model breaks when the window slides.
- Sun 2024 (Massive Activations): a few channels ~1000x the median at the first token / delimiters, appear in early layers, act as fixed biases.
- Gu 2024 (2410.10781): sink appears after effective training on enough data; stronger with larger lr and weight decay (AdamW); softmax sum-to-1 is the cause (sigmoid attention without normalization: no sink); extra learned key/value takes the sink.
- Qiu 2025 (2505.06708, Qwen gated attention): sigmoid gate after SDPA: first-token attention 46.7% -> 4.8%, max activation 1053 -> 94.
- Softpick 2504.20966: sink rate softmax 48%, softmax+1 24%, gate 5%, softpick 0%.
- OSP (ACL 2025): Muon alone reduces activation outliers only partly; needs single-scale norm + embedding projection.
- 2603.05498 (Spike, Sparse, Sink): pre-norm is what couples sinks and huge activations; huge-activation size varies independently of loss.
- 2603.17771: huge activations damp the gradient that the sink concentrates (RMSNorm Jacobian ~ 1/norm).
- 2605.08504: one early "massive emergence" layer (norm + MLP) makes them.
- No paper measures how Muon settings (lr, momentum, weight decay) change sinks / huge activations, or whether the gate still matters under Muon.

## Toy
sink.py: byte LM on ~7.8 MB local markdown (corpus.npz, not in the repo), RoPE, QK-norm, SwiGLU, RMSNorm, no biases.
Muon NS5 momentum 0.95 on matrices, Adam lr 0.01 on embedding/head/norms, warmup 30, linear decay over the last half.

## Runs

All runs: Muon (NS5, momentum 0.95) on hidden matrices, seed 0, one run each. Load from other sessions was 7-9 heavy processes, so wall times are 2-10x the 5 s budget.

| toy | setting | result |
|---|---|---|
| byte text LM, T 64, 1-3 layers (sink.py) | default | no sink: attention on token 0 <= 0.015 |
| Bigram-Backcopy, 1 layer 1 head (sink.py DATA=bb) | lr 0.005-0.04 | partial sink 0.07-0.14 at non-trigger queries, value drain v0 0.27-0.77, residual peak r0 1.9-4.5; not monotone in lr |
| BB 1L | extra key: learned KV / learned logit (gpt-oss) / softmax+1 | takes 0.99 / 0.63 / 0.04 of attention; all three cut r0 5.0 -> 2.0 and max/median 25 -> 10; loss unchanged (1.562) |
| BB 3L, const lr, 2000 steps | default | no sink (0.036), no growing peak |
| BB 1L | SGD on matrices | same weak sink (0.03-0.12): not caused by Muon |
| BB 1L | QK-norm off | sink 0.04-0.11, r0 5.6 -> 7.2 from 1000 to 4000 steps |
| 8-skill MoE toy, no start token (toyprobe BOS=0) | 260 -> 1040 steps | heads >0.3 on token 0: 0.375 -> 0.875, but token 0 is the skill token: blocking it +1.04 loss, v0 ~1, gate does not move it |
| 8-skill MoE toy + fixed start token (BOS=1) | 260 / 1040 steps | no start-token sink (0.084 / 0.060, no head >0.3); attention parks on the skill token (0.39 / 0.53); blocking the start token still +0.16 / +0.22 |
| 8-skill MoE toy + start token | width 32 / 64 / 128, 1040 steps | last-layer max/median 15.8 / 20.1 / 29.0 (grows with width and steps: 13.6 at 260) |
| same, width 64 | diagnosis | the outlier is one dimension that is largest on 96% of tokens (mean 14x median), not a token-0 activation |
| same | HBIAS=loginit (log-frequency output bias) | constant dimension gone (3.6x, 8% of tokens), loss same 1.429, but 4-bit activation loss rise worse 0.22 -> 0.30: not a recipe |
| same | NORM_MULT=0 (frozen norm gains) | constant dimension 93x, loss 1.72 (worse) |
| same | Muon WD 0.1 / 0.3 | 4-bit rise 0.22 -> 0.18 / 0.15, loss 1.433 -> 1.459 / 1.733: trade, and WD shrinking outliers is known (Moonlight) |

Toy-only cause sentence: strong first-token sinks need many layers and long contexts in which most heads must "do nothing" for most queries; with 16-64 tokens and 1-3 layers a head can average harmlessly, so no sink forms and every sink result above depends on toy size. Direction stopped: no effect passed the size/length check, nothing written to findings/.

Extrapolation (guess, not measured): the only effect that grew with size, the constant outlier dimension (max/median 16 -> 29 over 16x params), would be in the hundreds at 1B+, consistent with known outlier features; we found no Muon setting that removes it without a loss or 4-bit cost.
