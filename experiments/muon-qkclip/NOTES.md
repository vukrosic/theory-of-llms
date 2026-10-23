# QK-Clip (Kimi K2 MuonClip) vs QK-norm in a 2-layer Muon toy

Question: without QK-norm, does QK-Clip (per head, after each step) remove the lr blow-up above lr* ~ 0.043 (see ../muon-attention-logits), and what does it cost at the best lr?
Lab-change sentence: "Without QK-norm (e.g. with MLA), QK-Clip at tau = X recovers QK-norm's loss across the whole lr range / costs Y at the best lr; pick tau as ~k x the logit level at your best lr."

## Literature (4 extended searches, 2026-10-23)
- Kimi K2 (2507.20534): gamma_h = min(1, tau / max_logit_h) per head, from the current batch, applied after the optimizer step; Wq, Wk scaled by gamma^0.5 each (alpha=0.5). MLA: head-specific content q,k x sqrt(gamma), head-specific rotary q x gamma, shared rotary k untouched. K2 uses tau = 100; logits sit at the cap ~30% of training, then fall without changing tau. 12.7% of heads clipped at least once in first 70k steps.
- Only tau check in the paper: 0.5B-active/3B-total MoE, Muon vs MuonClip with tau = 30: loss curves nearly identical. No tau sweep, no rule for choosing tau, no lr sweep; no independent follow-up found. Third-party blogs/implementations just copy tau=100.
- Gemma 2 softcap (50 on attention logits) replaced by QK-norm in Gemma 3; no controlled softcap-vs-QK-norm loss comparison found (HF issue: loss 1.7459 vs 1.7452 on one sample).
- Not found: QK-Clip tested across an lr range (esp. above the lr where no-QK-norm Muon blows up), or tau scaled relative to the logit level at the best lr.

Variant tested: simple per-head version, all q,k dims (RoPE applied to all dims, no MLA shared-rope part), Wq/Wk rows of the head x sqrt(gamma), gamma from the training batch's max |q.k|/sqrt(d) after RoPE, applied after each step. Weight decay 0.

## Prediction (written before runs)
- tau=100: never triggers at lr 0.014 (logit ~10); at lr 0.057 / 0.113 it caps logit near 100 but that is still 10x higher than QK-norm's, so loss gap shrinks only part way (gap 0.024 -> ~0.01, 0.33 -> ~0.15).
- tau=30: loss gap at 0.113 mostly gone (<0.05); no cost at 0.014 (logit 10 < 30).
- tau=10: triggers at 0.014 (peak logit 9.8 ~ tau) with a small cost (~0.01-0.02); removes the gap at high lr.

## Results (1 seed, D16, B8, 800 steps, no QK-norm unless stated; loss / peak probe logit; raw: qkclip.jsonl; tau=5,3 added after tau=100..10 left a gap)
| tau | lr 0.014 (best) | lr 0.057 (1.3x lr*) | lr 0.113 (2.6x lr*) |
|---|---|---|---|
| none | 4.002 / 9.8 | 4.097 / 653 | 4.428 / 3160 |
| 100 | 4.002 / 9.8 (0 clips) | 4.087 / 120 | 4.303 / 141 |
| 30 | 4.002 / 9.8 (0 clips) | 4.083 / 44 | 4.192 / 45 |
| 10 | 4.002 / 9.8 (2 clips) | 4.064 / 15 | 4.157 / 24 |
| 5 | 4.004 / 6.3 | 4.059 / 8.2 | 4.071 / 14 |
| 3 | 4.013 / 4.5 | 4.052 / 5.6 | 4.080 / 11 |
| QK-norm (no clip) | 4.017 / 3.6 | 4.073 / 3.3 | 4.099 / 6.2 |
(init logit 2.1; noise ~0.01-0.02. Peak logit is the probe batch every 50 steps, so it can miss transient spikes.)

## Prediction check
Wrong in two ways: tau=100 helped less than predicted at 0.113 (4.303, gap to QK-norm still +0.20; logit stays 140 because it is checked only after the step), and the useful tau was much lower than 30: tau=30 and 10 still leave +0.09 and +0.06 at 0.113. Right: no cost at the best lr while tau > peak logit (9.8).

## Conclusion
In this toy, QK-Clip at tau = 5 (about 0.5x the best-lr logit, 2.4x the init logit) matches QK-norm's loss at 1.3x and 2.6x lr* (4.059 / 4.071 vs 4.073 / 4.099) and costs +0.002 at the best lr; tau = 100 or 30 (Kimi's values) leave the lr-0.113 loss +0.20 / +0.09 above QK-norm.
Caveats: 1 seed; tau sensitivity depends on the logit scale of this 16-dim model (best-lr peak 9.8) so the transferable rule is "tau around 0.5-1x the peak logit you see at your best lr", not tau = 5; simple per-head version without MLA shared-rope handling; no weight decay.
