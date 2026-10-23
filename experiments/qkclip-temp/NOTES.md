# Learnable per-head attention temperature as a QK-norm substitute

Lab-change: a lab that can't use QK-norm (MLA) with Muon would add one learnable scalar per head (log-temperature, trained by the Adam-type step) if it recovers what QK-norm gives.
Why: on copy-only data QK-norm beats no-QK-norm by 0.15-0.34 even when scores stay small (qkclip-data), so QK-norm's benefit there isn't stopping blow-up. Guess: its learnable gains act as a fast temperature knob; without it Muon must grow q,k weights to sharpen attention.
Prediction: TEMP=1 without QK-norm recovers most of the copy gap at lr 0.007/0.014 (3.53 → ~3.25 at 0.007), and does nothing on the full mix at the best lr.
Result (seed 0): copy-only 0.007 3.524 (none 3.526, QK-norm 3.190), 0.014 3.256 (none 3.251, QK-norm 3.052); full mix 0.014 4.002 (none 4.002), 0.057 4.082 / peak 125 (none 4.097 / 653).
Prediction wrong: a learnable temperature does not recover the copy gap. QK-norm's copy benefit likely comes from normalizing each token's q,k length (matching by direction), not from a temperature knob. Dropped.

## Query-only vs key-only norm (QKNORM=3 q only, 4 k only), seed 0
Lab-change: in MLA, queries are easy to normalize (not cached); if q-only norm is enough under Muon, MLA labs get QK-norm's benefit without touching the cached keys.
Prediction: either one alone bounds the blow-up partly (scores ~50-200 at 0.113 instead of 3160), loss at 0.113 between none (4.43) and QK-norm (4.10), ~4.2; copy gap half recovered.
Result (loss / peak score), seeds 0 | 1:
| | none | q-only | k-only | both |
|---|---|---|---|---|
| full mix lr 0.014 (s0) | 4.002 / 9.8 | 4.014 / 5.6 | 4.011 / 4.7 | 4.017 / 3.6 |
| full mix lr 0.057 (s0) | 4.097 / 653 | 4.059 / 13 | 4.066 / 13 | 4.073 / 3.3 |
| full mix lr 0.113 | 4.428 / 3160 \| 4.206 / 2597 | 4.175 / 123 \| 4.121 / 83 | 4.127 / 99 \| 4.117 / 115 | 4.099 / 6.2 \| 4.137 / 3.6 |
| copy only lr 0.007 | 3.526 \| 3.553 | 3.562 \| 3.533 | 3.035 \| 3.261 | 3.190 \| 3.401 |
- Either side alone stops the blow-up (scores ~100 instead of thousands) and keeps the loss within ~0.04 of full QK-norm at 8x best lr.
- On copy-only data the benefit comes entirely from the key norm: k-only ≥ both, q-only = none (2 seeds).
- Prediction wrong: one side alone is almost as good as both, not halfway.
3-seed means at lr 0.113: none 4.358, q-only 4.161 (4.175/4.121/4.186), k-only 4.132 (4.127/4.117/4.152), both 4.122. Either side alone recovers 83-96% of the gap.
