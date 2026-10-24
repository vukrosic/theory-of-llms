# A transcoder keeps the inputs and drops the sum

**Claim:** a transcoder (the tool behind attribution graphs: sparse latents that read an MLP's input and write its output) put in place of the adder's MLP rebuilds the parts that carry each number alone almost perfectly, and loses the part that carries their sum, which is the part the model needs. Status: proven (3 models).

**Example:** with 4 of 64 latents active per input, the transcoder explains 82% of the MLP output's variance. It rebuilds 98% of the "a alone" and "b alone" parts but only 71% of the part that depends on a + b, and the model's accuracy with the transcoder in place of its MLP falls from 99% to 72%. With 16 active latents the sum part is 98% rebuilt and accuracy is back to 98%.

![transcoder](../figures/transcoder-drops-the-sum.png)

| active latents (k of 64) | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| "each number alone" rebuilt (3 models) | 0.69-0.71 | 0.89-0.91 | 0.96-0.98 | 0.99 | 1.00 |
| "the sum" rebuilt | 0.38-0.40 | 0.44-0.67 | 0.71-0.89 | 0.94-0.97 | 0.98-0.99 |
| accuracy lost | 0.60-0.75 | 0.34-0.62 | 0.07-0.27 | 0.00-0.02 | 0.00-0.01 |

The same holds for a plain SAE on the MLP neurons (model 0, 64 latents, 4 active): 85% of the variance explained, 99% of "each number alone" rebuilt, 65% of the sum, accuracy 66%. At every sparsity the sum part is rebuilt worse than the input parts, so the transcoder's error is not noise: it is where the addition happens.

**What this builds on:** [Transcoders](https://arxiv.org/abs/2406.11944) and [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html), where whatever the replacement model misses is kept as "error nodes", and [Gao et al. 2024](https://arxiv.org/abs/2406.04093), who show SAE reconstructions cost more downstream loss than their variance suggests. **New here:** on a model with a known mechanism we can split the error by meaning, and it is the computed part (a + b), not the inputs.

Related: [a sparse autoencoder sees answers, not the clock](sae-sees-answers-not-clocks.md). Code: [experiments/sae-clock](../experiments/sae-clock/) (`transcoder.py`, `fig.py`).
