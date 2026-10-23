# Theory of LLMs

What do small language models actually build inside when they learn? We open up trained models and check every claim with an experiment.

![The adder's answer score is a sum of 5 waves that all peak at the right answer](figures/waves.png)

| What we found | One example | Page |
|---|---|---|
| A model that adds numbers invents a clock | every number is a point on a circle; 5 clocks of different speeds add up to a spike at the right answer | [clock-addition](ideas/clock-addition.md) |
| Each clock is run by 2-4 neurons out of 128 | removing 3 neurons: 99% -> 83%; removing 3 random ones: 98% | [neuron-teams](ideas/neuron-teams.md) |
| A neuron that looks like noise is a smooth wave in the model's own number order | sort numbers by 22 x n mod 47 and the noise becomes one hill | [neuron-teams](ideas/neuron-teams.md) |
| Weight decay is what makes it find the clock | without it the model memorizes and stays at 60% on new pairs | [weight-decay-forces-the-clock](ideas/weight-decay-forces-the-clock.md) |
| To copy text, a model builds a two-head machine | one head marks the previous word, a second finds the earlier copy and reads the next word | [copy-heads](ideas/copy-heads.md) |
| The copy machine appears late in training | not there at step 260, built by step 1,200 | [copy-heads](ideas/copy-heads.md) |
| Experts split by skill on their own | facts and counting share one expert, copy and sort another | [experts-split-by-skill](ideas/experts-split-by-skill.md) |
| Counts are stored in order on a bent line | 0, 1, 2, 3, 4 zeros sit in order; one straight readout gets 99% | [count-on-a-line](ideas/count-on-a-line.md) |

![A noisy neuron becomes one smooth hill when numbers are put in the model's own order](figures/reorder.png)

All ideas in reading order: [SUMMARY.md](SUMMARY.md). Open questions: [OPEN.md](OPEN.md). Code behind each page: [experiments/](experiments/). All models are trained with the Muon optimizer.
