# Numbers on a clock

**Claim:** a model trained to add numbers mod 47 places every number on a circle and adds by turning, using about 5 clocks of different speeds at once. Status: proven (3 seeds).

**Example:** the model's score for each possible answer is a sum of 5 waves that all peak at the correct answer; 77% of the scores are explained by these waves (0% for the same model before training).

![waves](../figures/waves.png)

It is the clock method, not the pizza method: the model is equally sure for every pair (spread 2-4% across a-b), while pizza would be less sure when a and b are far apart.

Each training run picks different clock speeds (seed 0: 13, 3, 22, 20, 11; seed 1: 17, 3, 1, 2, 9; seed 2: 16, 5, 23, 1, 4), with the same method.

Related: [neuron teams](neuron-teams.md), [weight decay](weight-decay-forces-the-clock.md). Code: [experiments/clock-addition](../experiments/clock-addition/).
