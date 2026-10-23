# Weight decay forces the clock

**Claim:** without weight decay the adder memorizes its training pairs and stays at about 60% on new pairs; with weight decay on the embeddings it switches to the clock and reaches 98%. Status: likely (seed 0).

Why: memorizing 1,300 pairs one by one needs big weights; weight decay keeps shrinking weights, so the small clock solution wins.

Related: [numbers on a clock](clock-addition.md).
