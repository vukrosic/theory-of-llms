# Contents

The order of the findings lives only here. Files have no numbers, so they can be moved freely.

## How a model stores things
- [Numbers on a clock](findings/clock-addition.md)
- [Counts on a line](findings/count-on-a-line.md)
- [Counts bend from a line into corners, so one steering direction can't set them](findings/ordered-values-bend.md)
- [Muon packs features the same way, and weight decay sets how many fit](findings/muon-superposition.md)

## How it moves information
- [The two-head copy machine](findings/copy-heads.md)

## How it splits work
- [Small neuron teams, one wave each](findings/neuron-teams.md)
- [Experts split by skill](findings/experts-split-by-skill.md)

## How learning unfolds
- [Fine-tuning on a few fixed tasks makes a model stop reading the prompt](findings/fine-tuning-stops-prompt-reading.md)
- [Weight decay forces the clock](findings/weight-decay-forces-the-clock.md)
- [Learning rate sets speed, not where the model ends up](findings/learning-rate-sets-speed.md)
- [What changes Muon's best learning rate](findings/what-changes-muon-learning-rate.md)
- [Without QK-norm, attention scores blow up above a learning rate threshold](findings/muon-attention-blowup.md)
- [When Muon needs QK-Clip, and how to set it](findings/muon-qkclip-rules.md)
- [Repeating data with Muon needs much higher weight decay](findings/muon-weight-decay-epochs.md)
- [With Muon, orthogonalize each MoE expert alone](findings/muon-moe-expert-layout.md)
- [Keep gradient clipping on for Muon matrices, with a low threshold](findings/muon-grad-clipping.md)

## What interpretability tools see
- [A sparse autoencoder sees answers, not the clock](findings/sae-sees-answers-not-clocks.md)
- [A transcoder keeps the inputs and drops the sum](findings/transcoders-drop-the-sum.md)
