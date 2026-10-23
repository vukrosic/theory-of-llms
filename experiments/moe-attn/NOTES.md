# moe-attn notes (2026-10-24)

Toy: facts [subject, rel, ans, rel, ans...] (attention carries subject, MLP stores facts) + copy. NS=256 (2048 facts): dense 63% vs MoE-8 82%, capacity-bound.
A lab researcher who reads this would change: how their Muon splits fused/3D expert weights (per-expert Newton-Schulz).

- Attention lr x2 hurts facts ~0.2 in both dense and MoE; x0.5 no gain. Not MoE-specific. Dropped.
- "MoE slows copy learning": copy-only 8 seeds dense 1.94±0.18, MoE-1 2.23±0.14, MoE-8 2.12±0.17 -> noise. Dropped.
- Expert layout under Muon (finding): each 0.848, fused 1.405, stack 1.473, flat 1.592, dense 1.564 (4 seeds, expert lr tuned). E=16: each 0.517, fused 1.30. D=64: 1.41 vs 1.79 (dense 1.92).
- Per-expert step x load^p (p=0.5,1,-0.5): no change (bias balancing keeps loads equal).
- DENSE0 (first layer dense MLP): no cost, known (DeepSeek). Dropped.
- Router input h / x0 (token embedding) / both: depends on whether the current token is a good key; no universal recipe; known (hash layers, StableMoE). Dropped.
- Routing MI (route_mi.py): layer 0 top-1 expert keys on relation (1.77 bits), layer 1 on subject (1.31); GENERIC "is" token: layer 1 subject 1.39.
- Granularity at fixed active+total (lr 0.02, seeds 0|1): E8K2 0.775|0.751, E16K4 0.753|0.642, E32K8 0.800|0.785, E64K16 1.140|0.867. Finer helps to E16, hurts at E64. Known (DeepSeekMoE, granularity laws). >5 s.
- Router on Muon (ROUTER_MUON=1) mult 0.25/1/4: 0.90/0.82/1.07 vs Adam 0.76. Worse. Dropped.
- Muon scale rule on narrow experts (E32K8, EF=11): keller 0.79 / none 0.85 / spectral 0.88 (best lr each, 2 seeds; spectral best maybe ~0.03). Small, noisy. Dropped.

## MoE vs dense: attention score blow-up without QK-norm (2026-10-24)
A lab researcher who reads this would change: the lr headroom / QK-Clip threshold when moving a dense Muon recipe (no QK-norm, e.g. MLA) to MoE.
Literature (3 extended searches): Kimi K2 saw logits >1000 in MoE with Muon; nobody compares MoE vs dense at matched active size.
Prediction: same blow-up lr for MoE and dense (attention never sees the experts); maybe MoE a bit earlier.
Result (2 seeds, lr 0.02-0.08): blow-up lr identical. Layer-1 peak score dense/MoE: 54/55 at 0.02, 70/71 at 0.028, 103/104 at 0.04, 232/150 at 0.057, 361/363 at 0.08. Prediction held, no MoE effect. Dropped.
- Fused gate_up / QKV under Muon: known (Dion3 2608.11612, Megatron #7077, Qwen3.8-Next per-head). Skipped.
- Expert layout at B=128 (512 tok/expert/step), 150 steps: dense 1.14, each 0.54, fused 0.98 -> holds; added to finding as point 4.
- Fine-grained E32K8 (EF=11<D): each 0.79, fused best 1.42 (x0.5) -> holds, added to finding point 2.
- Default muon toy (MOE=8 sampled.py full 260 0.014, EXP_FUSED=1 added to mix.py): dense 2.687, each 2.680, fused x0.5 2.689 / x1 2.718 (2 seeds). MoE gain there is only 0.007, so the layout can't matter; consistent.

## GQA (fewer key/value heads) x MoE (2026-10-24)
A lab researcher who reads this would change: how hard they compress KV (GQA groups / MLA rank) in an MoE vs a dense model.
Prediction: KVH=1 hurts MoE more than dense on facts (attention must deliver the subject that routing + experts use).
Result: dense-MoE facts gap KVH4 0.53, KVH2 0.67, KVH1 0.36 (2 seeds); both lose about equally, noisy. No interaction. Dropped. (Note: code edits change float order -> same seed can move 0.2-0.3 on dense; compare within one code version.)
- fusednorm (fused directions, each expert block rescaled to per-expert step size) x0.5: 1.24/1.34 ~ fused 1.28 -> directions, not step sizes. Added to finding.

## Night 2026-10-24, after the layout finding
- Megatron audit: `--moe-single-grouped-weight` packs experts into one TE GroupedTensor of logical shape [E*rows, cols] (2D); Megatron's plain Muon sends every 2D param to NS whole; only the layer-wise distributed optimizer asserts against it. Added to the finding (read from code, not run).
- Upcycling under Muon: SPARKLING (2602.02472) covers symmetry breaking under Muon for expansion incl. expert-inner; Drop-Upcycling covers re-init. Skipped.
- Real text (DATA=tiny, TinyStories 4k BPE): 12-18 s per run and MoE gain only 0.013 at 400 steps (not capacity-bound). Can't test the layout on text under the 5 s rule.
- Idle experts (LAZY knob): stale momentum gets full-size Muon steps when an expert gets 0 tokens, but that needs tokens/step < E/K, which lab batches never hit. Dropped without runs. Excitation (2602.21798) = utilization scaling (our LOADPOW), not Muon.
- Mamba2 fused in_proj under Muon: taken (Muon Meets Mamba 2608.03941, App. D: per-slice NS beats merged by 0.024).
