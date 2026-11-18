# Mixtral W4A16 routing handoff

## Outcome as of November 19, 2026

Both Hugging Face replacement checkpoints produced routing counts for
all five benchmarks on October 24. Every marginal count matrix showed
all eight experts active in every layer. These results supersede the
earlier blanket exclusion of Mixtral W4A16 from Jaccard analysis.
Use the replacement files with their checkpoint IDs recorded.

| Checkpoint | Benchmarks | Results |
|---|---|---|
| `MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-AWQ` | 5/5 | 8 of 8 experts active in every layer |
| `jarrelscy/Mixtral-8x22B-Instruct-v0.1-GPTQ-4bit` | 5/5 | 8 of 8 experts active in every layer |

Both trees live under
`/data/scratch/projects/uom00014/vllm/results_wide_w4a16_hf_h100/<org>--<model>/default/moe-<benchmark>/expert_counts.json`.
Benchmarks are MMLU, BBH, HumanEval, INCLUDE, and PopQA. The checkpoint
is part of the experiment identity even though the output path says
`default`.

## What happened in the original runs

The local prequantized Mixtral results averaged 2.52 active experts out
of 8 per layer, with layers 8–55 routing to fixed per-layer pairs. The
W8A16 baseline used all eight. The Marlin run (32517565) and the four
completed AWQ benchmarks of the Triton attempt (33143679) reproduced
the concentration. The Triton job timed out. The matching pattern
weakened the original backend-specific capture hypothesis.

The public checkpoints demonstrate that Mixtral's top‑2 routing can
retain full aggregate expert coverage under W4A16. The cause of the
earlier concentration remains unresolved. The local quantization recipe,
checkpoint differences, and evaluation path are possible contributors.
The earlier router-logit collapse explanation was a hypothesis without
GPU hook measurements. The obsolete local checkpoints and stale Mixtral
outputs were removed during the replacement work.

## Reproduction

The initial replacement jobs (33405490 and 33416017) exited with code 3
on October 20 because the sbatch looked for single-dash HF cache paths.
The weights were already staged. Use `models--org--repo`, offline HF
mode, and the checkpoint IDs above. Let vLLM autodetect quantization
from each checkpoint config. The replacement sweep uses TP=4 and the
vLLM analyzer in `examples/eval-moe`. Check all five JSONs and compare
`aggregate.marginal_expert_counts`; full coverage is a sanity check
before the cross-backend analysis.

SGLang Mixtral BF16 is a separate completed baseline. See the
[SGLang handoff](../sglang/SGLANG_HANDOFF.md) and the
[main evaluation handoff](EVAL_MOE_HANDOFF.md).
