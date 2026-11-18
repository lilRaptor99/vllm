# MoE expert-activation evaluation handoff

## Status as of November 19, 2026

The study compares empirical expert activation across vLLM, llama.cpp,
and SGLang, and across weight precisions. The core grid covers
OLMoE-1B-7B-0924-Instruct, DeepSeek-MoE-16B-Chat, Qwen3-30B-A3B, and
Mixtral-8x22B-Instruct-v0.1 on MMLU, BBH, HumanEval, INCLUDE, and PopQA.

| Backend and precision | Coverage and location |
|---|---|
| vLLM BF16, FP8, W8A16 | 20 H100 files per precision. MediaFlux trees are `results_wide_bf16_h100`, `results_wide_fp8_h100`, and `results_wide_w8a16_h100`. The October 8 W8A16 upload supersedes older pending notes. |
| vLLM W4A16 | Local AWQ/GPTQ results for OLMoE, Qwen3, and DeepSeek are under `results_wide_w4a16_awq_h100`. The replacement Mixtral AWQ/GPTQ results are under `results_wide_w4a16_hf_h100` on scratch, with five files per checkpoint verified. |
| SGLang BF16 | 20 files on scratch under `results_sglang_bf16_h100`. Mixtral TP=4 completed October 20, job 33416348. See the [SGLang handoff](../sglang/SGLANG_HANDOFF.md) for setup and job history. |
| llama.cpp | Analyzer and comparison tools live in `/data/projects/uom00014/llama-cpp-eval`. Verify matched cells and archive availability when assembling the comparison. |

The earlier local Mixtral W4A16 checkpoints showed severe routing
concentration. The October 24 Hugging Face replacement AWQ and GPTQ
runs used all eight experts in every layer across all five benchmarks.
Use those replacement results when assembling the W4A16 comparison.
The [Mixtral handoff](MIXTRAL_W4A16_BUG_HANDOFF.md) records the evidence
and the unresolved cause of the original collapse.

## Reproduction and data

- Main repo is vLLM, branch `eval-moe`. Entry points in
  [examples/eval-moe](examples/eval-moe) are `moe_expert_stats.py`,
  `moe_expert_stats_sglang.py`, `awq_quantize.py`, and the `spartan/`
  sbatches. Scratch root is `/data/scratch/projects/uom00014/vllm`.
- Spartan GPU jobs run via `sbatch`. Use the staged HF cache and keep
  its `models--org--repo` directory convention. The vLLM job stack used
  GCCcore 11.3.0, Python 3.11.3, and CUDA 12.4.1 on a CUDA 12.8-capable
  driver. Mixtral BF16 uses TP=4 on four H100s.
- W4A16 uses prequantized weights. The online W4 paths were abandoned
  after H100 MoE kernel and checkpoint-format constraints. DeepSeek
  requires group size 64 across the expert projections. Its down
  projection dimension of 10944 is divisible by 64, and a mixed
  BF16/int4 expert was rejected by the Marlin path. The re-prequant
  fix is `c016fe05`. The AWQ path also needed remote-code handling
  for DeepSeek config loading.
- The results contract is one `expert_counts.json` per model,
  precision, and benchmark, with architecture, totals, marginal
  counts, intra-layer and adjacent-layer pair counts, and per-subject
  layer counts. Check the files and logs even when Slurm reports success.

## Open analysis

Run or locate the three-way BF16 Jaccard report using the sister repo's
`examples/eval-moe-overview/aggregate_overview.py` and Jaccard scripts.
No completed SGLang-inclusive report was found in the locations checked.
Match model, benchmark, precision, and prompt settings before comparing,
and retain checkpoint provenance for W4A16. SGLang and replacement
Mixtral W4A16 outputs were verified on scratch. Their MediaFlux uploads
were not verified. Older vLLM BF16 and FP8 archives may contain
duplicate nested paths, so select the canonical tree.
