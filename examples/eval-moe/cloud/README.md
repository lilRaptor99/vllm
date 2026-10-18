# sglang BF16 Mixtral validation scripts

This directory contains the first-pass, low-volume validation flow for a 4x H100 BF16 Mixtral run, followed by the full sweep once the smoke test passes.

> **For the full background, gotchas, change log, and runbook** see
> [HANDOFF.md](HANDOFF.md) (725 lines). This README is the
> quick-start; the handoff is the authoritative reference.

## Phase 0: prepare the VM before the GPU test

Run this first on the VM that will eventually become the 4x H100 machine:

```bash
cd /home/pratheek/vllm
bash examples/eval-moe/cloud/prepare_sglang_mixtral_bf16_vm.sh
```

This creates the dedicated Python venv and downloads the Mixtral BF16 model into the local Hugging Face cache so the later H100 run can start immediately.

## Smoke validation (small sample, fast feedback)

```bash
cd /home/pratheek/vllm
bash examples/eval-moe/cloud/sglang_mixtral_bf16_4gpu_smoke.sh
```

Defaults:
- model: `mistralai/Mixtral-8x22B-Instruct-v0.1`
- GPUs: 4 H100
- benchmark: `mmlu`
- sample budget: `8`
- max tokens: `32`
- output dir: `${HOME}/sglang-mixtral-bf16-smoke`

This script:
1. checks that 4 GPUs are visible,
2. creates a dedicated sglang venv if needed,
3. installs the cu128-compatible sglang stack (
   `sglang==0.5.9` + matching dataset / numpy stack),
4. forces `enable_return_routed_experts` via a wrapper,
5. launches an sglang server with TP=4,
6. runs `moe_expert_stats_sglang.py` once against a tiny prompt set.

## Full sweep (after smoke validation passes)

```bash
cd /home/pratheek/vllm
bash examples/eval-moe/cloud/sglang_mixtral_bf16_4gpu_full.sh
```

Defaults:
- model: `mistralai/Mixtral-8x22B-Instruct-v0.1`
- GPUs: 4 H100
- benchmarks: `mmlu bbh humaneval popqa include`
- sample budget: `999999` (effectively full data)
- max tokens: `256`
- output dir: `${HOME}/sglang-mixtral-bf16-full`

This script follows the same server-side pattern as the Spartan sbatch jobs,
then loops over the full benchmark set and writes the standard
`expert_counts.json` tree consumed by the downstream Jaccard/heatmap tools.

## Important notes

- The `sglang` server requires the routed-experts capturer to be enabled on the
  server side; the wrapper forces this because `sglang==0.5.9` does not expose a
  reliable CLI flag for it.
- The scripts assume the repo checkout in `vllm` contains
  `examples/eval-moe/moe_expert_stats_sglang.py` and the required dataset helpers.
- Once the VM is resized to 4x H100, these are the validation steps before the
  wider sweep is run.
