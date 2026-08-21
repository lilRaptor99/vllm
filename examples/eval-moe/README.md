# MoE Expert Activation Statistics

This directory contains a tool for analyzing how a Mixture-of-Experts (MoE)
language model routes tokens through its experts across different
benchmarks. For every layer of the model, it counts how many tokens were
routed to each individual expert and writes the result to JSON.

This is **not** a quality / accuracy / performance benchmark — it only
captures routing statistics (the expert IDs chosen for each token). It is
useful for studying expert load balancing, layer specialization, and
benchmark-specific routing patterns.

## What it does

The script:

1. For every `(model, quant)` cell, loads the model with
   `vllm.LLM` and `enable_return_routed_experts=True`, which enables
   vLLM's built-in per-token, per-layer, per-topk expert-id
   capture on `CompletionOutput.routed_experts`.
2. Loads a configurable subset of three benchmarks from the HuggingFace
   Hub and runs them through the model with `max_tokens=256`:
   - **MMLU** (`cais/mmlu`, subset `"all"`, 5-shot multiple-choice prompts)
   - **Big-Bench-Hard** (`Joschka/big_bench_hard`, all 27 sub-tasks)
   - **HumanEval** (`openai/openai_humaneval`, function-completion prompts)
3. Aggregates the captured expert IDs into a `(num_layers, num_experts)`
   int64 count matrix per benchmark.
4. Writes one JSON file per benchmark, plus a cross-benchmark
   `summary.json` that sums counts across all benchmarks per layer.
   Output is laid out as
   `<output-dir>/<model_safe>/<quant_safe>/<benchmark>.json` so
   multiple models and quants don't overwrite each other.

## Usage

```bash
# Single model (canonical OLMoE workload):
python examples/eval-moe/moe_expert_stats.py \
    --models allenai/OLMoE-1B-7B-0924-Instruct \
    --benchmarks mmlu bbh humaneval \
    --num-samples 256 \
    --max-tokens 256 \
    --output-dir ./expert_stats_olmoe

# Multiple models + a quantization method:
python examples/eval-moe/moe_expert_stats.py \
    --models allenai/OLMoE-1B-7B-0924-Instruct \
              LiteLLMs/Mixtral-8x22B-Instruct-v0.1 \
    --quant awq \
    --benchmarks mmlu bbh humaneval \
    --output-dir ./expert_stats
```

Run `--help` for the full list of options. The defaults match the
canonical OLMoE-1B-7B analysis workload.

```bash
python examples/eval-moe/moe_expert_stats.py --help
```

### CLI options

| Option                     | Default                             | Description                                                                                                                                                                                                      |
| -------------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--models`                 | `allenai/OLMoE-1B-7B-0924-Instruct` | One or more HuggingFace model ids. Cells are evaluated as the Cartesian product `--models` × `--quant`. A fresh `LLM` is instantiated per model so each gets its own KV cache and routed-experts capture buffer. |
| `--quant`                  | (empty = no quant)                  | One or more vLLM quantization methods (e.g. `awq`, `fp8`, `gptq`, `bitsandbytes`). Forwarded to `LLM(quantization=...)`. Pass an empty token to skip.                                                            |
| `--benchmarks`             | `mmlu bbh humaneval`                | Subset of `{mmlu, bbh, humaneval}` to run.                                                                                                                                                                       |
| `--num-samples`            | `256`                               | Number of prompts to sample per benchmark.                                                                                                                                                                       |
| `--max-tokens`             | `256`                               | Generation length per prompt.                                                                                                                                                                                    |
| `--output-dir`             | `./expert_stats`                    | Parent directory for the output tree. Each cell writes to `<output-dir>/<model_safe>/<quant_safe>/...`. See "Output schema" below.                                                                               |
| `--seed`                   | `0`                                 | Random seed for prompt sampling (deterministic with `temperature=0`).                                                                                                                                            |
| `--max-model-len`          | `4096`                              | vLLM `max_model_len` setting.                                                                                                                                                                                    |
| `--dtype`                  | `auto`                              | vLLM dtype.                                                                                                                                                                                                      |
| `--gpu-memory-utilization` | `0.92`                              | vLLM GPU memory fraction.                                                                                                                                                                                        |

<!-- markdownlint-disable MD060 -->

| `--enforce-eager` | (off) | Force eager-mode execution (skip CUDA graphs). |
| `--tensor-parallel-size` | `1` | Number of GPUs for vLLM's tensor-parallel engine. For multi-GPU runs, the Spartan wrapper loads an NCCL module, prints the NVLink topology, and forwards this flag automatically (`--gres=gpu:N` → `--tensor-parallel-size N`). |

<!-- markdownlint-enable MD060 -->

## Output schema

The output tree is laid out by `(model, quant)` cell, with each cell
holding one JSON per benchmark plus a cross-benchmark summary:

```text
<output-dir>/
├── allenai--OLMoE-1B-7B-0924-Instruct/
│   ├── default/                       # the empty-quants case collapses to "default"
│   │   ├── mmlu.json
│   │   ├── bbh.json
│   │   ├── humaneval.json
│   │   └── summary.json
│   └── awq/
│       ├── mmlu.json
│       ├── bbh.json
│       ├── humaneval.json
│       └── summary.json
└── LiteLLMs--Mixtral-8x22B-Instruct-v0.1/
    └── awq/
        ├── mmlu.json
        ├── bbh.json
        ├── humaneval.json
        └── summary.json
```

`<model_safe>` is the HF repo id with `/` replaced by `--` (matching the
sister project's `model_safe`). `<quant_safe>` is the quant tag as-is,
or `default` if `--quant` is empty.

For each benchmark the script writes
`<output-dir>/<model_safe>/<quant_safe>/<benchmark>.json`:

```json
{
  "model": "allenai/OLMoE-1B-7B-0924-Instruct",
  "quant": "awq",
  "benchmark": "mmlu",
  "num_samples": 256,
  "num_layers": 16,
  "num_experts": 64,
  "top_k": 8,
  "total_tokens": 12345,
  "expert_counts": {
    "0":  {"0": 12, "1": 7, "2": 0, ..., "63": 9},
    "1":  {"0": 11, "1": 8, "2": 1, ..., "63": 7},
    ...
    "15": {"0": 5,  "1": 3, "2": 0, ..., "63": 12}
  },
  "extra": {"subjects_seen": 57}
}
```

`expert_counts[layer_id][expert_id]` is the number of tokens (across all
prompts in this benchmark) that were routed to `expert_id` at
`layer_id`. `top_k` votes per token, so the sum of `expert_counts[L]`
over all experts equals `total_tokens * top_k` for every layer.

The file `<output-dir>/<model_safe>/<quant_safe>/summary.json`
aggregates counts across all benchmarks per layer for that cell.

## Hardware requirements

End-to-end execution requires a GPU. The script supports `tp=1` and
`pp=1` only (this is enforced by vLLM when `enable_return_routed_experts`
is enabled). Tested against an A-100. The capture buffer consumes
`num_blocks * block_size * num_layers * top_k` bytes of pinned CPU
memory — order-of-gigabytes for OLMoE-1B-7B at default block sizes.

The aggregation logic (`aggregate_expert_counts`) is pure NumPy and runs
on CPU; nothing in the analyzer requires a GPU to import or smoke-test.

## Limitations

- `enable_return_routed_experts` is incompatible with pipeline
  parallelism (`pp > 1`), context parallelism (`DCP > 1`, `PCP > 1`),
  and KV connectors — see `vllm/config/vllm.py`.
- Captured expert IDs are **logical** (pre-EPLB remap), so they
  correspond to the experts as the model was trained.
- BBH aggregates across all 27 sub-tasks; there is no per-sub-task
  breakdown.
- Per-token arrays are not saved — only per-layer counts. If you need
  per-token data, modify `aggregate_expert_counts` to also retain the
  raw `(seq_len, num_layers, top_k)` arrays.
- Multi-model / multi-quant evaluation runs sequentially in one
  process. Each cell instantiates a fresh `LLM` (and tears it down
  after its benchmarks), so two models never need to fit in VRAM at
  the same time. Wall time scales linearly with `len(--models) × len(--quant)`.
- Determinism: same `--seed` + same `--num-samples` yields byte-identical
  JSONs across runs (sampling is `temperature=0`).
- `--quant` only changes vLLM's runtime compute method. To run against
  a separately-quantized checkpoint (e.g. an AWQ repo on HF), pass the
  AWQ repo id as a separate `--models` entry — vLLM treats each as a
  distinct weight set.

## Further reading

- vLLM docs: [LLM API](https://docs.vllm.ai/en/latest/api/offline_inference/llm.html)
- vLLM docs: [Sampling parameters](https://docs.vllm.ai/en/latest/api/inference_params.html#sampling-parameters)
