# MoE Expert Activation Statistics

This directory contains a tool for analyzing how a Mixture-of-Experts (MoE)
language model routes tokens through its experts across different
benchmarks. For every layer of the model, it counts how many tokens were
routed to each individual expert plus the per-layer and adjacent-layer
co-activation pair counts, and writes the result to JSON.

The output schema is **bit-compatible** with the sister project's
[`llama-cpp-eval` `examples/eval-moe-*`](https://github.com/lilRaptor99/llama-cpp-eval)
C++ binaries. The same downstream post-processing toolchain
(`examples/eval-moe-overview/aggregate_overview.py`,
`jaccard_sweep_from_cpp.py`, `cross_quant_jaccard_sweep_from_cpp.py`)
consumes both vLLM-produced and llama.cpp-produced outputs uniformly, so
this analyzer can be used as a vLLM-native replacement for the C++ binaries
on any architecture vLLM supports (tensor parallel included).

This is **not** a quality / accuracy / performance benchmark — it only
captures routing statistics (the expert IDs chosen for each token). It is
useful for studying expert load balancing, layer specialization,
co-activation patterns, and cross-dataset / cross-quantization
top-K alignment.

## What it does

The script:

1. For every `(model, quant)` cell, loads the model with
   `vllm.LLM` and `enable_return_routed_experts=True`, which enables
   vLLM's built-in per-token, per-layer, per-topk expert-id
   capture on `CompletionOutput.routed_experts`.
2. Loads a configurable subset of five benchmarks and runs them through
   the model with `max_tokens=256`:
   - **MMLU** (`cais/mmlu`, subset `"all"`, 5-shot multiple-choice prompts)
   - **Big-Bench-Hard** (`Joschka/big_bench_hard`, all 27 sub-tasks)
   - **HumanEval** (`openai/openai_humaneval`, function-completion prompts)
   - **PopQA** (`akariasai/PopQA`, relation-type completion)
   - **INCLUDE** (`CohereLabs/include-base-44`, 44 languages × multiple
     domains, 5-shot multiple-choice)
3. Aggregates the captured expert IDs into a `(num_layers, num_experts)`
   int64 count matrix **per row** (one record per MMLU subject, BBH
   sub-task, HumanEval task, PopQA relation type, or INCLUDE
   (language, domain) group). Also computes the per-row and aggregate
   `[L, E, E]` intra-layer and `[L-1, E, E]` adjacent-layer pair counts.
4. Writes one JSON file per `(model, quant, benchmark)` cell, with the
   schema documented below. Output is laid out as
   `<output-dir>/<model_safe>/<quant_safe>/moe-<bench>/expert_counts.json`.

## Usage

```bash
# Single model (canonical OLMoE workload):
python examples/eval-moe/moe_expert_stats.py \
    --models allenai/OLMoE-1B-7B-0924-Instruct \
    --benchmarks mmlu bbh humaneval popqa include \
    --num-samples 256 \
    --max-tokens 256 \
    --output-dir /data/scratch/projects/uom00014/vllm/results

# Multiple models + a quantization method:
python examples/eval-moe/moe_expert_stats.py \
    --models allenai/OLMoE-1B-7B-0924-Instruct \
              LiteLLMs/Mixtral-8x22B-Instruct-v0.1 \
    --quant awq \
    --benchmarks mmlu bbh humaneval popqa include \
    --output-dir /data/scratch/projects/uom00014/vllm/results
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
| `--benchmarks`             | `mmlu bbh humaneval`                | Subset of `{mmlu, bbh, humaneval, popqa, include}` to run.                                                                                                                                                      |
| `--num-samples`            | `256`                               | Sample budget per benchmark. For MMLU + BBH + INCLUDE this is a per-(subject/sub-task/langdom) budget; for HumanEval + PopQA it is a per-(task/prop) budget.                                                            |
| `--max-tokens`             | `256`                               | Autoregressive decode cap per prompt. Set 0 for prefill-only.                                                                                                                                                    |
| `--n-shots`                | `5`                                 | Few-shot exemplars from the per-dataset dev pool (mmlu, bbh, include). Set 0 for zero-shot.                                                                                                                       |
| `--output-dir`             | `./expert_stats`                    | Parent directory for the output tree. Each cell writes to `<output-dir>/<model_safe>/<quant_safe>/moe-<bench>/expert_counts.json`. See "Output schema" below.                                                     |
| `--seed`                   | `0`                                 | Random seed for prompt sampling (deterministic with `temperature=0`).                                                                                                                                            |
| `--max-model-len`          | `4096`                              | vLLM `max_model_len` setting.                                                                                                                                                                                    |
| `--dtype`                  | `auto`                              | vLLM dtype.                                                                                                                                                                                                      |
| `--gpu-memory-utilization` | `0.92`                              | vLLM GPU memory fraction.                                                                                                                                                                                        |

<!-- markdownlint-disable MD060 -->

| `--enforce-eager` | (off) | Force eager-mode execution (skip CUDA graphs). |
| `--tensor-parallel-size` | `1` | Number of GPUs for vLLM's tensor-parallel engine. For multi-GPU runs, the Spartan wrapper loads an NCCL module, prints the NVLink topology, and forwards this flag automatically (`--gres=gpu:N` → `--tensor-parallel-size N`). |

<!-- markdownlint-enable MD060 -->

## Output schema

The output tree is laid out by `(model, quant, benchmark)` cell:

```text
<output-dir>/
├── allenai--OLMoE-1B-7B-0924-Instruct/
│   ├── default/                                  # the empty-quants case collapses to "default"
│   │   ├── moe-mmlu/expert_counts.json
│   │   ├── moe-bbh/expert_counts.json
│   │   ├── moe-humaneval/expert_counts.json
│   │   ├── moe-popqa/expert_counts.json
│   │   └── moe-include/expert_counts.json
│   └── awq/
│       ├── moe-mmlu/expert_counts.json
│       └── ...
└── openai--gpt-oss-120b/                          # 2-GPU TP example
    └── default/
        ├── moe-mmlu/expert_counts.json
        └── ...
```

`<model_safe>` is the HF repo id with `/` replaced by `--` (matching the
sister project's `model_safe`). `<quant_safe>` is the quant tag as-is,
or `default` if `--quant` is empty. `moe-<bench>` is `moe-mmlu`,
`moe-bbh`, `moe-humaneval`, `moe-popqa`, or `moe-include`.

The expected aggregator outputs (per `(model, quant)` cell, written by
`examples/eval-moe-overview/aggregate_overview.py`):

```text
<output-dir>/<model_safe>/<quant_safe>/overall/
├── routing_heatmap_overview.png
├── routing_heatmap_overview_highlighted.png
├── top_experts_bars.png
├── top_experts.json
├── counts_total_overview.json
└── metadata_overview.json
```

And the Jaccard sweep outputs (per `(model, quant)` cell + per model):

```text
<output-dir>/<model_safe>/<quant_safe>/jaccard_sweep/
├── summary.csv
├── jaccard_pairwise.csv
├── jaccard_pairwise.png
├── jaccard_pairwise_K{N}.png  (for the 2x and 3x N_expert_used values)
├── jaccard_global.csv
├── jaccard_global.png
└── README.json

<output-dir>/<model_safe>/cross_quant_jaccard_sweep/
├── summary.csv
├── jaccard_pairwise.csv
├── jaccard_pairwise.png
├── jaccard_pairwise_K{N}.png
├── jaccard_global.csv
├── jaccard_global.png
└── README.json
```

### Per-benchmark JSON schema (`moe-<bench>/expert_counts.json`)

```jsonc
{
  "model":      "allenai/OLMoE-1B-7B-0924-Instruct",
  "model_arch": { "name": "olmoe", "n_layer": 16, "n_expert": 64, "n_expert_used": 8 },
  "config":     { "questions_per_subject": 256,
                  "n_shot": 5,
                  "few_shot_pool": "cais/mmlu dev split",
                  "prompt_format": "few_shot_chat" },
  "totals":     { "subjects_run": 57, "questions_total": 256, "tokens_total": 207358 },

  // Dataset-wide aggregate (sum of per-row counts). Optional in the
  // llama-cpp schema; always emitted by the vLLM analyzer.
  "aggregate": {
    "marginal_expert_counts": [[L, E] int64],
    "intra_pair_counts":      [[L, E, E] int64],
    "adjacent_pair_counts":   [[L-1, E, E] int64]
  },

  // Per-row record. The top-level key matches the benchmark:
  //   "subjects"   (mmlu, bbh)         - one entry per MMLU subject or BBH sub-task
  //   "tasks"      (humaneval)         - one entry per problem, key = "HumanEval/<task_id>"
  //   "props"      (popqa)             - one entry per relation type, key = the relation
  //   "by_langdom" (include)           - one entry per (language, domain), key = "<language>::<domain>"
  "subjects": {
    "abstract_algebra": {
      "questions": 4,
      "n_tokens": 8142,                       // mmlu + bbh (prefill only)
      "layer_expert_counts":   [[L, E] int64],
      "intra_pair_counts":     [[L, E, E] int64],
      "adjacent_pair_counts":  [[L-1, E, E] int64]
    }
  },
  // OR for humaneval / popqa / include:
  "tasks": {
    "HumanEval/0": {
      "questions": 1,
      "n_tokens_prefill":   312,              // humaneval + popqa + include
      "n_tokens_generated": 256,              // actual decoded, not the cap
      "layer_expert_counts":  [[L, E] int64],
      "intra_pair_counts":    [[L, E, E] int64],
      "adjacent_pair_counts": [[L-1, E, E] int64]
    }
  }
}
```

Field semantics:

| Field                                   | Shape         | Counts one increment per                                          |
| --------------------------------------- | ------------- | ----------------------------------------------------------------- |
| `layer_expert_counts[L][e]`             | `[L, E]`      | top-k slot in layer L where expert e fired                        |
| `intra_pair_counts[L][e_i][e_j]`        | `[L, E, E]`   | token where e_i and e_j both fired in layer L (`k × k` slot pairs) |
| `adjacent_pair_counts[L][e_i][e_j]`     | `[L-1, E, E]` | token where e_i fired in L and e_j fired in L+1 at the same position |

Identity invariants (verified by the post-processing):
- `layer_expert_counts[L][e] * k == sum_{e'} intra_pair_counts[L][e][e']` (each token's k top-k slots in L pair with k in L)
- `sum_{e_i, e_j} adjacent_pair_counts[L][e_i][e_j] == k * k * tokens_total` per layer pair
- `aggregate.X == sum_{rows} rows[*].X` for every X

The schema is the same one the sister project's
`llama-eval-moe-mmlu` / `-humaneval` / `-popqa` / `-include` C++ binaries
emit, so the same `aggregate_overview.py` / `jaccard_sweep_from_cpp.py`
/ `cross_quant_jaccard_sweep_from_cpp.py` scripts can read both.

## Spartan HPC

The `spartan/` directory wraps the analyzer as a SLURM job:

- `download-models.sh` — pre-stage HF model weights (login node)
- `download-datasets.sh` — pre-stage the 5 benchmark datasets (login node)
- `vllm-moe-eval.sbatch` — build the venv + run the analyzer + run
  post-processing (GPU node)

See `spartan/README.md` for the full workflow. Briefly:

```bash
# 1. login node: pre-download
cd /data/gpfs/projects/uom00014/vllm
bash examples/eval-moe/spartan/download-models.sh --models openai/gpt-oss-120b
bash examples/eval-moe/spartan/download-datasets.sh --datasets mmlu bbh humaneval popqa include

# 2. submit the GPU job
sbatch examples/eval-moe/spartan/vllm-moe-eval.sbatch
# or, to opt in to the post-processing visualisations + Jaccard sweeps:
sbatch --export=ALL,SKIP_PYTHON_PLOTS=0 examples/eval-moe/spartan/vllm-moe-eval.sbatch
```

The sbatch defaults to writing `<output-dir>/<model_safe>/<quant_safe>/moe-<bench>/expert_counts.json`
per cell, and (when `SKIP_PYTHON_PLOTS=0`) runs the sister-project's
`aggregate_overview.py` / `jaccard_sweep_from_cpp.py` /
`cross_quant_jaccard_sweep_from_cpp.py` scripts on the resulting tree
to produce the overview heatmaps + cross-dataset / cross-quant Jaccard
sweep artefacts.

## Hardware requirements

End-to-end execution requires a GPU. The script supports `tp=1` and
`pp=1` only (this is enforced by vLLM when `enable_return_routed_experts` is
enabled, see `vllm/config/vllm.py`). Tested against an A-100. The
capture buffer consumes `num_blocks * block_size * num_layers * top_k`
bytes of pinned CPU memory — order-of-gigabytes for OLMoE-1B-7B at
default block sizes.

The aggregation logic (`aggregate_expert_counts`) is pure NumPy and
runs on CPU; nothing in the analyzer requires a GPU to import or
smoke-test. A pure-NumPy sanity check (with a fake `RequestOutput` of
shape `(T, L, K)` int64) confirms the marginal + intra + adjacent
pair counts match the expected `(tokens × k)` totals.

## Limitations

- `enable_return_routed_experts` is incompatible with pipeline
  parallelism (`pp > 1`), context parallelism (`DCP > 1`, `PCP > 1`),
  and KV connectors — see `vllm/config/vllm.py`.
- Captured expert IDs are **logical** (pre-EPLB remap), so they
  correspond to the experts as the model was trained.
- BBH aggregates across all 27 sub-tasks, with one JSON record per
  sub-task that appears in the 256-sample. There is no separate
  per-question record.
- Per-token arrays are not saved — only per-layer counts and pair
  counts. If you need per-token data, modify `aggregate_expert_counts`
  to also retain the raw `(seq_len, num_layers, top_k)` arrays.
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
- The analyzer does **not** score answers (no `match_rate` /
  `n_correct` fields). This is intentional — the routing stats are the
  only output. The sister llama-cpp-eval C++ binaries do score
  humaneval + popqa + include; we keep the analyzer focused on
  routing.

## Further reading

- vLLM docs: [LLM API](https://docs.vllm.ai/en/latest/api/offline_inference/llm.html)
- vLLM docs: [Sampling parameters](https://docs.vllm.ai/en/latest/api/inference_params.html#sampling-parameters)
- Sister project: [`llama-cpp-eval` `examples/eval-moe-mmlu/README.md`](https://github.com/lilRaptor99/llama-cpp-eval/tree/eval-moe/examples/eval-moe-mmlu) — the C++ binary this analyzer's output schema is bit-compatible with.
- Sister project: [`llama-cpp-eval` `examples/eval-moe-overview/README.md`](https://github.com/lilRaptor99/llama-cpp-eval/tree/eval-moe/examples/eval-moe-overview) — the post-processing toolchain this analyzer's output is designed to feed.
