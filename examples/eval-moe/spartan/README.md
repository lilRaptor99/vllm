# Spartan HPC wrapper for `examples/eval-moe/moe_expert_stats.py`

Two artefacts that wrap the per-layer expert activation statistics
analyzer so it runs end-to-end on the Unimelb Spartan HPC under SLURM.

| File                                           | Where it runs               | Purpose                                                                                                                        |
| ---------------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| [`download-models.sh`](download-models.sh)     | Login node                  | Pre-populate the HF model cache (MoE safetensors repos the analyzer will load)                                                 |
| [`download-datasets.sh`](download-datasets.sh) | Login node                  | Pre-populate the HF dataset cache (`cais/mmlu`, `Joschka/big_bench_hard`, `openai/openai_humaneval`) the analyzer will consume |
| [`vllm-moe-eval.sbatch`](vllm-moe-eval.sbatch) | GPU node (`gpu-a100-short`) | Build venv + run the analyzer on the OLMoE-1B-7B model (or any MoE model)                                                      |
| [`README.md`](README.md)                       | —                           | This file                                                                                                                      |

The analyzer (`moe_expert_stats.py`) is unchanged at its core; the
wrapper just stages the data it needs and submits a SLURM job that
loads vLLM into a fresh venv on the scratch volume.

---

## Prerequisites

| What                                                                                                               | Why                                                               | Where to set it up                                                                                                                                                                                                |
| ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Spartan account + `gpu-a100-short` group membership                                                                | Run on A100 nodes                                                 | Spartan onboarding                                                                                                                                                                                                |
| This vllm checkout at `$REPO_ROOT` (default `/data/gpfs/projects/uom00014/vllm`)                                   | The analyzer under test                                           | `git clone` to the GPFS project path                                                                                                                                                                              |
| `/data/gpfs/projects/uom00014/vllm` (REPO_ROOT) and `/data/scratch/projects/uom00014/vllm` (SCRATCH_BASE) writable | venv + cache + results storage                                    | Project quota path on Spartan                                                                                                                                                                                     |
| `CUDA/12.4.1` + `Python/3.11.3` + `GCCcore/11.3.0` Lmod modules                                                    | Compiles + runs the analyzer                                      | Already pre-selected as defaults; override via `*_MODULE` env vars if your partition shows different versions. Note that `GCCcore/11.3.0` (the older compiler family) is required, not the newer `GCC/13.3.0`.    |
| `huggingface_hub` Python package on the **login node**                                                             | Pre-downloads the HF model weights (5-250 GB safetensors bundles) | `pip install --user huggingface_hub` on the login node (or just let `download-models.sh` install it for you). The .sbatch installs `vllm` + `numpy` + `datasets` into a fresh venv on the GPU node automatically. |
| `datasets` Python package on the **login node**                                                                    | Pre-downloads the 3 HF datasets                                   | `pip install --user datasets` on the login node (or just let `download-datasets.sh` install it for you). The .sbatch installs `vllm` + `numpy` + `datasets` into a fresh venv on the GPU node automatically.      |

The default `MODELS_OVERRIDE` is a public HF repo (`allenai/OLMoE-1B-7B-0924-Instruct`), so `HF_TOKEN` is not required. If you swap to a gated repo later, export `HF_TOKEN` before invoking either script.

---

## Workflow

### 1. Login node — pre-download the eval datasets

> **This step is required.** The Spartan GPU compute nodes are
> firewalled off from the public internet AND don't ship the `datasets`
> Python package. Skipping this step will cause the analyzer's
> `load_dataset(...)` calls to fail with `ModuleNotFoundError: No
module named 'datasets'` or `ConnectionError` mid-run.

```bash
# Default: all 3 datasets (mmlu, bbh, humaneval). Total ~280 MB.
cd <repo>/examples/eval-moe
bash spartan/download-datasets.sh

# Just one (e.g. for a smoke test):
bash spartan/download-datasets.sh --datasets mmlu

# Show what each dataset is without downloading:
bash spartan/download-datasets.sh --list

# Custom scratch base:
bash spartan/download-datasets.sh --scratch-base /data/scratch/projects/uom00014/vllm
```

The script auto-installs `datasets` to `~/.local` if it's missing,
then writes each dataset to
`${SCRATCH_BASE}/datasets/<name>/<split>/` as HuggingFace's native
Arrow-on-disk format. It's idempotent — re-runs skip any dataset whose
`<name>/<split>/` directory is already populated.

### 1b. Login node — pre-download the model weights (recommended)

> **Optional but strongly recommended.** Skipping this is fine if the
> GPU compute node can reach `huggingface.co` — `vllm.LLM(model=...)`
> will pull the weights on first use. But models are 5-250 GB; pulling
> them during compute-quota hours wastes that quota on I/O. Pre-stage
> on the login node where HF egress is unrestricted.

```bash
# Default: 5 canonical MoE models (incl. the analyzer's default).
# Worst-case ~300 GB total. Will probe each repo's safetensors
# metadata and ask for confirmation unless --yes.
cd <repo>/examples/eval-moe
bash spartan/download-models.sh

# Just the analyzer's default model (smallest, fastest):
bash spartan/download-models.sh --models allenai/OLMoE-1B-7B-0924-Instruct

# Skip the confirmation prompt:
bash spartan/download-models.sh --yes

# Pull only the weights + metadata (skip pytorch_model.bin, .gguf, etc.):
bash spartan/download-models.sh \
    --include "*.safetensors" \
    --include "*.json" \
    --include "tokenizer*" \
    --include "*.txt"

# Show what would be downloaded without doing it (also probes sizes):
bash spartan/download-models.sh --list

# Custom scratch base:
bash spartan/download-models.sh --scratch-base /data/scratch/projects/uom00014/vllm
```

The script auto-installs `huggingface_hub` to `~/.local` if it's
missing, then writes each repo to
`${SCRATCH_BASE}/hf_cache/hub/models--<org>--<name>/snapshots/<rev>/`
(the standard HF hub layout that `vllm.LLM(model=...)` resolves).
It's idempotent — re-runs skip any repo whose `config.json` is already
cached. `HF_TOKEN` is forwarded to `huggingface_hub` for gated repos.

### 2. Submit the GPU job

```bash
# Default: 1 A100 GPU, 8 CPUs, 120 GB RAM, 4-day wall clock.
# Default modules: CUDA/12.4.1, Python/3.11.3, GCCcore/11.3.0.
# Default analyzer: --models allenai/OLMoE-1B-7B-0924-Instruct
#                   --benchmarks mmlu bbh humaneval
#                   --num-samples 256 --max-tokens 256
cd <repo>/examples/eval-moe
sbatch spartan/vllm-moe-eval.sbatch

# Override modules if `module avail` shows different versions:
CUDA_MODULE=CUDA/12.4.1 PYTHON_MODULE=Python/3.11.3 GCC_MODULE=GCCcore/11.3.0 \
    sbatch spartan/vllm-moe-eval.sbatch

# Run a different model (Mixtral, DeepSeek-MoE, gpt-oss-120b, etc.):
sbatch --export=ALL,MODELS_OVERRIDE=LiteLLMs/Mixtral-8x22B-Instruct-v0.1 \
    spartan/vllm-moe-eval.sbatch

# Run multiple models + a quantization method (Cartesian product
# yields 2 model x 1 quant = 2 cells):
sbatch --export=ALL,MODELS_OVERRIDE="allenai/OLMoE-1B-7B-0924-Instruct LiteLLMs/Mixtral-8x22B-Instruct-v0.1",QUANTS_OVERRIDE=awq \
    spartan/vllm-moe-eval.sbatch

# Smaller subset for a smoke test:
sbatch --export=ALL,BENCHMARKS=mmlu,NUM_SAMPLES=32,MAX_TOKENS=64 \
    spartan/vllm-moe-eval.sbatch
```

If the venv install fails on the GPU node with a network error (pypi
unreachable), the .sbatch will exit with `[fatal] pip install failed`
and a clear stderr dump. The preflight installs to the venv under
`${SCRATCH_BASE}/venv`, which is cached between submissions; the
second run skips the heavy `pip install vllm` step entirely.

The .sbatch creates `${SCRATCH_BASE}/venv` on first run, installs
`vllm` + `numpy` + `datasets` into it via `pip`, then invokes
`examples/eval-moe/moe_expert_stats.py` with the configured argv.

### 3. Monitor

```bash
# Job status
squeue -j $JOB_ID

# Top-level job log
tail -f spartan/logs/vllm-moe-eval-$JOB_ID.out

# Per-cell analyzer output (if you tee'd it; the analyzer itself writes
# the JSONs to ${OUTPUT_DIR}/<benchmark>.json)
```

---

## Smoke testing

The full default run (OLMoE-1B-7B + 3 benchmarks × 256 prompts ×
256 generation tokens) takes ~30-60 minutes on a single A100. To
verify the pipeline end-to-end on a small subset:

```bash
sbatch \
  --export=ALL,BENCHMARKS=mmlu,NUM_SAMPLES=8,MAX_TOKENS=8,MAX_MODEL_LEN=1024 \
  spartan/vllm-moe-eval.sbatch
```

You should see:

- `nvidia-smi` print one A100 line in the banner.
- `[info] venv at <scratch>/venv created` (or reused) followed by
  `[info] python deps: vllm=ok numpy=ok datasets=ok` in the preflight.
- `[info] running: python examples/eval-moe/moe_expert_stats.py --models allenai/OLMoE-1B-7B-0924-Instruct ...`
- `[info] === cell: model=..., quant=<none> ===` at the start of each cell.
- Per-benchmark lines: `[info] Running benchmark mmlu with 8 prompts...`
  `[info] Wrote <OUTPUT_DIR>/allenai--OLMoE-1B-7B-0924-Instruct/default/mmlu.json with <N> tokens across 16 layers.`
- `[info] Wrote <OUTPUT_DIR>/summary.json` at the end.

---

## Env-var overrides

All variables are optional. Set them before `sbatch` (or pass via
`--export=...`).

| Variable             | Default                                                                            | Notes                                                                                                                                                                                                                               |
| -------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `REPO_ROOT`          | `/data/gpfs/projects/uom00014/vllm`                                                | Where the repo is cloned                                                                                                                                                                                                            |
| `SCRATCH_BASE`       | `/data/scratch/projects/uom00014/vllm` (or `${SCRATCH}/vllm` if `$SCRATCH` is set) | Root for venv / datasets / results / hf_cache                                                                                                                                                                                       |
| `CUDA_MODULE`        | `CUDA/12.4.1`                                                                      | Lmod module name (canonical gpu-a100-short default)                                                                                                                                                                                 |
| `PYTHON_MODULE`      | `Python/3.11.3`                                                                    | Lmod module name (canonical gpu-a100-short default)                                                                                                                                                                                 |
| `GCC_MODULE`         | `GCCcore/11.3.0`                                                                   | Lmod module name; **must be the older compiler family** that Python/3.11.3 is built against, not the newer GCC/13.3.0                                                                                                               |
| `MODELS_OVERRIDE`    | `allenai/OLMoE-1B-7B-0924-Instruct`                                                | Space-separated HF model ids. Each model gets its own LLM instance and routed-experts capture buffer.                                                                                                                               |
| `QUANTS_OVERRIDE`    | `<unset>`                                                                          | Space-separated vLLM quantization methods (e.g. `awq`, `fp8`, `gptq`, `bitsandbytes`). Cells are evaluated as `MODELS_OVERRIDE` × `QUANTS_OVERRIDE` (Cartesian product). Pass an empty token (or leave unset) to skip quantization. |
| `BENCHMARKS`         | `mmlu bbh humaneval`                                                               | Space-separated subset                                                                                                                                                                                                              |
| `NUM_SAMPLES`        | `256`                                                                              | Prompts per benchmark                                                                                                                                                                                                               |
| `MAX_TOKENS`         | `256`                                                                              | Generation length per prompt                                                                                                                                                                                                        |
| `MAX_MODEL_LEN`      | `4096`                                                                             | vLLM `max_model_len` setting                                                                                                                                                                                                        |
| `GPU_MEM_UTIL`       | `0.92`                                                                             | vLLM `gpu_memory_utilization`                                                                                                                                                                                                       |
| `ENFORCE_EAGER`      | `<unset>`                                                                          | If set, pass `--enforce-eager` to `LLM(...)` (skip CUDA graphs)                                                                                                                                                                     |
| `OUTPUT_DIR`         | `${SCRATCH_BASE}/results`                                                          | Where to write per-benchmark JSON files                                                                                                                                                                                             |
| `SEED`               | `0`                                                                                | Sampler + sampling seed                                                                                                                                                                                                             |
| `HF_TOKEN`           | `<unset>`                                                                          | Optional; for gated repos                                                                                                                                                                                                           |
| `SKIP_DATASET_CHECK` | `0`                                                                                | `1` = skip the dataset preflight warning                                                                                                                                                                                            |

---

## Results tree

Every `(model, quant)` cell the analyzer runs writes its own subtree:

```text
${OUTPUT_DIR}/
└── <model_safe>/
    └── <quant_safe>/
        ├── mmlu.json         # per-layer expert counts for MMLU
        ├── bbh.json          # per-layer expert counts for BBH
        ├── humaneval.json    # per-layer expert counts for HumanEval
        └── summary.json      # cross-benchmark per-layer totals
```

For the default 1-model / no-quant run the layout collapses to
`${OUTPUT_DIR}/allenai--OLMoE-1B-7B-0924-Instruct/default/`. Running
`--models A B --quant awq fp8` produces a 2 × 2 = 4-cell tree:

```text
${OUTPUT_DIR}/
├── A_safe/awq/    ├── A_safe/fp8/
├── B_safe/awq/    └── B_safe/fp8/
```

`<model_safe>` is the HF repo id with `/` replaced by `--`
(matching the sister project's convention). `<quant_safe>` is the
quant tag as-is, or `default` when `--quant` is empty.

Each per-benchmark JSON has the shape:

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
    "0":  {"0": 12, "1": 7, ..., "63": 9},
    "1":  {"0": 11, "1": 8, ..., "63": 7},
    ...
    "15": {"0": 5,  "1": 3, ..., "63": 12}
  },
  "extra": {"subjects_seen": 57}
}
```

`expert_counts[layer_id][expert_id]` is the number of tokens routed
to `expert_id` at `layer_id` across all prompts in the benchmark.
`top_k` votes per token, so the sum of `expert_counts[L]` over all
experts equals `total_tokens * top_k` for every layer.

See [`examples/eval-moe/README.md`](../README.md) for the full schema
and usage notes.

---

## Resuming / wiping

The analyzer overwrites the per-benchmark JSONs each run, so re-running
with the same `--benchmarks` flag simply replaces the corresponding
files. To resume after a pre-emption, re-submit:

```bash
sbatch spartan/vllm-moe-eval.sbatch
```

To start fresh:

```bash
rm -rf "${SCRATCH_BASE}/"venv,datasets,results,hf_cache
```

To wipe just the HF model cache (forces re-download but keeps
results):

```bash
rm -rf "${SCRATCH_BASE}/hf_cache"
```

To wipe just the venv (forces re-install on next run):

```bash
rm -rf "${SCRATCH_BASE}/venv"
```

---

## Troubleshooting

| Symptom                                                                                                                           | Likely cause                                                                                                                  | Fix                                                                                                                                                                                                                                                                                                               |
| --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `module: command not found`                                                                                                       | Lmod not in `.bashrc` on the GPU node                                                                                         | `source /usr/local/lmod/lmod/init/bash` before `module` calls, or trust the .sbatch's `module purge`                                                                                                                                                                                                              |
| `Lmod has detected the following error: The following module(s) are unknown: "CUDA/12.X"` (or similar)                            | The hard-coded default module name doesn't exist on `gpu-a100-short` (versions change over time)                              | `module avail cuda/python/gcc` on a `gpu-a100-short` node, then re-submit with the exact name                                                                                                                                                                                                                     |
| `Lmod has detected the following error: These module(s) or extension(s) exist but cannot be loaded as requested: "Python/3.11.3"` | The Python module isn't built against the GCC compiler you have loaded (e.g. you used `GCC/13.3.0` but need `GCCcore/11.3.0`) | `module spider Python/3.11.3` to see the required parent compiler, then re-submit with `GCC_MODULE=<spider-suggested-gcc>` (default is `GCCcore/11.3.0`)                                                                                                                                                          |
| `[fatal] CUDA_MODULE is empty`                                                                                                    | Forgot to export the module env vars                                                                                          | `module avail cuda/python/gcc` on `gpu-a100-short`, then re-submit with all three exported                                                                                                                                                                                                                        |
| `python3: command not found` after `module load`                                                                                  | Wrong PYTHON_MODULE                                                                                                           | `module avail python` on `gpu-a100-short` and set `PYTHON_MODULE`                                                                                                                                                                                                                                                 |
| `gcc: command not found` after `module load`                                                                                      | Wrong GCC_MODULE                                                                                                              | `module avail gcc` on `gpu-a100-short` and set `GCC_MODULE` (default is `GCCcore/11.3.0`)                                                                                                                                                                                                                         |
| `ggml_cuda_init: no CUDA devices found`                                                                                           | `CUDA_VISIBLE_DEVICES` empty or wrong GPU count                                                                               | Check `squeue -j $JOBID -o "%Gres"`; request `--gres=gpu:N` to match                                                                                                                                                                                                                                              |
| `[fatal] datasets not staged on scratch` warning at job start                                                                     | Login-node `download-datasets.sh` wasn't run (or didn't finish)                                                               | Run on the login node first: `bash spartan/download-datasets.sh --datasets <names>`. The .sbatch will still proceed; the analyzer will fail mid-run.                                                                                                                                                              |
| `ModuleNotFoundError: No module named 'datasets'` mid-run                                                                         | GPU node can't reach pypi + `download-datasets.sh` wasn't run                                                                 | Run on the login node first: `bash spartan/download-datasets.sh`                                                                                                                                                                                                                                                  |
| `ModuleNotFoundError: No module named 'vllm'`                                                                                     | venv install failed; .sbatch's pip retry gave up                                                                              | Re-submit; if it keeps failing, install manually after the job has allocated: `source <SCRATCH>/venv/bin/activate && pip install vllm numpy datasets`                                                                                                                                                             |
| `ConnectionError` from `datasets.load_dataset`                                                                                    | GPU node can't reach HF hub                                                                                                   | Same fix: pre-stage via login-node `download-datasets.sh` (the script auto-installs `datasets` and downloads the splits to scratch). Note that the **model weights** still need network access on the GPU node; if the GPU node has full outbound firewall on pypi+huggingface then the analyzer can't run there. |
| `ConnectionError` from `LLM(model=...)` weight download                                                                           | GPU node can't reach HF hub, AND `download-models.sh` wasn't run                                                              | Run on the login node first: `bash spartan/download-models.sh --models <id>`. The .sbatch reads from `${HF_CACHE}`, which is where `download-models.sh` writes.                                                                                                                                                   |
| OOM / `Killed` in job log                                                                                                         | 120B model + activations exceed `--mem`                                                                                       | Raise `--mem` (A100 node has 495 GB total) or use a smaller model/quant                                                                                                                                                                                                                                           |
| Analyzer hangs at first `LLM(...)` call                                                                                           | Cold-start weight download from HF hub is slow on shared bandwidth                                                            | First run only; subsequent runs reuse `${HF_CACHE}`; pre-stage via `download-models.sh` to skip entirely                                                                                                                                                                                                          |

---

## File map

```text
examples/eval-moe/
├── README.md                    # analyzer docs
├── moe_expert_stats.py          # the analyzer (unchanged by the spartan wrapper)
└── spartan/
    ├── README.md                # this file
    ├── download-models.sh       # login-node pre-download (HF model weights)
    ├── download-datasets.sh     # login-node pre-download (HF datasets)
    └── vllm-moe-eval.sbatch     # SLURM job (gpu-a100-short, 1 GPU, 4 days)
```

Related (not in this directory):

- [`examples/eval-moe/moe_expert_stats.py`](../moe_expert_stats.py) — the
  analyzer that the .sbatch invokes. Holds the `BENCHMARK_LOADERS`
  dict + the `DEFAULT_*` constants; the .sbatch's CLI flags
  (`--models`, `--quant`, `--benchmarks`, `--num-samples`,
  `--max-tokens`, `--max-model-len`, `--gpu-memory-utilization`,
  `--enforce-eager`, `--output-dir`, `--seed`, `--dtype`) all map 1:1
  to the analyzer's argparse. The analyzer evaluates the Cartesian
  product of `--models` × `--quant`, instantiating a fresh `LLM` per
  model so each cell has its own KV cache + routed-experts capture
  buffer.
