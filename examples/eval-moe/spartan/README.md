# Spartan HPC wrapper for `examples/eval-moe/moe_expert_stats.py`

Three artefacts that wrap the per-layer expert activation statistics
analyzer so it runs end-to-end on the Unimelb Spartan HPC under SLURM.

| File                                           | Where it runs               | Purpose                                                                                                                        |
| ---------------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| [`download-models.sh`](download-models.sh)     | Login node                  | Pre-populate the HF model cache (MoE safetensors repos the analyzer will load)                                                 |
| [`download-datasets.sh`](download-datasets.sh) | Login node                  | Pre-populate the 5 benchmark datasets (`cais/mmlu`, `Joschka/big_bench_hard`, `openai/openai_humaneval`, `akariasai/PopQA`, `CohereLabs/include-base-44`) the analyzer will consume |
| [`download-popqa.py`](download-popqa.py)       | Login node                  | PopQA pre-downloader (called by `download-datasets.sh`; mirrors the sister project's `download_popqa.py` for byte-compatible JSONL output) |
| [`download-include.py`](download-include.py)   | Login node                  | INCLUDE pre-downloader (called by `download-datasets.sh`; mirrors the sister project's `download_include.py`)                  |
| [`vllm-moe-eval.sbatch`](vllm-moe-eval.sbatch) | GPU node (`gpu-a100-short`) | Build venv + run the analyzer on the OLMoE-1B-7B model (or any MoE model) + optionally run the sister-project's `aggregate_overview.py` / Jaccard sweep tools |
| [`README.md`](README.md)                       | —                           | This file                                                                                                                      |

The analyzer (`moe_expert_stats.py`) emits JSON in the same schema as the
sister project's `llama-eval-moe-*` C++ binaries. The `.sbatch` optionally
runs the sister-project's `aggregate_overview.py` /
`jaccard_sweep_from_cpp.py` / `cross_quant_jaccard_sweep_from_cpp.py`
against the vLLM-produced tree — both projects' outputs are
schema-compatible, so the same toolchain produces the overview
heatmaps, top-K bar charts, and cross-dataset / cross-quantization
Jaccard sweep PNGs + CSVs.

---

## Prerequisites

| What                                                                                                               | Why                                                               | Where to set it up                                                                                                                                                                                                |
| ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Spartan account + `gpu-a100-short` group membership                                                                | Run on A100 nodes                                                 | Spartan onboarding                                                                                                                                                                                                |
| This vllm checkout at `$REPO_ROOT` (default `/data/gpfs/projects/uom00014/vllm`)                                   | The analyzer under test                                           | `git clone` to the GPFS project path                                                                                                                                                                              |
| `/data/gpfs/projects/uom00014/vllm` (REPO_ROOT) and `/data/scratch/projects/uom00014/vllm` (SCRATCH_BASE) writable | venv + cache + results storage                                    | Project quota path on Spartan                                                                                                                                                                                     |
| `CUDA/12.4.1` + `NCCL/2.22.3-CUDA-12.4.1` + `Python/3.11.3` + `GCCcore/11.3.0` Lmod modules                               | Compiles + runs the analyzer                                      | Already pre-selected as defaults; override via `*_MODULE` env vars if your partition shows different versions. Note that `GCCcore/11.3.0` (the older compiler family) is required, not the newer `GCC/13.3.0`. NCCL is only required for multi-GPU runs (auto-probed, must match the loaded CUDA). |
| `huggingface_hub` Python package on the **login node**                                                             | Pre-downloads the HF model weights (5-250 GB safetensors bundles) | `pip install --user huggingface_hub` on the login node (or just let `download-models.sh` install it for you). The .sbatch installs `vllm` + `numpy` + `datasets` into a fresh venv on the GPU node automatically. |
| `datasets` Python package on the **login node**                                                                    | Pre-downloads the 5 benchmark datasets                           | `pip install --user datasets` on the login node (or just let `download-datasets.sh` install it for you). The .sbatch installs `vllm` + `numpy` + `datasets` into a fresh venv on the GPU node automatically.      |
| (Optional) sister `llama-cpp-eval` repo at `/data/projects/uom00014/llama-cpp-eval`                                  | Post-processing (aggregate overview + Jaccard sweeps)            | Already on disk in this project; the sbatch calls its `examples/eval-moe-overview/` scripts directly. If the directory isn't there, set `SKIP_PYTHON_PLOTS=1` (the default).                                  |

The default `MODELS_OVERRIDE` is a public HF repo
(`allenai/OLMoE-1B-7B-0924-Instruct`), so `HF_TOKEN` is not required. If
you swap to a gated repo later, export `HF_TOKEN` before invoking
either script.

---

## Workflow

### 1. Login node — pre-download the eval datasets

> **This step is required.** The Spartan GPU compute nodes are
> firewalled off from the public internet AND don't ship the `datasets`
> Python package. Skipping this step will cause the analyzer's
> `load_dataset(...)` calls to fail with `ModuleNotFoundError: No
> module named 'datasets'` or `ConnectionError` mid-run.

```bash
# Default: all 5 datasets (mmlu, bbh, humaneval, popqa, include).
# Total ~500 MB.
cd <repo>/examples/eval-moe
bash spartan/download-datasets.sh

# Just one (e.g. for a smoke test):
bash spartan/download-datasets.sh --datasets mmlu

# Show what each dataset is without downloading:
bash spartan/download-datasets.sh --list

# Custom scratch base:
bash spartan/download-datasets.sh --scratch-base /data/scratch/projects/uom00014/vllm
```

The script auto-installs `datasets` to `~/.local` if it's missing, then
writes each dataset to `${SCRATCH_BASE}/datasets/<name>/` in one of two
layouts:

- **save_to_disk** (`mmlu`, `bbh`, `humaneval`): one Arrow directory per
  split under `<name>/<split>/`.
- **jsonl** (`popqa`, `include`): one `<name>.jsonl` file under `<name>/`
  plus sidecar partition lists (`props.txt` / `languages.txt` /
  `domains.txt` / `languages_domains.txt`).

It's idempotent — re-runs skip any dataset whose output is already
populated.

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

# Show what would be downloaded without downloading (also probes sizes):
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
# Default: 1 A100 GPU, 8 CPUs, 120 GB RAM, 3h55m wall clock.
# Default modules: CUDA/12.4.1, Python/3.11.3, GCCcore/11.3.0.
# Default analyzer: --models allenai/OLMoE-1B-7B-0924-Instruct
#                   --benchmarks mmlu bbh humaneval
#                   --num-samples 256 --max-tokens 256
cd <repo>/examples/eval-moe
sbatch spartan/vllm-moe-eval.sbatch

# Override modules if `module avail` shows different versions:
CUDA_MODULE=CUDA/12.4.1 NCCL_MODULE=NCCL/2.22.3-CUDA-12.4.1 \
PYTHON_MODULE=Python/3.11.3 GCC_MODULE=GCCcore/11.3.0 \
    sbatch spartan/vllm-moe-eval.sbatch

# Run a different model (Mixtral, DeepSeek-MoE, gpt-oss-120b, etc.):
sbatch --export=ALL,MODELS_OVERRIDE=LiteLLMs/Mixtral-8x22B-Instruct-v0.1 \
    spartan/vllm-moe-eval.sbatch

# Run all 5 benchmarks (mmlu + bbh + humaneval + popqa + include):
sbatch --export=ALL,BENCHMARKS="mmlu bbh humaneval popqa include" \
    spartan/vllm-moe-eval.sbatch

# Run multiple models + a quantization method (Cartesian product
# yields 2 model x 1 quant = 2 cells):
sbatch --export=ALL,MODELS_OVERRIDE="allenai/OLMoE-1B-7B-0924-Instruct LiteLLMs/Mixtral-8x22B-Instruct-v0.1",QUANTS_OVERRIDE=awq \
    spartan/vllm-moe-eval.sbatch

# Smaller subset for a smoke test:
sbatch --export=ALL,BENCHMARKS=mmlu,NUM_SAMPLES=8,MAX_TOKENS=8,MAX_MODEL_LEN=1024 \
    spartan/vllm-moe-eval.sbatch

# Multi-GPU (e.g. 2x A100 with NVLink on the same node) — `--gres` is
# the SBATCH directive and `NGPUS` is the runtime var the analyzer
# reads; they MUST match:
sbatch --gres=gpu:2 --export=ALL,NGPUS=2 \
    spartan/vllm-moe-eval.sbatch

# Multi-GPU with an explicit tensor split (e.g. 70/30 split across
# 2 GPUs because GPU-0 has more free memory):
sbatch --gres=gpu:2 --export=ALL,NGPUS=2,EXTRA_TENSOR_SPLIT="70,30" \
    spartan/vllm-moe-eval.sbatch

# Enable post-processing (overview heatmaps + cross-dataset / cross-quant
# Jaccard sweeps). The sister-project's `aggregate_overview.py` and the
# two Jaccard sweep scripts are invoked on the result tree at job end.
# Default: SKIP_PYTHON_PLOTS=1 (skip; the C++ JSON is enough for the
# 4-day HPC budget, visualisations + sweeps are run post-hoc on the
# login node).
sbatch --export=ALL,SKIP_PYTHON_PLOTS=0 \
    spartan/vllm-moe-eval.sbatch
```

The `.sbatch` only loads the NCCL module when `NGPUS > 1` (vLLM's
single-GPU engine doesn't need it). For multi-GPU runs it also runs
`nvidia-smi topo -m` to print the interconnect topology — NVLink
shows as `NV*`, NVSwitch as `SYS`, and PCIe/NVL as `PIX`/`PXB`. If
your partition only has PCIe, set `SPLIT_MODE=tensor` or
`SPLIT_MODE=layer` depending on whether you'd rather pipeline
activations (lower memory) or shard weights (more memory for
larger models).

If the venv install fails on the GPU node with a network error (pypi
unreachable), the .sbatch will exit with `[fatal] uv pip install failed`
and a clear stderr dump. The preflight installs to the venv under
`${SCRATCH_BASE}/venv`, which is cached between submissions; the
second run skips the heavy `uv pip install -e . --torch-backend=${TORCH_BACKEND}`
step entirely.

The .sbatch creates `${SCRATCH_BASE}/venv` on first run, then runs

```bash
VLLM_USE_PRECOMPILED=0 uv pip install \
    --python "${VENV_DIR}/bin/python" \
    numpy datasets==4.5.0 -e "${REPO_ROOT}" \
    --torch-backend=cu129
```

The `VLLM_USE_PRECOMPILED=0` flag forces a source build of vLLM (~60
min on A100). The precompiled wheel bundled with this branch is built
against CUDA 13, which is incompatible with the A-100 driver (CUDA
12.8) on `gpu-a100-short`; the source build links against the
module-loaded CUDA/12.4.1's `libcudart.so.12`, which works on any
partition with a CUDA-12.4+ driver. `--torch-backend=cu129` selects
the PyTorch CUDA-12.9 wheel index (the only one that publishes
`torch==2.13.0`, the version this vLLM branch pins in
`pyproject.toml`'s `[build-system].requires`). `datasets==4.5.0` pins
the same version used on the login node for the pre-staged datasets
(so the `save_to_disk` arrow files are byte-compatible across read +
write). `-e .` installs the local `${REPO_ROOT}` checkout so any
commits you pushed to the GPFS source tree are picked up
immediately.

If pyproject.toml is bumped to pin a different torch version in the
future, set `TORCH_BACKEND` to whichever PyTorch CUDA-X.Y index
publishes that pin (the preflight probes the index page and fails
fast with a clear error pointing at the right `TORCH_BACKEND` if the
pinned torch isn't on the chosen index).

Then it invokes `examples/eval-moe/moe_expert_stats.py` with the
configured argv. By default (`SKIP_PYTHON_PLOTS=1`) it just emits the
JSONs and exits. With `SKIP_PYTHON_PLOTS=0` it also runs
`aggregate_overview.py` + `jaccard_sweep_from_cpp.py` +
`cross_quant_jaccard_sweep_from_cpp.py` from the sister
`llama-cpp-eval` repo (`/data/projects/uom00014/llama-cpp-eval/examples/eval-moe-overview/`).

### 3. Monitor

```bash
# Job status
squeue -j $JOB_ID

# Top-level job log
tail -f spartan/logs/vllm-moe-eval-$JOB_ID.out

# Per-cell analyzer output (post-processing logs go to
# ${OUTPUT_DIR}/<model_safe>/<quant_safe>/{jaccard_sweep.log,cross_quant.log})
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
  `[info] deps installed` and `vllm=0.1.dev20193+...` in the preflight.
- `[info] running: python examples/eval-moe/moe_expert_stats.py --models allenai/OLMoE-1B-7B-0924-Instruct ...`
- `[info] === cell: model=..., quant=<none> ===` at the start of each cell.
- Per-benchmark lines: `[info] Running benchmark mmlu: 1 row(s), 8 prompt(s) total`
  `[info] Wrote <OUTPUT_DIR>/allenai--OLMoE-1B-7B-0924-Instruct/default/moe-mmlu/expert_counts.json (rows=8, tokens_prefill=..., tokens_generated=...)`
- (With `SKIP_PYTHON_PLOTS=0`): `[ok] aggregate_overview.py -> <output>/<model>/default/overall/...`

---

## Env-var overrides

All variables are optional. Set them before `sbatch` (or pass via
`--export=...`).

<!-- markdownlint-disable MD060 -->

| Variable             | Default                                                                            | Notes                                                                                                                                                                                                                                                                                         |
| -------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `REPO_ROOT`          | `/data/gpfs/projects/uom00014/vllm`                                                | Where the repo is cloned                                                                                                                                                                                                                                                                      |
| `SCRATCH_BASE`       | `/data/scratch/projects/uom00014/vllm` (or `${SCRATCH}/vllm` if `$SCRATCH` is set) | Root for venv / datasets / results / hf_cache                                                                                                                                                                                                                                                 |
| `CUDA_MODULE`        | `CUDA/12.4.1`                                                                      | Lmod module name (canonical gpu-a100-short default)                                                                                                                                                                                                                                           |
| `TORCH_BACKEND`      | `cu129`                                                                            | PyTorch CUDA wheel index tag (must publish the torch version pinned in `pyproject.toml`'s `[build-system].requires`). Only `cu129` publishes `torch==2.13.0` today; `cu124` maxes at 2.6.0, `cu128` at 2.11.0. The preflight probes the index and fails fast if the pinned torch isn't there. |
| `PYTHON_MODULE`      | `Python/3.11.3`                                                                    | Lmod module name (canonical gpu-a100-short default)                                                                                                                                                                                                                                           |
| `GCC_MODULE`         | `GCCcore/11.3.0`                                                                   | Lmod module name; **must be the older compiler family** that Python/3.11.3 is built against, not the newer GCC/13.3.0                                                                                                                                                                         |
| `NCCL_MODULE`        | `NCCL/2.22.3-CUDA-12.4.1`                                                           | Lmod module name for NCCL. The preflight walks `module avail NCCL/`, picks the build matching `${CUDA_MODULE}`, and loads it automatically when `NGPUS > 1`. Override before sbatch to pin a specific build. (On the login node `module avail` returns nothing in non-interactive sessions, so the override is usually required.) |
| `NGPUS`              | `1`                                                                                | Number of GPUs the job uses. Must match `#SBATCH --gres=gpu:N` (that directive can't reference shell vars). Forwarded to vLLM as `--tensor-parallel-size`.                                                                                                                                    |
| `SPLIT_MODE`         | `layer`                                                                            | vLLM's distributed-executor mode. `layer` = pipeline parallel; `tensor`/`row` = weight parallel (slower over PCIe without NVLink); `none` = single-GPU on a multi-GPU box.                                                                                                                    |
| `EXTRA_TENSOR_SPLIT` | `<unset>`                                                                          | Comma-separated weights for vLLM's `--tensor-split` (e.g. `50,50` for 2 GPUs, `25,25,25,25` for 4). When unset AND `NGPUS > 1`, the preflight auto-derives a uniform split so the model actually spreads layers instead of falling back to all-on-GPU-0.                                      |
| `MODELS_OVERRIDE`    | `allenai/OLMoE-1B-7B-0924-Instruct`                                                | Space-separated HF model ids. Each model gets its own LLM instance and routed-experts capture buffer.                                                                                                                                                                                         |
| `QUANTS_OVERRIDE`    | `<unset>`                                                                          | Space-separated vLLM quantization methods (e.g. `awq`, `fp8`, `gptq`, `bitsandbytes`). Cells are evaluated as `MODELS_OVERRIDE` × `QUANTS_OVERRIDE` (Cartesian product). Pass an empty token (or leave unset) to skip quantization.                                                           |
| `BENCHMARKS`         | `mmlu bbh humaneval`                                                               | Space-separated subset of `{mmlu, bbh, humaneval, popqa, include}`                                                                                                                                                                                                                            |
| `NUM_SAMPLES`        | `256`                                                                              | Sample budget per benchmark (per-subject for mmlu/bbh/include, per-task for humaneval, per-prop for popqa)                                                                                                                                                                                  |
| `MAX_TOKENS`         | `256`                                                                              | Autoregressive decode cap per prompt (set 0 for prefill-only)                                                                                                                                                                                                                                 |
| `N_SHOTS`            | `5`                                                                                | Few-shot exemplars from the per-dataset dev pool (mmlu, bbh, include). 0 = zero-shot.                                                                                                                                                                                                        |
| `MAX_MODEL_LEN`      | `4096`                                                                             | vLLM `max_model_len` setting                                                                                                                                                                                                                                                                  |
| `GPU_MEM_UTIL`       | `0.92`                                                                             | vLLM `gpu_memory_utilization`                                                                                                                                                                                                                                                                 |
| `ENFORCE_EAGER`      | `<unset>`                                                                          | If set, pass `--enforce-eager` to `LLM(...)` (skip CUDA graphs)                                                                                                                                                                                                                               |
| `OUTPUT_DIR`         | `${SCRATCH_BASE}/results`                                                          | Where to write per-benchmark JSON files (one subdir per `(model, quant, moe-<bench>)` cell)                                                                                                                                                                                                 |
| `SEED`               | `0`                                                                                | Sampler + sampling seed                                                                                                                                                                                                                                                                       |
| `HF_TOKEN`           | `<unset>`                                                                          | Optional; for gated repos                                                                                                                                                                                                                                                                     |
| `SKIP_DATASET_CHECK` | `0`                                                                                | `1` = skip the dataset preflight warning                                                                                                                                                                                                                                                      |
| `SKIP_PYTHON_PLOTS`  | `1`                                                                                | `1` = skip all post-processing (the default; 4-day HPC budget usually just wants the JSON). `0` = run the sister-project's `aggregate_overview.py` + `jaccard_sweep_from_cpp.py` + `cross_quant_jaccard_sweep_from_cpp.py` against the result tree (produces overview heatmaps + cross-dataset / cross-quant Jaccard sweep PNGs+CSVs). |

<!-- markdownlint-enable MD060 -->

---

## Results tree

Every `(model, quant, benchmark)` cell the analyzer runs writes its own
subtree under `${OUTPUT_DIR}/<model_safe>/<quant_safe>/moe-<bench>/expert_counts.json`,
in the same schema as the sister llama-cpp-eval C++ binaries:

```text
${OUTPUT_DIR}/
└── <model_safe>/
    └── <quant_safe>/
        ├── moe-mmlu/expert_counts.json
        ├── moe-bbh/expert_counts.json
        ├── moe-humaneval/expert_counts.json
        ├── moe-popqa/expert_counts.json
        └── moe-include/expert_counts.json
```

`<model_safe>` is the HF repo id with `/` replaced by `--`
(matching the sister project's convention). `<quant_safe>` is the
quant tag as-is, or `default` when `--quant` is empty.

For the default 1-model / no-quant / 3-benchmark run the layout
collapses to:

```text
${OUTPUT_DIR}/allenai--OLMoE-1B-7B-0924-Instruct/default/
├── moe-mmlu/expert_counts.json
├── moe-bbh/expert_counts.json
└── moe-humaneval/expert_counts.json
```

Running `--models A B --quant awq fp8 --benchmarks mmlu bbh humaneval popqa include`
produces a `2 × 2 = 4` cell tree with `5` benchmark subdirs each (20
JSON files total).

When `SKIP_PYTHON_PLOTS=0`, the sister-project's
`aggregate_overview.py` adds per-cell overview artefacts under
`${OUTPUT_DIR}/<model_safe>/<quant_safe>/overall/` (heatmap PNGs +
`counts_total_overview.json` + `top_experts.json` +
`metadata_overview.json`); the two Jaccard sweep scripts add
`jaccard_sweep/` (per cell, cross-dataset) and
`cross_quant_jaccard_sweep/` (per model, cross-quantization) with
their PNGs + CSVs.

The per-benchmark JSON has the schema documented in
[`examples/eval-moe/README.md`](../README.md#per-benchmark-json-schema-moebenchexpert_countsjson).
Briefly, the vLLM-produced and llama-cpp-produced JSONs are
**byte-compatible** so the same downstream tooling (aggregate
overview, Jaccard sweeps) reads both.

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

<!-- markdownlint-disable MD060 -->

| Symptom                                                                                                                           | Likely cause                                                                                                                  | Fix                                                                                                                                                                                                                                                                                                               |
| --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `module: command not found`                                                                                                       | Lmod not in `.bashrc` on the GPU node                                                                                         | `source /usr/local/lmod/lmod/init/bash` before `module` calls, or trust the .sbatch's `module purge`                                                                                                                                                                                                              |
| `Lmod has detected the following error: The following module(s) are unknown: "CUDA/12.X"` (or similar)                            | The hard-coded default module name doesn't exist on `gpu-a100-short` (versions change over time)                              | `module avail cuda/python/gcc` on a `gpu-a100-short` node, then re-submit with the exact name                                                                                                                                                                                                                     |
| `Lmod has detected the following error: These module(s) or extension(s) exist but cannot be loaded as requested: "Python/3.11.3"` | The Python module isn't built against the GCC compiler you have loaded (e.g. you used `GCC/13.3.0` but need `GCCcore/11.3.0`) | `module spider Python/3.11.3` to see the required parent compiler, then re-submit with `GCC_MODULE=<spider-suggested-gcc>` (default is `GCCcore/11.3.0`)                                                                                                                                                          |
| `[fatal] CUDA_MODULE is empty`                                                                                                    | Forgot to export the module env vars                                                                                          | `module avail cuda/python/gcc` on `gpu-a100-short`, then re-submit with all three exported                                                                                                                                                                                                                        |
| `python3: command not found` after `module load`                                                                                  | Wrong PYTHON_MODULE                                                                                                           | `module avail python` on `gpu-a100-short` and set `PYTHON_MODULE`                                                                                                                                                                                                                                                 |
| `gcc: command not found` after `module load`                                                                                      | Wrong GCC_MODULE                                                                                                              | `module avail gcc` on `gpu-a100-short` and set `GCC_MODULE` (default is `GCCcore/11.3.0`)                                                                                                                                                                                                                         |
| `[fatal] NGPUS=2 > 1 but no NCCL module is available`                                                                              | NCCL auto-probe failed (login node `module avail` returns nothing in non-interactive sessions)                                  | `module avail NCCL/` on a `gpu-a100-short` node, then re-submit with `NCCL_MODULE=<name>` (default is `NCCL/2.22.3-CUDA-12.4.1`)                                                                                                                                                                                       |
| `ggml_cuda_init: no CUDA devices found`                                                                                           | `CUDA_VISIBLE_DEVICES` empty or wrong GPU count                                                                               | Check `squeue -j $JOBID -o "%Gres"`; request `--gres=gpu:N` to match                                                                                                                                                                                                                                              |
| `[fatal] datasets not staged on scratch` warning at job start                                                                     | Login-node `download-datasets.sh` wasn't run (or didn't finish)                                                               | Run on the login node first: `bash spartan/download-datasets.sh --datasets <names>`. The .sbatch will still proceed; the analyzer will fail mid-run.                                                                                                                                                              |
| `ModuleNotFoundError: No module named 'datasets'` mid-run                                                                         | GPU node can't reach pypi + `download-datasets.sh` wasn't run                                                                 | Run on the login node first: `bash spartan/download-datasets.sh`                                                                                                                                                                                                                                                  |
| `ModuleNotFoundError: No module named 'vllm'`                                                                                     | venv install failed; .sbatch's `uv pip install` retry gave up                                                                 | Re-submit; if it keeps failing, see the install command in the `.sbatch` header and run it manually.                                                                                                                                                                                                              |
| `ConnectionError` from `datasets.load_dataset`                                                                                    | GPU node can't reach HF hub                                                                                                   | Same fix: pre-stage via login-node `download-datasets.sh` (the script auto-installs `datasets` and downloads the splits to scratch). Note that the **model weights** still need network access on the GPU node; if the GPU node has full outbound firewall on pypi+huggingface then the analyzer can't run there. |
| `ConnectionError` from `LLM(model=...)` weight download                                                                           | GPU node can't reach HF hub, AND `download-models.sh` wasn't run                                                              | Run on the login node first: `bash spartan/download-models.sh --models <id>`. The .sbatch reads from `${HF_CACHE}`, which is where `download-models.sh` writes.                                                                                                                                                   |
| OOM / `Killed` in job log                                                                                                         | 120B model + activations exceed `--mem`                                                                                       | Raise `--mem` (A100 node has 495 GB total) or use a smaller model/quant                                                                                                                                                                                                                                           |
| Analyzer hangs at first `LLM(...)` call                                                                                           | Cold-start weight download from HF hub is slow on shared bandwidth                                                            | First run only; subsequent runs reuse `${HF_CACHE}`; pre-stage via `download-models.sh` to skip entirely                                                                                                                                                                                                          |
| `[fatal] import datasets: AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'` (legacy failure mode)            | Wrong `datasets` version in the venv (mislabeled dist-info + pyarrow drift)                                                  | Already mitigated by `datasets==4.5.0` + `pyarrow<18` pins in the install line; if you see this, force a clean rebuild via `rm -rf $SCRATCH_BASE/venv && sbatch`                                                                                                                                              |
| `[fatal] KeyError: '_indexes'` mid-run (load_from_disk)                                                                            | Datasets cached by a different `datasets` version                                                                             | Re-stage the cache on the login node with the same `datasets` version (the sbatch pins `datasets==4.5.0` so the login node + GPU node match)                                                                                                                                                                       |

<!-- markdownlint-enable MD060 -->

---

## File map

```text
examples/eval-moe/
├── README.md                    # analyzer + schema docs
├── moe_expert_stats.py          # the analyzer
└── spartan/
    ├── README.md                # this file
    ├── download-models.sh       # login-node pre-download (HF model weights)
    ├── download-datasets.sh     # login-node pre-download (5 datasets; dispatches by layout)
    ├── download-popqa.py        # login-node pre-download (popqa jsonl)
    ├── download-include.py      # login-node pre-download (include jsonl)
    └── vllm-moe-eval.sbatch     # SLURM job (gpu-a100-short, 1 GPU, 3h55m)
```

Related (not in this directory):

- [`examples/eval-moe/moe_expert_stats.py`](../moe_expert_stats.py) — the
  analyzer that the .sbatch invokes. Holds the `BENCHMARK_LOADERS`
  dict + the `BENCHMARK_RECORD_KEY` / `BENCHMARK_PROMPT_FORMAT`
  constants; the .sbatch's CLI flags
  (`--models`, `--quant`, `--benchmarks`, `--num-samples`,
  `--max-tokens`, `--n-shots`, `--max-model-len`,
  `--gpu-memory-utilization`, `--enforce-eager`, `--output-dir`,
  `--seed`, `--dtype`) all map 1:1 to the analyzer's argparse. The
  analyzer evaluates the Cartesian product of `--models` × `--quant`,
  instantiating a fresh `LLM` per model so each cell has its own KV
  cache + routed-experts capture buffer.
- [`/data/projects/uom00014/llama-cpp-eval/examples/eval-moe-overview/`](../../../../../../projects/uom00014/llama-cpp-eval/examples/eval-moe-overview/) —
  the post-processing toolchain. The .sbatch calls these scripts
  directly when `SKIP_PYTHON_PLOTS=0`. They consume the
  `expert_counts.json` schema byte-for-byte regardless of whether the
  JSON was written by this vLLM analyzer or the C++ binary.
