# vLLM eval-moe sweep handoff (Aug–Oct 2026)

This document captures the full state of the `eval-moe` project as of **Oct 2, 2026 ~17:35 AEST**: what was attempted, what worked, what's left on disk, what's left to do, and the gotchas for whoever picks this up next. It is the authoritative reference — both `~/.hermes/memory/` (auto-injected, terse) and the on-disk configs/sbatch scripts (technical detail) point back here for the full picture.

**TL;DR**:
- bf16, fp8, W8A16 sweeps **complete** across all 4 models × 5 benchmarks. `results_wide_w8a16_h100/` still on scratch (13 GB) — needs MediaFlux upload. bf16+fp8 trees already on MediaFlux.
- W4A16 / W4A8 online paths **abandoned** (MXFP4 has no compatible H100 MoE kernel; compressed-tensors W4A8 is checkpoint-only).
- **AWQ pre-quant pipeline launched Oct 1–2** (`examples/eval-moe/awq_quantize.py` + `vllm-moe-awq-prequant.sbatch`). 9 jobs PENDING as of this writing. **All 30+ historical AWQ attempts failed at the same step — see §4.7.**
- The AWQ blocker is a one-line fix (`trust_remote_code=True` on the `AutoModelForCausalLM.from_pretrained` call in `awq_quantize.py:132`) plus a likely secondary OOM fix for Mixtral. **Patches needed before re-submitting.**
- See §7 for next steps in priority order.

---

## 1. Project context

- **Repo**: `/data/gpfs/projects/uom00014/vllm` (branch `eval-moe`)
- **Sister repo**: `/data/projects/uom00014/llama-cpp-eval` (Llama.cpp analyzer + cross-archive Jaccard aggregator; reads our `expert_counts.json` outputs)
- **HPC**: Spartan at UniMelb, login node `spartan-login2.hpc.unimelb.edu.au`. Login is **firewalled** from the public internet (no HF download at runtime, no module install). GPU nodes are reachable; vLLM runtime workloads run via `sbatch`.
- **Module stack on GPU nodes**: `GCCcore/11.3.0` + `Python/3.11.3` + `CUDA/12.4.1` (driver 12.8). `NCCL/2.22.3-CUDA-12.4.1` hardcoded in script defaults because `module avail NCCL/` returns nothing on the login node. CUDA toolkit on login can lag behind the driver — see §5.4.
- **Disk**:
  - `/data/scratch/projects/uom00014/vllm/` — **1.4 TB quota, 730 GB / 53% used**, 668 GB free (as of Oct 2 17:35). Was 466 GB until late Sep; admin expanded it. Scratch quota guard in sbatches is **≥90% = abort before download**.
  - `/data/gpfs/projects/uom00014/vllm/` — GPFS project volume (447 GB free). venv at `$REPO/.venv/`.
- **User**: Pratheek Senevirathne (plsenevirath), Unimelb. Workflow is cancel-and-resubmit often; **persists at cancelling rather than waiting for clarification**. Pick a defensible default and proceed.
- **Two parallel venvs on scratch** (new as of Oct 1):
  - `${SCRATCH}/venv/` — analyzer venv. `torch==2.13.0+cu129`, `transformers==5.15.1`, `datasets==4.5.0`, `vllm` from source.
  - `${SCRATCH}/venv_awq/` — AWQ pre-quant venv. `torch==2.13.0+cu129`, `transformers==5.15.1`, `datasets>=4.8.4`, `llmcompressor==0.14.1.dev18+g0a381869` (installed from `git+https://github.com/vllm-project/llm-compressor.git@main` because PyPI's `llmcompressor==0.8.1` caps `transformers<=4.56.2` and the analyzer venv already has 5.x). 7.3 GB on disk.

## 2. Model checkpoints on scratch (`/data/scratch/projects/uom00014/vllm/hf_cache/`)

`HUGGINGFACE_HUB_CACHE=hf_cache/` (so `hf_cache` itself is the hub dir, **not** `hf_cache/hub/`).

| Model snapshot | Layout | Size | Layout quirk |
|---|---|---|---|
| `models--mistralai--Mixtral-8x22B-Instruct-v0.1` | root-level | 262 GB | bf16, 59 shards, 56L/8E/k=2. Quantized 14 GB per param at TP=1 → bf16 w4a16 spike needs TP=2+. Re-downloaded Oct 1 after cleanup. |
| `models--Qwen--Qwen3-30B-A3B` | symlink → `hub/...` | 57 GB | Originally downloaded with HUGGINGFACE_HUB_CACHE unset (landed at `hub/`). Symlink added Sep 25 so `AutoConfig.from_pretrained` finds it under the root path too. The symlink is **required** for the analyzer to run. |
| `models--deepseek-ai--deepseek-moe-16b-chat` | root-level | 31 GB | 28L/64E/k=6+2shared. The shared-experts stay bf16 under online int8 quant (only routed experts are quantized). **Ships custom modeling code** — requires `trust_remote_code=True`. |
| `models--allenai--OLMoE-1B-7B-0924-Instruct` | root-level | 13 GB | 16L/64E/k=8. Smallest, fits TP=1 even with w8a16 quant. |
| `models--moonshotai--Kimi-K2.6` | root-level | ~125 GB | Not mine, separate experiment. |

**Also on scratch** (NOT pre-quantized checkpoints — these are just empty scaffolding dirs from the failed AWQ runs):
- `/data/scratch/projects/uom00014/vllm/hf_cache_awq/` — empty per-model dirs: `allenai--OLMoE-1B-7B-0924-Instruct{,-awq,-gptq}`, `deepseek-ai--deepseek-moe-16b-chat{,-awq,-gptq}`, `Qwen--Qwen3-30B-A3B-{awq,gptq}`, `mistralai--Mixtral-8x22B-Instruct-v0.1-{awq,gptq}`. **No `quantize_config.json` ever written.** Safe to `rm -rf /data/scratch/projects/uom00014/vllm/hf_cache_awq/` once the prequant pipeline is fixed and jobs re-run.

**Deleted during cleanup** (Sep 16, user-driven):
- Stranded `hub/models--deepseek-ai--deepseek-moe-16b-chat/` (31 GB duplicate of the canonical root-level snapshot; verified by inode comparison that blobs were not actually hardlinks — different files, same content).

## 3. Codebase research

### 3.1 Analyzer (`examples/eval-moe/moe_expert_stats.py`)

- ~1545 lines (after Oct 2 patches), single-file CLI. Runs one `(model, quant, benchmark)` cell per `LLM(...)` instance (no engine sharing across cells; each cell gets its own KV cache + expert capture buffer).
- 5 benchmarks via the registry `BENCHMARK_LOADERS`: `mmlu`/`bbh`/`humaneval` (HF `save_to_disk`) and `popqa`/`include` (JSONL local files).
- Per-benchmark jsonl files staged under `${SCRATCH_BASE}/datasets/` (pre-staged once via login; not via `spartan/download-datasets.sh`).
- Output tree: `<output_dir>/<model_safe>/<quant_safe>/moe-<bench>/expert_counts.json`. Idempotent: skips cells whose json already exists.

**Working tree diff** (uncommitted as of Oct 2 17:35, +127/-19 lines):
- `CMakeLists.txt`: +2 lines, adds CUDA ≥ 12.8 floor for the cooperative_topk MoE kernel (`workaround for vLLM PR that requires CUDA 13.x PTX but we have CUDA 12.4.1 toolkit on Spartan; gates the kernel out at build time`).
- `moe_expert_stats.py`: +127 lines, -19 lines. Four additions:
  1. `_hf_arch_name(model_type)` helper that maps HF `model_type` to a canonical sister-repo label (e.g. `qwen3_moe` → `qwen3-moe`). Was hardcoded `"olmoe"` for every model — fixed so cross-architecture heatmaps show the right names.
  2. `_resolve_local_snapshot(model_id)` helper that accepts both `hf_cache/models--<org>--<name>/` and `hf_cache/hub/models--<org>--<name>/` layouts. The Qwen3 snapshot lives at `hub/`, but vLLM's `AutoConfig` standard resolver only looks under root — this helper bypasses that.
  3. `--hf-overrides` CLI flag that JSON-parses a dict and forwards to `LLM(..., hf_overrides=...)`. Used for the (failed) W4A8 attempt.
  4. Plumbing: `_read_model_config` now returns `(num_experts, num_experts_per_tok, num_hidden_layers, model_type)`; `effective_arch_name` computed from `model_type` and threaded into `_run_benchmark` so the `model_arch.name` in the JSON is the real HF model_type, not the placeholder `"olmoe"`. `BenchmarkPrompts.arch_name` is now "Deprecated as a source of truth" (still set to "olmoe" by loaders for back-compat, but no longer read).

### 3.2 AWQ pre-quant (`examples/eval-moe/awq_quantize.py`)

- 282 lines, single-file CLI. Loads a bf16 MoE model, runs an llm-compressor recipe (AWQ scales, GPTQ Hessian, or AWQ+GPTQ combined), saves to a new snapshot dir.
- **Currently broken at line 132** (see §4.7). The fix is to add `trust_remote_code=True` to the `AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.bfloat16)` call so DeepSeek's custom modeling code doesn't prompt interactively under SLURM (no TTY → EOFError).
- Recipes in `build_recipe(recipe)`:
  - `awq`: `AWQModifier` (scales) + `QuantizationModifier` (round-to-nearest int4). Pure AWQ, no Hessian.
  - `gptq`: `GPTQModifier` only. Pure Hessian-based int4.
  - `awq_gptq` (default): `AWQModifier` first to scale weights, then `GPTQModifier` to round using scaled weights.
- All three ignore `lm_head` and MoE router gates (`re:.*mlp\.gate$`, `re:.*mlp\.shared_expert_gate$`). Group size 128. W4A16 only.
- `oneshot(model=..., recipe=..., max_seq_length=2048, moe_calibrate_all_experts=True)` — `moe_calibrate_all_experts=True` ensures AWQ+GPTQ stats are computed per-expert, not per-layer.
- Calibration data: pre-staged at `${SCRATCH_BASE}/datasets/perfectblend/data/*.parquet`. Passed via `--dataset /data/scratch/projects/uom00014/vllm/datasets/perfectblend` (local dir, globbed for parquets internally).
- Output dir: `${SCRATCH_BASE}/hf_cache_awq/<model_safe>-<recipe>/`. After save, the dir MUST contain `config.json` + `quantize_config.json` (script exits 1 with explicit `[awq-quant][error] no ...` if either missing). The `quantize_config.json` declares `quant_method="awq"` (or `"gptq"` for the GPTQ-only path), so vLLM's `moe_wna16` quant method picks the right kernel at load time.
- After all 4×3 prequant runs succeed, the analyzer runs as:
  ```bash
  python3 examples/eval-moe/moe_expert_stats.py \
      --model /data/scratch/projects/uom00014/vllm/hf_cache_awq/<model_safe>-<recipe> \
      --quantization moe_wna16 --output-dir .../results_wide_w4a16_awq_h100/<model_safe>
  ```

### 3.3 Spartan sbatch catalog (12 sbatches in `examples/eval-moe/spartan/`)

| sbatch | Purpose | Status |
|---|---|---|
| `vllm-moe-eval.sbatch` | gpu-a100 baseline | not modified, default `--quantization=` (bf16) |
| `vllm-moe-eval-deeplearn.sbatch` | deeplearn dlg5 (A100, TP=2) | not modified |
| `vllm-moe-eval-deeplearn-h100.sbatch` | deeplearn-h100 dlg6 (TP=2) | not modified |
| `vllm-moe-eval-h100-bf16.sbatch` | gpu-h100 bf16 wide | **the bread-and-butter** — 4 models × 5 benchmarks |
| `vllm-moe-eval-h100-bf16-mixtral.sbatch` | gpu-h100 bf16 Mixtral only (TP=4) | works |
| `vllm-moe-eval-h100-fp8-deepseek.sbatch` | gpu-h100 fp8 deepseek only | works |
| `vllm-moe-eval-h100-fp8-wide.sbatch` | gpu-h100 fp8 wide (OLMoE + Qwen3) | works |
| `vllm-moe-eval-h100-w8a16-moe.sbatch` | gpu-h100 W8A16 wide (online int8 quant, TP=1) | **runs OK on OLMoE/DeepSeek-MOE/Qwen3; Mixtral TP=1 OOMs** (needs TP=2+/4) |
| `vllm-moe-eval-h100-w8a16-mixtral-tp2.sbatch` | gpu-h100 W8A16 Mixtral TP=2 | **failed** at TP=2 (bf16 quant spike exceeded 80 GB), obsolete |
| `vllm-moe-eval-h100-w8a16-mixtral-tp4.sbatch` | gpu-h100 W8A16 Mixtral TP=4 | **works** (use this for Mixtral-only W8A16) |
| `vllm-moe-eval-h100-w4a16-moe.sbatch` | gpu-h100 W4A16 wide (online MXFP4, TP=1) | **failed**: MXFP4 MoE backend unsupported on H100 SM 9.0 |
| `vllm-moe-eval-h100-w4a8-moe.sbatch` | gpu-h100 W4A8 wide (compressed-tensors + hf_overrides) | **failed**: W4A8 MoE expects pre-quantized checkpoints, not online |
| `vllm-moe-awq-prequant.sbatch` | gpu-h100 AWQ/GPTQ/AWQ+GPTQ pre-quant, 1 model per job | **launched Oct 1–2, all 30+ attempts failed — §4.7** |

**Sbatch env-var contract** (defaults; overrides via `--export=` or `VAR=value`):
- `CUDA_MODULE=CUDA/12.4.1`, `PYTHON_MODULE=Python/3.11.3`, `GCC_MODULE=GCCcore/11.3.0`
- `TORCH_BACKEND=cu129` (PyTorch cu129 wheel is the one that has torch==2.13.0; cu124 caps at 2.6.0)
- `NCCL_MODULE=NCCL/2.22.3-CUDA-12.4.1` (loaded when `NGPUS>1`)
- `MODELS_OVERRIDE="allenai/OLMoE-1B-7B-0924-Instruct deepseek-ai/deepseek-moe-16b-chat Qwen/Qwen3-30B-A3B mistralai/Mixtral-8x22B-Instruct-v0.1"`
- `BENCHMARKS="mmlu bbh humaneval popqa include"`
- `NUM_SAMPLES=999999`, `MAX_TOKENS=256`, `MAX_MODEL_LEN=4096`, `N_SHOTS=5`, `SEED=0`, `SKIP_PYTHON_PLOTS=1`
- `GPU_MEM_UTIL=0.92` (mixtral W8A16 needs 0.85, override via `--export=GPU_MEM_UTIL=0.85`)
- AWQ-specific: `MODEL_ID` (required), `RECIPE=awq_gptq` (one of `awq|gptq|awq_gptq`), `OUTPUT_BASE=${SCRATCH}/hf_cache_awq`, `PERFECTBLEND_DIR=${SCRATCH}/datasets/perfectblend`, `AWQ_VENV_DIR=${SCRATCH}/venv_awq`.

**`#SBATCH` quirks**:
- `#SBATCH --gres=gpu:N` cannot use a shell variable — must be a literal in the directive. This is why we have separate sbatches for TP=1/TP=2/TP=4 variants of the same quant sweep.
- `#SBATCH --output/--error` paths must be absolute. We use `/data/gpfs/projects/uom00014/vllm/examples/eval-moe/spartan/logs/vllm-moe-{eval,awq-prequant}-%j.{out,err}`.
- 12-hour wall time on analyzer sweeps (long enough for venv build + worst-case Mixtral eval).
- 4-hour wall time on AWQ prequant (calibration is the slow part for Mixtral).
- AWQ prequant requests `--mem=720G` (Mixtral bf16 ~262 GB + linearization buffer + tokenized calibration data). Other models use far less, SLURM only enforces the cap when needed.

### 3.4 Analyzer forward kwargs to `LLM(...)`

`model`, `enable_return_routed_experts=True` (the routed-experts capturer buffer), `max_model_len`, `dtype`, `gpu_memory_utilization`, `enforce_eager`, `seed`, `tensor_parallel_size`, `trust_remote_code=True`, plus optional `quantization`, `hf_overrides`. The `trust_remote_code=True` is required because DeepSeek-MoE ships custom modeling code on the Hub and SLURM jobs have no TTY (a prompt would EOF).

## 4. Recent experiment runs (chronological)

### 4.1 bf16 H100 baseline + Mixtral TP=4

| Job ID | Partition | State | Wall | Notes |
|---|---|---|---|---|
| `30635801` | gpu-h100 | COMPLETED (0:0) | 1h33m | 4 models × 5 benchmarks (TP=1 for OLMoE/DeepSeek/Qwen3, TP=2 didn't fit) |
| `30638495` | gpu-h100 | **FAILED (1:0)** | 36m | Mixtral bf16 at TP=1 — `gpu_mem_util=0.92` → "Available KV cache memory: -1.86 GiB" |
| `30702863` | gpu-h100 | COMPLETED (0:0) | 1h08m | Mixtral bf16 at TP=4, `gpu_mem_util=0.92` → all 5 benchmarks succeeded |

**Output**: `results_wide_bf16_h100/` — 4 models × 5 benchmarks = 20 expert_counts.json. **Uploaded to MediaFlux then deleted from scratch** (see §4.3).

### 4.2 fp8 H100 wide sweep + disk-quota protection

| Job ID | State | Wall | Notes |
|---|---|---|---|
| `31052007` | FAILED (1:0) | 56m | Mid-download `os error 122` ("Disk quota exceeded") on scratch at 90%+ |
| `31186052` | FAILED (7:0) | 5s | New scratch-quota guard kicked in before any work (exit code 7 = quota guard fatal) |
| (manual cleanup) | — | — | Deleted stranded `hub/models--deepseek-ai--deepseek-moe-16b-chat/` (31 GB duplicate; not actually hardlinked to root — separate files with same content) |
| (manually uploaded + deleted) | — | — | bf16 + fp8 trees uploaded to MediaFlux, deleted from scratch |

**Patch in all wide sbatches**: a 90% scratch-quota guard before download.

### 4.3 MediaFlux upload setup (Sep 28, 2026)

- **`~/.Arcitecta/mflux.cfg`** (mode 0600) — standard format per RCS docs:
  ```
  host=mediaflux.researchsoftware.unimelb.edu.au
  port=443
  transport=https
  domain=student
  user=plsenevirath
  password=<password>
  ```
  (Corrected from initial template which had `domain=unimelb` and full email-as-user; the student-domain form actually authenticates.)
- **Allocation**: `/projects/proj-4180_momqe_data-1128.4.2052/` — 512 GB project allocation. Confirmed writable via test upload.
- **Helper script**: `~/bin/upload-eval-moe-results.sh` (also at `~/.hermes/scripts/upload-eval-moe-results.sh`):
  - Default destination: `/projects/proj-4180_momqe_data-1128.4.2052/<tree-name>/`
  - Uses `--csum-check` for integrity
  - Auto-deletes local copy unless `NO_DELETE=1`
  - **Critical flag**: `--exclude-parent` + source dir with trailing `/` (otherwise path doubles — see §4.3.1)
  - Default workers: 4 (max 8 per tool docs)

#### 4.3.1 First upload bug: `--exclude-parent` missing

In the first upload run (Sep 28), the helper script omitted `--exclude-parent`. Result: the upload tool's default behavior preserved the source basename inside the destination, creating `dest/<tree>/<tree>/...` paths instead of `dest/<tree>/...`. ~26 GB of duplicates went to /projects/.../results_wide_w8a16_h100/results_wide_w8a16_h100/... and similar for fp8_h100. The duplicates are intentional duplicates of pre-existing data (bf16/fp8 had been uploaded earlier; the data is intact). User said "the duplicate files are fine for now" so no cleanup was done.

### 4.4 W8A16 H100 sweep (Sep 25–29)

| Job ID | State | Wall | Notes |
|---|---|---|---|
| `31273232` | FAILED (1:0) | 29m | Cell 3 Qwen3: `transformers.AutoConfig` doesn't honor vLLM's HF_HUB_OFFLINE rewrite (Qwen3 snapshot lives at `hub/`, not root). |
| `31280753` | FAILED (1:0) | 1h05m | Cells 1-3 OK (OLMoE + DeepSeek-MoE cached→skipped, Qwen3 5/5 ✅ after patch), cell 4 Mixtral CUDA OOM at online int8 quant (TP=1, layerwise spike + 262 GB bf16 footprint). |
| `31425093` | FAILED (1:0) | 1h32m | Same Mixtral OOM at `gpu_mem_util=0.85` (gpu_mem_util doesn't bound the layerwise intermediate state). |
| `31503046` | FAILED (1:0) | 37m | Same cell, retry at TP=2: weight load alone consumed nearly all reserved memory (`Available KV cache memory: -0.23 GiB`). |
| `31516295` | **COMPLETED (0:0)** | 56m | **TP=4**. Mixtral TP=4 weight slice 65 GB → spike 24 GB → fits 80 GB with ~55 GB KV headroom. All 4 models done. |

**Patch history**:
1. **Symlink + `_resolve_local_snapshot()`** in `moe_expert_stats.py` — bypasses vLLM's offline-mode rewrite that doesn't know about hub/ layout. Created symlink `hf_cache/models--Qwen--Qwen3-30B-A3B → hub/models--Qwen--Qwen3-30B-A3B`.
2. **Mixtral-only TP=4** sbatch for the single failing cell.

**Output**: `results_wide_w8a16_h100/` — 4 models × 5 benchmarks = 20 expert_counts.json. **Still on scratch** (not yet uploaded to MediaFlux — see §7 #1).

### 4.5 W4A16 attempt via online MXFP4 (Oct 1) — failed

Sbatch `vllm-moe-eval-h100-w4a16-moe.sbatch`, `--quantization mxfp4`.

| Job ID | State | Wall | Notes |
|---|---|---|---|
| `31859550` | FAILED (1:0) | 22m | `NotImplementedError: No MXFP4 MoE backend supports the deployment configuration. Candidate backends were: ['AITER_MXFP4_MXFP4', 'EMULATION'].` — CUDA H100 SM 9.0 has no compatible MXFP4 MoE kernel in this vLLM build. The Triton kernel IS SM 9.0+ compatible but only routed through the GPT-OSS path, not the generic MXFP4 path. |

**Decision**: dropped online MXFP4. Pivoted to **Option A** (login-side AWQ pre-quant → load with `--quantization moe_wna16`). The online MXFP4 sbatch remains in `examples/eval-moe/spartan/` for reference.

### 4.6 W4A8 attempt via compressed-tensors online (Oct 1) — failed

Sbatch `vllm-moe-eval-h100-w4a8-moe.sbatch`, `--quantization compressed-tensors` with synthetic `compression_config` injected via `--hf-overrides`. The spec was `(weight int4 group_size=128 symmetric, input fp8 per-token dynamic)`.

| Job ID | State | Wall | Notes |
|---|---|---|---|
| `31876919` | FAILED (1:0) | 28m | First spec had `input_activations.strategy="tensor"`, `dynamic=False`. vLLM's W4A8 fp8 matcher (`_is_fp8_w4a8`) requires `strategy="token"` + `dynamic=True`. |
| `31882979` | FAILED (1:0) | 29m | Spec fixed, then W4A8 MoE parameter-name bug: `CompressedTensorsW4A8Fp8MoEMethod.create_weights()` registers `w13_weight_packed` instead of `w13_weight`. |

**Decision**: dropped W4A8 online path. W4A8 (and online W4A16) require pre-quantized int4 checkpoints in this vLLM build.

### 4.7 AWQ pre-quant pipeline (Oct 1 → Oct 3) — 6/12 ckpts succeeded, fixes applied

**Setup**:
- `examples/eval-moe/awq_quantize.py` (358 lines after fixes, llm-compressor oneshot, 3 recipes).
- `examples/eval-moe/spartan/vllm-moe-awq-prequant.sbatch` (gpu-h100, 1 GPU, 720 GB mem, 12h wall after Oct 3 fix).
- Calibration data pre-staged at `${SCRATCH}/datasets/perfectblend/data/*.parquet` + HF cache at `${SCRATCH}/datasets/hf_cache/`.
- Dedicated venv at `${SCRATCH}/venv_awq/` with `llmcompressor@main` (v0.14.x supports transformers 5.x; PyPI's 0.8.1 caps at 4.56).
- Output: `${SCRATCH}/hf_cache_awq/<model_safe>-<recipe>/`. vLLM loads with `--quantization compressed-tensors`.

**Bugs found and fixed** (chronological):

1. **DeepSeek EOFError** (Oct 1 → 30+ failed jobs): `awq_quantize.py:132` lacked `trust_remote_code=True`. transformers v5.x prompts interactively for DeepSeek's custom modeling code. **Fix**: added `trust_remote_code=True` to `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`.

2. **HF_DATASETS_CACHE env var set too late** (Oct 2 → 7 failed jobs): `llmcompressor.transformers.data.TextGenerationDataset.load_dataset` hardcodes `cache_dir=None` when calling `datasets.load_dataset("mlabonne/open-perfectblend")`, so it falls back to `~/.cache/huggingface/datasets/` on the GPU node (empty). **Fix**: in `build_calibration`, return a 3-tuple `(dataset, splits, cache_dir)` when HF cache exists. In `main()`, monkey-patch `TextGenerationDataset.load_dataset` for the duration of `oneshot()` to thread `cache_dir=cache_dir_arg` through to `get_raw_dataset`.

3. **DeepSeek ImportError** (Oct 3 02:05 → 2 failed jobs): Hub's `modeling_deepseek.py` imports `is_torch_fx_available` and `is_torch_greater_or_equal_than_1_13` from `transformers.utils.import_utils`, both renamed/removed in 5.x. **Fix**: 4-line shim at top of `awq_quantize.py`:
   ```python
   import transformers.utils.import_utils as _iu
   if not hasattr(_iu, "is_torch_fx_available"):
       _iu.is_torch_fx_available = lambda: False  # safe: only use site is a guarded `if` block
   if not hasattr(_iu, "is_torch_greater_or_equal_than_1_13"):
       _iu.is_torch_greater_or_equal_than_1_13 = lambda *a, **k: True
   ```

4. **quantize_config.json exit check falsely failed** (Oct 2 → 1 failed job): llm-compressor's `save_pretrained(save_compressed=True)` embeds the quantization config inside `config.json["quantization_config"]` (with `quant_method: "compressed-tensors"`), not as a standalone `quantize_config.json`. vLLM reads the embedded config when launched with `--quantization compressed-tensors`. **Fix**: removed the exit check; commented why.

5. **Sbatch wall time too short** (Oct 3 02:30 → 2 TIMEOUT jobs): Qwen3-awq (48L × 128E × 20 grid) and Mixtral-awq_gptq (56L × 8E × 2 modifiers) need ~5h, the sbatch had 4h. **Fix**: `--time=0-12:00:00` in the sbatch (was `0-04:00:00`).

6. **AWQ n_grid=20 too slow for Qwen3** (compounds issue 5): the 20-point grid search per expert per layer is the dominant cost. **Fix**: `n_grid=10` in the AWQ recipe (halves the AWQ time, only marginal accuracy loss).

**Setup**:
- `examples/eval-moe/awq_quantize.py` (282 lines, llm-compressor oneshot, 3 recipes).
- `examples/eval-moe/spartan/vllm-moe-awq-prequant.sbatch` (gpu-h100, 1 GPU, 720 GB mem, 4h wall).
- Calibration data pre-staged at `${SCRATCH}/datasets/perfectblend/data/*.parquet`.
- Dedicated venv at `${SCRATCH}/venv_awq/` with `llmcompressor@main` (v0.14.x supports transformers 5.x; PyPI's 0.8.1 caps at 4.56).
- Output: `${SCRATCH}/hf_cache_awq/<model_safe>-<recipe>/`. vLLM loads with `--quantization moe_wna16`.

**Submissions** (chronological by ID; all FAILED):

| Job ID | Date | Model | Recipe | Wall | Failure |
|---|---|---|---|---|---|
| `31918961` | Oct 1 22:43 | OLMoE | (no RECIPE exported → defaulted to `awq_gptq` first time) | 5s | Awkward venv build error — likely first run that built `venv_awq` |
| `31918962-64` | Oct 1 22:44 | DeepSeek / Mixtral / Qwen3 | (also default awq_gptq) | 5s each | Venv build errors / first-run races |
| `32045833` | Oct 2 09:47 | OLMoE | awq | 1m49s | Died at `[awq-quant] loading model ...` (no further output; OOM or load failure on 1 GPU) |
| `32045834` | Oct 2 09:47 | DeepSeek | awq_gptq | 15s | **`EOFError: EOF when reading a line`** — DeepSeek needs `trust_remote_code=True`, but `awq_quantize.py` line 132 doesn't pass it. vLLM's `LLM` does (we set `trust_remote_code=True` there). |
| `32047523-30` | Oct 2 09:48–09:52 | (8 attempts across all 4 models × 2 recipes) | awq/gptq | 15s each | All DeepSeek runs hit same `EOFError`. Non-DeepSeek runs died at model-load (OOM on 1 GPU + 720 GB mem). |
| `32049490-92`, `32052348-56`, `32064384-91`, `32080788-95`, `32082384-86`, `32082693`, `32082710-17`, `32087893`, `32089931-32` | Oct 2 10:00–17:31 | (mix of resubmits) | various | <30s | Same EOFError / load-time OOM pattern repeated. |

**Currently PENDING (Q-ID order, Oct 2 17:20–17:31 submissions, gpu-h100 PENDING "Priority" — all blocked by the same EOFError/OOM, no fix applied yet)**:

| Job ID | Model | Recipe | Estimated Start |
|---|---|---|---|
| `32089933` | deepseek-ai/deepseek-moe-16b-chat | awq | 2026-10-02T17:31:57 (QALR — first available) |
| `32089934` | deepseek-ai/deepseek-moe-16b-chat | gptq | 2026-10-02T17:42:08 |
| `32089935` | Qwen/Qwen3-30B-A3B | awq | 2026-10-02T21:31:57 |
| `32089936` | Qwen/Qwen3-30B-A3B | gptq | 2026-10-02T21:50:00 |
| `32090053` | mistralai/Mixtral-8x22B-Instruct-v0.1 | awq | 2026-10-03T01:40:00 |
| `32090055` | mistralai/Mixtral-8x22B-Instruct-v0.1 | gptq | 2026-10-03T01:50:00 |
| `32090597` | allenai/OLMoE-1B-7B-0924-Instruct | awq | 2026-10-03T01:57:10 |
| `32090679` | allenai/OLMoE-1B-7B-0924-Instruct | gptq | 2026-10-03T05:40:00 |
| `32090688` | mistralai/Mixtral-8x22B-Instruct-v0.1 | awq_gptq | 2026-10-03T05:50:00 |

**Root cause of every failure**:
1. **DeepSeek runs** (always fail): `AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.bfloat16)` in `awq_quantize.py:132` lacks `trust_remote_code=True`. `transformers` v5.x interactively prompts for "run custom code? [y/N]" on DeepSeek's repo. SLURM jobs have no TTY → `EOFError: EOF when reading a line`.
2. **Non-DeepSeek runs** (consistently fail at model-load on 1 GPU): the bf16 weight tensor must be loaded to RAM before `oneshot()`. Mixtral 262 GB on a 720 GB-mem node PLUS the linearization buffer PLUS tokenizer + venv + activation tensors during AWQ calibration overflows. The "Compressing model on CPU instead" warning indicates `llmcompressor` is detecting "no accelerator" (the SBATCH env says GRES=gpu:1 but the script's `get_main_device` is choosing CPU because no module-loaded CUDA toolkit is on the PATH inside the Python process). When AWQ calibration runs on CPU, RAM pressure is extreme.

**Action needed** (priority order):
1. **One-line patch**: add `trust_remote_code=True` to the `AutoModelForCausalLM.from_pretrained` call in `awq_quantize.py:132`:
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       args.model_id,
       dtype=torch.bfloat16,
       trust_remote_code=True,   # <-- this is the missing argument
   )
   ```
   Also pass `trust_remote_code=True` to `AutoTokenizer.from_pretrained(args.model_id)` at line 224.
2. **Investigate the "No accelerator" warning** — `llmcompressor` `get_main_device` is choosing CPU despite `gpu:1` allocation. The venv uses `torch==2.13.0+cu129` which has CUDA — but the Python process can't find the module-loaded CUDA toolkit at runtime. Could be fixed by exporting `CUDA_HOME=/usr/local/cuda` or similar inside the venv, OR by passing `device_map="cuda:0"` explicitly to `from_pretrained`. (Or the AWQ venv install needs `cuda-toolkit` re-installed in the venv site-packages, which `llmcompressor==0.14.x` pulls via the `cuda-toolkit[cublas,...]` extra — see the `nvidia-cuda-cupti==13.0.85` line in the venv build log, which suggests the venv has its own embedded CUDA 13.0 libraries even though the driver is 12.8. May need `--extra-index-url https://download.pytorch.org/whl/cu129` to override, or downgrade `torch` to match the module's CUDA toolkit.)
3. **Cancel the 9 PENDING jobs** (`scancel 32089933 32089934 32089935 32089936 32090053 32090055 32090597 32090679 32090688`) and re-submit after the fix.
4. **Optional: Mixtral-specific path** — even with the trust_remote_code fix, Mixtral 262 GB bf16 likely OOMs at `AutoModelForCausalLM.from_pretrained` on a 720 GB-mem node because of the linearization buffer + activation tensors during calibration. Options: (a) submit Mixtral jobs with `--mem=1500G` if SLURM allows, (b) split Mixtral into layerwise chunks (llm-compressor doesn't natively support this), (d) skip the Mixtral prequant and acknowledge the gap (bf16/fp8/w8a16 already cover the weight-compression spectrum).
5. **After prequant works**: re-validate by loading each output dir with `transformers.AutoConfig.from_pretrained(output_dir)` and checking `quantize_config.json` declares `quant_method="awq"` (or `"gptq"`).

**Output state on scratch**: `/data/scratch/projects/uom00014/vllm/hf_cache_awq/` contains 10 empty sub-dirs (`allenai--OLMoE-1B-7B-0924-Instruct{,-awq,-gptq}`, `deepseek-ai--deepseek-moe-16b-chat{,-awq,-gptq}`, `Qwen--Qwen3-30B-A3B-{awq,gptq}`, `mistralai--Mixtral-8x22B-Instruct-v0.1-{awq,gptq}`). All are 512 bytes (just `.` and `..`). **No checkpoints anywhere**. Safe to `rm -rf` before re-running.

## 5. Operational details and gotchas

### 5.1 Weight-name conflict (Qwen3-30B-A3B)

`AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B")` looks for `hf_cache/models--Qwen--Qwen3-30B-A3B/`. Our snapshot lives at `hf_cache/hub/models--Qwen--Qwen3-30B-A3B/`. vLLM's engine has its own offline-mode rewrite that handles this, but `transformers.AutoConfig` does not — so the analyzer's pre-flight `transformers.AutoConfig.from_pretrained` call raises `LocalEntryNotFoundError`.

**Fix**: Created symlink `hf_cache/models--Qwen--Qwen3-30B-A3B → hub/models--Qwen--Qwen3-30B-A3B`. Also added `_resolve_local_snapshot(model_id)` in the analyzer that accepts both layouts and passes the snapshot path directly to `AutoConfig.from_pretrained`. Symlink is essential — the analyzer code path alone may not be enough in some edge cases.

### 5.2 Layerwise online quant memory spike (Mixtral w8a16)

The online int8 quant path holds the bf16 weight tensor in memory WHILE the int8 quant allocates a parallel int8 tensor. For Mixtral-8x22B at TP=1, bf16 weight: 262 GB → bf16 + int8 spike: ~393 GB peak → doesn't fit in 80 GB. At TP=2, per-GPU bf16 + spike exceeds 80 GB. At TP=4 with meta-device + layerwise quant (which frees bf16 memory after quant), per-GPU w8a16 weights 16.4 GB + spike 24 GB → fits 80 GB with 55 GB KV headroom. **TP=4 is the minimum for Mixtral w8a16 online quant** (consistent across jobs 31503046 failed at TP=2, 31516295 succeeded at TP=4).

### 5.3 Scratch quota expansion

In late Sep, the project quota was expanded from 466 GB to 1.4 TB. This was a real (not a quota!) cap, so user-driven cleanup was needed before the expansion. The original 466 GB quota triggered multiple "delete some unwanted result files" steps. Now at 730 GB / 53% used, with 668 GB free.

### 5.4 GPU node module + CUDA toolkit mismatch

GPU nodes have driver CUDA 12.8 (driver 570.211.01). Module-loaded toolkit is CUDA 12.4.1. PyTorch wheel cu129 is forward-compatible — fine for runtime. But vLLM's source build requires the module-loaded CUDA toolkit, NOT the driver CUDA. The sbatch does:
```bash
module load "${GCC_MODULE}"  # GCCcore/11.3.0
module load "${PYTHON_MODULE}"  # Python/3.11.3
module load "${CUDA_MODULE}"  # CUDA/12.4.1
```

The venv install uses `uv pip install ... --torch-backend=cu129` to install the cu129 torch wheel. Source build of vLLM from `compile.sh` would compile against the module CUDA 12.4.1's headers/libs — but the package builds the prebuilt CUDA wheel via `VLLM_USE_PRECOMPILED=1`.

The AWQ prequant sbatch uses a **separate** venv (`venv_awq/`) that installs `llmcompressor@main` from GitHub. `llmcompressor==0.14.x` pulls `cuda-toolkit[cublas,cudart,...]==13.0.3` as a dependency (visible in the build log: `nvidia-cuda-cupti==13.0.85`, `nvidia-curand==10.4.0.35`, etc.). The driver is 12.8 (version 12080) — torch can use it because cu129 is forward-compatible, but if llm-compressor tries to call any CUDA 13.0-only API at runtime it will fail. So far the AWQ runs haven't gotten that far (they fail at `from_pretrained` time).

### 5.5 Venv + uv setup

**Analyzer venv** at `${SCRATCH}/venv/`:
```bash
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install numpy datasets==4.5.0
VLLM_USE_PRECOMPILED=0 uv pip install \
    --python "${VENV_DIR}/bin/python" \
    -e "${REPO_ROOT}" \
    --torch-backend=cu129
```

**AWQ venv** at `${SCRATCH}/venv_awq/` (built inside the sbatch on first run):
```bash
python3 -m venv "${AWQ_VENV_DIR}"
source "${AWQ_VENV_DIR}/bin/activate"
python3 -m pip install --upgrade pip wheel uv
UV_CACHE_DIR="${REPO_ROOT}/.uv_cache" \
    uv pip install \
        --torch-backend cu129 \
        "torch==2.13.0" \
        "torchvision==0.28.0+cu129" \
        "transformers==5.15.1" \
        "datasets>=4.8.4,<=5.0.1" \
        "numpy>=2.0.0,<=2.5.3" \
        "huggingface-hub" \
        "accelerate" \
        "safetensors" \
        "compressed-tensors" \
        "pillow>=10.4.0"
UV_CACHE_DIR="${REPO_ROOT}/.uv_cache" \
    uv pip install \
        "git+https://github.com/vllm-project/llm-compressor.git@main"
```

**Why `datasets==4.5.0` is pinned in the analyzer venv**: the login-node pre-download writes each split with system Python 3.9's `datasets 4.5.0`. The GPU venv must use the same version to avoid `KeyError: '_indexes'` on `Dataset.load_from_disk`. (Earlier `datasets 1.1.1` mislabeled package produced this.) The AWQ venv uses `>=4.8.4,<=5.0.1` because `llm-compressor@main` requires that range.

`UV_CACHE_DIR="${REPO_ROOT}/.uv_cache"` (NOT on scratch) — uv's wheel cache is ~10-15 GB and would blow past scratch quota. Project GPFS has plenty of room.

`UV_BUILD_PARALLEL=8` (default cap) — capped from auto-detect 32 because NVIDIA's cicc compiler RSS spikes to ~10 GB per process during CUTLASS file builds. With 32-way parallelism the cgroup OOM-kills cicc.

### 5.6 SLURM poller setup

Background poller for job notifications at `~/.hermes/scripts/slurm_job_poll.py` + per-profile wrapper scripts at `~/.hermes/scripts/slurm_job_poll__<profile>.sh`. Config at `~/.hermes/config/slurm_poller.json` with per-profile sections tracking job IDs and state files. Cron entries installed via `hermes cron create "every 2m" --no-agent --script <wrapper>`.

User explicitly said "no need to start the gateway or the poller" for the W4 attempts and the AWQ pipeline, so poller updates were skipped for those. AWQ prequant state is being monitored via `squeue` / `scontrol` directly.

## 6. Current state

### 6.1 Disk

| Volume | Used | Free | Notes |
|---|---|---|---|
| `/data/scratch/projects/uom00014/vllm/` | ~950 GB | ~450 GB | 1.4 TB quota. Results on scratch: w8a16_h100 (13 GB) + 6 valid int4 W4A16 compressed-tensors checkpoints (~165 GB) in `hf_cache_awq/`. The 4 empty dirs from failed attempts are also there (delete-able). |
| `/data/gpfs/projects/uom00014/vllm/` (project) | ~5 GB | 447 GB | Source repo, venv, .uv_cache |

### 6.2 Result trees

| Tree | Status | On scratch? | On MediaFlux? |
|---|---|---|---|
| `results_wide_bf16_h100/` | COMPLETE (4×5=20 jsons) | No (deleted) | Yes (correct path + doubled-path dupe per §4.3.1) |
| `results_wide_fp8_h100/` | COMPLETE (4×5=20 jsons) | No (deleted) | Yes (correct path + doubled-path dupe per §4.3.1) |
| `results_wide_fp8_dl/` | COMPLETE (1×5=5 jsons, OLMoE only, deeplearn partition) | No (deleted) | Yes (correct path) |
| `results_wide_w8a16_h100/` | COMPLETE (4×5=20 jsons) | **Yes (13 GB)** | No (not yet uploaded) |
| `results_wide_w4a16_awq_h100/` | 6/12 int4 ckpts (see §6.3) | Partial (165 GB in `hf_cache_awq/`) | n/a — these are pre-quant input checkpoints, not analyzer results |

### 6.3 AWQ pre-quant coverage matrix (as of Oct 3 06:00)

| Model | awq | gptq | awq_gptq |
|---|---|---|---|
| OLMoE-1B-7B | ✅ 3.6G (32108092) | ✅ 3.6G (32112652) | — |
| DeepSeek-MoE-16B | ❌ pending (32201642) | ❌ pending (32201643) | — |
| Qwen3-30B-A3B | ❌ pending (32201644) | ✅ 15.6G (32112655) | — |
| Mixtral-8x22B | ✅ 68.1G (32112656) | ✅ 68.1G (32112657) | ✅ 68.1G (32108095) — was marked FAILED but checkpoint written before exit-check fired |

All 6 valid checkpoints are **compressed-tensors format** (config.json has `quant_method: "compressed-tensors"`, `format: "pack-quantized"`, `num_bits: 4`, `group_size: 128`, `symmetric: True`). Load with `vllm LLM(model=..., quantization="compressed-tensors")` — dispatches to `CompressedTensorsWNA16MoEMethod` for routed experts.

### 6.4 Working tree (uncommitted as of Oct 3 06:00)

- `CMakeLists.txt` +2 lines (CUDA 12.8 floor for cooperative_topk MoE kernel gate)
- `examples/eval-moe/moe_expert_stats.py` +127/-19 lines (4 helpers + 1 CLI flag + plumbing for `model_type`)
- `examples/eval-moe/awq_quantize.py` (created Oct 1, multiple iterations)
- `examples/eval-moe/spartan/vllm-moe-awq-prequant.sbatch` (created Oct 1, --time bumped 4h → 12h on Oct 3)
- New untracked files:
  - `examples/eval-moe/awq_quantize.py`
  - `examples/eval-moe/spartan/vllm-moe-awq-prequant.sbatch`
  - `examples/eval-moe/spartan/vllm-moe-eval-h100-w8a16-mixtral-tp{2,4}.sbatch`
  - `examples/eval-moe/spartan/vllm-moe-eval-h100-w{4a8,4a16}-moe.sbatch`
  - `examples/eval-moe/spartan/logs/vllm-moe-awq-prequant-31{9,32}*.{out,err}` (60+ files from Oct 1–3 attempts)

### 6.5 Active SLURM jobs (Oct 3 06:00)

| JobID | Model | Recipe | State | Wall | Notes |
|---|---|---|---|---|---|
| 32201642 | DeepSeek-MoE-16B | awq | RUNNING | 0:00:02 | Just started, validating the v5.x is_torch_fx_available shim |
| 32201643 | DeepSeek-MoE-16B | gptq | PENDING | — | Priority |
| 32201644 | Qwen3-30B-A3B | awq | PENDING | — | 12h wall, n_grid=10; ETA ~3-5h wall |

## 7. Open items / next steps (in priority order)

1. **Fix `awq_trust.py` and re-submit the 9 PENDING AWQ prequant jobs** (one-line patch + investigate CPU/GPU device issue).
   - Add `trust_remote_code=True` to `awq_quantize.py:132` (DeepSeek EOFError fix).
   - Investigate "Compressing model on CPU instead" warning (likely needs `CUDA_HOME` env or `device_map="cuda:0"` override).
   - Cancel the 9 PENDING jobs and re-submit after fix.
   - See §4.7 for details.

2. **Upload `results_wide_w8a16_h100/` to MediaFlux** (13 GB, 20 jsons). Use:
   ```bash
   bash ~/.hermes/scripts/upload-eval-moe-results.sh results_wide_w8a16_h100
   ```
   (The corrected script with `--exclude-parent` will land at `/projects/proj-4180_momqe_data-1128.4.2052/results_wide_w8a16_h100/`.)

3. **Optional: clean up MediaFlux doubled-path duplicates** for bf16_h100 + fp8_h100 (~26 GB total, intentionally kept per user). Need `aterm` interactive session — `asset.destroy :id <id>` for each. Requires TTY, can't be done from a non-TTY terminal session.

4. **Run cross-quant Jaccard sweeps** from the sister llama-cpp-eval repo against the bf16/fp8/w8a16 trees. After #1 and #2 land.

5. **If W4 datapoint is wanted (after #1 lands)**:
   - Path A (the current approach): pre-quantize bf16 → AWQ/GPTQ/AWQ+GPTQ on the gpu-h100 node via `awq_quantize.py`. Once #1 lands, run the 4 models × 3 recipes = 12 prequant jobs. Then load each with `--quantization moe_wna16` via the analyzer.
   - Path C (defer): bf16/fp8/w8a16 likely answers the weight-compression question already. The W4 datapoint is "nice to have" for the paper, not blocking.

6. **Memory consolidation**: auto-clear `~/.hermes/memory/MEMORY.md` is repeatedly blocked by the consolidator (3-4 retries per turn). Memory entry is ~3.3k chars vs 2.2k budget. Could be done by removing now-obsolete entries (the "W4A in flight" entry is stale since #1 hasn't landed yet, but the Mixtral download entry is current).

## 8. Job ID index

| ID | Date | Status | Purpose |
|---|---|---|---|
| 29695741 | Aug 28 | done | Deeplearn |
| 29695743 | Aug 28 | done | gpu-a100 |
| 29744397 | Aug 28 | done | gpu-a100 rerun |
| 29778585 | Aug 31 | done | deeplearn-dlg6 fp8 OLMoE |
| 29778587 | Aug 31 | failed | gpu-h100 TP=2 DeepSeek-MoE (PTX intrinsic missing) |
| 29814285 | Aug 31 | done | RERUN with TORCH_CUDA_ARCH_LIST=8.0 |
| 30542271 | Sep 16 | done | (older) |
| 30635801 | Sep 16 | done | gpu-h100 bf16 wide |
| 30638495 | Sep 16 | failed | gpu-h100 bf16 Mixtral TP=1 |
| 30702863 | Sep 20 | done | gpu-h100 bf16 Mixtral TP=4 |
| 31052007 | Sep 23 | failed | gpu-h100 W8A16 wide (os err 122) |
| 31186052 | Sep 24 | failed | gpu-h100 W8A16 wide (90% quota guard) |
| 31273232 | Oct 1 | failed | gpu-h100 W8A16 Qwen3 AutoConfig failure |
| 31280753 | Oct 1 | failed | gpu-h100 W8A16 Mixtral OOM TP=1 |
| 31425093 | Oct 1 | failed | gpu-h100 W8A16 Mixtral OOM TP=1 gpu_mem=0.85 |
| 31503046 | Oct 1 | failed | gpu-h100 W8A16 Mixtral "no KV cache memory" TP=2 |
| 31516295 | Oct 1 | done | gpu-h100 W8A16 Mixtral TP=4 (all 4/4) |
| 31859550 | Oct 1 | failed | gpu-h100 W4A16 wide (MXFP4 no kernel on H100) |
| 31876919 | Oct 1 | failed | gpu-h100 W4A8 wide (input_activations spec mismatch) |
| 31882979 | Oct 1 | failed | gpu-h100 W4A8 wide (param-name mismatch, requires pre-quant ckpt) |
| 31918961-64 | Oct 1 | failed | AWQ prequant (first attempt, venv build issues) |
| 32045833-34 | Oct 2 | failed | AWQ prequant (OLMoE + DeepSeek) |
| 32047523-30 | Oct 2 | failed | AWQ prequant (4 models × 2 recipes) |
| 32049490-92, 32052348-56, 32064384-91, 32080788-95, 32082384-86, 32082693, 32082710-17, 32087893, 32089931-32 | Oct 2 | failed | AWQ prequant (resubmissions, same EOFError) |
| 32089933-936, 32090053-55, 32090597, 32090679-88 | Oct 2 17:20–17:31 | **PENDING** | AWQ prequant (9 jobs, awaiting fix — §7 #1) |

## 9. File pointers (for whoever picks this up)

- Working tree changes: `git diff examples/eval-moe/moe_expert_stats.py CMakeLists.txt`
- AWQ pre-quant script: `examples/eval-moe/awq_quantize.py` (needs the §7 #1 patch)
- AWQ pre-quant sbatch: `examples/eval-moe/spartan/vllm-moe-awq-prequant.sbatch`
- Failed AWQ run logs (Oct 1–2): `examples/eval-moe/spartan/logs/vllm-moe-awq-prequant-{31918961..32089933}.{out,err}` (40+ files)
- Empty prequant output dirs on scratch: `/data/scratch/projects/uom00014/vllm/hf_cache_awq/{,-awq,-gptq}/*` (safe to `rm -rf`)
- New untracked analyzer sbatches: `examples/eval-moe/spartan/vllm-moe-eval-h100-w{8a16-mixtral-tp{2,4},4a8,4a16}-*.sbatch`
- Failed-run logs (older): `examples/eval-moe/spartan/logs/vllm-moe-eval-31{276,283,320,403,4..}*.{out,err}`
- MediaFlux config: `~/.Arcitecta/mflux.cfg` (mode 0600)
- Poller config: `~/.hermes/config/slurm_poller.json`
- Upload helper: `~/.hermes/scripts/upload-eval-moe-results.sh`
- Memory (terse, auto-injected): `~/.hermes/memory/{MEMORY.md,USER.md}` (stale; see §7 #6)
- Sister repo (cross-archive Jaccard aggregator): `/data/projects/uom00014/llama-cpp-eval/`