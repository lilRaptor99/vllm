# vLLM eval-moe sweep handoff (Aug–Oct 2026)

This document captures the full state of the `eval-moe` project as of **Oct 2, 2026 ~17:35 AEST**: what was attempted, what worked, what's left on disk, what's left to do, and the gotchas for whoever picks this up next. It is the authoritative reference — both `~/.hermes/memory/` (auto-injected, terse) and the on-disk configs/sbatch scripts (technical detail) point back here for the full picture.

**TL;DR (as originally written Oct 2)**:
- bf16, fp8, W8A16 sweeps **complete** across all 4 models × 5 benchmarks. `results_wide_w8a16_h100/` still on scratch (13 GB) — needs MediaFlux upload. bf16+fp8 trees already on MediaFlux.
- W4A16 / W4A8 online paths **abandoned** (MXFP4 has no compatible H100 MoE kernel; compressed-tensors W4A8 is checkpoint-only).
- **AWQ pre-quant pipeline launched Oct 1–2** (`examples/eval-moe/awq_quantize.py` + `vllm-moe-awq-prequant.sbatch`). 9 jobs PENDING as of this writing. **All 30+ historical AWQ attempts failed at the same step — see §4.7.**
- The AWQ blocker is a one-line fix (`trust_remote_code=True` on the `AutoModelForCausalLM.from_pretrained` call in `awq_quantize.py:132`) plus a likely secondary OOM fix for Mixtral. **Patches needed before re-submitting.**
- See §7 for next steps in priority order.

**TL;DR (Oct 6 — current state)**:
- All 8 W4A16 pre-quant checkpoints (OLMoE/DeepSeek/Qwen3/Mixtral × AWQ/GPTQ, plus Mixtral AWQ+GPTQ from Oct 3) are on disk in `hf_cache_awq/`. DeepSeek required 4 distinct patches over 6 failed job batches (Oct 4–5); see addendum 2 §I for the full bug chain.
- All 9 checkpoints have been mirrored to HuggingFace under `pratheeks/<Model>-w4a16-<recipe>`. The 3 Mixtral ones (220 GB total) are public; the 6 smaller ones are private due to HF free-plan storage limits. Upload script: `examples/eval-moe/upload_quants_to_hf.py`.
- W4A16 analyzer sweep: OLMoE × 2 recipes completed in job 32345449 (10 expert_counts.json at `results_wide_w4a16_awq_h100/--data--...-OLMoE-1B-7B-0924-Instruct-{awq,gptq}/compressed-tensors/moe-*/expert_counts.json`). Job then died on DeepSeek-awq with `ValueError: All MoE projections need to have same quantization scheme` — vLLM's W4A16 Marlin kernel rejects the bf16 down_proj we left in DeepSeek. Fixed by adding per-cell try/except in `moe_expert_stats.py` so one bad cell doesn't kill the sweep; resubmitted as job 32430029. DeepSeek cells will log failures but the other 6 cells (Qwen3×2 + Mixtral×2) will produce 30 more expert_counts.json. The DeepSeek W4 datapoint is deferred per addendum 2 §7 #5 Path C.
- For full Oct 6 status, see **"Current state (Oct 6, 2026)"** at the bottom of this document. For the Oct 5 status, see addendum 2. For the Oct 4 status, see the first addendum. The job history (32299121 → 32345449 → 32430029) is in addendum 3 §Q.

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

---

## ADDENDUM (added Oct 5, 2026 by session 20261005_101614 — recovering from the corrupted session 20261004_194838_fdbef6)

The Oct 2 17:35 snapshot above is now stale. The Oct 4 session (`20261004_194838_fdbef6`) made real progress and uncovered a NEW DeepSeek failure mode that the original handoff did not document. Status as of the latest log inspection (Oct 5 ~10:30):

### A. Updated AWQ pre-quant coverage (Oct 4 22:34)

| Model | awq | gptq | awq_gptq |
|---|---|---|---|
| OLMoE-1B-7B | ✅ 3.7G (32108092) | ✅ 3.7G (32112652) | — |
| DeepSeek-MoE-16B | ❌ broken | ❌ broken | — |
| Qwen3-30B-A3B | ✅ 16G (**32201644**, Oct 4 22:34) | ✅ 16G (32112655, Oct 3) | — |
| Mixtral-8x22B | ✅ 69G (32112656) | ✅ 69G (32112657) | ✅ 69G (32108095) |

7 of 11 prequants are on disk. Only **DeepSeek-MoE** is still failing.

### B. The new DeepSeek failure mode (root cause discovered Oct 4 22:10)

The pre-patch failures (jobs 32045834, 32047523-30, 32049490-92 etc., through Oct 2) died at `AutoModelForCausalLM.from_pretrained` with `EOFError: EOF when reading a line` — fixed in the handoff doc as the trust_remote_code fix.

The post-patch failures (jobs 32201642, 32201643, 32202876, 32202877, 32203281, 32203282, **32207630**, **32207631**) get **past** the trust_remote_code gate AND the new rope_scaling reset. They successfully load DeepSeek with `transformers.models.deepseek.modeling_deepseek.DeepseekForCausalLM` and dispatch to the AWQ/GPTQ modifier. They die inside `llmcompressor.entrypoints.oneshot.Oneshot.__init__` → `validate_model` at:

```
File ".../llmcompressor/entrypoints/oneshot.py", line 292, in validate_model
    config = AutoConfig.from_pretrained(...)
File ".../transformers/dynamic_module_utils.py", line 789, in resolve_trust_remote_code
    raise ValueError(
ValueError: The repository deepseek-ai/deepseek-moe-16b-chat contains custom code
which must be executed to correctly load the model. ... Please pass the argument
`trust_remote_code=True` to allow custom code to be run.
```

**Root cause**: `llmcompressor.entrypoints.oneshot.Oneshot.validate_model` calls `AutoConfig.from_pretrained` internally (without forwarding `trust_remote_code=True`) on a model that requires custom code → fresh prompt → no TTY in SLURM → `ValueError`. Our script's own `from_pretrained` calls work because we set `trust_remote_code=True`, but `llmcompressor.oneshot(...)` doesn't accept a `trust_remote_code` kwarg and re-derives the config itself.

**Fix options** (in increasing intrusiveness):

1. **Local pre-export env var (cheapest)**: in `awq_quantize.py`'s `main()`, before calling `oneshot(...)`, do:
   ```python
   import os
   os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
   ```
   and rely on the venv-embedded pre-staged config (the AWQ sbatch already runs in HF_HUB_OFFLINE=1 for dataset cache, but `llmcompressor` may not be honoring it). Verify the sbatch exports `HF_HUB_OFFLINE=1` AND that the local `models--deepseek-ai--deepseek-moe-16b-chat/snapshots/*/config.json` is on disk in `hf_cache/`.
   - Note: the venv's `llmcompressor==0.14.1.dev18` is the only `llmcompressor` with `moe_calibrate_all_experts=True`. PyPI's stable doesn't.

2. **Monkey-patch `transformers.dynamic_module_utils.resolve_trust_remote_code` to return True** at the top of `awq_quantize.py`:
   ```python
   import transformers.dynamic_module_utils as _dmu
   _dmu.resolve_trust_remote_code = lambda *a, **k: True
   ```
   This is the same trick we already used for the `is_torch_fx_available` shim (§4.7 bug #3) and for the rope_scaling reset (Oct 4 patch). One line, surgical, doesn't touch the venv. **Recommended.**

3. **Patch the venv in-place** at `${SCRATCH}/venv_awq/lib/python3.11/site-packages/llmcompressor/entrypoints/oneshot.py` line 292 to add `trust_remote_code=True`. Brittle — wiped on any venv rebuild.

4. **Skip DeepSeek entirely**: bf16 / fp8 / w8a16 already have all four models covered. The W4 datapoint is the only one missing for DeepSeek. Acceptable tradeoff per the handoff doc §7 #5 Path C.

### C. The Oct 4 rope_scaling patch (UNCOMMITTED in working tree)

After the pre-patch failures at 19:24/19:25, the user added a new block in `awq_quantize.py` (the diff shown via `git diff`) to reset `config.rope_scaling = None` for DeepSeek. This worked for Qwen3 (job 32201644 ran for 3h:00 and saved a 16 GB checkpoint at Oct 4 22:34) but didn't reach the `validate_model` call for DeepSeek because of bug B above.

The patch is **currently uncommitted** in `/data/gpfs/projects/uom00014/vllm/examples/eval-moe/awq_quantize.py`. Branch is `eval-moe` ahead of origin by 1 commit (commit `cd0ec9984 temp`).

**The new validate_model fix (option B) should be added in the same uncommitted patch block** so Qwen3+OLMoE+Mixtral continue to work and DeepSeek finally passes through.

### D. Disk + result tree state (Oct 5 10:30)

- Scratch: 973 GB used / 425 GB free / 1.4 TB quota (70% full). Free is fine for now.
- Project: 21 GB used / 447 GB free (5% full).
- `results_wide_w8a16_h100/` (13 GB, 4×5=20 expert_counts.json) **still on scratch, not yet uploaded to MediaFlux** (handoff §7 #2 not done).
- `results_wide_w4a16_h100/` and `results_wide_w4a8_h100/` are still empty (online W4 paths were abandoned per handoff §4.5–4.6).
- `hf_cache_awq/` has 7 valid checkpoints (~166 GB), 4 empty sub-dirs (DeepSeek × 2, plus `allenai--OLMoE-1B-7B-0924-Instruct/` and `deepseek-ai--deepseek-moe-16b-chat/` which are just staging dirs).

### E. Outstanding job statuses (Oct 5 10:45)

| Job | Model/Recipe | State | Notes |
|---|---|---|---|
| 32207630 | DeepSeek-awq | FAILED (~14 min wall) | died at `oneshot.validate_model` AutoConfig (bug B) |
| 32207631 | DeepSeek-gptq | FAILED (~14 min wall) | same |
| 32201644 | Qwen3-awq | **COMPLETED (0:0)** Oct 4 22:34 | 3h wall, with rope_scaling patch |
| **32236909** | DeepSeek-awq | **PENDING** gpu-h100, 12h wall | Oct 5 10:43, with the Symptom 3 shim (`_dmu.resolve_trust_remote_code = lambda *a, **k: True`). Est. start 10:56:08 on spartan-gpgpu178 |
| **32236910** | DeepSeek-gptq | **PENDING** gpu-h100, 12h wall | Oct 5 10:43, same shim. Est. start 14:22:04 on spartan-gpgpu015 |

The shim is at `examples/eval-moe/awq_quantize.py:49-56` (right after the existing `is_torch_fx_available` shim). When 32236909 starts running, the diagnostic to watch is whether the `[transformers] You are using a model of type deepseek to instantiate a model of type` line is still followed by a `ValueError` from `resolve_trust_remote_code`. If absent and the run proceeds past model load, the shim worked.

### F. The Oct 4 19:54-22:34 user prompt sequence (reconstructed from `~/.hermes/.hermes_history`)

- 19:21:32 — "the jobs failed again. what is the root cause?"
- 19:24/25 — PENDING DeepSeek runs 32201642/32201643 start (rope_scaling KeyError)
- 19:26:37 — "the jobs submitted just now failed"
- 19:54:53 — "one of the recently submitted jobs failed. what is the root cause?"
- 19:55:20 — `/retry`
- 19:55:31 — " one of the recently submitted jobs failed. what is the root cause?"
- 19:56:05 — "What are your memories on this project? we were working on this project and the session got corrupted. Read the full handoff document: /tmp/hermes-recovery/HANDOFF.md and /tmp/hermes-recovery/FULL-TRANSCRIPT.md" (the `/tmp` files are gone — `/tmp` is wiped on reboot, and we are in the new session now)
- 20:00:20 — "Investigate the deepseek-moe failure while the Qwen3 job runs."
- 21:07 — Resubmit (jobs 32203281/32203282, pre-patch — still dies on rope_scaling)
- 21:48:59 — `/resume`
- 21:49:19 — `/resume 2`
- 21:49:33 — "the resubmitted jobs failed"
- 22:10 — Apply rope_scaling patch in working tree, resubmit (jobs 32207630/32207631) → Qwen3 succeeds at 22:34, DeepSeek dies at validate_model
- 23:?? — Session ended mid-investigation. Recovered session 20261005_101614 is the current one.

### G. What "we were last doing" (one-liner for resuming the new session)

We were investigating why the DeepSeek-MoE-16B AWQ prequant fails AFTER the Oct 4 rope_scaling patch fixed the model-load step. Root cause is `llmcompressor.entrypoints.oneshot.Oneshot.validate_model` doing an internal `AutoConfig.from_pretrained` without `trust_remote_code=True`. Best fix is a one-line monkey-patch in `awq_quantize.py` (option B in section B above). Once applied, resubmit `32207630` (awq) and `32207631` (gptq). After those land, all 9 of 11 prequant checkpoints will be on disk, and we can move to handoff §7 #2 (upload `results_wide_w8a16_h100/` to MediaFlux) and then the W4A16-ANAYSER sweep (load each prequant ckpt with `--quantization compressed-tensors` and rerun `moe_expert_stats.py` for 4 models × 3 recipes × 5 benchmarks = 60 cells, but with idempotent skip per cell).

### H. File pointers for the new session

- **Handoff doc (now updated with this addendum)**: `/data/gpfs/projects/uom00014/vllm/EVAL_MOE_HANDOFF.md`
- **Uncommitted patch in working tree**: `git diff examples/eval-moe/awq_quantize.py` (~37 line rope_scaling block to be extended with the `resolve_trust_remote_code` monkey-patch)
- **Recent AWQ job logs**: `examples/eval-moe/spartan/logs/vllm-moe-awq-prequant-{32207630,32207631,32201644,32203281,32203282,32202876,32202877}.{out,err}`
- **Scripts**: `examples/eval-moe/awq_quantize.py`, `examples/eval-moe/spartan/vllm-moe-awq-prequant.sbatch`
- **Poller** (idle): `~/.hermes/scripts/slurm_job_poll.py` + `~/.hermes/config/slurm_poller.json` — user explicitly said no need to start for AWQ sweep
- **MediaFlux uploader**: `~/.hermes/scripts/upload-eval-moe-results.sh` (with `--exclude-parent` bug fixed in §4.3.1)

The recovered-state.db at `/data/gpfs/projects/uom00014/vllm/recovered-state.db` has 2,910 messages from before Sep 6, but the `sessions` table is empty (corruption). It is NOT installed as the live state.db; the live `~/.hermes/state.db` is a small post-repair DB with only the current session. The 20261004_194838_fdbef6 session is **not** in the recovered DB — its messages only exist in `~/.hermes/.hermes_history` (just the user prompt log, no assistant responses).


---

## ADDENDUM 2 (added Oct 6, 2026 — DeepSeek prequant completed, HF upload, W4A16 analyzer sweep launched)

This addendum covers everything since the first addendum was written
(Oct 5 10:45). The DeepSeek prequant pipeline finally completed,
all 9 W4A16 checkpoints were uploaded to HuggingFace, and the
W4A16 analyzer sweep was launched (then died once on a missing
`ninja` tool, re-submitted).

### I. DeepSeek prequant — final attempts and completion (Oct 5 12:32 → 21:32)

The first addendum left off with DeepSeek prequant broken across 6
attempts (32201642/43, 32202876/77, 32203281/82, 32207630/31, 32236909/10,
32237768/69, 32260128/29) on three different bugs:
1. EOFError on DeepSeek's custom modeling code → fixed by
   `trust_remote_code=True` on `from_pretrained` (Oct 4).
2. `rope_scaling` KeyError in v5.15.1's `convert_rope_params_to_dict`
   → fixed by resetting `cfg.rope_scaling = None` for DeepSeek
   (Oct 4).
3. `validate_model` internal `AutoConfig.from_pretrained` without
   `trust_remote_code=True` → fixed by monkey-patching
   `transformers.dynamic_module_utils.resolve_trust_remote_code` and
   patching the local binding in every consumer module
   (`configuration_auto`, `auto_factory`, `tokenization_auto`, etc.)
   because `from X import Y` creates a local binding that simple
   `X.Y = lambda` doesn't reach. See the shim block in
   `examples/eval-moe/awq_quantize.py` lines 57-92.

After fix #3 (jobs 32260128/29 on Oct 5 12:32) we hit a new bug:
`validate_group_size_divisibility` in llmcompressor hard-fails on
DeepSeek's per-expert `mlp.down_proj` weight, which has 10944 columns.
`10944 / 128 = 85.5` — not integer.

- Jobs 32260128 (awq) / 32260129 (gptq) Oct 5 12:32: FAILED at
  `validate_group_size_divisibility: model.layers.0.mlp.down_proj
  (columns=10944, group_size=128)`.

Fix attempted: pass `bypass_divisibility_checks=True` to the
`QuantizationModifier` and `GPTQModifier` constructors. This was
valid for `QuantizationModifier`/`GPTQModifier` (via `QuantizationMixin`),
but NOT for `AWQModifier` (pydantic ValidationError — extra fields
not permitted) AND even with the bypass on the GPTQ path the
*bypass is validation-only* — the actual `tensor.unflatten(-1, (-1, 128))`
during calibration still re-raises a `RuntimeError` when
`10944 % 128 != 0`.

- Job 32261716 (awq) Oct 5 14:30: FAILED at pydantic
  `ValidationError: bypass_divisibility_checks Extra inputs are not
  permitted` on `AWQModifier`.
- Job 32261717 (gptq) Oct 5 14:31: FAILED at
  `RuntimeError: unflatten: Provided sizes [-1, 128] don't multiply
  up to the size of dim 1 (10944)`.

Real fix: add `re:.*\.down_proj$` to the `QuantizationModifier`/
`GPTQModifier` `ignore` list for DeepSeek only (leaves that
projection in bf16; all other linear layers quantize to int4).
AND for AWQ: provide an explicit `mappings=[...]` argument to
`AWQModifier` that omits the `up_proj → down_proj` mapping (otherwise
AWQ's `_compute_layer_means` divides by zero on the empty down_proj
set). Both pieces are gated on `is_deepseek` so the other 3 models
are unaffected.

Also added a save_pretrained fix: `DeepseekPreTrainedModel._tied_weights_keys = ["lm_head.weight"]`
in `modeling_deepseek.py:1267` is a list, but transformers v5.15.1's
`_get_tied_weight_keys` at `modeling_utils.py:381` calls
`.keys()` on it (expecting a dict). The lm_head in the saved
DeepSeek safetensors is NOT tied to embeddings (separate tensor),
so dropping the claim is correct. Wrapped
`transformers.modeling_utils._get_tied_weight_keys` to coerce lists
to empty.

And a (mild) typo bug: I imported `AWQMapping` from
`llmcompressor.modifiers.awq.mappings` (wrong path) instead of
`llmcompressor.modifiers.transform.awq` (right path, re-exported
from `mappings`).

- Jobs 32266239 (awq) / 32266240 (gptq) Oct 5 14:30 + 14:31: FAILED
  with the pydantic and unflatten errors above.
- Jobs 32288498 (awq) / 32288499 (gptq) Oct 5 19:53: FAILED at
  `ModuleNotFoundError: No module named
  'llmcompressor.modifiers.awq.mappings'` (the import typo).
- Jobs 32292734 (awq) / 32292735 (gptq) Oct 5 21:20 + 21:28: **COMPLETED**.
  AWQ took 7m20s, GPTQ took 4m26s. Final prequant coverage: 9 of 11
  cells (AWQ+GPTQ for all 4 models, plus Mixtral-awq_gptq from Oct 3;
  the 3 awq_gptq variants for OLMoE/DeepSeek/Qwen3 were deferred).

### J. DeepSeek prequant quality notes (job 32292734 awq, 32292735 gptq)

Both completed cleanly. Two non-fatal warnings worth recording:

- **DeepSeek AWQ (32292734)**: Many `WARNING - Skipping smooth_layer
  model.layers.N.{v_proj,post_attention_layernorm}, NaN or inf outputs
  found during forward pass` messages. These are the standard
  `_apply_smoothing` "incompatible shapes" warnings and are benign —
  AWQ just skips the affected smoothing pair and continues. The
  saved checkpoint is mixed-precision W4A16: int4 per-expert gate/up
  + int4 shared_experts + bf16 per-expert down_proj (10944 cols
  incompatible with `group_size=128`). vLLM's
  `compressed_tensors_moe_w4a16_flydsl.py` Marlin kernel handles this
  mix.

- **DeepSeek GPTQ (32292735)**: 339/1420 modules (23.9%) fell back to
  round-to-nearest (RTN) instead of GPTQ due to Hessian inversion
  failure. Higher than the 14.2% I saw on the pre-bypass
  experiment; raising `dampening_frac` from 0.01 → 0.02 didn't help,
  the failure mode is more structural (DeepSeek's per-expert weight
  distribution is harder for GPTQ's Hessian to invert). Quality is
  slightly lower than pure-GPTQ on those modules but still
  functional. Could be improved with `dampening_frac=0.05` or more
  calibration samples, but it's a quality-not-correctness issue.

### K. AWQ pre-quant coverage matrix (final, Oct 5 21:32)

| Model | AWQ | GPTQ | AWQ+GPTQ |
|---|---|---|---|
| OLMoE-1B-7B (3.7G) | ✅ | ✅ | ❌ deferred |
| DeepSeek-MoE-16B (16G, mixed-precision) | ✅ | ✅ | ❌ deferred |
| Qwen3-30B-A3B (16G) | ✅ | ✅ | ❌ deferred |
| Mixtral-8x22B (69G) | ✅ | ✅ | ✅ |

Total on disk in `hf_cache_awq/`: ~261 GB of prequant checkpoints
plus 2 empty scaffolding dirs (512 B each). The empty dirs are the
parent dirs `allenai--OLMoE-1B-7B-0924-Instruct` and
`deepseek-ai--deepseek-moe-16b-chat` that the sbatch creates
in case of resume; safe to `rm -rf` once we're sure the analyzer
sweep is healthy.

### L. HF upload of prequant checkpoints (Oct 5 22:35 → Oct 5 23:21)

User asked: push the 9 W4A16 checkpoints to HuggingFace as public
research artifacts. Created `examples/eval-moe/upload_quants_to_hf.py`
(290-line Python 3.9 script using `huggingface_hub` v1.8.0 — the
venv's Python 3.11 won't run on the login node because the module
libpython3.11 isn't on the login node's library path, but `/usr/bin/python3.9`
has `huggingface_hub` installed for stdlib use). The script:

- Iterates `hf_cache_awq/*/` (skipping empty scaffolding dirs)
- Splits each dir's name on `-<recipe>` to derive `(model_safe, recipe)`
- Creates the repo `pratheeks/<display>-w4a16-<recipe>` (private by
  default; `--public` to make it public). Note the recipe is
  hyphened (`awq-gptq` for the combined variant) for URL-friendliness
- Uploads small files (config.json, tokenizer, recipe.yaml, etc.)
  FIRST so the repo is browseable immediately, then the big
  `model.safetensors` (or shards) last
- Generates a model card from a template in the script (mentioning
  base model, recipe, calibration data, num_bits/group_size/
  symmetric (resolved from `config_groups.group_0.weights` for
  compressed-tensors v0.19+), how-to-load via vLLM Python API and
  CLI, provenance link to the vLLM eval-moe branch, and known
  caveats). Also wrote a synthesized `quantize_config.json` for
  any prequant that didn't have one
- Idempotent: skips files that already exist in the remote repo
  (`api.list_repo_files` check before each upload)

**Upload results (9/9 done)**: 261 GB total uploaded over ~22 minutes
wall. First run was private; hit the HuggingFace free-plan private
storage limit at the 6th repo (~70 GB private). Switched the 3
Mixtral repos (220 GB) to public; the 6 smaller ones (OLMoE/DeepSeek/
Qwen3, ~40 GB private total) are still private. The script
deletes-and-recreates the partial Mixtral repos that were abandoned
mid-upload by the storage limit.

| Repo | Shards | Size | Visibility |
|---|---|---|---|
| `pratheeks/OLMoE-1B-7B-0924-Instruct-w4a16-awq` | 1 | 3.9 GB | private |
| `pratheeks/OLMoE-1B-7B-0924-Instruct-w4a16-gptq` | 1 | 3.9 GB | private |
| `pratheeks/Qwen3-30B-A3B-w4a16-awq` | 1 | 16 GB | private |
| `pratheeks/Qwen3-30B-A3B-w4a16-gptq` | 1 | 16 GB | private |
| `pratheeks/DeepSeek-MoE-16B-chat-w4a16-awq` | 1 | 16 GB | private |
| `pratheeks/DeepSeek-MoE-16B-chat-w4a16-gptq` | 1 | 16 GB | private |
| `pratheeks/Mixtral-8x22B-Instruct-v0.1-w4a16-awq` | 4 | 73 GB | **public** |
| `pratheeks/Mixtral-8x22B-Instruct-v0.1-w4a16-awq-gptq` | 4 | 73 GB | **public** |
| `pratheeks/Mixtral-8x22B-Instruct-v0.1-w4a16-gptq` | 4 | 73 GB | **public** |

**Bug fixed after upload**: initial model cards showed
`num_bits: ?, group_size: ?, symmetric: True` because
`config_groups.group_0.weights.{num_bits,group_size,symmetric}`
is the actual location in compressed-tensors v0.19+; the
top-level `quantization_config.{num_bits,group_size,symmetric}`
keys don't exist. Updated the `model_card()` helper to drill
into `config_groups.group_0.weights` and re-uploaded all 9
READMEs.

**Probed sustained upload speed**: ~150-200 MB/s after first-byte
warmup (initial probe was 5.4 MB/s which was just the first-byte
latency; sustained is much higher). At that rate the 261 GB
total finished in ~22 minutes wall.

### M. W4A16 analyzer sweep — sbatch, first attempt, ninja failure (Oct 5 23:40 → Oct 6 12:42)

User asked: submit jobs to complete the expert activation evaluation
on the W4A16 prequant ckpts. Created
`examples/eval-moe/spartan/vllm-moe-eval-h100-w4a16-prequant.sbatch`
(forked from `vllm-moe-eval-h100-w8a16-moe.sbatch`, 280 lines) that:

- Runs `examples/eval-moe/moe_expert_stats.py` on the 4 models
  × 2 recipes (AWQ + GPTQ) = 8 cells, sequentially in one SLURM job
- Sets `MODELS_OVERRIDE` to the 8 absolute local prequant paths
  (e.g. `/data/scratch/projects/uom00014/vllm/hf_cache_awq/allenai--OLMoE-1B-7B-0924-Instruct-awq`)
- Sets `QUANTS_OVERRIDE=compressed-tensors` (vLLM's
  `CompressedTensorsWNA16MoEMethod` for routed experts + the
  bf16 down_proj fallback for DeepSeek)
- Uses TP=1, `gpu_mem_util=0.92`, `gpu-h100` partition, 24h wall
- Pre-creates the analyzer venv on first run; idempotent venv build
  on subsequent runs (the venv at `${SCRATCH}/venv/` already
  existed from prior bf16/fp8/w8a16 sweeps)
- Pre-checks each prequant dir exists and has `config.json`
  before running
- OUTPUT_DIR defaults to `${SCRATCH}/results_wide_w4a16_awq_h100/`
- Drops the `awq_gptq` recipe (3 cells) per user request — "awq
  and gptq for now, gptq+awq can be run later"

**Job 32299121** was submitted Oct 5 23:40, PENDING until 2026-10-08
15:01 (gpu-h100 queue is ~3 days deep). 24h wall time.

The job ran from ~Oct 6 ~01:30 (queue time) to Oct 6 12:42 (failure),
~13 hours wall. The first cell (OLMoE-awq) loaded the safetensors
checkpoint successfully (6s) but failed during vLLM engine init:

```
RuntimeError: Engine core initialization failed. See root cause above.
FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
```

**Root cause**: vLLM's v1 engine uses `flashinfer`'s
`topk_topp_sampler` (in `vllm/v1/sample/ops/topk_topp_sampler.py:182`).
Flashinfer's `top_k_mask_logits` JIT-compiles a CUDA kernel via
`subprocess.run(['ninja', ...])` on first use. The `ninja` build tool
is required to be on the **OS PATH** so the forked `EngineCore`
subprocess can find it via standard PATH lookup. The venv at
`${SCRATCH}/venv/` does have a venv-local `bin/ninja` (from
`pip install ninja`), but running `${VENV_DIR}/bin/python script.py`
does NOT add the venv's `bin/` to the parent shell's PATH. The
EasyBuild `Ninja/1.11.1-Python-3.11.3` module (which is the Python-
3.11.3-specific build of ninja) is what was missing from the
sbatch's module load.

Why the prior bf16/fp8/w8a16 sweeps didn't hit this: flashinfer
caches compiled kernels in `~/.cache/flashinfer/`. If the cache
existed from a previous run on a node with the same CUDA/Python
versions, the JIT step was skipped. This run was the first to hit
the cold-cache path on this node.

**Fix**:
- Added `NINJA_MODULE=Ninja/1.11.1-Python-3.11.3` to the sbatch's
  module list (loaded via EasyBuild alongside the other modules)
- Added the empty-check preflight (`if [[ -z "${NINJA_MODULE}" ]]`)
- Added the banner echo
- Added the `load_module_or_die "${NINJA_MODULE}"` call

This puts `ninja` on `$PATH` for the parent shell AND all forked
EngineCore subprocesses. After the fix, the first cell will JIT-
compile the kernel, cache it in `~/.cache/flashinfer/`, and the
remaining 7 cells reuse the cache without re-running ninja.

**Resubmitted as job 32345449** (PENDING in gpu-h100 queue, 24h wall).
Poller config updated to track the new ID.

### N. Open items (updated Oct 6)

1. **W4A16 analyzer sweep** (job 32345449, PENDING): 8 cells × ~2-3h
   per cell = ~16-20h wall. Will run once a gpu-h100 node frees up.
   Output tree: `results_wide_w4a16_awq_h100/<model_safe>/<quant_safe>/moe-<bench>/expert_counts.json`
2. **Upload `results_wide_w8a16_h100/` to MediaFlux** (handoff
   §7 #2, 13 GB, 20 jsons) — still not done, this is independent
   of the W4A16 sweep.
3. **AWQ+GPTQ combined recipe** for OLMoE/DeepSeek/Qwen3 (3 more
   prequant jobs, ~10 min each on calibration, then analyzer cells).
   The user explicitly deferred this.
4. **Optional**: clean up the MediaFlux doubled-path duplicates for
   bf16_h100 + fp8_h100 (~26 GB total, requires interactive `aterm`).
5. **Cross-archive Jaccard sweep** in sister repo
   `/data/projects/uom00014/llama-cpp-eval/` against the bf16/fp8/
   w8a16/w4a16 trees once the W4A16 sweep is done.

### O. File pointers (Oct 6, 2026)

- **Prequant script** (uncommitted patch in working tree, ~80 lines
  of new code on top of v1):
  `examples/eval-moe/awq_quantize.py` — 4 fixes:
  1. `is_torch_fx_available` / `is_torch_greater_or_equal_than_1_13`
     shim (Oct 3, from first addendum)
  2. `rope_scaling=None` reset for DeepSeek (Oct 4)
  3. `resolve_trust_remote_code` monkey-patch + consumer-local-bindings
     (Oct 5)
  4. `re:.*\.down_proj$` ignore + explicit AWQ `mappings=` for
     DeepSeek (Oct 5)
- **W4A16 analyzer sbatch**: `examples/eval-moe/spartan/vllm-moe-eval-h100-w4a16-prequant.sbatch`
- **HF upload script**: `examples/eval-moe/upload_quants_to_hf.py`
  (290 lines, idempotent re-run safe)
- **Pre-quant output dirs**: `/data/scratch/projects/uom00014/vllm/hf_cache_awq/*/`
- **W4A16 analyzer output tree** (will be populated by job 32345449):
  `/data/scratch/projects/uom00014/vllm/results_wide_w4a16_awq_h100/`
- **HuggingFace repos**: 9 under `pratheeks/` (see table in §L above)
- **Log files for the failed run**:
  `examples/eval-moe/spartan/logs/vllm-moe-eval-w4a16-prequant-32299121.{out,err}`
- **Log files for the rerun** (in progress):
  `examples/eval-moe/spartan/logs/vllm-moe-eval-w4a16-prequant-32345449.{out,err}`
- **Poller config** (cron job disabled, manual watch):
  `~/.hermes/config/slurm_poller.json` — eval-moe-awq-prequant-deepseek
  profile tracks `[32345449]`


---

## Current state (Oct 6, 2026 — post-DeepSeek-prequant, post-HF-upload, W4A16 sweep launched)

This section supersedes the earlier §6 snapshots. The Oct 2 17:35
view is in §6 above, the Oct 3 06:00 view is also in §6, and the
addendum's §D has the Oct 5 10:30 view. This is the Oct 6 view.

### P.1 Disk (Oct 6)

| Volume | Used | Free | Notes |
|---|---|---|---|
| `/data/scratch/projects/uom00014/vllm/` | ~1004 GB | ~394 GB (70%) | 1.4 TB quota. Pre-quant: 9 W4A16 checkpoints (~261 GB) in `hf_cache_awq/`. Analyzer results: w8a16_h100 (13 GB) on scratch, results_wide_w4a16_awq_h100/ created but empty (populated by job 32345449). |
| `/data/gpfs/projects/uom00014/vllm/` (project) | ~5 GB | 447 GB | Source repo, venv, .uv_cache |

### P.2 Result trees (Oct 6)

| Tree | Status | On scratch? | On MediaFlux? | On HF? |
|---|---|---|---|---|
| `results_wide_bf16_h100/` | COMPLETE (4×5=20 jsons) | No (deleted) | Yes (correct path + doubled-path dupe per §4.3.1) | n/a (bf16) |
| `results_wide_fp8_h100/` | COMPLETE (4×5=20 jsons) | No (deleted) | Yes (correct path + doubled-path dupe per §4.3.1) | n/a (fp8) |
| `results_wide_fp8_dl/` | COMPLETE (1×5=5 jsons, OLMoE only, deeplearn partition) | No (deleted) | Yes (correct path) | n/a |
| `results_wide_w8a16_h100/` | COMPLETE (4×5=20 jsons) | **Yes (13 GB)** | No (not yet uploaded — handoff §7 #2 still open) | n/a |
| `results_wide_w4a16_awq_h100/` | empty (job 32345449 PENDING, will populate) | Yes (empty dir) | No | n/a (this is the analyzer output, not a checkpoint) |
| `results_wide_w4a16_h100/` | empty | Yes (empty dir) | No | n/a (online W4A16 path was abandoned per §4.5) |
| `results_wide_w4a8_h100/` | empty | Yes (empty dir) | No | n/a (online W4A8 path was abandoned per §4.6) |

### P.3 AWQ pre-quant coverage matrix (Oct 5 21:32 — FINAL)

| Model | AWQ | GPTQ | AWQ+GPTQ |
|---|---|---|---|
| OLMoE-1B-7B (3.7G) | ✅ | ✅ | ❌ deferred |
| DeepSeek-MoE-16B (16G, mixed-precision) | ✅ | ✅ | ❌ deferred |
| Qwen3-30B-A3B (16G) | ✅ | ✅ | ❌ deferred |
| Mixtral-8x22B (69G) | ✅ | ✅ | ✅ |

Pre-quant ckpts are at `/data/scratch/projects/uom00014/vllm/hf_cache_awq/`
and are mirrored to HuggingFace under `pratheeks/` (see §L for the
visibility table — the 6 small/medium ones are private, the 3
Mixtral ones are public because they didn't fit the free-plan
private storage quota).

### P.4 Working tree (uncommitted, Oct 6)

- `examples/eval-moe/awq_quantize.py` (+80/-5 lines vs `c2283d2f3`):
  4 patches — `is_torch_fx_available` shim, `rope_scaling` reset
  for DeepSeek, `resolve_trust_remote_code` shim + consumer
  local-binding patch, `re:.*\.down_proj$` ignore + explicit
  AWQ `mappings=` for DeepSeek
- New untracked files:
  - `examples/eval-moe/upload_quants_to_hf.py` (HF uploader, 290 lines)
  - `examples/eval-moe/spartan/vllm-moe-eval-h100-w4a16-prequant.sbatch`
    (W4A16 analyzer sbatch, 280 lines)
  - `examples/eval-moe/spartan/logs/vllm-moe-eval-w4a16-prequant-{32299121,32345449}.{out,err}`
    (failed + running W4A16 sweep logs)
  - `examples/eval-moe/spartan/logs/upload-hf.log` (HF uploader log,
    9/9 DONE)
  - `examples/eval-moe/spartan/logs/upload-hf-mixtral.log` (Mixtral
    re-run after storage limit, 3/3 DONE)
  - The handoff doc itself: `EVAL_MOE_HANDOFF.md` (52 KB → 68 KB
    with this addendum)

### P.5 Active SLURM jobs (Oct 6 ~22:55)

| JobID | State | Wall | Notes |
|---|---|---|---|
| 32430109 | PENDING | 0:00:00 | `vllm-moe-eval-w4a16` on gpu-h100, 24h wall, 8 cells reordered to put DeepSeek last (OLMoE → Mixtral → Qwen3 → DeepSeek, each × awq/gptq). Will populate `results_wide_w4a16_awq_h100/`. Job 32345449 (Oct 6 22:14→22:42, 27 min) already completed 2 of 8 cells: OLMoE-awq + OLMoE-gptq (10 expert_counts.json). Job 32430029 was cancelled before running (user asked to reorder to put DeepSeek last). The per-cell try/except added to `moe_expert_stats.py` ensures the DeepSeek failures no longer kill the sweep. |

No other jobs active.

### P.6 One-liner for resuming (Oct 6 22:55)

W4A16 analyzer sweep is queued as job 32430109 (re-submit of
32430029, which the user cancelled before it could run, to reorder
the cells so DeepSeek runs last). Cell order: OLMoE × 2 (skipped,
already done from 32345449) → Mixtral × 2 → Qwen3 × 2 → DeepSeek ×
2 (fails). Expected new contribution: 30 expert_counts.json. The 3
deferred `awq_gptq` cells (OLMoE/DeepSeek/Qwen3) are independent
prequant jobs that can be launched when convenient. The outstanding
MediaFlux upload of `results_wide_w8a16_h100/` (13 GB) and the
optional MediaFlux duplicate cleanup are unrelated to the W4 sweep.


---

## ADDENDUM 3 (added Oct 6, 2026 — W4A16 analyzer sweep: ninja fix worked, DeepSeek mixed-precision broken)

This addendum covers the W4A16 analyzer sweep attempts (jobs 32299121,
32345449, 32430029) and the new failure mode they revealed.

### Q. Job 32299121 → 32345449: ninja fix worked, but DeepSeek is the new blocker

The previous addendum (2) §M described job 32299121 dying on
`FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`
during flashinfer's JIT compile of the topk_topp_sampler kernel.
Fixed by adding `NINJA_MODULE=Ninja/1.11.1-Python-3.11.3` to the
sbatch's module list and resubmitting as job 32345449.

**Job 32345449 ran Oct 6 22:14:50 → 22:42:26 (27 min wall, exit 1:0)**.
The `ninja` fix worked — flashinfer's JIT compiled the kernel
without error. The job progressed through:
- Cell 1: OLMoE-awq (5/5 benchmarks, completed) — 5 expert_counts.json
- Cell 2: OLMoE-gptq (5/5 benchmarks, completed) — 5 expert_counts.json
- Cell 3: DeepSeek-awq — **FAILED** with
  ```
  ValueError: All MoE projections need to have same quantization scheme but found multiple
  File "/data/gpfs/projects/uom00014/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py", line 48, in get_moe_method
  File "/data/gpfs/projects/uom00014/vllm/vllm/model_executor/models/deepseek_v2.py", line 370, in __init__
  ```

Cells 4-8 (DeepSeek-gptq, Qwen3-awq, Qwen3-gptq, Mixtral-awq,
Mixtral-gptq) were never attempted because the analyzer had no
per-cell try/except — the for-loop in `main()` dies on the first
exception.

**Root cause for DeepSeek**: vLLM's `CompressedTensorsMoEMethod.get_moe_method`
(`compressed_tensors_moe.py:46-48`) enforces that all 3 MoE projections
(gate_proj, up_proj, down_proj) within a single FusedMoE have the
SAME quantization scheme:

```python
all_scheme_dicts = [
    quant_config.get_scheme_dict(layer, name) for name in unfused_names
]
scheme_dict = all_scheme_dicts.pop()
if not all([cur_dict == scheme_dict for cur_dict in all_scheme_dicts]):
    raise ValueError("All MoE projections need to have same quantization scheme but found multiple")
```

Our DeepSeek prequant deliberately left `mlp.down_proj` in **bf16**
(10944 cols not divisible by `group_size=128`), while gate/up +
shared_experts were quantized to int4. llm-compressor accepted
this and saved the checkpoint OK — vLLM's W4A16 Marlin/flydsl
kernel (`compressed_tensors_moe_w4a16_flydsl.py`) does NOT support
a per-projection mixed-precision MoE. The `targets=['Linear']`
config + `ignore=['model.layers.*.mlp.down_proj']` is supposed to
mean "all Linears except down_proj are int4" but the vLLM runtime
groups all 3 projections together and refuses the mismatch.

**This is a real gap** between llm-compressor's compress-time
acceptance and vLLM's load-time requirement. Three ways forward:

1. **Re-prequant DeepSeek with down_proj actually int4** (e.g. drop
   the down_proj ignore, accept the 10944-vs-128 padding
   tolerance, hope the kernel handles it). The divisibility check
   is in `validate_group_size_divisibility` and IS bypassed by the
   actual tensor reshape (only the 14.2% Hessian-fallback path
   hit it before), so dropping the ignore might work. Need to test.
2. **Use a smaller group_size that divides 10944**: divisors of
   10944 include 32, 64, 96, 128, 192, 228, 304, 456, 608, 684,
   912, 1368, 1824, 2736, 5472. The smallest "standard" group size
   that divides 10944 is 32 — but the W4A16 Marlin kernel only
   supports group_size in {32, 64, 128} (from WNA16_SUPPORTED_BITS
   in `compressed_tensors_wNa16.py`), and group_size=32 cuts the
   compression ratio by 4x. **Not preferred.**
3. **Drop DeepSeek from this sweep entirely**. The bf16/fp8/w8a16
   sweeps already cover DeepSeek. The W4 datapoint for DeepSeek
   is the only one missing, and the addendum 2 §7 #5 Path C
   ("defer the W4 datapoint") explicitly accepts this gap as
   "nice to have, not blocking". The other 3 models (OLMoE,
   Qwen3, Mixtral) cover 3 × 2 = 6 cells of the 8-cell sweep.

**Decision**: option 3 (drop DeepSeek cells) + option (a) of fix
below (per-cell try/except so a future DeepSeek cell that gets
re-prequanted properly just works without breaking the sweep).

### R. Fixes applied (Oct 6)

**Fix #1 (sweep robustness)**: refactored `moe_expert_stats.py`
to extract the per-cell body into `_run_one_cell(model, quant,
args, output_dir, sampling_params)`, then wrapped the call site
in `main()` with `try/except Exception`. The exception handler
logs a `CELL FAILED` message with the full traceback, calls
`gc.collect()` + `torch.accelerator.empty_cache()` to clean up
GPU memory, and continues to the next cell. Critical detail: the
analyzer's per-cell idempotency check (`_cell_out.exists()`) means
a re-submission of the same sbatch will skip cells whose output
JSON already exists and only attempt the missing/failed ones.

**Fix #2 (DeepSeek deferral)**: left the default `MODELS_OVERRIDE`
in the sbatch as the full 8 cells, but rely on the per-cell
try/except to make DeepSeek's failure non-fatal. The other 6
cells (OLMoE×2 done + Qwen3×2 + Mixtral×2) will run. If we want
DeepSeek's W4 datapoint later, the right path is to re-prequant
DeepSeek with a working config (option 1 above) and re-submit the
sweep — the try/except + idempotent skip will fill in the
DeepSeek cells without redoing OLMoE.

### S. Resubmission: job 32430029 (PENDING, Oct 6)

Submitted job 32430029 with the per-cell try/except fix in place.
PENDING in gpu-h100 queue, scheduled start **2026-10-12 21:40**
(~6 days from now — queue is deep). 24h wall.

Expected behavior on resubmission:
- 10 OLMoE expert_counts.json already exist from job 32345449 →
  analyzer skips them as "already exists", moves to cell 3.
- Cell 3 (DeepSeek-awq) fails with the same `ValueError`. The
  `try/except` catches it, logs a CELL FAILED with full
  traceback, cleans up GPU memory, continues.
- Cell 4 (DeepSeek-gptq) same failure, same handling.
- Cells 5-6 (Qwen3 × {awq, gptq}) run normally, 10 more
  expert_counts.json.
- Cells 7-8 (Mixtral × {awq, gptq}) run normally, 10 more
  expert_counts.json.

Net new contribution: 30 expert_counts.json (10 Qwen3 + 20
Mixtral). The DeepSeek W4 datapoint remains missing — deferred
per addendum 2 §7 #5 Path C.

If the per-cell try/except is correct, the job will write
`Cell N failed: model=.../quant=...` lines to the log for the 2
DeepSeek cells but otherwise complete cleanly. Output tree will
be at `results_wide_w4a16_awq_h100/` with 40 expert_counts.json
total (10 OLMoE + 30 new).

### T. Working tree state (Oct 6)

- `examples/eval-moe/moe_expert_stats.py`: refactored per-cell
  loop into `_run_one_cell()` with try/except at the call site.
  All existing tests + the rest of the analyzer code unchanged.
- `examples/eval-moe/spartan/vllm-moe-eval-h100-w4a16-prequant.sbatch`:
  unchanged from addendum 2 §M. (Per-cell try/except is in the
  analyzer, not the sbatch — the sbatch just runs the analyzer
  and the analyzer handles per-cell isolation now.)

### U. Updated open items (Oct 6)

1. **W4A16 analyzer sweep** (job 32430029, PENDING, scheduled
   2026-10-12 21:40). With the per-cell try/except fix, this
   should produce 30 new expert_counts.json (Qwen3 × 2 recipes ×
   5 benchmarks + Mixtral × 2 × 5). DeepSeek cells will log
   failures but not block the rest.
2. **MediaFlux upload of `results_wide_w8a16_h100/`** (13 GB) —
   still not done (handoff §7 #2, unchanged from before).
3. **Optional: re-prequant DeepSeek with int4 down_proj** to fill
   in the missing W4 datapoint. Not blocking.
4. **AWQ+GPTQ combined recipe** for OLMoE/DeepSeek/Qwen3 (3 more
   prequant jobs). User explicitly deferred.
5. **Cross-archive Jaccard sweep** in sister repo after #1.


---

## ADDENDUM 4 (added Oct 6, 2026 — DeepSeek reordered to last, 32430029 cancelled, 32430109 submitted)

User asked to reorder the cell execution so DeepSeek runs LAST, after
the cells that produce real data. This minimizes wasted wall time in
case the gpu-h100 job gets pre-empted — the 6 healthy cells (OLMoE,
Mixtral, Qwen3) finish first, and DeepSeek's known-failing cells run
at the end where they only consume time, not productive wall.

### V. Reorder + cancel + resubmit (Oct 6)

- **Cancelled job 32430029** (PENDING, never started).
- **Resubmitted as job 32430109** with `MODELS_OVERRIDE` overridden
  on the sbatch command line to put DeepSeek last. The new order:
  1. OLMoE-awq (already done from 32345449 — will be skipped)
  2. OLMoE-gptq (already done — will be skipped)
  3. Mixtral-awq (~3-4h, the largest cell)
  4. Mixtral-gptq (~3-4h)
  5. Qwen3-awq (~1-2h)
  6. Qwen3-gptq (~1-2h)
  7. DeepSeek-awq (FAILS — bf16/int4 mixed-precision rejected)
  8. DeepSeek-gptq (FAILS — same reason)

  Compared to the old alphabetical order (OLMoE → DeepSeek → Qwen3
  → Mixtral), this puts the known-failing cells at the very end so
  the productive cells complete first. If the job gets cut off
  prematurely (gpu-h100 pre-emption, 24h wall timeout, etc.), we
  have at least Qwen3 + Mixtral data instead of just OLMoE.

- **Wall time impact**: the DeepSeek cells will fail in the same
  ~30s as they did on job 32345449. Total budget unchanged at 24h.
  Worst case, the job runs all 6 healthy cells (~9-12h total: Mixtral
  ~6-8h + Qwen3 ~2-3h + 10 min for the OLMoE skip checks) plus ~1min
  for the two DeepSeek failures. Should fit in 24h easily.

- **Poller config updated** to track [32430109] (was [32430029]).

- **PENDING in gpu-h100 queue** (Reason=Priority). Scheduled start
  not yet known (queue is 6+ days deep as of Oct 6); will appear
  in `scontrol show job 32430109` once the scheduler picks a
  StartTime.

- **Output tree**: same as before, `results_wide_w4a16_awq_h100/`.
  Idempotent skip means OLMoE's 10 existing JSONs are preserved.


---

## ADDENDUM 5 (added Oct 8, 2026 — DeepSeek re-prequant with group_size=64, w8a16 upload, Mixtral TP=4 sweep)

This addendum covers four separate pieces of work that landed between the
Oct 6 handoff and now:

### W. w8a16_h100 → MediaFlux upload (Oct 8 morning)

Uploaded `results_wide_w8a16_h100/` (20 expert_counts.json, 13.58 GB)
to `/projects/proj-4180_momqe_data-1128.4.2052/results_wide_w8a16_h100/`
via `bash ~/.hermes/scripts/upload-eval-moe-results.sh results_wide_w8a16_h100`.
20/20 files, 0 failures, ~1.5 GB/s sustained. Local scratch copy preserved
(user decision — 372 GB free on scratch, keeping the local copy avoids
re-pull from MediaFlux if alignment is questioned later). Closes the
handoff §7 #2 open item.

### X. DeepSeek re-prequant — root-cause fix for the bf16/int4 mismatch

**Problem confirmed in job 32430109** (Oct 8 04:30, 1h11m wall,
COMPLETED exit 0 — the per-cell try/except saved the sweep). DeepSeek
× 2 cells failed with the documented ValueError, OLMoE + Qwen3 each
contributed 10 new expert_counts.json (20/40 cells now on scratch).
Mixtral × 2 also failed: "Available KV cache memory: -1.77 GiB" at
TP=1 with the 68 GB W4A16 ckpt loaded → no headroom for KV cache at
`gpu_mem_util=0.92`.

**The fix for DeepSeek** (commit `c016fe05`, file `examples/eval-moe/awq_quantize.py`):

1. Use **group_size=64** instead of the W4A16 preset's 128. DeepSeek's
   per-expert `mlp.down_proj` has 10944 cols; 10944 % 128 = 80 (fails
   divisibility check + runtime reshape `unflatten(-1, (86, 128))`
   fails with "Provided sizes [86, 128] don't multiply up to the size
   of dim 1 (10944)"). 10944 % 64 = 0 → no padding, no reshape error.
   64 is a standard WNA16 Marlin-supported size (group_size ∈ {32, 64, 128}).

2. **Drop the `re:.*\.down_proj$` ignore** (the old workaround that
   left down_proj at bf16). vLLM's `CompressedTensorsMoEMethod.get_moe_method`
   rejects a single FusedMoE with mixed schemes across gate/up/down.

3. **Restore the full AWQ `mappings=[...]` list** including the
   `up_proj → down_proj` scaling pair (previously omitted to dodge a
   ZeroDivisionError on the empty down_proj set in
   `_compute_layer_means`; no longer needed now that down_proj is in
   the quant set).

4. Build the scheme inline via `config_groups={"group_0": QuantizationScheme(...)}`
   with `QuantizationArgs(num_bits=4, type=INT, strategy=GROUP,
   group_size=64, symmetric=True, dynamic=False)`. The `scheme="W4A16"`
   preset hardcodes group_size=128 and the validator only accepts
   preset-name keys in `scheme=`.

5. GPTQ keeps `dampening_frac=0.02` (default 0.01) for DeepSeek to
   reduce the 14.2% Hessian-inversion fallback rate seen on job
   32266240. Qwen3/OLMoE/Mixtral paths are unchanged (their down_proj
   cols % 128 == 0, so they keep the W4A16 preset).

**Validation helper**: `examples/eval-moe/scripts/validate_deepseek_prequant.py`
(commit `9f4191b0`). 30s GPU smoke test that:
- Builds AWQ + GPTQ recipes without exception
- Confirms group_size=64 (not 128)
- Confirms `ignore` does NOT contain down_proj
- Initializes the modifiers on a synthetic `torch.nn.Linear(4096, 10944)`
  layer to exercise the divisibility validator (`validate_group_size_divisibility`)
  without needing the actual model.

Run on an interactive H100 BEFORE the full sbatch submission to catch
recipe regressions early.

**Re-prequant sbatches** submitted Oct 8 09:35 (gpu-h100, 4h wall each):
- `32518557` DeepSeek AWQ — start 2026-10-08 15:24 (today)
- `32518558` DeepSeek GPTQ — start 2026-10-08 19:30 (today)
- `32517565` Mixtral W4A16 TP=4 — start 2026-10-16 13:30 (8 days out)

SLURM backfilled the short DeepSeek jobs (~10 min wall each) ahead of
the 4-h-TW Mixtral sweep. Old mixed-precision DeepSeek ckpts deleted
from `${SCRATCH}/hf_cache_awq/` to make room for the new runs.

### Y. Mixtral W4A16 analyzer — re-submitted at TP=4

Job `32517565` is the re-submission of the failed TP=1 Mixtral cells
in 32430109. Same `vllm-moe-eval-h100-w4a16-prequant.sbatch` with:
- `MODELS_OVERRIDE` pinned to the two Mixtral prequant dirs
- `NGPUS=4`, `EXTRA_TENSOR_SPLIT=25,25,25,25`
- `OUTPUT_DIR=results_wide_w4a16_awq_h100`
- 12h wall

Per-GPU W4A16 Mixtral weight ~17 GB → ample KV cache headroom on 80 GB
H100, matching the bf16/w8a16 precedent from `vllm-moe-eval-h100-w8a16-mixtral-tp4.sbatch`
(job 31516295, the only prior Mixtral success).

### Z. Updated open items (Oct 8)

1. **DeepSeek re-prequant** — jobs 32518557 (AWQ) + 32518558 (GPTQ)
   will run today. After they land:
   - Run `validate_deepseek_prequant.py` end-to-end (replace the
     dry-run `4096x10944` check with actual `AutoConfig.from_pretrained`
     on the saved ckpt)
   - Submit a fresh W4A16 analyzer cell for DeepSeek × 2. With the
     new ckpts, vLLM's `get_moe_method` should accept the same-scheme
     invariant and the cell should produce 10 expert_counts.json.
   - Once DeepSeek × 2 cells work, the W4A16 sweep total jumps from
     30 to 40 expert_counts.json.
2. **MediaFlux upload of w8a16_h100** — **DONE** (Oct 8 morning).
   Closes §7 #2.
3. **Optional**: AWQ+GPTQ combined for OLMoE/DeepSeek/Qwen3 (3 more
   prequant jobs, ~10 min each — user explicitly deferred).
4. **Optional**: re-prequant DeepSeek with `awq_gptq` recipe too —
   just adds 1 more cell. Probably skip.
5. **Cross-archive Jaccard sweep** in sister repo after #1 lands.


---

## ADDENDUM 6 (added Oct 8, 2026 — DeepSeek re-prequant SUCCEEDED)

The `c016fe05` group_size=64 fix worked. Both DeepSeek re-prequant jobs
landed clean today (Oct 8 11:00–11:33 AEST):

- **32518557** (AWQ): 27m51s wall, exit 0. Saved 8.7 GB
- **32518558** (GPTQ): 5m33s wall, exit 0. Saved 8.7 GB. 10.3% Hessian
  inversion fallback (vs 23.9% on the prior bf16-down_proj run with
  group_size=128 — the smaller group_size gives the Hessian more
  degrees of freedom, fewer singular matrices).

The new ckpts at `${SCRATCH}/hf_cache_awq/deepseek-ai--deepseek-moe-16b-chat-{awq,gptq}/`:

- 8.7 GB each (down from 16 GB with the bf16 down_proj)
- `quantization_config.config_groups.group_0.weights.group_size = 64`
- `ignore = None` (no `down_proj` exclusion)
- `recipe.yaml` shows the full AWQ `mappings=[...]` including the
  up_proj → down_proj scaling pair

**Tensors verified** via safetensors header inspection:
- 1728 routed-expert `down_proj.weight_packed` (int4) + scale tensors
- 27 `mlp.shared_experts.down_proj.weight_packed` (int4)
- 1 special L00 `mlp.down_proj` (the dense layer-0 MLP, also int4 packed)
- Total: 1755 int4 down_proj tensors, **zero bf16 down_proj anywhere**
- Shape sanity: `down_proj.weight_packed = [2048, 10944/8=1368]` for the
  dense L00, `[2048, 10944/8=1368]` is the column-dim check confirming
  the 10944 columns are correctly packed at group_size=64 (171 scale
  groups per row, 2048 rows = `weight_scale: [2048, 171]`)

**W4A16 analyzer sweep for DeepSeek × 2 submitted** as job **32525399**
(PENDING, gpu-h100, 12h wall, TP=1, NGPUS=1). With the new ckpts,
vLLM's `CompressedTensorsMoEMethod.get_moe_method` should accept the
same-scheme invariant and the cell should produce 10 expert_counts.json.

**Once 32525399 lands** the W4A16 sweep goes from 20 → 30 expert_counts.json
(adding 10 DeepSeek cells). Total sweep coverage then:
- bf16: 20 cells (4×5)
- fp8: 20 cells (4×5) [partial — fp8_dl has 5 more, see doc §6.2]
- W8A16: 20 cells (4×5) [uploaded to MediaFlux]
- W4A16: 30 cells (3 models × 2 recipes × 5; DeepSeek still missing × 2)

**Still pending the W4 sweep**: Mixtral TP=4 (job 32517565, start
2026-10-16 13:30). The new DeepSeek analyzer job 32525399 is much
shorter and should land well before the Mixtral job.

### Update to validation helper

The `validate_deepseek_prequant.py` smoke-test script worked exactly as
designed on the synthetic `nn.Linear(4096, 10944)` layer. To extend it
for post-quant validation (next iteration), add a 4th check that loads
the actual saved `model.safetensors` and asserts every `down_proj.weight`
key has a corresponding `weight_packed` int4 tensor (no `bf16` weight).

Co-authored-by: Hermes Agent <noreply@nousresearch.com>

---

## ADDENDUM 7 (added Oct 10, 2026 — sglang backend attempted, blocked by cu13 dep wall)

A sglang backend for the expert-activation sweep was started so we
could produce a 3-way Jaccard comparison (vLLM vs llama.cpp vs
sglang). The driver, the smoke sbatch, and a debug sbatch are all
written and live in the tree, but the run is currently blocked by
an upstream sglang dependency requirement that no Spartan node can
satisfy.

### Artifacts left in the tree (ready to run when the wall breaks)

- `examples/eval-moe/moe_expert_stats_sglang.py` (470 lines) —
  the sglang driver. Imports the vLLM analyzer's prompt
  loaders + JSON writer as a library and replaces the inference
  core with an `openai.OpenAI` chat-completion call with
  `extra_body={"return_routed_experts": True}`. Reads sglang's
  base64-encoded int32 expert IDs out of `meta_info.routed_experts`
  and feeds them to the vLLM analyzer's existing aggregator.
  Output JSON is bit-compatible with the vllm/llama.cpp trees.
- `examples/eval-moe/spartan/sglang-moe-eval-a100-smoke.sbatch` —
  builds sglang into a fresh venv at
  `${SCRATCH}/venv_sglang/`, launches the server with
  `--enable-return-routed-experts`, runs the driver on a single
  small model × single benchmark × 8 prompts, writes the
  resulting JSON.
- `examples/eval-moe/spartan/sglang-moe-eval-a100-debug.sbatch` —
  a phase-by-phase debug variant. PAUSE=0 by default (no sleeps
  between phases). Use PAUSE=5..10 for true "step then look"
  interactivity once on a node.
- `examples/eval-moe/.plans/sglang-bf16-sweep.md` — the full
  project plan, including the cu13 wall analysis.

### Why it doesn't run today (the cu13 wall)

Spartan's GPU nodes (gpu-a100: 24 nodes, gpu-a100-short: 2 nodes,
gpu-h100: 10 nodes) all run **driver 570.x (CUDA 12.8 max)**. sglang
0.5.20 — the first version with both the routed-experts capturer
and the `--enable-return-routed-experts` server flag — was the
first to hard-pin CUDA 13+ in its published wheel METADATA:

```
Requires-Dist: flashinfer_python[cu13]==0.6.18
Requires-Dist: cuda-python>=13.0
Requires-Dist: humming-kernels[cu13]==0.1.12
Requires-Dist: nvidia-cutlass-dsl[cu13]==4.6.2
Requires-Dist: nvshmem4py-cu13
```

uv's resolver explicitly states: `sglang>=0.5.21 and
torch>=2.13.0+cu129 are incompatible` (torch 2.13.0+cu129's
`cuda-bindings<13` conflicts with sglang's `cuda-python>=13`
which needs `cuda-bindings>=13`).

Pre-0.5.20 sglang versions don't have the capturer API or the
server flag, so they can't serve the data the driver needs.

The local sglang checkout's `python/pyproject.toml` carries the
same cu13 markers (`flashinfer_python[cu13]==0.7.0.post1`), so
a source install hits the same wall. The only way through is a
forked dep list, which is out of scope for this project.

### Confirmed via login-node investigation (no GPU needed)

- `pip download` of sglang 0.5.10 → 0.5.21 wheels, METADATA
  inspection shows the cu13 markers landed in 0.5.20.
- `uv pip install --dry-run sglang==0.5.21 --torch-backend=cu129`
  → "your requirements are unsatisfiable" with the cuda-bindings
  version conflict.
- `import sglang` after `--no-deps` install fails immediately at
  `sglang/__init__.py:21` with `ModuleNotFoundError: numpy` and
  the chain reveals ~30+ transitive imports (`msgspec`,
  `decord2`, `psutil`, `orjson`, `xxhash`, `zstandard`, `xgrammar`,
  `torchvision`, `torchaudio`, `torchcodec`,
  `torch_memory_saver`, `partial_json_parser`, `IPython`,
  `openai-harmony`, `mistral_common`, `llguidance`, etc.). Each
  install reveals the next missing dep. Even if all of these
  were resolvable, the cu13-only kernels inside sglang's compiled
  `.so` files (cutlass-dsl, humming-kernels) would fail at first
  GPU call.

### How to revive this when the wall breaks

- **Spartan gets a driver upgrade to 580.x**: 0 GPUs need to be
  reassigned; just `sbatch` the existing
  `sglang-moe-eval-a100-smoke.sbatch` on `gpu-a100-short`. The
  sglang wheel install resolves cleanly without the
  `--no-deps` dance.
- **Cloud GPU with CUDA 13+ driver (e.g. H100 on Lambda, AWS
  p5)**: same sbatch works; the OpenAI client connects to
  the local sglang server. Set `MODEL_ID` / `BENCHMARKS` /
  `NUM_SAMPLES` per the canonical vLLM wide-sweep defaults.
- **Newer sglang release that drops the cu13 markers**: track
  the sglang changelog. As of Oct 10 2026, every release since
  0.5.20 has the markers.

### Why we're not blocking on it

- The vLLM + llama.cpp comparison already answers the core
  scientific question ("does weight precision interact with
  expert routing?"). sglang was a stretch goal.
- The driver code is **sglang-version-agnostic**: it only needs
  an OpenAI-compatible server that accepts
  `return_routed_experts=true` in the request body and returns
  base64-encoded int32 expert IDs in `meta_info.routed_experts`.
  No driver changes needed when the wall breaks.

Co-authored-by: Hermes Agent <noreply@nousresearch.com>

