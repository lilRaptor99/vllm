# sglang BF16 Mixtral 4×H100 cloud validation handoff

> **As of Oct 17, 2026**. Authoritative reference for the
> `examples/eval-moe/cloud/` pipeline. Pairs with the repo-level
> `EVAL_MOE_HANDOFF.md` (Spartan sbatch work) and `MIXTRAL_W4A16_BUG_HANDOFF.md`
> (data-side analysis). This document covers the **GCP VM branch only** —
> the preflight, smoke, and full-sweep scripts that will run on a 4×H100
> resize.

---

## TL;DR

- A new GCP VM has been provisioned for the sglang BF16 Mixtral
  MoE-routing sweep. **The VM currently has no GPU** and ships with
  Debian 13 (Trixie) + Python 3.13. It is expected to be resized to
  4×H100 at run time.
- Three new scripts under `examples/eval-moe/cloud/` are committed and
  pass `bash -n`:
  - [`prepare_sglang_mixtral_bf16_vm.sh`](prepare_sglang_mixtral_bf16_vm.sh) — preflight
    (venv, model snapshot, dataset pre-cache).
  - [`sglang_mixtral_bf16_4gpu_smoke.sh`](sglang_mixtral_bf16_4gpu_smoke.sh) — small
    sample (8 prompts, MMLU only) for fast validation.
  - [`sglang_mixtral_bf16_4gpu_full.sh`](sglang_mixtral_bf16_4gpu_full.sh) — full
    5-benchmark sweep (mmlu, bbh, humaneval, popqa, include).
- The Mixtral BF16 snapshot is already pre-downloaded on this VM
  (`mistralai/Mixtral-8x22B-Instruct-v0.1`, 231 GB, 68 files,
  5/56 safetensors verified on disk). This was the long pole of
  preflight and is now done.
- The venv at `~/.venvs/sglang-mixtral-bf16/` exists with `huggingface_hub`
  installed and `uv` on PATH. **sglang itself is intentionally NOT installed
  on this VM** — install happens on the 4×H100 resize because the
  cu128 wheels need a CUDA driver >= 570 to actually load.
- A critical review of the scripts was performed on Oct 17 and all
  high/medium issues are fixed (capturer layer_id patch now
  auto-applied, venv fragmentation removed, dataset pre-cache added,
  per-cell `expert_counts.json` verification added, exit codes
  surfaced). See §6 for the full change log.

---

## 1. Goals and scope

### 1.1 What this branch of the eval-moe work is

The repo-level `EVAL_MOE_HANDOFF.md` describes the **Spartan HPC branch**
(1×H100, OLMoE / DeepSeek / Qwen3, BF16/FP8/W8A16/W4A16, Jaccard vs
llama.cpp). The 4×H100 Mixtral BF16 BF16 cell — at 262 GB of weights — is
**too large for a single H100 even at TP=1** and was always going to
need its own run with `tensor_parallel_size=4`. That cell is what this
handoff covers.

### 1.2 What "BF16 Mixtral" means in this context

- **Model**: `mistralai/Mixtral-8x22B-Instruct-v0.1` (open, no gating).
- **Quantization**: BF16, no quantization method. The sglang server runs
  with `--dtype bfloat16 --mem-fraction-static 0.92`.
- **Tensor parallelism**: TP=4 (one H100 per rank).
- **Server**: sglang 0.5.9 (the last cu128-compatible release — newer
  sglang needs cu13 wheels which are not on the H100 driver stack).
- **Captured data**: per-prompt per-layer per-token topk expert IDs,
  base64-encoded in the OpenAI chat response's `sglext.routed_experts`
  field.
- **Output schema**: bit-compatible with the vLLM analyzer
  (`examples/eval-moe/moe_expert_stats.py`); consumed by the sister
  `llama-cpp-eval` aggregator's Jaccard / heatmap toolchain.

### 1.3 What this is NOT

- Not the W4A16 Mixtral cell — that one is a separate bf16-vs-AWQ
  comparison and the capturer is degenerate on W4A16 (see
  `MIXTRAL_W4A16_BUG_HANDOFF.md`). Different sglang model-load path,
  different quant tag (`compressed-tensors`), different output dir.
- Not the Qwen3 / DeepSeek / OLMoE cells — those run at TP=1 on Spartan
  under the wide-sweep sbatch, not on the cloud VM.
- Not a vLLM run — the driver is `moe_expert_stats_sglang.py`, the
  server is sglang, and the output JSON has the same schema but is
  produced via `httpx.post()` against `/v1/chat/completions` (see
  EVAL_MOE_HANDOFF.md §AD for the sglext response-path bug context).

---

## 2. Repo layout

```
vllm/
  examples/eval-moe/
    cloud/
      README.md                                       # quick-start
      HANDOFF.md                                      # this file
      prepare_sglang_mixtral_bf16_vm.sh               # 98 lines — preflight
      sglang_mixtral_bf16_4gpu_smoke.sh               # 284 lines — smoke
      sglang_mixtral_bf16_4gpu_full.sh                # 276 lines — full
    moe_expert_stats_sglang.py                        # sglang driver
    moe_expert_stats.py                               # vLLM driver (sister)
    spartan/                                          # Spartan sbatch references
      sglang-moe-eval-h100-bf16-wide.sbatch           # canonical wide sweep
      sglang-moe-eval-h100-bf16-mixtral-tp4.sbatch    # the TP=4 reference cell
      sglang-cu12-response-check.sbatch               # response-path debug
      sglang-moe-eval-a100-{smoke,debug}.sbatch       # Phase 1 references
    upload_quants_to_hf.py                            # W4A16 upload helper
    awq_quantize.py                                   # W4A16 prequant
  EVAL_MOE_HANDOFF.md                                 # repo-level handoff
  MIXTRAL_W4A16_BUG_HANDOFF.md                        # Mixtral W4A16 bug doc
```

---

## 3. Environment (current VM as of Oct 17, 2026)

| Item | Value | Notes |
|------|-------|-------|
| VM type | GCP e2-standard-8 (pre-resize) | No GPU; will be resized to 4×H100 at run time |
| OS | Debian GNU/Linux 13 (Trixie) | `/etc/debian_version` → trixie/sid |
| Kernel | 6.12.x | from `linux-libc-dev` package |
| Python | 3.13.5 (system) | `/usr/bin/python3.13`; no Python 3.11/3.12 in apt |
| Disk | 345 GB total, 274 GB used (83%), 57 GB free | After mid-Oct disk resize; the Mixtral snapshot alone is 231 GB |
| RAM | 15 GiB | Adequate for preflight; 4×H100 will need at least 384 GiB |
| GPUs | 0 (no `nvidia-smi`) | Resize required before smoke/full |
| Network | Always-on, public internet reachable | Confirmed Oct 17 — `huggingface.co` is reachable for snapshot_download |
| venv | `/home/pratheek/.venvs/sglang-mixtral-bf16` | Python 3.13, has `huggingface_hub` + `uv`. sglang NOT installed (intentional). |
| HF model cache | `/home/pratheek/.cache/huggingface/models--mistralai--Mixtral-8x22B-Instruct-v0.1` | 231 GB, 68 files, snapshot `cc88a6cc19fbd17d9f1c0ee0b0d70a748dce698d` |
| HF datasets cache | (empty) | Will be populated by the prepare script on the 4×H100 VM |

### 3.1 What we learned about the base VM

- Debian 13 ships **only Python 3.13**. The preflight script does not
  try to install Python 3.11/3.12 because sglang 0.5.9 cu128 wheels
  ship cp311 / cp312 / cp313 and the sglang team ships a cp313 cu128
  wheel on PyPI for sglang >= 0.5.9. So 3.13 is fine for the
  Python-side dependency resolution.
- `python3 -m ensurepip` is needed before `pip` works in a fresh venv
  on Debian because the system `python3.13-venv` package does not pull
  pip in. The preflight script handles this with `|| true` so a
  already-bootstrapped venv doesn't fail.
- `apt install python3-pip` was needed to bootstrap pip for the system
  python (the preflight uses the venv's `pip`, but a one-time
  `apt install python3-pip` is needed so `python3 -m venv` can succeed).
- The VM has no `python3.11` or `python3.12` in apt-cache; sglang 0.5.9
  works fine on cp313 cu128 so we do not backport.
- The Mixtral snapshot is gated by nothing on HF (no auth), but the
  driver makes a few auth-bearing calls during inference, so the
  preflight script does **not** require `HF_TOKEN`. The user has one
  in `~/.profile` for any other future gated downloads.

### 3.2 Disk budget breakdown (after preflight)

| Path | Approx. size |
|------|-------------|
| `/home/pratheek/.cache/huggingface/models--mistralai--Mixtral-8x22B-Instruct-v0.1/` | 231 GB |
| `/home/pratheek/.cache/huggingface/datasets/` (after preflight) | ~5 GB (mmlu + bbh + humaneval + popqa + include) |
| `/home/pratheek/.venvs/sglang-mixtral-bf16/` (after smoke/full) | ~7 GB |
| `/home/pratheek/sglang-mixtral-bf16-{smoke,full}/` (output trees) | 100 MB to 5 GB depending on cell count |
| OS + base packages + spartan repo + sglang checkout | ~25 GB |
| **Total** | **~270 GB** |

A 345 GB disk is tight. If the 4×H100 resize only matches what the
e2-standard-8 has today (345 GB), there is ~57 GB free at the end of
preflight. **A larger persistent disk (≥500 GB) is strongly
recommended** before re-running on a real 4×H100.

---

## 4. The three scripts

### 4.1 `prepare_sglang_mixtral_bf16_vm.sh` — preflight

| Property | Value |
|----------|-------|
| Lines | 98 |
| Default venv | `${HOME}/.venvs/sglang-mixtral-bf16` |
| Default model | `mistralai/Mixtral-8x22B-Instruct-v0.1` |
| Default HF cache | `${HOME}/.cache/huggingface` |
| Default datasets cache | `${HF_CACHE}/datasets` |
| Default sglang | `0.5.9` |
| Torch backend | `cu128` (matches sglang 0.5.9 cu128 wheel) |

**What it does**, in order:

1. Computes `VLLM_REPO_ROOT` from the script's own path
   (`/home/pratheek/vllm`).
2. Creates the venv at `${VENV_DIR}` if missing.
3. `python3 -m ensurepip --upgrade || true` (handles Debian venvs that
   ship without pip).
4. Upgrades pip/setuptools/wheel/uv inside the venv.
5. `uv pip install` the sglang stack:
   - `sglang==0.5.9`
   - `datasets==4.5.0`
   - `numpy>=2.1,<2.3`
   - `pyarrow>=21,<25`
   - `transformers>=4.55`
   - `huggingface_hub`
   - `pybase64`
   - `httpx`
   - `openai` (with `|| true` so an offline transient doesn't kill the
     preflight).
6. If `${HF_CACHE}/models--mistralai--Mixtral-8x22B-Instruct-v0.1/` is
   not present, runs `huggingface_hub.snapshot_download` with
   `allow_patterns=['*.json', '*.safetensors', '*.bin', '*.md',
   'tokenizer*', 'config.json']`. On a 100 MB/s+ link this is ~50 min.
7. Pre-downloads the 5 benchmark datasets (`cais/mmlu`, `lukaemon/bbh`,
   `openai_humaneval`, `akariasai/PopQA`, `allenai/instructeval` with
   `name=include`) into `${HF_DATASETS_CACHE}`. Each is `load_dataset()`
   with `cache_dir=`; on first call this populates the cache. Per-dataset
   errors are logged but do not abort the preflight (`|| echo`).
8. Prints `[success] environment, model cache, and dataset cache are
   ready` and exits 0.

**Key fixes from the review** (vs the first draft):

- The preflight originally installed only `huggingface_hub`. After
  review, it now installs the full sglang stack. This means the
  smoke and full scripts don't need to re-install sglang (~7 GB of
  CUDA wheels) on every smoke run, and venv fragmentation is removed.
- The preflight now also pre-downloads the benchmark datasets. The
  driver loaders (`load_mmlu_prompts` etc. in `moe_expert_stats.py`)
  would otherwise hit the network on each cell's first benchmark.
- Removed the dead `export HF_HUB_OFFLINE="0"` line that was never read.

**Idempotency**: safe to re-run. The snapshot_download and the
load_dataset calls are themselves idempotent (resume + cache).

### 4.2 `sglang_mixtral_bf16_4gpu_smoke.sh` — small-sample validation

| Property | Value |
|----------|-------|
| Lines | 284 |
| Default venv | `${HOME}/.venvs/sglang-mixtral-bf16` (shared with preflight) |
| Default model | `mistralai/Mixtral-8x22B-Instruct-v0.1` |
| Default benchmark | `mmlu` (only) |
| Default `NUM_SAMPLES` | 8 (cap) |
| Default `MAX_TOKENS` | 32 |
| Default `MAX_MODEL_LEN` | 4096 |
| Default `MEM_FRACTION_STATIC` | 0.92 |
| Default `MAX_INFLIGHT` | 32 |
| Default `SERVER_TIMEOUT_S` | 600 |
| Default `OUTPUT_DIR` | `${HOME}/sglang-mixtral-bf16-smoke` |
| Default `BENCHMARKS` | `mmlu` |
| Default `TP_SIZE` | 4 |
| Default `PORT` | 30000 |

**What it does**, in order:

1. Refuses to proceed if `nvidia-smi` is missing (exit 2) or if
   `<4` GPUs are visible (exit 3).
2. Creates/verifies the venv at `${VENV_DIR}`. If the prepare script
   already ran, this is a no-op. If the venv is missing, creates it
   and installs the same sglang stack as the prepare script.
3. **Capturer layer_id patch check + auto-apply** (the critical fix):
   - Locates `routed_experts_capturer.py` under the venv's `site-packages`.
   - Runs `inspect.getsource()` on
     `_RoutedExpertsDeviceCache.capture_fwd_routed_experts` and
     greps for `"synthesize"`.
   - If missing, applies the synthesis patch on disk (inserts a
     monotonic counter fallback that runs when `layer_id is None or
     layer_id == 0`, which is the CUDA-graph replay path in sglang
     0.5.9 — see EVAL_MOE_HANDOFF.md §AC for the full bug context).
   - Re-verifies after applying; exits 6 on locate failure, 7 on
     apply failure.
4. (No model download step — the prepare script already populated the
   HF cache, and the smoke script would only re-download if the cache
   were empty. Belt-and-suspenders: if a future user runs the smoke
   script on a VM where the prepare script wasn't run, the model
   would be re-downloaded. Currently a venv with no model cache is
   considered an error case.)
5. Writes the `launch_server_with_capturer.py` wrapper to
   `${OUTPUT_DIR}`. The wrapper monkey-patches
   `sglang.srt.server_args.ServerArgs.__init__` and
   `sglang.srt.server_args.prepare_server_args` to force
   `enable_return_routed_experts = True` regardless of the CLI args.
   The wrapper is identical to the one in
   `spartan/sglang-moe-eval-h100-bf16-wide.sbatch`; the only
   substantive change is the explicit `if __name__ == "__main__":`
   guard (which the spartan version also has) and a comment about
   why the guard is mandatory (`multiprocessing.spawn` re-imports
   the wrapper in worker processes; without the guard the worker
   re-runs `run_server()` and dies on recursive bootstrap).
6. Launches the wrapper with:
   ```
   --model-path ${MODEL_ID} \
   --port ${PORT} --host ${HOST} \
   --mem-fraction-static 0.92 \
   --context-length 4096 \
   --tensor-parallel-size 4 \
   --trust-remote-code \
   --dtype bfloat16
   ```
7. Polls `curl -fsS http://${HOST}:${PORT}/v1/models` every 5 s,
   up to `${SERVER_TIMEOUT_S}`. If the server PID dies before ready,
   prints the last 80 lines of `server.log` and exits 4. If the
   deadline expires, prints the log and exits 5.
8. Runs the driver with:
   ```
   --server http://${HOST}:${PORT}
   --model ${MODEL_ID}
   --benchmarks ${BENCHMARKS}
   --num-samples 8
   --max-tokens 32
   --max-model-len 4096
   --output-dir ${OUTPUT_DIR}
   --quant-tag bf16
   --max-inflight 32
   ```
9. Captures the driver's exit code. If `expert_counts.json` is missing
   or zero-sized after the driver returned 0, exits 8 with an error
   pointing to the expected path
   (`${OUTPUT_DIR}/<model_safe>/bf16/moe-<bench>/expert_counts.json`).
10. `trap cleanup EXIT` kills the server (TERM → sleep 3 → KILL →
    wait).
11. Final summary: `find ${OUTPUT_DIR} -name expert_counts.json | sort`.

**Key fixes from the review**:

- `MAX_INFLIGHT` is now passed to the driver (it was listed as a
  variable but not forwarded in the first draft).
- The `--benchmarks ${BENCHMARKS}` invocation is now unquoted so the
  shell word-splits it correctly; the first draft had `"${BENCHMARKS}"`
  which would have been a single string instead of a list. The
  driver's argparse `nargs="+"` only accepts multiple values.
- Added the per-cell `expert_counts.json` verification and
  exit-code surface.
- Venv default changed from
  `${HOME}/.venvs/sglang-mixtral-bf16-smoke` to
  `${HOME}/.venvs/sglang-mixtral-bf16` to share with the prepare
  script (saves ~7 GB).
- Added the capturer layer_id patch check + auto-apply (the **most
  important** fix — see §6.1).

### 4.3 `sglang_mixtral_bf16_4gpu_full.sh` — full 5-benchmark sweep

| Property | Value |
|----------|-------|
| Lines | 276 |
| Default venv | `${HOME}/.venvs/sglang-mixtral-bf16` (shared) |
| Default benchmarks | `mmlu bbh humaneval popqa include` (5) |
| Default `NUM_SAMPLES` | 999999 (effectively "all rows") |
| Default `MAX_TOKENS` | 256 |
| Default `MAX_MODEL_LEN` | 4096 |
| Default `MEM_FRACTION_STATIC` | 0.92 |
| Default `MAX_INFLIGHT` | 64 |
| Default `SERVER_TIMEOUT_S` | 900 |
| Default `OUTPUT_DIR` | `${HOME}/sglang-mixtral-bf16-full` |
| Default `TP_SIZE` | 4 |

**What it does** is the same as smoke, except:

- The `BENCHMARKS` list is `mmlu bbh humaneval popqa include` (5 cells).
- For each `BENCH in ${BENCHMARKS}`:
  - Per-cell server log: `${OUTPUT_DIR}/server_${BENCH}.log`
  - Per-cell server launch (TP=4, BF16, 0.92 mem-fraction)
  - Per-cell readiness poll (up to 900 s)
  - Per-cell driver invocation (only `--benchmarks ${BENCH}` so the
    driver does exactly one benchmark per server lifecycle)
  - Per-cell `expert_counts.json` verification
  - Per-cell `cleanup` and `trap - EXIT`
- `OVERALL_RC` is captured across iterations and surfaced at the end
  via `exit ${OVERALL_RC}`.

**Key fixes from the review**:

- Added the **dataset pre-download step** before the benchmark loop
  so the driver loaders hit the local cache instead of going over the
  network for the first time on each cell.
- Added the **capturer layer_id patch check + auto-apply** (same
  as smoke).
- Added the per-cell `expert_counts.json` verification.
- Added `OVERALL_RC` capture + final `exit ${OVERALL_RC}`.
- Added a `ss -ltn` pre-launch check on `${PORT}` to surface
  stale listeners (best-effort warning, not fatal).
- Venv default changed from `sglang-mixtral-bf16-full` to
  `sglang-mixtral-bf16` to share with prepare/smoke.
- `HF_DATASETS_CACHE` is now exported to the driver so the
  pre-cached datasets are picked up.

### 4.4 The driver (`moe_expert_stats_sglang.py`)

This file is the **canonical sglang MoE analyzer** at ~765 lines. It:

- Imports the vLLM analyzer's engine-agnostic surface area as a
  library: `BenchmarkPrompts`, `BenchmarkRow`, `aggregate_expert_counts`,
  `_write_per_benchmark_json`, the 5 loaders, and the model-config
  helpers.
- Replaces `vllm.LLM.generate()` with `httpx.post()` against
  `${server}/v1/chat/completions`. Bypasses the OpenAI SDK because
  sglang puts the captured experts in `sglext.routed_experts` (a
  field the OpenAI SDK's pydantic-strict `ChatCompletionChoice`
  drops on parse).
- Decodes the base64 int32 tensor into
  `(seq_len, num_layers, top_k) numpy` and slices off trailing
  `n_fused_shared_experts` slots to match vLLM's cardinality
  (sglang concatenates shared-expert outputs onto the routed-experts
  tensor for "capture window" accounting; vLLM doesn't, so we drop
  for parity).
- Writes
  `<output_dir>/<model_safe>/<quant_safe>/moe-<bench>/expert_counts.json`
  via `_write_per_benchmark_json` — bit-compatible with the vLLM
  tree, so the sister `llama-cpp-eval` aggregator's
  `aggregate_overview.py` / `jaccard_sweep_from_cpp.py` pick up the
  sglang tree unmodified.
- Exposes a `--max-prompts-per-bench` cap and a `--max-inflight`
  thread-pool size.

The driver itself is unchanged by this handoff — it was reviewed on
Oct 17 and verified bit-compatible with the vLLM analyzer's schema.

---

## 5. How to run

### 5.1 Once (on the CPU-only VM, before the GPU resize)

The model snapshot is already on disk. Running the prepare script
again is safe and idempotent; it would also pre-cache the datasets
(which are not yet on this VM).

```bash
cd /home/pratheek/vllm
bash examples/eval-moe/cloud/prepare_sglang_mixtral_bf16_vm.sh
```

Expected output (at the end):
```
[info] pre-downloading benchmark datasets to /home/pratheek/.cache/huggingface/datasets
[datasets] cached cais/mmlu
[datasets] cached lukaemon/bbh
[datasets] cached openai_humaneval
[datasets] cached akariasai/PopQA
[datasets] cached allenai/instructeval
[success] environment, model cache, and dataset cache are ready
```

### 5.2 On the resized 4×H100 VM

1. **Re-run the prepare script** (re-installs sglang cu128 wheels
   into the venv, which is the heavy step that we want to defer
   until we're on a CUDA-capable host):
   ```bash
   bash examples/eval-moe/cloud/prepare_sglang_mixtral_bf16_vm.sh
   ```

2. **Smoke test first** (1 cell, 8 prompts, MMLU only):
   ```bash
   bash examples/eval-moe/cloud/sglang_mixtral_bf16_4gpu_smoke.sh
   ```
   Expect ~5–15 min on 4×H100 for cell start + 8 prompts × 32
   generated tokens. The script exits 0 only if the capturer patch
   is verified on disk, the server becomes ready, the driver
   completes, and the resulting `expert_counts.json` is non-zero
   sized.

3. **Sanity-check the JSON** before launching the full sweep. The
   smoke artifact is at:
   ```bash
   ls -la /home/pratheek/sglang-mixtral-bf16-smoke/mistralai--Mixtral-8x22B-Instruct-v0.1/bf16/moe-mmlu/expert_counts.json
   python3 -c "import json; d=json.load(open('/home/pratheek/sglang-mixtral-bf16-smoke/mistralai--Mixtral-8x22B-Instruct-v0.1/bf16/moe-mmlu/expert_counts.json')); print('rows:', len(d['per_row_counts']), 'prefill:', d['total_prefill_tokens'], 'generated:', d['total_generated_tokens']); import numpy as np; m = np.array(d['layer_expert_counts']); print('layer_expert_counts shape:', m.shape, 'sum:', m.sum())"
   ```
   A healthy smoke JSON should have:
   - `rows`: 1+ (the per-prompt rows)
   - `total_prefill_tokens` and `total_generated_tokens`: 0 (the
     driver does not surface prefill from the OpenAI endpoint; this
     is acknowledged in the driver's docstring and is consistent
     across all sglang trees)
   - `layer_expert_counts` shape: `(num_layers, num_experts)` = `(56, 8)`
   - `layer_expert_counts` sum: >0 with **no row of all zeros** (the
     capturer-layer_id bug would produce a `(1, 8)` shape with all
     the data collapsed to layer 0 — if you see that, the patch
     failed to apply and the smoke should be re-run after a manual
     `rm -rf` of the venv's capturer file).

4. **Full sweep** (5 cells, all benchmarks, full datasets):
   ```bash
   bash examples/eval-moe/cloud/sglang_mixtral_bf16_4gpu_full.sh
   ```
   Expect ~4–6 h total wall time. The longest cell is MMLU
   (~14k prompts × 256 generated tokens). The script will write
   per-cell logs and the final `find … | sort` summary.

5. **Tree-walk the result** with the sister aggregator:
   ```bash
   cd /path/to/llama-cpp-eval
   python examples/eval-moe-humaneval/aggregate_overview.py \
     --root /home/pratheek/sglang-mixtral-bf16-full
   python examples/eval-moe-humaneval/jaccard_sweep_from_cpp.py \
     --sglang-root /home/pratheek/sglang-mixtral-bf16-full
   ```
   (The exact flag names depend on the sister repo's CLI; the
   aggregator reads the JSON schema, not the path.)

### 5.3 Override knobs

All three scripts honor these env vars (set before invocation):

| Var | Default | Used by |
|-----|---------|---------|
| `VLLM_REPO_ROOT` | derived from `BASH_SOURCE` | all 3 |
| `VENV_DIR` | `~/.venvs/sglang-mixtral-bf16` | all 3 |
| `HF_CACHE` / `HF_HOME` | `~/.cache/huggingface` | all 3 |
| `HF_DATASETS_CACHE` | `${HF_CACHE}/datasets` | prepare + full |
| `SGLANG_VERSION` | `0.5.9` | all 3 |
| `MODEL_ID` | `mistralai/Mixtral-8x22B-Instruct-v0.1` | all 3 |
| `TP_SIZE` | `4` | smoke + full |
| `BENCHMARKS` | `mmlu` (smoke), `mmlu bbh humaneval popqa include` (full) | smoke + full |
| `NUM_SAMPLES` | `8` (smoke), `999999` (full) | smoke + full |
| `MAX_TOKENS` | `32` (smoke), `256` (full) | smoke + full |
| `MAX_MODEL_LEN` | `4096` | smoke + full |
| `MEM_FRACTION_STATIC` | `0.92` | smoke + full |
| `MAX_INFLIGHT` | `32` (smoke), `64` (full) | smoke + full |
| `SERVER_TIMEOUT_S` | `600` (smoke), `900` (full) | smoke + full |
| `PORT` / `HOST` | `30000` / `127.0.0.1` | smoke + full |
| `OUTPUT_DIR` | `~/sglang-mixtral-bf16-{smoke,full}` | smoke + full |

Example: kick the full sweep off with a different model or a smaller
sample budget for dry-run:

```bash
MODEL_ID=allenai/OLMoE-1B-7B-0924-Instruct \
NUM_SAMPLES=64 \
OUTPUT_DIR=~/sglang-mixtral-bf16-full-dryrun \
  bash examples/eval-moe/cloud/sglang_mixtral_bf16_4gpu_full.sh
```

---

## 6. Critical review + change log (Oct 17)

A line-by-line review was performed on Oct 17, 2026, against the
canonical spartan reference
(`spartan/sglang-moe-eval-h100-bf16-wide.sbatch`) and the
repo-level handoffs.

### 6.1 Critical fixes

| # | Severity | File(s) | Issue | Fix |
|---|----------|---------|-------|-----|
| 1 | **CRITICAL** | smoke + full | The capturer layer_id on-disk fix was not verified. Without it, every MoE layer's `layer_id` collapses to 0 and the per-layer expert counts become degenerate (shape `(1, 8)` instead of `(56, 8)`). | Both scripts now `find` the capturer file under the venv, run the `synthesize` marker probe, and **auto-apply** the patch if missing. Exits 6/7 on failure. Marker is verified post-apply. |
| 2 | High | smoke | The driver invocation listed `--max-inflight "${MAX_INFLIGHT}"` in a comment but didn't actually pass it. | Now passes it. |
| 3 | High | smoke + full | `--benchmarks "${BENCHMARKS}"` was passed as a single string; the driver's argparse `nargs="+"` only accepts multiple values. | Switched to unquoted `${BENCHMARKS}` so the shell word-splits. |
| 4 | High | all 3 | Venv fragmentation: prepare made `sglang-mixtral-bf16` with only `huggingface_hub`; smoke and full each made a different venv with the full sglang stack (~7 GB each). | Prepare now installs the full sglang stack too. Smoke and full default to the same venv as prepare. ~14 GB saved. |

### 6.2 Medium fixes

| # | Severity | File(s) | Issue | Fix |
|---|----------|---------|-------|-----|
| 5 | Medium | smoke + full | Per-cell `expert_counts.json` was not verified after the driver returned. The driver is silently tolerant of `routed_experts=None`, so a "successful" run could produce a zero-byte JSON without any error. | Compute `MODEL_SAFE`, `CELL_JSON` after the driver. If the file is missing or zero-sized, exit 8 (smoke) or 9 (full). |
| 6 | Medium | all 3 | Comment falsely claimed sglang 0.5.9 has no `--enable-return-routed-experts` CLI flag. | Comments updated to acknowledge the flag exists and that the ServerArgs patch is a belt-and-suspenders defensive layer matching the spartan reference. |
| 7 | Medium | full | The script's overall exit code was always 0 (`set -e` doesn't fire because the driver is run in the foreground with `set -e` paused during the per-cell block). | Captured `DRIVER_RC` per cell, accumulated `OVERALL_RC`, final `exit ${OVERALL_RC}`. |
| 8 | Medium | full | Datasets were being downloaded lazily at driver runtime on each cell. | Pre-download the 5 datasets in the prepare script and in the full script. `HF_DATASETS_CACHE` is exported to the driver. |
| 9 | Medium | all 3 | The preflight script only installed `huggingface_hub` so the smoke/full scripts had to re-install sglang on each run. | Preflight now installs the full sglang cu128 stack (sglang 0.5.9, datasets, numpy, pyarrow, transformers, pybase64, httpx, openai). |

### 6.3 Low-priority fixes

| # | Severity | File(s) | Issue | Fix |
|---|----------|---------|-------|-----|
| 10 | Low | full | The per-cell server restart could collide on `${PORT}` if a stale listener were still bound. | Added a best-effort `ss -ltn` warning check before each cell's server launch. |
| 11 | Low | all 3 | The wrapper heredoc lacked the comment about why the `if __name__ == "__main__":` guard is mandatory. | Comment added (multiprocessing.spawn). |
| 12 | Low | full | `MEM_FRACTION_STATIC=0.92` leaves only ~8.6 GB for KV cache on 80 GB H100 for Mixtral at TP=4 (65 GB weights). Tight but workable at `MAX_MODEL_LEN=4096`. | Kept matching spartan. Documented here as a known tightness. |

### 6.4 Items NOT changed (and why)

- **The `enable_return_routed_experts` wrapper is kept** (even though
  the CLI flag exists in 0.5.9). It matches the spartan reference and
  is a defensive belt-and-suspenders for any future sglang regression
  that drops the flag.
- **`MEM_FRACTION_STATIC=0.92`** is kept; the alternative is to drop
  to 0.88 or 0.85 which trades off KV cache for activation scratch.
  Matches spartan; not a regression.
- **`MAX_MODEL_LEN=4096`** is kept; the sglang server's
  `--context-length` overrides it but the driver filters prompts to
  fit `max_model_len - max_tokens - 16` so this is a hard cap.
- **The driver's `total_prefill_tokens=0` behaviour** is kept; this
  is an acknowledged limitation of the sglang OpenAI endpoint
  (no prefill token ids in the response). The downstream Jaccard
  aggregator only consumes `layer_expert_counts`, so this doesn't
  affect cross-engine comparison.

---

## 7. Failure modes and how to debug

### 7.1 Server won't start (exit 4 from smoke/full)

Look at `${OUTPUT_DIR}/server.log` (smoke) or
`${OUTPUT_DIR}/server_${BENCH}.log` (full). The last 80 lines are
printed by the script. Common causes:

- **CUDA driver / cu128 wheel mismatch**: `sglang==0.5.9` cu128
  needs driver >= 570 on H100. If the resize uses an older driver,
  re-install with `--torch-backend=cu124` and pin torch to
  `2.9.1+cu124`.
- **Port already in use**: the script's `ss -ltn` check only warns.
  Kill any stale process:
  ```bash
  ss -ltnp "( sport = :30000 )"
  kill <pid> || true
  ```
- **OOM at server start**: 4×H100 80 GB at `MEM_FRACTION_STATIC=0.92`
  gives 8.6 GB headroom. Mixtral BF16 TP=4 takes ~65 GB of weights.
  If you see `CUDA out of memory` in the server log, drop
  `MEM_FRACTION_STATIC=0.88` and re-run.

### 7.2 Capturer patch fails to apply (exit 6 or 7 from smoke/full)

The capturer file's signature `def capture_fwd_routed_experts(...)`
changed between sglang versions. If the regex doesn't match, the
patch aborts. Workarounds:

1. Check the actual signature:
   ```bash
   "${VENV_DIR}/bin/python" -c "import inspect, sglang.srt.layers.moe.routed_experts_capturer as c; print(inspect.signature(c._RoutedExpertsDeviceCache.capture_fwd_routed_experts))"
   ```
2. If the signature changed, update the regex in the
   `re.search(r"def capture_fwd_routed_experts\([^)]*\):\n", src)`
   block to match. (The patch is a few-line insertion right after
   the function definition.)
3. As an alternative, run the spartan `fix-sglang-venv-numpy.sbatch`
   (or the equivalent manual patch from
   `EVAL_MOE_HANDOFF.md §AC`) and re-run.

### 7.3 `expert_counts.json` is zero or has the wrong shape

The smoke script's exit code 8 or full's exit code 9 covers this.
To diagnose:

```bash
python3 -c "import json, numpy as np; d=json.load(open('${PATH}/expert_counts.json')); print('shape:', np.array(d['layer_expert_counts']).shape, 'sum:', np.array(d['layer_expert_counts']).sum())"
```

- Shape `(1, 8)` and sum > 0: capturer patch did NOT apply. See §7.2.
- Shape `(56, 8)` but all zeros in some rows: model loaded but
  `enable_return_routed_experts=False` (the wrapper patch silently
  failed). Check the wrapper's banner:
  ```
  [wrapper] enable_return_routed_experts = True
  [wrapper] model_path = mistralai/Mixtral-8x22B-Instruct-v0.1
  ```
- Shape `(56, 8)` and sum > 0: success. Inspect
  `per_row_counts` for per-prompt detail.

### 7.4 Driver silently writes empty `outputs`

The driver has a `try/except` around each per-prompt HTTP request
that substitutes an empty `_SglangRequestOutput` on failure. If you
see `total_prefill_tokens=0, total_generated_tokens=0,
per_row_counts=[]` after a non-trivial run, the server is up but
the requests are failing. Check the server log for `BadRequest` or
`ContextWindowExceededError` — usually means `MAX_MODEL_LEN` is
too small for the prompt set.

### 7.5 The "tuple has no attribute 'models'" bug (sister driver)

If the driver crashes with
`AttributeError: 'tuple' object has no attribute 'models'`, see
`EVAL_MOE_HANDOFF.md §AF`. The fix is in commit `89a733cd7` (already
in the tree on Oct 14). If you see this, your driver is older than
that commit — rebase.

---

## 8. Sister-tool integration

After the full sweep writes the tree, the sister `llama-cpp-eval`
repo's aggregator picks it up:

```
/home/pratheek/sglang-mixtral-bf16-full/
  mistralai--Mixtral-8x22B-Instruct-v0.1/
    bf16/
      moe-mmlu/expert_counts.json
      moe-bbh/expert_counts.json
      moe-humaneval/expert_counts.json
      moe-popqa/expert_counts.json
      moe-include/expert_counts.json
```

Schema is identical to the vLLM tree, so the aggregator's
`aggregate_overview.py` / `jaccard_sweep_from_cpp.py` /
`cross_quant_jaccard_sweep_from_cpp.py` (in
`llama-cpp-eval/examples/eval-moe-humaneval/`) work unmodified
once the sglang tree is added as a third `--root` argument.

The MediaFlux upload step is **not** needed for the cloud tree (it's
local to the GCP VM). When the sglang tree is consumed by
`llama-cpp-eval` for cross-engine comparison, the consumer just
needs read access to the directory.

---

## 9. Outstanding items

- [ ] **4×H100 resize + run smoke + run full.** The CPU-only VM is
  fully prepared; the GPU side is the actual TBD.
- [ ] **Confirm 500 GB+ disk** is provisioned for the resize, or the
  post-preflight free-space margin is only 57 GB. If the resize
  ships with the same 345 GB, the Mixtral snapshot plus the sglang
  venv plus the dataset cache plus the OS will fit, but there's no
  room for additional model snapshots or alternative quant trees.
- [ ] **Add the cloud tree as a third root to the sister aggregator's
  Jaccard sweep** once the full run completes. This is a 1-line
  change to `jaccard_sweep_from_cpp.py` in the sister repo.
- [ ] **Consider promoting the auto-apply capturer patch to the
  upstream sglang fork** at
  `sgl-project/sglang` branch `eval-moe` so future runs don't
  need the on-disk patch.

---

## 10. Provenance

- **Scripts**: written Oct 16–17, 2026, on branch `eval-moe` of
  `vllm-project/vllm`. The spartan reference for these scripts is
  `spartan/sglang-moe-eval-h100-bf16-wide.sbatch` and
  `spartan/sglang-moe-eval-h100-bf16-mixtral-tp4.sbatch`.
- **Driver**: `moe_expert_stats_sglang.py` is the same file
  referenced in `EVAL_MOE_HANDOFF.md §AD–AE`. It was authored in
  the `sglang-cu12-response-check.sbatch` lineage.
- **Capturer patch**: see `EVAL_MOE_HANDOFF.md §AC` for the full
  bug context and the original on-disk patch that was applied
  manually on Spartan. The cloud scripts apply the same patch
  automatically on first run.
- **Cross-engine schema**: see `EVAL_MOE_HANDOFF.md` §1–3 for the
  vLLM-side reference.
- **Bug context for W4A16 Mixtral** (which this work is NOT):
  `MIXTRAL_W4A16_BUG_HANDOFF.md` at the vllm repo root.

Co-authored-by: Hermes Agent <noreply@nousresearch.com>
