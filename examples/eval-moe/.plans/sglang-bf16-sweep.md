# Plan: sglang bf16 MoE expert-activation sweep (Oct 2026)

## Goal

Re-run the bf16 H100 expert-activation sweep (`results_wide_bf16_h100/`,
4 models × 5 benchmarks = 20 expert_counts.json) on **sglang** instead of
vLLM. The output must be **bit-compatible** with the existing vLLM and
llama.cpp JSON schema so the sister-repo aggregator
(`llama-cpp-eval/examples/eval-moe-humaneval/heatmap_from_cpp.py`,
`aggregate_overview.py`, `jaccard_sweep_from_cpp.py`) consumes it
unmodified. Outcome: a 3-way Jaccard sweep over (vLLM, llama.cpp, sglang)
to answer "does the inference backend measurably change expert routing?"

## Phase 0 — Recon (done in this turn)

### What I read

- `vllm/examples/eval-moe/moe_expert_stats.py` (1638 lines): the canonical
  analyzer. `aggregate_expert_counts()` consumes vLLM's
  `CompletionOutput.routed_experts` tensor of shape `(seq_len, L, top_k)`
  and bins it into per-row `[L,E]`, `[L,E,E]` intra, `[L-1,E,E]` adjacent
  pair counts. `_write_per_benchmark_json()` is the output writer — schema
  is documented at the top of that function and re-described in the
  llama-cpp C++ header comments.
- `vllm/examples/eval-moe/spartan/vllm-moe-eval-h100-bf16.sbatch` (1027
  lines): the bf16 baseline sbatch. Module stack, scratch layout, env
  contract are all standard.
- `llama-cpp-eval/examples/eval-moe-humaneval/heatmap_from_cpp.py` (391
  lines): the C++ schema doc is the source of truth for downstream
  consumers. The expected JSON top-level keys (per
  `BENCHMARK_RECORD_KEY`):
  - mmlu / bbh → `"subjects"`
  - humaneval → `"tasks"`
  - popqa → `"props"`
  - include → `"by_langdom"`
  Plus required fields: `model`, `model_arch.{name,n_layer,n_expert,
  n_expert_used}`, `config`, `totals`, `aggregate.{marginal_expert_counts,
  intra_pair_counts, adjacent_pair_counts}`.
- `sglang/python/sglang/srt/state_capturer/routed_experts.py` (177
  lines): the analog of vLLM's `enable_return_routed_experts`. Returns
  the per-token per-layer topk tensor via `meta_info.routed_experts` as
  **base64-encoded int32** (helper `extract_routed_experts_from_meta_info`
  decodes it). Shape is `(tokens, layers, top_k)` — same as vLLM.
  `num_fused_shared_experts` are dropped via the `[:topk_size]` truncation
  in `_get_local_slice`, matching vLLM's "shared experts are NOT in
  routed_experts" semantics.
- `sglang/python/sglang/srt/entrypoints/openai/protocol.py` (line 376):
  the OpenAI request schema already exposes `return_routed_experts:
  bool = False` and `routed_experts_start_len: int = 0`. So the client
  is just a vanilla `openai` SDK chat-completion call with one extra
  field.
- `sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:2014` and
  `serving_completions.py:269`: server side stuffs
  `meta_info.routed_experts` into the response. We decode on the client.
- sglang's four target models are all present:
  `python/sglang/srt/models/{olmoe,deepseek_v2,qwen3_moe,mixtral}.py`.

### Key design constraints discovered

1. **sglang needs a Rust toolchain.** `pyproject.toml` requires
   `setuptools-rust>=1.11` and pulls in a Cargo workspace from
   `../rust/`. sglang also pins `torch==2.14.1` in build-system — but
   vLLM's venv at `${SCRATCH}/venv/` already has torch 2.13.0+cu129 and
   the wheel is forward-compatible with CUDA 12.4.1 driver. **We can
   install sglang with `pip install -e .` and the existing torch will
   satisfy the runtime, but the Rust ext must still be built.** Cargo
   build is ~5-8 min on login.
2. **gpu-a100-short has only 2 nodes** (per `sinfo` output). A 4-GPU
   bf16 Mixtral run will need either gpu-a100 (25 nodes) or partition
   backfill. The 123 GB / 4-day hard cap the user mentioned only
   applies to a100-short; a100 is no-cap. **For bf16 80 GB Mixtral
   single-GPU OOMs (262 GB model)** — same as vLLM; TP=4 needed. The
   user explicitly said "a100-short" for the first smoke job, so we
   start with OLMoE + DeepSeek + Qwen3 (TP=1) on a100-short, then a
   separate Mixtral TP=4 job on a100.
3. **No `enable_return_routed_experts` for sglang-cli argument**: I
   confirmed the flag is in `arg_groups/fields/exec_.py:85` (line
   checked). Need to verify the CLI flag spelling (`--enable-return-
   routed-experts`?) and pass it on server launch.
4. **API transport**: sglang is OpenAI-compatible HTTP, so the client
   side is a regular `openai.OpenAI(base_url=...)` chat completion. No
   custom RPC needed.
5. **Datasets are pre-staged** at `${SCRATCH}/datasets/{mmlu,bbh,
   humaneval,popqa,include,perfectblend}` — the vllm analyzer reads
   from here via absolute paths. The sglang analyzer should do the
   same (the existing vllm loaders are HF-dataset-loader based, not
   sglang-specific; we can reuse the entire prompt-construction
   surface area by importing `vllm.examples.eval_moe.moe_expert_stats`
   as a library).

## Phase 1 — Smoke job on gpu-a100-short (the immediate ask)

Single sbatch that:
1. Loads modules (CUDA/12.4.1, Python/3.11.3, GCCcore/11.3.0,
   Ninja/1.11.1-Python-3.11.3).
2. Sets `UV_CACHE_DIR=${REPO}/.uv_cache` (avoid blowing scratch
   quota on uv wheels).
3. Installs sglang into a fresh `${SCRATCH}/venv_sglang/`
   (separate from the vLLM venv; ~10 GB on disk after the
   Rust ext is built). Reuses the same `UV_BUILD_PARALLEL=8`.
4. Launches the sglang server in the background as
   `python -m sglang.launch_server --model
   allenai/OLMoE-1B-7B-0924-Instruct --enable-return-routed-experts
   --port 30000 --host 127.0.0.1 --mem-fraction-static 0.92
   --max-model-len 4096 --trust-remote-code`. Polls the
   `/v1/models` endpoint until 200 OK (max 300 s).
5. Runs `python moe_expert_stats_sglang.py --server
   http://127.0.0.1:30000 --model
   allenai/OLMoE-1B-7B-0924-Instruct --benchmarks mmlu
   --num-samples 8 --max-tokens 64 --output-dir
   $SCRATCH/results_smoke_sglang` — a single benchmark, 8 samples,
   to verify end-to-end before we commit to a 12-hour bf16 sweep.
6. Kills the server, dumps logs to spartan/logs/.

The Python driver (next phase) handles prompt construction, batching,
and the JSON write; the sbatch just orchestrates.

## Phase 2 — The Python driver

New file: `examples/eval-moe/moe_expert_stats_sglang.py` (~470 lines).

### Architecture

Import the vLLM analyzer's **prompt-construction + schema-write**
parts (no copy-paste) and replace the inference core:

```python
from examples.eval_moe.moe_expert_stats import (
    BenchmarkPrompts,
    BenchmarkRow,
    BENCHMARK_RECORD_KEY,
    BENCHMARK_RUN_KEY,
    BENCHMARK_QUESTIONS_CONFIG_KEY,
    BENCHMARK_PROMPT_FORMAT,
    BENCHMARK_FEWSHOT_POOL,
    _write_per_benchmark_json,
    _safe_id,
    load_mmlu_prompts, load_bbh_prompts, load_humaneval_prompts,
    load_popqa_prompts, load_include_prompts,
    _tokenize_prompts,
    _hf_arch_name,
    _resolve_local_snapshot,
    _read_model_config,
    aggregate_expert_counts,
)
```

Why this works: the vLLM analyzer's downstream half (rows → JSON
schema) is engine-agnostic. The only engine-specific code is
`_build_llm()` + `llm.generate()` + reading `completion.routed_experts`.
We replace those three with an `openai.OpenAI(...)` chat-completion
call + an inline base64 decode of `meta_info.routed_experts`. We
deliberately do NOT import `sglang.srt.state_capturer.routed_experts`
in the driver — that pulls in a chain of SRT-internal modules (CUDA
backends, forward-batch info, runtime context) that fail at import
time on a non-GPU login node. The 5-line base64 decode is inlined;
the production-grade helper in sglang's tree does the same thing.

### Aggregation parity

`aggregate_expert_counts()` (the function in the vLLM analyzer) takes
`outputs: Iterable[Any]` and reads `completion.routed_experts` — the
shape is `(seq_len, num_layers, top_k)` for BOTH engines, so we feed
it a list of fake `RequestOutput`-like objects. The cleanest path is
to make a tiny shim that copies the relevant fields off an sglang
`ChatCompletion` response and into a duck-typed object the aggregator
can walk. ~30 lines of shim, no copy of the aggregation logic.

### Output paths

Mirror the vLLM layout so cross-engine Jaccard tools can sweep both:

```
${SCRATCH}/results_wide_bf16_sglang_a100/
  <model_safe>/<quant_safe>/moe-<bench>/expert_counts.json
```

`<model_safe>` matches the vLLM analyzer's `_safe_id()` so the
aggregator's tree-walk picks up both engines identically. Quant is
always the literal string `bf16` (no quant method applied at the
sglang server) — that lets the sister aggregator group by quant
across engines.

### What we DO NOT do in v1

- No quant variants (just bf16). The user asked for bf16 first; W4A16
  / W8A16 sglang support is a separate workstream and out of scope
  here.
- No per-cell try/except on the analyzer side — single cell, single
  job, simpler. Wrap at sbatch level if we later batch.
- No new aggregator changes. The downstream
  `jaccard_sweep_from_cpp.py` is a tree-walker; as long as the sglang
  tree is at a different root (`results_wide_bf16_sglang_a100/`),
  adding a new root to its arg is a 1-line change.

## Phase 3 — bf16 wide sweep (the real ask, after smoke passes)

For 3 small models (OLMoE, DeepSeek, Qwen3) on gpu-a100-short:
- 3 cells × 5 benchmarks = 15 JSONs.
- ~1.5h per model, ~4.5h wall total in one job.
- Single sbatch reuses the smoke sbatch; just bumps
  `MODELS_OVERRIDE` to all 3 and `BENCHMARKS` to the full list.

For Mixtral on gpu-a100 (TP=4):
- 1 cell × 5 benchmarks = 5 JSONs.
- 12h wall, 4 GPUs, mem-fraction-static 0.85.
- Forked sbatch (mirrors `vllm-moe-eval-h100-w8a16-mixtral-tp4.sbatch`
  structure but for sglang).

## Phase 4 — Cross-engine Jaccard sweep

Once the sglang tree is populated, add the new root to the sister
aggregator and re-run. Expected output: 3 panels (vLLM, llama.cpp,
sglang) of expert-routing Jaccard similarity per (model, quant,
benchmark). The interesting scientific question — does inference
backend bias routing? — gets answered by the off-diagonal Jaccard
between same-(model, quant) cells across engines. If the per-engine
trees are nearly identical (Jaccard > 0.95), the answer is "no,
backend doesn't matter, only weights do." If they diverge, that's a
publishable finding.

## Open questions before writing the smoke sbatch

1. **Module availability on a100-short**: A100 nodes have driver
   CUDA 12.4 with `NCCL/2.22.3-CUDA-12.4.1`. Same as gpu-h100 per
   the handoff. Confirm `module spider NCCL/` on an a100-short node
   before submission. (Login can be firewalled; use the handoff's
   `auto_nccl_for_cuda` probe pattern.)
2. **sglang CLI flag spelling**: `--enable-return-routed-experts`
   vs `--enable-return-routed-experts`. Verify in
   `python/sglang/srt/arg_groups/fields/exec_.py` before pasting
   into the sbatch.
3. **sglang's `torch==2.14.1` build-system pin**: vLLM venv has
   `torch==2.13.0+cu129`. We install sglang into a fresh venv so
   this conflict doesn't matter, but the source build of sglang
   may try to rebuild torch. Pin with `UV_PYTHON=...` + skip
   torch install via `--no-deps` for torch, then install
   `torch==2.13.0+cu129` separately, OR accept the 2.14.1 build
   and verify it doesn't break on CUDA 12.4.1 driver (cu130
   forward-compat).
4. **Disk budget**: scratch at 73% (383 GB free). A fresh sglang
   venv is ~10 GB. The bf16 sweep writes ~5-10 GB of new JSONs.
   Plenty of headroom; no quota guard needed.
5. **Mixtral pre-staging**: the model is already on scratch at
   `hf_cache/models--mistralai--Mixtral-8x22B-Instruct-v0.1/`
   (262 GB). No additional download. The smoke job does NOT need
   Mixtral.

## UPDATE (Oct 10, 2026): sglang 0.5.20+ has a hard cu13 wall

After two failed sbatch runs we have hard evidence that **sglang
0.5.20 and later cannot run on the current Spartan CUDA 12.8
driver stack.** Findings (in order of investigation):

1. **sglang 0.5.20 wheel deps include `flashinfer_python[cu13]==0.6.18`,
   `cuda-python>=13.0`, `humming-kernels[cu13]`, `nvidia-cutlass-dsl[cu13]`.
   uv's resolver reports `sglang>=0.5.21 and torch>=2.13.0+cu129 are
   incompatible` because torch 2.13.0+cu129 needs
   `cuda-bindings<13` but sglang needs `cuda-python>=13` (which brings
   `cuda-bindings>=13`).** Workaround attempts:
   - Use older sglang: 0.5.19 doesn't have the capturer API or
     `--enable-return-routed-experts` server flag. 0.5.20 is the
     first version with both, but also the first version to require
     cu13. Pre-0.5.20 is a dead end.
   - Install with `--no-deps` and provide cu128 flashinfer 0.6.17
     ourselves: the Python import chain requires the full dep tree
     (torchvision, msgspec, decord2, etc.) before sglang will even
     import. Each install reveals the next missing dep. Not
     impossible but ~30+ packages to chase, and the cu13-only kernels
     inside sglang's compiled `.so` files (e.g. `nvidia-cutlass-dsl[cu13]`)
     will fail at first GPU call.
2. **All A100 and H100 nodes on Spartan run driver 570.x (CUDA 12.8
   max).** Per `spartan-weather`: gpu-a100 24/24 nodes used; gpu-a100-short
   2/2 used; gpu-h100 100% busy. No node has the CUDA 13+ driver
   (580.x) sglang 0.5.20+ requires.
3. **The local sglang checkout's `python/pyproject.toml` also pins cu13
   deps** (`flashinfer_python[cu13]==0.7.0.post1`,
   `humming-kernels[cu13]==0.1.12`). So a source install with the local
   repo hits the same wall. The only way around it would be to fork
   the dep list in `pyproject.toml`.

### Decision

Document the blocker in the eval-moe handoff and the sglang
subdir. Keep the driver + smoke sbatch + debug sbatch as artifacts
that are ready to run on any future sglang-compatible HPC (or after
Spartan upgrades to a CUDA 13+ driver). Do not block the project on
sglang — the vLLM + llama.cpp comparison already answers the core
"do weight precision and expert-routing method interact?" question.

If a CUDA 13+ node becomes available later:

- The sglang wheel install resolves cleanly (no `--no-deps` dance).
- The driver code is unchanged; it only needs the sglang server to
  accept `extra_body={"return_routed_experts": True}` and return
  base64-encoded int32 expert IDs in `meta_info.routed_experts`.
- Resubmit `sglang-moe-eval-a100-smoke.sbatch` (default
  `SGLANG_VERSION=0.5.21`).
- If the model architecture changes (e.g. a new gpt-oss checkpoint
  that's not in the vllm analyzer's model registry), the driver's
  `_read_model_config` will catch it cleanly — same fallback as the
  vllm analyzer.

## Open questions before writing the smoke sbatch

## Files this plan will produce

- `examples/eval-moe/moe_expert_stats_sglang.py` — the driver.
- `examples/eval-moe/spartan/sglang-moe-eval-a100-smoke.sbatch` —
  the immediate smoke job (1 model, 1 benchmark, 8 samples).
- `examples/eval-moe/spartan/sglang-moe-eval-a100-bf16.sbatch` —
  full bf16 wide sweep on 3 small models.
- `examples/eval-moe/spartan/sglang-moe-eval-a100-bf16-mixtral.sbatch`
  — Mixtral TP=4 only.

The user explicitly asked for the smoke job as the immediate
deliverable. Phases 2-4 are gated on the smoke passing.

## What I am NOT changing in the vllm repo

- No edits to `moe_expert_stats.py`. We import it as a library
  (Python allows importing from any path on `sys.path`; the sbatch
  adds `${REPO}/examples/eval-moe` to `PYTHONPATH`).
- No new sbatches in the vLLM spartan dir; the sglang sbatches
  live alongside but are clearly named.
- No changes to the sister aggregator; that comes in Phase 4.
