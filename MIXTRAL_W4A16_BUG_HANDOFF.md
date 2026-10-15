# Mixtral W4A16 routed-experts capturer bug — handoff (Oct 2026)

This document captures the **diagnosis and in-flight fix** for a bug
discovered on 2026-10-16 in the Mixtral-8x22B-Instruct-v0.1 W4A16
(AWQ + GPTQ) analyzer results at
`/data/scratch/projects/uom00014/vllm/results_wide_w4a16_awq_h100/`.
The bf16 / fp8 / W8A16 Mixtral cells were healthy, so the bug is
specific to the W4A16 + compressed-tensors path on Mixtral.

This doc is the canonical source. `EVAL_MOE_HANDOFF.md` Addendum 9
contains a shorter summary with cross-references back here.

## TL;DR

- **Bug**: For Mixtral W4A16 (AWQ + GPTQ), the capturer writes
  per-layer-deterministic pairs of expert IDs into the output JSON.
  48 of 56 layers show exactly 2 distinct experts, with both
  positions populated to the same total count (= total prefill +
  generated tokens for the benchmark). The pattern is consistent
  across all 14k+ prompts × 256 generated tokens.
- **Where**: vLLM `routed_experts_capturer.py`. Specifically, the
  `BaseRouter.set_capture_fn` path that Mixtral W4A16 falls into
  on H100 (sm_90) via the `MarlinExperts` kernel.
- **Root cause**: not yet pinned to a single line — most likely a
  MarlinExperts + Mixtral top_k=2 specific issue (the capturer is
  bound correctly; the router's topk_ids appear to be deterministic
  per-layer for the W4A16 path on Mixtral). The bf16 router gate in
  the prequant ckpt is verified intact (2.5k unique values per gate,
  range [-0.14, 0.13]).
- **Fix in flight**: force the WNA16 oracle to pick the
  `TritonWNA16Experts` kernel via vLLM's `--moe-backend triton`
  config (added to the analyzer as a new `--moe-backend` CLI flag).
  Triton's WNA16 path uses the same `BaseRouter` capture_fn
  plumbing that produces healthy data for `TritonExperts` (W8A16).
- **Test job**: job 33143679, PENDING on gpu-h100 at 2026-10-16
  10:24 AEST. Mixtral × {AWQ, GPTQ} × 5 benchmarks at TP=4, 4h wall.

## 1. Bug observation

For Mixtral-8x22B-Instruct-v0.1 (n_expert=8, n_experts_per_tok=2, 56
layers), the W4A16 AWQ/GPTQ JSONs at `results_wide_w4a16_awq_h100/`
show:

```
Recipe: AWQ (mmlu benchmark)
  marginal_expert_counts shape=(56, 8) zeros: 307/448 (68.5%)
  non-zero experts per layer: min=2, max=8, mean=2.5
  layers with full 8 experts:  1/56 (only L=0)
  layers with <= 2 experts:   48/56

Recipe: GPTQ (mmlu benchmark)
  marginal_expert_counts shape=(56, 8) zeros: 304/448 (67.9%)
  non-zero experts per layer: min=2, max=8, mean=2.6
  layers with full 8 experts:  1/56 (only L=0)
  layers with <= 2 experts:   48/56
```

For the W8A16 baseline (same model, same benchmarks):

```
Recipe: W8A16 (mmlu benchmark)
  marginal_expert_counts shape=(56, 8) zeros: 0/448 (0%)
  non-zero experts per layer: min=8, max=8, mean=8.0
```

The degenerate pattern in the W4A16 data, layer-by-layer
(`aggregate.marginal_expert_counts[8:12]` for the AWQ mmlu cell):

```
L= 8: [14649685, 0, 0, 0, 0, 0, 14649685, 0]    # E0 + E6
L= 9: [14649685, 0, 0, 0, 0, 0, 0, 14649685]    # E0 + E7
L=10: [0, 0, 0, 0, 14649685, 14649685, 0, 0]    # E4 + E5
L=11: [0, 14649685, 0, 0, 0, 0, 14649685, 0]    # E1 + E6
L=12: [14649685, 0, 0, 0, 0, 0, 14649685, 0]    # E0 + E6 (same as L=8)
L=13: [0, 14649685, 0, 0, 0, 0, 0, 14649685]    # E0 + E7 (same as L=9)
L=14: [0, 0, 0, 0, 0, 14649685, 14649685, 0]    # E5 + E6
L=15: [0, 0, 0, 0, 0, 0, 14649685, 14649685]    # E6 + E7
```

The pair `(E0, E6)` at L=8 and L=12 (and many similar repeats)
isn't a "stochastic router picked the same pair twice" — it's
deterministic per-layer. Each layer has its own fixed pair, and
**both positions are populated to exactly the same count**
(`14649685` = total prefill + generated tokens for mmlu across all
14,042 prompts). Per-subject verification: `abstract_algebra` (100
prompts) layer-8 slice is `[64841, 0, 0, 0, 0, 0, 64841, 0]` — so
**every single token** of **every one of 100 prompts** picks
`(E0, E6)` at L=8.

## 2. What's NOT the bug

Eliminated alternatives:

- **Prequant ckpt corruption** — verified the bf16 router gate in
  the AWQ prequant is intact:
  ```
  $ python -m safetensors …/mistralai--Mixtral-8x22B-Instruct-v0.1-awq/
  layer 1 gate: shape=(8, 6144) dtype=torch.bfloat16
    unique values: 2491, range [-0.1357421875, 0.1259765625]
    first 20 vals: [-0.0061, 0.0091, -0.0062, 0.0124, …]
  ```
  56 gates, all `bf16`, all with thousands of unique values. If the
  router could run, it would produce diverse top-2 selections.
- **Other models on the same W4A16 path are healthy**:
  ```
  OLMoE     16L/64E/k=8   zeros: 0/1024   (0%)
  Qwen3     48L/128E/k=8  zeros: 44/6144  (0.7%) — sparse selection
  Mixtral   56L/8E/k=2   zeros: 307/448  (68.5%)  ← BROKEN
  DeepSeek  28L/64E/k=6  zeros: 63/1792  (3.5%)  — consistent but
                                            some routing imbalance
  ```
  The bug is specific to **Mixtral's top_k=2** (smallest of the 4
  models), not the W4A16 path generally.
- **Dtype/narrowing bug** — `device_buffer` is `torch.int32`,
  scheduler buffer is `np.uint8`. For expert IDs 0-7 the int32→
  uint8 narrowing is lossless. Not the cause.
- **The vLLM TrtLlmMxint4ExpertsMonolithic kernel** (my initial
  suspect — see §3.1) is NOT used on H100 (sm_90). The kernel
  requires `is_device_capability_family(100)` (Blackwell), so the
  oracle falls through to MARLIN.

## 3. The kernel actually used (H100 sm_90)

Job 32517565 (the Mixtral W4A16 AWQ TP=4 run) log shows:

```
(Worker_TP0 pid=276985) INFO 10-13 10:43:51 [compressed_tensors_wNa16.py:137] Using MacheteLinearKernel for CompressedTensorsWNA16
(Worker_TP0 pid=276985) INFO 10-13 10:43:51 [compressed_tensors_moe.py:132] Using CompressedTensorsWNA16MoEMethod
(Worker_TP0 pid=276985) INFO 10-13 10:43:51 [int_wna16.py:297] Using 'MARLIN' WNA16 MoE backend.
(Worker_TP0 pid=276985) INFO 10-13 10:43:56 [int_wna16.py:408] Using MoEPrepareAndFinalizeNoDPEPModular
(Worker_TP0 pid=276985) INFO 10-13 10:43:56 [int_wna16.py:409] Using MarlinExperts
```

Oracle priority for WNA16 on H100 (`int_wna16.py:117-124`):
```python
return [
    WNA16MoEBackend.FLASHINFER_TRTLLM,   # excluded: sm_90 only supports sm_100
    WNA16MoEBackend.MARLIN,              # ← picked
    WNA16MoEBackend.BATCHED_MARLIN,
    WNA16MoEBackend.TRITON,              # ← our workaround picks this
    WNA16MoEBackend.HUMMING,
    WNA16MoEBackend.EMULATION,
]
```

`MarlinExperts` extends `FusedMoEExpertsModular` →
`is_monolithic() -> False`. So the capturer binding goes through the
`BaseRouter` branch:

```python
# routed_experts_capturer.py:283-296
if quant_method.is_monolithic:
    ...
elif isinstance(module.router, BaseRouter):
    module.router.set_capture_fn(capture_fn)  # ← Mixtral W4A16 lands here
    num_bound += 1
```

And `BaseRouter._select_experts` (base_router.py:295-297) fires
`self.capture_fn(topk_ids)` BEFORE EPLB mapping. The capturer
writes `topk_ids` into `device_buffer[:, layer_id, :]`. So the
capturer is correctly bound — but the `topk_ids` it receives from
the router for layers 8+ is degenerate.

### 3.1. What I thought the bug was (incorrect first hypothesis)

I initially suspected `TrtLlmMxint4ExpertsMonolithic` (monolithic
FlashInfer kernel) because of `supports_routing_replay_capture()`
returning True and the routing_replay_out mechanism at
`modular_kernel.py:1061-1063`. **Wrong** — that kernel doesn't run
on H100 because of the sm_90 family guard. The actual kernel is
`MarlinExperts` (non-monolithic, BaseRouter path). I corrected this
in §3 above.

### 3.2. Where the actual bug likely lives

The data signature (per-layer-deterministic 2-expert pairs across
all tokens) is consistent with one of:

a. **`vllm_topk_softmax` C++ kernel** (`ops.topk_softmax` via
`vllm._custom_ops.topk_softmax`) producing degenerate output for
top_k=2 when called from `MarlinExperts.apply`'s code path. The
kernel allocates `topk_ids = torch.empty(M, topk, dtype=int32, ...)`
and the kernel should fill it; if some slot is left unwritten for
top_k=2, the empty buffer's garbage would show up as the
deterministic pattern we see.

b. **`fused_marlin_moe` C++/CUDA kernel** treating small
(top_k=2, n_expert=8) configurations as a special case that
breaks routing-replay semantics. The kernel's `moe_align_block_size`
step might fail silently for `block_size = min(top_k, 4) = 2`.

c. **Capturer-side timing issue** — the `device_buffer` slot for
each layer is being read before the write completes (CUDA stream
ordering bug specific to MarlinExperts' intermediate launch).

I can't pin this further without GPU access. The Triton workaround
(§4) avoids the issue entirely.

## 4. The fix in flight: force Triton backend

The simplest, lowest-risk fix is to **bypass `MarlinExperts`** by
asking the WNA16 oracle to pick `TritonWNA16Experts` directly. This
is exposed via vLLM's `--moe-backend` CLI flag and `LLM(...)`
kwarg.

### 4.1. Analyzer patch

`examples/eval-moe/moe_expert_stats.py` (+23 lines, Oct 16 2026):

```diff
def _build_llm(
    ...,
    hf_overrides: dict[str, Any] | None = None,
+    moe_backend: str | None = None,
) -> Any:
    ...
+    if moe_backend:
+        kwargs["moe_backend"] = moe_backend
    return LLM(**kwargs)

def _parse_args():
    ...
+    parser.add_argument(
+        "--moe-backend", default=None,
+        help="Override vLLM's MoE backend selection. ...",
+    )

# at the _build_llm call site:
-    hf_overrides=(...),
+    hf_overrides=(...),
+    moe_backend=args.moe_backend,
```

### 4.2. Why TritonWNA16Experts should work

- It's `FusedMoEExpertsModular` (non-monolithic) → same
  `BaseRouter.set_capture_fn` flow as W8A16 `TritonExperts`.
- W8A16 already produces healthy routing data on the same model
  (all 8 experts active at every layer), proving the capturer
  binding works for `TritonExperts` + Mixtral.
- Triton's `_compute_routing` is the same `vllm_topk_softmax`
  kernel that `MarlinExperts` would use; the kernel itself is not
  the differentiator (it's the post-topk path in MarlinExperts
  that's broken).
- `TritonWNA16Experts._supports_quant_scheme` accepts
  `kInt4Static` (group_size=128, our prequant ckpt config).

### 4.3. The sbatch

`examples/eval-moe/spartan/vllm-moe-eval-h100-w4a16-prequant-mixtral-triton.sbatch`
(new file, Oct 16 2026):

- Mirrors `vllm-moe-eval-h100-w4a16-prequant.sbatch` but:
  - `MODELS_OVERRIDE` pinned to Mixtral × {AWQ, GPTQ} only
  - `--moe-backend triton` baked in (via `MOE_BACKEND=triton`
    env var → forwarded to `--moe-backend "${MOE_BACKEND}"`)
  - `NGPUS=4` (Mixtral needs TP=4 — bf16 footprint 262 GB,
    prequant 69 GB, KV cache headroom)
  - 4h wall time (the prior MARLIN TP=4 Mixtral cell took 1h09m;
    Triton is typically 1.5-2x slower for WNA16, so worst case
    ~3h per cell, fits in 4h with margin)
  - Auto-wipes the 10 stale Mixtral W4A16 expert_counts.json
    files before running, so the analyzer's idempotent skip
    doesn't skip them
  - Pre-flight guard that confirms `moe_expert_stats.py --help`
    contains `--moe-backend` (so we don't silently fall back to
    "auto" on an older tree that lacks the analyzer patch)

### 4.4. Job 33143679 (in flight)

- **Job ID**: 33143679
- **Name**: `vllm-moe-w4a16-mixtral-triton`
- **Submitted**: 2026-10-16 10:24 AEST
- **State**: PENDING on gpu-h100 (estimated start 2026-10-21)

## 5. Verification recipe (when 33143679 completes)

```bash
cd /data/gpfs/projects/uom00014/vllm

# 1. Banner check — should show "Using 'TRITON' WNA16 MoE backend"
tail spartan/logs/vllm-moe-eval-w4a16-prequant-mixtral-triton-33143679.out \
  | grep -E "WNA16|Triton"

# 2. 10 JSONs written?
find /data/scratch/projects/uom00014/vllm/results_wide_w4a16_awq_h100 \
     -path '*Mixtral*' -name expert_counts.json | wc -l
# expect: 10

# 3. Compare to W8A16 baseline
python3 -c "
import json, numpy as np
d = json.load(open('/data/scratch/projects/uom00014/vllm/results_wide_w4a16_awq_h100/--data--scratch--projects--uom00014--vllm--hf_cache_awq--mistralai--Mixtral-8x22B-Instruct-v0.1-awq/compressed-tensors/moe-mmlu/expert_counts.json'))
a = np.array(d['aggregate']['marginal_expert_counts'])
print(f'W4A16-AWQ mmlu zeros: {(a==0).sum()}/{a.size}')
print(f'non-zero experts per layer: min={(a>0).sum(axis=1).min()} max={(a>0).sum(axis=1).max()}')

b = json.load(open('/data/scratch/projects/uom00014/vllm/results_wide_w8a16_h100/mistralai--Mixtral-8x22B-Instruct-v0.1/int8_per_channel_weight_only/moe-mmlu/expert_counts.json'))
a = np.array(b['aggregate']['marginal_expert_counts'])
print(f'W8A16   mmlu zeros: {(a==0).sum()}/{a.size}')
print(f'non-zero experts per layer: min={(a>0).sum(axis=1).min()} max={(a>0).sum(axis=1).max()}')
"
# expect: W4A16 zeros <= 20%, all 8 experts active at every layer.
# If Triton path works, should match W8A16.
```

If the new data is healthy (diverse top-k across all layers): the
fix is confirmed; commit the analyzer patch and document the
workaround upstream (vllm-project/vllm issue). If Triton fails to
load the ckpt or the capturer data is still degenerate: fall back
to Option B (in-tree patch forcing the WNA16 oracle to skip
MarlinExperts on sm_90).

## 6. File pointers

| What | Where |
|------|-------|
| Analyzer patch (this commit) | `examples/eval-moe/moe_expert_stats.py` (+23 lines) |
| New sbatch | `examples/eval-moe/spartan/vllm-moe-eval-h100-w4a16-prequant-mixtral-triton.sbatch` |
| Stale Mixtral W4A16 JSONs | `/data/scratch/projects/uom00014/vllm/results_wide_w4a16_awq_h100/--data--scratch--projects--uom00014--vllm--hf_cache_awq--mistralai--Mixtral-8x22B-Instruct-v0.1-{awq,gptq}/compressed-tensors/moe-*/expert_counts.json` |
| Working baseline (W8A16) | `/data/scratch/projects/uom00014/vllm/results_wide_w8a16_h100/mistralai--Mixtral-8x22B-Instruct-v0.1/int8_per_channel_weight_only/moe-*/expert_counts.json` |
| vLLM handoff summary | `EVAL_MOE_HANDOFF.md` Addendum 9 |
| Prior MarlinExperts run | job 32517565, log at `examples/eval-moe/spartan/logs/vllm-moe-eval-w4a16-prequant-32517565.out` |
| In-flight Triton run | job 33143679, log at `examples/eval-moe/spartan/logs/vllm-moe-eval-w4a16-prequant-mixtral-triton-33143679.out` |

## 7. Outstanding items after the fix lands

1. **If TritonWNA16Experts produces healthy data** (the expected
   case): submit a one-line upstream issue to vllm-project/vllm
   describing the MarlinExperts capturer bug for top_k=2 on H100.
   The workaround (--moe-backend triton) should be documented in
   the issue.

2. **If TritonWNA16Experts fails or is also degenerate**: escalate
   to Option B — patch `int_wna16.py:280-298` to skip MarlinExperts
   for top_k=2 on sm_90. Patch needs a unit test before pushing.

3. **Once the new 10 JSONs are healthy**, re-run the 3-way Jaccard
   aggregator with the Mixtral W4A16 cells included. Currently the
   sister `llama-cpp-eval/examples/eval-moe-overview/` aggregator
   would see degenerate (constant) routing data for Mixtral W4A16
   cells and produce misleading results — verify the new cells
   don't break the Jaccard output.

4. **Option B if needed**: patch `int_wna16.py` to make the oracle
   skip MarlinExperts for `top_k=2, n_expert <= 8` on `sm_90` (or
   more generally: skip MarlinExperts when its routing-capture
   path is known-buggy). This is a one-line conditional change.
   Risk: Triton's WNA16 kernel may not be tested for small
   `top_k`, so add a unit test that constructs a `top_k=2,
   n_expert=8, group_size=128` MoE layer and runs the capturer
   through it before pushing.

Co-authored-by: Hermes Agent <noreply@nousresearch.com>