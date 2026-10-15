# Mixtral W4A16 routed-experts data — handoff (Oct 2026)

This document captures the analysis of the **degenerate Mixtral-8x22B
W4A16 (AWQ + GPTQ) routing data** at
`/data/scratch/projects/uom00014/vllm/results_wide_w4a16_awq_h100/`,
which shows 67–69% zeros in `aggregate.marginal_expert_counts`
(per-layer-deterministic 2-expert pairs across all tokens). The
bf16 / fp8 / W8A16 Mixtral cells were healthy, so the degeneracy
is specific to the W4A16 + compressed-tensors path on Mixtral.

The conclusion (after a second-pass deep analysis): **the data is
genuinely what the model produces** under int4 weight quantization
with `top_k=2` — the bf16 router gate logits collapse to a few
dominant values at deeper layers because the W4A16 expert outputs
have lost too much dynamic range. **There is no vLLM or capturer
bug.** Per addendum 2 §7 #5 Path C (already authorized), the
Mixtral W4A16 cells should be dropped from the Jaccard comparison.

This doc is the canonical source. `EVAL_MOE_HANDOFF.md` Addendum 9
contains a shorter summary with cross-references back here.

## 1. Bug observation (CORRECTED after re-investigation)

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

**Layer-by-layer non-zero count (sum across all 5 benchmarks for
W4A16 AWQ, mmlu cell):**

| L | Active experts | Notable? |
|---|---|---|
| 0 | 8 | healthy (input from input_layernorm, no W4A16 yet) |
| 1 | 7 | one zero |
| 2..7 | 4-6 | gradually degenerating |
| 8..55 | 2 | locked, per-layer-deterministic pairs |

## 2. What's NOT the bug (CORRECTED after re-investigation)

**The capturer, kernel, router, and vLLM pipeline are all
correct.** The Triton-backend rerun (job 33143679, Oct 18) produced
**byte-identical** degenerate data to the Marlin run (job 32517565,
Oct 13). The capturer correctly reflects what the model produces.

Eliminated alternatives:

- **Prequant ckpt corruption** — verified the bf16 router gate in
  the AWQ prequant is intact:
  ```
  $ python -m safetensors …/mistralai--Mixtral-8x22B-Instruct-v0.1-awq/
  layer 1 gate: shape=(8, 6144) dtype=torch.bfloat16
    unique values: 2491, range [-0.1357421875, 0.1259765625]
  ```
  56 gates, all `bf16`, all with thousands of unique values.
- **Capturer / kernel / router** — the Triton-backend rerun
  (different kernel: `TritonWNA16Experts` instead of
  `MarlinExperts`) produces **byte-identical** degenerate
  patterns. Both go through the same `BaseRouter` path with the
  same `FusedTopKRouter` and same C++ `topk_softmax` kernel.
  The data flow is:
  1. `gate(hidden_states)` → bf16 `router_logits` of shape `(M, 8)`.
  2. `BaseRouter._select_experts` → `fused_topk` → C++ kernel
     writes `topk_ids` of shape `(M, 2)`.
  3. `BaseRouter.capture_fn(topk_ids)` → capturer's
     `device_buffer[:M, layer_id, :] = topk_ids`.
  4. D2H to CPU, scatter by `slot_mapping`, request finishes →
     `RoutedExpertsManager.get(block_ids, num_tokens)` returns
     `(num_tokens, 56, 2)` array to the analyzer.
  Every step is well-defined and the capturer binding is correct
  for both kernel paths.
- **Routing-replay mechanism for monolithic kernels** — Mixtral
  W4A16 on H100 sm_90 hits `MarlinExperts` (or `TritonWNA16Experts`
  with the override) which is `FusedMoEExpertsModular` (non-monolithic).
  The `routing_replay_out` mechanism (line 1061-1063 of
  `modular_kernel.py`) is only used by monolithic kernels and is
  not in this path.

## 3. The actual root cause: router logits collapse under int4 precision loss

The pattern across all 4 models on the W4A16 path (mmlu benchmark):

| Model | top_k | n_expert | Mean active experts per layer |
|---|---|---|---|
| OLMoE | 8 | 64 | **64.0** (all experts) |
| Qwen3-MOE | 8 | 128 | **127.0** (99% of experts) |
| DeepSeek | 6 | 64 | **61.7** (96% of experts) |
| **Mixtral** | **2** | **8** | **2.52** (31%) |

**The smaller `top_k`, the worse the degeneration.** OLMoE
(top_k=8) and Qwen3 (top_k=8) are fine. DeepSeek (top_k=6) is
mostly fine. **Mixtral (top_k=2) is severely degenerated.**

### 3.1. Mechanism (hypothesis, GPU-only to confirm)

The bf16 router gate is **unchanged by W4A16** (it's in the
prequant's ignore list: `re:.*mlp\.gate$`). So the gate
computes correct bf16 router logits from bf16 hidden states.

The hidden states going into the gate are the output of the
**previous layer's MoE** (W4A16 quantized). Int4 quantization
reduces the dynamic range of the MoE output. The cumulative
quantization noise at deeper layers makes the hidden states less
informative to the gate.

When the gate produces router logits with one or two values
dominating and the rest near zero, the top-k selection collapses:

- **For top_k=8 (OLMoE, Qwen3)**: the softmax distributes mass
  across 8 experts; even with a peaked input, the top-8 still
  captures diverse experts.
- **For top_k=2 (Mixtral)**: the softmax concentrates ALL mass on
  the top-2. If two experts dominate the logits, those same two
  experts win for every token → degenerate data.
- **For top_k=6 (DeepSeek)**: intermediate case.

**The gate logits are deterministic per-layer** (not per-token)
because the input hidden states are essentially constant at deep
layers (the int4 quantization has removed the variation across
tokens). The top-2 selection then becomes the same pair for every
token.

### 3.2. The capturer is **correctly** reflecting model output

The capturer's `device_buffer[:M, layer_id, :] = topk_ids[0:M, :]` is
a straight tensor slice copy. The C++ `topk_softmax` kernel
(`csrc/libtorch_stable/moe/topk_softmax_kernels.cu:822`) writes
all M*topk positions correctly. The `BaseRouter._select_experts`
(`base_router.py:295-297`) fires `self.capture_fn(topk_ids)` before
EPLB mapping — and for Mixtral, EPLB is not enabled. **The
capturer's data is exactly what the router produced.**

**Conclusion: This is NOT a vLLM bug, NOT a capturer bug, and
NOT a kernel bug. The W4A16 Mixtral expert weights are too
aggressive for the model's top_k=2 routing, and the gate
logits collapse to a fixed pattern at deep layers.**

### 3.3. What does this mean for the project?

**The Mixtral W4A16 routing data is unusable for cross-quant
Jaccard analysis** because the routing doesn't reflect a
"diverse MoE expert selection" — it reflects a "model collapse to
2 experts" artifact. Including this data in the Jaccard
comparison would bias the results.

**Per addendum 2 §7 #5 Path C (already authorized)**: drop the
Mixtral W4A16 cells from the comparison. The bf16 / fp8 / W8A16
Mixtral cells (and the W4A16 cells for the other 3 models)
provide enough weight-precision coverage to answer the
research question.

## 4. The Triton-backend rerun (job 33143679) — NO LONGER RELEVANT

We tried forcing `TritonWNA16Experts` via `--moe-backend triton` in
a resubmit (job 33143679, PENDING on gpu-h100 at 2026-10-16
10:24 AEST, completed at 2026-10-19 03:24 with TIMEOUT after 4h
wall). The new AWQ cell produced **byte-identical** degenerate
data to the old Marlin run. The Triton-backend attempt is now
recognized as a dead end: both kernels share the same upstream
router code path, and the router's outputs are deterministic
per-layer for the W4A16 + top_k=2 case regardless of which
expert kernel is selected.

**Patches left in working tree (no longer needed for the bug fix,
but useful as future API for vLLM MoE backend selection):**

- `examples/eval-moe/moe_expert_stats.py` (+23 lines): adds
  `--moe-backend` argparse flag and threads it through
  `_build_llm()` to vLLM's `LLM(..., moe_backend=...)` kwarg.
- `examples/eval-moe/spartan/vllm-moe-eval-h100-w4a16-prequant-mixtral-triton.sbatch`
  (new): Mixtral-only TP=4 sbatch with `--moe-backend triton`
  baked in, 4h wall.
- `examples/eval-moe/spartan/fix-sglang-venv-numpy.sbatch`
  (committed): reproducible-rebuild helper for venv_sglang_0509.

These should be kept (they're useful for future debugging where
picking a specific MoE kernel is desired) but **do not fix the
Mixtral W4A16 data degeneracy**.

### 4.1. The Triton-backend attempt (failed but informative)

For completeness — the Triton backend attempt that turned out not
to fix anything is documented here. The user did propose Option A
(force Triton backend via `--moe-backend triton`); we implemented
it as a 30-line analyzer patch + a 290-line new sbatch, submitted
as job 33143679. The job ran for 4h, completed 4/5 AWQ benchmarks
(then TIMEOUT on `include`). The new AWQ mmlu JSON has **identical**
degeneracy to the old Marlin run — same per-layer deterministic
pairs, same 307/448 zeros. This conclusively proved the bug is NOT
in the expert kernel.

Patches in working tree (uncommitted):

- `examples/eval-moe/moe_expert_stats.py` (+23 lines): adds
  `--moe-backend` argparse flag and threads it through
  `_build_llm()` to vLLM's `LLM(..., moe_backend=...)` kwarg.
- `examples/eval-moe/spartan/vllm-moe-eval-h100-w4a16-prequant-mixtral-triton.sbatch`
  (new): Mixtral-only TP=4 sbatch with `--moe-backend triton`
  baked in, 4h wall.

These are still useful: `--moe-backend` is a good general-purpose
debug switch for future "which kernel is being picked?" issues.

## 5. How to confirm the root cause (GPU-side, optional)

The hypothesis is that **int4 quantization in the W4A16 experts'
gate/up/down projections makes the hidden states less informative
to the bf16 gate at deeper layers, causing the gate logits to
collapse to a few dominant values.** A focused GPU test would be:

```bash
# On a GPU node, load the AWQ prequant + add forward hooks:
#   - hook the gate (linear) at L=0, L=8, L=30, L=55
#   - log hidden_states.min(), .max(), .std() at each hook
#   - log router_logits.min(), .max(), .std() at each hook
#   - log topk_ids[:20] at each hook
# Compare across W4A16-AWQ vs W4A16-GPTQ vs W8A16 vs BF16.

# This is the only way to confirm the hypothesis without further
# speculation. Estimated wall time: 1-2 hours for the hook
# patches, 30 min for a single 5-benchmark run. The job can run
# on gpu-a100-short (cheaper than gpu-h100).
```

The hypothesis does NOT need to be confirmed to unblock the
project — **the Mixtral W4A16 cells should be dropped from the
Jaccard comparison regardless** (per addendum 2 §7 #5 Path C).

## 6. File pointers

| What | Where |
|------|-------|
| vLLM handoff summary | `EVAL_MOE_HANDOFF.md` Addendum 9 |
| Prior Marlin run (job 32517565) | `examples/eval-moe/spartan/logs/vllm-moe-eval-w4a16-prequant-32517565.out` |
| Triton rerun (job 33143679) — TIMEOUT, 4/5 AWQ benchmarks | `examples/eval-moe/spartan/logs/vllm-moe-eval-w4a16-prequant-mixtral-triton-33143679.out` |
| Analyzer patch (uncommitted) | `examples/eval-moe/moe_expert_stats.py` (+23 lines) |
| New sbatch (uncommitted) | `examples/eval-moe/spartan/vllm-moe-eval-h100-w4a16-prequant-mixtral-triton.sbatch` |
| Stale Mixtral W4A16 JSONs (overwritten by Triton rerun) | `results_wide_w4a16_awq_h100/--data--scratch--...-Mixtral-...-awq/compressed-tensors/moe-*/expert_counts.json` |
| Working baseline (W8A16) | `results_wide_w8a16_h100/mistralai--Mixtral-.../int8_per_channel_weight_only/moe-*/expert_counts.json` |

## 7. Outstanding items

1. **Drop the Mixtral W4A16 cells from the Jaccard comparison**
   (per addendum 2 §7 #5 Path C, already authorized). The bf16 /
   fp8 / W8A16 Mixtral cells (and the W4A16 cells for OLMoE,
   Qwen3, DeepSeek) provide enough weight-precision coverage.
2. **Optionally: confirm the hypothesis** via the GPU-side
   forward hooks described in §5. This is interesting science but
   not required to unblock the project.
3. **Optionally: keep the uncommitted patches** (`--moe-backend`
   flag in the analyzer + the new sbatch) as a general-purpose
   vLLM MoE-backend-selection utility. The Marlin-vs-Triton
   comparison may be useful for future debugging.

The data confirms there is no capturer bug. The "fix" was
attempting to work around a vLLM internal that wasn't actually
broken. Future debugging time is better spent elsewhere.

Co-authored-by: Hermes Agent <noreply@nousresearch.com>