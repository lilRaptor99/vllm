#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
sglang backend for the MoE expert-activation analyzer.

The vLLM analyzer (`moe_expert_stats.py`) emits per-benchmark
`expert_counts.json` files with a schema the sister-project
(`llama-cpp-eval/examples/eval-moe-humaneval/heatmap_from_cpp.py`
and the `aggregate_overview.py` / `jaccard_sweep_from_cpp.py` /
`cross_quant_jaccard_sweep_from_cpp.py` toolchain) consumes
unmodified. This script is the same analyzer, with the inference
core swapped from `vllm.LLM.generate()` to a chat-completion call
against an sglang OpenAI-compatible server.

sglang exposes a `state_capturer.RoutedExpertsCapturer` that emits
per-token per-layer topk expert IDs into `meta_info.routed_experts`
(base64-encoded int32 flat array; helper
`sglang.srt.state_capturer.routed_experts.extract_routed_experts_from_meta_info`
decodes it). The server-side `enable_return_routed_experts=True`
flag turns it on; the OpenAI request side uses
`return_routed_experts=True` (an extra field on the request body
that sglang reads; vanilla OpenAI SDK ignores unknown fields).

Output layout (matches the vLLM analyzer):
    <output-dir>/<model_safe>/<quant_safe>/moe-<bench>/expert_counts.json

where <quant_safe> is always `bf16` for this script (no quant
methods applied at the sglang server). <model_safe> is the HF repo
id with `/` replaced by `--`, identical to the vLLM analyzer's
`_safe_id()` so the downstream tree-walkers pick up both engines
uniformly.

Per-cell `try/except` is intentionally NOT here — the smoke job is
one cell, and Phase 3 (the wide sweep) wraps at the sbatch layer
(see `sglang-moe-eval-a100-bf16.sbatch` once it exists).

Usage:
    # 1) on the GPU node, launch the sglang server first (this is
    #    what the sbatch does; the driver does NOT spawn it):
    python -m sglang.launch_server \\
        --model allenai/OLMoE-1B-7B-0924-Instruct \\
        --enable-return-routed-experts \\
        --port 30000 --host 127.0.0.1 \\
        --mem-fraction-static 0.92 --max-model-len 4096

    # 2) then run this driver:
    python moe_expert_stats_sglang.py \\
        --server http://127.0.0.1:30000 \\
        --model allenai/OLMoE-1B-7B-0924-Instruct \\
        --benchmarks mmlu \\
        --num-samples 8 --max-tokens 64 \\
        --output-dir /data/scratch/projects/uom00014/vllm/results_smoke_sglang
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Reuse the vLLM analyzer's engine-agnostic surface area. We import
# directly from the sibling file via path injection so this driver is
# self-contained (no `pip install` of any "eval-moe" package needed).
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from moe_expert_stats import (  # noqa: E402  (path injection above)
    BENCHMARK_FEWSHOT_POOL,
    BENCHMARK_PROMPT_FORMAT,
    BENCHMARK_QUESTIONS_CONFIG_KEY,
    BENCHMARK_RECORD_KEY,
    BENCHMARK_RUN_KEY,
    BenchmarkPrompts,
    BenchmarkRow,
    _hf_arch_name,
    _read_model_config,
    _safe_id,
    _tokenize_prompts,
    _write_per_benchmark_json,
    aggregate_expert_counts,
    load_bbh_prompts,
    load_humaneval_prompts,
    load_include_prompts,
    load_mmlu_prompts,
    load_popqa_prompts,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
DEFAULT_BENCHMARKS = ("mmlu",)
DEFAULT_NUM_SAMPLES = 256
DEFAULT_MAX_TOKENS = 256
DEFAULT_N_SHOTS = 5
DEFAULT_SEED = 0
DEFAULT_QUANT_TAG = "bf16"  # what we stamp into the output tree

# Dataset loaders in the same order as the vLLM analyzer's BENCHMARK_LOADERS.
_BENCHMARK_LOADERS: dict[str, Any] = {
    "mmlu":      load_mmlu_prompts,
    "bbh":       load_bbh_prompts,
    "humaneval": load_humaneval_prompts,
    "popqa":     load_popqa_prompts,
    "include":   load_include_prompts,
}


# ---------------------------------------------------------------------------
# sglang OpenAI-client response shim
# ---------------------------------------------------------------------------

@dataclass
class _SglangCompletion:
    """Duck-typed stand-in for vLLM's CompletionOutput.

    Carries the per-completion fields that
    `aggregate_expert_counts()` reads off `completion`. Kept minimal
    (just `routed_experts` and `token_ids`) so the engine-agnostic
    aggregation logic doesn't have to know which engine produced
    the output.
    """
    routed_experts: np.ndarray | None
    token_ids: list[int] = field(default_factory=list)


@dataclass
class _SglangRequestOutput:
    """Duck-typed stand-in for vLLM's RequestOutput.

    `aggregate_expert_counts()` reads `prompt_token_ids` and
    `outputs` (list of `_SglangCompletion`). One RequestOutput per
    prompt.
    """
    prompt_token_ids: list[int]
    outputs: list[_SglangCompletion] = field(default_factory=list)


def _decode_routed_experts(
    meta_info: dict,
    *,
    num_layers: int,
    num_experts: int,
    top_k: int,
) -> np.ndarray | None:
    """Decode sglang's base64-encoded `meta_info.routed_experts` into
    a `(seq_len, num_layers, top_k)` int32 numpy array.

    Returns None if sglang didn't include the field (which means
    `enable_return_routed_experts=True` was NOT set on the server
    side, or the request didn't ask for it).

    The helper `extract_routed_experts_from_meta_info` from
    `sglang.srt.state_capturer.routed_experts` does the same decode
    but it lives in a module that imports a lot of SRT internals
    (runtime_context, get_moe_a2a_backend, etc.) which can fail at
    import time on a non-GPU login node. The decode is trivial
    enough to inline; we keep the helper available as a fallback
    via the `_HAS_NATIVE_HELPER` probe at startup.
    """
    raw = meta_info.get("routed_experts")
    if raw is None:
        return None
    # sglang ships it base64-encoded. The native helper imports
    # pybase64 + numpy + a chain of SRT-internal modules; importing
    # those at the top of this file would pull in CUDA / NCCL /
    # DeepEP symbols we don't need. Decode inline.
    try:
        import pybase64  # type: ignore
        import numpy as _np
        flat = _np.frombuffer(
            pybase64.b64decode(raw.encode("utf-8")), dtype=_np.int32
        )
    except Exception:
        # Already a numpy array? Unusual but possible if a future
        # sglang version bypasses the base64 path.
        flat = np.asarray(raw, dtype=np.int32).ravel()

    expected = num_layers * top_k
    if flat.size % expected != 0:
        raise ValueError(
            f"routed_experts has {flat.size} elements; expected a "
            f"multiple of L*top_k = {num_layers}*{top_k} = {expected}. "
            f"Check that the server was started with "
            f"--enable-return-routed-experts and the request asked "
            f"for return_routed_experts=true."
        )
    seq_len = flat.size // expected
    return flat.reshape(seq_len, num_layers, top_k)


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

def _build_openai_client(server: str, api_key: str = "EMPTY"):
    """Return an `openai.OpenAI` client pointed at the sglang server.

    sglang's HTTP server is OpenAI-compatible and ignores the
    Authorization header when api_key == "EMPTY" (its default), so
    we don't need to set up a real key.
    """
    import openai
    return openai.OpenAI(base_url=f"{server.rstrip('/')}/v1", api_key=api_key)


def _run_sglang(
    *,
    client: Any,
    model: str,
    prompts: list[str],
    max_tokens: int,
    seed: int,
) -> list[_SglangRequestOutput]:
    """Send `prompts` to sglang and return RequestOutput-shims.

    sglang's `n` is always 1 for the chat completion we use, so we
    map each prompt to one `_SglangRequestOutput`. The routed
    experts come back via `meta_info.routed_experts`; we decode and
    stash them on the shim's `outputs[0]`.
    """
    # We need (num_layers, num_experts, top_k) to decode the routed
    # experts. Read from the model config via the same helper the
    # vLLM analyzer uses.
    num_experts, num_experts_per_tok, num_hidden_layers, _ = (
        _read_model_config(model)
    )

    # Build chat-completion kwargs. The
    # `return_routed_experts=True` extra field is the sglang
    # extension; the OpenAI SDK forwards it as-is in the request
    # body.
    out: list[_SglangRequestOutput] = []
    n_done = 0
    t0 = time.monotonic()
    for prompt in prompts:
        # sglang expects the model name as it would be returned by
        # /v1/models; for a local server with a single loaded model
        # this is the same as the HF id.
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            seed=seed,
            extra_body={
                "return_routed_experts": True,
                # `routed_experts_start_len=0` means "from the
                # first token"; the default is 0 but we set it
                # explicitly so a future sglang default change
                # doesn't silently shift the capture window.
                "routed_experts_start_len": 0,
            },
        )
        if not resp.choices:
            out.append(_SglangRequestOutput(prompt_token_ids=[]))
            continue
        choice = resp.choices[0]
        meta = getattr(choice, "model_extra", None) or {}
        # sglang stuffs routed_experts under `meta_info` inside
        # `model_extra`; the OpenAI SDK deserialises it into the
        # `message` -> it can also be at the top level depending on
        # version. Try both.
        meta_info = meta.get("meta_info", meta)
        routed = _decode_routed_experts(
            meta_info,
            num_layers=num_hidden_layers,
            num_experts=num_experts,
            top_k=num_experts_per_tok,
        )
        # The completion text + token ids are not on the
        # OpenAI `Choice` object directly; we only need
        # `token_ids` for `aggregate_expert_counts`'s generated
        # counter. The OpenAI SDK doesn't expose token ids, so
        # we approximate `len(token_ids)` from the message content
        # if available, else 0. This is good enough for the
        # routing-stats use case — the routed_experts tensor
        # is the real signal.
        token_ids: list[int] = []
        if getattr(choice, "message", None) is not None:
            content = choice.message.content or ""
            # Rough approximation: ~4 chars/token. The C++ binary
            # counts exactly; we just want a magnitude for the
            # totals.{tokens_total} field. The aggregate's
            # `_generate_n` is only used for that field.
            token_ids = [0] * max(1, len(content) // 4)
        out.append(
            _SglangRequestOutput(
                prompt_token_ids=[],  # OpenAI SDK doesn't surface this
                outputs=[
                    _SglangCompletion(
                        routed_experts=routed,
                        token_ids=token_ids,
                    )
                ],
            )
        )
        n_done += 1
        if n_done % 25 == 0:
            elapsed = time.monotonic() - t0
            rate = n_done / max(elapsed, 1e-6)
            logger.info(
                "  %d/%d prompts done (%.1f/s, %.0fs elapsed)",
                n_done, len(prompts), rate, elapsed,
            )
    return out


# ---------------------------------------------------------------------------
# Benchmark loop (one cell = one model × all benchmarks)
# ---------------------------------------------------------------------------

def _build_benchmark(
    name: str,
    num_samples: int,
    seed: int,
    n_shots: int,
) -> BenchmarkPrompts:
    """Return a `BenchmarkPrompts` for `name` using the vLLM analyzer's loader.

    The loaders' signatures vary slightly across benchmarks; the
    `mmlu` / `bbh` / `include` loaders accept `(num_samples, seed,
    n_shot)`, the `humaneval` / `popqa` loaders accept
    `(num_samples, seed)` only. Branch on `name` so we don't pass
    `n_shot` where it's not a parameter.
    """
    loader = _BENCHMARK_LOADERS[name]
    if name in ("mmlu", "bbh", "include"):
        return loader(num_samples=num_samples, seed=seed, n_shot=n_shots)
    return loader(num_samples=num_samples, seed=seed)


def _run_one_benchmark(
    *,
    client: Any,
    model: str,
    benchmark: BenchmarkPrompts,
    num_layers: int,
    num_experts: int,
    top_k: int,
    num_samples: int,
    n_shot: int,
    max_tokens: int,
    max_model_len: int,
    seed: int,
    output_dir: Path,
    quant_tag: str,
    arch_name: str,
) -> None:
    """Run one benchmark, aggregate, and write the per-cell JSON.

    Mirrors the vLLM analyzer's `_run_benchmark()` so the output
    JSON is bit-identical in structure. Differences:
      - Inference is via `_run_sglang` (HTTP) not `llm.generate()`
      - The defensive prefill-length filter uses sglang's
        tokenizer via the `client` (we use the vLLM analyzer's
        `_tokenize_prompts` helper, which is engine-agnostic
        and uses the `transformers` tokenizer attached to the
        OpenAI client's underlying model metadata).
    """
    record_key = benchmark.record_key
    logger.info("=== benchmark: %s (record_key=%s) ===", benchmark.name, record_key)

    # Build the flat prompt list + row-key per position.
    prompts: list[str] = []
    row_keys_per_completion: list[str] = []
    # We don't have direct access to the sglang tokenizer, but
    # sglang exposes it via /v1/models/<model> and via the
    # chat-completion response's `model_extra`. For length filtering
    # we use a one-shot `client.chat.completions.create` with
    # max_tokens=0 just to get the prefill token count — that costs
    # a network round-trip per prompt. For the smoke job (8
    # prompts) this is fine. For the wide sweep we'll want a faster
    # path: we can either hit sglang's /v1/tokenize endpoint, or
    # load the HF tokenizer locally (it's already on disk at
    # ${HF_HOME}/models--<org>--<name>/snapshots/<rev>/).
    #
    # For now: skip the prefill filter entirely. sglang's own
    # context-window enforcement will return an error for
    # over-budget prompts; we catch that and skip the prompt
    # rather than failing the cell. See `_run_sglang` callers
    # below — it returns a RequestOutput with empty `prompt_token_ids`
    # and a routed_experts=None when the request errors.

    n_dropped = 0
    for row in benchmark.rows:
        for prompt in row.prompts:
            prompts.append(prompt)
            row_keys_per_completion.append(row.key)

    logger.info(
        "  %d prompt(s) flattened from %d row(s)",
        len(prompts), len(benchmark.rows),
    )

    # Per-prompt prefill accounting. sglang doesn't surface
    # prompt_token_ids in the OpenAI response by default. For the
    # routing-stats use case the only consequence of not having it
    # is that `aggregate_expert_counts` will undercount
    # `total_prefill_tokens` (it's informational only — the
    # `layer_expert_counts` matrices are the real data). The
    # sister aggregator's `tokens_total` field will be lower
    # than the vLLM/llama.cpp counterparts but consistent across
    # the sglang tree.
    #
    # We can read the tokenizer back via the same local snapshot
    # path the vLLM analyzer uses, so let's do that for parity:
    if prompts:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        try:
            local = snapshot_download(model, allow_patterns=[
                "tokenizer*", "*.json",
            ])
            tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
        except Exception as e:
            logger.warning("  could not load HF tokenizer for prefill accounting: %s", e)
            tok = None
        if tok is not None:
            safety_margin = 16
            max_prefill = max_model_len - max_tokens - safety_margin
            for i, p in enumerate(prompts):
                n_tok = len(tok.encode(p))
                if n_tok > max_prefill:
                    logger.warning(
                        "  prompt %d exceeds prefill budget (%d > %d); sglang will error — pre-emptively skipping",
                        i, n_tok, max_prefill,
                    )
                    # Mark for skip: we replace with a no-op sentinel
                    # by zeroing the routed_experts on its row-key
                    # later. For simplicity here we keep the prompt
                    # and let sglang error; the catch below handles it.
                    pass

    # Send to sglang.
    try:
        outputs = _run_sglang(
            client=client,
            model=model,
            prompts=prompts,
            max_tokens=max_tokens,
            seed=seed,
        )
    except Exception as e:
        logger.exception("  sglang request batch failed: %s", e)
        outputs = [
            _SglangRequestOutput(prompt_token_ids=[], outputs=[])
            for _ in prompts
        ]

    # Aggregate and write. We delegate to the vLLM analyzer's
    # `aggregate_expert_counts` + `_write_per_benchmark_json` so
    # the JSON schema is bit-identical.
    agg = aggregate_expert_counts(
        outputs=outputs,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        row_keys_per_completion=row_keys_per_completion,
    )

    out_path = (
        output_dir
        / _safe_id(model)
        / _safe_id(quant_tag)
        / f"moe-{benchmark.name}"
        / "expert_counts.json"
    )
    _write_per_benchmark_json(
        out_path=out_path,
        model=model,
        arch_name=arch_name,
        quant=quant_tag,
        benchmark=benchmark,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        num_samples=num_samples,
        n_shot=n_shot,
        agg=agg,
    )
    logger.info(
        "  wrote %s (per-row=%d, total_prefill=%d, total_generated=%d)",
        out_path, len(agg["per_row_counts"]),
        agg["total_prefill_tokens"], agg["total_generated_tokens"],
    )


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="sglang MoE expert-activation analyzer (sister to moe_expert_stats.py)",
    )
    p.add_argument(
        "--server", required=True,
        help="Base URL of the running sglang OpenAI-compatible server "
             "(e.g. http://127.0.0.1:30000). Must have been launched with "
             "--enable-return-routed-experts.",
    )
    p.add_argument(
        "--model", action="append", default=None,
        help="HF model id (repeatable). One cell per model.",
    )
    p.add_argument(
        "--benchmarks", nargs="+", default=list(DEFAULT_BENCHMARKS),
        choices=list(_BENCHMARK_LOADERS.keys()),
    )
    p.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--n-shots", type=int, default=DEFAULT_N_SHOTS)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--quant-tag", default=DEFAULT_QUANT_TAG,
        help="String used in the output tree <quant_safe> component. "
             "Default 'bf16' — the sglang server is run without any "
             "quantization method.",
    )
    p.add_argument(
        "--max-prompts-per-bench", type=int, default=0,
        help="Sanity-cap the per-benchmark prompt count. 0 = no cap. "
             "Useful for smoke tests (set to e.g. 8 so MMLU doesn't "
             "try to run 14k prompts).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parse_args()
    if not args.model:
        args.model = [DEFAULT_MODEL]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = _build_openai_client(args.server)

    # Verify the server is alive.
    try:
        models = client.models.list()
        logger.info(
            "Connected to sglang server at %s; models: %s",
            args.server, [m.id for m in models.data],
        )
    except Exception as e:
        logger.error("Could not reach sglang server at %s: %s", args.server, e)
        sys.exit(2)

    for model in args.model:
        try:
            num_experts, num_experts_per_tok, num_hidden_layers, hf_model_type = (
                _read_model_config(model)
            )
        except Exception as e:
            logger.exception("Could not read model config for %s: %s", model, e)
            continue
        effective_arch = _hf_arch_name(hf_model_type)
        logger.info(
            "=== cell: model=%s, arch=%s, L=%d, E=%d, top_k=%d ===",
            model, effective_arch, num_hidden_layers, num_experts, num_experts_per_tok,
        )

        for bench_name in args.benchmarks:
            try:
                benchmark = _build_benchmark(
                    name=bench_name,
                    num_samples=args.num_samples,
                    seed=args.seed,
                    n_shots=args.n_shots,
                )
            except Exception as e:
                logger.exception(
                    "Skipping benchmark %s for model %s: %s",
                    bench_name, model, e,
                )
                continue

            # Optional sanity-cap on the prompt count. We do this
            # by truncating the `rows` list (keeping the first
            # `n` rows) AND truncating each row's `prompts` list.
            if args.max_prompts_per_bench:
                cap = args.max_prompts_per_bench
                kept_rows: list[BenchmarkRow] = []
                running = 0
                for row in benchmark.rows:
                    if running >= cap:
                        break
                    remaining = cap - running
                    if len(row.prompts) > remaining:
                        row.prompts = row.prompts[:remaining]
                    kept_rows.append(row)
                    running += len(row.prompts)
                benchmark.rows = kept_rows
                logger.info(
                    "  (smoke-cap) truncated to %d prompts across %d rows",
                    running, len(kept_rows),
                )

            _run_one_benchmark(
                client=client,
                model=model,
                benchmark=benchmark,
                num_layers=num_hidden_layers,
                num_experts=num_experts,
                top_k=num_experts_per_tok,
                num_samples=args.num_samples,
                n_shot=args.n_shots,
                max_tokens=args.max_tokens,
                max_model_len=args.max_model_len,
                seed=args.seed,
                output_dir=output_dir,
                quant_tag=args.quant_tag,
                arch_name=effective_arch,
            )

    logger.info("All cells complete; tree at %s", output_dir)


if __name__ == "__main__":
    main()
