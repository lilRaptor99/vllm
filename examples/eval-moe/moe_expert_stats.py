# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Per-layer expert activation statistics for MoE models.

This script runs an MoE model on a set of benchmarks (MMLU, Big-Bench-Hard,
HumanEval) and writes one JSON file per benchmark that records, for every
layer of the model, how many tokens were routed to each expert.

The capture mechanism is vLLM's built-in `enable_return_routed_experts`,
which streams the per-token, per-layer, per-topk expert-id array out of
`CompletionOutput.routed_experts`. We only aggregate it here.

Usage:
    python examples/eval-moe/moe_expert_stats.py \\
        --models allenai/OLMoE-1B-7B-0924-Instruct \\
        --benchmarks mmlu bbh humaneval \\
        --num-samples 256 \\
        --max-tokens 256 \\
        --output-dir ./expert_stats_olmoe

    # multiple models + a quantization method (e.g. AWQ on the same repo):
    python examples/eval-moe/moe_expert_stats.py \\
        --models allenai/OLMoE-1B-7B-0924-Instruct \\
                LiteLLMs/Mixtral-8x22B-Instruct-v0.1 \\
        --quant awq \\
        --benchmarks mmlu bbh humaneval \\
        --output-dir ./expert_stats
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
DEFAULT_MODELS: tuple[str, ...] = (DEFAULT_MODEL,)
DEFAULT_QUANTS: tuple[str, ...] = ("",)  # empty = no quantization override
DEFAULT_BENCHMARKS = ("mmlu", "bbh", "humaneval")
DEFAULT_NUM_SAMPLES = 256
DEFAULT_MAX_TOKENS = 256

# MMLU letter choices shown to the model.
MMLU_LETTERS = ("A", "B", "C", "D")

# Standard 5-shot MMLU prompt template.
MMLU_5SHOT_TEMPLATE = (
    "The following are multiple choice questions (with answers) about {subject}.\n\n"
)
MMLU_QUESTION_TEMPLATE = "{question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:"


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkPrompts:
    """A list of prompts loaded from a benchmark dataset."""

    name: str
    prompts: list[str]
    extra: dict[str, object]


def _sample_indices(n: int, k: int, seed: int) -> list[int]:
    """Return `k` distinct indices in `[0, n)`, deterministically shuffled."""
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    return indices[:k]


def load_mmlu_prompts(num_samples: int, seed: int) -> BenchmarkPrompts:
    """Load `num_samples` random MMLU test prompts with a 5-shot prefix.

    Returns the formatted prompt as a string ending in "Answer:". The model
    answer letter (A/B/C/D) is not part of the prompt — we just want to
    observe expert routing on the question text.
    """
    # `datasets` ships no type stubs, so we annotate every dataset-typed
    # local as `Any`. The runtime behaviour is unchanged.
    from datasets import load_dataset  # type: ignore[import-untyped]

    test_ds: Any = load_dataset("cais/mmlu", "all", split="test")
    dev_ds: Any = load_dataset("cais/mmlu", "all", split="dev")

    # Group 5-shot examples by subject. Rows are heterogeneous dicts so we
    # annotate as `Any` for the static checker.
    dev_by_subject: dict[str, list[Any]] = {}
    for ex in dev_ds:
        dev_by_subject.setdefault(ex["subject"], []).append(ex)

    n = len(test_ds)
    # Pre-index examples by their position in the dataset so we can address
    # them in O(1) instead of scanning.
    # `cast(Iterable[Any], test_ds)` silences the cascade of partial-unknowns
    # from `enumerate[Unknown] -> list[tuple[int, Unknown]] -> Any`.
    indexed: list[tuple[int, Any]] = list(enumerate(cast(Iterable[Any], test_ds)))
    indices = _sample_indices(n, num_samples, seed)

    prompts: list[str] = []
    subjects_seen: set[str] = set()
    for idx in indices:
        _, ex = indexed[idx]
        subject: str = ex["subject"]
        subjects_seen.add(subject)
        # Rotate the dev list so we don't always lead with the same 5
        # examples within a subject.
        dev_list = dev_by_subject[subject]
        offset = idx % len(dev_list)
        fewshot = dev_list[offset : offset + 5]
        if len(fewshot) < 5:
            fewshot = fewshot + dev_list[: 5 - len(fewshot)]

        prompt = MMLU_5SHOT_TEMPLATE.format(subject=subject)
        for fs in fewshot:
            prompt += MMLU_QUESTION_TEMPLATE.format(
                question=fs["question"],
                a=fs["choices"][0],
                b=fs["choices"][1],
                c=fs["choices"][2],
                d=fs["choices"][3],
            )
            prompt += f" {MMLU_LETTERS[fs['answer']]}\n\n"
        prompt += MMLU_QUESTION_TEMPLATE.format(
            question=ex["question"],
            a=ex["choices"][0],
            b=ex["choices"][1],
            c=ex["choices"][2],
            d=ex["choices"][3],
        )
        prompts.append(prompt)

    return BenchmarkPrompts(
        name="mmlu",
        prompts=prompts,
        extra={"subjects_seen": len(subjects_seen)},
    )


def load_bbh_prompts(num_samples: int, seed: int) -> BenchmarkPrompts:
    """Load `num_samples` random prompts from the 27 BBH sub-tasks."""

    from datasets import get_dataset_config_names, load_dataset  # type: ignore

    subsets: list[str] = list(get_dataset_config_names("Joschka/big_bench_hard"))
    # Concatenate per-subset examples into a flat list, then sample from it.
    all_examples: list[dict[str, str]] = []
    for subset in subsets:
        ds: Any = load_dataset("Joschka/big_bench_hard", subset, split="test")
        for ex in ds:
            all_examples.append({"input": ex["input"], "target": ex["target"]})

    indices = _sample_indices(len(all_examples), num_samples, seed)
    prompts = [all_examples[i]["input"] for i in indices]

    return BenchmarkPrompts(
        name="bbh",
        prompts=prompts,
        extra={"num_subsets_used": len(subsets)},
    )


def load_humaneval_prompts(num_samples: int, seed: int) -> BenchmarkPrompts:
    """Load `num_samples` random HumanEval prompts (function signature + docstring)."""
    from datasets import load_dataset  # type: ignore[import-untyped]

    ds: Any = load_dataset("openai/openai_humaneval", split="test")
    indices = _sample_indices(len(ds), num_samples, seed)
    prompts = [ds[i]["prompt"] for i in indices]

    return BenchmarkPrompts(
        name="humaneval",
        prompts=prompts,
        extra={"total_available": len(ds)},
    )


BENCHMARK_LOADERS = {
    "mmlu": load_mmlu_prompts,
    "bbh": load_bbh_prompts,
    "humaneval": load_humaneval_prompts,
}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_expert_counts(
    outputs: Iterable[Any],
    num_layers: int,
    num_experts: int,
    top_k: int,
) -> tuple[np.ndarray, int]:
    """Aggregate per-token topk expert IDs into per-layer counts.

    Args:
        outputs: iterable of `RequestOutput` objects (any object whose
            `.outputs[i].routed_experts` is a `(seq_len, num_layers, top_k)`
            array, or `None`).
        num_layers: number of MoE layers in the model.
        num_experts: total number of experts per layer.
        top_k: number of experts selected per token.

    Returns:
        Tuple of (counts, total_tokens) where counts has shape
        `(num_layers, num_experts)` with dtype int64 and total_tokens is the
        sum of `seq_len` across all prompt+generated tokens routed.
    """
    counts = np.zeros((num_layers, num_experts), dtype=np.int64)
    total_tokens = 0

    for request_output in outputs:
        for completion in request_output.outputs:
            routed: Any = completion.routed_experts
            if routed is None:
                continue
            if routed.ndim != 3:
                raise ValueError(
                    f"Expected routed_experts to be 3D "
                    f"(seq_len, num_layers, top_k); got shape {routed.shape}"
                )
            if routed.shape[1] != num_layers:
                raise ValueError(
                    f"routed_experts has {routed.shape[1]} layers but model "
                    f"config has {num_layers}"
                )
            if routed.shape[2] != top_k:
                raise ValueError(
                    f"routed_experts has top_k={routed.shape[2]} but model "
                    f"config has top_k={top_k}"
                )
            # Flatten per layer: shape (seq_len, num_layers, top_k)
            # -> per-layer: reshape to (seq_len * top_k, num_layers), then
            # transpose to (num_layers, seq_len * top_k) and bincount.
            flat = np.ascontiguousarray(routed).reshape(-1, num_layers, top_k)
            for layer_id in range(num_layers):
                layer_ids = flat[:, layer_id, :].ravel()
                counts[layer_id] += np.bincount(
                    layer_ids.astype(np.int64, copy=False),
                    minlength=num_experts,
                )
            total_tokens += routed.shape[0]

    return counts, total_tokens


# ---------------------------------------------------------------------------
# Model config + LLM driver
# ---------------------------------------------------------------------------


def _read_model_config(model: str) -> tuple[int, int, int]:
    """Read num_experts, num_experts_per_tok, num_hidden_layers from HF config."""
    # `transformers` ships no type stubs for `from_pretrained`, so cast the
    # result to `Any` to silence the partial-unknown cascade.
    from transformers import AutoConfig  # type: ignore[import-untyped]

    hf_config: Any = cast(
        Any,
        AutoConfig.from_pretrained(model),  # type: ignore[no-untyped-def]
    )
    num_experts = int(hf_config.num_experts)
    num_experts_per_tok = int(hf_config.num_experts_per_tok)
    num_hidden_layers = int(hf_config.num_hidden_layers)
    return num_experts, num_experts_per_tok, num_hidden_layers


def _safe_id(name: str, fallback: str = "default") -> str:
    """Sanitize a model id or quant tag for use as a directory name.

    Replaces ``/`` with ``--`` (matching the convention used by the
    sister project's `model_safe`) and rejects any character outside
    ``[A-Za-z0-9._-]`` so the value is safe to embed in a filesystem
    path. Empty strings collapse to `fallback` so the
    no-quantization case still produces a stable directory name
    (``results/<model>/default/...``).
    """
    cleaned = "".join(c if c.isalnum() or c in "._-" else "-" for c in name)
    if name != cleaned:
        # Preserve the slash->double-dash rule explicitly; the alnum
        # filter above would turn `/` into `-` (single), but the
        # sister convention is `--`.
        cleaned = name.replace("/", "--")
        cleaned = "".join(c if c.isalnum() or c in "._-" else "-" for c in cleaned)
    if not cleaned:
        return fallback
    return cleaned


def _cell_dir(output_dir: Path, model: str, quant: str) -> Path:
    """Per-(model, quant) directory under `output_dir`."""
    return output_dir / _safe_id(model) / _safe_id(quant)


def _build_llm(
    model: str,
    quant: str,
    max_model_len: int,
    dtype: str,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    seed: int,
) -> Any:
    """Instantiate a fresh vLLM `LLM` for one (model, quant) cell.

    `quant` is forwarded to vLLM as the `quantization=` kwarg; pass an
    empty string to skip (vLLM's own default is `None`). Each cell
    gets its own KV cache so the slot buffer that backs
    `enable_return_routed_experts` is scoped to this model+quant.
    """
    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": model,
        "enable_return_routed_experts": True,
        "max_model_len": max_model_len,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
        "seed": seed,
    }
    if quant:
        kwargs["quantization"] = quant
    return LLM(**kwargs)


def _run_benchmark(
    llm: Any,
    benchmark: BenchmarkPrompts,
    sampling_params: Any,
    num_layers: int,
    num_experts: int,
    top_k: int,
    model: str,
    quant: str,
) -> dict[str, Any]:
    """Run one benchmark and aggregate per-layer expert counts."""
    logger.info(
        "Running benchmark %s with %d prompts...",
        benchmark.name,
        len(benchmark.prompts),
    )
    outputs: Any = llm.generate(benchmark.prompts, sampling_params, use_tqdm=False)
    counts, total_tokens = aggregate_expert_counts(
        outputs,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
    )

    expert_counts: dict[str, dict[str, int]] = {}
    for layer_id in range(num_layers):
        expert_counts[str(layer_id)] = {
            str(expert_id): int(counts[layer_id, expert_id])
            for expert_id in range(num_experts)
        }

    return {
        "benchmark": benchmark.name,
        "model": model,
        "quant": quant,
        "num_samples": len(benchmark.prompts),
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "total_tokens": total_tokens,
        "expert_counts": expert_counts,
        "extra": benchmark.extra,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _write_summary(
    output_dir: Path,
    model: str,
    quant: str,
    per_benchmark_results: Sequence[dict[str, Any]],
    num_layers: int,
    num_experts: int,
    top_k: int,
) -> None:
    """Write a per-layer summary aggregating across benchmarks."""
    grand_total = np.zeros((num_layers, num_experts), dtype=np.int64)
    for result in per_benchmark_results:
        for layer_id in range(num_layers):
            counts = result["expert_counts"][str(layer_id)]
            for expert_id, count in counts.items():
                grand_total[layer_id, int(expert_id)] += count

    summary: dict[str, Any] = {
        "model": model,
        "quant": quant,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "benchmarks": [r["benchmark"] for r in per_benchmark_results],
        "expert_counts": {
            str(layer_id): {
                str(expert_id): int(grand_total[layer_id, expert_id])
                for expert_id in range(num_experts)
            }
            for layer_id in range(num_layers)
        },
    }
    _write_json(output_dir / "summary.json", summary)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one or more MoE models over MMLU/BBH/HumanEval and write "
            "per-layer expert activation statistics for every "
            "(model, quant) cell."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help=(
            "One or more HuggingFace model ids to analyze. A separate "
            "LLM instance is built per model so each gets its own KV "
            "cache + routed-experts capture buffer."
        ),
    )
    parser.add_argument(
        "--quant",
        nargs="+",
        default=list(DEFAULT_QUANTS),
        help=(
            "vLLM quantization method(s) to apply to every model. "
            "Forwarded as `LLM(quantization=...)`; pass an empty token "
            "to skip (default). Repeat the flag is not supported — "
            "use a single `--quant <m1> <m2> ...` invocation. "
            "Cells are evaluated as a Cartesian product of "
            "--models x --quant."
        ),
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(DEFAULT_BENCHMARKS),
        choices=sorted(BENCHMARK_LOADERS),
        help="Which benchmarks to run.",
    )
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--output-dir",
        default="./expert_stats",
        help=(
            "Parent directory for the output tree. Each cell lands at "
            "<output-dir>/<model_safe>/<quant_safe>/<benchmark>.json "
            "+ summary.json. <model_safe> is the HF repo id with '/' "
            "replaced by '--'; <quant_safe> is the quant tag (or "
            "'default' if --quant is empty)."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    n_cells = len(args.models) * len(args.quant)
    logger.info(
        "Eval plan: %d model(s) x %d quant(s) = %d cell(s); benchmarks=%s",
        len(args.models),
        len(args.quant),
        n_cells,
        args.benchmarks,
    )

    for model in args.models:
        for quant in args.quant:
            cell_label = f"model={model}, quant={quant or '<none>'}"
            logger.info("=== cell: %s ===", cell_label)

            num_experts, num_experts_per_tok, num_hidden_layers = _read_model_config(
                model
            )
            logger.info(
                "Model config: num_experts=%d, num_experts_per_tok=%d, "
                "num_hidden_layers=%d",
                num_experts,
                num_experts_per_tok,
                num_hidden_layers,
            )

            llm = _build_llm(
                model=model,
                quant=quant,
                max_model_len=args.max_model_len,
                dtype=args.dtype,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enforce_eager=args.enforce_eager,
                seed=args.seed,
            )

            cell_dir = _cell_dir(output_dir, model, quant)
            cell_dir.mkdir(parents=True, exist_ok=True)

            per_benchmark_results: list[dict[str, Any]] = []
            for benchmark_name in args.benchmarks:
                loader = BENCHMARK_LOADERS[benchmark_name]
                benchmark = loader(num_samples=args.num_samples, seed=args.seed)
                result = _run_benchmark(
                    llm=llm,
                    benchmark=benchmark,
                    sampling_params=sampling_params,
                    num_layers=num_hidden_layers,
                    num_experts=num_experts,
                    top_k=num_experts_per_tok,
                    model=model,
                    quant=quant,
                )
                _write_json(cell_dir / f"{benchmark_name}.json", result)
                per_benchmark_results.append(result)
                logger.info(
                    "Wrote %s with %d tokens across %d layers.",
                    cell_dir / f"{benchmark_name}.json",
                    result["total_tokens"],
                    result["num_layers"],
                )

            _write_summary(
                output_dir=cell_dir,
                model=model,
                quant=quant,
                per_benchmark_results=per_benchmark_results,
                num_layers=num_hidden_layers,
                num_experts=num_experts,
                top_k=num_experts_per_tok,
            )
            logger.info("Wrote %s", cell_dir / "summary.json")

            # Free GPU memory before the next cell so two models don't
            # have to fit in VRAM simultaneously. vLLM's LLM class
            # does not expose a public shutdown() - we have to rely on
            # GC + an explicit empty_cache() call. Without the cache
            # flush, the next cell's LLM(...) would OOM even though
            # `del llm` cleared the Python reference, because PyTorch's
            # caching allocator holds onto the freed blocks.
            import gc

            import torch

            del llm
            gc.collect()
            if torch.accelerator.is_available():
                torch.accelerator.empty_cache()

    logger.info("All %d cell(s) complete; tree at %s", n_cells, output_dir)


if __name__ == "__main__":
    # Make deterministic; harmless on Linux but reduces noise on shared hosts.
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
