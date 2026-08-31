# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Per-layer expert activation statistics for MoE models.

This script runs an MoE model on a configurable subset of benchmarks
(MMLU, BBH, HumanEval, PopQA, INCLUDE) and writes one JSON file per
benchmark that records, for every layer of the model, how many tokens
were routed to each expert (the "marginal" counts) plus the per-layer
and adjacent-layer co-activation pair counts (mirroring the schema
emitted by the sister project's `llama-eval-moe-*` C++ binaries so the
same downstream post-processing / Jaccard-sweep tools can consume
both).

The capture mechanism is vLLM's built-in
`enable_return_routed_experts=True`, which streams the per-token,
per-layer, per-topk expert-id array out of
`CompletionOutput.routed_experts`. We aggregate it here into:
  * per-row marginal [L, E] int64 count matrices (one record per
    MMLU subject, BBH sub-task, HumanEval task, PopQA relation
    type, or INCLUDE (language, domain) group)
  * per-row [L, E, E] intra-layer and [L-1, E, E] adjacent-layer
    co-activation pair counts
  * per-benchmark [L, E] / [L, E, E] / [L-1, E, E] aggregate
    (sum across rows) for `aggregate_overview.py` to consume

The output schema is intentionally bit-compatible with the JSON
files the sister `llama-cpp-eval/examples/eval-moe-*` C++ binaries
write, so the same post-processing toolchain
(`examples/eval-moe-overview/aggregate_overview.py`,
`jaccard_sweep_from_cpp.py`, `cross_quant_jaccard_sweep_from_cpp.py`)
can read vLLM-produced and llama.cpp-produced outputs uniformly.
See the `examples/eval-moe/README.md` for the full schema doc.

Per-row record keys follow the convention from
`aggregate_overview._RECORD_KEYS`:
  * MMLU  -> "subjects"   (key = MMLU subject name)
  * BBH   -> "subjects"   (key = BBH sub-task name; `few_shot_prompts`
                            filtered)
  * HumanEval -> "tasks"  (key = task_id, e.g. "HumanEval/146")
  * PopQA -> "props"     (key = PopQA relation type, e.g. "occupation")
  * INCLUDE -> "by_langdom"  (key = "<language>::<domain>")

Per-row token accounting:
  * MMLU + BBH        -> "n_tokens" (one number; prefill + 5-shot +
                                          test question tokens;
                                          no autoregressive decode)
  * HumanEval+PopQA+INCLUDE -> "n_tokens_prefill" + "n_tokens_generated"
                                  (prefill only vs full autoregressive
                                   decode; mirrors the C++ binaries)

Usage:
    python examples/eval-moe/moe_expert_stats.py \\
        --models allenai/OLMoE-1B-7B-0924-Instruct \\
        --benchmarks mmlu bbh humaneval \\
        --num-samples 256 \\
        --max-tokens 256 \\
        --output-dir /data/scratch/projects/uom00014/vllm/results

    # multiple models + a quantization method (e.g. AWQ on the same repo):
    python examples/eval-moe/moe_expert_stats.py \\
        --models allenai/OLMoE-1B-7B-0924-Instruct \\
                LiteLLMs/Mixtral-8x22B-Instruct-v0.1 \\
        --quant awq \\
        --benchmarks mmlu bbh humaneval \\
        --output-dir /data/scratch/projects/uom00014/vllm/results
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
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

# Default parent directory where `spartan/download-datasets.sh` writes
# the pre-staged datasets. Layout:
#   ${DATASET_DIR}/<ds_short>/<split>            (mmlu, bbh, humaneval - arrow save_to_disk)
#   ${DATASET_DIR}/<ds_short>/<ds_short>.jsonl   (popqa, include - per-row jsonl)
# The analyzer reads from this path when present and falls back to a
# network load only when the cache is missing. Override via the
# `DATASET_DIR` env var for non-Spartan hosts.
DEFAULT_DATASET_DIR = os.environ.get(
    "DATASET_DIR", "/data/scratch/projects/uom00014/vllm/datasets"
)

# MMLU letter choices shown to the model.
MMLU_LETTERS = ("A", "B", "C", "D")

# Standard 5-shot MMLU prompt template.
MMLU_5SHOT_TEMPLATE = (
    "The following are multiple choice questions (with answers) about {subject}.\n\n"
)
MMLU_QUESTION_TEMPLATE = "{question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:"

# Per-benchmark metadata. The keys here are the top-level JSON record
# keys the `aggregate_overview._load_one` probe expects
# (tuple order: tasks, subjects, props, by_langdom). The C++ side uses
# the same key per dataset.
BENCHMARK_RECORD_KEY: dict[str, str] = {
    "mmlu": "subjects",
    "bbh": "subjects",
    "humaneval": "tasks",
    "popqa": "props",
    "include": "by_langdom",
}
# Per-benchmark `totals.<x>_run` key (the count of distinct records
# under the record key, e.g. "subjects_run" for mmlu).
BENCHMARK_RUN_KEY: dict[str, str] = {
    "mmlu": "subjects_run",
    "bbh": "subjects_run",
    "humaneval": "tasks_run",
    "popqa": "props_run",
    "include": "langdoms_run",
}
# Per-benchmark `config.<x>` key (the per-row question budget).
BENCHMARK_QUESTIONS_CONFIG_KEY: dict[str, str] = {
    "mmlu": "questions_per_subject",
    "bbh": "questions_per_subject",
    "humaneval": "questions_per_task",
    "popqa": "questions_per_prop",
    "include": "questions_per_langdom",
}
# Per-benchmark prompt-format tag (for the `config.prompt_format` field).
BENCHMARK_PROMPT_FORMAT: dict[str, str] = {
    "mmlu": "few_shot_chat",
    "bbh": "few_shot_chat",
    "humaneval": "completion",
    "popqa": "completion_constrained",
    "include": "few_shot_inlang_5shot",
}
# Per-benchmark `few_shot_pool` config field (only set for benchmarks
# that use a 5-shot format from a held-out split).
BENCHMARK_FEWSHOT_POOL: dict[str, str] = {
    "mmlu": "cais/mmlu dev split",
    "bbh": "self",  # BBH exposes a `few_shot_prompts` meta-config on the HF Hub
    "humaneval": "",
    "popqa": "",
    "include": "self",  # 5-shot from the same (lang, dom) group
}

# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkRow:
    """One record under the benchmark's top-level record key.

    Attributes:
        key: The record key (MMLU subject, BBH sub-task, HumanEval task id,
             PopQA relation type, or INCLUDE `<language>::<domain>`).
             Must be unique within a benchmark.
        prompts: One or more prompts belonging to this record. For most
                 benchmarks it's a single prompt per record; can be multiple
                 for batched workloads.
        prefill_tokens: Total prefill token count across all prompts in the
                        row (computed offline by the tokenizer, exact).
        generated_tokens: Total generated token count across all prompts
                          in the row (matches the model's `max_tokens` cap;
                          the actual generated count comes back from
                          `aggregate_expert_counts`).
    """

    key: str
    prompts: list[str] = field(default_factory=list)
    prefill_tokens: int = 0
    generated_tokens: int = 0


@dataclass
class BenchmarkPrompts:
    """A benchmark's per-row prompt grouping + meta.

    Attributes:
        name: Short benchmark name (e.g. "mmlu"). Used as the output
              directory suffix (`moe-<name>/expert_counts.json`).
        record_key: The top-level JSON key under which per-row records
                     will be emitted ("subjects" / "tasks" / "props" /
                     "by_langdom"). Derived from `BENCHMARK_RECORD_KEY`.
        rows: One `BenchmarkRow` per record (subject / task / prop /
              langdom).
        arch_name: A short architecture hint for the model's MoE
                   block (e.g. "olmoe", "mixtral", "gpt-oss"). Currently
                   informational only; downstream tools that need a
                   specific name can re-derive it from the HF config.
    """

    name: str
    record_key: str
    rows: list[BenchmarkRow] = field(default_factory=list)
    arch_name: str = "unknown"


def _sample_indices(n: int, k: int, seed: int) -> list[int]:
    """Return `k` distinct indices in `[0, n)`, deterministically shuffled."""
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    return indices[:k]


def _try_local_or_remote(
    local_dir: str,
    hf_id: str,
    subset: str | None,
    split: str,
) -> Any:
    """Load a dataset split from the local cache, or fall back to HF Hub.

    The Spartan login-node pre-download (`spartan/download-datasets.sh`)
    writes each split to `${DATASET_DIR}/<ds>/<split>/` as a HuggingFace
    `Dataset.save_to_disk()` arrow directory. When present,
    `load_from_disk(...)` is preferred because (a) the GPU compute nodes
    are firewalled off the public internet, so `load_dataset(...)` over
    the network always fails, and (b) `load_dataset("cais/mmlu", "all",
    ...)` on recent `datasets` versions also requires a matching
    `cais/mmlu/mmlu.py` loading script on the HF Hub, which is a
    separate download and can also fail behind the firewall. Reading the
    pre-staged arrow files is version-stable across `datasets` releases
    and needs no network.

    `subset` is the multi-config name (e.g. ``"all"`` for MMLU, a
    sub-task for BBH). It's not consumed by `load_from_disk` (the
    subset was already baked into the arrow files at save time) but is
    accepted so the call signature mirrors `load_dataset(...)`.
    """
    import os

    from datasets import load_dataset, load_from_disk  # type: ignore

    if os.path.isdir(local_dir):
        # The marker files `dataset_info.json` + at least one
        # `data-*.arrow` shard are the canonical signature of a
        # `save_to_disk` directory. The shards may live either
        # directly under `local_dir` (datasets < 3 default) or in a
        # `data/` subdirectory (datasets >= 3 default), so check
        # both layouts. Checking the marker files explicitly
        # avoids loading a half-written directory during a parallel
        # re-download.
        has_info = os.path.isfile(os.path.join(local_dir, "dataset_info.json"))
        has_shard = any(
            n.startswith("data-") and n.endswith(".arrow")
            for n in os.listdir(local_dir)
        ) or any(
            n.startswith("data-") and n.endswith(".arrow")
            for n in os.listdir(os.path.join(local_dir, "data"))
        )
        if has_info and has_shard:
            return load_from_disk(local_dir)
    # Local cache not present (e.g. the user skipped the login-node
    # pre-download and is running the analyzer somewhere with internet
    # access). Fall through to the network path.
    kwargs: dict[str, Any] = {"split": split}
    if subset is not None:
        # `datasets >= 3` renamed `name=` to `config_name=`; both are
        # accepted by the current `load_dataset` signature, so prefer
        # `config_name` and let older releases silently ignore it.
        kwargs["config_name"] = subset
        kwargs["name"] = subset
    return load_dataset(hf_id, **kwargs)


def _try_local_or_remote_jsonl(local_path: str, hf_id: str, split: str) -> list[dict]:
    """Load a per-row JSONL from local cache, or fall back to HF Hub.

    The Spartan pre-download for PopQA / INCLUDE writes a single
    `popqa.jsonl` / `include.jsonl` file with one row per question
    (see `spartan/download-popqa.py` and
    `spartan/download-include.py`). When the file is present we read
    it directly; otherwise we fall back to a network load via
    `datasets.load_dataset` (only reachable on a host with internet).

    Returns the raw list of dicts (no schema validation; the C++
    sister-project's row format is mirrored here so the two stay
    byte-compatible).
    """
    import os

    from datasets import load_dataset  # type: ignore

    if os.path.isfile(local_path):
        rows: list[dict] = []
        with open(local_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    # Local cache not present - fall through to the network path.
    return list(load_dataset(hf_id, split=split))


def _tokenize_prompts(
    tokenizer: Any, prompts: Sequence[str], add_bos: bool
) -> list[int]:
    """Tokenize a list of prompts and return the per-prompt prefill length.

    Used to populate `BenchmarkRow.prefill_tokens` for the loaders that
    build long multi-shot prompts (mmlu 5-shot, bbh 5-shot, include 5-shot).
    We need the exact prefill token count to populate the per-row
    `n_tokens` / `n_tokens_prefill` field matching the llama-cpp binary.

    The tokenizer is the vLLM `LLM.get_tokenizer()` instance - it exposes
    the same `encode(..., add_special_tokens=...)` API as a HF tokenizer.
    """
    out: list[int] = []
    for p in prompts:
        ids = tokenizer.encode(p, add_special_tokens=add_bos)
        out.append(len(ids))
    return out


def load_mmlu_prompts(
    num_samples: int, seed: int, n_shot: int = 5
) -> BenchmarkPrompts:
    """Load `num_samples` random MMLU test prompts, grouped per subject.

    Returns a `BenchmarkPrompts` with one `BenchmarkRow` per MMLU
    subject that appears in the 256-sample (typically up to 57
    subjects; some get 4-5 questions, others get fewer). Each row's
    `prompts` is a list of MMLU `n_shot`-shot formatted questions
    (default 5, matching the standard MMLU evaluation prompt;
    pass `--n-shots 0` for zero-shot). Token counts are pre-tokenised
    at load time so the per-row `n_tokens` matches the actual prompt
    length vLLM sees.
    """
    test_ds: Any = _try_local_or_remote(
        local_dir=f"{DEFAULT_DATASET_DIR}/mmlu/test",
        hf_id="cais/mmlu",
        subset="all",
        split="test",
    )
    dev_ds: Any = _try_local_or_remote(
        local_dir=f"{DEFAULT_DATASET_DIR}/mmlu/dev",
        hf_id="cais/mmlu",
        subset="all",
        split="dev",
    )

    # Group 5-shot examples by subject. Rows are heterogeneous dicts so we
    # annotate as `Any` for the static checker.
    dev_by_subject: dict[str, list[Any]] = {}
    for ex in dev_ds:
        dev_by_subject.setdefault(ex["subject"], []).append(ex)

    n = len(test_ds)
    # Pre-index examples by their position in the dataset so we can address
    # them in O(1) instead of scanning.
    indexed: list[tuple[int, Any]] = list(enumerate(cast(Iterable[Any], test_ds)))
    indices = _sample_indices(n, num_samples, seed)

    # Build per-subject row list. Subjects can repeat (one row per
    # sampled question within that subject), so we accumulate then
    # re-key the list to one row per subject.
    per_subject: dict[str, list[str]] = {}
    for idx in indices:
        _, ex = indexed[idx]
        subject: str = ex["subject"]
        dev_list = dev_by_subject[subject]
        if n_shot > 0:
            offset = idx % len(dev_list)
            fewshot = dev_list[offset : offset + n_shot]
            if len(fewshot) < n_shot:
                # Subject's dev pool is smaller than `n_shot`. Wrap around
                # to the start of the same pool so we still get a full
                # `n_shot`-shot prompt.
                fewshot = fewshot + dev_list[: n_shot - len(fewshot)]
        else:
            fewshot = []

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
        per_subject.setdefault(subject, []).append(prompt)

    rows: list[BenchmarkRow] = []
    for subj in sorted(per_subject):
        rows.append(BenchmarkRow(key=subj, prompts=per_subject[subj]))
    return BenchmarkPrompts(
        name="mmlu",
        record_key="subjects",
        rows=rows,
        arch_name="olmoe",  # default; downstream tools can re-derive
    )


def load_bbh_prompts(num_samples: int, seed: int, n_shot: int = 5) -> BenchmarkPrompts:
    """Load `num_samples` random BBH prompts, grouped per sub-task.

    Returns a `BenchmarkPrompts` with one `BenchmarkRow` per BBH
    sub-task (up to 27; `few_shot_prompts` is filtered). Each row
    contains the 0-shot question text (the C++ binary also runs
    0-shot for BBH; we keep the same default for vLLM parity).
    `n_shot` is accepted for signature parity with the other loaders
    but is ignored - BBH ships its own per-task few-shot exemplars
    that are baked into the prompt text on the HF Hub side.
    """
    hf_id = "Joschka/big_bench_hard"
    local_root = f"{DEFAULT_DATASET_DIR}/bbh"
    subsets: list[str] = []
    if os.path.isdir(local_root):
        for entry in sorted(os.listdir(local_root)):
            if entry == "few_shot_prompts":
                continue
            if os.path.isdir(os.path.join(local_root, entry, "test")):
                subsets.append(entry)
    if not subsets:
        from datasets import get_dataset_config_names  # type: ignore

        subsets = [
            c
            for c in get_dataset_config_names(hf_id)
            if c != "few_shot_prompts"
        ]

    # Per-subset example list, then we sample N total across all
    # subsets and regroup by subset key for the output rows.
    per_subset: dict[str, list[str]] = {}
    for subset in subsets:
        local_dir = f"{local_root}/{subset}/test"
        ds: Any = _try_local_or_remote(
            local_dir=local_dir,
            hf_id=hf_id,
            subset=subset,
            split=subset,
        )
        # `Joschka/big_bench_hard` exposes the user-facing prompt
        # under the column name `input` on older `datasets` releases
        # and `question` on `datasets >= 3`. Accept both.
        prompt_key = "input" if "input" in ds.column_names else "question"
        for ex in ds:
            per_subset.setdefault(subset, []).append(ex[prompt_key])

    # Sample num_samples across the concatenation of all subsets, then
    # regroup by subset key.
    all_examples: list[tuple[str, str]] = []
    for subset, prompts in per_subset.items():
        for p in prompts:
            all_examples.append((subset, p))
    indices = _sample_indices(len(all_examples), num_samples, seed)
    sampled = [all_examples[i] for i in indices]

    grouped: dict[str, list[str]] = {}
    for subset, p in sampled:
        grouped.setdefault(subset, []).append(p)

    rows: list[BenchmarkRow] = []
    for subset in sorted(grouped):
        rows.append(BenchmarkRow(key=subset, prompts=grouped[subset]))
    return BenchmarkPrompts(
        name="bbh",
        record_key="subjects",
        rows=rows,
        arch_name="olmoe",
    )


def load_humaneval_prompts(
    num_samples: int, seed: int, n_shot: int = 5
) -> BenchmarkPrompts:
    """Load `num_samples` random HumanEval prompts, one record per problem.

    Each `BenchmarkRow` key is the HumanEval `task_id` directly
    (e.g. `"HumanEval/146"` — matches the C++ sister binary's row
    key format; the upstream HF column already carries the
    `"HumanEval/"` prefix, so we don't add it again). The full
    HumanEval set has 164 problems; if `num_samples >= 164` we keep
    them all. `n_shot` is accepted for signature parity but
    HumanEval is a pure-completion benchmark with no few-shot
    exemplars.
    """
    ds: Any = _try_local_or_remote(
        local_dir=f"{DEFAULT_DATASET_DIR}/humaneval/test",
        hf_id="openai/openai_humaneval",
        subset=None,
        split="test",
    )

    indices = _sample_indices(len(ds), num_samples, seed)
    rows: list[BenchmarkRow] = []
    for i in indices:
        rows.append(
            BenchmarkRow(
                key=str(ds[i]["task_id"]),
                prompts=[ds[i]["prompt"]],
            )
        )
    return BenchmarkPrompts(
        name="humaneval",
        record_key="tasks",
        rows=rows,
        arch_name="olmoe",
    )


def load_popqa_prompts(
    num_samples: int, seed: int, n_shot: int = 5
) -> BenchmarkPrompts:
    """Load PopQA questions, one record per relation type (prop).

    Each `BenchmarkRow` key is the relation type (e.g. `occupation`,
    `place_of_birth`). The prompt format is a constrained
    `completion` style: "Q: ...\nA:" (matches the C++ binary's
    `prompt_format = "completion_constrained"`). `num_samples` is
    taken as a soft per-prop budget and subsampled evenly across
    props (matches `download_popqa.py --limit N`). `n_shot` is
    accepted for signature parity but PopQA is a pure-completion
    benchmark with no few-shot exemplars.
    """
    hf_id = "akariasai/PopQA"
    local_jsonl = f"{DEFAULT_DATASET_DIR}/popqa/popqa.jsonl"
    try:
        rows_raw = _try_local_or_remote_jsonl(local_jsonl, hf_id, "test")
    except Exception:
        rows_raw = []

    if not rows_raw:
        from datasets import load_dataset  # type: ignore

        rows_raw = list(load_dataset(hf_id, split="test"))

    # Group by prop.
    by_prop: dict[str, list[dict]] = {}
    for r in rows_raw:
        by_prop.setdefault(r["prop"], []).append(r)

    # Subsample per-prop evenly if num_samples is a per-prop budget.
    per_prop: dict[str, list[dict]] = {}
    if num_samples > 0:
        for prop, items in sorted(by_prop.items()):
            n = min(num_samples, len(items))
            # Deterministic subsample by question text (matches the
            # `download_popqa.py --limit` ordering).
            items_sorted = sorted(items, key=lambda r: r.get("question", ""))
            per_prop[prop] = items_sorted[:n]

    def _build_prompt(r: dict) -> str:
        q = str(r.get("question", "")).strip()
        return f"Q: {q}\nA:"

    rows: list[BenchmarkRow] = []
    for prop in sorted(per_prop):
        rows.append(
            BenchmarkRow(
                key=prop,
                prompts=[_build_prompt(r) for r in per_prop[prop]],
            )
        )
    return BenchmarkPrompts(
        name="popqa",
        record_key="props",
        rows=rows,
        arch_name="olmoe",
    )


def load_include_prompts(
    num_samples: int, seed: int, n_shot: int = 5
) -> BenchmarkPrompts:
    """Load INCLUDE questions, one record per (language, domain) group.

    Each `BenchmarkRow` key is `<language>::<domain>`. `n_shot`-shot
    from the same (lang, dom) group, formatted like MMLU (Q +
    A/B/C/D + "Answer:"). `num_samples` is a per-(lang, dom) budget
    for the **test** rows; the exemplars are drawn from a separate
    slice of the same group so a row never appears as both exemplar
    and test item (which would inflate the routing signal on the
    exemplar's tokens). Test split only (per the paper, validation is
    a format-error probe).
    """
    hf_id = "CohereLabs/include-base-44"
    local_jsonl = f"{DEFAULT_DATASET_DIR}/include/include.jsonl"
    try:
        rows_raw = _try_local_or_remote_jsonl(local_jsonl, hf_id, "test")
    except Exception:
        rows_raw = []

    if not rows_raw:
        from datasets import load_dataset  # type: ignore

        rows_raw = list(load_dataset(hf_id, "Dutch", split="test"))

    # Group by (language, domain).
    by_langdom: dict[str, list[dict]] = {}
    for r in rows_raw:
        if r.get("split", "test") != "test":
            continue
        key = f"{r.get('language', '?')}::{r.get('domain', 'Unknown')}"
        by_langdom.setdefault(key, []).append(r)

    # Sort each group by question for deterministic subsample.
    for k, v in by_langdom.items():
        v.sort(key=lambda r: r.get("question", ""))

    # Take `n_shot + num_samples` rows per (lang, dom). The first
    # `n_shot` rows serve as exemplars; the remaining `num_samples`
    # rows are the test items. Keeping them disjoint avoids the
    # contamination where the model's routing signal on the exemplar
    # tokens gets re-counted on the test prompt that contains them.
    per_langdom: dict[str, tuple[list[dict], list[dict]]] = {}
    if num_samples > 0:
        for key, items in sorted(by_langdom.items()):
            ex_n = min(n_shot, len(items)) if n_shot > 0 else 0
            test_n = min(num_samples, len(items) - ex_n)
            per_langdom[key] = (items[:ex_n], items[ex_n : ex_n + test_n])

    def _build_prompt(test_row: dict, exemplars: list[dict]) -> str:
        # Mirrors llama-cpp's build_fewshot_user_text: Q + 4 options + "Answer:".
        parts: list[str] = []
        for ex in exemplars:
            parts.append(str(ex.get("question", "")).strip())
            for opt in ex.get("options", []):
                parts.append(str(opt))
            ans = ex.get("answer", 0)
            letter = chr(ord("A") + int(ans)) if isinstance(ans, int) else "A"
            parts.append(f"Answer: {letter}\n")
        parts.append(str(test_row.get("question", "")).strip())
        for opt in test_row.get("options", []):
            parts.append(str(opt))
        parts.append("Answer:")
        return "\n".join(parts)

    rows: list[BenchmarkRow] = []
    for key in sorted(per_langdom):
        exemplars, test_items = per_langdom[key]
        prompts = [_build_prompt(it, exemplars) for it in test_items]
        rows.append(BenchmarkRow(key=key, prompts=prompts))
    return BenchmarkPrompts(
        name="include",
        record_key="by_langdom",
        rows=rows,
        arch_name="olmoe",
    )


BENCHMARK_LOADERS = {
    "mmlu": load_mmlu_prompts,
    "bbh": load_bbh_prompts,
    "humaneval": load_humaneval_prompts,
    "popqa": load_popqa_prompts,
    "include": load_include_prompts,
}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_expert_counts(
    outputs: Iterable[Any],
    num_layers: int,
    num_experts: int,
    top_k: int,
    row_keys_per_completion: list[str],
) -> dict[str, Any]:
    """Aggregate per-token topk expert IDs into per-row + aggregate counts.

    Walks each `RequestOutput` in `outputs`, then each
    `completion` inside it. The `routed_experts` tensor on each
    completion has shape `(seq_len, num_layers, top_k)` (vLLM's
    `enable_return_routed_experts` contract).

    For every (token, layer) cell we increment the per-row
    marginal counter `counts[row_key][layer, expert]` once for each
    of the `top_k` expert slots. We additionally compute the
    per-row intra-layer `[L, E, E]` and adjacent-layer `[L-1, E, E]`
    co-activation pair counts (one increment per token per
    `(j1, j2) in top_k^2` pair). The pair counts are identical
    in shape and semantics to the sister-project's C++ binary
    output so `aggregate_overview.py` consumes them as-is.

    Returns a dict with:
      * `per_row_counts`     : dict[row_key, np.ndarray (L, E) int64]
      * `per_row_intra`      : dict[row_key, np.ndarray (L, E, E) int64]
      * `per_row_adj`        : dict[row_key, np.ndarray (L-1, E, E) int64]
                              (omitted for the last layer pair; rows
                              with n_layer < 2 get an empty entry)
      * `total_prefill_tokens`  : int (sum across all completions)
      * `total_generated_tokens`: int (sum across all completions)
      * `marginal`           : np.ndarray (L, E) int64 (sum across rows)
      * `intra`              : np.ndarray (L, E, E) int64 (sum across rows)
      * `adj`                : np.ndarray (L-1, E, E) int64 (sum across rows)
      * `arch_name`          : always "unknown" here; the caller tags
                              the model arch (used for the JSON
                              `model_arch.name` field).

    `row_keys_per_completion` is a flat list (one row_key per
    `RequestOutput`) that the caller builds by walking the
    `BenchmarkPrompts.rows` in order and concatenating each row's
    `prompts` list. Completions in `outputs` are then matched
    positionally to those keys.
    """
    if num_layers <= 0 or num_experts <= 0 or top_k <= 0:
        raise ValueError(
            f"aggregate_expert_counts: bad shape "
            f"(num_layers={num_layers}, num_experts={num_experts}, top_k={top_k})"
        )

    # Initialise per-row accumulators lazily so a row with zero
    # completions still appears in the output (with an all-zero
    # [L, E] matrix), matching the C++ binary's behaviour. We do the
    # initialisation up front (before the request walk) so a row that
    # is requested by `row_keys_per_completion` but never gets a
    # non-None `routed_experts` still appears in the output.
    per_row_counts: dict[str, np.ndarray] = {}
    per_row_intra: dict[str, np.ndarray] = {}
    per_row_adj: dict[str, np.ndarray] = {}
    seen_keys: set[str] = set(row_keys_per_completion)
    for rk in seen_keys:
        per_row_counts[rk] = np.zeros((num_layers, num_experts), dtype=np.int64)
        per_row_intra[rk] = np.zeros((num_layers, num_experts, num_experts), dtype=np.int64)
        if num_layers >= 2:
            per_row_adj[rk] = np.zeros((num_layers - 1, num_experts, num_experts), dtype=np.int64)

    marginal = np.zeros((num_layers, num_experts), dtype=np.int64)
    intra = np.zeros((num_layers, num_experts, num_experts), dtype=np.int64)
    adj = (
        np.zeros((max(num_layers - 1, 0), num_experts, num_experts), dtype=np.int64)
        if num_layers >= 2
        else np.zeros((0, num_experts, num_experts), dtype=np.int64)
    )

    total_prefill_tokens = 0
    total_generated_tokens = 0

    # `outputs` is one element per `prompt` in the order we sent them.
    # Walk it positionally; pair with `row_keys_per_completion`.
    out_iter = iter(outputs)
    for completion_idx, request_output in enumerate(out_iter):
        row_key = row_keys_per_completion[completion_idx]

        # Request-level token accounting. vLLM's `RequestOutput` exposes
        # `prompt_token_ids` (list[int]) and each `CompletionOutput`
        # exposes `token_ids` (list[int]) + `finish_reason` for the
        # prefill + autoregressive decode counts.
        prompt_token_ids: list[int] = list(
            getattr(request_output, "prompt_token_ids", []) or []
        )
        prefill_n = len(prompt_token_ids)
        generated_n = 0
        for completion in request_output.outputs:
            generated_n += len(getattr(completion, "token_ids", []) or [])
        total_prefill_tokens += prefill_n
        total_generated_tokens += generated_n

        for completion in request_output.outputs:
            routed: Any = completion.routed_experts
            if routed is None:
                continue
            arr = np.asarray(routed)
            if arr.ndim != 3:
                raise ValueError(
                    f"Expected routed_experts to be 3D "
                    f"(seq_len, num_layers, top_k); got shape {arr.shape}"
                )
            if arr.shape[1] != num_layers:
                raise ValueError(
                    f"routed_experts has {arr.shape[1]} layers but model "
                    f"config has {num_layers}"
                )
            if arr.shape[2] != top_k:
                raise ValueError(
                    f"routed_experts has top_k={arr.shape[2]} but model "
                    f"config has top_k={top_k}"
                )

            row_counts = per_row_counts[row_key]
            row_intra = per_row_intra[row_key]
            row_adj = per_row_adj.get(row_key)

            # Cast to int64 once; downstream code uses np.bincount which
            # is faster on a contiguous int64 array.
            flat = np.ascontiguousarray(arr, dtype=np.int64)  # (T, L, K)
            T = flat.shape[0]

            # Marginal: for each layer, bincount the K expert ids per
            # token, sum across tokens.
            for layer_id in range(num_layers):
                layer_ids = flat[:, layer_id, :].ravel()
                bc = np.bincount(layer_ids, minlength=num_experts)
                row_counts[layer_id] += bc
                marginal[layer_id] += bc

            # Intra-layer pair counts: for each layer, for each token,
            # for each (j1, j2) in top_k^2, increment
            # `row_intra[L, e1, e2]`. Mirror of the C++ binary's
            # tally_pairs_for_question() inner loop. Vectorised with
            # broadcasting on the (T, K) topk slice.
            for layer_id in range(num_layers):
                topk = flat[:, layer_id, :]  # (T, K)
                # bincount on the outer product of (T*K) x (T*K) is O((T*K)^2)
                # per layer, which is fine for typical prefill lengths
                # (~1K tokens, K=8 -> 64M ops per layer) and matches the
                # C++ binary's O(T*K*K) cost. Use np.add.at for unbuffered
                # scatter to handle repeated expert ids correctly.
                T_, K_ = topk.shape
                e1 = np.broadcast_to(topk[:, :, None], (T_, K_, K_)).ravel()
                e2 = np.broadcast_to(topk[:, None, :], (T_, K_, K_)).ravel()
                bc = np.bincount(
                    e1 * num_experts + e2, minlength=num_experts * num_experts
                )
                bc_2d = bc.reshape(num_experts, num_experts)
                row_intra[layer_id] += bc_2d
                intra[layer_id] += bc_2d

            # Adjacent-layer pair counts: for each (L, L+1) pair,
            # for each token, for each (j1, j2) in top_k^2, increment
            # `row_adj[L, e1, e2]`. Same vectorisation as intra.
            if num_layers >= 2 and row_adj is not None:
                for L_ in range(num_layers - 1):
                    e1 = np.broadcast_to(
                        flat[:, L_, :, None], (T, top_k, top_k)
                    ).ravel()
                    e2 = np.broadcast_to(
                        flat[:, L_ + 1, None, :], (T, top_k, top_k)
                    ).ravel()
                    bc = np.bincount(
                        e1 * num_experts + e2, minlength=num_experts * num_experts
                    )
                    bc_2d = bc.reshape(num_experts, num_experts)
                    row_adj[L_] += bc_2d
                    adj[L_] += bc_2d

    return {
        "per_row_counts": per_row_counts,
        "per_row_intra": per_row_intra,
        "per_row_adj": per_row_adj,
        "total_prefill_tokens": total_prefill_tokens,
        "total_generated_tokens": total_generated_tokens,
        "marginal": marginal,
        "intra": intra,
        "adj": adj,
        "arch_name": "unknown",
    }


# ---------------------------------------------------------------------------
# Model config + LLM driver
# ---------------------------------------------------------------------------


def _read_model_config(model: str) -> tuple[int, int, int]:
    """Read num_experts, num_experts_per_tok, num_hidden_layers from HF config.

    `trust_remote_code=True` is required for architectures like DeepSeek-MoE
    that ship custom modeling code on the Hub. Without it, AutoConfig raises
    on a non-interactive stdin (SLURM jobs have no TTY), so the prompt for
    trust confirmation EOFs before the user can answer.

    Different MoE architectures name these fields differently:
      - OLMoE / gpt-oss : num_experts
      - Mixtral         : num_local_experts
      - DeepSeek-MoE    : n_routed_experts (plus n_shared_experts, but those
                          are NOT in vLLM's routed_experts tensor — they're
                          always-on experts, not part of the routing pool)

    The values are used downstream as the bincount minlength / matrix shape
    in `aggregate_expert_counts`, so they must match the cardinality of the
    IDs that vLLM actually emits in `routed_experts` for that architecture.
    """
    from transformers import AutoConfig  # type: ignore[import-untyped]

    hf_config: Any = cast(
        Any,
        AutoConfig.from_pretrained(model, trust_remote_code=True),  # type: ignore[no-untyped-def]
    )

    def _resolve(candidates: list[str]) -> int:
        for name in candidates:
            if hasattr(hf_config, name):
                v = getattr(hf_config, name)
                if v is not None:
                    return int(v)
        available = sorted(
            k for k in vars(hf_config).keys()
            if "expert" in k.lower() or "hidden" in k.lower() or "layer" in k.lower()
        )
        raise AttributeError(
            f"{model!r}: none of {candidates} found on the HF config "
            f"(model_type={getattr(hf_config, 'model_type', '?')!r}). "
            f"Available expert/layer/hidden attrs: {available}"
        )

    num_experts = _resolve(["n_routed_experts", "num_experts", "num_local_experts"])
    num_experts_per_tok = _resolve(["num_experts_per_tok", "top_k"])
    num_hidden_layers = _resolve(["num_hidden_layers", "n_layer", "num_layers"])
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
        cleaned = name.replace("/", "--")
        cleaned = "".join(c if c.isalnum() or c in "._-" else "-" for c in cleaned)
    if not cleaned:
        return fallback
    return cleaned


def _build_llm(
    model: str,
    quant: str,
    max_model_len: int,
    dtype: str,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    seed: int,
    tensor_parallel_size: int = 1,
) -> Any:
    """Instantiate a fresh vLLM `LLM` for one (model, quant) cell.

    `quant` is forwarded to vLLM as the `quantization=` kwarg; pass an
    empty string to skip (vLLM's own default is `None`). Each cell
    gets its own KV cache so the slot buffer that backs
    `enable_return_routed_experts` is scoped to this model+quant.

    `tensor_parallel_size` is forwarded as vLLM's `tensor_parallel_size`.
    vLLM automatically picks the right distributed executor based on
    the visible GPUs (multi-GPU + NVLink uses the V1 engine's
    tensor-parallel path; single-GPU is a no-op).

    `trust_remote_code=True` is forwarded so architectures that ship
    custom modeling code on the Hub (e.g. DeepSeek-MoE) load without an
    interactive prompt — SLURM jobs have no TTY, so a prompt would EOF.
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
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
    }
    if quant:
        kwargs["quantization"] = quant
    return LLM(**kwargs)


# ---------------------------------------------------------------------------
# Per-benchmark driver
# ---------------------------------------------------------------------------


def _matrix_to_nested_list(arr: np.ndarray) -> list[list[int]]:
    """Cast a 2D int64 array to a nested list of Python ints for JSON."""
    return [[int(v) for v in row] for row in arr]


def _matrix3d_to_nested_list(arr: np.ndarray) -> list[list[list[int]]]:
    """Cast a 3D int64 array to a nested list of Python ints for JSON."""
    return [
        [[int(v) for v in col] for col in row]
        for row in arr
    ]


def _write_per_benchmark_json(
    out_path: Path,
    model: str,
    arch_name: str,
    quant: str,
    benchmark: BenchmarkPrompts,
    num_layers: int,
    num_experts: int,
    top_k: int,
    num_samples: int,
    n_shot: int,
    agg: dict[str, Any],
) -> None:
    """Write the llama-cpp-compatible `expert_counts.json` for one cell.

    Schema is documented in `examples/eval-moe/README.md`. Brief summary:
    * `model` / `model_arch` / `config` / `totals` mirror the C++ binary.
    * `aggregate` block holds dataset-wide marginal + pair counts so
      `aggregate_overview.py` can produce its `overall/` overview
      without re-aggregating from per-row records.
    * The top-level record key (`subjects` / `tasks` / `props` /
      `by_langdom`) holds one entry per row, each with `layer_expert_counts`,
      `intra_pair_counts`, `adjacent_pair_counts`, plus per-benchmark
      token accounting (`n_tokens` for mmlu/bbh; `n_tokens_prefill` +
      `n_tokens_generated` for humaneval/popqa/include).
    """
    record_key = benchmark.record_key
    run_key = BENCHMARK_RUN_KEY[benchmark.name]
    questions_config_key = BENCHMARK_QUESTIONS_CONFIG_KEY[benchmark.name]
    prompt_format = BENCHMARK_PROMPT_FORMAT[benchmark.name]
    fewshot_pool = BENCHMARK_FEWSHOT_POOL[benchmark.name]

    # Per-row record dict.
    per_row_records: dict[str, dict[str, Any]] = {}
    L = num_layers
    E = num_experts
    for row in benchmark.rows:
        # The defensive prompt-length filter can drop all prompts in
        # a row, in which case the row's key never makes it into
        # `agg["per_row_counts"]`. Synthesise an all-zero record so
        # the JSON's per-row shape stays consistent.
        if row.key in agg["per_row_counts"]:
            per_key = agg["per_row_counts"][row.key]
            per_key_intra = agg["per_row_intra"][row.key]
            per_key_adj = agg["per_row_adj"].get(row.key)
        else:
            per_key = np.zeros((L, E), dtype=np.int64)
            per_key_intra = np.zeros((L, E, E), dtype=np.int64)
            per_key_adj = (
                np.zeros((max(L - 1, 0), E, E), dtype=np.int64)
                if L >= 2
                else None
            )
        rec: dict[str, Any] = {
            "questions": len(row.prompts),
            "layer_expert_counts": _matrix_to_nested_list(per_key),
            "intra_pair_counts": _matrix3d_to_nested_list(per_key_intra),
        }
        if per_key_adj is not None and per_key_adj.shape[0] > 0:
            rec["adjacent_pair_counts"] = _matrix3d_to_nested_list(per_key_adj)
        else:
            rec["adjacent_pair_counts"] = []

        # Token accounting: MMLU + BBH use `n_tokens` (prefill only;
        # no autoregressive decode). HumanEval + PopQA + INCLUDE use
        # the explicit `n_tokens_prefill` + `n_tokens_generated` pair
        # so downstream tools can reconstruct total token volume.
        if benchmark.name in ("mmlu", "bbh"):
            rec["n_tokens"] = int(per_key.sum())
        else:
            # Use the row-level prefill / generated counts captured at
            # the vLLM boundary via `request_output.prompt_token_ids`
            # and `completion.token_ids`. The aggregate function
            # already summed these into the global totals; we
            # attribute per-row by averaging the row's prefill across
            # its completions (good enough for the routing stats use
            # case - per-row prefill vs generated split is informational).
            # The exact per-row split is captured by the row's
            # `prefill_tokens` / `generated_tokens` set at load time
            # (from the tokenizer) - we use that for accuracy.
            rec["n_tokens_prefill"] = int(row.prefill_tokens)
            rec["n_tokens_generated"] = int(row.generated_tokens)
        per_row_records[row.key] = rec

    totals = {
        run_key: len(per_row_records),
        "questions_total": int(sum(r["questions"] for r in per_row_records.values())),
        "tokens_total": int(agg["total_prefill_tokens"] + agg["total_generated_tokens"]),
    }

    config: dict[str, Any] = {
        questions_config_key: num_samples,
        "prompt_format": prompt_format,
    }
    if n_shot > 0:
        config["n_shot"] = n_shot
    if fewshot_pool:
        config["few_shot_pool"] = fewshot_pool

    payload: dict[str, Any] = {
        "model": model,
        "model_arch": {
            "name": arch_name,
            "n_layer": num_layers,
            "n_expert": num_experts,
            "n_expert_used": top_k,
        },
        "config": config,
        "totals": totals,
        "aggregate": {
            "marginal_expert_counts": _matrix_to_nested_list(agg["marginal"]),
            "intra_pair_counts": _matrix3d_to_nested_list(agg["intra"]),
            "adjacent_pair_counts": _matrix3d_to_nested_list(agg["adj"]),
        },
        record_key: per_row_records,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _run_benchmark(
    llm: Any,
    benchmark: BenchmarkPrompts,
    sampling_params: Any,
    num_layers: int,
    num_experts: int,
    top_k: int,
    model: str,
    quant: str,
    num_samples: int,
    n_shot: int,
    tokenizer: Any,
    output_dir: Path,
    max_model_len: int,
    max_tokens: int,
) -> None:
    """Run one benchmark, aggregate, and write the per-cell JSON.

    Flattens `benchmark.rows` into a single `prompts` list, runs
    `llm.generate(...)` on it, then aggregates per-token routed
    experts into the per-row + aggregate count matrices and writes
    them to `<output-dir>/<model_safe>/<quant_safe>/moe-<bench>/expert_counts.json`.
    """
    # Flatten rows into a single prompts list; record the row key
    # for each prompt position so we can route the per-completion
    # counts back to the right per-row matrix.
    prompts: list[str] = []
    row_keys_per_completion: list[str] = []
    # Defensive filter: drop prompts whose prefill exceeds the model's
    # context window (`max_model_len - max_tokens - safety_margin`).
    # Otherwise vLLM raises `VLLMValidationError` mid-batch and the
    # whole cell aborts. We keep the row's other (in-bounds) prompts
    # so the row still appears in the output. This shows up most on
    # INCLUDE 5-shot (some (lang, dom) groups have long exemplars)
    # and MMLU 5-shot on long subjects (professional_law etc).
    safety_margin = 16
    max_prefill = max_model_len - max_tokens - safety_margin
    n_dropped = 0
    for row in benchmark.rows:
        # Refresh the row's prefill token count from the vLLM tokenizer
        # (more accurate than a static precompute). vLLM's tokenizer
        # attribute path has changed across releases: in some it's
        # `llm.llm_engine.tokenizer.tokenizer`, in newer releases it's
        # `llm.get_tokenizer()` (the stable public API) wrapped in a
        # pool with a `_tokenizer` attribute. We use the public
        # `llm.get_tokenizer()` API and probe for the inner tokenizer
        # via `_tokenizer` (newer) or `tokenizer` (older) so we work
        # across vLLM versions.
        tokenizer_obj = llm.get_tokenizer()
        inner = getattr(tokenizer_obj, "_tokenizer", None) or getattr(
            tokenizer_obj, "tokenizer", None
        ) or tokenizer_obj
        add_bos = bool(getattr(inner, "add_bos_token", False))
        per_prompt_prefill = _tokenize_prompts(tokenizer_obj, row.prompts, add_bos)
        # Filter out prompts whose prefill exceeds the budget; keep the
        # in-bounds ones. Updates per-row prefill_tokens to reflect only
        # the kept prompts.
        kept_prompts: list[str] = []
        kept_prefill: list[int] = []
        for prompt, n_tok in zip(row.prompts, per_prompt_prefill):
            if n_tok > max_prefill:
                n_dropped += 1
                logger.warning(
                    "  dropping over-budget prompt: row=%r, prefill_tokens=%d, max=%d",
                    row.key,
                    n_tok,
                    max_prefill,
                )
                continue
            kept_prompts.append(prompt)
            kept_prefill.append(n_tok)
        row.prompts = kept_prompts
        row.prefill_tokens = sum(kept_prefill)
        # `generated_tokens` is the model's `max_tokens` cap; the actual
        # generated count comes back from `aggregate_expert_counts`
        # via `RequestOutput.outputs[i].token_ids`. We initialise to 0
        # and let the aggregation overwite it; the JSON write uses the
        # aggregated value, not this initial.
        row.generated_tokens = 0
        for p in row.prompts:
            prompts.append(p)
            row_keys_per_completion.append(row.key)

    if n_dropped:
        logger.warning(
            "Dropped %d over-budget prompt(s) from benchmark %s "
            "(prefill > %d = max_model_len=%d - max_tokens=%d - margin=%d); "
            "rows with zero kept prompts will appear with empty arrays",
            n_dropped,
            benchmark.name,
            max_prefill,
            max_model_len,
            max_tokens,
            safety_margin,
        )

    logger.info(
        "Running benchmark %s: %d row(s), %d prompt(s) total",
        benchmark.name,
        len(benchmark.rows),
        len(prompts),
    )
    outputs: Any = llm.generate(prompts, sampling_params, use_tqdm=False)

    agg = aggregate_expert_counts(
        outputs=outputs,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        row_keys_per_completion=row_keys_per_completion,
    )

    # Refine per-row `generated_tokens` from the actual completions:
    # the row-level count we wrote above was 0; we now set it from
    # the vLLM-side actual generated count by walking the outputs
    # again. (For HumanEval / PopQA / INCLUDE we want the actual
    # generated count, not the cap, so the JSON's
    # `n_tokens_generated` reflects what was really decoded.)
    generated_per_row: dict[str, int] = {rk: 0 for rk in agg["per_row_counts"]}
    for completion_idx, request_output in enumerate(outputs):
        rk = row_keys_per_completion[completion_idx]
        for completion in request_output.outputs:
            generated_per_row[rk] = generated_per_row.get(rk, 0) + len(
                getattr(completion, "token_ids", []) or []
            )
    for row in benchmark.rows:
        row.generated_tokens = generated_per_row.get(row.key, 0)

    out_path = output_dir / _safe_id(model) / _safe_id(quant) / f"moe-{benchmark.name}" / "expert_counts.json"
    _write_per_benchmark_json(
        out_path=out_path,
        model=model,
        arch_name=benchmark.arch_name,
        quant=quant,
        benchmark=benchmark,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        num_samples=num_samples,
        n_shot=n_shot,
        agg=agg,
    )
    logger.info(
        "Wrote %s (rows=%d, tokens_prefill=%d, tokens_generated=%d)",
        out_path,
        len(benchmark.rows),
        agg["total_prefill_tokens"],
        agg["total_generated_tokens"],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one or more MoE models over MMLU/BBH/HumanEval/PopQA/INCLUDE "
            "and write per-layer expert activation statistics (with pair counts) "
            "for every (model, quant, benchmark) cell, in the same JSON schema as "
            "the sister llama-cpp-eval C++ binaries so the same post-processing "
            "toolchain can consume both."
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
        help="Which benchmarks to run (subset of {mmlu, bbh, humaneval, popqa, include}).",
    )
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Autoregressive decode cap per prompt (default 256). Set 0 for prefill-only.",
    )
    parser.add_argument(
        "--n-shots",
        type=int,
        default=5,
        help="Few-shot exemplars from the per-dataset dev pool (mmlu, bbh, include). 0 for zero-shot.",
    )
    parser.add_argument(
        "--output-dir",
        default="./expert_stats",
        help=(
            "Parent directory for the output tree. Each cell lands at "
            "<output-dir>/<model_safe>/<quant_safe>/moe-<bench>/expert_counts.json. "
            "Can be overridden via EVAL_MOE_OUTPUT_DIR for sbatch integration."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help=(
            "Number of GPUs to spread each model across (vLLM's "
            "tensor_parallel_size). Default 1. Set to NGPUS from the "
            "sbatch to use the full GPU allocation."
        ),
    )
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
                tensor_parallel_size=args.tensor_parallel_size,
            )

            # Tokenizer for per-prompt prefill counts (so the
            # per-row `n_tokens_prefill` field matches what vLLM
            # actually sees).
            tokenizer = llm.get_tokenizer()

            for benchmark_name in args.benchmarks:
                # Idempotency check: if this (model, quant, benchmark)
                # cell's JSON already exists, skip the workload. The
                # canonical output path mirrors the sbatch's
                # `--output-dir` arg. To force a re-run, delete the
                # JSON before submitting.
                _cell_out = (
                    output_dir
                    / _safe_id(model)
                    / _safe_id(quant)
                    / f"moe-{benchmark_name}"
                    / "expert_counts.json"
                )
                if _cell_out.exists():
                    logger.info(
                        "Skipping cell: model=%s, quant=%s, benchmark=%s "
                        "(output already exists at %s; delete to rerun)",
                        model,
                        quant or "<none>",
                        benchmark_name,
                        _cell_out,
                    )
                    continue

                loader = BENCHMARK_LOADERS[benchmark_name]
                benchmark = loader(
                    num_samples=args.num_samples,
                    seed=args.seed,
                    n_shot=args.n_shots,
                )
                _run_benchmark(
                    llm=llm,
                    benchmark=benchmark,
                    sampling_params=sampling_params,
                    num_layers=num_hidden_layers,
                    num_experts=num_experts,
                    top_k=num_experts_per_tok,
                    model=model,
                    quant=quant,
                    num_samples=args.num_samples,
                    n_shot=args.n_shots,
                    tokenizer=tokenizer,
                    output_dir=output_dir,
                    max_model_len=args.max_model_len,
                    max_tokens=args.max_tokens,
                )

            # Free GPU memory + tear down torch.distributed before the
            # next cell so two models don't have to fit in VRAM
            # simultaneously.
            import gc
            import torch

            del llm
            gc.collect()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                with contextlib.suppress(Exception):
                    torch.distributed.destroy_process_group()
            if torch.accelerator.is_available():
                torch.accelerator.empty_cache()

    logger.info("All %d cell(s) complete; tree at %s", n_cells, output_dir)


if __name__ == "__main__":
    # Make deterministic; harmless on Linux but reduces noise on shared hosts.
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
