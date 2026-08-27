#!/usr/bin/env python3
# type: ignore
#
# from __future__ import annotations lets us use PEP 604 unions (`dict | None`)
# and PEP 585 generics (`list[str]`) on Python 3.7+ without a syntax error.
from __future__ import annotations

"""Download the CohereLabs/include-base-44 dataset (per-language) and emit a
consolidated JSONL plus partition lists, ready to be consumed by
`examples/eval-moe/moe_expert_stats.py`.

INCLUDE is a multilingual knowledge / reasoning MCQ benchmark across 44
languages (Romanou et al., 2024, arXiv:2411.19799). The HuggingFace dataset
exposes a separate HF `config` per language - 45 real language configs (the
45th is a duplicate `Dutch-Flemish` variant whose `main` parquet has no live
rows; we use `Dutch` instead).

Schema dispatch - two variants exist in the wild:

  - **Schema A** (12 columns): the standard schema used by 44 configs
    including `Dutch`. Columns:
        language, country, domain, subject, regional_feature, level,
        question, option_a, option_b, option_c, option_d, answer

  - **Schema B** (9 columns): legacy / Flemish variant using a single
    `choices: list[str]` column instead of `option_a..d`. Not reachable
    in our 44-language subset, but we keep a defensive branch in case
    `CohereLabs` re-uploads the missing parquet.

Mirrors `llama-cpp-eval/examples/eval-moe-include/download_include.py`
(kept byte-compatible so both projects can share a single cached
JSONL on a shared scratch volume).

Outputs (default under ${DATASET_DIR}/include/, override with --outdir):
  - include.jsonl               one row per question
  - languages.txt               one language per line, sorted, unique
  - domains.txt                 one domain per line, sorted, unique
  - languages_domains.txt       one "<language>::<domain>" per line, sorted
  - metadata.json               per-language + per-(lang,dom) row counts
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Canonical 44-language list (1:1 with the paper).
INCLUDE_LANGUAGES: list[str] = sorted([
    "Albanian", "Arabic", "Armenian", "Azerbaijani", "Basque", "Belarusian",
    "Bengali", "Bulgarian", "Chinese", "Croatian", "Dutch", "Estonian",
    "Finnish", "French", "Georgian", "German", "Greek", "Hebrew", "Hindi",
    "Hungarian", "Indonesian", "Italian", "Japanese", "Kazakh", "Korean",
    "Lithuanian", "Malay", "Malayalam", "Nepali", "North Macedonian",
    "Persian", "Polish", "Portuguese", "Russian", "Serbian", "Spanish",
    "Tagalog", "Tamil", "Telugu", "Turkish", "Ukrainian", "Urdu", "Uzbek",
    "Vietnamese",
])
assert len(INCLUDE_LANGUAGES) == 44, f"expected 44 languages, got {len(INCLUDE_LANGUAGES)}"


def _letter(i: int) -> str:
    return chr(ord("A") + i)


def _langdom(language: str, domain: str) -> str:
    """Canonical "<language>::<domain>" key. Whitespace-stable."""
    return f"{language}::{domain}"


def _normalise_row(row: dict, split: str, language: str) -> dict | None:
    """Apply schema dispatch + build a normalised JSONL dict."""
    q = str(row.get("question") or "").strip()
    if not q:
        return None

    if "option_a" in row:
        opts = [
            str(row.get("option_a") or ""),
            str(row.get("option_b") or ""),
            str(row.get("option_c") or ""),
            str(row.get("option_d") or ""),
        ]
    elif "choices" in row:
        choices = list(row.get("choices") or [])
        if len(choices) != 4:
            return None
        opts = [str(c) for c in choices]
    else:
        return None

    if any(not o.strip() for o in opts):
        return None

    try:
        answer = int(row.get("answer"))
    except (TypeError, ValueError):
        return None
    if answer < 0 or answer > 3:
        return None

    def _opt(key: str) -> str:
        v = row.get(key)
        return "" if v is None else str(v)

    domain = _opt("domain") or "Unknown"
    return {
        "split":            split,
        "language":         language,
        "country":          _opt("country"),
        "domain":           domain,
        "subject":          _opt("subject"),
        "regional_feature": _opt("regional_feature"),
        "level":            _opt("level"),
        "question":         q,
        "options":          [f"{_letter(i)}. {opts[i].strip()}" for i in range(4)],
        "answer":           answer,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--outdir",
        default=os.environ.get(
            "DATASET_DIR", "/data/scratch/projects/uom00014/vllm/datasets"
        )
        + "/include",
        help="output directory for include.jsonl + partition lists.",
    )
    p.add_argument("--languages", nargs="*", default=None,
                   help="restrict to these languages only (default: all 44).")
    p.add_argument("--domains", nargs="*", default=None,
                   help="restrict to these domains only (default: all).")
    p.add_argument("--limit-per-langdom", type=int, default=0,
                   help="keep at most N rows per (language, domain) group (0 = no limit).")
    p.add_argument("--include-validation", action="store_true",
                   help="also download and emit the validation split.")
    p.add_argument("--seed", type=int, default=42, help="subsample seed (default 42).")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    languages = args.languages if args.languages else INCLUDE_LANGUAGES
    print(f"[include] target: {len(languages)} language(s) -> {outdir.resolve()}", flush=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("error: the 'datasets' package is required.", file=sys.stderr)
        return 1

    splits = ["test"] + (["validation"] if args.include_validation else [])
    out_jsonl = outdir / "include.jsonl"

    rows_by_langdom: dict[str, list[dict]] = {}
    counts_by_langdom: dict[str, int] = {}
    counts_by_language: dict[str, int] = {}
    counts_by_domain: dict[str, int] = {}
    counts_by_split: dict[str, int] = {}
    n_kept = n_skipped = 0
    seen_languages: set[str] = set()
    seen_domains: set[str] = set()
    seen_langdoms: set[str] = set()

    for language in languages:
        n_lang = 0
        for split in splits:
            print(f"[include] loading {language}/{split} ...", flush=True)
            try:
                ds = load_dataset("CohereLabs/include-base-44", language, split=split)
            except Exception as e:
                print(f"[include]   {language}/{split} not available: {e}", file=sys.stderr, flush=True)
                continue

            print(f"[include]   {language}/{split}: {len(ds)} raw rows", flush=True)

            for row in ds:
                normalised = _normalise_row(row, split, language)
                if normalised is None:
                    n_skipped += 1
                    continue
                domain = normalised["domain"]
                if args.domains and domain not in args.domains:
                    continue
                key = _langdom(language, domain)
                rows_by_langdom.setdefault(key, []).append(normalised)
                n_lang += 1

        if n_lang == 0:
            print(f"[include]   {language}: 0 rows kept across splits", flush=True)
            continue

        counts_by_language[language] = counts_by_language.get(language, 0) + n_lang
        seen_languages.add(language)
        print(f"[include]   {language}: {n_lang} rows kept", flush=True)

    keep_ids: set[int] = set()
    for lst in rows_by_langdom.values():
        if args.limit_per_langdom > 0 and len(lst) > args.limit_per_langdom:
            lst.sort(key=lambda r: r["question"])
            for i in range(args.limit_per_langdom):
                keep_ids.add(id(lst[i]))
        else:
            for r in lst:
                keep_ids.add(id(r))

    with out_jsonl.open("w") as f:
        for lst in rows_by_langdom.values():
            for obj in lst:
                if id(obj) not in keep_ids:
                    continue
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

                key = _langdom(obj["language"], obj["domain"])
                counts_by_langdom[key] = counts_by_langdom.get(key, 0) + 1
                counts_by_domain[obj["domain"]] = counts_by_domain.get(obj["domain"], 0) + 1
                counts_by_split[obj["split"]] = counts_by_split.get(obj["split"], 0) + 1
                seen_domains.add(obj["domain"])
                seen_langdoms.add(key)
                n_kept += 1

    (outdir / "languages.txt").write_text("\n".join(sorted(seen_languages)) + "\n")
    (outdir / "domains.txt").write_text("\n".join(sorted(seen_domains)) + "\n")
    langdom_keys = sorted(seen_langdoms)
    (outdir / "languages_domains.txt").write_text("\n".join(langdom_keys) + "\n")

    metadata = {
        "dataset":       "CohereLabs/include-base-44",
        "paper":         "Romanou et al. 2024 (arXiv:2411.19799)",
        "license":       "apache-2.0",
        "rows_total":    n_kept,
        "rows_skipped":  n_skipped,
        "splits":        counts_by_split,
        "n_languages":   len(seen_languages),
        "n_domains":     len(seen_domains),
        "n_langdoms":    len(seen_langdoms),
        "counts_by_language": {k: counts_by_language[k] for k in sorted(counts_by_language)},
        "counts_by_domain":   {k: counts_by_domain[k]   for k in sorted(counts_by_domain)},
        "counts_by_langdom":  {k: counts_by_langdom[k]  for k in sorted(counts_by_langdom)},
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"[include] wrote {n_kept} rows to {out_jsonl} (skipped {n_skipped})")
    print(f"[include] wrote {len(seen_languages)} languages to {outdir / 'languages.txt'}")
    print(f"[include] wrote {len(seen_domains)} domains to {outdir / 'domains.txt'}")
    print(f"[include] wrote {len(seen_langdoms)} (language, domain) keys to {outdir / 'languages_domains.txt'}")
    print(f"[include] wrote metadata to {outdir / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
