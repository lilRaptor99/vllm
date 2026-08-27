#!/usr/bin/env python3
# type: ignore
"""Download the akariasai/PopQA dataset and emit a consolidated JSONL
plus a props list, ready to be consumed by
`examples/eval-moe/moe_expert_stats.py`.

PopQA ships with a single `test` split (no dev/val). Each row carries:

    id, subj, prop, obj, subj_id, prop_id, obj_id, s_aliases, o_aliases,
    s_uri, o_uri, s_wiki_title, o_wiki_title, s_pop, o_pop, question,
    possible_answers

We keep the analytical fields (id, subj, prop, obj, s_pop, o_pop, question,
possible_answers) and drop the rest.

Mirrors `llama-cpp-eval/examples/eval-moe-popqa/download_popqa.py`
(kept byte-compatible so both projects can share a single cached
JSONL on a shared scratch volume).

Outputs (default under ${DATASET_DIR}/popqa/, override with --outdir):
  - popqa.jsonl     one row per question
  - props.txt       one prop (relation type) per line, sorted, unique
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--outdir",
        default=os.environ.get(
            "DATASET_DIR", "/data/scratch/projects/uom00014/vllm/datasets"
        )
        + "/popqa",
        help="output directory for popqa.jsonl and props.txt",
    )
    p.add_argument(
        "--props",
        nargs="*",
        default=None,
        help="restrict to these relation types only (default: all).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="subsample evenly across props so each keeps at most N rows (0 = no limit).",
    )
    p.add_argument("--seed", type=int, default=42, help="subsample seed (default 42).")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "error: the 'datasets' package is required.\n"
            "  pip install --user datasets",
            file=sys.stderr,
        )
        return 1

    print(f"[popqa] loading akariasai/PopQA into {outdir}", flush=True)
    try:
        ds = load_dataset("akariasai/PopQA", split="test")
    except Exception as e:
        print(f"[popqa] failed to load dataset: {e}", file=sys.stderr)
        return 1

    print(f"[popqa]   total rows: {len(ds)}", flush=True)

    # --props filter
    allow = set(args.props) if args.props else None
    if allow:
        ds = ds.filter(lambda r: r["prop"] in allow)
        print(f"[popqa]   after --props filter: {len(ds)} rows", flush=True)

    # Always sort by (prop, id) for determinism + even subsampling.
    ds = ds.sort(("prop", "id"))

    # --limit evenly per prop
    if args.limit > 0:
        from collections import defaultdict

        by_prop: dict[str, list[int]] = defaultdict(list)
        for i, prop in enumerate(ds["prop"]):
            by_prop[prop].append(i)
        keep = set()
        for prop, idxs in by_prop.items():
            idxs = idxs[: args.limit]
            keep.update(idxs)
        ds = ds.filter(lambda r, idx: idx in keep, with_indices=True)
        print(f"[popqa]   after --limit {args.limit}/prop: {len(ds)} rows", flush=True)

    out_jsonl = outdir / "popqa.jsonl"
    seen_props: set[str] = set()
    n_kept = 0
    n_skipped = 0

    with out_jsonl.open("w") as f:
        for row in ds:
            possible = list(row.get("possible_answers") or [])
            if not possible or not str(row.get("question") or "").strip():
                n_skipped += 1
                continue

            obj = {
                "id":                int(row["id"]),
                "subj":              str(row["subj"]),
                "prop":              str(row["prop"]),
                "obj":               str(row["obj"]),
                "s_pop":             int(row.get("s_pop") or 0),
                "o_pop":             int(row.get("o_pop") or 0),
                "question":          str(row["question"]),
                "possible_answers":  [str(a) for a in possible],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            seen_props.add(obj["prop"])
            n_kept += 1

    props_sorted = sorted(seen_props)
    (outdir / "props.txt").write_text("\n".join(props_sorted) + "\n")

    print(f"[popqa] wrote {n_kept} rows to {out_jsonl} (skipped {n_skipped})")
    print(f"[popqa] wrote {len(props_sorted)} unique props to {outdir / 'props.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
