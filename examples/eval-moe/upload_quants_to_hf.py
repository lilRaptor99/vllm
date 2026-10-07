#!/usr/bin/env python3
"""
upload_quants_to_hf.py - Upload the W4A16 prequant checkpoints from
/data/scratch/projects/uom00014/vllm/hf_cache_awq/ to a HuggingFace namespace.

For each <model_safe>-<recipe>/ directory under hf_cache_awq/, create a
matching HF repo (e.g. pratheeks/OLMoE-1B-7B-0924-Instruct-w4a16-awq) and
upload:
  - model.safetensors (the big one)
  - the small JSON/tokenizer files
  - a generated README.md model card
  - a generated quantize_config.json (or copy from disk if it exists)

Repos are created PRIVATE (research artifacts, not for public distribution
unless the user promotes them later). Pass --public to create them public.

Usage:
  upload_quants_to_hf.py [--namespace USER] [--public] [--only MIXTRAL-AWQ]
                          [--skip MIXTRAL-AWQ_GPTQ] [--scratch-base PATH]

Env:
  HF_TOKEN  required, must have write scope on the namespace
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, whoami


# Mapping from local dir suffix -> (display name, base HF model id,
# recipe human name, recipe description)
RECIPE_INFO = {
    "awq":      ("AWQ-only",        "AWQ scale + round-to-nearest int4"),
    "gptq":     ("GPTQ-only",       "Hessian-based int4 (GPTQ)"),
    "awq_gptq": ("AWQ + GPTQ combined",
                 "AWQ scale first, then GPTQ Hessian int4"),
}


# Pretty model name from the safetensors dir name
MODEL_DISPLAY = {
    "allenai--OLMoE-1B-7B-0924-Instruct":    "OLMoE-1B-7B-0924-Instruct",
    "deepseek-ai--deepseek-moe-16b-chat":    "DeepSeek-MoE-16B-chat",
    "Qwen--Qwen3-30B-A3B":                  "Qwen3-30B-A3B",
    "mistralai--Mixtral-8x22B-Instruct-v0.1": "Mixtral-8x22B-Instruct-v0.1",
}


def model_card(model_safe: str, recipe: str, config: dict) -> str:
    """Build a short model card. Mentions base model, recipe, load example,
    calibration data, and known caveats."""
    display = MODEL_DISPLAY.get(model_safe, model_safe.replace("--", "/"))
    base_id = display  # the upstream HF id is the same as the display name
    recipe_name, recipe_desc = RECIPE_INFO[recipe]
    # compressed-tensors v0.19+ nests the per-layer config under
    # config_groups.group_0.{weights,input_activations,output_activations}
    qc = (config.get("quantization_config") or {})
    fmt = qc.get("format", "?")
    cg0 = (qc.get("config_groups") or {}).get("group_0") or {}
    w = cg0.get("weights") or {}
    nb = w.get("num_bits", qc.get("num_bits", "?"))
    gs = w.get("group_size", qc.get("group_size", "?"))
    sym = w.get("symmetric", qc.get("symmetric", True))
    quant_method = qc.get("quant_method", "compressed-tensors")
    # Special note for DeepSeek (down_proj stays bf16)
    deepseek_note = ""
    if "deepseek" in model_safe.lower():
        deepseek_note = (
            "\n**Note**: DeepSeek-MoE-16B's per-expert `mlp.down_proj` weight "
            "matrix has 10944 columns, which is not divisible by `group_size=128`. "
            "That projection is left in **bf16**; all other linear layers "
            "(gate, up, down for shared experts, per-expert gate/up) are "
            "quantized to int4. Load with `--quantization compressed-tensors`."
        )

    return f"""---
license: other
tags:
- vllm
- moe
- int4
- w4a16
- {recipe}
- {quant_method}
base_model: {base_id}
---

# {display} — W4A16 {recipe_name}

Pre-quantized {recipe_desc} version of [{base_id}](https://huggingface.co/{base_id}),
produced with `llm-compressor` (v0.14.x) using the
[`mlabonne/open-perfectblend`](https://huggingface.co/datasets/mlabonne/open-perfectblend)
calibration dataset (512 samples, seq_len 2048).

## Recipe summary

- **scheme**: W4A16 (int4 weights, bf16 activations)
- **format**: `{fmt}` (compressed-tensors)
- **num_bits**: {nb}, **group_size**: {gs}, **symmetric**: {sym}
- **AWQ**: `duo_scaling="both"`, `n_grid=10`
- **GPTQ**: `dampening_frac=0.01` (0.02 for DeepSeek)
- **Ignore**: `lm_head`, `.*mlp.gate$`, `.*mlp.shared_expert_gate$`
  (plus `.*\\.down_proj$` for DeepSeek — see note below)
{deepseek_note}

## How to load

```python
from vllm import LLM, SamplingParams
llm = LLM(
    model="REPLACE_WITH_REPO_ID",
    quantization="compressed-tensors",
    trust_remote_code=True,  # DeepSeek only
)
out = llm.generate(["Hello, my name is"], SamplingParams(max_tokens=32))
```

Or via the vLLM CLI:

```bash
vllm serve REPLACE_WITH_REPO_ID --quantization compressed-tensors \\
    --trust-remote-code   # DeepSeek only
```

## Provenance

Pre-quantized as part of the vLLM eval-moe project on Spartan HPC
(`/data/gpfs/projects/uom00014/vllm`, branch `eval-moe`). See
[`examples/eval-moe/awq_quantize.py`](https://github.com/vllm-project/vllm/blob/eval-moe/examples/eval-moe/awq_quantize.py)
for the exact recipe and run scripts in
`examples/eval-moe/spartan/vllm-moe-awq-prequant.sbatch`.

## Known caveats

- **DeepSeek AWQ**: a small fraction of `_apply_smoothing` calls produce
  NaN/inf activations on the calibration batch and are silently skipped
  (matches the behavior of the upstream llm-compressor examples).
- **DeepSeek GPTQ**: ~24% of per-expert modules fall back to
  round-to-nearest (RTN) instead of GPTQ because the Hessian is
  ill-conditioned for some DeepSeek weight distributions. The
  quantization scheme is unchanged; quality is slightly lower than
  pure-GPTQ on those modules.
- The awq_gptq variant is only produced for Mixtral in this snapshot.
"""


def model_card_path_quant_config(model_safe: str, recipe: str, src_dir: Path) -> dict:
    """Build a small quantize_config.json to upload alongside the model.
    Most of the config is embedded in config.json under quantization_config,
    but a few loaders look for a top-level quantize_config.json. We just
    copy whatever is in src_dir if present, else synthesize from config.json."""
    qc = src_dir / "quantize_config.json"
    if qc.exists():
        return json.loads(qc.read_text())
    cfg = json.loads((src_dir / "config.json").read_text())
    qc_inner = (cfg.get("quantization_config") or {})
    return qc_inner


def upload_one(api: HfApi, namespace: str, model_safe: str, recipe: str,
               src_dir: Path, public: bool, log_path: Path) -> bool:
    """Create the repo, write README + quantize_config, upload files.
    Returns True on success, False on error (logged)."""
    display = MODEL_DISPLAY.get(model_safe, model_safe.replace("--", "/"))
    repo_id = f"{namespace}/{display}-w4a16-{recipe.replace('_', '-')}"
    safe_repo = repo_id.replace("/", "__")
    log_line = lambda msg: (log_path.open("a").write(f"[{safe_repo}] {msg}\n") if log_path else None)

    log_line(f"=== uploading {repo_id} from {src_dir} ===")
    try:
        # 1) create the repo (idempotent)
        url = api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=not public,
            exist_ok=True,
        )
        log_line(f"repo ready: {url}")

        # 2) read source config to render the model card
        cfg = json.loads((src_dir / "config.json").read_text())
        readme = model_card(model_safe, recipe, cfg)
        # patch the placeholder REPO_ID into the actual repo_id
        readme = readme.replace("REPLACE_WITH_REPO_ID", repo_id)
        (src_dir / "README.md").write_text(readme)

        # 3) write a quantize_config.json next to the model if not present
        qc = model_card_path_quant_config(model_safe, recipe, src_dir)
        (src_dir / "quantize_config.json").write_text(json.dumps(qc, indent=2))

        # 4) upload small files first (so the repo is browseable immediately)
        existing = set(api.list_repo_files(repo_id, repo_type="model"))
        small_files = [
            "config.json",
            "configuration_deepseek.py",  # DeepSeek only — noop for others
            "modeling_deepseek.py",       # DeepSeek only
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "recipe.yaml",
            "quantize_config.json",
            "README.md",
        ]
        for fname in small_files:
            p = src_dir / fname
            if not p.exists():
                continue
            if fname in existing:
                log_line(f"  skip {fname} (already in repo)")
                continue
            log_line(f"  uploading {fname} ({p.stat().st_size} B)")
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=fname,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"add {fname}",
            )

        # 5) upload the big model.safetensors (or model-*.safetensors shards)
        shards = sorted(src_dir.glob("model*.safetensors"))
        if not shards:
            log_line(f"  ERROR: no model*.safetensors found in {src_dir}")
            return False
        for shard in shards:
            if shard.name in existing:
                log_line(f"  skip {shard.name} (already in repo)")
                continue
            size_gb = shard.stat().st_size / 1e9
            log_line(f"  uploading {shard.name} ({size_gb:.1f} GB) ...")
            t0 = time.time()
            api.upload_file(
                path_or_fileobj=str(shard),
                path_in_repo=shard.name,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"add {shard.name}",
            )
            dt = time.time() - t0
            speed = size_gb / dt if dt > 0 else 0
            log_line(f"  uploaded {shard.name} in {dt:.0f}s ({speed:.1f} GB/s)")

        # 6) upload the safetensors index if present
        idx = src_dir / "model.safetensors.index.json"
        if idx.exists() and idx.name not in existing:
            log_line(f"  uploading {idx.name}")
            api.upload_file(
                path_or_fileobj=str(idx),
                path_in_repo=idx.name,
                repo_id=repo_id,
                repo_type="model",
                commit_message="add model.safetensors.index.json",
            )

        log_line(f"=== DONE {repo_id} ===")
        return True
    except Exception as e:
        log_line(f"  ERROR: {type(e).__name__}: {e}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--namespace", default=None,
                   help="HF namespace to upload under (default: HF_USERNAME env or 'pratheeks')")
    p.add_argument("--public", action="store_true",
                   help="create repos public (default: private)")
    p.add_argument("--scratch-base",
                   default="/data/scratch/projects/uom00014/vllm/hf_cache_awq",
                   help="local dir holding the per-recipe prequant ckpts")
    p.add_argument("--only", nargs="*", default=None,
                   help="only upload these <model_safe>-<recipe> (e.g. allenai--OLMoE-1B-7B-0924-Instruct-awq)")
    p.add_argument("--skip", nargs="*", default=None,
                   help="skip these <model_safe>-<recipe>")
    p.add_argument("--log", default="/data/gpfs/projects/uom00014/vllm/examples/eval-moe/spartan/logs/upload-hf.log",
                   help="log file path")
    args = p.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN env var is required", file=sys.stderr)
        sys.exit(2)

    namespace = args.namespace or os.environ.get("HF_NAMESPACE") or os.environ.get("HF_USER") or "pratheeks"
    # Verify the token is good and matches the namespace
    me = whoami(token=os.environ["HF_TOKEN"])
    if me.get("name") != namespace:
        print(f"HF_TOKEN belongs to '{me.get('name')}', not '{namespace}'. "
              f"Pass --namespace {me.get('name')} to use it.", file=sys.stderr)
        sys.exit(2)

    api = HfApi(token=os.environ["HF_TOKEN"])
    print(f"uploading as namespace={namespace}, public={args.public}, scratch_base={args.scratch_base}")
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)

    # Build the list of (model_safe, recipe, src_dir) tuples
    jobs = []
    for entry in sorted(Path(args.scratch_base).iterdir()):
        if not entry.is_dir() or not (entry / "config.json").exists():
            continue
        name = entry.name
        if "-" in name:
            # split off the recipe suffix; convention is <model_safe>-<recipe>
            for recipe in ("awq_gptq", "awq", "gptq"):  # longer first to avoid false split
                suffix = f"-{recipe}"
                if name.endswith(suffix):
                    model_safe = name[: -len(suffix)]
                    jobs.append((model_safe, recipe, entry))
                    break

    if args.only:
        only_set = set(args.only)
        jobs = [j for j in jobs if f"{j[0]}-{j[1]}" in only_set]
    if args.skip:
        skip_set = set(args.skip)
        jobs = [j for j in jobs if f"{j[0]}-{j[1]}" not in skip_set]

    print(f"will upload {len(jobs)} repos:")
    for ms, recipe, d in jobs:
        size = sum(p.stat().st_size for p in d.glob("model*.safetensors")) / 1e9
        print(f"  {ms:50s} {recipe:10s} {size:6.1f} GB")

    total = 0
    ok = 0
    log_path = Path(args.log)
    for ms, recipe, d in jobs:
        total += 1
        if upload_one(api, namespace, ms, recipe, d, args.public, log_path):
            ok += 1
    print(f"\n{ok}/{total} uploads succeeded. log: {log_path}")


if __name__ == "__main__":
    main()
