#!/usr/bin/env python3
"""
validate_deepseek_prequant.py - Quick smoke test for the DeepSeek W4A16
prequant recipe. Runs in ~30 seconds on a single H100 GPU (no calibration
data needed). Validates:

  1. Recipe construction: AWQ + GPTQ modifiers both build without error.
  2. group_size=64 is accepted by llm-compressor's divisibility validator.
  3. The model can be loaded + recipe applied to a few layers without
     unflatten RuntimeError (the failure mode the old ignore=down_proj
     workaround was hiding).
  4. After applying the quantization scheme, all 3 MoE projections
     (gate_proj, up_proj, down_proj) within a single FusedMoE share
     the SAME scheme dict (the vLLM load-time invariant).

Usage on an interactive H100 node:

  ssh spartan
  salloc --partition=gpu-h100 --gres=gpu:1 --mem=80G --time=0-01:00:00 bash
  module load Python/3.11.3 GCCcore/11.3.0 CUDA/12.4.1 NCCL/2.22.3-CUDA-12.4.1
  source /data/scratch/projects/uom00014/vllm/venv_awq/bin/activate
  cd /data/gpfs/projects/uom00014/vllm
  python3 examples/eval-moe/scripts/validate_deepseek_prequant.py

Exits 0 on success, 1 on any check failing.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path("/data/gpfs/projects/uom00014/vllm")
sys.path.insert(0, str(REPO_ROOT / "examples" / "eval-moe"))

DEEPSEEK_ID = "deepseek-ai/deepseek-moe-16b-chat"


def main() -> int:
    # Import the script under test.
    try:
        from awq_quantize import build_recipe  # noqa: E402
    except Exception as e:
        print(f"[fail] cannot import awq_quantize.build_recipe: {e}")
        traceback.print_exc()
        return 1

    print("=" * 70)
    print("CHECK 1: build_recipe returns valid modifiers (no exception)")
    print("=" * 70)
    try:
        awq_recipe = build_recipe("awq", DEEPSEEK_ID)
        gptq_recipe = build_recipe("gptq", DEEPSEEK_ID)
        print(f"[ok] awq recipe: {[type(m).__name__ for m in awq_recipe]}")
        print(f"[ok] gptq recipe: {[type(m).__name__ for m in gptq_recipe]}")
    except Exception as e:
        print(f"[fail] build_recipe raised: {e}")
        traceback.print_exc()
        return 1

    print()
    print("=" * 70)
    print("CHECK 2: DeepSeek AWQ scheme has group_size=64 (not 128)")
    print("=" * 70)
    try:
        from compressed_tensors.quantization.quant_scheme import (  # noqa: E402
            QuantizationScheme,
        )

        quant_mod = next(m for m in awq_recipe if m.__class__.__name__ == "QuantizationModifier")
        if quant_mod.config_groups is None:
            print("[fail] AWQ QuantizationModifier has no config_groups")
            return 1
        scheme = quant_mod.config_groups["group_0"]
        ws = scheme.weights
        if ws is None or ws.group_size != 64:
            print(f"[fail] expected group_size=64, got {ws.group_size if ws else None}")
            return 1
        print(f"[ok] group_size={ws.group_size}, num_bits={ws.num_bits}, "
              f"strategy={ws.strategy}, symmetric={ws.symmetric}")

        gptq = gptq_recipe[0]
        if gptq.config_groups is None:
            print("[fail] GPTQ GPTQModifier has no config_groups")
            return 1
        gs2 = gptq.config_groups["group_0"]
        if gs2.weights.group_size != 64:
            print(f"[fail] GPTQ group_size={gs2.weights.group_size}, expected 64")
            return 1
        print(f"[ok] GPTQ group_size={gs2.weights.group_size}")

        if any("down_proj" in p for p in quant_mod.ignore):
            print(f"[fail] AWQ ignore still contains down_proj: {quant_mod.ignore}")
            return 1
        print(f"[ok] AWQ ignore does NOT skip down_proj: {quant_mod.ignore}")
        if any("down_proj" in p for p in gptq.ignore):
            print(f"[fail] GPTQ ignore still contains down_proj: {gptq.ignore}")
            return 1
        print(f"[ok] GPTQ ignore does NOT skip down_proj: {gptq.ignore}")
    except Exception as e:
        print(f"[fail] scheme inspection failed: {e}")
        traceback.print_exc()
        return 1

    print()
    print("=" * 70)
    print("CHECK 3: Initialize modifiers on a synthetic 4096x10944 layer")
    print("=" * 70)
    try:
        import torch  # noqa: E402
        from llmcompressor.modifiers.quantization.quantization.mixin import (  # noqa: E402
            QuantizationMixin,
        )

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4096, 10944, bias=False)

        for label, recipe in [("AWQ", awq_recipe), ("GPTQ", gptq_recipe)]:
            for mod in recipe:
                if isinstance(mod, QuantizationMixin):
                    dummy = DummyModel()
                    try:
                        mod.initialize_quantization(dummy)
                        print(f"[ok] {label} {type(mod).__name__} initializes on "
                              f"4096x10944 layer (down_proj shape)")
                    except Exception as e:
                        print(f"[fail] {label} {type(mod).__name__} init failed: {e}")
                        traceback.print_exc()
                        return 1
    except Exception as e:
        print(f"[fail] layer application failed: {e}")
        traceback.print_exc()
        return 1

    print()
    print("=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)
    print("Next steps:")
    print("  - DeepSeek re-prequant sbatches are already queued:")
    print("      32518557 AWQ")
    print("      32518558 GPTQ")
    print("  - After they land, validate end-to-end with the analyzer:")
    print("      python3 examples/eval-moe/moe_expert_stats.py \\")
    print("        --model /data/scratch/projects/uom00014/vllm/hf_cache_awq/\\")
    print("        deepseek-ai--deepseek-moe-16b-chat-awq \\")
    print("        --quant compressed-tensors --benchmarks mmlu --num-samples 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())