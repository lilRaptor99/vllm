#!/usr/bin/env python3
"""
awq_quantize.py - Pre-quantize a bf16 MoE model to int4 (W4A16) using one of
three llm-compressor recipes: AWQ-only, GPTQ-only, or AWQ+GPTQ combined.

Output goes to a new local dir that the analyzer can load via
`--quantization moe_wna16`. The output checkpoint carries a
`quantize_config.json` declaring `quant_method="awq"` (both AWQ and AWQ+GPTQ
set this; GPTQ-only sets `"gptq"`), so vLLM's `moe_wna16` quant method picks
up the right kernel at load time.

Usage:
  python3 awq_quantize.py <model_id> <output_dir> --recipe awq
  python3 awq_quantize.py <model_id> <output_dir> --recipe gptq
  python3 awq_quantize.py <model_id> <output_dir> --recipe awq_gptq

Environment (set by sbatch wrapper):
  HF_HUB_OFFLINE  (model); the calibration dataset is pre-staged
                    under ${SCRATCH}/datasets/{hf_cache,perfectblend}
                    by the pre-download-perfectblend.sh script.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------------
# transformers v5.x compatibility shim
# ------------------------------------------------------------------
# Some HF Hub modeling files (e.g. modeling_deepseek.py on
# deepseek-ai/deepseek-moe-16b-chat) were authored against a newer
# transformers than the 5.15.1 in this venv. They import
# `is_torch_fx_available` and `is_torch_greater_or_equal_than_1_13`
# from `transformers.utils.import_utils`, but those names were renamed
# or removed in 5.x. Stub them with safe no-ops so the import
# succeeds; the only use site of `is_torch_fx_available` in
# modeling_deepseek.py is a `if is_torch_fx_available(): torch.fx.wrap(...)`
# block that we explicitly want to skip (we don't run FX tracing).
import transformers.utils.import_utils as _iu
if not hasattr(_iu, "is_torch_fx_available"):
    _iu.is_torch_fx_available = lambda: False  # type: ignore[attr-defined]
if not hasattr(_iu, "is_torch_greater_or_equal_than_1_13"):
    _iu.is_torch_greater_or_equal_than_1_13 = lambda *a, **k: True  # type: ignore[attr-defined]

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.utils import load_context


RECIPES = ("awq", "gptq", "awq_gptq")


def build_recipe(recipe: str) -> list:
    """Return the recipe list for the requested int4 quantization strategy.

    All three target W4A16 with group_size 128 and skip lm_head + MoE
    router gates (integer-arithmetic sensitive to quantization noise).
    `moe_calibrate_all_experts=True` (passed to oneshot()) ensures AWQ+GPTQ
    stats are computed per-expert, not per-layer.

    Note: AWQModifier is a *transform* (it scales weights) and doesn't
    quantize. It needs to be paired with a QuantizationMixin. We use
    QuantizationModifier for the awq-only path (round-to-nearest int4
    after AWQ scaling) and GPTQModifier for the other two (Hessian-based
    int4 quantization).
    """
    ignore = [
        "lm_head",
        "re:.*mlp\\.gate$",
        "re:.*mlp\\.shared_expert_gate$",
    ]
    if recipe == "awq":
        return [
            AWQModifier(duo_scaling="both", n_grid=10),
            QuantizationModifier(targets="Linear", scheme="W4A16", ignore=ignore),
        ]
    if recipe == "gptq":
        return [
            GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignore),
        ]
    if recipe == "awq_gptq":
        return [
            AWQModifier(duo_scaling="both", n_grid=10),
            GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignore),
        ]
    raise ValueError(f"unknown recipe: {recipe!r}")


def build_calibration(args, tokenizer):
    """Build the calibration dataset for oneshot().

    Two paths:

    1. **HF_DATASETS_CACHE path** (preferred): `build-hf-dataset-cache.sh`
       builds a proper HF datasets cache at `${SCRATCH}/datasets/hf_cache/`
       that `load_dataset("mlabonne/open-perfectblend", cache_dir=...)` can
       resolve offline. We pass `dataset="perfectblend"` to oneshot and let
       the registry handle the rest.

    2. **Parquet-direct path** (fallback): we load the parquet shards via
       `datasets.load_dataset("parquet", data_files=...)` and pass the
       resulting `Dataset` to oneshot as a non-registered Dataset. To get
       proper chat-template + tokenization we monkey-patch
       PerfectBlendDataset.__init__ to skip the hardcoded Hub name, then run
       through the regular pipeline.
    """
    dataset_root = Path(args.dataset)
    hf_cache = dataset_root.parent / "hf_cache"
    if hf_cache.exists():
        print(f"[awq-quant] using HF_DATASETS_CACHE={hf_cache}")
        # The natural registry path: pass the registered name "perfectblend".
        return ("perfectblend", f"train[:{args.num_samples}]", str(hf_cache))

    # Parquet-direct fallback (no HF cache available).
    print(f"[awq-quant][warn] no HF cache at {hf_cache}; using parquet-direct")
    parquet_files = sorted(str(p) for p in dataset_root.glob("**/*.parquet"))
    if not parquet_files:
        print(f"[awq-quant][error] no parquet files found under {args.dataset}",
              file=sys.stderr)
        sys.exit(1)
    print(f"[awq-quant] found {len(parquet_files)} parquet shards")

    import datasets as hf_datasets
    calibration_ds = hf_datasets.load_dataset(
        "parquet", data_files={"train": parquet_files}, split="train",
    )
    if args.num_samples and len(calibration_ds) > args.num_samples:
        calibration_ds = calibration_ds.select(range(args.num_samples))

    # Wrap in PerfectBlendDataset for chat-template + tokenization pipeline.
    from llmcompressor.transformers.data.perfectblend import PerfectBlendDataset
    from llmcompressor.args import DatasetArguments
    import llmcompressor.transformers.data.data_helpers as _dh

    # Monkey-patch PerfectBlendDataset.__init__ to skip the hardcoded Hub
    # name; we already have a local parquet Dataset via raw_kwargs.
    original_init = PerfectBlendDataset.__init__

    def patched_init(self, dataset_args, split, processor):
        # Skip the `dataset_args.dataset = "mlabonne/open-perfectblend"` line.
        super(PerfectBlendDataset, self).__init__(
            dataset_args=dataset_args, split=split, processor=processor
        )
        dataset_args.text_column = "conversations"
        if (
            self.tokenizer is not None
            and getattr(self.tokenizer, "chat_template", None) is None
        ):
            self.tokenizer.chat_template = self.DEFAULT_CHAT_TEMPLATE
            import loguru
            loguru.logger.warning(
                "tokenizer.chat_template is not set, using default for "
                f"{self.__class__.__name__}"
            )

    PerfectBlendDataset.__init__ = patched_init
    try:
        dataset_args_ = DatasetArguments(
            dataset="parquet",
            dataset_path=str(args.dataset),
            splits=f"train[:{args.num_samples}]",
            max_seq_length=args.max_seq_length,
            num_calibration_samples=args.num_samples,
            concatenate_data=False,
            dataloader_num_workers=0,
            preprocessing_num_workers=None,
            overwrite_cache=False,
            streaming=False,
        )
        # Override the file-extension probe for parquet shards.
        original_get_custom = _dh.get_custom_datasets_from_path
        _dh.get_custom_datasets_from_path = (
            lambda path, ext="json": original_get_custom(path, ext="parquet")
        )
        return PerfectBlendDataset(
            dataset_args=dataset_args_,
            split=dataset_args_.splits,
            processor=tokenizer,
        )
    finally:
        PerfectBlendDataset.__init__ = original_init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", help="HF model id or local snapshot path")
    parser.add_argument("output_dir", help="Where to save the int4 ckpt")
    parser.add_argument(
        "--recipe",
        choices=RECIPES,
        default="awq_gptq",
        help=(
            "Which int4 recipe to run: 'awq' (AWQ scales only), "
            "'gptq' (GPTQ Hessian only), or 'awq_gptq' (AWQ then GPTQ). "
            "Default 'awq_gptq' for best accuracy."
        ),
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=2048,
        help="Calibration sequence length (default 2048)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=512,
        help="Number of calibration samples (default 512)",
    )
    parser.add_argument(
        "--dataset", default="perfectblend",
        help=(
            "Local directory containing calibration parquet shards (the "
            "pre-download-perfectblend.sh helper stages them at "
            "${SCRATCH}/datasets/perfectblend). The script globs "
            "**/*.parquet under this path. If unset, falls back to the "
            "string 'perfectblend' which requires Hub access."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        print(f"[awq-quant] removing existing output_dir {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[awq-quant] loading model {args.model_id!r} (recipe={args.recipe}) ...")
    with load_context():
        # trust_remote_code=True is REQUIRED for DeepSeek-MoE (custom
        # modeling code on the Hub) and harmless for the other models.
        # Without it, transformers v5.x prompts interactively via input()
        # to ask "Do you wish to run the custom code? [y/N]" and SLURM
        # has no TTY -> EOFError on every DeepSeek run.
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True,
    )

    recipe = build_recipe(args.recipe)
    recipe_label = {
        "awq": "AWQ-only",
        "gptq": "GPTQ-only",
        "awq_gptq": "AWQ+GPTQ combined",
    }[args.recipe]

    calibration = build_calibration(args, tokenizer)
    if isinstance(calibration, tuple):
        if len(calibration) == 3:
            dataset_arg, splits_arg, cache_dir_arg = calibration
        else:
            dataset_arg, splits_arg = calibration
            cache_dir_arg = None
        print(
            f"[awq-quant] running {recipe_label} (HF cache path): "
            f"splits={splits_arg!r}, seq_len={args.max_seq_length}, "
            f"cache_dir={cache_dir_arg!r}"
        )
        if cache_dir_arg:
            # PerfectBlendDataset.load_dataset (inherited from
            # TextGenerationDataset) hardcodes cache_dir=None when calling
            # load_dataset, so datasets falls back to ~/.cache/huggingface/
            # — empty on a GPU node. Monkey-patch load_dataset on the
            # registered PerfectBlendDataset class for the duration of
            # this oneshot() call to thread our local HF_DATASETS_CACHE
            # through to load_dataset().
            from llmcompressor.transformers.data.perfectblend import (
                PerfectBlendDataset,
            )
            from llmcompressor.transformers.data import TextGenerationDataset
            _original_load = TextGenerationDataset.load_dataset

            def _patched_load_dataset(self):
                # Mirror the parent's body but pass cache_dir explicitly
                # to get_raw_dataset.
                from llmcompressor.transformers.data.data_helpers import (
                    get_raw_dataset,
                )
                logger = __import__("loguru").logger
                logger.debug(
                    f"Loading dataset {self.dataset_args.dataset} "
                    f"(cache_dir={cache_dir_arg!r})"
                )
                return get_raw_dataset(
                    self.dataset_args,
                    cache_dir=cache_dir_arg,
                    split=self.split,
                    streaming=self.dataset_args.streaming,
                    **self.dataset_args.raw_kwargs,
                )

            TextGenerationDataset.load_dataset = _patched_load_dataset
            try:
                oneshot(
                    model=model,
                    dataset=dataset_arg,
                    splits=splits_arg,
                    recipe=recipe,
                    max_seq_length=args.max_seq_length,
                    moe_calibrate_all_experts=True,
                )
            finally:
                TextGenerationDataset.load_dataset = _original_load
        else:
            oneshot(
                model=model,
                dataset=dataset_arg,
                splits=splits_arg,
                recipe=recipe,
                max_seq_length=args.max_seq_length,
                moe_calibrate_all_experts=True,
            )
    else:
        print(
            f"[awq-quant] running {recipe_label} (parquet-direct path), "
            f"seq_len={args.max_seq_length}"
        )
        # PerfectBlendDataset is a Mapping (preprocesses on __getitem__); it
        # doesn't expose __len__ until it's wrapped in a DataLoader. Pass it
        # directly to oneshot(), which constructs the DataLoader internally.
        oneshot(
            model=model,
            dataset=calibration,
            recipe=recipe,
            max_seq_length=args.max_seq_length,
            moe_calibrate_all_experts=True,
        )

    print(f"[awq-quant] saving to {output_dir}")
    model.save_pretrained(str(output_dir), save_compressed=True)
    tokenizer.save_pretrained(str(output_dir))

    src_gen_cfg = Path(args.model_id) / "generation_config.json"
    if src_gen_cfg.is_file():
        shutil.copy2(src_gen_cfg, output_dir / "generation_config.json")

    files = sorted(p.name for p in output_dir.iterdir())
    print(f"[awq-quant] done. output_dir contents: {files}")
    if not (output_dir / "config.json").exists():
        print(f"[awq-quant][error] no config.json in output_dir!", file=sys.stderr)
        sys.exit(1)
    # Note: we DON'T require a standalone quantize_config.json. llm-compressor's
    # save_pretrained(save_compressed=True) embeds the quantization_config inside
    # config.json (under the "quantization_config" key, with
    # quant_method="compressed-tensors"). vLLM reads that embedded config when
    # launched with --quantization compressed-tensors, so the file is correct
    # as-is. The previous check for a standalone quantize_config.json was based
    # on an outdated assumption (moe_wna16 path) and falsely failed every run
    # that successfully produced a checkpoint.


if __name__ == "__main__":
    main()