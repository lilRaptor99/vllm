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

# llmcompressor.entrypoints.oneshot.Oneshot.validate_model (oneshot.py:292)
# does its own `AutoConfig.from_pretrained(...)` on the model without
# forwarding `trust_remote_code=True`. For DeepSeek-MoE (which ships custom
# modeling code) this triggers a fresh interactive prompt and dies with
# `ValueError: ... must pass trust_remote_code=True` under SLURM (no TTY).
# Force the resolver to always return True.
#
# IMPORTANT: every call site does
#   from ...dynamic_module_utils import resolve_trust_remote_code
# which creates a *local* binding in the consumer module. Just reassigning
# the attribute on dynamic_module_utils leaves those local bindings
# pointing at the original function (the failure mode we hit on jobs
# 32236909/32236910 with the v1 shim). Walk sys.modules and overwrite the
# local binding in any already-loaded transformers module that has it, so
# the patched function is what gets called.
import sys
import transformers.dynamic_module_utils as _dmu
import transformers.models.auto.configuration_auto as _ca
import transformers.models.auto.auto_factory as _af
import transformers.models.auto.tokenization_auto as _ta
import transformers.models.auto.image_processing_auto as _ia
import transformers.models.auto.video_processing_auto as _va
import transformers.models.auto.feature_extraction_auto as _fea
import transformers.models.auto.processing_auto as _pa


def _force_trust_remote_code(*_a, **_k):
    return True


_dmu.resolve_trust_remote_code = _force_trust_remote_code  # type: ignore[assignment]
for _mod in (
    _ca,
    _af,
    _ta,
    _ia,
    _va,
    _fea,
    _pa,
    sys.modules.get("transformers.generation.utils"),
):
    if _mod is not None and hasattr(_mod, "resolve_trust_remote_code"):
        _mod.resolve_trust_remote_code = _force_trust_remote_code  # type: ignore[attr-defined]
del _mod, _ca, _af, _ta, _ia, _va, _fea, _pa

# DeepseekPreTrainedModel declares `_tied_weights_keys = ["lm_head.weight"]`
# (a list) in modeling_deepseek.py:1267. transformers v5.15.1's
# `_get_tied_weight_keys` (modeling_utils.py:381) expects a dict and
# calls `tied.keys()` — AttributeError on a list. The lm_head in the
# saved DeepSeek-MoE safetensors is *not* tied to embeddings (separate
# tensor), so clearing the tied-weights claim is correct for this
# checkpoint. Wrap the function to coerce lists to empty dicts (and
# dicts-without-tensor-mapping to empty dicts) so the iteration works
# for any old-style custom modeling code.
import transformers.modeling_utils as _mu
_orig_get_tied = _mu._get_tied_weight_keys


def _patched_get_tied_weight_keys(module):
    out = []
    for name, sub in module.named_modules():
        tied = getattr(sub, "_tied_weights_keys", None)
        if tied is None:
            continue
        if isinstance(tied, list):
            # Old API: a list of key names that are tied to the previous
            # one. v5 can't model that generically — just drop them all
            # so the iteration below is empty. Safe for DeepSeek because
            # the saved checkpoint stores lm_head as its own tensor.
            continue
        if not isinstance(tied, dict):
            continue
        out.extend([f"{name}.{k}" if name else k for k in tied.keys()])
    return out


_mu._get_tied_weight_keys = _patched_get_tied_weight_keys
del _orig_get_tied

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQMapping, AWQModifier
from llmcompressor.utils import load_context

# Imported here (instead of at module top) so the file is syntactically
# import-clean for the repo's analyzer venv which lacks compressed_tensors.
# build_recipe() runs only inside venv_awq which has it.
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)


RECIPES = ("awq", "gptq", "awq_gptq")


def build_recipe(recipe: str, model_id: str = "") -> list:
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

    DeepSeek-MoE-16B's per-expert `mlp.down_proj` weight has 10944 columns,
    which is NOT divisible by 128 (10944 / 128 = 85.5). The W4A16 preset's
    group_size=128 would (a) fail llm-compressor's divisibility validation
    AND (b) fail the runtime reshape `unflatten(-1, (86, 128))` which expects
    11008 cols not 10944. vLLM's CompressedTensorsMoEMethod further requires
    that all 3 MoE projections (gate, up, down) within a single FusedMoE
    share a quantization scheme, so leaving down_proj in bf16 (the prior
    approach via `ignore=["re:.*\\.down_proj$"]`) causes a load-time
    `ValueError: All MoE projections need to have same quantization scheme`
    in vLLM 0.1.dev20195+gf7160722f (job 32345449 2026-10-06).

    Fix: quantize DeepSeek at group_size=64 (10944 % 64 == 0 exactly,
    so no reshape error, no padding waste). This is a standard WNA16
    Marlin-supported group size ({32, 64, 128}). The compression ratio
    drop is tiny in absolute terms because the extra scale bytes per
    row are dwarfed by the per-token int4 data bytes. Per-expert
    down_proj goes from bf16 4096x10944 = 86 MB to int4 group=64 =
    ~5.6 MB (same 15x compression as group=128).

    For AWQ we restore the full `mappings=[...]` including the
    up_proj -> down_proj pair (now that down_proj is in the quant set,
    AWQ's `_compute_layer_means` won't divide by zero). The previous
    "no down_proj mapping" hack is no longer needed. We still pass
    `dampening_frac=0.02` for GPTQ to reduce the 14.2% Hessian
    inversion failure rate on this branch of the Hessian's structure
    (job 32266240 2026-10-05).

    OLMoE/Qwen3/Mixtral all have down_proj columns % 128 == 0 (16384,
    768, 1024 respectively) so they keep the W4A16 preset (group_size=128)
    and the standard AWQ mapping registry.
    """
    ignore = [
        "lm_head",
        "re:.*mlp\\.gate$",
        "re:.*mlp\\.shared_expert_gate$",
    ]
    is_deepseek = "deepseek" in (model_id or "").lower()

    # DeepSeek-specific scheme: int4 weights, group_size=64 (not 128) so
    # the 10944-col down_proj divides evenly. Built inline via config_groups
    # because the "W4A16" preset hardcodes group_size=128 and the `scheme=`
    # validator only accepts preset-name keys or whole-scheme dicts.
    from compressed_tensors.quantization import QuantizationScheme
    from compressed_tensors.quantization.quant_args import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )

    deepseek_weights_args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        group_size=64,
        symmetric=True,
        dynamic=False,
    )
    deepseek_config_groups = {
        "group_0": QuantizationScheme(
            targets=["Linear"],
            weights=deepseek_weights_args,
        ),
    }

    if is_deepseek:
        # Full DeepSeek AWQ mappings — down_proj is now in the quant set
        # (since ignore no longer skips it), so include the up_proj ->
        # down_proj scaling pair.
        deepseek_awq_mappings = [
            AWQMapping(
                smooth_layer="re:.*input_layernorm$",
                balance_layers=["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
            ),
            AWQMapping(
                smooth_layer="re:.*v_proj$",
                balance_layers=["re:.*o_proj$"],
            ),
            AWQMapping(
                smooth_layer="re:.*post_attention_layernorm$",
                balance_layers=["re:.*gate_proj$", "re:.*up_proj$"],
            ),
            AWQMapping(
                smooth_layer="re:.*up_proj$",
                balance_layers=["re:.*down_proj$"],
            ),
        ]
    else:
        deepseek_awq_mappings = None  # not used; AWQModifier picks its own

    if recipe == "awq":
        awq = (
            AWQModifier(mappings=deepseek_awq_mappings, duo_scaling="both", n_grid=10)
            if is_deepseek
            else AWQModifier(duo_scaling="both", n_grid=10)
        )
        if is_deepseek:
            return [
                awq,
                QuantizationModifier(
                    config_groups=deepseek_config_groups,
                    ignore=ignore,
                ),
            ]
        return [
            awq,
            QuantizationModifier(targets="Linear", scheme="W4A16", ignore=ignore),
        ]
    if recipe == "gptq":
        if is_deepseek:
            return [
                GPTQModifier(
                    config_groups=deepseek_config_groups,
                    ignore=ignore,
                    dampening_frac=0.02,
                ),
            ]
        return [
            GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignore),
        ]
    if recipe == "awq_gptq":
        awq = (
            AWQModifier(mappings=deepseek_awq_mappings, duo_scaling="both", n_grid=10)
            if is_deepseek
            else AWQModifier(duo_scaling="both", n_grid=10)
        )
        if is_deepseek:
            return [
                awq,
                GPTQModifier(
                    config_groups=deepseek_config_groups,
                    ignore=ignore,
                    dampening_frac=0.02,
                ),
            ]
        gptq_kwargs = {"targets": "Linear", "scheme": "W4A16", "ignore": ignore}
        return [
            awq,
            GPTQModifier(**gptq_kwargs),
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
        #
        # Force `rope_scaling=None` in the model config before construction.
        # The Hub's modeling_deepseek.py is for a newer transformers than
        # v5.15.1; in v5.x, `PreTrainedConfig.__init__` calls
        # `convert_rope_params_to_dict` which materializes a missing/null
        # `rope_scaling` into a truthy `rope_parameters` dict like
        # `{"rope_theta": 10000}` (no "type" key). The Hub code's
        # `if rope_scaling is None: ... else: rope_scaling["type"]` then
        # crashes with `KeyError: 'type'`. Forcing None on the config class
        # attribute (read by `getattr(config, "rope_scaling", None)`) makes
        # the dynamic module's rope init take the no-scaling branch.
        try:
            from transformers import AutoConfig  # type: ignore
            _cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
            if not getattr(_cfg, "rope_scaling", None) or (
                isinstance(_cfg.rope_scaling, dict)
                and "type" not in _cfg.rope_scaling
            ):
                # Reset the cached field so the dynamic module sees None.
                try:
                    _cfg.rope_scaling = None
                except (AttributeError, ValueError):
                    # Some v5.x configs make rope_scaling a property; reset
                    # via __dict__ if so.
                    if "rope_scaling" in _cfg.__dict__:
                        _cfg.__dict__["rope_scaling"] = None
                print(f"[awq-quant] reset config.rope_scaling=None for {args.model_id!r}")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                config=_cfg,
            )
        except Exception as _e:
            print(f"[awq-quant][warn] config-normalize path failed ({_e}); falling back to direct load")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                dtype=torch.bfloat16,
                trust_remote_code=True,
            )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True,
    )

    recipe = build_recipe(args.recipe, args.model_id)
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