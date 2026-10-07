#!/usr/bin/env bash
#
# spartan/vllm-moe-eval-h100-w4a16-prequant.sbatch - SLURM job that runs the
# vLLM MoE expert activation statistics analyzer on the Unimelb Spartan
# HPC on gpu-h100, loading the pre-quantized W4A16 compressed-tensors
# checkpoints that awq_quantize.py produced under
# ${SCRATCH_BASE}/hf_cache_awq/<model_safe>-<recipe>/.
#
# Forked from spartan/vllm-moe-eval-h100-w8a16-moe.sbatch on 2026-10-05
# to switch the load path from "online int8 quant on bf16 weights" to
# "load the int4 compressed-tensors checkpoint directly". The analyzer
# (examples/eval-moe/moe_expert_stats.py) is unchanged; we just point
# --model at the local prequant dir and pass --quantization
# compressed-tensors so vLLM's CompressedTensorsWNA16MoEMethod is
# selected for the routed experts (and the bf16 down_proj on DeepSeek
# is left alone, per the ignore pattern baked into the prequant).
#
# Per-model footprint of the prequant ckpt on disk (compressed-tensors
# pack-quantized, num_bits=4, group_size=128, symmetric):
#   OLMoE-1B-7B          3.7 GB on disk, ~4 GB on GPU at TP=1
#   DeepSeek-MoE-16B    16   GB on disk, ~17 GB on GPU at TP=1
#                          (per-expert down_proj stays bf16, +3 GB)
#   Qwen3-30B-A3B       16   GB on disk, ~17 GB on GPU at TP=1
#   Mixtral-8x22B       69   GB on disk, ~70 GB on GPU at TP=1
#                          (bf16 Mixtral: 262 GB; the W4A16 path is the
#                           only way to fit on a single 80 GB H100)
#
# So all 4 fit TP=1 on a single 80 GB H100, but only just barely for
# Mixtral (gpu_mem_util=0.92 leaves ~7 GB for KV at --max_model_len
# 4096, which is tight). The script defaults to a single SLURM job
# that runs all 4 cells sequentially.
#
# Why one job, not four:
#   The Python venv + build cache is loaded once; the analyzer is
#   loaded once; each cell is a separate `python moe_expert_stats.py`
#   invocation. Total wall time is ~3-4h per cell = 12-16h total, so
#   the sbatch --time is 0-24:00:00 to leave headroom.
#
# Pre-reqs
# --------
# 1. The prequant ckpts exist under ${SCRATCH_BASE}/hf_cache_awq/.
#    Produced by examples/eval-moe/awq_quantize.py via
#    examples/eval-moe/spartan/vllm-moe-awq-prequant.sbatch. Coverage
#    as of the v1 launch (2026-10-05): OLMoE{awq,gptq}, DeepSeek{awq,
#    gptq}, Qwen3{awq,gptq}, Mixtral{awq,gptq,awq_gptq} (7 of 11
#    recipes; the 3 awq_gptq for non-Mixtral models are deferred).
# 2. The HF datasets the analyzer reads have been pre-populated under
#    ${SCRATCH_BASE}/datasets. Run on the login node first:
#        bash spartan/download-datasets.sh
# 3. The vllm + datasets venv is built under ${SCRATCH_BASE}/venv/.
#    Created on the first analyzer sbatch run; cached thereafter.
#
# Quick start
# -----------
#   # 1. login node: pre-download the datasets
#   ssh spartan
#   cd <repo>/examples/eval-moe
#   bash spartan/download-datasets.sh
#
#   # 2. submit the GPU job. Defaults: 4 cells sequentially (all 4
#   #    models x 2 recipes = 8 cells; idempotent skip if json exists).
#   sbatch spartan/vllm-moe-eval-h100-w4a16-prequant.sbatch
#
#   # 3. tail logs
#   tail -f spartan/logs/vllm-moe-eval-h100-w4a16-prequant-<jobid>.out
#
# To run a single model/recipe cell, override:
#   sbatch --export=MODELS_OVERRIDE="${SCRATCH}/hf_cache_awq/Qwen--Qwen3-30B-A3B-awq",QUANTS_OVERRIDE=compressed-tensors,OUTPUT_DIR=... spartan/vllm-moe-eval-h100-w4a16-prequant.sbatch
#
# Env-var overrides (export before sbatch, or pass via --export=...)
# ------------------------------------------------------------------
#   REPO_ROOT       path to the cloned vllm repo
#   SCRATCH_BASE    root for venv / datasets / results / hf_cache_awq
#   MODELS_OVERRIDE space-separated prequant dirs (default: all 4 models
#                                                  x 2 recipes = 8 cells)
#   QUANTS_OVERRIDE quantization tag for the analyzer  (default: compressed-tensors)
#   BENCHMARKS      space-separated subset            (default: mmlu bbh humaneval popqa include)
#   NUM_SAMPLES     prompts per benchmark              (default: 999999)
#   OUTPUT_DIR      where to write JSONs               (default: results_wide_w4a16_awq_h100)
#   NGPUS           1 for OLMoE/DeepSeek/Qwen3, 2 for Mixtral
#   See vllm-moe-eval-h100-w8a16-moe.sbatch for the rest of the env-var
#   contract (CUDA_MODULE, PYTHON_MODULE, GCC_MODULE, NCCL_MODULE,
#   TORCH_BACKEND, UV_CACHE_DIR, etc.) - all identical between the
#   two sbatches.

#SBATCH --job-name=vllm-moe-eval-w4a16
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=0-24:00:00
#SBATCH --output=/data/gpfs/projects/uom00014/vllm/examples/eval-moe/spartan/logs/vllm-moe-eval-w4a16-prequant-%j.out
#SBATCH --error=/data/gpfs/projects/uom00014/vllm/examples/eval-moe/spartan/logs/vllm-moe-eval-w4a16-prequant-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pratheek.senevirathne@student.unimelb.edu.au

set -euo pipefail

# ---------------------------------------------------------------- paths
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="/data/gpfs/projects/uom00014/vllm/examples/eval-moe/spartan/logs"
mkdir -p "${LOG_DIR}"

CUDA_MODULE="${CUDA_MODULE:-CUDA/12.4.1}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3}"
GCC_MODULE="${GCC_MODULE:-GCCcore/11.3.0}"
TORCH_BACKEND="${TORCH_BACKEND:-cu129}"
NCCL_MODULE="${NCCL_MODULE:-NCCL/2.22.3-CUDA-12.4.1}"
# flashinfer's topk_topp_sampler JIT-compiles a CUDA kernel via the
# `ninja` build tool on first use. The venv has a venv-local
# `${VENV_DIR}/bin/ninja` from `pip install ninja`, but vLLM's
# EngineCore subprocess forks without inheriting the venv's bin on
# PATH (running `${VENV_DIR}/bin/python script.py` doesn't add the
# venv's bin to the parent shell's PATH). The EasyBuild Ninja module
# is the reliable source — module load puts ninja on $PATH for the
# parent and all forked children. The exact name varies by Python
# version; the default `Ninja/1.11.1-Python-3.11.3` matches the
# Python/3.11.3 module above. Override only if your Python module
# is different.
NINJA_MODULE="${NINJA_MODULE:-Ninja/1.11.1-Python-3.11.3}"
NGPUS="${NGPUS:-1}"
SPLIT_MODE="${SPLIT_MODE:-layer}"
EXTRA_TENSOR_SPLIT="${EXTRA_TENSOR_SPLIT:-}"

REPO_ROOT="${REPO_ROOT:-/data/gpfs/projects/uom00014/vllm}"
SCRATCH_BASE="${SCRATCH_BASE:-${SCRATCH:-/data/scratch/projects/uom00014}/vllm}"
VENV_DIR="${VENV_DIR:-${SCRATCH_BASE}/venv}"
DATASETS_DIR="${DATASETS_DIR:-${SCRATCH_BASE}/datasets}"
HF_CACHE="${HF_CACHE:-${SCRATCH_BASE}/hf_cache}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/.uv_cache}"
export UV_CACHE_DIR
UV_BUILD_PARALLEL="${UV_BUILD_PARALLEL:-8}"
export UV_BUILD_PARALLEL

mkdir -p "${DATASETS_DIR}" "${HF_CACHE}" "${VENV_DIR}" "${UV_CACHE_DIR}"

# The prequant checkpoints live under hf_cache_awq/, NOT hf_cache/. We
# point HUGGINGFACE_HUB_CACHE at the parent so vLLM can still resolve
# any incidental HF lookups (e.g. tokenizer config that the prequant
# save_pretrained copy may have missed), but the prequant dir itself
# is referenced by absolute path in MODELS_OVERRIDE so vLLM bypasses
# the HF cache and loads directly from disk.
PREQUANT_BASE="${PREQUANT_BASE:-${SCRATCH_BASE}/hf_cache_awq}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Cell defaults: all 4 models x 2 recipes (awq, gptq). awq_gptq is
# deferred (separate sbatch later). Each MODELS_OVERRIDE entry is an
# ABSOLUTE LOCAL PATH to a prequant dir; the analyzer passes that
# straight to LLM(model=path) which loads the safetensors directly.
# We list the 8 default cells explicitly rather than globbing so the
# user can comment out any cell they don't want to run today.
MODELS_OVERRIDE="${MODELS_OVERRIDE:-${PREQUANT_BASE}/allenai--OLMoE-1B-7B-0924-Instruct-awq ${PREQUANT_BASE}/allenai--OLMoE-1B-7B-0924-Instruct-gptq ${PREQUANT_BASE}/deepseek-ai--deepseek-moe-16b-chat-awq ${PREQUANT_BASE}/deepseek-ai--deepseek-moe-16b-chat-gptq ${PREQUANT_BASE}/Qwen--Qwen3-30B-A3B-awq ${PREQUANT_BASE}/Qwen--Qwen3-30B-A3B-gptq ${PREQUANT_BASE}/mistralai--Mixtral-8x22B-Instruct-v0.1-awq ${PREQUANT_BASE}/mistralai--Mixtral-8x22B-Instruct-v0.1-gptq}"
# compressed-tensors is the quant method baked into every prequant
# checkpoint's config.json (the loader dispatches to
# CompressedTensorsWNA16MoEMethod for the routed experts). Note the
# analyzer's --quant flag is the vLLM engine flag; the same name works.
QUANTS_OVERRIDE="${QUANTS_OVERRIDE:-compressed-tensors}"
BENCHMARKS="${BENCHMARKS:-mmlu bbh humaneval popqa include}"
NUM_SAMPLES="${NUM_SAMPLES:-999999}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
N_SHOTS="${N_SHOTS:-5}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
ENFORCE_EAGER="${ENFORCE_EAGER:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_BASE}/results_wide_w4a16_awq_h100}"
SEED="${SEED:-0}"
SKIP_PYTHON_PLOTS="${SKIP_PYTHON_PLOTS:-1}"

# ---------------------------------------------------------------- banner
echo "============================================================"
echo "[$(date -Iseconds)] vllm-moe-eval-w4a16-prequant starting"
echo "[$(date -Iseconds)] job_id        = ${SLURM_JOB_ID:-}"
echo "[$(date -Iseconds)] node          = ${SLURM_JOB_NODELIST:-}"
echo "[$(date -Iseconds)] cpus          = ${SLURM_CPUS_PER_TASK:-}"
echo "[$(date -Iseconds)] gpus          = ${CUDA_VISIBLE_DEVICES:-<unset>} (NGPUS=${NGPUS})"
echo "[$(date -Iseconds)] REPO_ROOT     = ${REPO_ROOT}"
echo "[$(date -Iseconds)] SCRATCH_BASE  = ${SCRATCH_BASE}"
echo "[$(date -Iseconds)] PREQUANT_BASE = ${PREQUANT_BASE}"
echo "[$(date -Iseconds)] HF_HOME       = ${HF_CACHE}"
echo "[$(date -Iseconds)] MODELS        = ${MODELS_OVERRIDE}"
echo "[$(date -Iseconds)] QUANTS        = ${QUANTS_OVERRIDE:-<unset>}"
echo "[$(date -Iseconds)] BENCHMARKS    = ${BENCHMARKS}"
echo "[$(date -Iseconds)] NUM_SAMPLES   = ${NUM_SAMPLES}"
echo "[$(date -Iseconds)] MAX_TOKENS    = ${MAX_TOKENS}"
echo "[$(date -Iseconds)] MAX_MODEL_LEN = ${MAX_MODEL_LEN}"
echo "[$(date -Iseconds)] GPU_MEM_UTIL  = ${GPU_MEM_UTIL}"
echo "[$(date -Iseconds)] ENFORCE_EAGER = ${ENFORCE_EAGER:-<unset>}"
echo "[$(date -Iseconds)] OUTPUT_DIR    = ${OUTPUT_DIR}"
echo "[$(date -Iseconds)] SEED          = ${SEED}"
echo "[$(date -Iseconds)] SKIP_PYTHON_PLOTS = ${SKIP_PYTHON_PLOTS}"
echo "[$(date -Iseconds)] CUDA_MODULE   = ${CUDA_MODULE}"
echo "[$(date -Iseconds)] PYTHON_MODULE = ${PYTHON_MODULE}"
echo "[$(date -Iseconds)] GCC_MODULE    = ${GCC_MODULE}"
echo "[$(date -Iseconds)] TORCH_BACKEND = ${TORCH_BACKEND}"
echo "[$(date -Iseconds)] NCCL_MODULE   = ${NCCL_MODULE:-<auto-probed below>}"
echo "[$(date -Iseconds)] NINJA_MODULE  = ${NINJA_MODULE}"
echo "============================================================"

# Preflight: prequant dirs must exist
for d in ${MODELS_OVERRIDE}; do
    if [[ ! -d "${d}" ]]; then
        echo "[fatal] MODELS_OVERRIDE entry '${d}' is not a directory" >&2
        echo "[fatal] Run examples/eval-moe/awq_quantize.py first to produce prequant ckpts under ${PREQUANT_BASE}/" >&2
        exit 3
    fi
    if [[ ! -f "${d}/config.json" ]]; then
        echo "[fatal] ${d} has no config.json (incomplete prequant run?)" >&2
        exit 3
    fi
done

# ---------------------------------------------------------------- modules
if [[ -z "${GCC_MODULE}" ]]; then echo "[fatal] GCC_MODULE empty" >&2; exit 3; fi
if [[ -z "${PYTHON_MODULE}" ]]; then echo "[fatal] PYTHON_MODULE empty" >&2; exit 3; fi
if [[ -z "${CUDA_MODULE}" ]]; then echo "[fatal] CUDA_MODULE empty" >&2; exit 3; fi
if [[ -z "${NINJA_MODULE}" ]]; then echo "[fatal] NINJA_MODULE empty" >&2; exit 3; fi

set +u
module purge
load_module_or_die() {
    local name="$1"
    local stderr_log
    stderr_log="$(mktemp)"
    if ! module load "${name}" 2>"${stderr_log}"; then
        echo "[fatal] module load ${name} failed:" >&2
        sed 's/^/  /' "${stderr_log}" >&2
        echo "[fatal] Hint: run 'module spider ${name}' on a gpu-a100-short node to see the load chain." >&2
        rm -f "${stderr_log}"; exit 3
    fi
    rm -f "${stderr_log}"
}
load_module_or_die "${GCC_MODULE}"
load_module_or_die "${PYTHON_MODULE}"
load_module_or_die "${CUDA_MODULE}"
load_module_or_die "${NINJA_MODULE}"
if [[ "${NGPUS}" -gt 1 ]]; then load_module_or_die "${NCCL_MODULE}"; fi
module list
set -u

# ---------------------------------------------------------------- venv
# Build the analyzer venv on first run. uv's wheel cache lives on
# ${UV_CACHE_DIR} (GPFS) so the second submission skips the heavy
# build when pyproject.toml + the editable source hash haven't changed.
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[$(date -Iseconds)] building venv at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/pip" install --quiet --upgrade pip wheel uv
    VLLM_USE_PRECOMPILED=0 \
    UV_CACHE_DIR="${UV_CACHE_DIR}" \
        "${VENV_DIR}/bin/uv" pip install --python "${VENV_DIR}/bin/python" \
        --torch-backend "${TORCH_BACKEND}" \
        -e "${REPO_ROOT}"
    # Pin datasets==4.5.0: the login-node pre-download wrote each
    # split with system Python 3.9's datasets 4.5.0. The GPU venv
    # must use the same version to avoid KeyError: '_indexes' on
    # Dataset.load_from_disk. The AWQ prequant venv at
    # ${SCRATCH}/venv_awq/ uses datasets>=4.8.4 because
    # llm-compressor@main requires it; the analyzer venv does not.
    "${VENV_DIR}/bin/pip" install --quiet "datasets==4.5.0" "numpy>=2.0.0,<=2.5.3"
else
    echo "[$(date -Iseconds)] venv already built at ${VENV_DIR}; skipping"
fi

# ---------------------------------------------------------------- preflight
# Confirm the analyzer can read each prequant dir's config. Cheap and
# catches path/permission issues before we burn 12h on a job that
# can't find its inputs.
"${VENV_DIR}/bin/python" - <<EOF
import os
from transformers import AutoConfig
for path in """${MODELS_OVERRIDE}""".split():
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    qc = getattr(cfg, "quantization_config", None) or {}
    print(f"OK {path}: model_type={cfg.model_type} quant_method={qc.get('quant_method')}")
EOF

# ---------------------------------------------------------------- run
cd "${REPO_ROOT}/examples/eval-moe"

# Build the QUANTS arg as a shell-quoted space-separated list.
# MODELS_OVERRIDE is already pre-tokenized by bash word splitting.
QUANTS_SHELL=""
for q in ${QUANTS_OVERRIDE}; do QUANTS_SHELL="${QUANTS_SHELL} '${q}'"; done

# Run the analyzer. It accepts --models as nargs="+", --quant as
# nargs="+", so the Cartesian product (models x quants) is built
# automatically. Each cell is independently skipped if the json
# already exists at the output path (idempotent re-runs).
"${VENV_DIR}/bin/python" moe_expert_stats.py \
    --models ${MODELS_OVERRIDE} \
    --quant ${QUANTS_OVERRIDE} \
    --benchmarks ${BENCHMARKS} \
    --num-samples "${NUM_SAMPLES}" \
    --max-tokens "${MAX_TOKENS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --n-shots "${N_SHOTS}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --tensor-parallel-size "${NGPUS}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    ${ENFORCE_EAGER:+--enforce-eager}

echo "[$(date -Iseconds)] vllm-moe-eval-w4a16-prequant done"
