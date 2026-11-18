#!/usr/bin/env bash
#
# spartan/vllm-moe-eval-h100-w4a16-hf-mixtral-tp4.sbatch
#
# Mixtral W4A16 rerun using HuggingFace prequantized weights (instead
# of the local prequant ckpts at ${SCRATCH_BASE}/hf_cache_awq/, which
# produced degenerate routed-experts data — see EVAL_MOE_HANDOFF.md
# Addendum 9 + MIXTRAL_W4A16_BUG_HANDOFF.md).
#
# The two HF model IDs are:
#   - mistral-community/Mixtral-8x22B-v0.1-AWQ          (AWQ, base v0.1)
#   - jarrelscy/Mixtral-8x22B-Instruct-v0.1-GPTQ-4bit  (GPTQ, Instruct)
#
# Both pre-staged under ${HF_CACHE}/ by a login-node `hf download` run
# (the GPU nodes are firewalled from HF Hub).
#
# The HF AWQ config has quant_method="awq" with bits=4, group_size=128,
# modules_to_not_convert=["gate"]. The HF GPTQ config has
# quant_method="gptq" with bits=4, group_size=128, desc_act=false.
# vLLM auto-detects both via the HF config's quantization_config —
# we do NOT pass --quant, so each model uses its native handler
# (awq_marlin for AWQ, gptq_marlin for GPTQ).
#
# IMPORTANT differences from the local prequant sweep:
#   - NO --moe-backend triton (the Triton workaround attempt was a
#     dead end — both Marlin and Triton produce byte-identical
#     degenerate capturer data for W4A16 + top_k=2 Mixtral)
#   - NO --moe-backend override at all (let vLLM pick)
#   - 8h wall (4h was tight for the 1h09m Marlin run; the Triton
#     attempt hit the 4h wall mid-include. With Marlin's been-killed-
#     on-H100 routing-replay path now the default, we expect the
#     same ~1h09m/cell as the local prequant AWQ run — 5 cells ×
#     2 models = 10 cells ≈ 12h... so we run 1 model at a time
#     to fit 8h)
#
# To switch which HF model runs, override MODELS_OVERRIDE.
# To force a different quant handler, set QUANTS_OVERRIDE explicitly.
# ----------------------------------------------------------------

#SBATCH --job-name=vllm-moe-w4a16-hf-mixtral-tp4
#SBATCH --partition=gpu-h100
#SBATCH --account=uom00014
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=8:00:00
#SBATCH --output=spartan/logs/vllm-moe-eval-w4a16-hf-mixtral-tp4-%j.out
#SBATCH --error=spartan/logs/vllm-moe-eval-w4a16-hf-mixtral-tp4-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pratheek.senevirathne@student.unimelb.edu.au
#SBATCH --export=ALL
#SBATCH --signal=B:SIGTERM@60

set -euo pipefail
shopt -s nullglob

# ---------------------------------------------------------------- modules
GCC_MODULE="${GCC_MODULE:-GCCcore/11.3.0}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3}"
CUDA_MODULE="${CUDA_MODULE:-CUDA/12.4.1}"
# NCCL/2.18.3-GCCcore-11.3.0 does not exist on Spartan — `module spider
# NCCL` lists only CUDA-suffixed builds (2.12.12-CUDA-11.7.0 through
# 2.27.7-CUDA-12.8.0). Use the same NCCL/2.22.3-CUDA-12.4.1 that every
# other vLLM sbatch in this repo uses; loading a missing module is a
# hard [fatal] exit 3 in this script.
NCCL_MODULE="${NCCL_MODULE:-NCCL/2.22.3-CUDA-12.4.1}"
# Ninja must match the active Python module — module load puts
# ninja on $PATH for the parent and all forked children. The
# default `Ninja/1.11.1-Python-3.11.3` matches Python/3.11.3.
NINJA_MODULE="${NINJA_MODULE:-Ninja/1.11.1-Python-3.11.3}"
NGPUS="${NGPUS:-4}"
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

# vLLM uses HF_HUB_CACHE (or HUGGINGFACE_HUB_CACHE) to find the
# model. We pre-staged both HF models under ${HF_CACHE}/ via login-
# node `hf download`. With HF_HUB_OFFLINE=1 the GPU node will NOT
# try to reach HF Hub (firewalled) — it just resolves the model
# id via the local cache.
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Cell defaults: 2 HF model IDs (AWQ + GPTQ). The analyzer will
# build cells as a Cartesian product of (models x quants). We pass
# an empty --quant so vLLM auto-detects from each model's HF config
# (awq_marlin for AWQ, gptq_marlin for GPTQ).
#
# IMPORTANT: the user asked for the INSTRUCT variant of the AWQ
# (MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-AWQ), not the base
# v0.1 (mistral-community/Mixtral-8x22B-v0.1-AWQ). The Instruct
# variant matches the GPTQ counterpart (both v0.1-Instruct), and
# is the right reference for instruction-tuned benchmark scoring.
HF_AWQ_MODEL="MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-AWQ"
HF_GPTQ_MODEL="jarrelscy/Mixtral-8x22B-Instruct-v0.1-GPTQ-4bit"

# Default: run the AWQ cell first (lowest risk, gives us the
# cleanest comparison to the local prequant AWQ cell from job
# 32517565). Run GPTQ in a follow-up submission by overriding
# MODELS_OVERRIDE.
MODELS_OVERRIDE="${MODELS_OVERRIDE:-${HF_AWQ_MODEL}}"
QUANTS_OVERRIDE="${QUANTS_OVERRIDE:-}"  # empty => vLLM auto-detect
BENCHMARKS="${BENCHMARKS:-mmlu bbh humaneval popqa include}"
NUM_SAMPLES="${NUM_SAMPLES:-999999}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
N_SHOTS="${N_SHOTS:-5}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
ENFORCE_EAGER="${ENFORCE_EAGER:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_BASE}/results_wide_w4a16_hf_h100}"
SEED="${SEED:-0}"
SKIP_PYTHON_PLOTS="${SKIP_PYTHON_PLOTS:-1}"
MOE_BACKEND="${MOE_BACKEND:-}"  # do NOT force triton — see header comment

# ---------------------------------------------------------------- banner
echo "============================================================"
echo "[$(date -Iseconds)] vllm-moe-eval-w4a16-hf-mixtral-tp4 starting"
echo "[$(date -Iseconds)] job_id        = ${SLURM_JOB_ID:-}"
echo "[$(date -Iseconds)] node          = ${SLURM_JOB_NODELIST:-}"
echo "[$(date -Iseconds)] cpus          = ${SLURM_CPUS_PER_TASK:-}"
echo "[$(date -Iseconds)] gpus          = ${CUDA_VISIBLE_DEVICES:-<unset>} (NGPUS=${NGPUS})"
echo "[$(date -Iseconds)] REPO_ROOT     = ${REPO_ROOT}"
echo "[$(date -Iseconds)] SCRATCH_BASE  = ${SCRATCH_BASE}"
echo "[$(date -Iseconds)] HF_HOME       = ${HF_CACHE}"
echo "[$(date -Iseconds)] HF_HUB_OFFLINE= ${HF_HUB_OFFLINE}"
echo "[$(date -Iseconds)] MODELS        = ${MODELS_OVERRIDE}"
echo "[$(date -Iseconds)] QUANTS        = ${QUANTS_OVERRIDE:-<auto>}"
echo "[$(date -Iseconds)] MOE_BACKEND   = ${MOE_BACKEND:-<auto>}"
echo "[$(date -Iseconds)] BENCHMARKS    = ${BENCHMARKS}"
echo "[$(date -Iseconds)] NUM_SAMPLES   = ${NUM_SAMPLES}"
echo "[$(date -Iseconds)] MAX_TOKENS    = ${MAX_TOKENS}"
echo "[$(date -Iseconds)] MAX_MODEL_LEN = ${MAX_MODEL_LEN}"
echo "[$(date -Iseconds)] GPU_MEM_UTIL  = ${GPU_MEM_UTIL}"
echo "[$(date -Iseconds)] ENFORCE_EAGER = ${ENFORCE_EAGER:-<unset>}"
echo "[$(date -Iseconds)] OUTPUT_DIR    = ${OUTPUT_DIR}"
echo "[$(date -Iseconds)] SEED          = ${SEED}"
echo "[$(date -Iseconds)] CUDA_MODULE   = ${CUDA_MODULE}"
echo "[$(date -Iseconds)] PYTHON_MODULE = ${PYTHON_MODULE}"
echo "============================================================"

# Preflight: HF model must be pre-staged. With HF_HUB_OFFLINE=1,
# vLLM resolves the model id through HF_HUB_CACHE; the
# cache-must-have-config check below catches "forgot to run
# `hf download`" before we burn 8h on a job that can't find its
# inputs.
# HF Hub cache layout is `models--ORG--REPO/` (each `/` becomes `-`
# AND a `--` separator is inserted between components). For a model
# id like `MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-AWQ` the on-disk
# path is `models--MaziyarPanahi--Mixtral-8x22B-Instruct-v0.1-AWQ` —
# note the DOUBLE dash between org and repo. Building that from the
# model id with `tr '/' '-'` (single dash) or `awk '{print $1 "-" $2}'`
# produces the WRONG path; the preflight then exits 3 in 2 seconds
# even though the weights are perfectly pre-staged. Using bash
# parameter expansion to replace every '/' with '--' produces the
# canonical HF Hub cache path correctly.
for m in ${MODELS_OVERRIDE}; do
    HF_DIR="${HF_CACHE}/models--${m//\//--}"
    HF_SNAP="${HF_DIR}/snapshots"
    if [[ ! -d "${HF_SNAP}" ]]; then
        echo "[fatal] ${m} is not pre-staged under ${HF_DIR}" >&2
        echo "[fatal] Run this on the login node first:" >&2
        echo "[fatal]   export HF_HUB_CACHE=${HF_CACHE}" >&2
        echo "[fatal]   hf download ${m} --repo-type model --include '*.json' '*.safetensors' 'tokenizer*' '*.jinja' '*.model' '*.txt' '*.py' 'README*'" >&2
        exit 3
    fi
    if ! ls "${HF_SNAP}"/*/config.json >/dev/null 2>&1; then
        echo "[fatal] ${m} is staged but has no config.json in any snapshot" >&2
        echo "[fatal] Re-run hf download — the previous attempt was interrupted" >&2
        exit 3
    fi
done

# Auto-wipe the 10 stale Mixtral W4A16 expert_counts.json files
# (5 benchmarks × 2 recipes) from the local prequant sweep, so the
# analyzer's idempotent skip doesn't skip them. They live under
# ${SCRATCH_BASE}/results_wide_w4a16_awq_h100/ — a separate dir
# from the new HF results. We just want them deleted so the
# aggregator can be re-run from scratch later.
LEGACY_DIR="${SCRATCH_BASE}/results_wide_w4a16_awq_h100"
if [[ -d "${LEGACY_DIR}" ]]; then
    echo "[$(date -Iseconds)] auto-wiping stale Mixtral W4A16 expert_counts.json from ${LEGACY_DIR}"
    find "${LEGACY_DIR}" -name 'expert_counts.json' -path '*Mixtral*' -delete
    find "${LEGACY_DIR}" -name 'driver_*.log' -delete
fi

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
        --torch-backend "auto" \
        -e "${REPO_ROOT}"
    # Pin datasets==4.5.0: the login-node pre-download wrote each
    # split with system Python 3.9's datasets 4.5.0. The GPU venv
    # must use the same version to avoid KeyError: '_indexes' on
    # Dataset.load_from_disk.
    "${VENV_DIR}/bin/pip" install --quiet "datasets==4.5.0" "numpy>=2.0.0,<=2.5.3"
else
    echo "[$(date -Iseconds)] venv already built at ${VENV_DIR}; skipping"
fi

# ---------------------------------------------------------------- preflight
# Confirm the analyzer can read each HF model's config. Cheap and
# catches path/permission issues before we burn 8h on a job that
# can't find its inputs.
"${VENV_DIR}/bin/python" - <<EOF
import os
from transformers import AutoConfig
for m in """${MODELS_OVERRIDE}""".split():
    cfg = AutoConfig.from_pretrained(m, trust_remote_code=True)
    qc = getattr(cfg, "quantization_config", None) or {}
    print(f"OK {m}: model_type={cfg.model_type} quant_method={qc.get('quant_method')} bits={qc.get('bits') or qc.get('w_bits')}")
EOF

# ---------------------------------------------------------------- run
cd "${REPO_ROOT}/examples/eval-moe"

# Build the QUANTS arg as a shell-quoted space-separated list.
# MODELS_OVERRIDE is already pre-tokenized by bash word splitting.
QUANTS_SHELL=""
if [[ -n "${QUANTS_OVERRIDE}" ]]; then
    for q in ${QUANTS_OVERRIDE}; do QUANTS_SHELL="${QUANTS_SHELL} '${q}'"; done
fi

# Run the analyzer. It accepts --models as nargs="+", --quant as
# nargs="+", so the Cartesian product (models x quants) is built
# automatically. Each cell is independently skipped if the json
# already exists at the output path (idempotent re-runs).
"${VENV_DIR}/bin/python" moe_expert_stats.py \
    --models ${MODELS_OVERRIDE} \
    ${QUANTS_OVERRIDE:+--quant ${QUANTS_OVERRIDE}} \
    --benchmarks ${BENCHMARKS} \
    --num-samples "${NUM_SAMPLES}" \
    --max-tokens "${MAX_TOKENS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --n-shots "${N_SHOTS}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --tensor-parallel-size "${NGPUS}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    ${ENFORCE_EAGER:+--enforce-eager} \
    ${MOE_BACKEND:+--moe-backend "${MOE_BACKEND}"}

echo "[$(date -Iseconds)] vllm-moe-eval-w4a16-hf-mixtral-tp4 done"
