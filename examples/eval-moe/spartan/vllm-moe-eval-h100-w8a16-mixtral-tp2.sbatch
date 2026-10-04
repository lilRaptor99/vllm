#!/usr/bin/env bash
#
# spartan/vllm-moe-eval-h100-w8a16-mixtral-tp2.sbatch - SLURM job that
# runs JUST the Mixtral-8x22B-Instruct-v0.1 cell of the w8a16 sweep at
# TP=2 on gpu-h100. Forked from spartan/vllm-moe-eval-h100-w8a16-moe.sbatch
# on 2026-09-29 after both TP=1 attempts (31280753 at gpu_mem_util=0.92
# and 31425093 at 0.85) failed at the same point: the layerwise online
# int8 quant step in the mixtral loader holds a full bf16 expert
# weight shard (~384 MiB) in memory WHILE the int8 quant allocates
# a parallel int8 tensor, and that intermediate state exceeds even
# 0.85*80 GiB on a single H100 (Mixtral bf16 disk = 262 GiB).
#
# TP=2 halves the per-GPU weight footprint (262/2 = 131 GiB bf16 ->
# ~65 GiB w8a16), and more importantly halves the per-rank shard size
# that the layerwise loader holds during the quant spike. With 2 GPUs
# the quant spike should fit comfortably under 0.92 of one 80 GiB H100.
#
# Why a separate sbatch instead of editing the original:
#   - The full w8a16 sweep sbatch hardcodes --gres=gpu:1 in the
#     #SBATCH directive and the comment above explains why it can't
#     use a shell variable. Editing it to gpu:2 would re-do the
#     already-cached cells (OLMoE, DeepSeek-MoE, Qwen3-30B) which the
#     analyzer would skip on the JSON-exists check, but the engine
#     load + warmup per cell is ~50-90s of pure waste when the goal
#     is just to retry Mixtral.
#   - This Mixtral-only TP=2 sbatch is one cell, one engine load, one
#     80-120 minute run.
#
# IMPORTANT - Mirror the w8a16 sbatch's env-var contract exactly so the
# analyzer invocation matches what's already documented in the README.
# N_SHOTS, NUM_SAMPLES, MAX_MODEL_LEN, BENCHMARKS, SEED, OUTPUT_DIR
# all default to the same values as the full sweep so the per-cell
# JSONs are Jaccard-comparable with the other w8a16 cells already on
# scratch (well, on MediaFlux since we uploaded + deleted those).
# Output tree:
#   ${SCRATCH_BASE}/results_wide_w8a16_h100/mistralai--Mixtral-8x22B-Instruct-v0.1/int8_per_channel_weight_only/{moe-mmlu,moe-bbh,moe-humaneval,moe-popqa,moe-include}/expert_counts.json
# (this is the SAME path the failed TP=1 cell would have written to,
# so the analyzer's "skip if output exists" check still works for a
# hypothetical second TP=2 retry.)

# ---------------------------------------------------------------- SLURM
#SBATCH --job-name=vllm-moe-eval-h100-w8a16-mixtral-tp2
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=0-12:00:00
#SBATCH --output=/data/gpfs/projects/uom00014/vllm/examples/eval-moe/spartan/logs/vllm-moe-eval-%j.out
#SBATCH --error=/data/gpfs/projects/uom00014/vllm/examples/eval-moe/spartan/logs/vllm-moe-eval-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pratheek.senevirathne@student.unimelb.edu.au

# ---------------------------------------------------------------- setup
# Everything below is lifted from spartan/vllm-moe-eval-h100-w8a16-moe.sbatch
# with three changes:
#   1. NGPUS=2 (must mirror --gres=gpu:2; see comment in the parent sbatch
#      about why this can't be a single #SBATCH line).
#   2. MODELS_OVERRIDE pinned to just Mixtral (skip the other 3 cells).
#   3. NCCL_MODULE is now REQUIRED (auto-loaded) since NGPUS > 1.

set -euo pipefail

SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="/data/gpfs/projects/uom00014/vllm/examples/eval-moe/spartan/logs"
mkdir -p "${LOG_DIR}"

CUDA_MODULE="${CUDA_MODULE:-CUDA/12.4.1}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3}"
GCC_MODULE="${GCC_MODULE:-GCCcore/11.3.0}"
TORCH_BACKEND="${TORCH_BACKEND:-cu129}"
NCCL_MODULE="${NCCL_MODULE:-NCCL/2.22.3-CUDA-12.4.1}"
NGPUS=2
SPLIT_MODE="${SPLIT_MODE:-layer}"
EXTRA_TENSOR_SPLIT="${EXTRA_TENSOR_SPLIT:-50,50}"

REPO_ROOT="${REPO_ROOT:-/data/gpfs/projects/uom00014/vllm}"
SCRATCH_BASE="${SCRATCH_BASE:-${SCRATCH:-/data/scratch/projects/uom00014}/vllm}"

VENV_DIR="${SCRATCH_BASE}/venv"
DATASETS_DIR="${SCRATCH_BASE}/datasets"
RESULTS_DIR="${SCRATCH_BASE}/results"
HF_CACHE="${SCRATCH_BASE}/hf_cache"
UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/.uv_cache}"
export UV_CACHE_DIR

UV_BUILD_PARALLEL="${UV_BUILD_PARALLEL:-8}"
export UV_BUILD_PARALLEL

mkdir -p "${DATASETS_DIR}" "${RESULTS_DIR}" "${HF_CACHE}" "${VENV_DIR}" "${UV_CACHE_DIR}"

# Cell config - pinned to Mixtral only.
MODELS_OVERRIDE="${MODELS_OVERRIDE:-mistralai/Mixtral-8x22B-Instruct-v0.1}"
QUANTS_OVERRIDE="${QUANTS_OVERRIDE:-int8_per_channel_weight_only}"
BENCHMARKS="${BENCHMARKS:-mmlu bbh humaneval popqa include}"
NUM_SAMPLES="${NUM_SAMPLES:-999999}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
N_SHOTS="${N_SHOTS:-5}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_BASE}/results_wide_w8a16_h100}"
SEED="${SEED:-0}"
SKIP_PYTHON_PLOTS="${SKIP_PYTHON_PLOTS:-1}"
SKIP_DATASET_CHECK="${SKIP_DATASET_CHECK:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-}"

export HF_HOME="${HF_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HUB_OFFLINE

# ---------------------------------------------------------------- modules
module purge
module load "${GCC_MODULE}"
module load "${PYTHON_MODULE}"
module load "${CUDA_MODULE}"
# NCCL is required for TP>1 (vLLM tensor-parallel needs cross-GPU
# collectives). Auto-detect from `module avail NCCL/` if not pinned.
if [[ -z "${NCCL_MODULE}" ]]; then
    NCCL_MODULE="$(module avail NCCL/ 2>&1 \
        | awk -v cuda="${CUDA_MODULE}" '/NCCL\/.*-CUDA-/ { split($1,a,"-"); split(a[1],b,"/"); ver=b[2]; gsub(/\/.*/, "", cuda); if (ver == cuda) { print $1; exit } }' \
        || true)"
fi
if [[ -n "${NCCL_MODULE}" ]]; then
    module load "${NCCL_MODULE}" || echo "[warn] failed to load NCCL module ${NCCL_MODULE}"
fi

# ---------------------------------------------------------------- preflight
# (Same preflight as the parent sbatch - condensed; see parent for full
# commentary on why each guard exists.)

if [[ ! -d "${REPO_ROOT}/vllm" ]]; then
    echo "[fatal] REPO_ROOT=${REPO_ROOT} does not look like a vllm checkout" >&2
    exit 2
fi
if [[ ! -f "${REPO_ROOT}/examples/eval-moe/moe_expert_stats.py" ]]; then
    echo "[fatal] ${REPO_ROOT}/examples/eval-moe/moe_expert_stats.py missing" >&2
    exit 2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
fi

# Scratch quota guard - keep the same threshold as the parent.
_scratch_used_pct="$(df --output=pcent "${HF_CACHE}" 2>/dev/null | tail -1 | tr -d ' %' || echo 100)"
if [[ "${_scratch_used_pct}" =~ ^[0-9]+$ ]] && [[ "${_scratch_used_pct}" -ge 90 ]]; then
    echo "[fatal] scratch quota >= 90% used (${_scratch_used_pct}%). Clean up before resubmitting." >&2
    exit 7
fi

# Verify Mixtral snapshot is present.
_mixtral_layout=""
for _layout in "${HF_CACHE}/models--mistralai--Mixtral-8x22B-Instruct-v0.1" \
               "${HF_CACHE}/hub/models--mistralai--Mixtral-8x22B-Instruct-v0.1"; do
    if [[ -d "${_layout}" ]] && compgen -G "${_layout}/snapshots/*/*.safetensors" >/dev/null 2>&1; then
        _mixtral_layout="${_layout}"
        break
    fi
done
if [[ -z "${_mixtral_layout}" ]]; then
    echo "[fatal] Mixtral snapshot not found under ${HF_CACHE}/models--... or ${HF_CACHE}/hub/models--..." >&2
    echo "[fatal] HF_HUB_OFFLINE=${HF_HUB_OFFLINE} so the resolver will NOT download." >&2
    exit 6
fi
echo "[info] Mixtral snapshot: ${_mixtral_layout}"

# ---------------------------------------------------------------- install
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[info] creating venv at ${VENV_DIR} ..."
    python3 -m venv "${VENV_DIR}"
fi
export PATH="${VENV_DIR}/bin:${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-0}"
if ! VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED}" uv pip install \
        --python "${VENV_DIR}/bin/python" \
        "numpy" \
        "datasets==4.5.0" \
        -e "${REPO_ROOT}" \
        --torch-backend="${TORCH_BACKEND}" 2>"${VENV_DIR}/install.err"; then
    echo "[fatal] uv pip install failed:" >&2
    sed 's/^/  /' "${VENV_DIR}/install.err" >&2
    exit 4
fi

# Bundle-path setup for CUDA shared libs (same as parent sbatch).
_nvidia_lib_paths=""
_venv_lib_root="$("${VENV_DIR}/bin/python" -c 'import sys, os; p=os.path.dirname(os.path.dirname(sys.executable)); print(os.path.join(p, "lib"))' 2>/dev/null || true)"
if [[ -n "${_venv_lib_root}" && -d "${_venv_lib_root}" ]]; then
    while IFS= read -r _nvidia_lib; do
        if [[ -n "${_nvidia_lib_paths}" ]]; then
            _nvidia_lib_paths="${_nvidia_lib}:${_nvidia_lib_paths}"
        else
            _nvidia_lib_paths="${_nvidia_lib}"
        fi
    done < <(find "${_venv_lib_root}" -maxdepth 5 -type d -path '*/nvidia/*/lib' 2>/dev/null | sort -u)
fi
if [[ -n "${_nvidia_lib_paths}" ]]; then
    export LD_LIBRARY_PATH="${_nvidia_lib_paths}:${LD_LIBRARY_PATH:-}"
    _cudart12="$(find "${_venv_lib_root}" -name 'libcudart.so.12' 2>/dev/null | head -1)"
    if [[ -n "${_cudart12}" ]] && [[ ! -e "${_cudart12%.12}.13" ]]; then
        ln -s "libcudart.so.12" "${_cudart12%.12}.13" || true
    fi
fi
unset _nvidia_lib_paths _venv_lib_root _nvidia_lib _cudart12

if ! "${VENV_DIR}/bin/python" -c \
    "import vllm, numpy, datasets; print(f'vllm={vllm.__version__} numpy={numpy.__version__} datasets={datasets.__version__}')" \
    >"${VENV_DIR}/sanity.out" 2>&1; then
    echo "[fatal] venv sanity failed:" >&2
    sed 's/^/  /' "${VENV_DIR}/sanity.out" >&2
    exit 4
fi
cat "${VENV_DIR}/sanity.out"

# ---------------------------------------------------------------- run
cd "${REPO_ROOT}"

COMMON_ARGS=(
    --benchmarks ${BENCHMARKS}
    --num-samples "${NUM_SAMPLES}"
    --max-tokens "${MAX_TOKENS}"
    --n-shots "${N_SHOTS}"
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEM_UTIL}"
    --output-dir "${OUTPUT_DIR}"
    --seed "${SEED}"
    --dtype "auto"
    --tensor-parallel-size "${NGPUS}"
)

models=()
quants=()
read -ra models <<<"${MODELS_OVERRIDE}"
read -ra quants <<<"${QUANTS_OVERRIDE}"
if [[ ${#quants[@]} -eq 0 ]]; then
    quants=("")
fi

cell_idx=0
for m in "${models[@]}"; do
    for q in "${quants[@]}"; do
        cell_idx=$((cell_idx + 1))
        if [[ -z "${q}" ]]; then
            CELL_ARGS=(--models "${m}" "${COMMON_ARGS[@]}")
        else
            CELL_ARGS=(--models "${m}" --quant "${q}" "${COMMON_ARGS[@]}")
        fi
        echo "============================================================"
        echo "[$(date -Iseconds)] cell ${cell_idx}: ${m}${q:+ @ ${q}} (TP=${NGPUS})"
        echo "============================================================"
        "${VENV_DIR}/bin/python" examples/eval-moe/moe_expert_stats.py "${CELL_ARGS[@]}"
    done
done

echo "[$(date -Iseconds)] all cells complete; tree at ${OUTPUT_DIR}"
