#!/usr/bin/env bash
#
# spartan/vllm-moe-eval-h100-w4a16-prequant-mixtral-triton.sbatch
#
# ONE-OFF rerun of the Mixtral W4A16 cells with --moe-backend triton to
# bypass the MarlinExperts kernel that produces degenerate routed-experts
# capture data on H100 (see EVAL_MOE_HANDOFF.md addendum 9 — the W4A16
# AWQ/GPTQ JSONs on scratch at results_wide_w4a16_awq_h100/ for Mixtral
# show 67-69% zeros across the (layer, expert) marginal bins because the
# capturer is bound to BaseRouter but the actual topk_ids coming back are
# deterministic per-layer pairs). TritonWNA16Experts goes through the same
# BaseRouter capture_fn path that TritonExperts uses for W8A16 (which
# produces healthy data on the same model).
#
# Only Mixtral is touched (OLMoE/Qwen3/DeepSeek W4A16 cells are fine).
# Mixtral needs TP=4 because the bf16 snapshot is 262 GB (the W4A16
# prequant is 69 GB per recipe, so it fits in 80 GB at TP=1, but the
# analyzer needs a fresh LLM per cell and re-using the TP=4 engine from
# the previous Mixtral W8A16 TP=4 run makes the wall-time estimate
# straightforward).
#
# Pre-reqs (same as vllm-moe-eval-h100-w4a16-prequant.sbatch):
#   - ${SCRATCH_BASE}/hf_cache_awq/mistralai--Mixtral-8x22B-Instruct-v0.1-{awq,gptq}/
#     exist on scratch (69 GB each, ~138 GB total)
#   - ${SCRATCH_BASE}/datasets/{mmlu,bbh,humaneval,popqa,include}/ are populated
#   - ${SCRATCH_BASE}/venv/ already has vllm + datasets (vLLM >= the
#     commit that adds --moe-backend kwarg support, currently HEAD)
#
# What it does:
#   - Removes the 10 stale Mixtral W4A16 expert_counts.json files so
#     the idempotent skip doesn't skip them
#   - Runs 2 cells (Mixtral AWQ + GPTQ) at TP=4 with --moe-backend triton
#   - 4 h wall time (the prior Marlin run took ~1h09m per cell for
#     Mixtral TP=4; Triton is typically 1.5-2x slower for WNA16, so
#     ~3-4 h per cell is the worst case, fits in 4 h with margin)
#
# Quick start:
#   cd /data/gpfs/projects/uom00014/vllm
#   sbatch examples/eval-moe/spartan/vllm-moe-eval-h100-w4a16-prequant-mixtral-triton.sbatch

# ---------------------------------------------------------------- SLURM
#SBATCH --job-name=vllm-moe-w4a16-mixtral-triton
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=0-04:00:00
#SBATCH --output=/data/gpfs/projects/uom00014/vllm/examples/eval-moe/spartan/logs/vllm-moe-eval-w4a16-prequant-mixtral-triton-%j.out
#SBATCH --error=/data/gpfs/projects/uom00014/vllm/examples/eval-moe/spartan/logs/vllm-moe-eval-w4a16-prequant-mixtral-triton-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pratheek.senevirathne@student.unimelb.edu.au

set -euo pipefail

SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="/data/gpfs/projects/uom00014/vllm/examples/eval-moe/spartan/logs"
mkdir -p "${LOG_DIR}"

CUDA_MODULE="${CUDA_MODULE:-CUDA/12.4.1}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3}"
GCC_MODULE="${GCC_MODULE:-GCCcore/11.3.0}"
TORCH_BACKEND="${TORCH_BACKEND:-cu129}"
NCCL_MODULE="${NCCL_MODULE:-NCCL/2.22.3-CUDA-12.4.1}"
NINJA_MODULE="${NINJA_MODULE:-Ninja/1.11.1-Python-3.11.3}"
NGPUS=4

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

# Cell config: ONLY Mixtral. Both AWQ and GPTQ recipes.
MODELS_OVERRIDE="${MODELS_OVERRIDE:-${SCRATCH_BASE}/hf_cache_awq/mistralai--Mixtral-8x22B-Instruct-v0.1-awq ${SCRATCH_BASE}/hf_cache_awq/mistralai--Mixtral-8x22B-Instruct-v0.1-gptq}"
QUANTS_OVERRIDE="${QUANTS_OVERRIDE:-compressed-tensors}"
BENCHMARKS="${BENCHMARKS:-mmlu bbh humaneval popqa include}"
NUM_SAMPLES="${NUM_SAMPLES:-999999}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
N_SHOTS="${N_SHOTS:-5}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_BASE}/results_wide_w4a16_awq_h100}"
SEED="${SEED:-0}"
SKIP_PYTHON_PLOTS="${SKIP_PYTHON_PLOTS:-1}"
SKIP_DATASET_CHECK="${SKIP_DATASET_CHECK:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-}"
# Force the Triton MoE backend to skip MarlinExperts — see sbatch header.
MOE_BACKEND="${MOE_BACKEND:-triton}"

export HF_HOME="${HF_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HUB_OFFLINE

# ---------------------------------------------------------------- banner
echo "============================================================"
echo "[$(date -Iseconds)] vllm-moe-w4a16-mixtral-triton starting"
echo "[$(date -Iseconds)] job_id        = ${SLURM_JOB_ID:-}"
echo "[$(date -Iseconds)] node          = ${SLURM_JOB_NODELIST:-}"
echo "[$(date -Iseconds)] MODELS        = ${MODELS_OVERRIDE}"
echo "[$(date -Iseconds)] QUANTS        = ${QUANTS_OVERRIDE}"
echo "[$(date -Iseconds)] BENCHMARKS    = ${BENCHMARKS}"
echo "[$(date -Iseconds)] NUM_SAMPLES   = ${NUM_SAMPLES}"
echo "[$(date -Iseconds)] MAX_TOKENS    = ${MAX_TOKENS}"
echo "[$(date -Iseconds)] NGPUS         = ${NGPUS}"
echo "[$(date -Iseconds)] GPU_MEM_UTIL  = ${GPU_MEM_UTIL}"
echo "[$(date -Iseconds)] MOE_BACKEND   = ${MOE_BACKEND}"
echo "[$(date -Iseconds)] VENV_DIR      = ${VENV_DIR}"
echo "[$(date -Iseconds)] OUTPUT_DIR    = ${OUTPUT_DIR}"
echo "============================================================"

# ---------------------------------------------------------------- modules
module purge
module load "${GCC_MODULE}"
module load "${PYTHON_MODULE}"
module load "${CUDA_MODULE}"
module load "${NINJA_MODULE}"
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
if [[ ! -d "${REPO_ROOT}/vllm" ]]; then
    echo "[fatal] REPO_ROOT=${REPO_ROOT} does not look like a vllm checkout" >&2
    exit 2
fi
if [[ ! -f "${REPO_ROOT}/examples/eval-moe/moe_expert_stats.py" ]]; then
    echo "[fatal] ${REPO_ROOT}/examples/eval-moe/moe_expert_stats.py missing" >&2
    exit 2
fi
# Confirm the analyzer has the --moe-backend flag (the patch landed in
# this repo on 2026-10-16; if you're running this on an older tree the
# flag won't exist and the analyzer will silently fall back to "auto").
if ! "${VENV_DIR}/bin/python" "${REPO_ROOT}/examples/eval-moe/moe_expert_stats.py" --help 2>&1 | grep -q -- "--moe-backend"; then
    echo "[fatal] moe_expert_stats.py does not accept --moe-backend." >&2
    echo "        Pull the patch that adds the --moe-backend argparse flag," >&2
    echo "        or apply it from the working tree." >&2
    exit 8
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
fi

# Scratch quota guard (>=90% = abort).
_scratch_used_pct="$(df --output=pcent "${HF_CACHE}" 2>/dev/null | tail -1 | tr -d ' %' || echo 100)"
if [[ "${_scratch_used_pct}" =~ ^[0-9]+$ ]] && [[ "${_scratch_used_pct}" -ge 90 ]]; then
    echo "[fatal] scratch quota >= 90% used (${_scratch_used_pct}%). Clean up before resubmitting." >&2
    exit 7
fi

# Verify Mixtral snapshots exist at one of the two layouts.
for m in ${MODELS_OVERRIDE}; do
    _layout=""
    for _try in \
            "${m}/snapshots" \
            "${m}"; do
        if compgen -G "${_try}/*.safetensors" >/dev/null 2>&1; then
            _layout="${_try}"
            break
        fi
    done
    if [[ -z "${_layout}" ]]; then
        echo "[fatal] ${m}: no safetensors at root or under snapshots/" >&2
        exit 6
    fi
    echo "[info] ${m}: safetensors at ${_layout}"
done

# Wipe the 10 stale Mixtral W4A16 expert_counts.json files so the
# analyzer's idempotent skip doesn't skip them. The prequant ckpts are
# unchanged — only the broken JSON outputs are removed.
for m in ${MODELS_OVERRIDE}; do
    safe=$(basename "${m}")
    cell_dir="${OUTPUT_DIR}/--data--scratch--projects--uom00014--vllm--hf_cache_awq--${safe}/compressed-tensors"
    if [[ -d "${cell_dir}" ]]; then
        echo "[info] wiping stale JSONs under ${cell_dir}"
        rm -fv "${cell_dir}"/moe-*/expert_counts.json
    fi
done

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
cd "${REPO_ROOT}/examples/eval-moe"

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
    --moe-backend "${MOE_BACKEND}"
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
        echo "[$(date -Iseconds)] cell ${cell_idx}: ${m}${q:+ @ ${q}} (TP=${NGPUS}, moe_backend=${MOE_BACKEND})"
        echo "============================================================"
        "${VENV_DIR}/bin/python" moe_expert_stats.py "${CELL_ARGS[@]}"
    done
done

echo "[$(date -Iseconds)] all cells complete; tree at ${OUTPUT_DIR}"