#!/usr/bin/env bash
#
# spartan/sglang-moe-eval-a100-smoke.sbatch - SLURM smoke job that builds
# sglang, launches it with --enable-return-routed-experts, runs a tiny
# MoE expert-activation capture (1 model, 1 benchmark, ~8 prompts) via
# examples/eval-moe/moe_expert_stats_sglang.py, and writes the resulting
# expert_counts.json in the same schema as the vLLM analyzer
# (examples/eval-moe/moe_expert_stats.py) so the sister-repo aggregator
# (llama-cpp-eval/examples/eval-moe-humaneval/heatmap_from_cpp.py and
# friends) can read it unmodified.
#
# Target: gpu-a100-short. 2 nodes in the partition, queue is short
# (a couple of hours even on busy days). a100-short has a 4-day wall
# cap and a 123 GB / job memory limit; we ask for 64 GB which is
# plenty for sglang + OLMoE (13 GB) + a small client venv.
#
# This is a smoke job. Pass overrides via --export=... on the
# sbatch command line (see bottom of this file) to switch the
# model/benchmark/sample count.
#
# ---------------------------------------------------------------- SLURM
#SBATCH --job-name=sglang-moe-smoke
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=spartan/logs/sglang-moe-smoke-%j.out
#SBATCH --error=spartan/logs/sglang-moe-smoke-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pratheek.senevirathne@student.unimelb.edu.au

set -euo pipefail

# ---------------------------------------------------------------- paths
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "${SLURM_SUBMIT_DIR}/spartan/logs"

CUDA_MODULE="${CUDA_MODULE:-CUDA/12.4.1}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3}"
GCC_MODULE="${GCC_MODULE:-GCCcore/11.3.0}"
NINJA_MODULE="${NINJA_MODULE:-Ninja/1.11.1-Python-3.11.3}"
TORCH_BACKEND="${TORCH_BACKEND:-cu129}"

# sglang's pyproject.toml + setup.py live under python/ (the
# repo root is a Cargo-workspace parent).
REPO_ROOT="${REPO_ROOT:-/data/gpfs/projects/uom00014/sglang/python}"
# Sibling vllm repo where moe_expert_stats.py + the dataset helpers live.
# The driver imports them via sys.path injection; this MUST point at the
# vllm checkout that has the eval-moe dir.
SISTER_VLLM_ROOT="${SISTER_VLLM_ROOT:-/data/gpfs/projects/uom00014/vllm}"
SCRATCH_BASE="${SCRATCH_BASE:-${SCRATCH:-/data/scratch/projects/uom00014}/vllm}"

# Fresh venv for sglang (separate from the vLLM venv at ${SCRATCH}/venv
# so the two toolchains don't fight over torch 2.13 vs 2.14).
VENV_DIR="${VENV_DIR:-${SCRATCH_BASE}/venv_sglang}"
DATASETS_DIR="${DATASETS_DIR:-${SCRATCH_BASE}/datasets}"
HF_CACHE="${HF_CACHE:-${SCRATCH_BASE}/hf_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_BASE}/results_smoke_sglang}"

# Pin the uv wheel cache to the project GPFS volume so the sglang
# build doesn't blow past the scratch quota (~10-15 GB of wheels).
UV_CACHE_DIR="${UV_CACHE_DIR:-${SISTER_VLLM_ROOT}/.uv_cache}"
export UV_CACHE_DIR
UV_BUILD_PARALLEL="${UV_BUILD_PARALLEL:-8}"
export UV_BUILD_PARALLEL

# Smoke defaults: tiny so we can verify end-to-end in <30 min.
# Override via --export=... on the sbatch line.
MODEL_ID="${MODEL_ID:-allenai/OLMoE-1B-7B-0924-Instruct}"
BENCHMARKS="${BENCHMARKS:-mmlu}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
MAX_TOKENS="${MAX_TOKENS:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.92}"
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S:-600}"

mkdir -p "${DATASETS_DIR}" "${OUTPUT_DIR}" "${HF_CACHE}" "${VENV_DIR}" "${UV_CACHE_DIR}"

# ---------------------------------------------------------------- banner
echo "============================================================"
echo "[$(date -Iseconds)] sglang-moe-smoke starting"
echo "[$(date -Iseconds)] job_id        = ${SLURM_JOB_ID:-}"
echo "[$(date -Iseconds)] node          = ${SLURM_JOB_NODELIST:-}"
echo "[$(date -Iseconds)] cpus          = ${SLURM_CPUS_PER_TASK:-}"
echo "[$(date -Iseconds)] gpus          = ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[$(date -Iseconds)] REPO_ROOT     = ${REPO_ROOT}"
echo "[$(date -Iseconds)] SISTER_VLLM_ROOT = ${SISTER_VLLM_ROOT}"
echo "[$(date -Iseconds)] SCRATCH_BASE  = ${SCRATCH_BASE}"
echo "[$(date -Iseconds)] VENV_DIR      = ${VENV_DIR}"
echo "[$(date -Iseconds)] HF_HOME       = ${HF_CACHE}"
echo "[$(date -Iseconds)] OUTPUT_DIR    = ${OUTPUT_DIR}"
echo "[$(date -Iseconds)] MODEL_ID      = ${MODEL_ID}"
echo "[$(date -Iseconds)] BENCHMARKS    = ${BENCHMARKS}"
echo "[$(date -Iseconds)] NUM_SAMPLES   = ${NUM_SAMPLES}"
echo "[$(date -Iseconds)] MAX_TOKENS    = ${MAX_TOKENS}"
echo "[$(date -Iseconds)] MAX_MODEL_LEN = ${MAX_MODEL_LEN}"
echo "[$(date -Iseconds)] MEM_FRACTION_STATIC = ${MEM_FRACTION_STATIC}"
echo "[$(date -Iseconds)] CUDA_MODULE   = ${CUDA_MODULE}"
echo "[$(date -Iseconds)] PYTHON_MODULE = ${PYTHON_MODULE}"
echo "[$(date -Iseconds)] GCC_MODULE    = ${GCC_MODULE}"
echo "[$(date -Iseconds)] NINJA_MODULE  = ${NINJA_MODULE}"
echo "[$(date -Iseconds)] TORCH_BACKEND = ${TORCH_BACKEND}"
echo "[$(date -Iseconds)] PORT          = ${PORT}"
echo "============================================================"

# ---------------------------------------------------------------- modules
if [[ -z "${GCC_MODULE}" || -z "${PYTHON_MODULE}" || -z "${CUDA_MODULE}" ]]; then
    echo "[fatal] GCC_MODULE / PYTHON_MODULE / CUDA_MODULE must all be set" >&2
    exit 3
fi

set +u
module purge
load_module_or_die() {
    local name="$1"
    local stderr_log
    stderr_log="$(mktemp)"
    if ! module load "${name}" 2>"${stderr_log}"; then
        echo "[fatal] module load ${name} failed:" >&2
        sed 's/^/  /' "${stderr_log}" >&2
        echo "[fatal] Hint: run 'module spider ${name}' on a gpu-a100-short node." >&2
        rm -f "${stderr_log}"
        exit 3
    fi
    rm -f "${stderr_log}"
}
load_module_or_die "${GCC_MODULE}"
load_module_or_die "${PYTHON_MODULE}"
load_module_or_die "${CUDA_MODULE}"
load_module_or_die "${NINJA_MODULE}"
module list
set -u

# Post-load sanity.
if ! command -v python3 >/dev/null 2>&1; then
    echo "[fatal] python3 not on PATH after loading ${PYTHON_MODULE}" >&2
    exit 3
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "[warn] nvcc not on PATH; sglang's CUDA-extension build may fail"
fi
if ! command -v ninja >/dev/null 2>&1; then
    echo "[warn] ninja not on PATH after loading ${NINJA_MODULE}; sglang's JIT builds may fail"
fi
echo "[info] python : $(python3 --version)"
echo "[info] gcc    : $(gcc --version | head -1)"
echo "[info] nvcc   : $(nvcc --version 2>/dev/null | tail -1 || echo '<not on PATH>')"
echo "[info] ninja  : $(ninja --version 2>/dev/null || echo '<not on PATH>')"

# Make sure C++ sees the loaded GCC and CUDA toolkit is on PATH.
export CC="${CC:-$(command -v gcc)}"
export CXX="${CXX:-$(command -v g++)}"
if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${CUDA_HOME:-/usr/local/cuda}/lib64"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Make the module-loaded Python's libpython visible to the venv
# (the sglang venv's binaries link against
# ${EBROOTPYTHON}/lib/libpython3.11.so.1.0, which is NOT on the
# default login LD_LIBRARY_PATH). Without this, `python -V` from
# inside the venv errors with "libpython3.11.so.1.0: cannot
# open shared object file".
if [[ -n "${EBROOTPYTHON:-}" ]]; then
    export LD_LIBRARY_PATH="${EBROOTPYTHON}/lib:${LD_LIBRARY_PATH}"
fi

# ---------------------------------------------------------------- HF cache
# Pre-existing snapshots live at ${HF_HOME}/hub/models--<org>--<name>/
# (the layout used when HUGGINGFACE_HUB_CACHE was unset at download
# time). New downloads (if any) land at ${HF_HOME}/models--<org>--<name}/.
# Symlink the Qwen3 snapshot if needed (matches the vLLM workflow
# documented in the project handoff).
if [[ -d "${HF_CACHE}/hub/models--Qwen--Qwen3-30B-A3B" && ! -e "${HF_CACHE}/models--Qwen--Qwen3-30B-A3B" ]]; then
    ln -s "hub/models--Qwen--Qwen3-30B-A3B" "${HF_CACHE}/models--Qwen--Qwen3-30B-A3B"
    echo "[info] symlinked Qwen3 snapshot under ${HF_CACHE}"
fi
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}"
export HF_HOME="${HF_CACHE}"

# ---------------------------------------------------------------- venv
# Build sglang in a fresh venv. Two paths:
#
# (A) install from PyPI's manylinux wheel (default, fast). sglang
#     0.5.21 ships `sglang-0.5.21-cp311-cp311-manylinux_2_34_x86_64.whl`
#     with the Rust extensions pre-built, so we DON'T need a
#     Rust toolchain on PATH and DON'T need to compile from source.
#
# (B) editable install from the local checkout (SGLANG_FROM_SOURCE=1).
#     Falls back to source only if (A) fails (e.g. wheel compat break).
#     Requires `Rust/1.78.0-Python-3.11.3` or `Rust/1.87.0` on PATH
#     (a Lmod module) — see sglang's pyproject.toml which pins
#     `setuptools-rust>=1.11`.
#
# We pin torch to the cu129 wheel (same index the vLLM venv uses) to
# keep the runtime forward-compatible with the CUDA 12.4.1 driver on
# Spartan. The cu129 wheel's `torch.version.cuda` is 12.9 — sglang
# 0.5.21 wants cu128 but the runtime is forward-compatible.
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[info] creating sglang venv at ${VENV_DIR} ..."
    python3 -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/pip" install --upgrade pip wheel uv

    # Pin torch to the BASE 2.13.0 wheel (cu128, no +cu129 suffix)
    # NOT the cu129 wheel. Reason: sglang 0.5.21's deps resolve
    # to torch 2.13.0 (cu128) and force a `torch==2.13.0` (cu130) +
    # `torchvision==cu129` mismatch unless we ourselves pin cu128.
    # The 12.4.1 driver on Spartan is forward-compat with cu128
    # (driver 12080 supports up through CUDA 13.0 user-space APIs).
    # We do NOT pre-install torchvision: sglang's text-only MoE
    # models don't need it, and pre-installing it triggers the
    # cu128/cu129/cu130 mismatch.
    echo "[info] installing torch (base cu128 wheel) + sglang's other deps ..."
    UV_CACHE_DIR="${UV_CACHE_DIR}" \
        uv pip install --python "${VENV_DIR}/bin/python" \
            "torch==2.13.0" \
            "numpy>=2.0.0,<=2.5.3" \
            "huggingface-hub" \
            "openai" \
            "pillow" \
            "aiohttp" \
            "pyzmq" \
            "interegular" \
            "ninja"

    if [[ "${SGLANG_FROM_SOURCE:-0}" == "1" ]]; then
        # Source path. The Rust ext compiles against the module
        # CUDA via setuptools-rust. Need Rust on PATH.
        load_module_or_die "Rust/1.78.0-Python-3.11.3"
        load_module_or_die "setuptools-rust/1.6.0-Python-3.11.3"
        echo "[info] installing sglang from ${REPO_ROOT} (Rust ext will compile) ..."
        cd "${REPO_ROOT}"
        UV_CACHE_DIR="${UV_CACHE_DIR}" \
            uv pip install --python "${VENV_DIR}/bin/python" \
                -e . 2>&1 | tail -40
    else
        # Wheel path (default). Pin sglang's version explicitly.
        # sglang 0.5.21 is the most recent with the capturer API
        # the driver was written against.
        SGLANG_VERSION="${SGLANG_VERSION:-0.5.21}"
        echo "[info] installing sglang==${SGLANG_VERSION} from PyPI (precompiled wheel) ..."
        # sglang's wheel will NOT downgrade our torch (we just
        # pinned torch==2.13.0 above, and sglang asks for the
        # same). The uv resolver will keep our cu128 torch.
        UV_CACHE_DIR="${UV_CACHE_DIR}" \
            uv pip install --python "${VENV_DIR}/bin/python" \
                "sglang==${SGLANG_VERSION}" 2>&1 | tail -50
    fi
else
    echo "[info] reusing existing venv at ${VENV_DIR}"
fi

# Sanity: torch + cuda runtime visible. The wheel's torch is
# plain `torch==2.13.0` (cu128). With driver 12.8 (Spartan
# gpu-a100 nodes), torch's `cuda.is_available()` should return
# True and `torch.cuda.get_device_name(0)` should report the GPU.
# If `cuda avail  : False` shows up here, the wheel is using a
# cu130 torch that needs a newer driver - in that case pin
# SGLANG_TORCH_INDEX to the cu128 index explicitly.
"${VENV_DIR}/bin/python" -c "
import torch
print('torch       :', torch.__version__)
print('cuda avail  :', torch.cuda.is_available())
print('cuda device :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
print('cuda runtime:', torch.version.cuda)
print('cuda archs  :', torch.cuda.get_arch_list())
"
# Verify sglang importable + check the wheel's version.
"${VENV_DIR}/bin/python" -c "
import sglang
from sglang import version as v
print('sglang      :', v.__version__ if hasattr(v, '__version__') else 'unknown')
import importlib.metadata
print('sglang wheel:', importlib.metadata.version('sglang'))
"

# ---------------------------------------------------------------- server
SERVER_LOG="${OUTPUT_DIR}/sglang-server.log"
echo "[info] launching sglang server in background, log -> ${SERVER_LOG}"
HUGGINGFACE_HUB_CACHE="${HF_CACHE}" \
HF_HOME="${HF_CACHE}" \
    "${VENV_DIR}/bin/python" -m sglang.launch_server \
        --model "${MODEL_ID}" \
        --enable-return-routed-experts \
        --port "${PORT}" --host "${HOST}" \
        --mem-fraction-static "${MEM_FRACTION_STATIC}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --trust-remote-code \
        --dtype bfloat16 \
        > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "[info] server pid = ${SERVER_PID}"

# Trap to ensure the server is killed on exit (any reason).
cleanup() {
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[info] killing sglang server pid ${SERVER_PID} ..."
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        sleep 2
        kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Wait for /v1/models to return 200.
echo "[info] polling http://${HOST}:${PORT}/v1/models (timeout ${SERVER_TIMEOUT_S}s) ..."
deadline=$(( $(date +%s) + SERVER_TIMEOUT_S ))
while :; do
    if curl -s -o /dev/null -w "%{http_code}" \
            "http://${HOST}:${PORT}/v1/models" 2>/dev/null \
            | grep -q "^200$"; then
        echo "[info] sglang server is up"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[fatal] sglang server process died before becoming ready. Last 60 lines of log:" >&2
        tail -60 "${SERVER_LOG}" >&2
        exit 4
    fi
    if [[ $(date +%s) -ge ${deadline} ]]; then
        echo "[fatal] sglang server did not become ready in ${SERVER_TIMEOUT_S}s. Last 60 lines of log:" >&2
        tail -60 "${SERVER_LOG}" >&2
        exit 4
    fi
    sleep 5
done

# Sanity dump of the server's own model listing.
echo "[info] /v1/models response:"
curl -s "http://${HOST}:${PORT}/v1/models" | head -c 500
echo ""

# ---------------------------------------------------------------- driver
echo "[info] running the sglang analyzer driver ..."
PYTHONPATH="${SISTER_VLLM_ROOT}/examples/eval-moe:${PYTHONPATH:-}" \
HUGGINGFACE_HUB_CACHE="${HF_CACHE}" \
HF_HOME="${HF_CACHE}" \
    "${VENV_DIR}/bin/python" \
    "${SISTER_VLLM_ROOT}/examples/eval-moe/moe_expert_stats_sglang.py" \
    --server "http://${HOST}:${PORT}" \
    --model "${MODEL_ID}" \
    --benchmarks ${BENCHMARKS} \
    --num-samples "${NUM_SAMPLES}" \
    --max-tokens "${MAX_TOKENS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-prompts-per-bench "${NUM_SAMPLES}" \
    --output-dir "${OUTPUT_DIR}" \
    --quant-tag bf16

# ---------------------------------------------------------------- summary
echo "[info] result tree at ${OUTPUT_DIR}:"
find "${OUTPUT_DIR}" -type f | sort
echo ""
echo "[info] expert_counts.json (head, first 40 lines of any):"
find "${OUTPUT_DIR}" -name expert_counts.json -print -exec head -40 {} \; -exec echo "..." \; 2>/dev/null | head -80
echo ""
echo "[info] sglang server tail (last 20 lines):"
tail -20 "${SERVER_LOG}"

echo "[$(date -Iseconds)] sglang-moe-smoke done"
