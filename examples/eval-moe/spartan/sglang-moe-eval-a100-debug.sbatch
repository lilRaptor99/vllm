#!/usr/bin/env bash
#
# sglang-moe-eval-a100-debug.sbatch
#
# Interactive-style debug sbatch for the sglang cu13 wall.
# Runs each step, prints the outcome, then sleeps N seconds so
# the human can tail the log and intervene if needed.
#
# Strategy: install sglang 0.5.20+ from source with --no-deps,
# pin flashinfer-python to the vllm-venv's 0.6.17 (cu128),
# bypass cuda-python>=13.0 by skipping it (sglang imports it
# lazily only for some kernels we don't use for routing capture).
#
# This is a "best effort to break the cu13 wall" attempt.
#
# ---------------------------------------------------------------- SLURM
#SBATCH --job-name=sglang-debug
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=spartan/logs/sglang-debug-%j.out
#SBATCH --error=spartan/logs/sglang-debug-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pratheek.senevirathne@student.unimelb.edu.au

set -euo pipefail
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "${SLURM_SUBMIT_DIR}/spartan/logs"

CUDA_MODULE="${CUDA_MODULE:-CUDA/12.4.1}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3}"
GCC_MODULE="${GCC_MODULE:-GCCcore/11.3.0}"
NINJA_MODULE="${NINJA_MODULE:-Ninja/1.11.1-Python-3.11.3}"

REPO_ROOT="${REPO_ROOT:-/data/gpfs/projects/uom00014/sglang/python}"
SISTER_VLLM_ROOT="${SISTER_VLLM_ROOT:-/data/gpfs/projects/uom00014/vllm}"
SCRATCH_BASE="${SCRATCH_BASE:-${SCRATCH:-/data/scratch/projects/uom00014}/vllm}"

VENV_DIR="${VENV_DIR:-${SCRATCH_BASE}/venv_sglang_debug}"
DATASETS_DIR="${DATASETS_DIR:-${SCRATCH_BASE}/datasets}"
HF_CACHE="${HF_CACHE:-${SCRATCH_BASE}/hf_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_BASE}/results_debug_sglang}"

UV_CACHE_DIR="${UV_CACHE_DIR:-${SISTER_VLLM_ROOT}/.uv_cache}"
export UV_CACHE_DIR
UV_BUILD_PARALLEL="${UV_BUILD_PARALLEL:-8}"
export UV_BUILD_PARALLEL

MODEL_ID="${MODEL_ID:-allenai/OLMoE-1B-7B-0924-Instruct}"
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
PAUSE="${PAUSE:-0}"  # seconds to sleep between phases (set 5-10 for "interactive")

mkdir -p "${DATASETS_DIR}" "${OUTPUT_DIR}" "${HF_CACHE}" "${VENV_DIR}" "${UV_CACHE_DIR}"

# Helper: print a phase banner and optionally pause.
phase() {
    echo ""
    echo "============================================================"
    echo "[$(date -Iseconds)] PHASE: $1"
    echo "============================================================"
    if [[ "${PAUSE}" -gt 0 ]]; then
        sleep "${PAUSE}"
    fi
}

# ---------------------------------------------------------------- modules
set +u
module purge
load_module_or_die() {
    local name="$1"
    local stderr_log
    stderr_log="$(mktemp)"
    if ! module load "${name}" 2>"${stderr_log}"; then
        echo "[fatal] module load ${name} failed:" >&2
        sed 's/^/  /' "${stderr_log}" >&2
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
export LD_LIBRARY_PATH="${EBROOTPYTHON:-/usr/lib}/lib:${LD_LIBRARY_PATH:-}"
echo "[info] torch baseline: $(python3 -c 'import torch; print(torch.__version__, torch.version.cuda, "is_available="+str(torch.cuda.is_available()))' 2>&1 | tail -1)"

phase "BUILD VENV (cu129 torch + sglang --no-deps + manual cu128 flashinfer)"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    python3 -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/pip" install --upgrade pip wheel uv

    echo "[info] torch==2.13.0+cu129 (matches vllm-venv) ..."
    UV_CACHE_DIR="${UV_CACHE_DIR}" \
        uv pip install --python "${VENV_DIR}/bin/python" \
            --torch-backend cu129 \
            "torch==2.13.0" \
            "torchvision==0.28.0+cu129" \
            "torchaudio==2.11.0+cu129" \
            "numpy<2.5"

    echo "[info] sglang runtime deps (no cuda-python>=13, no cu13 markers) ..."
    UV_CACHE_DIR="${UV_CACHE_DIR}" \
        uv pip install --python "${VENV_DIR}/bin/python" \
            "openai==2.6.1" \
            "aiohttp" \
            "pyzmq" \
            "transformers==5.15.1" \
            "tokenizers==0.23.2" \
            "huggingface-hub" \
            "datasets" \
            "pybase64" \
            "compressed-tensors" \
            "fastapi" \
            "uvicorn" \
            "ninja" \
            "interegular" \
            "pillow" \
            "prometheus-client" \
            "partial_json_parser" \
            "orjson" \
            "outlines==0.1.11" \
            "packaging" \
            "requests" \
            "rich" \
            "tiktoken" \
            "torch_memory_saver>=0.0.9.post1" \
            "llguidance<2.0.0,>=1.7.6" \
            "IPython"

    # The CRITICAL pin: flashinfer-python 0.6.17 (cu128, what vllm-venv has).
    # sglang 0.5.20+ wants 0.6.18[cu13]. If 0.6.17 is ABI-compat (likely
    # only a few new APIs differ), this gets us past the cu13 wall. If not,
    # the server will error at startup and we'll know.
    echo "[info] flashinfer-python 0.6.17 (cu128, from PyPI) ..."
    "${VENV_DIR}/bin/pip" install "flashinfer-python==0.6.17"

    # sglang 0.5.20+ from PyPI without deps.
    echo "[info] sglang==0.5.20 --no-deps ..."
    "${VENV_DIR}/bin/pip" install "sglang==0.5.20" --no-deps
else
    echo "[info] reusing ${VENV_DIR}"
fi

phase "VERIFY DEPS"
"${VENV_DIR}/bin/python" -c "
import torch
print('torch       :', torch.__version__)
print('cuda avail  :', torch.cuda.is_available())
print('cuda device :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
print('cuda runtime:', torch.version.cuda)
import flashinfer
print('flashinfer  :', getattr(flashinfer, '__version__', 'unknown'))
import sglang
import importlib.metadata
print('sglang wheel:', importlib.metadata.version('sglang'))
"

phase "LAUNCH sglang server (timeout 5 min)"
SERVER_LOG="${OUTPUT_DIR}/sglang-server.log"
HUGGINGFACE_HUB_CACHE="${HF_CACHE}" \
HF_HOME="${HF_CACHE}" \
    "${VENV_DIR}/bin/python" -m sglang.launch_server \
        --model-path "${MODEL_ID}" \
        --enable-return-routed-experts \
        --port "${PORT}" --host "${HOST}" \
        --mem-fraction-static 0.85 \
        --context-length 4096 \
        --trust-remote-code \
        > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "[info] server pid = ${SERVER_PID}"

cleanup() {
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        sleep 2
        kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Wait up to 5 min for /v1/models.
DEADLINE=$(( $(date +%s) + 300 ))
while :; do
    if curl -s -o /dev/null -w "%{http_code}" "http://${HOST}:${PORT}/v1/models" 2>/dev/null | grep -q "^200$"; then
        echo "[info] sglang server is up"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[fatal] sglang server died. Tail of log:"
        tail -60 "${SERVER_LOG}"
        exit 4
    fi
    if [[ $(date +%s) -ge ${DEADLINE} ]]; then
        echo "[fatal] sglang server did not become ready in 5 min. Last 80 lines:"
        tail -80 "${SERVER_LOG}"
        exit 4
    fi
    sleep 5
done

phase "HIT /v1/chat/completions WITH return_routed_experts"
curl -s "http://${HOST}:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL_ID}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2?\"}],
        \"max_tokens\": 16,
        \"temperature\": 0.0,
        \"return_routed_experts\": true
    }" | head -c 4000
echo ""

phase "DONE"
echo "Server log tail:"
tail -40 "${SERVER_LOG}"
