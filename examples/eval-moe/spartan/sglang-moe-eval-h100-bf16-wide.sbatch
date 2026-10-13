#!/usr/bin/env bash
#
# sglang-moe-eval-h100-bf16-wide.sbatch - sglang 0.5.9 + capturer sweep
#
# Full wide sweep of the sglang cu12 backend. For each (model, benchmark)
# cell, runs the canonical examples/eval-moe/moe_expert_stats_sglang.py
# driver against an in-process sglang server and writes per-cell
# `expert_counts.json` in the EXACT schema the vLLM analyzer produces
# (so the sister llama-cpp-eval aggregator's Jaccard toolchain reads
# the sglang tree unmodified).
#
# Models: OLMoE-1B-7B-0924-Instruct + deepseek-moe-16b-chat (BF16 only).
# Benchmarks: mmlu, bbh, humaneval, popqa, include (all 5).
# Samples per benchmark: 999999 (cap; loaders return the whole dataset
# up to that, so this means "the full benchmark").
#
# Target: gpu-h100. 1 GPU. 12 h wall time. The driver fires requests
# to sglang with up to MAX_INFLIGHT in flight at once so sglang's
# continuous-batching engine can saturate the H100. For the smaller
# benchmarks (popqa, include, humaneval) this finishes in seconds;
# mmlu is the long one (~14k prompts × 256 gen tokens).
#
# Reuses venv_sglang_0509 (sglang 0.5.9 + sgl-kernel 0.3.21 cu128 wheel +
# flashinfer-python 0.6.7.post1 + torch 2.9.1+cu128) which has the
# capturer layer_id on-disk patch applied (see handoff §X).
#
# Output tree:
#   <OUTPUT_DIR>/<model_safe>/bf16/moe-<bench>/expert_counts.json
# matches the vLLM analyzer's tree exactly.
#

# ---------------------------------------------------------------- SLURM
#SBATCH --job-name=sglang-moe-h100-wide
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-12:00:00
#SBATCH --output=spartan/logs/sglang-moe-h100-wide-%j.out
#SBATCH --error=spartan/logs/sglang-moe-h100-wide-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pratheek.senevirathne@student.unimelb.edu.au

set -euo pipefail

SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "${SLURM_SUBMIT_DIR}/spartan/logs"

CUDA_MODULE="${CUDA_MODULE:-CUDA/12.4.1}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3}"
GCC_MODULE="${GCC_MODULE:-GCCcore/11.3.0}"
NINJA_MODULE="${NINJA_MODULE:-Ninja/1.11.1-Python-3.11.3}"

SCRATCH_BASE="${SCRATCH_BASE:-${SCRATCH:-/data/scratch/projects/uom00014}/vllm}"
SISTER_VLLM_ROOT="${SISTER_VLLM_ROOT:-/data/gpfs/projects/uom00014/vllm}"
VENV_DIR="${VENV_DIR:-${SCRATCH_BASE}/venv_sglang_0509}"
HF_CACHE="${HF_CACHE:-${SCRATCH_BASE}/hf_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_BASE}/results_sglang_bf16_h100}"

# Two models, BF16 only. Both fit on 1× H100 80 GB at TP=1
# (OLMoE 13 GB, DeepSeek 16B 31 GB).
MODELS_CSV="${MODELS_CSV:-allenai/OLMoE-1B-7B-0924-Instruct,deepseek-ai/deepseek-moe-16b-chat}"
BENCHMARKS="${BENCHMARKS:-mmlu bbh humaneval popqa include}"
NUM_SAMPLES="${NUM_SAMPLES:-999999}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_INFLIGHT="${MAX_INFLIGHT:-64}"
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.92}"
SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S:-600}"
QUANT_TAG="${QUANT_TAG:-bf16}"

mkdir -p "${OUTPUT_DIR}" "${HF_CACHE}"

# ---------------------------------------------------------------- banner
echo "============================================================"
echo "[$(date -Iseconds)] sglang-moe-h100-wide starting"
echo "[$(date -Iseconds)] job_id        = ${SLURM_JOB_ID:-}"
echo "[$(date -Iseconds)] node          = ${SLURM_JOB_NODELIST:-}"
echo "[$(date -Iseconds)] MODELS_CSV    = ${MODELS_CSV}"
echo "[$(date -Iseconds)] BENCHMARKS    = ${BENCHMARKS}"
echo "[$(date -Iseconds)] NUM_SAMPLES   = ${NUM_SAMPLES}"
echo "[$(date -Iseconds)] MAX_TOKENS    = ${MAX_TOKENS}"
echo "[$(date -Iseconds)] MAX_INFLIGHT  = ${MAX_INFLIGHT}"
echo "[$(date -Iseconds)] VENV_DIR      = ${VENV_DIR}"
echo "[$(date -Iseconds)] OUTPUT_DIR    = ${OUTPUT_DIR}"
echo "============================================================"

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

# ---------------------------------------------------------------- venv check
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[fatal] expected venv at ${VENV_DIR}" >&2
    echo "        (the working cu12 install with capturer + sglext plumbing)." >&2
    echo "        Run sglang-cu12-response-check.sbatch once first to build it." >&2
    exit 4
fi
echo "[info] using venv ${VENV_DIR}"

# Verify the capturer layer_id on-disk patch is still in place. If
# someone wiped and rebuilt the venv, this check fails and we abort
# BEFORE starting the sweep (which would silently produce zeros).
PATCH_CHECK=$("${VENV_DIR}/bin/python" -c "
import inspect
from sglang.srt.layers.moe.routed_experts_capturer import _RoutedExpertsDeviceCache
src = inspect.getsource(_RoutedExpertsDeviceCache.capture_fwd_routed_experts)
print('PATCHED' if 'synthesize' in src else 'UNPATCHED')
" 2>&1 | tail -1)
if [[ "${PATCH_CHECK}" != "PATCHED" ]]; then
    echo "[fatal] capturer layer_id fix is NOT on disk in ${VENV_DIR}" >&2
    echo "        Last good venv had this patch applied (handoff §X)." >&2
    echo "        If the venv was rebuilt, re-apply:" >&2
    echo "          python -c 'from pathlib import Path;'" >&2
    echo "          p = Path(\"${VENV_DIR}/lib/python3.11/site-packages/sglang/srt/layers/moe/routed_experts_capturer.py\");'" >&2
    echo "          ... (see EVAL_MOE_HANDOFF.md for the exact patch)" >&2
    exit 5
fi
echo "[info] capturer layer_id fix verified on disk"

# ---------------------------------------------------------------- HF cache
# Symlink Qwen3 snapshot from hub/ to root if needed (matches the vLLM
# workflow; harmless if not present).
if [[ -d "${HF_CACHE}/hub/models--Qwen--Qwen3-30B-A3B" && ! -e "${HF_CACHE}/models--Qwen--Qwen3-30B-A3B" ]]; then
    ln -s "hub/models--Qwen--Qwen3-30B-A3B" "${HF_CACHE}/models--Qwen--Qwen3-30B-A3B"
fi
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}"
export HF_HOME="${HF_CACHE}"

# ---------------------------------------------------------------- server launcher wrapper
# sglang 0.5.9 has the capturer and ServerArgs.enable_return_routed_experts,
# but no CLI flag for it. Patch via a wrapper.
WRAPPER="${OUTPUT_DIR}/launch_server_with_capturer.py"
cat > "${WRAPPER}" << 'PYEOF'
import sys
import warnings
warnings.filterwarnings("ignore")

def _main():
    import sglang.srt.server_args as _sa
    import sglang.srt.entrypoints.http_server as _http  # noqa: F401

    _orig_init = _sa.ServerArgs.__init__
    def _patched_init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        self.enable_return_routed_experts = True
    _sa.ServerArgs.__init__ = _patched_init

    _orig_prepare = _sa.prepare_server_args
    def _patched_prepare(argv):
        args = _orig_prepare(argv)
        args.enable_return_routed_experts = True
        return args
    _sa.prepare_server_args = _patched_prepare

    from sglang.launch_server import run_server
    sa = _sa.prepare_server_args(sys.argv[1:])
    print(f"[wrapper] enable_return_routed_experts = {getattr(sa, 'enable_return_routed_experts', '<not present>')}", flush=True)
    print(f"[wrapper] model-path = {sa.model_path}", flush=True)
    run_server(sa)

# if __name__ guard is REQUIRED: sglang uses multiprocessing.spawn
# which re-imports this file in the worker. Without the guard, the
# worker re-runs run_server() and dies on recursive bootstrap.
if __name__ == '__main__':
    _main()
PYEOF
echo "[info] wrapper written: ${WRAPPER}"

# ---------------------------------------------------------------- sweep loop
# Parse MODELS_CSV into an array
IFS=',' read -r -a MODELS <<< "${MODELS_CSV}"

# Track per-cell timings for the final summary.
declare -A CELL_START CELL_END

# Top-level driver so we can exit cleanly on first failure (so we
# don't burn 12 h producing nothing). The driver itself does NOT
# swallow exceptions; if it crashes on a cell, the wrapper exits
# non-zero and we stop the loop.
set +e
TOTAL_CELLS=$(( ${#MODELS[@]} * $(echo ${BENCHMARKS} | wc -w) ))
CELL_NUM=0
OVERALL_RC=0
for MODEL_ID in "${MODELS[@]}"; do
    for BENCH in ${BENCHMARKS}; do
        CELL_NUM=$((CELL_NUM + 1))
        echo ""
        echo "============================================================"
        echo "[$(date -Iseconds)] CELL ${CELL_NUM}/${TOTAL_CELLS}: model=${MODEL_ID} bench=${BENCH}"
        echo "============================================================"

        # Per-cell output dir (driver will create this itself).
        CELL_OUT="${OUTPUT_DIR}/$(echo ${MODEL_ID} | tr '/' '-')/${QUANT_TAG}/moe-${BENCH}"
        mkdir -p "${CELL_OUT}"

        # Server log goes to a per-cell file so failures are easy to find.
        SERVER_LOG="${OUTPUT_DIR}/server_$(echo ${MODEL_ID} | tr '/' '-')_${BENCH}.log"
        SERVER_PID=""

        cleanup_server() {
            if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
                echo "[cleanup] killing sglang server pid=${SERVER_PID}"
                kill -TERM "${SERVER_PID}" 2>/dev/null || true
                sleep 3
                kill -KILL "${SERVER_PID}" 2>/dev/null || true
                wait "${SERVER_PID}" 2>/dev/null || true
            fi
        }
        trap cleanup_server EXIT

        # Launch server
        echo "[cell] launching sglang server for ${MODEL_ID} (log: ${SERVER_LOG})"
        HUGGINGFACE_HUB_CACHE="${HF_CACHE}" \
        HF_HOME="${HF_CACHE}" \
            "${VENV_DIR}/bin/python" "${WRAPPER}" \
                --model-path "${MODEL_ID}" \
                --port "${PORT}" --host "${HOST}" \
                --mem-fraction-static "${MEM_FRACTION_STATIC}" \
                --context-length "${MAX_MODEL_LEN}" \
                --trust-remote-code \
                --dtype bfloat16 \
                > "${SERVER_LOG}" 2>&1 &
        SERVER_PID=$!

        # Wait for /v1/models to return 200.
        DEADLINE=$(( $(date +%s) + SERVER_TIMEOUT_S ))
        SERVER_OK=0
        while :; do
            if curl -s -o /dev/null -w "%{http_code}" "http://${HOST}:${PORT}/v1/models" 2>/dev/null | grep -q "^200$"; then
                SERVER_OK=1
                break
            fi
            if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
                echo "[fatal] server died before becoming ready. Last 60 lines of log:" >&2
                tail -60 "${SERVER_LOG}" >&2
                OVERALL_RC=1
                break 2
            fi
            if [[ $(date +%s) -ge ${DEADLINE} ]]; then
                echo "[fatal] server did not become ready in ${SERVER_TIMEOUT_S}s. Last 60 lines of log:" >&2
                tail -60 "${SERVER_LOG}" >&2
                OVERALL_RC=1
                break 2
            fi
            sleep 5
        done
        if [[ ${SERVER_OK} -ne 1 ]]; then
            cleanup_server
            continue
        fi
        echo "[cell] sglang server is up"

        # Run the cell. The driver handles per-prompt concurrency via
        # ThreadPoolExecutor(max_workers=MAX_INFLIGHT); sglang batches
        # the requests internally.
        echo "[cell] running driver: ${BENCH} (NUM_SAMPLES=${NUM_SAMPLES}, MAX_INFLIGHT=${MAX_INFLIGHT})"
        CELL_T0=$(date +%s)
        PYTHONPATH="${SISTER_VLLM_ROOT}/examples/eval-moe:${PYTHONPATH:-}" \
        HUGGINGFACE_HUB_CACHE="${HF_CACHE}" \
        HF_HOME="${HF_CACHE}" \
            "${VENV_DIR}/bin/python" \
            "${SISTER_VLLM_ROOT}/examples/eval-moe/moe_expert_stats_sglang.py" \
                --server "http://${HOST}:${PORT}" \
                --model "${MODEL_ID}" \
                --benchmarks "${BENCH}" \
                --num-samples "${NUM_SAMPLES}" \
                --max-tokens "${MAX_TOKENS}" \
                --max-model-len "${MAX_MODEL_LEN}" \
                --output-dir "${OUTPUT_DIR}" \
                --quant-tag "${QUANT_TAG}" \
                --max-inflight "${MAX_INFLIGHT}" 2>&1 | tee -a "${OUTPUT_DIR}/driver_${SLURM_JOB_ID}.log"
        RC=${PIPESTATUS[0]}
        CELL_T1=$(date +%s)
        CELL_ELAPSED=$(( CELL_T1 - CELL_T0 ))
        if [[ ${RC} -ne 0 ]]; then
            echo "[cell] driver FAILED (rc=${RC}, elapsed=${CELL_ELAPSED}s)"
            OVERALL_RC=${RC}
            cleanup_server
            break
        fi
        echo "[cell] driver OK (elapsed=${CELL_ELAPSED}s)"

        # Verify the output JSON was written.
        if [[ -f "${CELL_OUT}/expert_counts.json" ]]; then
            SIZE=$(stat -c %s "${CELL_OUT}/expert_counts.json")
            echo "[cell] wrote ${CELL_OUT}/expert_counts.json (${SIZE} bytes)"
        else
            echo "[cell] WARNING: ${CELL_OUT}/expert_counts.json not found after driver exit"
        fi

        cleanup_server
        # Disable the per-cell cleanup (we want the trap to be a no-op
        # for the next cell's setup). Re-arm for the next iteration.
        trap - EXIT
    done
done
set -e

# ---------------------------------------------------------------- summary
echo ""
echo "============================================================"
echo "[$(date -Iseconds)] sweep complete"
echo "============================================================"
echo "[summary] expert_counts files written:"
find "${OUTPUT_DIR}" -name expert_counts.json | sort
echo ""
echo "[summary] per-file sizes:"
find "${OUTPUT_DIR}" -name expert_counts.json -exec ls -la {} \; 2>/dev/null
echo ""
echo "[summary] exit code: ${OVERALL_RC}"
echo "[$(date -Iseconds)] done"