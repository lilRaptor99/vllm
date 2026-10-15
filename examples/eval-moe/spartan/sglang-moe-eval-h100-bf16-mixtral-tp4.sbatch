#!/usr/bin/env bash
#
# spartan/sglang-moe-eval-h100-bf16-mixtral-tp4.sbatch - SLURM job that
# runs JUST the Mixtral-8x22B-Instruct-v0.1 cell of the sglang bf16
# wide sweep at TP=4 on gpu-h100.
#
# Forked from spartan/sglang-moe-eval-h100-bf16-wide.sbatch on
# 2026-10-16 to add the Mixtral TP=4 cell (the wide sweep hardcodes
# --gres=gpu:1 which is insufficient for the 262 GB bf16 Mixtral
# snapshot). Mirrors the vLLM Mixtral TP=4 precedent (job 31516295,
# the only prior Mixtral success):
#   - TP=1: doesn't fit (bf16 weights alone are 262 GB; H100 has 80 GB)
#   - TP=2: weight load itself exhausts reserved memory
#   - TP=4: per-GPU bf16 weights = 65 GB; ample KV cache headroom on 80 GB
#
# Why a separate sbatch instead of editing the wide sweep:
#   - The wide sweep hardcodes --gres=gpu:1 in the #SBATCH directive
#     and the comment above explains why it can't use a shell variable.
#     Editing it to gpu:4 would re-do the already-cached cells (OLMoE,
#     DeepSeek, Qwen3) which the analyzer would skip on the JSON-exists
#     check, but the engine load + warmup per cell is ~50-90s of pure
#     waste when the goal is just to retry Mixtral.
#   - This Mixtral-only TP=4 sbatch is one model, 5 cells, one engine
#     load per cell, ~4-6 h total wall time.
#
# IMPORTANT - Mirror the wide sweep's env-var contract exactly so the
# per-cell JSONs are Jaccard-comparable with the other bf16 cells
# already on scratch (jobs 32969746, 33009563):
#   NUM_SAMPLES, MAX_TOKENS, MAX_MODEL_LEN, BENCHMARKS, SEED,
#   OUTPUT_DIR all default to the same values as the wide sweep.
#
# Output tree:
#   ${SCRATCH_BASE}/results_sglang_bf16_h100/mistralai--Mixtral-8x22B-Instruct-v0.1/bf16/{moe-mmlu,moe-bbh,moe-humaneval,moe-popqa,moe-include}/expert_counts.json
# (the SAME path the wide sweep would write to, so the analyzer's
# "skip if output exists" check still works for a hypothetical second
# TP=4 retry.)

# ---------------------------------------------------------------- SLURM
#SBATCH --job-name=sglang-moe-h100-bf16-mixtral-tp4
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=0-12:00:00
#SBATCH --output=/data/gpfs/projects/uom00014/vllm/examples/eval-moe/spartan/logs/sglang-moe-eval-%j.out
#SBATCH --error=/data/gpfs/projects/uom00014/vllm/examples/eval-moe/spartan/logs/sglang-moe-eval-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pratheek.senevirathne@student.unimelb.edu.au

set -euo pipefail

SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="/data/gpfs/projects/uom00014/vllm/examples/eval-moe/spartan/logs"
mkdir -p "${LOG_DIR}"

CUDA_MODULE="${CUDA_MODULE:-CUDA/12.4.1}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3}"
GCC_MODULE="${GCC_MODULE:-GCCcore/11.3.0}"
NINJA_MODULE="${NINJA_MODULE:-Ninja/1.11.1-Python-3.11.3}"
# NCCL is required for TP>1 (sglang tensor-parallel needs cross-GPU
# collectives just like vLLM). Auto-load if pinned.
NCCL_MODULE="${NCCL_MODULE:-NCCL/2.22.3-CUDA-12.4.1}"
NGPUS=4

SCRATCH_BASE="${SCRATCH_BASE:-${SCRATCH:-/data/scratch/projects/uom00014}/vllm}"
SISTER_VLLM_ROOT="${SISTER_VLLM_ROOT:-/data/gpfs/projects/uom00014/vllm}"
VENV_DIR="${VENV_DIR:-${SCRATCH_BASE}/venv_sglang_0509}"
HF_CACHE="${HF_CACHE:-${SCRATCH_BASE}/hf_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_BASE}/results_sglang_bf16_h100}"

# Cell config - pinned to Mixtral only.
MODELS_CSV="${MODELS_CSV:-mistralai/Mixtral-8x22B-Instruct-v0.1}"
BENCHMARKS="${BENCHMARKS:-mmlu bbh humaneval popqa include}"
NUM_SAMPLES="${NUM_SAMPLES:-999999}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_INFLIGHT="${MAX_INFLIGHT:-64}"
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
# Per-GPU weights = 65 GB at TP=4; reserve ~24% for KV cache + activations
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.92}"
SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S:-900}"
QUANT_TAG="${QUANT_TAG:-bf16}"

mkdir -p "${OUTPUT_DIR}" "${HF_CACHE}"

# ---------------------------------------------------------------- banner
echo "============================================================"
echo "[$(date -Iseconds)] sglang-moe-h100-bf16-mixtral-tp4 starting"
echo "[$(date -Iseconds)] job_id        = ${SLURM_JOB_ID:-}"
echo "[$(date -Iseconds)] node          = ${SLURM_JOB_NODELIST:-}"
echo "[$(date -Iseconds)] MODELS_CSV    = ${MODELS_CSV}"
echo "[$(date -Iseconds)] BENCHMARKS    = ${BENCHMARKS}"
echo "[$(date -Iseconds)] NUM_SAMPLES   = ${NUM_SAMPLES}"
echo "[$(date -Iseconds)] MAX_TOKENS    = ${MAX_TOKENS}"
echo "[$(date -Iseconds)] MAX_INFLIGHT  = ${MAX_INFLIGHT}"
echo "[$(date -Iseconds)] NGPUS         = ${NGPUS}"
echo "[$(date -Iseconds)] MEM_FRACTION_STATIC = ${MEM_FRACTION_STATIC}"
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
if [[ -z "${NCCL_MODULE}" ]]; then
    NCCL_MODULE="$(module avail NCCL/ 2>&1 \
        | awk -v cuda="${CUDA_MODULE}" '/NCCL\/.*-CUDA-/ { split($1,a,"-"); split(a[1],b,"/"); ver=b[2]; gsub(/\/.*/, "", cuda); if (ver == cuda) { print $1; exit } }' \
        || true)"
fi
if [[ -n "${NCCL_MODULE}" ]]; then
    load_module_or_die "${NCCL_MODULE}"
fi
module list
set -u
export LD_LIBRARY_PATH="${EBROOTPYTHON:-/usr/lib}/lib:${LD_LIBRARY_PATH:-}"

# ---------------------------------------------------------------- preflight
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
    exit 5
fi
echo "[info] capturer layer_id fix verified on disk"

# Verify Mixtral snapshot is present at one of the two layouts.
_mixtral_layout=""
for _layout in \
        "${HF_CACHE}/models--mistralai--Mixtral-8x22B-Instruct-v0.1" \
        "${HF_CACHE}/hub/models--mistralai--Mixtral-8x22B-Instruct-v0.1"; do
    if [[ -d "${_layout}" ]] && compgen -G "${_layout}/snapshots/*/*.safetensors" >/dev/null 2>&1; then
        _mixtral_layout="${_layout}"
        break
    fi
done
if [[ -z "${_mixtral_layout}" ]]; then
    echo "[fatal] Mixtral snapshot not found under ${HF_CACHE}/models--... or ${HF_CACHE}/hub/models--..." >&2
    echo "[fatal] Login is firewalled from HF Hub; pre-stage the snapshot on scratch." >&2
    exit 6
fi
echo "[info] Mixtral snapshot: ${_mixtral_layout}"

# Scratch quota guard (>=90% = abort).
_scratch_used_pct="$(df --output=pcent "${HF_CACHE}" 2>/dev/null | tail -1 | tr -d ' %' || echo 100)"
if [[ "${_scratch_used_pct}" =~ ^[0-9]+$ ]] && [[ "${_scratch_used_pct}" -ge 90 ]]; then
    echo "[fatal] scratch quota >= 90% used (${_scratch_used_pct}%). Clean up before resubmitting." >&2
    exit 7
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
fi

# ---------------------------------------------------------------- HF cache
if [[ -d "${HF_CACHE}/hub/models--Qwen--Qwen3-30B-A3B" && ! -e "${HF_CACHE}/models--Qwen--Qwen3-30B-A3B" ]]; then
    ln -s "hub/models--Qwen--Qwen3-30B-A3B" "${HF_CACHE}/models--Qwen--Qwen3-30B-A3B"
fi
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

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
    print(f"[wrapper] tensor_parallel_size = {getattr(sa, 'tensor_parallel_size', '<unset>')}", flush=True)
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
        echo "[$(date -Iseconds)] CELL ${CELL_NUM}/${TOTAL_CELLS}: model=${MODEL_ID} bench=${BENCH} (TP=${NGPUS})"
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

        # Launch server with TP=4 (the only difference from the wide sweep).
        echo "[cell] launching sglang server for ${MODEL_ID} (TP=${NGPUS}, log: ${SERVER_LOG})"
        HUGGINGFACE_HUB_CACHE="${HF_CACHE}" \
        HF_HOME="${HF_CACHE}" \
        HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
            "${VENV_DIR}/bin/python" "${WRAPPER}" \
                --model-path "${MODEL_ID}" \
                --port "${PORT}" --host "${HOST}" \
                --mem-fraction-static "${MEM_FRACTION_STATIC}" \
                --context-length "${MAX_MODEL_LEN}" \
                --tensor-parallel-size "${NGPUS}" \
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
        echo "[cell] sglang server is up (TP=${NGPUS})"

        # Run the cell. The driver handles per-prompt concurrency via
        # ThreadPoolExecutor(max_workers=MAX_INFLIGHT); sglang batches
        # the requests internally.
        echo "[cell] running driver: ${BENCH} (NUM_SAMPLES=${NUM_SAMPLES}, MAX_INFLIGHT=${MAX_INFLIGHT})"
        CELL_T0=$(date +%s)
        PYTHONPATH="${SISTER_VLLM_ROOT}/examples/eval-moe:${PYTHONPATH:-}" \
        HUGGINGFACE_HUB_CACHE="${HF_CACHE}" \
        HF_HOME="${HF_CACHE}" \
        HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
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