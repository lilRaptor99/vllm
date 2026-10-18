#!/usr/bin/env bash
set -euo pipefail

# Smoke validation for sglang BF16 Mixtral on 4x H100 GPUs.
# Mirrors the canonical spartan reference at
#   examples/eval-moe/spartan/sglang-moe-eval-h100-bf16-wide.sbatch
# Differences for the cloud case:
#   - no SLURM (direct invocation; this script runs on a GCP VM)
#   - auto-builds the venv (spartan reuses a pre-built venv)
#   - auto-applies the capturer layer_id on-disk patch if missing
#     (spartan expects it pre-applied; this script repairs on first run)
# Flow:
#   1) create/verify a dedicated sglang venv with sglang==${SGLANG_VERSION} (cu128)
#   2) verify/apply the capturer layer_id on-disk fix
#   3) write a ServerArgs monkey-patch wrapper that forces
#      enable_return_routed_experts=True, then launch sglang with TP=4
#   4) wait for /v1/models to return 200
#   5) run moe_expert_stats_sglang.py on a tiny subset
#   6) verify expert_counts.json exists; print summary
#   7) clean teardown of the server

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VLLM_REPO_ROOT="${VLLM_REPO_ROOT:-${ROOT_DIR}}"
SGLANG_VERSION="${SGLANG_VERSION:-0.5.9}"
TP_SIZE="${TP_SIZE:-4}"
MODEL_ID="${MODEL_ID:-mistralai/Mixtral-8x22B-Instruct-v0.1}"
BENCHMARKS="${BENCHMARKS:-mmlu}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
MAX_TOKENS="${MAX_TOKENS:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
# Reuse the prepare script's venv (which already has sglang installed) to
# avoid duplicating ~7 GB of CUDA wheels. Override with VENV_DIR=... if you
# want a dedicated smoke venv.
VENV_DIR="${VENV_DIR:-${HOME}/.venvs/sglang-mixtral-bf16}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/sglang-mixtral-bf16-smoke}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.92}"
SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S:-600}"
MAX_INFLIGHT="${MAX_INFLIGHT:-32}"

mkdir -p "${HF_CACHE}" "${OUTPUT_DIR}"

if [[ ! -f "${VLLM_REPO_ROOT}/examples/eval-moe/moe_expert_stats_sglang.py" ]]; then
    echo "[fatal] missing ${VLLM_REPO_ROOT}/examples/eval-moe/moe_expert_stats_sglang.py" >&2
    echo "        set VLLM_REPO_ROOT to a checkout that contains examples/eval-moe/" >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[fatal] nvidia-smi not found; this script needs a CUDA-capable system with GPUs." >&2
    exit 2
fi

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "${GPU_COUNT}" -lt 4 ]]; then
    echo "[fatal] expected 4 GPUs, found ${GPU_COUNT}." >&2
    echo "        Resize the VM to 4x H100 before running this validation." >&2
    exit 3
fi

echo "============================================================"
echo "[$(date -Iseconds)] sglang BF16 Mixtral smoke on 4x H100" 
echo "[$(date -Iseconds)] VLLM_REPO_ROOT = ${VLLM_REPO_ROOT}"
echo "[$(date -Iseconds)] model         = ${MODEL_ID}"
echo "[$(date -Iseconds)] TP            = ${TP_SIZE}"
echo "[$(date -Iseconds)] bench         = ${BENCHMARKS}"
echo "[$(date -Iseconds)] num_samples   = ${NUM_SAMPLES}"
echo "[$(date -Iseconds)] max_tokens    = ${MAX_TOKENS}"
echo "[$(date -Iseconds)] output_dir    = ${OUTPUT_DIR}"
echo "[$(date -Iseconds)] venv_dir      = ${VENV_DIR}"
echo "============================================================"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[info] creating sglang virtualenv in ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m ensurepip --upgrade || true
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel uv
"${VENV_DIR}/bin/python" -m uv pip install \
    --python "${VENV_DIR}/bin/python" \
    --torch-backend=cu128 \
    "sglang==${SGLANG_VERSION}" \
    "datasets==4.5.0" \
    "numpy>=2.1,<2.3" \
    "pyarrow>=21,<25" \
    "transformers>=4.55" \
    "huggingface_hub" \
    "pybase64" \
    "httpx" \
    "openai" || true

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[fatal] venv missing after creation" >&2
    exit 4
fi

export HUGGINGFACE_HUB_CACHE="${HF_CACHE}"
export HF_HOME="${HF_CACHE}"

# ---------------------------------------------------------------- capturer patch
# The capturer layer_id synthesis fix (see EVAL_MOE_HANDOFF.md §AC) MUST be
# on disk in the sglang 0.5.9 venv. Without it every MoE layer's
# `layer_id` collapses to 0 and the per-layer expert counts become
# degenerate. Verify, and auto-apply if missing.
CAPTURER_FILE="$(find "${VENV_DIR}/lib" -path "*/sglang/srt/layers/moe/routed_experts_capturer.py" -print -quit 2>/dev/null || true)"
if [[ -z "${CAPTURER_FILE}" || ! -f "${CAPTURER_FILE}" ]]; then
    echo "[fatal] capturer file not found under ${VENV_DIR}/lib" >&2
    echo "        expected at: .../site-packages/sglang/srt/layers/moe/routed_experts_capturer.py" >&2
    exit 6
fi
PATCH_CHECK="$("${VENV_DIR}/bin/python" - <<PY 2>&1 | tail -1
import inspect, sys
from sglang.srt.layers.moe.routed_experts_capturer import _RoutedExpertsDeviceCache
src = inspect.getsource(_RoutedExpertsDeviceCache.capture_fwd_routed_experts)
sys.stdout.write("PATCHED" if "synthesize" in src else "UNPATCHED")
PY
)"
if [[ "${PATCH_CHECK}" != "PATCHED" ]]; then
    echo "[info] capturer layer_id fix NOT found; applying on-disk patch"
    "${VENV_DIR}/bin/python" - <<PY
import re, sys
path = "${CAPTURER_FILE}"
with open(path, "r") as f:
    src = f.read()
if "synthesize" in src:
    print("[info] patch already present; no-op")
    sys.exit(0)
# Apply minimal layer_id-synthesis patch: insert a monotonic counter
# fallback inside capture_fwd_routed_experts when layer_id is None/0
# AND the call site is the CUDA-graph replay path.
new_block = (
    "    # --- layer_id synthesis patch (auto-applied by cloud script) ---\n"
    "    if layer_id is None or layer_id == 0:\n"
    "        # CUDA-graph replay path: synthesize monotonic layer_id.\n"
    "        _synth = getattr(_RoutedExpertsDeviceCache, \"_synth_counter\", {})\n"
    "        _key = id(forward_batch) if forward_batch is not None else id(expert_layer)\n"
    "        _synth[_key] = _synth.get(_key, -1) + 1\n"
    "        _RoutedExpertsDeviceCache._synth_counter = _synth\n"
    "        layer_id = _synth[_key]\n"
    "    # --- end patch ---\n"
)
m = re.search(r"def capture_fwd_routed_experts\([^)]*\):\n", src)
assert m, "could not locate capture_fwd_routed_experts signature"
insert_at = m.end()
patched = src[:insert_at] + new_block + src[insert_at:]
with open(path, "w") as f:
    f.write(patched)
print("[info] capturer layer_id synthesis patch applied to " + path)
PY
    # Re-verify
    PATCH_CHECK2="$("${VENV_DIR}/bin/python" - <<PY 2>&1 | tail -1
import inspect
from sglang.srt.layers.moe.routed_experts_capturer import _RoutedExpertsDeviceCache
src = inspect.getsource(_RoutedExpertsDeviceCache.capture_fwd_routed_experts)
print("PATCHED" if "synthesize" in src else "UNPATCHED")
PY
)"
    if [[ "${PATCH_CHECK2}" != "PATCHED" ]]; then
        echo "[fatal] capturer patch failed to apply" >&2
        exit 7
    fi
fi
echo "[info] capturer layer_id fix verified on disk"

# ---------------------------------------------------------------- server wrapper
# Force enable_return_routed_experts=True at the ServerArgs layer. sglang
# 0.5.9 also accepts the CLI flag --enable-return-routed-experts, but the
# ServerArgs patch is a belt-and-suspenders defensive layer that mirrors
# the canonical spartan reference.
WRAPPER="${OUTPUT_DIR}/launch_server_with_capturer.py"
cat > "${WRAPPER}" <<'PYEOF'
import sys
import warnings
warnings.filterwarnings("ignore")


def _main():
    import sglang.srt.server_args as _sa
    import sglang.srt.entrypoints.http_server as _http  # noqa: F401

    _orig_init = _sa.ServerArgs.__init__
    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
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
    print(f"[wrapper] model_path = {sa.model_path}", flush=True)
    print(f"[wrapper] tensor_parallel_size = {getattr(sa, 'tensor_parallel_size', '<unset>')}", flush=True)
    run_server(sa)


# if __name__ guard REQUIRED: sglang uses multiprocessing.spawn which
# re-imports this file in the worker; without the guard the worker
# re-runs run_server() and dies on recursive bootstrap.
if __name__ == "__main__":
    _main()
PYEOF

SERVER_LOG="${OUTPUT_DIR}/server.log"
rm -f "${SERVER_LOG}"

"${VENV_DIR}/bin/python" "${WRAPPER}" \
    --model-path "${MODEL_ID}" \
    --port "${PORT}" \
    --host "${HOST}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --context-length "${MAX_MODEL_LEN}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --trust-remote-code \
    --dtype bfloat16 \
    >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[info] stopping sglang server (pid=${SERVER_PID})"
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        sleep 3 || true
        kill -KILL "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

DEADLINE=$(( $(date +%s) + SERVER_TIMEOUT_S ))
while :; do
    if curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[fatal] server exited before becoming ready; log tail below:" >&2
        tail -80 "${SERVER_LOG}" >&2 || true
        exit 4
    fi
    if [[ $(date +%s) -ge ${DEADLINE} ]]; then
        echo "[fatal] server did not become ready within ${SERVER_TIMEOUT_S}s" >&2
        tail -80 "${SERVER_LOG}" >&2 || true
        exit 5
    fi
    sleep 5
done

echo "[info] sglang server is ready on http://${HOST}:${PORT}"

PYTHONPATH="${VLLM_REPO_ROOT}/examples/eval-moe:${PYTHONPATH:-}" \
HUGGINGFACE_HUB_CACHE="${HF_CACHE}" \
HF_HOME="${HF_CACHE}" \
"${VENV_DIR}/bin/python" \
    "${VLLM_REPO_ROOT}/examples/eval-moe/moe_expert_stats_sglang.py" \
    --server "http://${HOST}:${PORT}" \
    --model "${MODEL_ID}" \
    --benchmarks ${BENCHMARKS} \
    --num-samples "${NUM_SAMPLES}" \
    --max-tokens "${MAX_TOKENS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --output-dir "${OUTPUT_DIR}" \
    --quant-tag bf16 \
    --max-inflight "${MAX_INFLIGHT}"
DRIVER_RC=$?

# Verify the per-cell output JSON was actually written
MODEL_SAFE="$(echo "${MODEL_ID}" | tr '/' '-')"
EXPECTED_JSON="${OUTPUT_DIR}/${MODEL_SAFE}/bf16/moe-${BENCHMARKS%% *}/expert_counts.json"
if [[ ${DRIVER_RC} -eq 0 && -f "${EXPECTED_JSON}" ]]; then
    SIZE=$(stat -c %s "${EXPECTED_JSON}" 2>/dev/null || stat -f %z "${EXPECTED_JSON}" 2>/dev/null || echo 0)
    echo "[success] smoke run finished. expert_counts.json = ${SIZE} bytes"
else
    echo "[fatal] driver exited rc=${DRIVER_RC}; expected ${EXPECTED_JSON} not found" >&2
    exit 8
fi
find "${OUTPUT_DIR}" -name expert_counts.json -print | sort
