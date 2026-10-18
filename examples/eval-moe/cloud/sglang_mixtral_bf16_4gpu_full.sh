#!/usr/bin/env bash
set -euo pipefail

# Full BF16 Mixtral sweep on 4x H100 GPUs.
# This is the follow-up to the smoke run once the validation is confirmed.
# Mirrors the canonical spartan reference at
#   examples/eval-moe/spartan/sglang-moe-eval-h100-bf16-wide.sbatch

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VLLM_REPO_ROOT="${VLLM_REPO_ROOT:-${ROOT_DIR}}"
SGLANG_VERSION="${SGLANG_VERSION:-0.5.9}"
TP_SIZE="${TP_SIZE:-4}"
MODEL_ID="${MODEL_ID:-mistralai/Mixtral-8x22B-Instruct-v0.1}"
BENCHMARKS="${BENCHMARKS:-mmlu bbh humaneval popqa include}"
NUM_SAMPLES="${NUM_SAMPLES:-999999}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
# Reuse the smoke/prepare venv to avoid duplicating ~7 GB of CUDA wheels.
VENV_DIR="${VENV_DIR:-${HOME}/.venvs/sglang-mixtral-bf16}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/sglang-mixtral-bf16-full}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.92}"
SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S:-900}"
MAX_INFLIGHT="${MAX_INFLIGHT:-64}"

mkdir -p "${HF_CACHE}" "${OUTPUT_DIR}"

if [[ ! -f "${VLLM_REPO_ROOT}/examples/eval-moe/moe_expert_stats_sglang.py" ]]; then
    echo "[fatal] missing ${VLLM_REPO_ROOT}/examples/eval-moe/moe_expert_stats_sglang.py" >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[fatal] nvidia-smi not found; this needs a 4-GPU CUDA VM." >&2
    exit 2
fi
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "${GPU_COUNT}" -lt 4 ]]; then
    echo "[fatal] expected 4 GPUs, found ${GPU_COUNT}." >&2
    exit 3
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
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

export HUGGINGFACE_HUB_CACHE="${HF_CACHE}"
export HF_HOME="${HF_CACHE}"

# Pre-download the datasets to the HF cache so the driver doesn't hit
# network on each cell. The driver loaders (load_mmlu_prompts etc.)
# cache under HF_DATASETS_CACHE which defaults to ${HF_HOME}/datasets.
echo "[info] pre-downloading benchmark datasets to ${HF_HOME}/datasets"
HF_DATASETS_CACHE="${HF_HOME}/datasets" \
"${VENV_DIR}/bin/python" - <<'PY' || echo "[warn] dataset pre-cache step failed; will retry at driver runtime"
import os
from datasets import load_dataset
# Each tuple: (hf_repo, subset_or_None, config_or_None)
items = [
    ("cais/mmlu", "all", None),
    ("lukaemon/bbh", None, None),
    ("openai_humaneval", None, None),
    ("akariasai/PopQA", None, None),
    ("allenai/instructeval", None, "include"),
]
for spec in items:
    repo, sub, conf = spec
    try:
        kwargs = {"cache_dir": os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))}
        if sub is not None:
            kwargs["name"] = sub
        if conf is not None:
            kwargs["name"] = conf
        load_dataset(repo, **kwargs)
        print(f"[datasets] cached {repo}")
    except Exception as e:
        print(f"[datasets] FAILED to cache {repo}: {e}")
PY

# ---------------------------------------------------------------- capturer patch
CAPTURER_FILE="$(find "${VENV_DIR}/lib" -path "*/sglang/srt/layers/moe/routed_experts_capturer.py" -print -quit 2>/dev/null || true)"
if [[ -z "${CAPTURER_FILE}" || ! -f "${CAPTURER_FILE}" ]]; then
    echo "[fatal] capturer file not found under ${VENV_DIR}/lib" >&2
    exit 6
fi
PATCH_CHECK="$("${VENV_DIR}/bin/python" - <<PY 2>&1 | tail -1
import sys
from sglang.srt.layers.moe.routed_experts_capturer import _RoutedExpertsDeviceCache
import inspect
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
new_block = (
    "    # --- layer_id synthesis patch (auto-applied by cloud script) ---\n"
    "    if layer_id is None or layer_id == 0:\n"
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
fi
echo "[info] capturer layer_id fix verified on disk"

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


if __name__ == "__main__":
    _main()
PYEOF

OVERALL_RC=0
for BENCH in ${BENCHMARKS}; do
    echo "============================================================"
    echo "[$(date -Iseconds)] starting benchmark=${BENCH}"
    echo "============================================================"

    SERVER_LOG="${OUTPUT_DIR}/server_${BENCH}.log"
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

    # Pre-flight: clear any stale listener on PORT from a prior cell so
    # the new server can bind. Best-effort; warn rather than fail.
    if command -v ss >/dev/null 2>&1; then
        if ss -ltn "( sport = :${PORT} )" 2>/dev/null | grep -q LISTEN; then
            echo "[warn] port ${PORT} already in use; trying anyway"
        fi
    fi

    cleanup() {
        if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
            kill -TERM "${SERVER_PID}" 2>/dev/null || true
            sleep 3 || true
            kill -KILL "${SERVER_PID}" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT

    DEADLINE=$(( $(date +%s) + SERVER_TIMEOUT_S ))
    while :; do
        if curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
            break
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "[fatal] server exited before becoming ready" >&2
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

    echo "[info] server ready; running benchmark ${BENCH}"
    PYTHONPATH="${VLLM_REPO_ROOT}/examples/eval-moe:${PYTHONPATH:-}" \
    HUGGINGFACE_HUB_CACHE="${HF_CACHE}" \
    HF_HOME="${HF_CACHE}" \
    HF_DATASETS_CACHE="${HF_HOME}/datasets" \
    "${VENV_DIR}/bin/python" \
        "${VLLM_REPO_ROOT}/examples/eval-moe/moe_expert_stats_sglang.py" \
        --server "http://${HOST}:${PORT}" \
        --model "${MODEL_ID}" \
        --benchmarks "${BENCH}" \
        --num-samples "${NUM_SAMPLES}" \
        --max-tokens "${MAX_TOKENS}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --output-dir "${OUTPUT_DIR}" \
        --quant-tag bf16 \
        --max-inflight "${MAX_INFLIGHT}"
    DRIVER_RC=$?
    if [[ ${DRIVER_RC} -ne 0 ]]; then
        echo "[error] driver for ${BENCH} exited rc=${DRIVER_RC}"
        OVERALL_RC=${DRIVER_RC}
    fi

    cleanup
    trap - EXIT

    # Per-cell verification: confirm the expert_counts.json was actually
    # written. If the driver reported success but the file is missing,
    # that's a silent failure mode — bail out.
    MODEL_SAFE="$(echo "${MODEL_ID}" | tr '/' '-')"
    CELL_JSON="${OUTPUT_DIR}/${MODEL_SAFE}/bf16/moe-${BENCH}/expert_counts.json"
    if [[ -f "${CELL_JSON}" ]]; then
        SIZE=$(stat -c %s "${CELL_JSON}" 2>/dev/null || stat -f %z "${CELL_JSON}" 2>/dev/null || echo 0)
        echo "[info] benchmark ${BENCH} complete; ${SIZE} bytes -> ${CELL_JSON}"
    else
        echo "[error] benchmark ${BENCH} did not write ${CELL_JSON}" >&2
        OVERALL_RC=9
    fi
done

echo "[success] full BF16 Mixtral sweep finished under ${OUTPUT_DIR}"
find "${OUTPUT_DIR}" -name expert_counts.json -print | sort
echo "[summary] exit code: ${OVERALL_RC}"
exit ${OVERALL_RC}
