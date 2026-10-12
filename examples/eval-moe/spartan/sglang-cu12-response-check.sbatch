#!/usr/bin/env bash
#
# sglang-cu12-response-check.sbatch
#
# The Oct 11 session left the sglang cu12 backend in this state:
#  - venv_sglang_0509 installs cleanly on cu12 driver (sglang 0.5.9 +
#    sgl-kernel 0.3.21 cu128 wheel + flashinfer 0.6.7.post1 + torch
#    2.9.1+cu128 + a torchvision shim).
#  - The capturer imports OK.
#  - The server starts with --enable-return-routed-experts via the
#    monkey-patch (sglang 0.5.9 doesn't expose that flag on the CLI).
#  - The OpenAI chat response comes back successfully but DOES NOT
#    contain a routed_experts field at the place the test was looking.
#
# Goal of this job: dump the FULL non-streaming response from /v1/chat/completions
# to disk and print the top-level keys + the contents of `sglext` so we can
# see whether the capturer output landed at the response level, the choice
# level, or got dropped entirely. Once we know where it is, the driver code
# in examples/eval-moe/moe_expert_stats_sglang.py gets a one-line fix.
#
# This is the "find the right JSON path" debug step. Reuses the
# already-built venv at /data/scratch/projects/uom00014/vllm/venv_sglang_0509
# (last good Oct 11 install, job 32714323).
#

#SBATCH --job-name=sglang-cu12-rspchk
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=spartan/logs/sglang-cu12-rspchk-%j.out
#SBATCH --error=spartan/logs/sglang-cu12-rspchk-%j.err
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
VENV_DIR="${VENV_DIR:-${SCRATCH_BASE}/venv_sglang_0509}"
HF_CACHE="${HF_CACHE:-${SCRATCH_BASE}/hf_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_BASE}/results_debug_sglang_0509}"

MODEL_ID="${MODEL_ID:-allenai/OLMoE-1B-7B-0924-Instruct}"
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
MAX_TOKENS="${MAX_TOKENS:-32}"
SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S:-600}"

mkdir -p "${OUTPUT_DIR}" "${HF_CACHE}"

echo "============================================================"
echo "[$(date -Iseconds)] sglang-cu12-rspchk starting"
echo "[$(date -Iseconds)] job_id        = ${SLURM_JOB_ID:-}"
echo "[$(date -Iseconds)] node          = ${SLURM_JOB_NODELIST:-}"
echo "[$(date -Iseconds)] MODEL_ID      = ${MODEL_ID}"
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
    echo "[fatal] expected venv at ${VENV_DIR} (last good cu12 install)."
    echo "        Run sbatch with the build script first, or set VENV_DIR."
    exit 4
fi
echo "[info] using venv ${VENV_DIR}"
"${VENV_DIR}/bin/python" -c "
import torch, sglang
print('torch       :', torch.__version__)
print('cuda avail  :', torch.cuda.is_available())
print('cuda device :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
print('cuda runtime:', torch.version.cuda)
print('sglang      :', sglang.__version__ if hasattr(sglang, '__version__') else 'no __version__')
try:
    from sglang.srt.layers.moe.routed_experts_capturer import get_global_experts_capturer, _global_expert_capturer
    print('capturer    : imported, global type =', type(_global_expert_capturer).__name__)
except Exception as e:
    print('capturer    : import FAILED:', e)
"

# ---------------------------------------------------------------- server launcher
# sglang 0.5.9 has the capturer and ServerArgs.enable_return_routed_experts,
# but no CLI flag for it. Patch the dataclass default via a tiny wrapper.
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

    # ----- capturer plumbing: handled by on-disk patch -----
    # The sglang 0.5.9 capturer at
    #   venv_sglang_0509/lib/python3.11/site-packages/sglang/
    #     srt/layers/moe/routed_experts_capturer.py
    # has a band-aid `layer_id = layer_id if layer_id is not None else 0`
    # that collapses all layers to layer 0 for models whose TopK doesn't
    # thread layer_id through (OLMoE, Mixtral). We patched this file
    # directly: when layer_id is None, we synthesize one via a monotonic
    # per-device-cache counter that resets at num_hidden_layers. This
    # is process-independent (it's the actual capturer code), so the
    # fix applies in BOTH the parent process and the scheduler worker.
    print("[wrapper] capturer layer_id fix is on-disk in venv_sglang_0509", flush=True)
    print("[wrapper] (parent-process monkey-patches do not propagate to", flush=True)
    print("[wrapper]  sglang's multiprocessing.spawn workers; on-disk is.", flush=True)

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

SERVER_LOG="${OUTPUT_DIR}/sglang-server.log"
echo "[info] launching server (logging to ${SERVER_LOG})"
HUGGINGFACE_HUB_CACHE="${HF_CACHE}" \
HF_HOME="${HF_CACHE}" \
    "${VENV_DIR}/bin/python" "${WRAPPER}" \
        --model-path "${MODEL_ID}" \
        --port "${PORT}" --host "${HOST}" \
        --mem-fraction-static "${MEM_FRACTION_STATIC}" \
        --context-length "${CONTEXT_LENGTH}" \
        --trust-remote-code \
        --dtype bfloat16 \
        > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "[info] server pid = ${SERVER_PID}"

cleanup() {
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[cleanup] TERM server pid=${SERVER_PID}"
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        sleep 2
        kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# wait for /v1/models
DEADLINE=$(( $(date +%s) + SERVER_TIMEOUT_S ))
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
        echo "[fatal] sglang server did not become ready in ${SERVER_TIMEOUT_S}s. Last 80 lines:"
        tail -80 "${SERVER_LOG}"
        exit 4
    fi
    sleep 5
done

# Print the server-config-flag dump in log evidence (verifies monkey-patch landed)
echo ""
echo "=== Server process /v1/models snapshot ==="
curl -s "http://${HOST}:${PORT}/v1/models" | python3 -m json.tool | head -40

# ---------------------------------------------------------------- response check
export RESP_DUMP="${OUTPUT_DIR}/response_${SLURM_JOB_ID}.json"
export RESP_DECODED="${OUTPUT_DIR}/response_decoded_${SLURM_JOB_ID}.json"

cat > "${OUTPUT_DIR}/hit_endpoint.py" << 'PYEOF'
import base64
import json
import os
import sys
import urllib.request

host = os.environ["HOST"]
port = os.environ["PORT"]
model = os.environ["MODEL_ID"]
max_tokens = int(os.environ.get("MAX_TOKENS", "32"))
dump_file = os.environ["RESP_DUMP"]
decoded_file = os.environ["RESP_DECODED"]

# Three prompts: math, code, factual. Vary length to see if any of them
# triggers the capturer to populate the response.
prompts = [
    [{"role": "user", "content": "What is 2+2?"}],
    [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number."}],
    [{"role": "user", "content": "The capital of France is"}],
]
results = []
for i, msgs in enumerate(prompts):
    body = {
        "model": model,
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "return_routed_experts": True,
    }
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read())
    except Exception as e:
        print(f"[hit] prompt {i} failed: {e}")
        results.append({"prompt_idx": i, "error": str(e)})
        continue

    # Inspect EVERY plausible location for the routed_experts blob.
    def _walk(obj, path="$"):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "routed_experts":
                    yield path + "." + k, v
                yield from _walk(v, path + "." + k)
        elif isinstance(obj, list):
            for j, v in enumerate(obj):
                yield from _walk(v, path + f"[{j}]")

    locations = list(_walk(data))
    print(f"\n[hit] prompt {i} (msgs={msgs[0]['content'][:40]!r}):")
    print(f"  top-level keys: {sorted(data.keys())}")
    for path, val in locations:
        s = f"<{type(val).__name__}"
        if isinstance(val, str):
            s += f" len={len(val)}"
        elif isinstance(val, list):
            s += f" len={len(val)}"
        s += ">"
        print(f"  routed_experts at {path}: {s}")

    # Decode the first base64 hit we find, dump to disk.
    # PRESERVE the b64 string so the downstream round-trip step can
    # re-decode the full buffer (not just first_50 / last_10).
    b64_str = None
    for path, val in locations:
        if isinstance(val, str) and len(val) > 8:
            b64_str = val
            break
    decoded = None
    if b64_str:
        try:
            raw = base64.b64decode(b64_str.encode())
            import numpy as np
            arr = np.frombuffer(raw, dtype=np.int32)
            if arr.size > 0:
                decoded = {
                    "path": locations[0][0] if locations else "",
                    "raw_size_bytes": len(raw),
                    "n_int32": int(arr.size),
                    "first_50": arr[:50].tolist(),
                    "last_10": arr[-10:].tolist() if arr.size > 50 else [],
                }
        except Exception as e:
            print(f"    base64 decode failed: {e}")
    results.append({
        "prompt_idx": i,
        "msgs": msgs,
        "max_tokens": max_tokens,
        "top_level_keys": sorted(data.keys()),
        "routed_experts_locations": [{"path": p, "type": type(v).__name__,
                                     "len": (len(v) if hasattr(v, '__len__') else None)}
                                    for p, v in locations],
        "routed_experts_b64": b64_str,  # PRESERVED for round-trip
        "decoded": decoded,
        "choices_full": data.get("choices", []),
    })

with open(dump_file, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n[hit] wrote {dump_file}")
# Print a condensed summary file too (decoded)
summary = []
for r in results:
    if r.get("decoded"):
        d = r["decoded"]
        summary.append({
            "prompt_idx": r["prompt_idx"],
            "path": d["path"],
            "n_int32": d["n_int32"],
            "first_5_int32": d["first_50"][:5],
        })
with open(decoded_file, "w") as f:
    json.dump(summary, f, indent=2)
print(f"[hit] wrote {decoded_file}")
print(f"[hit] decoded summary: {json.dumps(summary, indent=2)}")
PYEOF

echo ""
echo "=== /v1/chat/completions probes (3 prompts) ==="
HOST="${HOST}" PORT="${PORT}" MODEL_ID="${MODEL_ID}" MAX_TOKENS="${MAX_TOKENS}" \
RESP_DUMP="${RESP_DUMP}" RESP_DECODED="${RESP_DECODED}" \
    "${VENV_DIR}/bin/python" "${OUTPUT_DIR}/hit_endpoint.py"

echo ""
echo "=== Files written ==="
ls -la "${OUTPUT_DIR}" | head -20
echo ""
echo "=== Final summary (from decoded summary file) ==="
if [[ -f "${RESP_DECODED}" ]]; then
    cat "${RESP_DECODED}"
fi

echo ""
echo "=== Server log tail ==="
tail -30 "${SERVER_LOG}"

echo ""
echo "[$(date -Iseconds)] done"
echo ""
echo "=== Round-trip: decode capturer buffer into per-layer expert counts ==="
export RESP_DUMP NUM_LAYERS NUM_EXPERTS TOP_K MODEL_NAME
NUM_LAYERS=16
NUM_EXPERTS=64
TOP_K=8
MODEL_NAME="${MODEL_ID}"
"${VENV_DIR}/bin/python" << 'PYEOF'
import base64, json, os
import numpy as np
resp_dump = os.environ["RESP_DUMP"]
n_layers = int(os.environ["NUM_LAYERS"])
n_experts = int(os.environ["NUM_EXPERTS"])
top_k = int(os.environ["TOP_K"])
model_name = os.environ["MODEL_NAME"]

with open(resp_dump) as f:
    data = json.load(f)

# Re-decode the b64 strings from the responses
all_layer_counts = np.zeros((n_layers, n_experts), dtype=np.int64)
total_tokens = 0
for r in data:
    p = r['prompt_idx']
    # The dump file structure: each prompt dict has 'routed_experts_b64'
    # set by hit_endpoint.py. Use it directly.
    b64_str = r.get('routed_experts_b64')
    if not b64_str:
        print(f'prompt {p}: no routed_experts_b64 in dump, skipping')
        continue
    raw = base64.b64decode(b64_str.encode())
    arr = np.frombuffer(raw, dtype=np.int32)
    # Reshape to (num_tokens, n_layers, top_k)
    if arr.size % (n_layers * top_k) != 0:
        print(f'prompt {p}: arr.size={arr.size} not divisible by {n_layers * top_k}')
        continue
    n_tokens = arr.size // (n_layers * top_k)
    arr3d = arr.reshape(n_tokens, n_layers, top_k)
    # Per-layer expert histogram
    for L in range(n_layers):
        for tok in arr3d[:, L, :].ravel():
            all_layer_counts[L, tok] += 1
    total_tokens += n_tokens
    print(f'prompt {p}: decoded {arr.size} int32 → {n_tokens} tokens × {n_layers} layers × {top_k} top_k')

print()
print(f'TOTAL: {total_tokens} sequence positions across {len(data)} prompts')
print(f'Per-layer count matrix shape: {all_layer_counts.shape}')
print(f'Total expert selections: {all_layer_counts.sum()} (should be {total_tokens * n_layers * top_k})')
print()
print('Per-layer top-3 most-picked experts:')
for L in range(n_layers):
    top3 = np.argsort(all_layer_counts[L])[-3:][::-1]
    top3_counts = all_layer_counts[L, top3]
    print(f'  L{L:2d}: experts {top3.tolist()} with counts {top3_counts.tolist()}')

# Save as analyzer-compatible JSON
out_json = os.path.join(os.path.dirname(resp_dump), f'expert_counts_smoke_{os.environ.get("SLURM_JOB_ID", "x")}.json')
result = {
    "model": model_name,
    "n_layers": n_layers,
    "n_experts": n_experts,
    "top_k": top_k,
    "n_prompts": len(data),
    "n_tokens": total_tokens,
    "layer_expert_counts": all_layer_counts.tolist(),
}
with open(out_json, 'w') as f:
    json.dump(result, f, indent=2)
print(f'\nWrote {out_json}')
PYEOF