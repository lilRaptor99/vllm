#!/usr/bin/env bash
#
# spartan/download-models.sh - login-node pre-download for the HF MoE
# model repos consumed by `examples/eval-moe/moe_expert_stats.py`.
#
# Why this exists: vLLM's `LLM(model=...)` resolves files via the
# standard HF hub cache layout
# ($HF_HOME/hub/models--<org>--<name>/snapshots/<rev>/...).
# Pre-seeding that cache from the login node (where HF egress is
# unrestricted and bandwidth is unmetered) means the GPU job starts
# warm and never has to pull a 5-250 GB safetensors bundle during
# compute-quota hours.
#
# vLLM does NOT use GGUF — it downloads the HF repo's native weights
# (safetensors / pytorch_model.bin) and config files. Unlike the
# sister `llama-cpp-eval/download-models.sh`, there is therefore no
# `--quant` knob and no allow-pattern magic: each repo is pulled whole
# (or with `--include` if you want to skip non-weight files). The set
# of weights downloaded is exactly what `vllm.LLM` would fetch on first
# use.
#
# IMPORTANT: keep DEFAULT_MODELS in sync with `DEFAULT_MODEL` in
# `examples/eval-moe/moe_expert_stats.py`. They intentionally duplicate
# rather than source because the analyzer defines its default at module
# scope; this script must also work standalone on the login node where
# the analyzer may not have been imported yet.
#
# Behaviour:
#   - Resolves SCRATCH_BASE the same way the .sbatch does.
#   - Auto-installs `huggingface_hub` to ~/.local if missing (login
#     node only).
#   - For each repo, calls `huggingface_hub.snapshot_download` and
#     writes to ${HF_CACHE:-${SCRATCH_BASE}/hf_cache}/hub/.
#   - Estimates total disk usage and (unless --yes) asks for
#     confirmation.
#   - Idempotent: re-running skips repos whose snapshots/<rev>/
#     config.json already exists.
#   - Logs everything; the GPU job's LLM(...) calls become no-ops on
#     the cache once this script has populated it.

set -uo pipefail

# ---------------------------------------------------------------- paths

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ------------------------------------------------------------- model list
#
# IMPORTANT: keep in sync with `DEFAULT_MODEL` in
# examples/eval-moe/moe_expert_stats.py. The first entry is the
# analyzer's own default; the rest are convenient alternates that the
# same analyzer can run via `--model <id>` on the sbatch command line.
readonly DEFAULT_MODELS=(
    "allenai/OLMoE-1B-7B-0924-Instruct"
    # "mistralai/Mixtral-8x22B-Instruct-v0.1"
    # "deepseek-ai/deepseek-moe-16b-chat"
    # "unsloth/gpt-oss-120b"
)

# --------------------------------------------------------- defaults / state

SCRATCH_BASE=""
MODELS=("${DEFAULT_MODELS[@]}")
INCLUDE_PATTERNS=()        # empty = pull entire repo
ASSUME_YES=0
LIST_ONLY=0
PRINT_USAGE=0

# ----------------------------------------------------------- logging funcs

_log()      { printf '[%s] %s\n' "${1}" "${*:2}"; }
_log_info() { _log "info"  "$@"; }
_log_warn() { _log "warn"  "$@"; }
_log_err()  { _log "error" "$@" >&2; }
_die()      { _log_err "$@"; exit 1; }

# ----------------------------------------------------------- path resolution
#
# Resolves SCRATCH_BASE exactly the same way the SLURM job does so the
# pre-download lands in the same cache the analyzer reads from.
# Override with --scratch-base for re-runs against a different scratch
# volume.
resolve_scratch_base() {
    if [[ -n "$SCRATCH_BASE" ]]; then
        printf '%s' "$SCRATCH_BASE"
        return
    fi
    if [[ -n "${SCRATCH:-}" ]]; then
        printf '%s/vllm' "${SCRATCH%/}"
        return
    fi
    printf '%s/vllm' "/data/scratch/projects/uom00014"
}

# Translate a HF repo id like "allenai/OLMoE-1B-7B-0924-Instruct" to
# the directory name HF uses inside $HF_HOME/hub/.
#
#     models--allenai--OLMoE-1B-7B-0924-Instruct
#
# This is what huggingface_hub.snapshot_download writes; the idempotency
# probe uses it to look up an already-cached snapshot.
hf_cache_dir_for() {
    local repo_id="$1"
    printf '%s' "models--${repo_id//\//--}"
}

# ---------------------------------------------------------------- python env
#
# `huggingface_hub` is the only thing we need on the login node. It's
# tiny (~5 MB) so we install to ~/.local rather than fighting with
# EasyBuild modules. The GPU node's .sbatch also installs it as a
# transitive dep of vllm, so this install is independent of the GPU
# stack.
ensure_huggingface_hub() {
    if python3 -c "import huggingface_hub" >/dev/null 2>&1; then
        return 0
    fi
    echo "[info] 'huggingface_hub' not importable on PATH; pip install --user huggingface_hub ..."
    local err
    err="$(mktemp)"
    if ! python3 -m pip install --user --quiet huggingface_hub 2>"${err}"; then
        if grep -q "externally-managed-environment\|break-system-packages" "${err}"; then
            echo "[info] PEP 668 in play; retrying with --break-system-packages ..."
            if ! python3 -m pip install --user --quiet --break-system-packages \
                    huggingface_hub 2>"${err}"; then
                echo "[fatal] pip install huggingface_hub failed:" >&2
                sed 's/^/  /' "${err}" >&2
                rm -f "${err}"
                return 1
            fi
        else
            echo "[fatal] pip install huggingface_hub failed:" >&2
            sed 's/^/  /' "${err}" >&2
            rm -f "${err}"
            return 1
        fi
    fi
    rm -f "${err}"
    export PATH="${HOME}/.local/bin:${PATH}"
}

# --------------------------------------------------------------- size probe
#
# Best-effort size estimate via the HF API. The `huggingface_hub`
# `HfApi.model_info()` exposes `safetensors.total` for repos that
# publish that field (most modern ones do). If unavailable, we fall
# back to "unknown" and let the user decide. Returns the size in MiB
# or -1 on probe failure.
probe_repo_size_mib() {
    local repo_id="$1"
    python3 - "${repo_id}" <<'PY' 2>/dev/null || echo "-1"
import sys

try:
    from huggingface_hub import HfApi
except Exception:
    raise SystemExit("-1")

repo_id = sys.argv[1]
try:
    info = HfApi().model_info(repo_id)
except Exception as e:
    print(f"probe-failed: {e}", file=sys.stderr)
    raise SystemExit("-1")

# Try safetensors metadata first.
size = 0
try:
    if info.safetensors is not None and info.safetensors.total is not None:
        size = int(info.safetensors.total)
except Exception:
    pass

# Fall back to summing siblings. Slightly slower because it lists every
# file, but works for repos that don't publish safetensors metadata.
if size == 0:
    try:
        for s in info.siblings:
            if s.size and (s.rfilename.endswith(".safetensors")
                           or s.rfilename.endswith(".bin")
                           or s.rfilename.endswith(".gguf")):
                size += s.size
    except Exception:
        pass

if size == 0:
    raise SystemExit("-1")

print(size // (1024 * 1024))
PY
}

# --------------------------------------------------------------- download
#
# Download one (repo_id) into $HF_CACHE/hub/. Idempotent: skips if the
# snapshot dir already contains a config.json.
download_one() {
    local repo_id="$1"
    local log="${CACHE_DIR}/${repo_id//\//__}.download.log"
    local cache_dir="${CACHE_DIR}/$(hf_cache_dir_for "${repo_id}")"

    mkdir -p "${CACHE_DIR}"

    # Idempotency: any snapshot dir with config.json means we're done.
    if compgen -G "${cache_dir}/snapshots/*/config.json" >/dev/null; then
        echo "[info] ${repo_id}: already cached, skip"
        return 0
    fi

    echo "[info] ${repo_id}: downloading -> ${cache_dir} ..."

    if ! python3 - "${repo_id}" "${CACHE_DIR}" $(printf -- '--include %q ' "${INCLUDE_PATTERNS[@]}") >>"${log}" 2>&1 <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
cache_dir = sys.argv[2]
extra = sys.argv[3:]

kwargs = {"cache_dir": cache_dir, "allow_patterns": None}
i = 0
while i < len(extra):
    flag = extra[i]
    val = extra[i + 1]
    if flag == "--include":
        # Multiple --include flags are additive; huggingface_hub accepts
        # a single list of globs, so we accumulate.
        if kwargs["allow_patterns"] is None:
            kwargs["allow_patterns"] = []
        kwargs["allow_patterns"].append(val)
    i += 2

# Pass through HF_TOKEN if set.
if "HF_TOKEN" in os.environ:
    kwargs["token"] = os.environ["HF_TOKEN"]

print(f"  cache_dir = {cache_dir}", flush=True)
print(f"  allow_patterns = {kwargs['allow_patterns']}", flush=True)

path = snapshot_download(repo_id, **kwargs)
print(f"  -> {path}", flush=True)
PY
    then
        echo "[fatal] ${repo_id} download failed; log: ${log}" >&2
        return 1
    fi
    echo "[info] ${repo_id}: ok"
    return 0
}

# ----------------------------------------------------------------- main CLI

print_usage() {
    cat <<'EOF'
Usage: bash spartan/download-models.sh [options]

Options:
  --models <id> [<id> ...]      subset of hardcoded HF repo ids
                                (default: 5 canonical MoE repos)
  --include <glob>              glob pattern(s) forwarded to
                                huggingface_hub.snapshot_download.
                                Repeatable; e.g. --include "*.safetensors"
                                --include "*.json" --include "tokenizer*"
                                limits the download to weights + metadata.
                                (default: pull entire repo)
  --scratch-base <path>         override SCRATCH_BASE
                                (default: ${SCRATCH}/vllm or
                                 /data/scratch/projects/uom00014/vllm)
  --list                        show what would be downloaded without doing it
  --yes                         skip the disk-usage confirmation prompt
  -h, --help                    show this message and exit

Examples:
  # Pre-download the analyzer's default model only:
  bash spartan/download-models.sh --models allenai/OLMoE-1B-7B-0924-Instruct

  # Pre-download all 5 canonical MoE models (~300 GB total worst-case):
  bash spartan/download-models.sh --yes

  # Just the safetensors + config (skip pytorch_model.bin + .gguf):
  bash spartan/download-models.sh --include "*.safetensors" --include "*.json" \
                                  --include "tokenizer*" --include "*.txt"

  # Custom scratch base:
  bash spartan/download-models.sh --scratch-base /data/scratch/projects/uom00014/vllm

Environment:
  SCRATCH                       if set, used as the parent of --scratch-base.
                                 Default: unset; --scratch-base falls back to
                                 /data/scratch/projects/uom00014/vllm.
  HF_TOKEN                      optional; passed through to huggingface_hub
                                 for gated repos.

EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --models)
                shift; MODELS=()
                while [[ $# -gt 0 && "$1" != --* && "$1" != -* ]]; do
                    MODELS+=("$1"); shift
                done
                [[ ${#MODELS[@]} -gt 0 ]] || {
                    echo "[fatal] --models requires at least one id" >&2
                    exit 1
                }
                ;;
            --include)
                shift
                if [[ $# -eq 0 || "$1" == --* || "$1" == -* ]]; then
                    echo "[fatal] --include requires a glob argument" >&2
                    exit 1
                fi
                INCLUDE_PATTERNS+=("$1"); shift
                ;;
            --scratch-base)   SCRATCH_BASE="$2"; shift 2 ;;
            --list)           LIST_ONLY=1; shift ;;
            --yes)            ASSUME_YES=1; shift ;;
            -h|--help)        PRINT_USAGE=1; shift ;;
            *)
                echo "[fatal] unknown argument: $1 (use --help)" >&2
                exit 1
                ;;
        esac
    done
}

# ------------------------------------------------------------------ main

main() {
    parse_args "$@"
    if [[ $PRINT_USAGE -eq 1 ]]; then
        print_usage
        return 0
    fi

    SCRATCH_BASE="$(resolve_scratch_base)"
    CACHE_DIR="${SCRATCH_BASE}/hf_cache"
    mkdir -p "${CACHE_DIR}"

    echo "============================================================"
    echo "[$(date -Iseconds)] vllm-moe-eval model pre-download"
    echo "[$(date -Iseconds)] REPO_ROOT    = ${REPO_ROOT}"
    echo "[$(date -Iseconds)] SCRATCH_BASE = ${SCRATCH_BASE}"
    echo "[$(date -Iseconds)] HF_CACHE     = ${CACHE_DIR}"
    echo "[$(date -Iseconds)] MODELS       = ${MODELS[*]}"
    echo "[$(date -Iseconds)] INCLUDE      = ${INCLUDE_PATTERNS[*]:-<all>}"
    echo "============================================================"

    # Estimate disk usage up front. Probe each repo's safetensors total
    # via the HF API; fall back to "unknown" on probe failure.
    total_mib=0
    unknown=0
    echo "[info] probing repo sizes via HF API ..."
    for m in "${MODELS[@]}"; do
        mib="$(probe_repo_size_mib "${m}")"
        if [[ "${mib}" == "-1" ]]; then
            printf '   %-60s   unknown\n' "${m}"
            unknown=$(( unknown + 1 ))
        else
            printf '   %-60s   %s MiB\n' "${m}" "${mib}"
            total_mib=$(( total_mib + mib ))
        fi
    done
    if [[ ${unknown} -gt 0 ]]; then
        echo "[warn] ${unknown} repo(s) size unknown - total estimate is a lower bound"
    fi
    if [[ ${total_mib} -gt 0 ]]; then
        echo "[info] total download plan: ~$((total_mib / 1024)) GiB (lower bound)"
    else
        echo "[info] total download plan: unknown"
    fi

    if [[ $LIST_ONLY -eq 1 ]]; then
        echo "[info] --list: would download ${MODELS[*]}; exiting"
        return 0
    fi

    if [[ $ASSUME_YES -ne 1 ]]; then
        echo -n "[?] proceed? [y/N] "
        read -r ans
        case "${ans}" in
            y|Y|yes|YES) ;;
            *)
                echo "[info] aborted"
                return 0
                ;;
        esac
    fi

    ensure_huggingface_hub || exit 1

    local m
    for m in "${MODELS[@]}"; do
        download_one "${m}" || {
            echo "[warn] ${m} download failed; continuing with the rest" >&2
        }
    done

    echo "============================================================"
    echo "[$(date -Iseconds)] pre-download complete"
    echo "[$(date -Iseconds)] cached at: ${CACHE_DIR}"
    echo "[$(date -Iseconds)] re-running this script is a no-op (idempotent)"
    echo "============================================================"
}

main "$@"
