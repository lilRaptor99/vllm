#!/usr/bin/env bash
#
# spartan/download-datasets.sh - login-node pre-download for the 3
# benchmarks consumed by `examples/eval-moe/moe_expert_stats.py`:
#
#   mmlu      (cais/mmlu,                  subset "all", test+dev splits)
#   bbh       (Joschka/big_bench_hard,    all 27 sub-tasks, test split)
#   humaneval (openai/openai_humaneval,    test split)
#
# Why this exists: the GPU compute nodes on Spartan are firewalled off
# from the public internet for outbound HTTPS, AND they don't ship the
# `datasets` Python package. Both are required by `moe_expert_stats.py`'s
# lazy `from datasets import load_dataset(...)` blocks. Pre-running the
# downloads on the login node (which has internet + an easy
# `pip install --user datasets`) means the GPU job starts with every
# dataset already on disk and never tries to fetch one mid-eval.
#
# IMPORTANT: keep DEFAULT_DATASETS in sync with the `BENCHMARK_LOADERS`
# dict at the top of `examples/eval-moe/moe_expert_stats.py`. They
# intentionally duplicate rather than source because the analyzer defines
# its own list at module scope; this script must also work standalone on
# the login node where the analyzer may not have been imported yet.
#
# Behaviour:
#   - Resolves SCRATCH_BASE the same way the .sbatch does.
#   - Auto-installs `datasets` to ~/.local if missing (login node only).
#   - For each dataset, calls `datasets.load_dataset(...)` and writes the
#     splits to `${SCRATCH_BASE}/datasets/<name>/` as HuggingFace's
#     native Arrow-on-disk format (one directory per split). The
#     analyzer reads them via `datasets.load_dataset(<path>, split=...)`
#     so the layout matches what HF Datasets expects.
#   - Idempotent: re-running skips datasets whose directory already
#     exists and contains the requested splits.
#   - Logs everything; the GPU job's benchmark loaders become no-ops
#     once this script has populated the cache.

set -uo pipefail

# ---------------------------------------------------------------- paths

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ------------------------------------------------------------- dataset list
#
# IMPORTANT: keep in sync with `BENCHMARK_LOADERS` in
# examples/eval-moe/moe_expert_stats.py. The keys are the analyzer's
# short names; the values are the (path, subset_or_none, splits) tuples
# consumed by `download_one`.
readonly -A DATASET_HF_ID=(
    [mmlu]="cais/mmlu"
    [bbh]="Joschka/big_bench_hard"
    [humaneval]="openai/openai_humaneval"
)
readonly -A DATASET_SUBSET=(
    [mmlu]="all"
    [bbh]="__all__"          # special sentinel - download every sub-task
    [humaneval]=""
)
# Splits to stage. The analyzer reads:
#   mmlu:      test + dev
#   bbh:       test (across all 27 sub-tasks)
#   humaneval: test
readonly -A DATASET_SPLITS=(
    [mmlu]="test dev"
    [bbh]="test"
    [humaneval]="test"
)

# ----------------------------------------------------------- path resolution
#
# Resolves SCRATCH_BASE exactly the same way the SLURM job does so the
# pre-download lands in the same cache the analyzer reads from.
# Override with --scratch-base for re-runs against a different scratch
# volume.
SCRATCH_BASE=""
LIST_ONLY=0
ASSUME_YES=0
PRINT_USAGE=0

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

# --------------------------------------------------------------- python env
#
# The `datasets` package is required both here (to pre-download) and on
# the GPU node (to re-load via load_dataset(<path>, split=...)). On the
# login node we install it to ~/.local; on the GPU node the .sbatch
# installs it before invoking the analyzer. We never install anything
# system-wide.
ensure_datasets() {
    if python3 -c "import datasets" >/dev/null 2>&1; then
        return 0
    fi
    echo "[info] 'datasets' not importable on PATH; pip install --user datasets ..."
    local err
    err="$(mktemp)"
    if ! python3 -m pip install --user --quiet datasets 2>"${err}"; then
        if grep -q "externally-managed-environment\|break-system-packages" "${err}"; then
            echo "[info] PEP 668 in play; retrying with --break-system-packages ..."
            if ! python3 -m pip install --user --quiet --break-system-packages \
                    datasets 2>"${err}"; then
                echo "[fatal] pip install datasets failed:" >&2
                sed 's/^/  /' "${err}" >&2
                rm -f "${err}"
                return 1
            fi
        else
            echo "[fatal] pip install datasets failed:" >&2
            sed 's/^/  /' "${err}" >&2
            rm -f "${err}"
            return 1
        fi
    fi
    rm -f "${err}"
    export PATH="${HOME}/.local/bin:${PATH}"
}

# ---------------------------------------------------------- download helper
#
# Downloads one (short_name, hf_id, subset, splits) entry. Writes to
# ${out_dir}/. Re-runs are no-ops when the directory exists and
# contains every requested split.
download_one() {
    local short_name="$1"
    local hf_id="${DATASET_HF_ID[${short_name}]}"
    local subset="${DATASET_SUBSET[${short_name}]}"
    local splits="${DATASET_SPLITS[${short_name}]}"
    local out_dir="${DATASETS_DIR}/${short_name}"
    local log="${out_dir}.download.log"

    if [[ -z "${hf_id}" ]]; then
        echo "[fatal] unknown short_name '${short_name}'" >&2
        return 1
    fi
    mkdir -p "${out_dir}"

    # Idempotency check: every requested split must exist under out_dir.
    local need_download=0
    local split
    for split in ${splits}; do
        if [[ ! -d "${out_dir}/${split}" ]]; then
            need_download=1
            break
        fi
    done
    if [[ "${need_download}" -eq 0 ]]; then
        echo "[info] ${short_name}: already prepared (${splits}), skip"
        return 0
    fi

    echo "[info] ${short_name}: downloading ${hf_id} -> ${out_dir} ..."
    local subset_flag=""
    if [[ -n "${subset}" && "${subset}" != "__all__" ]]; then
        subset_flag="--subset ${subset}"
    fi

    # Special-case BBH: enumerate the 27 sub-task configs and download
    # each one to its own subdirectory. The analyzer reads them by
    # calling datasets.get_dataset_config_names(...) then
    # load_dataset(<name>, <config>, split="test") for each, so the
    # cache layout must mirror that.
    if [[ "${subset}" == "__all__" ]]; then
        download_bbh_subtasks "${hf_id}" "${out_dir}" "${log}"
        return $?
    fi

    local split_args=""
    for split in ${splits}; do
        split_args+="--split ${split} "
    done

    if ! python3 - "${hf_id}" "${out_dir}" "${subset_flag}" ${split_args} >>"${log}" 2>&1 <<'PY'
import sys
from datasets import load_dataset

(hf_id, out_dir) = sys.argv[1], sys.argv[2]
extra = sys.argv[3:]

kwargs = {}
positional = []
i = 0
while i < len(extra):
    flag = extra[i]
    val = extra[i + 1]
    if flag == "--subset":
        kwargs["name"] = val
    elif flag == "--split":
        positional.append(val)
    i += 2

# `name` (subset) is fixed across splits; load each split independently
# so we get one directory per split under out_dir.
for split in positional:
    print(f"  - downloading split={split!r} ...")
    ds = load_dataset(hf_id, split=split, **kwargs)
    ds.save_to_disk(f"{out_dir}/{split}")
    print(f"  - saved {split} ({len(ds)} rows) to {out_dir}/{split}")
PY
    then
        echo "[fatal] ${short_name} download failed; log: ${log}" >&2
        return 1
    fi
    echo "[info] ${short_name}: ok"
}

# BBH is a multi-config dataset. Download every sub-task to its own
# directory so the analyzer's get_dataset_config_names + load_dataset
# call chain reads from local cache. Each sub-task directory has its
# own test split.
download_bbh_subtasks() {
    local hf_id="$1"
    local out_dir="$2"
    local log="$3"

    python3 - "${hf_id}" "${out_dir}" >>"${log}" 2>&1 <<'PY' || return $?
import sys
from datasets import get_dataset_config_names, load_dataset

hf_id = sys.argv[1]
out_dir = sys.argv[2]

configs = get_dataset_config_names(hf_id)
print(f"  - {len(configs)} sub-tasks: {configs}")
for cfg in configs:
    cfg_dir = f"{out_dir}/{cfg}"
    split_dir = f"{cfg_dir}/test"
    try:
        # Idempotency: skip configs whose test split already exists.
        from datasets import Dataset
        Dataset.load_from_disk(split_dir)
        print(f"  - {cfg}: already cached, skip")
        continue
    except Exception:
        pass
    print(f"  - {cfg}: downloading ...")
    ds = load_dataset(hf_id, cfg, split="test")
    ds.save_to_disk(split_dir)
    print(f"  - {cfg}: saved {len(ds)} rows to {split_dir}")
PY
}

# --------------------------------------------------------------- CLI parsing

print_usage() {
    cat <<'EOF'
Usage: bash spartan/download-datasets.sh [options]

Options:
  --datasets <ds> [<ds> ...]   subset of {mmlu, bbh, humaneval}
                                (default: all three)
  --scratch-base <path>        override SCRATCH_BASE
                                (default: ${SCRATCH}/vllm or
                                 /data/scratch/projects/uom00014/vllm)
  --list                       show what would be downloaded without doing it
  --yes                        skip the disk-usage confirmation prompt
  -h, --help                   show this message and exit

Examples:
  # Download everything (mmlu + bbh + humaneval):
  bash spartan/download-datasets.sh

  # Just MMLU (fastest smoke test):
  bash spartan/download-datasets.sh --datasets mmlu

  # Custom scratch base (e.g. for a different project quota):
  bash spartan/download-datasets.sh --scratch-base /data/scratch/projects/uom00014/vllm

Environment:
  SCRATCH                      if set, used as the parent of --scratch-base.
                                Default: unset; --scratch-base falls back to
                                /data/scratch/projects/uom00014/vllm.

EOF
}

DATASETS=(mmlu bbh humaneval)

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --datasets)
                shift; DATASETS=()
                while [[ $# -gt 0 && "$1" != --* && "$1" != -* ]]; do
                    DATASETS+=("$1"); shift
                done
                [[ ${#DATASETS[@]} -gt 0 ]] || {
                    echo "[fatal] --datasets requires at least one name" >&2
                    exit 1
                }
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

validate_choices() {
    local -a ALL=(mmlu bbh humaneval)
    local ds
    for ds in "${DATASETS[@]}"; do
        local known=0
        local k
        for k in "${ALL[@]}"; do
            if [[ "$k" == "$ds" ]]; then known=1; break; fi
        done
        [[ $known -eq 1 ]] || {
            echo "[fatal] unknown dataset '$ds' (allowed: ${ALL[*]})" >&2
            exit 1
        }
    done
}

# ------------------------------------------------------------------ main

main() {
    parse_args "$@"
    if [[ $PRINT_USAGE -eq 1 ]]; then
        print_usage
        return 0
    fi
    validate_choices

    SCRATCH_BASE="$(resolve_scratch_base)"
    DATASETS_DIR="${SCRATCH_BASE}/datasets"
    mkdir -p "${DATASETS_DIR}"

    echo "============================================================"
    echo "[$(date -Iseconds)] vllm-moe-eval dataset pre-download"
    echo "[$(date -Iseconds)] REPO_ROOT    = ${REPO_ROOT}"
    echo "[$(date -Iseconds)] SCRATCH_BASE = ${SCRATCH_BASE}"
    echo "[$(date -Iseconds)] DATASETS_DIR = ${DATASETS_DIR}"
    echo "[$(date -Iseconds)] DATASETS     = ${DATASETS[*]}"
    echo "============================================================"

    # Estimate disk usage up front so the user can abort if it's too big.
    # Worst case (mmlu full ~250 MB + bbh ~30 MB + humaneval ~3 MB) is
    # ~300 MB total - small enough that we don't bother asking for
    # confirmation. Just print it.
    total_mb=0
    case "${DATASETS[*]}" in
        *mmlu*)      total_mb=$((total_mb + 250)) ;;
    esac
    case "${DATASETS[*]}" in
        *bbh*)       total_mb=$((total_mb + 30))  ;;
    esac
    case "${DATASETS[*]}" in
        *humaneval*) total_mb=$((total_mb + 3))   ;;
    esac
    echo "[info] estimated cache footprint: ~${total_mb} MB"

    if [[ $LIST_ONLY -eq 1 ]]; then
        echo "[info] --list: would download ${DATASETS[*]}; exiting"
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

    ensure_datasets || exit 1

    local ds
    for ds in "${DATASETS[@]}"; do
        download_one "${ds}" || {
            echo "[warn] ${ds} download failed; continuing with the rest" >&2
        }
    done

    echo "============================================================"
    echo "[$(date -Iseconds)] pre-download complete"
    echo "[$(date -Iseconds)] staged at: ${DATASETS_DIR}"
    echo "[$(date -Iseconds)] re-running this script is a no-op (idempotent)"
    echo "============================================================"
}

main "$@"
