#!/usr/bin/env bash
#
# spartan/download-datasets.sh - login-node pre-download for the 5
# benchmarks consumed by `examples/eval-moe/moe_expert_stats.py`:
#
#   mmlu      (cais/mmlu,                  subset "all", test+dev splits)
#   bbh       (Joschka/big_bench_hard,    all 27 sub-tasks, test split)
#   humaneval (openai/openai_humaneval,    test split)
#   popqa     (akariasai/PopQA,            test split - per-row jsonl)
#   include   (CohereLabs/include-base-44, per-row jsonl across 44 languages)
#
# Why this exists: the GPU compute nodes on Spartan are firewalled off
# from the public internet for outbound HTTPS, AND they don't ship the
# `datasets` Python package. Pre-running the downloads on the login
# node (which has internet + `pip install --user datasets`) means the
# GPU job starts with every dataset already on disk and never tries to
# fetch one mid-eval.
#
# IMPORTANT: keep DEFAULT_DATASETS in sync with `BENCHMARK_LOADERS`
# in `examples/eval-moe/moe_expert_stats.py`. They intentionally
# duplicate rather than source because the analyzer defines its own
# list at module scope; this script must also work standalone on the
# login node where the analyzer may not have been imported yet.
#
# Behaviour:
#   - Resolves SCRATCH_BASE the same way the .sbatch does.
#   - Auto-installs `datasets` to ~/.local if missing (login node only).
#   - For each dataset, dispatches to its own downloader (mmlu/bbh/
#     humaneval use a save_to_disk-based internal path; popqa/include
#     shell out to dedicated `download-popqa.py` / `download-include.py`
#     helpers under this directory).
#   - Idempotent: re-running skips datasets whose output already exists.
#   - Logs everything; the GPU job's benchmark loaders become no-ops
#     once this script has populated the cache.

set -uo pipefail

# ---------------------------------------------------------------- paths

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ------------------------------------------------------------- dataset list
#
# Two layouts:
#   * "save_to_disk" (mmlu, bbh, humaneval): each split is its own
#     arrow directory under ${DATASET_DIR}/<ds>/<split>/. The
#     `internal_save_to_disk` helper drives the load + save.
#   * "jsonl" (popqa, include): one file per dataset
#     (${DATASET_DIR}/<ds>/<ds>.jsonl) plus sidecar partition lists.
#     The `dispatch_external_downloader` helper invokes the per-dataset
#     Python downloader in this directory.
readonly -A DATASET_LAYOUT=(
    [mmlu]="save_to_disk"
    [bbh]="save_to_disk"
    [humaneval]="save_to_disk"
    [popqa]="jsonl"
    [include]="jsonl"
)
# HuggingFace ids (used by the save_to_disk path only).
readonly -A DATASET_HF_ID=(
    [mmlu]="cais/mmlu"
    [bbh]="Joschka/big_bench_hard"
    [humaneval]="openai/openai_humaneval"
)
# Subset / config hint (passed to load_dataset for the save_to_disk path).
readonly -A DATASET_SUBSET=(
    [mmlu]="all"
    [bbh]="__all__"          # special sentinel - download every sub-task
    [humaneval]=""
)
# Splits to stage under ${DATASET_DIR}/<ds>/<split>/. The analyzer reads:
#   mmlu:      test + dev
#   bbh:       test (across all 27 sub-tasks)
#   humaneval: test
readonly -A DATASET_SPLITS=(
    [mmlu]="test dev"
    [bbh]="test"
    [humaneval]="test"
)
# Markers that prove the save_to_disk cache is fully populated for a
# dataset. If every required file under out_dir is present, the download
# is skipped on a re-run.
readonly -A DATASET_DONE_MARKER=(
    [mmlu]="test dev"   # both subdirs must exist
    [bbh]="__all__"     # special: every sub-task subdir must have test/
    [humaneval]="test"
)
# Per-dataset downloader (only for jsonl layout).
readonly -A JSONL_DOWNLOADER=(
    [popqa]="${SCRIPT_DIR}/download-popqa.py"
    [include]="${SCRIPT_DIR}/download-include.py"
)
# Per-dataset done marker for jsonl layout.
readonly -A JSONL_DONE_MARKER=(
    [popqa]="popqa.jsonl"
    [include]="include.jsonl"
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

# ---------------------------------------------------------- download helpers
#
# save_to_disk path: download a (hf_id, subset, splits) entry to
# ${out_dir}/. Re-runs are no-ops when the directory exists and
# contains every requested split.
download_save_to_disk() {
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
    # subset_args and split_args are passed via array expansion so each
    # flag + its value become a separate argv element. The previous
    # "subset_flag=\"--subset ${subset}\"" string form put the whole
    # "--subset all" into a single argv element, which the parser
    # below then silently skipped (it only matches on the exact flag
    # name). Pass as separate tokens instead.
    local subset_args=()
    if [[ -n "${subset}" && "${subset}" != "__all__" ]]; then
        subset_args=(--subset "${subset}")
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

    local split_args=()
    for split in ${splits}; do
        split_args+=(--split "${split}")
    done

    if ! python3 - "${hf_id}" "${out_dir}" "${subset_args[@]}" "${split_args[@]}" >>"${log}" 2>&1 <<'PY'
import sys
from datasets import load_dataset

(hf_id, out_dir) = sys.argv[1], sys.argv[2]
extra = sys.argv[3:]

# `datasets >= 3` renamed the `name=` kwarg to `config_name=`. The old
# `name=` is silently dropped, which then produces the misleading
# "Config name is missing" error. Probe for whichever kwarg the
# installed `datasets` understands and use that.
import inspect
try:
    _sig = inspect.signature(load_dataset)
except (TypeError, ValueError):
    _sig = None
_subset_kwarg = "config_name"
if _sig is not None and "config_name" not in _sig.parameters and "name" in _sig.parameters:
    _subset_kwarg = "name"

kwargs = {}
splits = []
i = 0
# Walk argv token-by-token: any `--flag` consumes the next non-flag
# token as its value. A token starting with `--` that is followed by
# another flag (or end-of-args) is silently skipped (the previous
# parser crashed with IndexError in that case; we now ignore it).
while i < len(extra):
    flag = extra[i]
    if flag == "--subset" and i + 1 < len(extra) and not extra[i + 1].startswith("--"):
        kwargs[_subset_kwarg] = extra[i + 1]
        i += 2
    elif flag == "--split" and i + 1 < len(extra) and not extra[i + 1].startswith("--"):
        splits.append(extra[i + 1])
        i += 2
    elif flag == "--split":
        # `--split` without a following value: skip the flag itself.
        i += 1
    else:
        # Unknown / stray token: skip it (don't crash the parser).
        i += 1

if not splits:
    sys.exit("no `--split` arguments reached the python helper; check the shell quoting in download_one()")

# `name`/`config_name` (subset) is fixed across splits; load each split
# independently so we get one directory per split under out_dir.
for split in splits:
    print(f"  - downloading split={split!r} (subset={kwargs.get(_subset_kwarg)!r}) ...")
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
import os
import sys
from datasets import get_dataset_config_names, load_dataset

hf_id = sys.argv[1]
out_dir = sys.argv[2]

configs = get_dataset_config_names(hf_id)
print(f"  - {len(configs)} sub-tasks: {configs}")
# `Joschka/big_bench_hard` exposes `few_shot_prompts` as a config
# alongside the 27 real tasks. It's a meta-config of in-context
# examples, not a scoring task, and would dilute the routing signal
# if we included it. Filter it out before downloading.
configs = [c for c in configs if c != "few_shot_prompts"]
print(f"  - {len(configs)} real sub-tasks after filter: {configs}")
for cfg in configs:
    cfg_dir = f"{out_dir}/{cfg}"
    split_dir = f"{cfg_dir}/test"
    # Idempotency: a previously-downloaded sub-task has both an Arrow
    # metadata file and at least one data shard. We check for the
    # marker files explicitly instead of catching a broad Exception
    # around Dataset.load_from_disk - the broad-except version silently
    # swallowed real errors (corrupt cache, permission denied) and
    # kept re-downloading on every run.
    if (
        os.path.isfile(os.path.join(split_dir, "dataset_info.json"))
        and (
            any(
                n.startswith("data-") and n.endswith(".arrow")
                for n in os.listdir(split_dir)
            )
            or any(
                n.startswith("data-") and n.endswith(".arrow")
                for n in os.listdir(os.path.join(split_dir, "data"))
            )
        )
    ):
        print(f"  - {cfg}: already cached, skip")
        continue
    print(f"  - {cfg}: downloading ...")
    # In `datasets >= 3`, the BBH sub-task name is exposed as the
    # dataset's only split (not `test`). `load_dataset(hf, cfg,
    # split="test")` raises "Unknown split" - use `split=cfg`.
    ds = load_dataset(hf_id, cfg, split=cfg)
    ds.save_to_disk(split_dir)
    print(f"  - {cfg}: saved {len(ds)} rows to {split_dir}")
PY
}

# jsonl path: shell out to the per-dataset Python downloader
# (download-popqa.py, download-include.py). These mirror the
# sister llama-cpp-eval/scripts so the two projects share the same
# on-disk cache layout.
download_jsonl() {
    local short_name="$1"
    local script="${JSONL_DOWNLOADER[${short_name}]}"
    local out_dir="${DATASETS_DIR}/${short_name}"
    local marker="${JSONL_DONE_MARKER[${short_name}]}"
    local log="${out_dir}.download.log"

    mkdir -p "${out_dir}"

    if [[ -f "${out_dir}/${marker}" ]]; then
        echo "[info] ${short_name}: already prepared (${marker}), skip"
        return 0
    fi

    echo "[info] ${short_name}: downloading via ${script} -> ${out_dir} ..."
    if ! python3 "${script}" --outdir "${out_dir}" >>"${log}" 2>&1; then
        echo "[fatal] ${short_name} download failed; log: ${log}" >&2
        return 1
    fi
    echo "[info] ${short_name}: ok"
}

download_one() {
    local short_name="$1"
    case "${DATASET_LAYOUT[${short_name}]:-}" in
        save_to_disk)
            download_save_to_disk "${short_name}"
            ;;
        jsonl)
            download_jsonl "${short_name}"
            ;;
        *)
            echo "[fatal] unknown layout for '${short_name}'" >&2
            return 1
            ;;
    esac
}

# --------------------------------------------------------------- CLI parsing

print_usage() {
    cat <<'EOF'
Usage: bash spartan/download-datasets.sh [options]

Options:
  --datasets <ds> [<ds> ...]   subset of {mmlu, bbh, humaneval, popqa, include}
                                (default: all five)
  --scratch-base <path>        override SCRATCH_BASE
                                (default: ${SCRATCH}/vllm or
                                 /data/scratch/projects/uom00014/vllm)
  --list                       show what would be downloaded without doing it
  --yes                        skip the disk-usage confirmation prompt
  -h, --help                   show this message and exit

Examples:
  # Download everything (mmlu + bbh + humaneval + popqa + include):
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

DATASETS=(mmlu bbh humaneval popqa include)

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
    local -a ALL=(mmlu bbh humaneval popqa include)
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
    # Worst case (mmlu ~250 MB + bbh ~30 MB + humaneval ~3 MB + popqa ~50 MB
    # + include ~150 MB) is ~500 MB total - small enough that we don't
    # bother asking for confirmation. Just print it.
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
    case "${DATASETS[*]}" in
        *popqa*)     total_mb=$((total_mb + 50))  ;;
    esac
    case "${DATASETS[*]}" in
        *include*)   total_mb=$((total_mb + 150)) ;;
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
