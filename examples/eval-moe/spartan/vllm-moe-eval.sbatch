#!/usr/bin/env bash
#
# spartan/vllm-moe-eval.sbatch - SLURM job that runs the vLLM MoE expert
# activation statistics analyzer on the Unimelb Spartan HPC
# (gpu-a100-short partition by default).
#
# The job is naturally idempotent: the analyzer overwrites per-benchmark
# JSONs under ${RESULTS_DIR}, so re-submitting after a pre-emption
# re-runs the model end-to-end (vLLM doesn't expose a "skip cell"
# switch). The pre-downloaded datasets in ${DATASETS_DIR} are reused.
#
# Prerequisites
# -------------
# 1. This vllm checkout (or a fork) is on disk at $REPO_ROOT
#    (default: /data/gpfs/projects/uom00014/vllm). Must already have a
#    working .venv/ with vllm + numpy + datasets installed, OR be willing
#    to let the .sbatch's preflight create one.
# 2. The HF datasets the analyzer reads have been pre-populated under
#    $SCRATCH_BASE/datasets. Run on the login node first:
#        bash spartan/download-datasets.sh
#    (or --datasets mmlu for a smoke test). GPU compute nodes on
#    Spartan are firewalled off from the public internet, so datasets
#    MUST be staged on the login node.
# 3. The HF model weights the analyzer loads will be downloaded on first
#    use (vllm.LLM(model=...) pulls via huggingface_hub). They land in
#    $HF_HOME; we set that to ${SCRATCH_BASE}/hf_cache so the cache is
#    on the fast scratch volume, not the GPFS project volume.
#    Recommended: pre-download on the login node to avoid pulling
#    5-250 GB during compute-quota hours:
#        bash spartan/download-models.sh --models <id> [<id> ...]
#    The script is idempotent and writes to the same ${HF_CACHE} the
#    .sbatch reads from.
#
# Quick start
# -----------
#   # 1. login node: pre-download the datasets
#   ssh spartan
#   cd <repo>/examples/eval-moe
#   bash spartan/download-datasets.sh
#
#   # 2. submit the GPU job (defaults: gpu-a100-short, gpu:1, 4-day
#   #    wall clock, CUDA/12.4.1 + Python/3.11.3 + GCCcore/11.3.0).
#   sbatch spartan/vllm-moe-eval.sbatch
#
#   # 3. tail logs
#   tail -f spartan/logs/vllm-moe-eval-<jobid>.out
#
# Env-var overrides (export before sbatch, or pass via --export=...)
# ------------------------------------------------------------------
#   REPO_ROOT       path to the cloned vllm repo
#                                                  (default: /data/gpfs/projects/uom00014/vllm)
#   SCRATCH_BASE    root for venv / datasets / results / hf_cache
#                                                  (default: /data/scratch/projects/uom00014/vllm)
#   CUDA_MODULE     Lmod module name for CUDA     (default: CUDA/12.4.1)
#   PYTHON_MODULE   Lmod module name for Python   (default: Python/3.11.3)
#   GCC_MODULE      Lmod module name for GCC      (default: GCCcore/11.3.0)
#   MODELS_OVERRIDE space-separated HF model ids (default: allenai/OLMoE-1B-7B-0924-Instruct)
#                                                  Cells are evaluated as MODELS x QUANTS.
#   QUANTS_OVERRIDE space-separated vLLM quant tags (default: unset = no quant)
#                                                  Forwarded as --quant <m1> <m2> ... to the analyzer.
#   BENCHMARKS      space-separated subset of {mmlu,bbh,humaneval}
#                                                  (default: all three)
#   NUM_SAMPLES     prompts per benchmark         (default: 256)
#   MAX_TOKENS      generation length             (default: 256)
#   MAX_MODEL_LEN   vLLM max_model_len            (default: 4096)
#   GPU_MEM_UTIL    vLLM gpu_memory_utilization   (default: 0.92)
#   ENFORCE_EAGER   pass --enforce-eager to LLM()  (default: unset)
#   OUTPUT_DIR      where to write JSONs          (default: <SCRATCH_BASE>/results)
#   SEED            sampler + sampling seed       (default: 0)
#   HF_TOKEN        optional, for gated repos     (default: unset)
#   SKIP_DATASET_CHECK  1 = skip the dataset preflight warning
#                                                  (default: 0)

# ---------------------------------------------------------------- SLURM
#SBATCH --job-name=vllm-moe-eval
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=0-03:55:00
#SBATCH --output=spartan/logs/vllm-moe-eval-%j.out
#SBATCH --error=spartan/logs/vllm-moe-eval-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pratheek.senevirathne@student.unimelb.edu.au

set -euo pipefail

# ---------------------------------------------------------------- paths
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "${SLURM_SUBMIT_DIR}/spartan/logs"

# Default values (overridable via env before sbatch). Module names are
# taken from the canonical gpu-a100 / gpu-a100-short EasyBuild tree.
#   CUDA_MODULE     CUDA/12.4.1        (from Core)
#   PYTHON_MODULE   Python/3.11.3      (from Compiler/GCCcore/11.3.0)
#   GCC_MODULE      GCCcore/11.3.0     (from Core; older compiler that
#                                     Python/3.11.3 is built against)
# Load order matters: GCCcore before Python (Python/3.11.3 depends on
# it), CUDA independently (Python/3.11.3 has CUDA-aware numpy/scipy
# wheels but doesn't require CUDA at runtime). If your partition shows
# different versions, override via the env vars above.
CUDA_MODULE="${CUDA_MODULE:-CUDA/12.4.1}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3}"
GCC_MODULE="${GCC_MODULE:-GCCcore/11.3.0}"

REPO_ROOT="${REPO_ROOT:-/data/gpfs/projects/uom00014/vllm}"
SCRATCH_BASE="${SCRATCH_BASE:-${SCRATCH:-/data/scratch/projects/uom00014}/vllm}"

VENV_DIR="${SCRATCH_BASE}/venv"
DATASETS_DIR="${SCRATCH_BASE}/datasets"
RESULTS_DIR="${SCRATCH_BASE}/results"
HF_CACHE="${SCRATCH_BASE}/hf_cache"

mkdir -p "${DATASETS_DIR}" "${RESULTS_DIR}" "${HF_CACHE}" "${VENV_DIR}"

# Analyzer knobs (forwarded to moe_expert_stats.py). Defaults match the
# script's own DEFAULT_* constants, so unset = run the canonical OLMoE
# analysis workload. Override any of these via --export=ALL,NAME=value.
# MODELS_OVERRIDE and QUANTS_OVERRIDE are space-separated lists; the
# analyzer's --models and --quant flags are nargs="+". Cells are
# evaluated as the Cartesian product of (MODELS x QUANTS).
MODELS_OVERRIDE="${MODELS_OVERRIDE:-allenai/OLMoE-1B-7B-0924-Instruct}"
QUANTS_OVERRIDE="${QUANTS_OVERRIDE:-}"
BENCHMARKS="${BENCHMARKS:-mmlu bbh humaneval}"
NUM_SAMPLES="${NUM_SAMPLES:-256}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
ENFORCE_EAGER="${ENFORCE_EAGER:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_DIR}}"
SEED="${SEED:-0}"

# ---------------------------------------------------------------- banner
echo "============================================================"
echo "[$(date -Iseconds)] vllm-moe-eval starting"
echo "[$(date -Iseconds)] job_id        = ${SLURM_JOB_ID:-}"
echo "[$(date -Iseconds)] node          = ${SLURM_JOB_NODELIST:-}"
echo "[$(date -Iseconds)] cpus          = ${SLURM_CPUS_PER_TASK:-}"
echo "[$(date -Iseconds)] gpus          = ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[$(date -Iseconds)] REPO_ROOT     = ${REPO_ROOT}"
echo "[$(date -Iseconds)] SCRATCH_BASE  = ${SCRATCH_BASE}"
echo "[$(date -Iseconds)] HF_HOME       = ${HF_CACHE}"
echo "[$(date -Iseconds)] MODELS        = ${MODELS_OVERRIDE}"
echo "[$(date -Iseconds)] QUANTS        = ${QUANTS_OVERRIDE:-<unset>}"
echo "[$(date -Iseconds)] BENCHMARKS    = ${BENCHMARKS}"
echo "[$(date -Iseconds)] NUM_SAMPLES   = ${NUM_SAMPLES}"
echo "[$(date -Iseconds)] MAX_TOKENS    = ${MAX_TOKENS}"
echo "[$(date -Iseconds)] MAX_MODEL_LEN = ${MAX_MODEL_LEN}"
echo "[$(date -Iseconds)] GPU_MEM_UTIL  = ${GPU_MEM_UTIL}"
echo "[$(date -Iseconds)] ENFORCE_EAGER = ${ENFORCE_EAGER:-<unset>}"
echo "[$(date -Iseconds)] OUTPUT_DIR    = ${OUTPUT_DIR}"
echo "[$(date -Iseconds)] SEED          = ${SEED}"
echo "[$(date -Iseconds)] CUDA_MODULE   = ${CUDA_MODULE}"
echo "[$(date -Iseconds)] PYTHON_MODULE = ${PYTHON_MODULE}"
echo "[$(date -Iseconds)] GCC_MODULE    = ${GCC_MODULE}"
echo "============================================================"

# ---------------------------------------------------------------- modules
# Preflight: error out with a clear message if any *_MODULE variable is
# empty. Without this, an empty CUDA_MODULE would silently become
# `module load ""` which Lmod might accept as a no-op - giving the user
# a confusing later error (no nvcc on PATH) instead of the right one.
if [[ -z "${GCC_MODULE}" ]]; then
    echo "[fatal] GCC_MODULE is empty. Run 'module avail gcc' on a gpu-a100-short node and export GCC_MODULE=<name> before sbatch." >&2
    exit 3
fi
if [[ -z "${PYTHON_MODULE}" ]]; then
    echo "[fatal] PYTHON_MODULE is empty. Run 'module avail python' on a gpu-a100-short node and export PYTHON_MODULE=<name> before sbatch." >&2
    exit 3
fi
if [[ -z "${CUDA_MODULE}" ]]; then
    echo "[fatal] CUDA_MODULE is empty. Run 'module avail cuda' on a gpu-a100-short node and export CUDA_MODULE=<name> before sbatch." >&2
    exit 3
fi

# `set -u` would crash on the module command if it isn't installed; we
# unconditionally `module purge` first so we know exactly what we're
# getting. `load_module_or_die` wraps `module load` so a dependency
# failure surfaces as a clear fatal message.
set +u
module purge
load_module_or_die() {
    local name="$1"
    local stderr_log
    stderr_log="$(mktemp)"
    if ! module load "${name}" 2>"${stderr_log}"; then
        echo "[fatal] module load ${name} failed:" >&2
        sed 's/^/  /' "${stderr_log}" >&2
        echo "[fatal] Hint: run 'module spider ${name}' on a gpu-a100-short node to see the load chain." >&2
        rm -f "${stderr_log}"
        exit 3
    fi
    rm -f "${stderr_log}"
}
load_module_or_die "${GCC_MODULE}"
load_module_or_die "${PYTHON_MODULE}"
load_module_or_die "${CUDA_MODULE}"
module list
set -u

# Post-load sanity check: make sure the binaries we expect are actually
# on PATH.
if ! command -v python3 >/dev/null 2>&1; then
    echo "[fatal] python3 not on PATH after loading ${PYTHON_MODULE}. Pick a different PYTHON_MODULE." >&2
    exit 3
fi
if ! command -v gcc >/dev/null 2>&1; then
    echo "[fatal] gcc not on PATH after loading ${GCC_MODULE}. Pick a different GCC_MODULE." >&2
    exit 3
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "[warn] nvcc not on PATH after loading ${CUDA_MODULE} - vLLM may still work via its bundled CUDA runtime, but the .sbatch's CUDA sanity check below will fail"
fi
echo "[info] python: $(python3 --version)"
echo "[info] gcc   : $(gcc --version | head -1)"
echo "[info] nvcc  : $(nvcc --version 2>/dev/null | tail -1 || echo '<not on PATH>')"

# Make sure C++ sees the loaded GCC and that nvcc is on PATH.
export CC="${CC:-$(command -v gcc)}"
export CXX="${CXX:-$(command -v g++)}"
if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${CUDA_HOME:-/usr/local/cuda}/lib64"

# OpenMP: per the Spartan docs, set OMP_NUM_THREADS to match the SLURM
# requested CPUs so multi-threaded math libraries don't oversubscribe.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ----------------------------------------------- HF cache for the analyzer
# vLLM's HF resolver reads $HF_HOME (and $HUGGINGFACE_HUB_CACHE) before
# falling back to $HOME/.cache/huggingface. By exporting both here, we
# guarantee the model weights and the dataset cache (if anything slipped
# through the pre-download) land under the scratch volume rather than
# the GPFS project volume.
export HF_HOME="${HF_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}"

# (NB: no `VLLM_USE_V1` here - that env var was removed in vLLM 0.7+,
# and newer versions log a WARNING + may misbehave if it's set. The
# V1 engine is the only engine in current vLLM.)

# ---------------------------------------------------------------- preflight
# 1. Make sure the repo is in place.
if [[ ! -d "${REPO_ROOT}/vllm" ]]; then
    echo "[fatal] REPO_ROOT=${REPO_ROOT} does not look like a vllm checkout (no vllm/ subdir)" >&2
    exit 2
fi
if [[ ! -f "${REPO_ROOT}/examples/eval-moe/moe_expert_stats.py" ]]; then
    echo "[fatal] ${REPO_ROOT}/examples/eval-moe/moe_expert_stats.py missing - the analyzer was renamed or moved" >&2
    exit 2
fi

# 2. Quick CUDA sanity check.
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
else
    echo "[warn] nvidia-smi not found on PATH; CUDA may not be active"
fi

# 3. Make sure the datasets have been pre-populated. GPU compute nodes
#    on Spartan are firewalled off from the public internet AND don't
#    ship the `datasets` python package, so we MUST have staged these
#    from the login node. If any expected dataset is missing, warn
#    loudly (don't hard-fail - the analyzer's load_dataset will retry
#    and fail with a clearer error).
if [[ "${SKIP_DATASET_CHECK:-0}" -ne 1 ]]; then
    missing_datasets=()
    for ds in ${BENCHMARKS}; do
        case "${ds}" in
            mmlu)
                # Needs both test + dev splits.
                if [[ ! -d "${DATASETS_DIR}/mmlu/test" || ! -d "${DATASETS_DIR}/mmlu/dev" ]]; then
                    missing_datasets+=("${ds}")
                fi
                ;;
            bbh)
                # Needs >=1 sub-task subdirectory.
                n_cfg=$(find "${DATASETS_DIR}/bbh" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
                if [[ "${n_cfg}" -eq 0 ]]; then
                    missing_datasets+=("${ds}")
                fi
                ;;
            humaneval)
                if [[ ! -d "${DATASETS_DIR}/humaneval/test" ]]; then
                    missing_datasets+=("${ds}")
                fi
                ;;
            *)
                echo "[warn] BENCHMARKS contains unknown dataset '${ds}'; analyzer will refuse" >&2
                ;;
        esac
    done
    if [[ ${#missing_datasets[@]} -gt 0 ]]; then
        echo "[warn] ${#missing_datasets[@]} dataset(s) not staged on scratch: ${missing_datasets[*]}"
        echo "[warn]   (looked under ${DATASETS_DIR}/<ds>/)"
        echo "[warn] Run on the login node first:"
        echo "[warn]   bash spartan/download-datasets.sh --datasets ${missing_datasets[*]}"
        echo "[warn] The GPU job will continue and let the analyzer try+fail,"
        echo "[warn] but staging on the login node avoids wasted compute-quota time."
    fi
fi

# 4. Ensure a venv exists with vllm + numpy + datasets. We create the
#    venv on the SCRATCH volume (not in the repo) so it survives across
#    job submissions. If a usable venv already exists we reuse it;
#    otherwise we create one and pip install vllm into it.
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[info] creating venv at ${VENV_DIR} ..."
    # --system-site-packages lets us reuse the module-loaded Python's
    # CUDA-aware numpy/scipy wheels without re-downloading them.
    python3 -m venv --system-site-packages "${VENV_DIR}"
    echo "[info] venv created"
fi
# Always re-export PATH so the rest of the script picks up the venv's
# python (which has vllm installed). pip / python go through this venv.
# We also pull `uv` (the Astral pip replacement vLLM uses for its own
# install instructions - see AGENTS.md §"Installing dependencies")
# onto PATH if it's missing.
export PATH="${VENV_DIR}/bin:${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
    echo "[info] uv not on PATH; installing to ~/.local ..."
    err="$(mktemp)"
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh 2>"${err}"; then
        echo "[fatal] uv install failed:" >&2
        sed 's/^/  /' "${err}" >&2
        rm -f "${err}"
        exit 4
    fi
    rm -f "${err}"
    export PATH="${HOME}/.local/bin:${PATH}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Make sure the analyzer's hard dependencies are present. The CRITICAL
# one is vLLM: we install it editable from $REPO_ROOT with
# VLLM_USE_PRECOMPILED=1 so the prebuilt CUDA wheels land in the venv
# (a plain `pip install vllm` would pull the CPU-only PyPI wheel and
# fail at import with `No module named 'vllm._C_stable_libtorch'`).
# `numpy` and `datasets` are pure-Python and are added to the same
# `uv pip install` so they're all version-aligned in the lockfile.
echo "[info] installing vllm + numpy + datasets from $REPO_ROOT (editable) ..."
if ! VLLM_USE_PRECOMPILED=1 uv pip install \
        --python "${VENV_DIR}/bin/python" \
        "numpy" \
        "datasets" \
        -e "${REPO_ROOT}" \
        --torch-backend=auto 2>"${VENV_DIR}/install.err"; then
    echo "[fatal] uv pip install failed:" >&2
    sed 's/^/  /' "${VENV_DIR}/install.err" >&2
    exit 4
fi
echo "[info] deps installed"

# Final venv sanity check.
if ! python3 -c "import vllm, numpy, datasets; print(f'vllm={vllm.__version__} numpy={numpy.__version__} datasets={datasets.__version__}')" 2>/dev/null; then
    echo "[fatal] venv is broken; pip install reported success but import failed" >&2
    exit 4
fi

# ---------------------------------------------------------------- run
# Compose the analyzer's argv. Each ${VAR:+...} guard means the flag
# only appears when its env var is set, so unset vars fall back to the
# script's own defaults. ENFORCE_EAGER is a flag-style arg, so we use
# a separate conditional.
RUN_ARGS=(
    --models ${MODELS_OVERRIDE}
    --benchmarks ${BENCHMARKS}
    --num-samples "${NUM_SAMPLES}"
    --max-tokens "${MAX_TOKENS}"
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEM_UTIL}"
    --output-dir "${OUTPUT_DIR}"
    --seed "${SEED}"
    --dtype "auto"
)
if [[ -n "${QUANTS_OVERRIDE}" ]]; then
    RUN_ARGS+=( --quant ${QUANTS_OVERRIDE} )
fi
if [[ -n "${ENFORCE_EAGER}" ]]; then
    RUN_ARGS+=( --enforce-eager )
fi

echo "============================================================"
echo "[$(date -Iseconds)] running: python examples/eval-moe/moe_expert_stats.py ${RUN_ARGS[*]}"
echo "============================================================"

cd "${REPO_ROOT}"
python3 examples/eval-moe/moe_expert_stats.py "${RUN_ARGS[@]}"

echo "============================================================"
echo "[$(date -Iseconds)] vllm-moe-eval finished (rc=$?)"
echo "results root: ${OUTPUT_DIR}"
echo "hf cache    : ${HF_CACHE}"
echo "logs        : ${SLURM_SUBMIT_DIR}/spartan/logs/vllm-moe-eval-${SLURM_JOB_ID}.{out,err}"
echo "============================================================"
