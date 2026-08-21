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
#   TORCH_BACKEND   uv --torch-backend tag         (default: cu129)
#                                                  Matches a PyTorch CUDA wheel index that
#                                                  contains the torch version pinned in
#                                                  pyproject.toml's [build-system].requires
#                                                  (torch==2.13.0 at time of writing).
#                                                  cu124 maxes at torch 2.6.0, so the
#                                                  default is cu129 (the only index that
#                                                  publishes torch 2.13.0 today). The
#                                                  wheel is forward-compatible with the
#                                                  CUDA 12.4.1 driver/runtime on Spartan
#                                                  as long as `nvidia-smi` reports CUDA
#                                                  >= 12.9. Override with e.g. cu118 for
#                                                  CUDA/11.8.x if your node has it, BUT
#                                                  you'll then need a vllm branch whose
#                                                  pyproject pins a torch version that
#                                                  exists in that index.
#   NCCL_MODULE     Lmod module name for NCCL     (default: <unset>)
#                                                  Auto-detected from `module avail NCCL/`
#                                                  by matching the loaded CUDA major.minor.
#                                                  Loaded automatically when NGPUS > 1
#                                                  (vLLM's tensor-parallel engine needs NCCL
#                                                  for cross-GPU collectives). Override
#                                                  before sbatch if the probe picks the
#                                                  wrong build.
#   NGPUS           number of GPUs the job uses   (default: 1)
#                                                  Must match `#SBATCH --gres=gpu:N` (that
#                                                  directive can't reference shell vars).
#                                                  Forwarded to vLLM as
#                                                  `--tensor-parallel-size ${NGPUS}`.
#   SPLIT_MODE      vLLM distributed-executor mode (default: layer)
#                                                  "layer" = pipeline parallel across GPUs;
#                                                  "tensor" / "row" = weight parallel (slower
#                                                  over PCIe without NVLink); "none" = single
#                                                  GPU mode regardless of allocation.
#   EXTRA_TENSOR_SPLIT  comma-separated fraction  (default: <unset>)
#                                                  Forwarded to vLLM as `--tensor-split`.
#                                                  e.g. "50,50" for 2 GPUs, "25,25,25,25"
#                                                  for 4. Unset = auto-derive a uniform
#                                                  split so multi-GPU actually spreads
#                                                  layers instead of falling back to
#                                                  all-on-GPU-0.
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
# IMPORTANT: `#SBATCH --gres=gpu:N` cannot use a shell variable, so
# edit the value below AND keep NGPUS in sync (the body uses NGPUS
# to auto-derive the tensor split, etc.). Default 1 = single GPU.
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
# uv's `--torch-backend` tag. MUST point at a PyTorch index that
# publishes the torch version pinned in pyproject.toml's
# [build-system].requires (currently torch==2.13.0). cu124 maxes at
# torch 2.6.0; cu128 maxes at torch 2.11.0; only cu129 publishes
# torch 2.13.0 today. The wheel's runtime is forward-compatible with
# CUDA 12.4.1 toolkits as long as `nvidia-smi` reports driver CUDA
# >= 12.9. Override with cu118 / cu126 / etc. ONLY if your vllm
# branch's pyproject pins a torch version that exists in that index.
TORCH_BACKEND="${TORCH_BACKEND:-cu129}"
# NCCL_MODULE is auto-detected (probed from `module avail NCCL/`)
# but NOT loaded by default. Loaded automatically when NGPUS > 1,
# since vLLM's tensor-parallel engine needs NCCL for cross-GPU
# collectives. Override before sbatch if the probe picks the wrong one.
NCCL_MODULE="${NCCL_MODULE:-}"
# Number of GPUs the job requests. Mirrors `#SBATCH --gres=gpu:`;
# update both together.
NGPUS="${NGPUS:-1}"
# vLLM `--split-mode` for tensor parallelism (forwarded to LLM as
# `distributed_executor_backend` for some paths, or to the engine).
# "layer" shards layers across GPUs (pipeline parallel);
# "row" / "tensor" do weight-parallel sharding (slower over PCIe).
SPLIT_MODE="${SPLIT_MODE:-layer}"
# Comma-separated fraction list summing to ~1.0 (e.g. "50,50" for 2
# GPUs). Empty = auto-derive a uniform split so multi-GPU actually
# uses the GPUs. Set to "none" to force single-GPU on a multi-GPU
# allocation (useful for A/B testing).
EXTRA_TENSOR_SPLIT="${EXTRA_TENSOR_SPLIT:-}"

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
echo "[$(date -Iseconds)] gpus          = ${CUDA_VISIBLE_DEVICES:-<unset>} (NGPUS=${NGPUS})"
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
echo "[$(date -Iseconds)] TORCH_BACKEND = ${TORCH_BACKEND}"
echo "[$(date -Iseconds)] NCCL_MODULE   = ${NCCL_MODULE:-<auto-probed below>}"
echo "[$(date -Iseconds)] SPLIT_MODE    = ${SPLIT_MODE}"
echo "[$(date -Iseconds)] EXTRA_TENSOR_SPLIT = ${EXTRA_TENSOR_SPLIT:-<auto-derived for NGPUS=${NGPUS}>}"
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

# Auto-detect the default NCCL module that matches the loaded CUDA.
# `module avail NCCL/...-CUDA-<X>.<Y>` typically lays modules out in
# 3 columns per line, so we have to scan each whitespace-separated
# token individually rather than rely on awk on the whole line.
# Prefers the default version (the one Lmod marks with `(D)`) when
# multiple builds match. Echoes the picked module name to stdout;
# empty output means no match found.
#
# Implementation note: bash's `[[ ... =~ ... ]]` regex has subtle
# quoting rules when the pattern contains variables, so we use
# `grep -E` for portable, predictable matching.
#
# Usage:
#     nccl="$(auto_nccl_for_cuda CUDA/12.4.1)"
auto_nccl_for_cuda() {
    local cuda_mod="$1"
    # CUDA/12.4.1 -> 12.4 (major.minor)
    local cuda_ver="${cuda_mod#CUDA/}"
    cuda_ver="${cuda_ver%.*}"
    local avail_out
    avail_out="$(module avail NCCL/ 2>/dev/null || true)"
    local first_match="" default_match=""
    local line tok prev_match_tok
    while IFS= read -r line; do
        # Skip lines that don't contain any NCCL/ token.
        [[ "${line}" == *"NCCL/"* ]] || continue
        # Walk the whitespace-separated tokens. We track
        # `prev_match_tok` = the most recently seen NCCL/... token
        # whose name matched our CUDA version regex. When a "(D)"
        # token appears AFTER such a token, we retroactively promote
        # that matched token to `default_match`.
        prev_match_tok=""
        for tok in ${line}; do
            if [[ "${tok}" == "(D)" ]]; then
                # If the previous matching token exists and we
                # haven't already promoted one, promote it now.
                if [[ -n "${prev_match_tok}" && -z "${default_match}" ]]; then
                    default_match="${prev_match_tok}"
                    prev_match_tok=""  # consumed
                fi
                continue
            fi
            local stripped="${tok%*(D)}"
            # Match e.g. NCCL/2.22.3-CUDA-12.4.1 - the trailing dot
            # before the patch number is mandatory so we don't
            # accidentally pick up a build with the same major.minor
            # but for a different CUDA (CUDA-12.4 vs CUDA-12.4.1).
            if ! echo "${stripped}" | grep -qE "NCCL/[^ ]*-CUDA-${cuda_ver//./\\.}\\.[0-9]"; then
                # Non-NCCL/non-(D) token (e.g. UCX-CUDA/...). Reset
                # the prev_match_tok so a later (D) doesn't promote
                # something across a non-NCCL entry.
                prev_match_tok=""
                continue
            fi
            # Glued-on (D) marker: e.g. NCCL/X(D) is one token. We
            # already stripped it above, so we know it's the match.
            if [[ "${tok}" == *"(D)"* && -z "${default_match}" ]]; then
                default_match="${stripped}"
                prev_match_tok=""
                continue
            fi
            if [[ -z "${first_match}" ]]; then
                first_match="${stripped}"
            fi
            prev_match_tok="${stripped}"
        done
    done <<<"${avail_out}"
    if [[ -n "${default_match}" ]]; then
        printf '%s\n' "${default_match}"
    elif [[ -n "${first_match}" ]]; then
        printf '%s\n' "${first_match}"
    fi
}
# If the user set NCCL_MODULE explicitly, respect it. Otherwise probe
# the available NCCL builds and pick the one matching ${CUDA_MODULE}.
if [[ -z "${NCCL_MODULE}" ]]; then
    probed_nccl="$(auto_nccl_for_cuda "${CUDA_MODULE}")"
    if [[ -n "${probed_nccl}" ]]; then
        NCCL_MODULE="${probed_nccl}"
        echo "[info] auto-detected NCCL_MODULE=${NCCL_MODULE} for ${CUDA_MODULE} (not loaded by default - opt in with NCCL_MODULE on the sbatch command line)"
    else
        echo "[info] no NCCL/...-CUDA-${CUDA_MODULE#CUDA/} found via 'module avail' - skipping (vLLM doesn't need NCCL on a single-GPU job)"
    fi
fi

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
# Capture the active Python version once (e.g. 3.11 -> "311") so the
# torch-wheel preflight below can build the cp${X}-cp${X} substring
# that matches PyTorch's manylinux wheel filenames.
PYTHON_VER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "")"
PYTHON_VER_NO_DOT="$(python3 -c 'import sys; print("%d%d" % sys.version_info[:2])' 2>/dev/null || echo "")"

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

# ------------------------------------------- multi-GPU / NVLink handling
# Only relevant when NGPUS > 1. vLLM's tensor-parallel engine needs
# NCCL for cross-GPU collectives. We load the auto-detected NCCL
# module here (the auto-detect probe runs above). If the probe
# couldn't find an NCCL module but the user asked for multi-GPU, we
# fatal out with a clear message.
if [[ "${NGPUS}" -gt 1 ]]; then
    if [[ -z "${NCCL_MODULE}" ]]; then
        echo "[fatal] NGPUS=${NGPUS} > 1 but no NCCL module is available for ${CUDA_MODULE}." >&2
        echo "[fatal] Run on a login node: module avail NCCL/ to find one, then export NCCL_MODULE=<name> before sbatch." >&2
        exit 5
    fi
    echo "[info] loading NCCL module ${NCCL_MODULE} (NGPUS=${NGPUS} requires it for cross-GPU collectives) ..."
    load_module_or_die "${NCCL_MODULE}"
    # EasyBuild exposes NCCL's install prefix via ${EBROOTNCCL}; set
    # both NCCL_HOME (which llama.cpp's FindNCCL looks at) and add
    # lib to LD_LIBRARY_PATH so runtime dlopen works too.
    if [[ -n "${EBROOTNCCL:-}" ]]; then
        export NCCL_HOME="${NCCL_HOME:-${EBROOTNCCL}}"
        export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH}"
    fi
    echo "[info] nccl: EBROOTNCCL=${EBROOTNCCL:-<unset>}  NCCL_HOME=${NCCL_HOME:-<unset>}"

    # Probe GPU topology via nvidia-smi topo -m. NVLink between GPUs
    # gives ~600 GB/s tensor-parallel BW on A100; PCIe only gives
    # ~32 GB/s and is generally too slow for >70B models. We log the
    # result so a future failure is easier to diagnose; vLLM doesn't
    # expose a per-tensor-parallel-mode switch at the Python level
    # but the choice of split-mode below matters.
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
        echo "[info] GPU topology probe:"
        nvidia-smi topo -m 2>/dev/null | head -n "$((NGPUS + 1))" || true
        # Auto-set EXTRA_TENSOR_SPLIT when the user didn't pass one
        # AND we have multiple GPUs to spread across. We compute a
        # uniform split: 50/50 for 2 GPUs, 25/25/25/25 for 4, etc.
        if [[ -z "${EXTRA_TENSOR_SPLIT}" && "${SPLIT_MODE}" != "none" ]]; then
            _auto_frac="$(awk -v n="${NGPUS}" 'BEGIN{ printf "%.4f", 100.0/n }')"
            _auto_split=""
            _auto_i=0
            while [[ ${_auto_i} -lt ${NGPUS} ]]; do
                _auto_split+="${_auto_frac}"
                _auto_i=$(( _auto_i + 1 ))
                [[ ${_auto_i} -lt ${NGPUS} ]] && _auto_split+=","
            done
            EXTRA_TENSOR_SPLIT="${_auto_split}"
            echo "[info] auto-derived EXTRA_TENSOR_SPLIT=${EXTRA_TENSOR_SPLIT} for NGPUS=${NGPUS}"
            unset _auto_frac _auto_split _auto_i
        fi
    fi

    # vLLM's V1 engine spins up its own process group internally;
    # we don't need to set RANK / WORLD_SIZE manually here.
else
    # Single-GPU: skip NCCL + tensor split entirely.
    :
fi

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

# 2. Quick CUDA sanity check. We print both the toolkit version
#    (from `nvcc --version`, controlled by ${CUDA_MODULE}) and the
#    driver version (from `nvidia-smi`, baked into the node's kernel
#    modules). PyTorch wheels are built against a specific CUDA
#    version (cu${X} = the toolkit the wheel was compiled with) but
#    require a driver CUDA >= that version at runtime -- so the
#    driver version is what actually has to be >= ${TORCH_BACKEND}'s
#    cu-major.minor.
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
    _drv_cuda="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
    if [[ -n "${_drv_cuda}" ]]; then
        echo "[info] nvidia driver (supports CUDA runtime up to): ${_drv_cuda}"
    fi
    unset _drv_cuda
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
#    otherwise we create one and uv-pip-install vllm into it.
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[info] creating venv at ${VENV_DIR} ..."
    # NB: deliberately NOT --system-site-packages. On Spartan the
    # module-loaded Python/3.11.3 bundles its own `torch` + `vllm` in
    # /apps/easybuild-..., and `--system-site-packages` would let
    # those leak into our venv ahead of the editable install - so
    # `import vllm` inside the venv would resolve to the EasyBuild
    # copy (which may not match the editable source we just installed).
    # A clean venv + `uv pip install --torch-backend=cu124` (below)
    # gives us a self-contained venv with the right CUDA torch.
    python3 -m venv "${VENV_DIR}"
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

# Preflight: confirm the torch version pinned in pyproject.toml's
# [build-system].requires is actually published on the chosen
# PyTorch index. The full install below would otherwise fail with
# `Because there is no version of torch==2.13.0 and you require
# torch==2.13.0, we can conclude that your requirements are
# unsatisfiable.` -- which is correct, but only AFTER `uv` has spent
# ~30s resolving 200+ transitive packages. Doing the check up-front
# saves the wasted time AND lets us print an actionable error
# pointing at the right TORCH_BACKEND value.
_pin_probe="$(awk -F '"' '/^requires/ {flag=1; next} flag && /"torch *==/ {gsub(/[", ]/, "", $2); print $2; exit} flag && /^]/ {exit}' \
    "${REPO_ROOT}/pyproject.toml" 2>/dev/null || true)"
_pin_ver="${_pin_probe##*==}"
if [[ -n "${_pin_ver}" && "${_pin_ver}" != "${_pin_probe}" ]]; then
    # The pytorch.org simple index URL for --torch-backend=<TAG> is
    # https://download.pytorch.org/whl/<TAG>/torch/. Fetch the index
    # page and look for a wheel filename matching the pinned version.
    # We avoid HEAD requests (some CDNs return 403) and just parse the
    # HTML index page, which PyTorch always serves.
    _index_url="https://download.pytorch.org/whl/${TORCH_BACKEND}/torch/"
    _hits="$(curl -fsSL --max-time 15 "${_index_url}" 2>/dev/null \
        | grep -oE "torch-${_pin_ver//./\\.}\\+${TORCH_BACKEND}-cp${PYTHON_VER_NO_DOT}-cp${PYTHON_VER_NO_DOT}-manylinux_2_28_x86_64\\.whl" \
        | head -1 || true)"
    if [[ -z "${_hits}" ]]; then
        echo "[fatal] pyproject.toml pins torch==${_pin_ver}, but the ${TORCH_BACKEND} PyTorch index has no such wheel for cp${PYTHON_VER_NO_DOT} (Python ${PYTHON_VER})." >&2
        echo "[fatal]   searched: ${_index_url}" >&2
        echo "[fatal] Most likely causes:" >&2
        echo "[fatal]   - ${TORCH_BACKEND} is wrong for this vllm branch (cu124 maxes at torch 2.6.0; cu128 maxes at 2.11.0; only cu129 publishes torch 2.13.0 today)." >&2
        echo "[fatal]   - this vllm checkout pins a torch version that doesn't exist on any PyTorch index (then you need an older vllm or a torch index patch)." >&2
        echo "[fatal] Fix: re-submit with TORCH_BACKEND set to the tag whose index publishes torch==${_pin_ver}. See https://download.pytorch.org/whl/" >&2
        exit 4
    fi
    echo "[info] pyproject pins torch==${_pin_ver}; ${TORCH_BACKEND} index has the cp${PYTHON_VER_NO_DOT} wheel. Continuing."
    unset _pin_probe _pin_ver _index_url _hits
else
    unset _pin_probe _pin_ver
fi

# Make sure the analyzer's hard dependencies are present. The CRITICAL
# one is vLLM: we install it editable from $REPO_ROOT with
# VLLM_USE_PRECOMPILED=1 so the prebuilt CUDA wheels land in the venv
# (a plain `pip install vllm` would pull the CPU-only PyPI wheel and
# fail at import with `No module named 'vllm._C_stable_libtorch'`).
# `numpy` and `datasets` are pure-Python and are added to the same
# `uv pip install` so they're all version-aligned in the lockfile.
echo "[info] installing vllm + numpy + datasets from $REPO_ROOT (editable) ..."
# --torch-backend=${TORCH_BACKEND} pulls torch/torchvision/torchaudio
# from pytorch.org's CUDA-X.Y index (matches the module-loaded CUDA).
# --torch-backend=auto would look at the existing torch on PATH and
# pick that - which on a fresh venv is the CPU-only PyPI wheel,
# leading to a CPU-only vllm install and the well-known
# `No module named 'vllm._C_stable_libtorch'` import error.
if ! VLLM_USE_PRECOMPILED=1 uv pip install \
        --python "${VENV_DIR}/bin/python" \
        "numpy" \
        "datasets" \
        -e "${REPO_ROOT}" \
        --torch-backend="${TORCH_BACKEND}" 2>"${VENV_DIR}/install.err"; then
    echo "[fatal] uv pip install failed:" >&2
    sed 's/^/  /' "${VENV_DIR}/install.err" >&2
    exit 4
fi
echo "[info] deps installed"

# Bundle-path setup for CUDA shared libs. The PyTorch CUDA wheels
# install a tree of `nvidia/<pkg>/lib/` directories inside the venv's
# site-packages -- for cu129 that's `nvidia/cu13/lib/libcudart.so.13`
# (CUDA 13 runtime), `nvidia/cu13/lib/libnvrtc.so.13`, and various
# cu12 helper libs (cublas, cufft, curand, cusparse, etc.). The
# dynamic linker doesn't search those paths by default, and vLLM's
# `_C_stable_libtorch.so` links `libcudart.so.13` directly -- so
# without this, the first `import vllm` raises
# `ImportError: libcudart.so.13: cannot open shared object file`.
#
# We walk the venv's site-packages, collect every `nvidia/*/lib`
# directory that actually exists, and prepend them to LD_LIBRARY_PATH.
# Doing it via a glob (rather than hard-coding paths) means this also
# picks up future nvidia-* packages without code changes.
_nvidia_lib_paths=""
# Resolve `${VENV}/lib` from the installed python rather than guessing.
# On CPython 3.10+ this is `lib/python3.X/site-packages/`; on older
# layouts (or `uv venv --relocatable`) it can be `lib/site-packages/`.
# The discovery loop below uses `*/site-packages` so either works.
_venv_lib_root="$("${VENV_DIR}/bin/python" -c 'import sys, os; p=os.path.dirname(os.path.dirname(sys.executable)); print(os.path.join(p, "lib"))' 2>/dev/null || true)"
if [[ -n "${_venv_lib_root}" && -d "${_venv_lib_root}" ]]; then
    # CPython 3.11 uses `python3.11/` directly; CPython 3.10 used
    # `python3.10/`; the `*/site-packages` glob below handles both.
    while IFS= read -r _nvidia_lib; do
        if [[ -n "${_nvidia_lib_paths}" ]]; then
            _nvidia_lib_paths="${_nvidia_lib}:${_nvidia_lib_paths}"
        else
            _nvidia_lib_paths="${_nvidia_lib}"
        fi
    done < <(find "${_venv_lib_root}" -maxdepth 5 -type d -path '*/nvidia/*/lib' 2>/dev/null | sort -u)
fi
if [[ -n "${_nvidia_lib_paths}" ]]; then
    export LD_LIBRARY_PATH="${_nvidia_lib_paths}:${LD_LIBRARY_PATH:-}"
    echo "[info] LD_LIBRARY_PATH prepended with $(echo "${_nvidia_lib_paths}" | tr ':' '\n' | wc -l) nvidia lib dirs (e.g. libcudart.so.13, libnvrtc.so.13, libcublas.so.12)"
else
    echo "[warn] no nvidia/*/lib dirs found under ${_venv_lib_root}; libcudart.so.13 may be missing at import time"
fi
unset _nvidia_lib_paths _venv_lib_root _nvidia_lib

# Final venv sanity check. We capture and print the actual import
# error (rather than swallowing it with `2>/dev/null`) so a future
# failure points at the root cause immediately. We use the venv's
# python explicitly (not bare `python3`) so we never accidentally
# fall back to the module-loaded EasyBuild Python if PATH ever loses
# its venv prefix.
if ! "${VENV_DIR}/bin/python" -c \
    "import vllm, numpy, datasets; print(f'vllm={vllm.__version__} numpy={numpy.__version__} datasets={datasets.__version__}')" \
    >"${VENV_DIR}/sanity.out" 2>&1; then
    echo "[fatal] venv is broken; uv pip install reported success but the sanity import failed." >&2
    echo "[fatal] import output:" >&2
    sed 's/^/  /' "${VENV_DIR}/sanity.out" >&2
    echo "[fatal] common causes:" >&2
    echo "[fatal]   - ${TORCH_BACKEND} wheel links a CUDA runtime (e.g. libcudart.so.13) that isn't on LD_LIBRARY_PATH. The fix above should have prepended the venv's nvidia/*/lib paths; if not, check that the cu${TORCH_BACKEND#cu}/lib subdir exists under the venv's site-packages." >&2
    echo "[fatal]   - the editable install landed in a different Python than ${VENV_DIR}/bin/python; verify with: ls -la ${VENV_DIR}/bin/python*" >&2
    echo "[fatal]   - the module-loaded EasyBuild Python (${PYTHON_MODULE}) is shadowing the venv; explicitly call \${VENV_DIR}/bin/python instead of bare python3." >&2
    exit 4
fi
cat "${VENV_DIR}/sanity.out"

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
    --tensor-parallel-size "${NGPUS}"
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
# Use the venv's python explicitly. `source activate` already put it
# at the front of PATH, but if anything between the activate and here
# rewrote PATH (e.g. a stray module load), `python3` could resolve to
# the EasyBuild Python instead, which has its own (different) vllm
# install AND doesn't have libcudart.so.13 on its loader path. Using
# the absolute path makes the analyzer invocation independent of PATH.
"${VENV_DIR}/bin/python" examples/eval-moe/moe_expert_stats.py "${RUN_ARGS[@]}"

echo "============================================================"
echo "[$(date -Iseconds)] vllm-moe-eval finished (rc=$?)"
echo "results root: ${OUTPUT_DIR}"
echo "hf cache    : ${HF_CACHE}"
echo "logs        : ${SLURM_SUBMIT_DIR}/spartan/logs/vllm-moe-eval-${SLURM_JOB_ID}.{out,err}"
echo "============================================================"
