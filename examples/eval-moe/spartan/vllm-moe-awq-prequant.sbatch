#!/usr/bin/env bash
#
# spartan/vllm-moe-awq-prequant.sbatch - SLURM job that pre-quantizes a
# single bf16 MoE model to int4 (W4A16) using llm-compressor's AWQ + GPTQ
# pipeline. Output goes to ${SCRATCH}/hf_cache_awq/<model_safe>/.
#
# Submit one of these per model (parallelizable on gpu-h100):
#
#   sbatch --export=MODEL_ID=allenai/OLMoE-1B-7B-0924-Instruct \
#          spartan/vllm-moe-awq-prequant.sbatch
#   sbatch --export=MODEL_ID=deepseek-ai/deepseek-moe-16b-chat \
#          spartan/vllm-moe-awq-prequant.sbatch
#   sbatch --export=MODEL_ID=Qwen/Qwen3-30B-A3B \
#          spartan/vllm-moe-awq-prequant.sbatch
#   sbatch --export=MODEL_ID=mistralai/Mixtral-8x22B-Instruct-v0.1 \
#          spartan/vllm-moe-awq-prequant.sbatch
#
# After all 4 jobs complete, run the analyzer with --quantization moe_wna16
# pointing at each new ${SCRATCH}/hf_cache_awq/<model_safe>/ snapshot.
#
# This sbatch is fork-from-vllm-moe-eval-h100-bf16.sbatch: same module
# stack, same venv, same scratch-quota guard.

#SBATCH --job-name=vllm-moe-awq-prequant
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=720G  # Mixtral bf16 (~262 GB) + linearization buffer + python venv
#                      + tokenized calibration data. Other 3 models use far less,
#                      SLURM only enforces the cap when needed.
#SBATCH --time=0-12:00:00  # 4h was too tight: Qwen3-awq needs ~5h (48L x 128E x 20 grid),
                               # Mixtral-awq_gptq needs ~5h (56L x 8E x 2 modifiers).
                               # 12h covers worst case (Qwen3-awq_gptq) with margin.
#SBATCH --output=/data/gpfs/projects/uom00014/vllm/examples/eval-moe/spartan/logs/vllm-moe-awq-prequant-%j.out
#SBATCH --error=/data/gpfs/projects/uom00014/vllm/examples/eval-moe/spartan/logs/vllm-moe-awq-prequant-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pratheek.senevirathne@student.unimelb.edu.au

set -euo pipefail

# ---------------------------------------------------------------------------
# Module + env (matches the existing analyzer sbatch's preamble).
# ---------------------------------------------------------------------------
module purge
module load "${GCC_MODULE:-GCCcore/11.3.0}"
module load "${PYTHON_MODULE:-Python/3.11.3}"
module load "${CUDA_MODULE:-CUDA/12.4.1}"
module load "${NCCL_MODULE:-NCCL/2.22.3-CUDA-12.4.1}" 2>/dev/null || true

# SCRATCH_BASE defaults to the user's per-project scratch.
SCRATCH_BASE="${SCRATCH_BASE:-/data/scratch/projects/uom00014/vllm}"
HF_CACHE="${HF_CACHE:-${SCRATCH_BASE}/hf_cache}"

# AWQ output dir: ${SCRATCH}/hf_cache_awq/<model_safe>/
OUTPUT_BASE="${OUTPUT_BASE:-${SCRATCH_BASE}/hf_cache_awq}"

# Required env vars. Models and the calibration dataset are pre-staged
# under ${HF_CACHE} / ${SCRATCH_BASE}/datasets. GPU nodes may be firewall-
# restricted from HF Hub, so HF_HUB_OFFLINE=1 keeps all loads local.
# The pre-download-perfectblend.sh helper stages mlabonne/open-perfectblend
# parquet shards on the login node (which has Hub access) into
# ${SCRATCH_BASE}/datasets/perfectblend/.
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
# Avoid CUDA OOM on activation tensors during AWQ/GPTQ calibration.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Point llm-compression's `load_dataset("perfectblend")` at the local dir
# rather than the Hub. The datasets library resolves a local directory
# path before attempting the Hub if the dir exists.
PERFECTBLEND_DIR="${PERFECTBLEND_DIR:-${SCRATCH_BASE}/datasets/perfectblend}"
if [[ -d "${PERFECTBLEND_DIR}" ]]; then
    echo "[awq-prequant] pointing perfectblend at ${PERFECTBLEND_DIR}"
fi

# ---------------------------------------------------------------------------
# Model arg (required) + recipe arg (default awq_gptq).
# Recipe controls the int4 quantization strategy:
#   awq       - AWQ scales + plain round-to-nearest int4
#   gptq      - GPTQ Hessian-based int4 only
#   awq_gptq  - AWQ scales first, then GPTQ rounds using the scaled weights
# ---------------------------------------------------------------------------
: "${MODEL_ID:?MODEL_ID must be exported before sbatch (e.g. --export=MODEL_ID=...) }"
: "${RECIPE:=awq_gptq}"
case "${RECIPE}" in
    awq|gptq|awq_gptq) ;;
    *) echo "[fatal] invalid RECIPE='${RECIPE}' (expected awq|gptq|awq_gptq)"; exit 9 ;;
esac
MODEL_SAFE="${MODEL_ID//\//--}"
OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_SAFE}-${RECIPE}"

# ---------------------------------------------------------------------------
# Scratch quota guard (same as the analyzer sbatch).
# ---------------------------------------------------------------------------
_scratch_used_pct="$(df --output=pcent "${SCRATCH_BASE}" 2>/dev/null | tail -1 | tr -d ' %' || echo 100)"
if [[ "${_scratch_used_pct}" =~ ^[0-9]+$ ]] && [[ "${_scratch_used_pct}" -ge 90 ]]; then
    echo "[fatal] scratch quota >= 90% used (${_scratch_used_pct}%); aborting."
    exit 7
fi

# ---------------------------------------------------------------------------
# AWQ pre-quant venv (separate from analyzer venv). llmcompressor 0.8.1 has
# a hard constraint of `transformers<=4.56.2` which is incompatible with
# vLLM (requires v5+).  To work around it: install llmcompressor from the
# GitHub `main` branch (v0.14.x, requires transformers>=5.15.0) into a
# separate venv so we don't pollute the analyzer venv.
#
#   GPU node layout:
#     ${SCRATCH}/venv/         -> analyzer (torch 2.13.0+cu129, transformers 5.15.1)
#     ${SCRATCH}/venv_awq/     -> AWQ pre-quant (torch 2.13.0+cu129 + llm-compressor main)
#
# IMPORTANT: both venvs use torch==2.13.0+cu129 because the GPU driver on
# Spartan is CUDA 12.8 (version 12080). torch==2.14.x requires CUDA 13.0+
# driver and WILL NOT WORK here — model.load_state_dict etc. will raise
# "The NVIDIA driver on your system is too old (found version 12080)".
# ---------------------------------------------------------------------------
AWQ_VENV_DIR="${AWQ_VENV_DIR:-${SCRATCH_BASE}/venv_awq}"
ANALYZER_VENV_DIR="${VENV_DIR:-${SCRATCH_BASE}/venv}"
mkdir -p "$(dirname "${AWQ_VENV_DIR}")"
if [[ ! -x "${AWQ_VENV_DIR}/bin/python3" ]]; then
    echo "[awq-prequant] AWQ venv missing; building at ${AWQ_VENV_DIR}"
    # Use the system Python 3.11 from the loaded module to create the venv.
    python3 -m venv "${AWQ_VENV_DIR}"
    # shellcheck disable=SC1091
    source "${AWQ_VENV_DIR}/bin/activate"
    python3 -m pip install --upgrade pip wheel uv
    # Match the analyzer venv's torch 2.13.0+cu129 (so CUDA works the same),
    # transformers 5.15.1, datasets 4.8.4+ (llm-compressor main requires this).
    UV_CACHE_DIR="${REPO_ROOT:-/data/gpfs/projects/uom00014/vllm}/.uv_cache" \
        uv pip install \
            --torch-backend cu129 \
            "torch==2.13.0" \
            "torchvision==0.28.0+cu129" \
            "transformers==5.15.1" \
            "datasets>=4.8.4,<=5.0.1" \
            "numpy>=2.0.0,<=2.5.3" \
            "huggingface-hub" \
            "accelerate" \
            "safetensors" \
            "compressed-tensors" \
            "pillow>=10.4.0"
    # Install llm-compressor from GitHub main (v0.14.x supports transformers
    # 5.15+, unlike v0.8.1 from PyPI which caps at transformers 4.56).
    UV_CACHE_DIR="${REPO_ROOT:-/data/gpfs/projects/uom00014/vllm}/.uv_cache" \
        uv pip install \
            "git+https://github.com/vllm-project/llm-compressor.git@main"
    deactivate
fi
# shellcheck disable=SC1091
source "${AWQ_VENV_DIR}/bin/activate"
echo "[$(date -Iseconds)] AWQ venv active: $(python3 -c 'import sys; print(sys.prefix)')"
python3 -c "import llmcompressor, transformers; print('llmcompressor:', llmcompressor.__version__); print('transformers:', transformers.__version__)"

echo "[$(date -Iseconds)] MODEL_ID      = ${MODEL_ID}"
echo "[$(date -Iseconds)] RECIPE        = ${RECIPE}"
echo "[$(date -Iseconds)] MODEL_SAFE    = ${MODEL_SAFE}"
echo "[$(date -Iseconds)] OUTPUT_DIR    = ${OUTPUT_DIR}"
echo "[$(date -Iseconds)] SCRATCH_BASE  = ${SCRATCH_BASE}"
echo "[$(date -Iseconds)] SCRATCH_USED  = ${_scratch_used_pct}%"

# ---------------------------------------------------------------------------
# Run the AWQ + GPTQ W4A16 quantization.
# ---------------------------------------------------------------------------
cd "${REPO_ROOT:-/data/gpfs/projects/uom00014/vllm}"

python3 examples/eval-moe/awq_quantize.py \
    "${MODEL_ID}" \
    "${OUTPUT_DIR}" \
    --recipe "${RECIPE}" \
    --dataset "${PERFECTBLEND_DIR}" \
    --max-seq-length 2048 \
    --num-samples 512

echo "[$(date -Iseconds)] pre-quantization (recipe=${RECIPE}) done."