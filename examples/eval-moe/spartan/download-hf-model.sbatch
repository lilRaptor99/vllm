#!/usr/bin/env bash
#
# spartan/download-hf-model.sbatch
#
# One-off sbatch to pre-stage a HuggingFace model to scratch.
# Intended for login-node policy compliance: Spartan's HPC banner
# warns "Do not run programs or code on the login node" — HF
# downloads of large model files (>1 GB) should run via sbatch.
#
# Usage:
#   sbatch --export=HF_MODEL_ID=mistral-community/Mixtral-8x22B-v0.1-AWQ,HF_HUB_CACHE_DIR=/data/scratch/projects/uom00014/vllm/hf_cache,JOB_NAME=dl-awq spartan/download-hf-model.sbatch
#
# Required env vars (export them to sbatch):
#   HF_MODEL_ID       - the HF model id, e.g. "mistral-community/Mixtral-8x22B-v0.1-AWQ"
#   HF_HUB_CACHE_DIR  - where to store the downloaded files
#   JOB_NAME          - short name for the sbatch job (cosmetic; not
#                       used in the actual --job-name because SLURM
#                       #SBATCH directives don't expand variables)
#
# Note: we use gpu-a100-short (PCIe, 4h max wall) because Python/3.11.3
# is available there. The single GPU is just a "tax" for the right
# module set; the download doesn't touch the GPU. Use a backfill slot
# -- the workload is purely network-bound.
# Optional env vars (default to safe values):
#   INCLUDE_PATTERNS  - glob patterns to include; default matches
#                       safetensors + tokenizer + configs
# ----------------------------------------------------------------

#SBATCH --job-name=dl-hf
#SBATCH --partition=gpu-a100-short
#SBATCH --account=uom00014
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=spartan/logs/download-hf-model-%j.out
#SBATCH --error=spartan/logs/download-hf-model-%j.err
#SBATCH --export=ALL
#SBATCH --signal=B:SIGTERM@60

set -euo pipefail

if [[ -z "${HF_MODEL_ID:-}" ]]; then
    echo "[fatal] HF_MODEL_ID is empty — set it to e.g. mistral-community/Mixtral-8x22B-v0.1-AWQ" >&2
    exit 3
fi
if [[ -z "${HF_HUB_CACHE_DIR:-}" ]]; then
    echo "[fatal] HF_HUB_CACHE_DIR is empty — set it to the destination cache dir" >&2
    exit 3
fi

INCLUDE_PATTERNS="${INCLUDE_PATTERNS:-*.json *.safetensors tokenizer* *.jinja *.model *.txt *.py README*}"

mkdir -p "${HF_HUB_CACHE_DIR}"

set +u
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
set -u

# Build the --include arg. Use a single --include with comma-joined
# patterns to avoid the "Ignoring --include since filenames have
# been explicitly set" warning (which fires when any positional
# FILENAMES arg is present).
INCLUDE_GLOB=$(echo "${INCLUDE_PATTERNS}" | tr ' ' ',')

echo "============================================================"
echo "[$(date -Iseconds)] download-hf-model starting"
echo "[$(date -Iseconds)] job_id         = ${SLURM_JOB_ID:-}"
echo "[$(date -Iseconds)] node           = ${SLURM_JOB_NODELIST:-}"
echo "[$(date -Iseconds)] HF_MODEL_ID    = ${HF_MODEL_ID}"
echo "[$(date -Iseconds)] HF_HUB_CACHE_DIR = ${HF_HUB_CACHE_DIR}"
echo "[$(date -Iseconds)] include        = ${INCLUDE_GLOB}"
echo "============================================================"

# HF_HUB_CACHE controls where the standard HF layout
# (models--*/{blobs,snapshots,refs,trees}) is written.
export HF_HUB_CACHE="${HF_HUB_CACHE_DIR}"

# Use hf (v2 CLI) — newer than huggingface-cli. We use snapshot_download
# via the python API because the hf CLI's --include flag has a
# non-obvious "Ignoring --include since filenames have been explicitly
# set" warning when any non-flag token follows the repo id, which can
# silently skip file types we need. The snapshot_download call is
# equivalent in network behavior but with predictable allow_patterns
# semantics.
python3 -c "
import os
os.environ.pop('HF_HUB_OFFLINE', None)
os.environ.pop('TRANSFORMERS_OFFLINE', None)
from huggingface_hub import snapshot_download
allow_patterns = '${INCLUDE_PATTERNS}'.split()
print('allow_patterns:', allow_patterns)
path = snapshot_download(
    '${HF_MODEL_ID}',
    allow_patterns=allow_patterns,
)
print('Snapshot at:', path)
"

echo "[$(date -Iseconds)] download-hf-model done"
