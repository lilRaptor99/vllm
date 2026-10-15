#!/usr/bin/env bash
#
# fix-sglang-venv-numpy.sbatch - one-shot: pin datasets 4.5.0 + numpy 2.x
# in venv_sglang_0509 so the sglang-moe driver can import datasets.load_*
# without the pyarrow/numpy version conflict.
#

#SBATCH --job-name=fix-sglang-venv
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-00:15:00
#SBATCH --output=spartan/logs/fix-sglang-venv-%j.out
#SBATCH --error=spartan/logs/fix-sglang-venv-%j.err

set -euo pipefail
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "${SLURM_SUBMIT_DIR}/spartan/logs"

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
load_module_or_die "GCCcore/11.3.0"
load_module_or_die "Python/3.11.3"
load_module_or_die "CUDA/12.4.1"
load_module_or_die "Ninja/1.11.1-Python-3.11.3"
set -u
export LD_LIBRARY_PATH="${EBROOTPYTHON:-/usr/lib}/lib:${LD_LIBRARY_PATH:-}"

VENV="/data/scratch/projects/uom00014/vllm/venv_sglang_0509"
SISTER_VLLM_ROOT="${SISTER_VLLM_ROOT:-/data/gpfs/projects/uom00014/vllm}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${SISTER_VLLM_ROOT}/.uv_cache}"
export UV_CACHE_DIR

echo "[$(date -Iseconds)] numpy/datasets version BEFORE fix:"
"${VENV}/bin/python" -m pip list 2>/dev/null | grep -iE "^(numpy|datasets|pyarrow|pyyaml) " || true

# Strategy: the analyzer loaders use datasets.load_dataset /
# datasets.load_from_disk. We pre-downloaded all benchmark data with
# datasets 4.5.0, so the safest pin is datasets==4.5.0 (matches the
# vLLM analyzer venv and the local parquet format). datasets 4.5.0
# requires pyarrow which is incompatible with our old numpy 1.26.
# Two options tried in order:
#
#   (a) Upgrade numpy to 2.1.x. pyarrow and pandas are numpy-2 safe.
#       We must verify torch 2.9.1+cu128 still loads under numpy 2.
#
#   (b) If (a) breaks torch, fall back to downgrading pyarrow to the
#   last version compatible with both datasets 4.5.0 and numpy 1.26.
#
echo "[$(date -Iseconds)] option (a): upgrade numpy to 2.1.x"
"${VENV}/bin/python" -m pip install --upgrade 'numpy>=2.1,<2.3' 'pyarrow>=17,<21' 2>&1 | tail -20 || {
    echo "[warn] numpy 2 upgrade failed; trying option (b): downgrade pyarrow"
    "${VENV}/bin/python" -m pip install 'pyarrow==17.0.0' 2>&1 | tail -10
}

# Pin datasets 4.5.0 to match the vLLM analyzer's pin and the local
# pre-download schema. This avoids the 5.x series that pulls a newer
# pyarrow + requires numpy 2.
echo "[$(date -Iseconds)] pin datasets==4.5.0"
"${VENV}/bin/python" -m pip install 'datasets==4.5.0' 2>&1 | tail -10

echo "[$(date -Iseconds)] numpy/datasets/pyarrow AFTER fix:"
"${VENV}/bin/python" -m pip list 2>/dev/null | grep -iE "^(numpy|datasets|pyarrow|pyyaml) " || true

# Smoke test: import torch + load a saved dataset
echo "[$(date -Iseconds)] torch import smoke test:"
"${VENV}/bin/python" -c "
import torch
print('torch', torch.__version__, 'cuda_avail=', torch.cuda.is_available())
import numpy as np
print('numpy', np.__version__)
import datasets
print('datasets', datasets.__version__)
import pyarrow
print('pyarrow', pyarrow.__version__)
# Try loading the pre-staged mmlu dataset
from datasets import load_from_disk
ds = load_from_disk('/data/scratch/projects/uom00014/vllm/datasets/mmlu')
print('mmlu loaded: rows=', len(ds), 'cols=', ds.column_names)
"

# Clear the .pyc cache so any imports pick up new package versions
find "${VENV}/lib/python3.11/site-packages/datasets" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
find "${VENV}/lib/python3.11/site-packages/pyarrow" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
find "${VENV}/lib/python3.11/site-packages/numpy" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
echo "[$(date -Iseconds)] pyc cache cleared"