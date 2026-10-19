#!/usr/bin/env bash
#
# spartan/install-pip-package.sbatch
#
# One-off sbatch to install a pip package into an existing venv.
# Intended for login-node policy compliance: Spartan's HPC banner
# warns "Do not run programs or code on the login node". Quick
# installs on small venvs (e.g. `uv pip install protobuf`) should
# run via sbatch to avoid the warning.
#
# Usage:
#   sbatch --export=PIP_TARGET=/path/to/venv,PIP_PACKAGES="protobuf" examples/eval-moe/spartan/install-pip-package.sbatch
#
# Required env vars (export them to sbatch):
#   PIP_TARGET   - absolute path to the venv to install into
#   PIP_PACKAGES - space-separated list of pip packages
# ----------------------------------------------------------------

#SBATCH --job-name=install-pkg
#SBATCH --partition=gpu-a100-short
#SBATCH --account=uom00014
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=spartan/logs/install-pip-package-%j.out
#SBATCH --error=spartan/logs/install-pip-package-%j.err
#SBATCH --export=ALL
#SBATCH --signal=B:SIGTERM@60

set -euo pipefail

if [[ -z "${PIP_TARGET:-}" ]]; then
    echo "[fatal] PIP_TARGET is empty — set it to e.g. /data/scratch/projects/uom00014/vllm/venv_sglang_0509" >&2
    exit 3
fi
if [[ -z "${PIP_PACKAGES:-}" ]]; then
    echo "[fatal] PIP_PACKAGES is empty — set it to e.g. 'protobuf==4.25.3'" >&2
    exit 3
fi

set +u
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
set -u

echo "============================================================"
echo "[$(date -Iseconds)] install-pip-package starting"
echo "[$(date -Iseconds)] job_id       = ${SLURM_JOB_ID:-}"
echo "[$(date -Iseconds)] node         = ${SLURM_JOB_NODELIST:-}"
echo "[$(date -Iseconds)] PIP_TARGET   = ${PIP_TARGET}"
echo "[$(date -Iseconds)] PIP_PACKAGES = ${PIP_PACKAGES}"
echo "============================================================"

# Use uv (bundled in the venv) to install. uv respects the
# existing venv and won't clobber it.
"${PIP_TARGET}/bin/uv" pip install --python "${PIP_TARGET}/bin/python" ${PIP_PACKAGES}

echo "[$(date -Iseconds)] install-pip-package done"
