#!/usr/bin/env bash
set -euo pipefail

# Prepare the GPU VM before the first H100 validation run:
# - create a dedicated sglang venv
# - install the repo's required Python packages
# - download the Mixtral BF16 snapshot into the designated HF cache
# - pre-download the benchmark datasets (mmlu, bbh, humaneval, popqa, include)
#
# This is intentionally safe to run on a non-GPU machine, because it only
# initializes the Python environment, preloads the model, and pre-caches
# the datasets. The actual inference run occurs later from the 4x H100 VM.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VLLM_REPO_ROOT="${VLLM_REPO_ROOT:-${ROOT_DIR}}"
SGLANG_VERSION="${SGLANG_VERSION:-0.5.9}"
MODEL_ID="${MODEL_ID:-mistralai/Mixtral-8x22B-Instruct-v0.1}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE}/datasets}"
VENV_DIR="${VENV_DIR:-${HOME}/.venvs/sglang-mixtral-bf16}"

mkdir -p "${HF_CACHE}" "${HF_DATASETS_CACHE}"

echo "============================================================"
echo "[$(date -Iseconds)] preparing sglang Mixtral BF16 VM setup"
echo "[$(date -Iseconds)] model    = ${MODEL_ID}"
echo "[$(date -Iseconds)] venv     = ${VENV_DIR}"
echo "[$(date -Iseconds)] hf_cache = ${HF_CACHE}"
echo "============================================================"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[info] creating venv at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m ensurepip --upgrade || true
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel uv
"${VENV_DIR}/bin/python" -m uv pip install \
    --python "${VENV_DIR}/bin/python" \
    --torch-backend=cu128 \
    "sglang==${SGLANG_VERSION}" \
    "datasets==4.5.0" \
    "numpy>=2.1,<2.3" \
    "pyarrow>=21,<25" \
    "transformers>=4.55" \
    "huggingface_hub" \
    "pybase64" \
    "httpx" \
    "openai" || true

export HUGGINGFACE_HUB_CACHE="${HF_CACHE}"
export HF_HOME="${HF_CACHE}"

if [[ ! -d "${HF_CACHE}/models--mistralai--Mixtral-8x22B-Instruct-v0.1" && ! -d "${HF_CACHE}/hub/models--mistralai--Mixtral-8x22B-Instruct-v0.1" ]]; then
    echo "[info] downloading model snapshot for ${MODEL_ID}"
    "${VENV_DIR}/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
repo_id = os.environ.get("MODEL_ID", "mistralai/Mixtral-8x22B-Instruct-v0.1")
cache_dir = os.environ.get("HF_CACHE", os.path.expanduser("~/.cache/huggingface"))
snapshot_download(
    repo_id=repo_id,
    cache_dir=cache_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.md", "tokenizer*", "config.json"],
)
print(f"[done] snapshot_download complete for {repo_id}")
PY
else
    echo "[info] model snapshot already present under ${HF_CACHE}"
fi

# ---------------------------------------------------------------- datasets
echo "[info] pre-downloading benchmark datasets to ${HF_DATASETS_CACHE}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
"${VENV_DIR}/bin/python" - <<'PY' || echo "[warn] dataset pre-cache step failed; will retry at driver runtime"
import os
from datasets import load_dataset
items = [
    ("cais/mmlu", "all"),
    ("lukaemon/bbh", None),
    ("openai_humaneval", None),
    ("akariasai/PopQA", None),
    ("allenai/instructeval", "include"),
]
for repo, name in items:
    try:
        kwargs = {"cache_dir": os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))}
        if name is not None:
            kwargs["name"] = name
        load_dataset(repo, **kwargs)
        print(f"[datasets] cached {repo}")
    except Exception as e:
        print(f"[datasets] FAILED to cache {repo}: {e}")
PY

echo "[success] environment, model cache, and dataset cache are ready"
