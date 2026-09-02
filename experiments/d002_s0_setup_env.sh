#!/bin/bash
# D002 / P1 — S0.1: build the llmmap-gpu env with a CUDA aarch64 torch build.
# The existing `llmmap` env is deliberately left untouched so D001 stays reproducible.
set -e
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
export PIP_CACHE_DIR=/work/11280/zimuq1/vista/.cache/pip
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap

echo "=== create env ==="
conda create -n llmmap-gpu python=3.11 -y -c conda-forge --override-channels 2>&1 | tail -4
conda activate llmmap-gpu

# Requirements FIRST, CUDA torch LAST -- order matters, and cu126 is the wrong index.
#
# PyPI's aarch64 torch wheel is CPU-only (that is what produced the 2.7.1+cpu build
# behind D001/P1 F5). The obvious fix, cu126, does NOT work here: that index has no
# aarch64 build for 2.7.1 (it goes 2.6.0 -> 2.9.0), so pip resolves to 2.14.0+cu126
# and requirements.txt's `torch==2.7.1` pin then downgrades it straight back to the
# CPU wheel -- silently, with a zero exit code.
#
# cu128 publishes exactly 2.7.1+cu128, which PEP 440 treats as satisfying `==2.7.1`.
# So we keep the pinned version and change only the CUDA variant. CUDA 12.8 is also
# available as a Vista module, and H200 is sm_90 -- well within cu128's support.
echo "=== requirements ==="
pip install --quiet -r requirements.txt accelerate scikit-learn 2>&1 | tail -6

echo "=== torch (CUDA aarch64, cu128 -- must come last) ==="
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -6

echo "=== versions ==="
python -c "import torch; print('torch', torch.__version__, '| cuda build:', torch.version.cuda)"
echo "ENV_DONE"
