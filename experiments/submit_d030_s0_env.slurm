#!/bin/bash
#SBATCH -J d030-s0-env
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/s0env-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/s0env-%j.err
# D030 / S0.1-S0.2 (P1, approved 2026-10-09): side venvs + staging.
#  d030-zp : ZeroPrint's environment.yml pins for the packages its prepare()/fingerprint
#            code imports (gptqmodel/optimum/peft/faiss are not imported there and are
#            skipped); torch 2.7.1+cu128 instead of +cu126 (no aarch64 cu126 wheel, ENV.md).
#  d030-met: venv over `llmmap` (CPU) + the pinned MET package (--no-deps; numpy/torch
#            already there) + datasets/pyarrow for MET's own prompt builder.
set -eo pipefail
W=/work/11280/zimuq1/vista
EXT=$W/ext
export HF_HOME=$SCRATCH/hf-cache
export HUGGINGFACE_API_KEY=$(cat $W/.hf_token)
export HF_TOKEN=$HUGGINGFACE_API_KEY
export NLTK_DATA=$W/ext/nltk_data
export GENSIM_DATA_DIR=$W/ext/gensim-data
echo "=== node: $(hostname) start: $(date) ==="

PYBASE=$W/miniconda3/envs/llmmap/bin/python      # python 3.11, as ZeroPrint's env
if [ ! -x $W/envs/d030-zp/bin/python ]; then
  $PYBASE -m venv $W/envs/d030-zp
fi
ZP=$W/envs/d030-zp/bin/python
$ZP -m pip install -q --upgrade "pip==25.3"
$ZP -m pip install -q "torch==2.7.1+cu128" --index-url https://download.pytorch.org/whl/cu128
$ZP -m pip install -q "transformers==4.54.1" "tokenizers==0.21.4" "sentence-transformers==5.2.0" \
    "datasets==4.5.0" "huggingface-hub==0.36.0" "gensim==4.4.0" "nltk==3.9.2" "numpy==1.26.4" \
    "pandas==2.3.3" "pyarrow==22.0.0" "scikit-learn==1.8.0" "scipy==1.17.0" "sentencepiece==0.2.1" \
    "accelerate==1.12.0" "pyyaml==6.0.3" "safetensors==0.7.0" "fsspec==2025.10.0" "protobuf==6.33.4"
$ZP -m pip freeze > /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/env_d030-zp.txt

if [ ! -x $W/envs/d030-met/bin/python ]; then
  $PYBASE -m venv --system-site-packages $W/envs/d030-met
fi
MET=$W/envs/d030-met/bin/python
$MET -m pip install -q "datasets==4.5.0" "pyarrow==22.0.0"
$MET -m pip install -q --no-deps -e $EXT/model-equality-testing/model_equality_testing
$MET -m pip freeze > /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/env_d030-met.txt

# ---- staging (S0.1) ----
mkdir -p $NLTK_DATA $GENSIM_DATA_DIR
$ZP - <<'EOF'
import os, nltk
for r in ["stopwords", "punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    print(r, nltk.download(r, download_dir=os.environ["NLTK_DATA"], quiet=True))
import gensim.downloader as api
m = api.load("glove-wiki-gigaword-100"); print("glove", len(m.key_to_index))
from sentence_transformers import SentenceTransformer
s = SentenceTransformer("sentence-transformers/all-mpnet-base-v2"); print("mpnet", s.get_sentence_embedding_dimension())
from datasets import load_dataset
ds = load_dataset("openai_humaneval", split="test"); print("humaneval", len(ds))
EOF
echo "=== end: $(date) ==="
