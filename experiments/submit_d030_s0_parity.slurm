#!/bin/bash
#SBATCH -J d030-s0-parity
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/s0parity-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/s0parity-%j.err
# D030 / S0.5(c) + Call 5 counts: tokenizer-only parity for all 85 models, per generation env.
set -eo pipefail
W=/work/11280/zimuq1/vista
cd $W/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments
export HF_HOME=$SCRATCH/hf-cache
export HF_MODEL_CACHE=$SCRATCH/hf-cache
export HUGGINGFACE_API_KEY=$(cat $W/.hf_token)
export TOKENIZERS_PARALLELISM=false
echo "=== node: $(hostname) start: $(date) ==="
$W/envs/llmmap-gpu-v2/bin/python experiments/d030_checks.py decoding
$W/envs/llmmap-gpu-v2/bin/python experiments/d030_checks.py parity --gen-env llmmap-gpu-v2 --tag v2
$W/miniconda3/envs/llmmap-gpu/bin/python experiments/d030_checks.py parity --gen-env llmmap-gpu --tag old
$W/envs/llmmap-internlm/bin/python experiments/d030_checks.py parity --gen-env llmmap-internlm --tag internlm
echo "=== end: $(date) ==="
