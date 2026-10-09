#!/bin/bash
#SBATCH -J d030-s0-freeze
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/s0freeze-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/s0freeze-%j.err
# D030 / S0.3-S0.4: freeze MET prompts, ZeroPrint prompts (official prepare(), run once), then
# config draws / T_ZP order / LLMMAP-OS overlap / manifest. No generation.
set -eo pipefail
W=/work/11280/zimuq1/vista
cd $W/LLMmap-project/LLMmap
export HF_HOME=$SCRATCH/hf-cache
export HUGGINGFACE_API_KEY=$(cat $W/.hf_token)
export HF_TOKEN=$HUGGINGFACE_API_KEY
export NLTK_DATA=$W/ext/nltk_data
export GENSIM_DATA_DIR=$W/ext/gensim-data
export TOKENIZERS_PARALLELISM=false
echo "=== node: $(hostname) start: $(date) ==="
$W/envs/d030-met/bin/python experiments/d030_met_prompts.py
$W/envs/d030-zp/bin/python experiments/d030_zp_prepare.py
$W/miniconda3/envs/llmmap/bin/python experiments/d030_s0.py
echo "=== end: $(date) ==="
