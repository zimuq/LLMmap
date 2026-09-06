#!/bin/bash
#SBATCH -J d6-internlm
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D006/shards/internlm-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D006/shards/internlm-%j.err
# Dedicated env: sentencepiece 0.1.99 (llmmap-gpu has 0.2.2, which rejects
# internlm2's vocabulary). Everything else inherited via --system-site-packages,
# so transformers/torch match the other 36 shards exactly.
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments
export HF_HOME=$SCRATCH/hf-cache HF_MODEL_CACHE=$SCRATCH/hf-cache
export HUGGINGFACE_API_KEY=$(cat /work/11280/zimuq1/vista/.hf_token)
echo "=== $(hostname) $(date) internlm (dedicated env) ==="
/work/11280/zimuq1/vista/envs/llmmap-internlm/bin/python \
    experiments/d006_s4_shard.py --model internlm/internlm2_5-7b-chat
echo "=== end $(date) ==="
