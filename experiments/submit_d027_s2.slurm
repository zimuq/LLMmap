#!/bin/bash
#SBATCH -J d027-s2
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D027/logs/s2-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D027/logs/s2-%j.err
# S1-embed -> S2 train -> S3 analysis, all in the unchanged llmmap-gpu env
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments HF_HOME=$SCRATCH/hf-cache HF_MODEL_CACHE=$SCRATCH/hf-cache HF_HUB_OFFLINE=1
export HUGGINGFACE_API_KEY=$(cat /work/11280/zimuq1/vista/.hf_token)
echo "=== node: $(hostname) start: $(date) ==="
python experiments/d027_embed.py && python experiments/d027_train.py && python experiments/d027_analysis.py
echo "=== end: $(date) ==="
