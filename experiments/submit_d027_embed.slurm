#!/bin/bash
#SBATCH -J d027-emb
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D027/logs/embed-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D027/logs/embed-%j.err
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false 
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments HF_HOME=$SCRATCH/hf-cache HF_MODEL_CACHE=$SCRATCH/hf-cache HF_HUB_OFFLINE=1
export HUGGINGFACE_API_KEY=$(cat /work/11280/zimuq1/vista/.hf_token)
echo "=== node: $(hostname) start: $(date) ==="
python experiments/d027_embed.py
echo "=== end: $(date) ==="
