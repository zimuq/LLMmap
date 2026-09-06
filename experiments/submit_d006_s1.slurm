#!/bin/bash
#SBATCH -J d006-s1
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D006/slurm-s1-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D006/slurm-s1-%j.err

source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments
export HF_HOME=$SCRATCH/hf-cache
export HF_MODEL_CACHE=$SCRATCH/hf-cache
export HUGGINGFACE_API_KEY=$(cat /work/11280/zimuq1/vista/.hf_token)

echo "=== node: $(hostname) start: $(date) ==="
python experiments/d006_s1.py
echo "=== end: $(date) ==="
