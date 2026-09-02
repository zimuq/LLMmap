#!/bin/bash
#SBATCH -J d003-genpool
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D003/slurm-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D003/slurm-%j.err
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export HF_HOME=$SCRATCH/hf-cache
export HUGGINGFACE_API_KEY=$(tr -d '\r\n' < /work/11280/zimuq1/vista/.hf_token)
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments
echo "=== node: $(hostname) start: $(date) ==="
python experiments/d003_generate.py --target 250
echo "=== end: $(date) ==="
