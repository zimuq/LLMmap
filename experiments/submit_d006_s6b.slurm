#!/bin/bash
#SBATCH -J d006-s6b
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D006/slurm-s6b-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D006/slurm-s6b-%j.err
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments
export HF_HOME=$SCRATCH/hf-cache HF_MODEL_CACHE=$SCRATCH/hf-cache
export HUGGINGFACE_API_KEY=$(cat /work/11280/zimuq1/vista/.hf_token)

echo "=== $(hostname) $(date) ==="
# S5 first: S6 warns loudly if the manifest is not READY, and the manifest is
# what records which shards are even eligible. Re-running S5 is cheap and
# idempotent; running S6 against a stale manifest is not.
true

python experiments/d006_s6b_norm_check.py
echo "=== end $(date) ==="
