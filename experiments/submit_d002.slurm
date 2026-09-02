#!/bin/bash
#SBATCH -J d002-bench
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D002/slurm-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D002/slurm-%j.err
# NOTE: no --gres / --gpus-per-task. Vista does not support them (Slurm rejects
# job scripts containing them); GH nodes are exclusive so the H200 is present.

source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu

export HF_HOME=$SCRATCH/hf-cache
export HUGGINGFACE_API_KEY=$(tr -d '\r\n' < /work/11280/zimuq1/vista/.hf_token)
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16

cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
export PYTHONPATH=.

echo "=== node: $(hostname)  start: $(date) ==="
nvidia-smi || echo "nvidia-smi unavailable"
echo
python experiments/d002_benchmark.py --n_confs 10 --max_new_tokens 100 200
echo "=== end: $(date) ==="
