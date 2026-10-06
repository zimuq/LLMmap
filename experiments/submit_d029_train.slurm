#!/bin/bash
#SBATCH -J d029tr
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D029/logs/train-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D029/logs/train-%j.err
# D029 S1: train one frozen arm (arg 1 = arm name in chains.json).
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=32 TOKENIZERS_PARALLELISM=false PYTHONPATH=.:experiments
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
echo "=== node: $(hostname) start: $(date) ==="; free -g
echo "arm: $1"; python experiments/d029_train.py "$1"
echo "=== end: $(date) ==="
