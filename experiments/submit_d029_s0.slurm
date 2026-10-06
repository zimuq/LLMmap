#!/bin/bash
#SBATCH -J d029s0
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D029/logs/s0-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D029/logs/s0-%j.err
# D029 S0: frozen inputs, ML query embeddings, loader + v1 code-path checks, chains, pointers. No training.
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=32 TOKENIZERS_PARALLELISM=false PYTHONPATH=.:experiments
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
echo "=== node: $(hostname) start: $(date) ==="; free -g
mkdir results/D029/s0.lock 2>/dev/null || { echo "another S0 copy holds results/D029/s0.lock -- exiting"; exit 0; }
python experiments/d029_s0.py
echo "=== end: $(date) ==="
