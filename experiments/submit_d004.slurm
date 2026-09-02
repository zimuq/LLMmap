#!/bin/bash
#SBATCH -J d004-reanalysis
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:40:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D004/slurm-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D004/slurm-%j.err
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=16
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments
echo "=== node: $(hostname) start: $(date) ==="
python experiments/d004_analyze.py
echo "=== end: $(date) ==="
