#!/bin/bash
#SBATCH -J d024ana
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D024/analysis-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D024/analysis-%j.err
# D024 S3: analysis (CPU).
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTHONPATH=.:experiments
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
echo "=== node: $(hostname) start: $(date) ==="
python experiments/d024_analysis.py
echo "=== end: $(date) ==="
