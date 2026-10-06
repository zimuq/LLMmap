#!/bin/bash
#SBATCH -J d029an
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D029/logs/analysis-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D029/logs/analysis-%j.err
# D029 S2: the single S_test read (guardrail first), then figures (llmmap env has matplotlib).
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=16 PYTHONPATH=.:experiments
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
echo "=== node: $(hostname) start: $(date) ==="
python experiments/d029_analysis.py && /work/11280/zimuq1/vista/miniconda3/envs/llmmap/bin/python experiments/d029_plots.py curves
echo "=== end: $(date) ==="
