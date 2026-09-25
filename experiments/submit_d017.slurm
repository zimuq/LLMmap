#!/bin/bash
#SBATCH -J d017
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D017/d017-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D017/d017-%j.err

# D017 S2-S5. S0/S1 already done: concat primary (C=1.0), meanpool secondary (C=3.0).
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=16 PYTHONPATH=.:experiments
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
set -x
python experiments/d017_linear.py || exit 1
