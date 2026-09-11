#!/bin/bash
#SBATCH -J d010
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D010/d010-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D010/d010-%j.err

# D010 S2-S6. Call 1: energy (scale-free). Call 2: exact search. Plus the
# review's k=8 MMD robustness check.
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTHONPATH=.:experiments
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
set -x
python experiments/d010_select.py   || exit 1
python experiments/d010_evaluate.py || exit 1
