#!/bin/bash
#SBATCH -J d013
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D013/d013-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D013/d013-%j.err

# D013 S4-S6. Selection (S1-S3) already verified against D008's frozen chains.
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTHONPATH=.:experiments
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
set -x
python experiments/d013_evaluate.py || exit 1
