#!/bin/bash
#SBATCH -J d019
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:20:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D019/d019-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D019/d019-%j.err
# D019 S0-S3. P1 approved 2026-09-27, no amendments. 10 attention fits, ~3-4 min.
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTHONPATH=.:experiments
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
python experiments/d019_hard_model_recall.py
