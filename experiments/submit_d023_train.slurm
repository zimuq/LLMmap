#!/bin/bash
#SBATCH -J d023train
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D023/train-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D023/train-%j.err
# D023 S2+S3: train frozen ILP-ADD / Greedy-ADD chains, then analysis. P1 approved w/ amendments.
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTHONPATH=.:experiments
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
set -e
python experiments/d023_train.py
python experiments/d023_analysis.py
