#!/bin/bash
#SBATCH -J d022A
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D022/partA-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D022/partA-%j.err
# D022 S1 Part A: G1, G2, judge over all 259 queries (query text only). P1 approved w/ amendments.
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTHONPATH=.:experiments
export HF_HOME=$SCRATCH/hf-cache HF_HUB_OFFLINE=1
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
python experiments/d022_partA.py
