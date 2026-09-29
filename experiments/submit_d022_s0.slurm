#!/bin/bash
#SBATCH -J d022s0
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D022/s0-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D022/s0-%j.err
# D022 S0 (pre-P1): detector + judge cost pilot on 10 queries. Query text only.
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTHONPATH=.:experiments
export HF_HOME=$SCRATCH/hf-cache HF_HUB_OFFLINE=1
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
python experiments/d022_s0.py
