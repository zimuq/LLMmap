#!/bin/bash
#SBATCH -J d021s0
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D021/s0-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D021/s0-%j.err
# D021 S0 (pre-P1, read-only): inventory, incumbents, delta check, Sigma pilot. CPU only.
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=16 PYTHONPATH=.:experiments
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
python experiments/d021_s0.py
