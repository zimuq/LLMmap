#!/bin/bash
#SBATCH -J d030-posthoc
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:20:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/posthoc-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/posthoc-%j.err
# D030 post hoc (descriptive, after labels): intervals on unselected v2 pair sets.
set -eo pipefail
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments OMP_NUM_THREADS=16
/work/11280/zimuq1/vista/miniconda3/envs/llmmap-gpu/bin/python experiments/d030_posthoc.py
