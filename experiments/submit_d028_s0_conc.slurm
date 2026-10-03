#!/bin/bash
#SBATCH -J d028-conc
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D028/logs/s0-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D028/logs/s0-%j.err
# usage: sbatch submit_d028_s0.slurm <task> [models] [mode]
export OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments HF_HOME=$SCRATCH/hf-cache HF_MODEL_CACHE=$SCRATCH/hf-cache HF_HUB_OFFLINE=1
export HUGGINGFACE_API_KEY=$(cat /work/11280/zimuq1/vista/.hf_token)
PY=/work/11280/zimuq1/vista/envs/llmmap-gpu-v2/bin/python
echo "=== node: $(hostname) start: $(date) ==="
$PY experiments/d028_s0_concurrency.py
echo "=== end: $(date) ==="
