#!/bin/bash
#SBATCH -J d030-s0-checks
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/s0checks-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/s0checks-%j.err
# D030 / S0.5(a)(b) on pilot gen_ref / all_ref rows (never test_timing, Review A3), then s0.json.
set -eo pipefail
W=/work/11280/zimuq1/vista
cd $W/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16
export HF_HOME=$SCRATCH/hf-cache
echo "=== node: $(hostname) start: $(date) ==="
$W/envs/d030-met/bin/python experiments/d030_checks.py met
$W/envs/d030-zp/bin/python experiments/d030_checks.py zp --model meta-llama/Meta-Llama-3.1-8B-Instruct
$W/miniconda3/envs/llmmap/bin/python experiments/d030_s0_report.py
echo "=== end: $(date) ==="
