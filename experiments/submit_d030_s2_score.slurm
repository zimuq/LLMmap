#!/bin/bash
#SBATCH -J d030-s2-score
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/s2score-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/s2score-%j.err
# D030 / P2 S2.2 (score matrices, code-path asserts, counts, Call 1) then S3 (the single read) + tail plot.
# Requires results/D030/s2_freeze.json committed first (S2.0). Any assert failure stops the job (Review B3).
set -eo pipefail
W=/work/11280/zimuq1/vista
cd $W/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16
export HF_HOME=$SCRATCH/hf-cache HF_MODEL_CACHE=$SCRATCH/hf-cache
GPU=$W/miniconda3/envs/llmmap-gpu/bin/python
echo "=== node: $(hostname) job $SLURM_JOB_ID start: $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git log -1 --format='HEAD %h %s' -- results/D030/s2_freeze.json
$GPU experiments/d030_s2_score.py met --variant hamming
$GPU experiments/d030_s2_score.py met --variant e5
$W/envs/d030-met/bin/python experiments/d030_s2_score.py met_assert
$W/envs/d030-zp/bin/python experiments/d030_s2_score.py zp --variant 512
$W/envs/d030-zp/bin/python experiments/d030_s2_score.py zp --variant 200
$GPU experiments/d030_s2_score.py os
$GPU experiments/d030_s2_score.py counts
$GPU experiments/d030_s2_score.py call1
echo "=== S2 done: $(date) ==="
$GPU experiments/d030_s3_analysis.py
$W/miniconda3/envs/llmmap/bin/python experiments/d030_plots.py
echo "=== end: $(date) ==="
