#!/bin/bash
#SBATCH -J d030-prestage
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/prestage-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/prestage-%j.err
# D030 pre-S1: existence + sha of all 85 pinned snapshots + mpnet + e5 (Review A2); re-stage if needed.
W=/work/11280/zimuq1/vista
cd $W/LLMmap-project/LLMmap
export HF_HOME=$SCRATCH/hf-cache HUGGINGFACE_API_KEY=$(cat $W/.hf_token) NPROC=48 HF_HUB_DISABLE_PROGRESS_BARS=1
echo "=== node: $(hostname) start: $(date) ==="
$W/envs/llmmap-gpu-v2/bin/python experiments/d030_s1_prestage.py
echo "exit $? === end: $(date) ==="
