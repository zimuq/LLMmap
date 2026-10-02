#!/bin/bash
#SBATCH -J d027-stage
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D027/logs/stage-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D027/logs/stage-%j.err
# Staging runs on a compute node: the login node's 100-thread user cap cannot
# host hf_xet / parallel downloads (D027 S0; compute-node egress measured OK,
# job 1041847: HF API 200, 32.7 MB/s single stream).
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
export HF_HOME=$SCRATCH/hf-cache
echo "=== node: $(hostname) start: $(date) ==="
/work/11280/zimuq1/vista/envs/llmmap-gpu-v2/bin/python experiments/d027_stage.py
echo "=== end: $(date) ==="
