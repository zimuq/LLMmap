#!/bin/bash
#SBATCH -J d030-s1-post
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/s1post-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/s1post-%j.err
# D030 / S1 health report (after all generation jobs, afterany). No score of any kind.
W=/work/11280/zimuq1/vista
cd $W/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments
$W/miniconda3/envs/llmmap/bin/python experiments/d030_generate.py --health --model ALL --tzp 25
