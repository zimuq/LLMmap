#!/bin/bash
#SBATCH -J llmmap-ablation
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/slurm-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/slurm-%j.err

cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
./run_ablation_suite.sh
