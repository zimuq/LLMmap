#!/bin/bash
#SBATCH -J d027-pre
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D027/logs/pre-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D027/logs/pre-%j.err
# usage: sbatch [-p gh-dev -t 01:00:00] submit_d027_preflight.slurm <list> [v2|old|internlm]
ENVTAG=${2:-v2}
case $ENVTAG in
  v2)       PY=/work/11280/zimuq1/vista/envs/llmmap-gpu-v2/bin/python ;;
  old)      PY=/work/11280/zimuq1/vista/miniconda3/envs/llmmap-gpu/bin/python ;;
  internlm) PY=/work/11280/zimuq1/vista/envs/llmmap-internlm/bin/python ;;
esac
export OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments HF_HOME=$SCRATCH/hf-cache HF_MODEL_CACHE=$SCRATCH/hf-cache HF_HUB_OFFLINE=1
export HUGGINGFACE_API_KEY=$(cat /work/11280/zimuq1/vista/.hf_token)
echo "=== node: $(hostname) env: $ENVTAG start: $(date) ==="
$PY experiments/d027_preflight.py --models "$1" --env-tag $ENVTAG ${3:+--batch1}
echo "=== end: $(date) ==="
