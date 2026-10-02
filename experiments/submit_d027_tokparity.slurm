#!/bin/bash
#SBATCH -J d027-tokp
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D027/logs/tokp-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D027/logs/tokp-%j.err
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments HF_HOME=$SCRATCH/hf-cache HF_MODEL_CACHE=$SCRATCH/hf-cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HUGGINGFACE_API_KEY=$(cat /work/11280/zimuq1/vista/.hf_token)
echo "=== node: $(hostname) start: $(date) ==="
/work/11280/zimuq1/vista/miniconda3/envs/llmmap-gpu/bin/python experiments/d027_tokparity.py --env-tag old &
/work/11280/zimuq1/vista/envs/llmmap-internlm/bin/python experiments/d027_tokparity.py --env-tag internlm --models internlm/internlm2_5-7b-chat &
/work/11280/zimuq1/vista/envs/llmmap-gpu-v2/bin/python experiments/d027_tokparity.py --env-tag v2 &
wait
/work/11280/zimuq1/vista/miniconda3/envs/llmmap-gpu/bin/python experiments/d027_tokparity.py --compare
echo "=== end: $(date) ==="
