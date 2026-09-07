#!/bin/bash
#SBATCH -J d007-f1-reembed
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D007/slurm-f1-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D007/slurm-f1-%j.err
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments
export HF_HOME=$SCRATCH/hf-cache HF_MODEL_CACHE=$SCRATCH/hf-cache
export HUGGINGFACE_API_KEY=$(cat /work/11280/zimuq1/vista/.hf_token)
echo "=== pass 1: 36 models, llmmap-gpu ==="
python experiments/d007_f1_reembed_trunc.py --budget 100
echo "=== pass 2: internlm, dedicated env (sentencepiece 0.1.99) ==="
/work/11280/zimuq1/vista/envs/llmmap-internlm/bin/python \
    experiments/d007_f1_reembed_trunc.py --budget 100 --only internlm/internlm2_5-7b-chat
echo "=== end $(date) ==="
