#!/bin/bash
#SBATCH -J d025train
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D025/train-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D025/train-%j.err
# D025 S2: train one frozen arm (arg 1 = arm name, e.g. GLOBAL, "SPEC:Falcon3-10B-Instruct|Falcon3-7B-Instruct").
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTHONPATH=.:experiments
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
echo "=== arm: $1 node: $(hostname) start: $(date) ==="
python experiments/d025_train.py "$1"
echo "=== end: $(date) ==="
