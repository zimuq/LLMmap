#!/bin/bash
#SBATCH -J d001-extract
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D001/slurm-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D001/slurm-%j.err

source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap

export HF_MODEL_CACHE=/work/11280/zimuq1/vista/.cache/huggingface
export HF_HOME=/work/11280/zimuq1/vista/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
export PYTHONPATH=.

echo "=== node: $(hostname)  start: $(date) ==="
python experiments/d001_extract.py --device cpu
echo "=== end: $(date) ==="
