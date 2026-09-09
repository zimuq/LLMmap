#!/bin/bash
#SBATCH -J d009
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D009/d009-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D009/d009-%j.err

# D009 S0-S8. Closed-set grid (Calls 1-2 as approved) + S8 open-set at k=8.
# D009_SMOKE=1 exercises every path at a fraction of the size first.
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTHONPATH=.:experiments
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
set -x
python experiments/d009_train.py   || exit 1
python experiments/d009_eval.py    || exit 1
python experiments/d009_openset.py || exit 1
