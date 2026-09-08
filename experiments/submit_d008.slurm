#!/bin/bash
#SBATCH -J d008
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D008/d008-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D008/d008-%j.err

# D008 S0-S7. Pure post-processing on the frozen corpus + D007 tensor (I6).
# Set D008_SMOKE=1 to exercise every code path at a fraction of the size first.
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=.:experiments
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
set -x
python experiments/d008_select.py      || exit 1
python experiments/d008_metrics.py     || exit 1
python experiments/d008_gamma.py       || exit 1
python experiments/d008_sensitivity.py || exit 1
python experiments/d008_frontier.py    || exit 1
