#!/bin/bash
#SBATCH -J d031
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D031/logs/%x-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D031/logs/%x-%j.err
# D031 S1/S2/S3/S4: arguments = the python script and its arguments, e.g.
#   sbatch -J d031val  submit_d031_job.slurm d031_train.py val
#   sbatch -J d031lin2 submit_d031_job.slurm d031_train.py lin v2
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONPATH=.:experiments
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
echo "=== node: $(hostname) start: $(date) args: $* ==="; free -g
python experiments/"$@"
echo "=== end: $(date) ==="
