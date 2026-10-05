#!/bin/bash
#SBATCH -J d028-ten
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D028/logs/post-tensor-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D028/logs/post-tensor-%j.err
# post part 2: frozen v2 tensor (D007 statistics, 85-model loader) + reproduction check + tail plot
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments HF_HOME=$SCRATCH/hf-cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
G=/work/11280/zimuq1/vista/miniconda3/envs/llmmap-gpu/bin/python
P=/work/11280/zimuq1/vista/miniconda3/envs/llmmap/bin/python
echo "=== node: $(hostname) start: $(date) ==="
OMP_NUM_THREADS=1 $G experiments/d028_build_tensor.py --jobs 70 || exit 1
$P experiments/d028_plots.py
echo "=== end: $(date) ==="
