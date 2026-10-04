#!/bin/bash
#SBATCH -J d028-b1
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 47:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D028/logs/b1-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D028/logs/b1-%j.err
# usage: sbatch submit_d028_conc.slurm <model> <nparts> <first_part> <k>
# Review Call 1: batch 1, k concurrent processes on one GPU, disjoint config parts.
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments HF_HOME=$SCRATCH/hf-cache HF_MODEL_CACHE=$SCRATCH/hf-cache HF_HUB_OFFLINE=1
export HUGGINGFACE_API_KEY=$(cat /work/11280/zimuq1/vista/.hf_token)
case ${ENVTAG:-v2} in
  v2)       PY=/work/11280/zimuq1/vista/envs/llmmap-gpu-v2/bin/python ;;
  old)      PY=/work/11280/zimuq1/vista/miniconda3/envs/llmmap-gpu/bin/python ;;
  internlm) PY=/work/11280/zimuq1/vista/envs/llmmap-internlm/bin/python ;;
esac
echo "=== node: $(hostname) env: ${ENVTAG:-v2} start: $(date) ==="
for b in pool ml; do
  for ((i=$3; i<$3+$4; i++)); do
    OMP_NUM_THREADS=4 $PY experiments/d028_generate.py --model "$1" --block $b --mode batch1 --part $i --nparts $2 &
  done
  wait
done
echo "=== end: $(date) ==="
