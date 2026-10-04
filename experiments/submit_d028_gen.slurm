#!/bin/bash
#SBATCH -J d028-gen
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D028/logs/gen-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D028/logs/gen-%j.err
# usage: ENVTAG=v2|old|internlm sbatch submit_d028_gen.slurm <model-list> "<blocks, e.g. pool ml>"
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
for m in $(cat "$1"); do for b in $2; do
  $PY experiments/d028_generate.py --model "$m" --block $b --mode d006 --env-tag ${ENVTAG:-v2} || echo "FAILED $m $b"
done; done
echo "=== end: $(date) ==="
