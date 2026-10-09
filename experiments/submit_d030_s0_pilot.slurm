#!/bin/bash
#SBATCH -J d030-pilot
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/pilot-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/pilot-%j.err
# D030 / S0.6 cost pilot. Usage: sbatch --export=ALL,MODEL=<hf id>,K=<processes>,REDUCED=<0|1> this
# K>1 = D028's batch-1-concurrent mode (K processes on one GPU, disjoint unit ranges).
W=/work/11280/zimuq1/vista
cd $W/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_HOME=$SCRATCH/hf-cache
export HF_MODEL_CACHE=$SCRATCH/hf-cache
export HUGGINGFACE_API_KEY=$(cat $W/.hf_token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$W/envs/llmmap-gpu-v2/bin/python
R=""; [ "${REDUCED:-0}" = "1" ] && R="--reduced"
echo "=== node: $(hostname) model: $MODEL K=$K start: $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
T0=$(date +%s)
if [ "${K:-1}" -gt 1 ]; then
  for i in $(seq 0 $((K-1))); do
    $PY experiments/d030_generate.py --model "$MODEL" --pilot $R --part $i --nparts $K > results/D030/logs/pilot-${SLURM_JOB_ID}-p$i.log 2>&1 &
  done
  wait
else
  $PY experiments/d030_generate.py --model "$MODEL" --pilot $R
fi
echo "node_wall_s $(( $(date +%s) - T0 ))"
echo "=== end: $(date) ==="
