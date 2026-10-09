#!/bin/bash
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/s1-%x-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/s1-%x-%j.err
# D030 / S1 generation (P1 §3; Review stop A: T_ZP = 25, our renderer, double-BOS control).
# Either LIST=<bucket file> (padded models, sequential) or MODEL=<hf id> K=<n> (batch-1-concurrent).
# Each model: generate MET + ZP rows (resumable), then its --dblbos row if it is one of the 4
# control models, then --finalize. A failing model does not stop the bucket.
W=/work/11280/zimuq1/vista
cd $W/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export HF_HOME=$SCRATCH/hf-cache HF_MODEL_CACHE=$SCRATCH/hf-cache
export HUGGINGFACE_API_KEY=$(cat $W/.hf_token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FIN=$W/miniconda3/envs/llmmap/bin/python
pyfor() {  # generation python for a model, from the manifest's gen_env (D028 precedent)
  case $($FIN -c "import json,sys;print(json.load(open('confs/baselines/baseline_ext_v1.json'))['models'][sys.argv[1]]['gen_env'])" "$1") in
    llmmap-gpu-v2) echo $W/envs/llmmap-gpu-v2/bin/python ;;
    llmmap-gpu) echo $W/miniconda3/envs/llmmap-gpu/bin/python ;;
    llmmap-internlm) echo $W/envs/llmmap-internlm/bin/python ;;
  esac
}
DBL="meta-llama/Meta-Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.3 google/gemma-2-9b-it meta-llama/Llama-2-7b-chat-hf"
echo "=== node: $(hostname) job $SLURM_JOB_ID start: $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
run_model() {
  local M=$1 PY=$(pyfor "$1") L=results/D030/logs/s1-$(echo $1 | tr / _)
  echo "--- $M ($PY) $(date)"
  $PY experiments/d030_generate.py --model "$M" --tzp 25 > $L.log 2>&1 || echo "GENERATE FAILED: $M"
  if [[ " $DBL " == *" $M "* ]]; then
    $PY experiments/d030_generate.py --model "$M" --tzp 25 --dblbos > $L.dblbos.log 2>&1 || echo "DBLBOS FAILED: $M"
    $FIN experiments/d030_generate.py --finalize --dblbos --model "$M" --tzp 25
  fi
  $FIN experiments/d030_generate.py --finalize --model "$M" --tzp 25
}
if [ -n "$LIST" ]; then
  while read -r M; do [ -n "$M" ] && run_model "$M"; done < "$LIST"
else
  PY=$(pyfor "$MODEL"); L=results/D030/logs/s1-$(echo $MODEL | tr / _)
  for i in $(seq 0 $((K-1))); do
    $PY experiments/d030_generate.py --model "$MODEL" --tzp 25 --part $i --nparts $K > $L.p$i.log 2>&1 &
  done
  wait
  $FIN experiments/d030_generate.py --finalize --model "$MODEL" --tzp 25
fi
echo "=== end: $(date) ==="
