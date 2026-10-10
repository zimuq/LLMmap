#!/bin/bash
#SBATCH -J d030-s1-post
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/s1post-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/s1post-%j.err
# D030 / S1 health report (after all generation jobs, afterany). No score of any kind.
# Re-finalizes every model first so all final.json carry the same stats definition
# (collapse counted only over prompt groups with >= 2 samples; fixed during S1).
W=/work/11280/zimuq1/vista
cd $W/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments
PY=$W/miniconda3/envs/llmmap/bin/python
for M in $($PY -c "import json;print(' '.join(json.load(open('confs/baselines/baseline_ext_v1.json'))['models']))"); do
  $PY experiments/d030_generate.py --finalize --model "$M" --tzp 25
done
for M in meta-llama/Meta-Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.3 google/gemma-2-9b-it meta-llama/Llama-2-7b-chat-hf; do
  $PY experiments/d030_generate.py --finalize --dblbos --model "$M" --tzp 25
done
$PY experiments/d030_generate.py --health --model ALL --tzp 25
