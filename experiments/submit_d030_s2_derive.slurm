#!/bin/bash
#SBATCH -J d030-s2-derive
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/s2derive-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D030/logs/s2derive-%j.err
# D030 / P2 S2.1 derive + S2.0 freeze (Review D030/P2: B1 within-prompt sigma). No score computed.
# ZP-200 re-tokenisation (CPU, 3 generation envs) runs alongside e5 (GPU); then mpnet, LLMMAP-OS, freeze.
set -o pipefail
W=/work/11280/zimuq1/vista
cd $W/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export HF_HOME=$SCRATCH/hf-cache HF_MODEL_CACHE=$SCRATCH/hf-cache
export HUGGINGFACE_API_KEY=$(cat $W/.hf_token)
GPU=$W/miniconda3/envs/llmmap-gpu/bin/python
ZP=$W/envs/d030-zp/bin/python
L=results/D030/logs
echo "=== node: $(hostname) job $SLURM_JOB_ID start: $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$W/envs/llmmap-gpu-v2/bin/python experiments/d030_s2_derive.py zp200 --env llmmap-gpu-v2 > $L/s2_zp200_v2.log 2>&1 &
P1=$!
$W/miniconda3/envs/llmmap-gpu/bin/python experiments/d030_s2_derive.py zp200 --env llmmap-gpu > $L/s2_zp200_gpu.log 2>&1 &
P2=$!
$W/envs/llmmap-internlm/bin/python experiments/d030_s2_derive.py zp200 --env llmmap-internlm > $L/s2_zp200_internlm.log 2>&1 &
P3=$!
$GPU experiments/d030_s2_derive.py met_e5 || { echo "MET_E5 FAILED"; exit 1; }
echo "met_e5 done $(date)"
wait $P1 && wait $P2 && wait $P3 || { echo "ZP200 FAILED"; tail -5 $L/s2_zp200_*.log; exit 1; }
echo "zp200 done $(date)"
$ZP experiments/d030_s2_derive.py mpnet || { echo "MPNET FAILED"; exit 1; }
echo "mpnet done $(date)"
$GPU experiments/d030_s2_derive.py os || { echo "OS FAILED"; exit 1; }
echo "os done $(date)"
$GPU experiments/d030_s2_derive.py freeze || { echo "FREEZE FAILED"; exit 1; }
echo "=== end: $(date) ==="
