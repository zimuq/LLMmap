#!/bin/bash
#SBATCH -J d031pk
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D031/logs/%x-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D031/logs/%x-%j.err
# D031 S2/S3 packed job: $1 and $2 are two command strings (python script + args, run under experiments/),
# run concurrently on one node (CPU linear pool + GPU attention / budget). "-" = none. Each sequence in a
# string may chain several commands with ';'.
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export TOKENIZERS_PARALLELISM=false PYTHONPATH=.:experiments
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
echo "=== node: $(hostname) start: $(date) A: $1 | B: $2 ==="; free -g
run() { local tag=$1; shift; IFS=';' read -ra cmds <<< "$*"; for c in "${cmds[@]}"; do
  echo "[$tag] >>> $c $(date +%T)"; eval "python experiments/$c" || { echo "[$tag] FAILED: $c"; return 1; }; done; }
pids=()
[ "$1" != "-" ] && { OMP_NUM_THREADS=8 D031_WORKERS=6 run A "$1" > >(sed -u 's/^/[A] /') 2>&1 & pids+=($!); }
[ "$2" != "-" ] && { OMP_NUM_THREADS=16 run B "$2" > >(sed -u 's/^/[B] /') 2>&1 & pids+=($!); }
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
echo "=== end: $(date) rc=$rc ==="; exit $rc
