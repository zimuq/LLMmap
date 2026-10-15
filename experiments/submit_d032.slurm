#!/bin/bash
#SBATCH -J d032
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A TG-NAIRR250513
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D032/logs/%x-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D032/logs/%x-%j.err
# D032 S0-S3: $1 and $2 are two command streams run concurrently on one node ("-" = none). A stream is a
# ';'-separated list of commands, each evaluated verbatim from the repo root, e.g.
#   "OMP_NUM_THREADS=8 D031_THREADS=8 D031_WORKERS=8 python experiments/d032_train.py lin v2 0 2"
# Thread counts are always stated in the command (D031 R §7 item 1: lbfgs bit-exactness depends on them).
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export TOKENIZERS_PARALLELISM=false PYTHONPATH=.:experiments PYTHONFAULTHANDLER=1
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
echo "=== node: $(hostname) start: $(date) A: $1 | B: $2 ==="; free -g
run() { local tag=$1; shift; IFS=';' read -ra cmds <<< "$*"; for c in "${cmds[@]}"; do
  echo "[$tag] >>> $c $(date +%T)"; eval "$c" || { echo "[$tag] FAILED: $c"; return 1; }; done; }
pids=()
[ "$1" != "-" ] && { run A "$1" > >(sed -u 's/^/[A] /') 2>&1 & pids+=($!); }
[ "$2" != "-" ] && { run B "$2" > >(sed -u 's/^/[B] /') 2>&1 & pids+=($!); }
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
echo "=== end: $(date) rc=$rc ==="; exit $rc
