#!/bin/bash
#SBATCH -J d033
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A TG-NAIRR250513
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D033/logs/%x-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D033/logs/%x-%j.err
# D033: $1..$4 are up to four command streams run concurrently on one node ("-" or absent = none). A stream is
# a ';'-separated list of commands, each evaluated verbatim from the repo root. Thread counts are stated in the
# command (D031 R §7 item 1).
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export TOKENIZERS_PARALLELISM=false PYTHONPATH=.:experiments PYTHONFAULTHANDLER=1
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
echo "=== node: $(hostname) start: $(date) | $1 | $2 | $3 | $4 ==="; free -g; nvidia-smi --query-gpu=name,memory.total --format=csv
run() { local tag=$1; shift; IFS=';' read -ra cmds <<< "$*"; for c in "${cmds[@]}"; do
  echo "[$tag] >>> $c $(date +%T)"; eval "$c" || { echo "[$tag] FAILED: $c"; return 1; }; done; }
pids=(); tags=(A B C D); i=0
for s in "$1" "$2" "$3" "$4"; do
  t=${tags[$i]}; i=$((i+1))
  [ -n "$s" ] && [ "$s" != "-" ] && { run $t "$s" > >(sed -u "s/^/[$t] /") 2>&1 & pids+=($!); }
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
echo "=== end: $(date) rc=$rc ==="; exit $rc
