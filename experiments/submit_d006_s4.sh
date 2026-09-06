#!/bin/bash
# D006/S4 -- submit one job per model. 37 shards; qgh allows 20 running / 40
# submitted per user, so all 37 can sit in the queue and Slurm backfills as
# slots free. Largest models first so the long pole starts in round 1.
set -euo pipefail
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
mkdir -p results/D006/shards data/corpus_v1

MODELS=$(python - <<'PY'
import csv
rows=[r for r in csv.DictReader(open("results/D001/model_metadata.csv"))
      if r["proprietary"].strip().lower() not in ("true","1","yes")
      and float(r["params_b"])<=14]
assert len(rows)==37, f"A1 anchor violated: {len(rows)} models, expected 37"
for r in sorted(rows, key=lambda r: -float(r["params_b"])):
    print(r["model"])
PY
)

# Models that already have a queued/running job. Keyed on the FULL model name
# read back out of each job's batch script -- NOT on the Slurm job name, which
# is truncated to 20 chars and therefore collides across genuinely different
# models (granite-3.0/3.1, Phi-3-medium-128k/4k, Mistral v0.1/v0.2/v0.3,
# Llama-3.2-1B/3B). Keying on the truncated name once caused four live jobs to
# be cancelled as false "duplicates".
LIVE=$(for j in $(squeue -u "$USER" -h -o "%i"); do
         scontrol write batch_script "$j" - 2>/dev/null | grep -oP -- '--model \K\S+' | head -1
       done)

# Models known to be unrunnable, so the auto top-up loop does not resubmit them
# every few minutes. Each entry needs a reason and a decision owner.
#   internlm/internlm2_5-7b-chat -- will not load under transformers 4.51.3:
#     repo ships no tokenizer.json, forcing a SentencePiece->fast conversion
#     that fails; use_fast=False returns a bool instead of a tokenizer.
#     Escalated to design side (D006 BLOCKED criterion). Remove this entry
#     once that is resolved.
BLOCKED="internlm/internlm2_5-7b-chat"

n=0
for m in $MODELS; do
  if echo "$BLOCKED" | grep -qxF "$m"; then
    echo "BLOCKED (see submitter comment): $m"; continue
  fi
  slug=${m//\//__}
  if echo "$LIVE" | grep -qxF "$m"; then
    echo "skip (job already queued/running): $m"; continue
  fi
  if [ -f "data/corpus_v1/${slug}.status.json" ] && \
     grep -q '"status": "COMPLETE"' "data/corpus_v1/${slug}.status.json" 2>/dev/null; then
    echo "skip (already COMPLETE): $m"; continue
  fi
  # Skip models whose weights are not staged. docs/ENV.md: compute-node egress
  # to the HF Hub is untested, so a shard must never be the thing that
  # discovers a model is ungated-but-undownloaded. Re-running this script after
  # the gate clears picks them up.
  cachedir="$SCRATCH/hf-cache/hub/models--${m//\//--}"
  if [ ! -d "$cachedir" ] || [ -z "$(find "$cachedir" \( -name '*.safetensors' -o -name '*.bin' \) -print -quit 2>/dev/null)" ]; then
    echo "SKIP (weights not staged, likely gated): $m"; continue
  fi
  n=$((n+1))
  sbatch --parsable \
    -J "d6-${slug:0:20}" -p gh -N 1 -n 1 -t 12:00:00 \
    -o "/work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D006/shards/${slug}-%j.out" \
    -e "/work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D006/shards/${slug}-%j.err" \
    --wrap "source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
            conda activate llmmap-gpu
            export OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
            cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
            export PYTHONPATH=.:experiments
            export HF_HOME=\$SCRATCH/hf-cache HF_MODEL_CACHE=\$SCRATCH/hf-cache
            export HUGGINGFACE_API_KEY=\$(cat /work/11280/zimuq1/vista/.hf_token)
            echo \"=== \$(hostname) \$(date) $m ===\"
            python experiments/d006_s4_shard.py --model $m
            echo \"=== end \$(date) ===\"" \
    | xargs -I{} echo "submitted {}  $m"
done
echo "--- $n shard job(s) submitted ---"
