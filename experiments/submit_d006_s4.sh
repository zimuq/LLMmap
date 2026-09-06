#!/bin/bash
# D006/S4 -- submit one job per model. 37 shards; qgh allows 20 running / 40
# submitted per user, so all 37 can sit in the queue and Slurm backfills as
# slots free. Largest models first so the long pole starts in round 1.
set -euo pipefail
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap

# Single-instance lock. This script is run by an auto top-up monitor, and two
# concurrent runs could both read the live-job list before either submits,
# double-submitting a model -- two processes appending to one shard file. Fail
# fast instead of racing.
exec 9>/tmp/.d006_submit.lock
flock -n 9 || { echo "another submitter is running; skipping this pass"; exit 0; }
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
# `|| true` is load-bearing: under `set -euo pipefail`, a job whose batch script
# has no `--model` (any non-shard job -- a diagnostic, S5/S6) makes grep exit 1,
# pipefail propagates it, and the whole submitter dies BEFORE printing anything.
# That silently blocked the 8 gated shards from ever being submitted.
LIVE=$(for j in $(squeue -u "$USER" -h -o "%i"); do
         scontrol write batch_script "$j" - 2>/dev/null | grep -oP -- '--model \K\S+' | head -1 || true
       done)

# Models known to be unrunnable, so the auto top-up loop does not resubmit them
# every few minutes. Each entry needs a reason and a decision owner.
#   internlm/internlm2_5-7b-chat -- RESOLVED 2026-09-06, no longer blocked. It
#     needs sentencepiece 0.1.99 (0.2.x rejects its vocabulary), use_fast=False,
#     trust_remote_code, and einops -- all provided by envs/llmmap-internlm, and
#     it is submitted by experiments/submit_d006_internlm.slurm rather than this
#     script, because it must run under that venv's python. Its shard is
#     COMPLETE. Kept out of this script's loop for that reason alone, not
#     because anything is wrong. See docs/ENV.md.
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
