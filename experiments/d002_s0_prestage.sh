#!/bin/bash
# D002 / P1 — S0.3: pre-stage benchmark model weights on the LOGIN node.
# Compute-node external egress on Vista is untested; downloading here removes that
# risk from the timed run. Weights go to $SCRATCH (P1.a): no quota there, and they
# are freely re-downloadable, so they must not consume the 1 TB $WORK budget.
set -e
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap
export HF_HOME=$SCRATCH/hf-cache
TOK=$(tr -d '\r\n' < /work/11280/zimuq1/vista/.hf_token)

python - "$TOK" <<'PY'
import sys, time
from huggingface_hub import snapshot_download
tok = sys.argv[1]
for m in ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]:
    t = time.time()
    p = snapshot_download(m, token=tok,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    print(f"STAGED {m}  {time.time()-t:.0f}s  -> {p}", flush=True)
PY

du -sh "$SCRATCH/hf-cache"
echo "PRESTAGE_DONE"
