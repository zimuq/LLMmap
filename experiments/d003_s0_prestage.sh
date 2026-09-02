#!/bin/bash
# D003 S0 — pre-stage the generator on the login node.
# Generator overruled by design side (D003 Review, Call 2) to the strictly
# decoupled OLMo-2: AllenAI has no presence in the 52-model universe.
set -e
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap
export HF_HOME=$SCRATCH/hf-cache
TOK=$(tr -d '\r\n' < /work/11280/zimuq1/vista/.hf_token)
python - "$TOK" <<'PY'
import sys, time
from huggingface_hub import snapshot_download
t=time.time()
p=snapshot_download("allenai/OLMo-2-1124-13B-Instruct", token=sys.argv[1],
                    allow_patterns=["*.json","*.safetensors","*.txt","*.model"])
print(f"STAGED OLMo-2-13B  {time.time()-t:.0f}s -> {p}", flush=True)
PY
du -sh $SCRATCH/hf-cache
echo "D003_PRESTAGE_DONE"
