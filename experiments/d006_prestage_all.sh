#!/bin/bash
# Pre-stage all 37 A1 model weights from a LOGIN node (docs/ENV.md: login-node
# downloads are the recommended pattern; compute-node egress is untested).
# Needed regardless of the C7 ceiling, so it runs while that is settled.
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export HF_HOME=$SCRATCH/hf-cache HF_MODEL_CACHE=$SCRATCH/hf-cache
export HF_TOKEN=$(cat /work/11280/zimuq1/vista/.hf_token)
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
python - <<'PY'
import csv, sys
from huggingface_hub import snapshot_download
rows=[r for r in csv.DictReader(open("results/D001/model_metadata.csv"))
      if r["proprietary"].strip().lower() not in ("true","1","yes")
      and float(r["params_b"])<=14]
assert len(rows)==37, f"A1 anchor: {len(rows)}"
rows.sort(key=lambda r:-float(r["params_b"]))
ok=fail=0
for i,r in enumerate(rows,1):
    m=r["model"]
    try:
        snapshot_download(m, allow_patterns=['*.json','*.safetensors','*.bin','*.model','*.txt'],
                          max_workers=4)
        ok+=1; print(f"[{i}/37] ok    {m}", flush=True)
    except Exception as e:
        fail+=1; print(f"[{i}/37] FAIL  {m}: {type(e).__name__}: {str(e)[:160]}", flush=True)
print(f"\nstaged ok={ok} fail={fail}")
PY
du -sh $SCRATCH/hf-cache
echo "=== prestage-all done $(date) ==="
