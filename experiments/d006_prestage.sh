#!/bin/bash
# D006/S1 -- pre-stage pilot model weights from a LOGIN node.
# Per docs/ENV.md: login-node downloads are the recommended pattern;
# compute-node egress to the HF Hub is untested. Compute stays off login nodes.
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export HF_HOME=$SCRATCH/hf-cache
export HF_MODEL_CACHE=$SCRATCH/hf-cache
export HF_TOKEN=$(cat /work/11280/zimuq1/vista/.hf_token)
export HF_HUB_ENABLE_HF_TRANSFER=0
for m in \
  Qwen/Qwen2.5-0.5B-Instruct \
  Qwen/Qwen2.5-3B-Instruct \
  microsoft/Phi-3-mini-4k-instruct \
  microsoft/Phi-3-mini-128k-instruct \
  HuggingFaceH4/zephyr-7b-beta \
  microsoft/Phi-3-medium-4k-instruct \
  intfloat/multilingual-e5-large-instruct ; do
  echo "=== $m $(date +%H:%M:%S) ==="
  python -c "
import sys
from huggingface_hub import snapshot_download
snapshot_download('$m', allow_patterns=['*.json','*.safetensors','*.bin','*.model','*.txt'],
                  max_workers=4)
print('  ok')
" || echo "  FAILED: $m"
done
echo "=== prestage done $(date) ==="
du -sh $SCRATCH/hf-cache
