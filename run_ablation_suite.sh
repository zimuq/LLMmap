#!/bin/bash
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap
export HF_MODEL_CACHE=/work/11280/zimuq1/vista/.cache/huggingface
export HF_HOME=/work/11280/zimuq1/vista/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap

M=300
OUT=ablation_results.txt
> "$OUT"

run() {
  name="$1"; shift
  slug=$(echo "$name" | tr -c 'a-zA-Z0-9' '_' | cut -c1-40)
  rawlog="ablation_raw_${slug}.log"
  echo "===== $name =====" | tee -a "$OUT"
  python ablation_query_removal.py ./data/pretrained_models/default -k 3 -m $M "$@" > "$rawlog" 2>&1
  status=$?
  if [ $status -ne 0 ]; then
    echo "  [FAILED exit=$status -- see $rawlog]" | tee -a "$OUT"
  else
    grep -E "^(Total queries|  \[|Top-)" "$rawlog" | tee -a "$OUT"
  fi
  echo "" | tee -a "$OUT"
}

run "baseline (no perturbation)"
run "perturb_1_harmful_refusal_only" --perturb 1
run "perturb_1_3_7_full_alignment_family" --perturb 1 3 7
run "control_perturb_0_5_6_banner_grabbing" --perturb 0 5 6

echo "ALL_DONE" | tee -a "$OUT"
