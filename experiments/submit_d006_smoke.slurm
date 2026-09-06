#!/bin/bash
#SBATCH -J d006-smoke
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D006/slurm-smoke-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D006/slurm-smoke-%j.err
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments
export HF_HOME=$SCRATCH/hf-cache HF_MODEL_CACHE=$SCRATCH/hf-cache
export HUGGINGFACE_API_KEY=$(cat /work/11280/zimuq1/vista/.hf_token)
export D006_CORPUS=$SCRATCH/d006_smoke
rm -rf $D006_CORPUS; mkdir -p $D006_CORPUS
echo "=== smoke: 2 configs/pool on the SMALLEST and LARGEST models ==="
python experiments/d006_s4_shard.py --model Qwen/Qwen2.5-0.5B-Instruct --limit-configs 2
echo "--- rc=$? ---"
python experiments/d006_s4_shard.py --model microsoft/Phi-3-medium-4k-instruct --limit-configs 2
echo "--- rc=$? ---"
echo "=== output shape check ==="
python - <<'PY'
import json,glob,os
for f in sorted(glob.glob(os.environ["D006_CORPUS"]+"/*.jsonl")):
    rows=[json.loads(l) for l in open(f)]
    n=sum(len(r["traces"]) for r in rows)
    pools={}
    for r in rows: pools[r["dataset"]]=pools.get(r["dataset"],0)+1
    empt=sum(1 for r in rows for _,a in r["traces"] if not a.strip())
    print(f"{os.path.basename(f)}: {len(rows)} configs {pools}, {n} responses, {empt} empty")
    print("   sample:", json.dumps(rows[0]["traces"][0][1][:110]))
PY
echo "=== end $(date) ==="
