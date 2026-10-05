#!/bin/bash
#SBATCH -J d028-emb
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D028/logs/post-embed-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D028/logs/post-embed-%j.err
# post part 1: reuse link + finalize every (model, block) + I5 embedding (restartable: existing .npy are skipped)
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments HF_HOME=$SCRATCH/hf-cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
G=/work/11280/zimuq1/vista/miniconda3/envs/llmmap-gpu/bin/python
P=/work/11280/zimuq1/vista/miniconda3/envs/llmmap/bin/python
echo "=== node: $(hostname) start: $(date) ==="
$G experiments/d028_link.py || exit 1
$G - <<'PY'
import json, subprocess, sys
uni = json.load(open("results/D027/universe_v2.json"))["models"]
reuse = set(json.load(open("results/D028/reuse.json")))
for m in uni:
    for b in (["ml"] if m in reuse else ["pool", "ml"]):
        subprocess.run([sys.executable, "experiments/d028_generate.py", "--finalize", "--model", m, "--block", b], check=True)
PY
OMP_NUM_THREADS=16 $G experiments/d028_embed.py || exit 1
echo "=== end: $(date) ==="
