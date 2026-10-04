#!/bin/bash
#SBATCH -J d028-post
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D028/logs/post-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D028/logs/post-%j.err
# S1 finalize + reuse link -> S2 embed -> S3 tensor -> tail plot (llmmap-gpu; plot in llmmap)
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
        subprocess.run([sys.executable, "experiments/d028_generate.py", "--finalize", "--model", m, "--block", b])
PY
OMP_NUM_THREADS=16 $G experiments/d028_embed.py || exit 1
OMP_NUM_THREADS=1 $G experiments/d028_build_tensor.py --jobs 68 || exit 1
$P experiments/d028_plots.py
echo "=== end: $(date) ==="
