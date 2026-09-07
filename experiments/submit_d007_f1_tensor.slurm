#!/bin/bash
#SBATCH -J d007-f1-tensor
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D007/slurm-f1t-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D007/slurm-f1t-%j.err
# gh-dev rather than gh: gh is 566/571 allocated with 178 queued (ETA +7h) and
# this is ~40 min of work. Both stages are restartable -- the re-embed skips
# shards already written, the tensor build is minutes -- so a 2h cap is safe.
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
export TOKENIZERS_PARALLELISM=false
export HF_HOME=$SCRATCH/hf-cache HF_MODEL_CACHE=$SCRATCH/hf-cache
export HUGGINGFACE_API_KEY=$(cat /work/11280/zimuq1/vista/.hf_token)
export D007_CACHE=$SCRATCH/d007_cubes
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments
echo "=== $(hostname) $(date) ==="

echo "### F1 pass 1: 36 models (llmmap-gpu)"
OMP_NUM_THREADS=16 python experiments/d007_f1_reembed_trunc.py --budget 100
echo "### F1 pass 2: internlm (dedicated env, sentencepiece 0.1.99)"
OMP_NUM_THREADS=16 /work/11280/zimuq1/vista/envs/llmmap-internlm/bin/python \
    experiments/d007_f1_reembed_trunc.py --budget 100 --only internlm/internlm2_5-7b-chat

N=$(ls data/corpus_v1/embeddings_tok100/*.npy 2>/dev/null | wc -l)
echo "### tok100 shards: $N/37"

# one BLAS thread per worker: parallelism is over cells; letting each
# LogisticRegression spawn threads oversubscribes 72 cores badly
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "### tensor, budget 200"
python experiments/d007_build_tensor.py --budget 200
if [ "$N" -eq 37 ]; then
  echo "### tensor, budget 100 (F1 comparison)"
  python experiments/d007_build_tensor.py --budget 100
else
  echo "SKIP budget 100: only $N/37 tok100 shards"
fi
echo "=== end $(date) ==="
