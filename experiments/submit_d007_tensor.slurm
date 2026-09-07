#!/bin/bash
#SBATCH -J d007-tensor
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -o /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D007/slurm-tensor-%j.out
#SBATCH -e /work/11280/zimuq1/vista/LLMmap-project/LLMmap/results/D007/slurm-tensor-%j.err
source /work/11280/zimuq1/vista/miniconda3/etc/profile.d/conda.sh
conda activate llmmap-gpu
# one BLAS thread per worker: the parallelism is over cells, and letting each
# LogisticRegression spawn its own threads oversubscribes 72 cores badly
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export D007_CACHE=$SCRATCH/d007_cubes
cd /work/11280/zimuq1/vista/LLMmap-project/LLMmap
export PYTHONPATH=.:experiments
echo "=== $(hostname) $(date) ==="
echo "--- budget 200 (as generated) ---"
python experiments/d007_build_tensor.py --budget 200
if [ -d data/corpus_v1/embeddings_tok100 ] && \
   [ "$(ls data/corpus_v1/embeddings_tok100/*.npy 2>/dev/null | wc -l)" -eq 37 ]; then
  echo "--- budget 100 (F1 comparison) ---"
  python experiments/d007_build_tensor.py --budget 100
else
  echo "SKIP budget 100: tok100 embeddings incomplete ($(ls data/corpus_v1/embeddings_tok100/*.npy 2>/dev/null | wc -l)/37)"
fi
echo "=== end $(date) ==="
