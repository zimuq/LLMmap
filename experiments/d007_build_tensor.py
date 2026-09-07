"""
D007 / S2+S3 — build the separability tensor, with F2's winner's-curse correction.

Produces, for one analysis token budget:

  S_probe        (259, 666)  5-fold CV probe AUC on all 75 configs   [A3 primary]
  S_energy_raw   (259, 666)  energy distance, raw                    [diagnostic]
  S_energy_sf    (259, 666)  energy distance, scale-free             [aggregatable, F3]
  S_probe_A      (259, 666)  probe AUC on config half A only
  S_probe_B      (259, 666)  probe AUC on config half B only
  counts         (666, 2)    per-cell sample counts (constant (75,75), F4)

WHY A AND B EXIST (F2, approved). I4 aggregates per pair as `max_q S[q][p]`.
Each cell is estimated from 75 vs 75 points in 1024-d, so it carries real
variance, and **the max over 259 noisy estimates exceeds the true max** — a
winner's curse. The bias runs upward, which inflates per-pair coverage, which
raises the 10th percentile, which is exactly what triggers T1.6's
`TAIL STILL ABSENT` — a STOP and a METHOD revision escalation that could be
caused by estimator noise rather than by the models.

The correction is a clean split-half:
    select   q* = argmax_q S_probe_A[q][p]      (on half A)
    evaluate       S_probe_B[q*][p]             (on half B, never used to select)
`max_q A` and `B[q*]` are computed at the SAME sample size, so their difference
isolates selection bias rather than confounding it with sample size. S2a's
self-pair control (mean AUC 0.4904 against a true 0.5) already established that
the probe is not fitting noise at n=75, so what remains is selection, which is
precisely what this measures.

S2a also found `S_probe` saturating (20.4% of cells >= 0.99), so per A3's gate
`S_energy` is authoritative for interpretation. Both tensors are built regardless.

Usage:
    PYTHONPATH=.:experiments python experiments/d007_build_tensor.py --budget 200
    PYTHONPATH=.:experiments python experiments/d007_build_tensor.py --budget 100
"""
import os
import json
import time
import argparse
import itertools

import numpy as np
from joblib import Parallel, delayed

from d007_lib import load_corpus, probe_auc, energy

OUTDIR = "./results/D007"
CACHE = os.environ.get("D007_CACHE", "/tmp/d007_cubes")
SCHEMA = "cdqd-tensor-v1"          # I7, distinct from the corpus's tag
SPLIT_SEED = 20260907


def prepare_cubes(budget):
    """Materialise per-model (n_queries, n_configs, dim) build-split cubes to
    disk once, so parallel workers memory-map them instead of each holding a
    copy. 37 x 259 x 75 x 1024 float32 is ~2.9 GB in total."""
    d = os.path.join(CACHE, f"tok{budget}")
    os.makedirs(d, exist_ok=True)
    marker = os.path.join(d, "_models.json")
    if os.path.exists(marker):
        return json.load(open(marker)), d
    models, X = load_corpus(pool="build", token_budget=budget)
    for m in models:
        np.save(os.path.join(d, m.replace("/", "__") + ".npy"), X[m])
    json.dump(models, open(marker, "w"))
    return models, d


def pair_worker(cache_dir, a, b, split_a, split_b):
    """All 259 queries for ONE model pair. Returns 5 arrays of length n_queries."""
    A = np.load(os.path.join(cache_dir, a.replace("/", "__") + ".npy"), mmap_mode="r")
    B = np.load(os.path.join(cache_dir, b.replace("/", "__") + ".npy"), mmap_mode="r")
    nq = A.shape[0]
    pr = np.empty(nq, np.float32); er = np.empty(nq, np.float32)
    es = np.empty(nq, np.float32); pa = np.empty(nq, np.float32)
    pb = np.empty(nq, np.float32)
    for q in range(nq):
        Aq = np.asarray(A[q], dtype=np.float32)
        Bq = np.asarray(B[q], dtype=np.float32)
        pr[q] = probe_auc(Aq, Bq)
        er[q], es[q] = energy(Aq, Bq)
        pa[q] = probe_auc(Aq[split_a], Bq[split_a])
        pb[q] = probe_auc(Aq[split_b], Bq[split_b])
    return pr, er, es, pa, pb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("D007_JOBS", "68")))
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    t0 = time.time()
    models, cache_dir = prepare_cubes(args.budget)
    shape0 = np.load(os.path.join(cache_dir, models[0].replace("/", "__") + ".npy"),
                     mmap_mode="r").shape
    nq, nc, dim = shape0
    pairs = list(itertools.combinations(models, 2))
    print(f"budget={args.budget}  {len(models)} models  {nq} queries  "
          f"{nc} configs/cloud  dim {dim}  {len(pairs)} pairs  "
          f"({nq*len(pairs):,} cells)  prep {time.time()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(nc)
    split_a, split_b = np.sort(perm[:nc // 2]), np.sort(perm[nc // 2:2 * (nc // 2)])
    print(f"F2 split-half: |A|={len(split_a)} |B|={len(split_b)} "
          f"(disjoint, equal size so max_A vs B[q*] isolates SELECTION bias, "
          f"not sample size)", flush=True)

    t0 = time.time()
    res = Parallel(n_jobs=args.jobs, verbose=5, batch_size=1)(
        delayed(pair_worker)(cache_dir, a, b, split_a, split_b) for a, b in pairs)
    wall = time.time() - t0

    S = {k: np.empty((nq, len(pairs)), np.float32)
         for k in ("probe", "energy_raw", "energy_sf", "probe_A", "probe_B")}
    for j, (pr, er, es, pa, pb) in enumerate(res):
        S["probe"][:, j] = pr; S["energy_raw"][:, j] = er
        S["energy_sf"][:, j] = es; S["probe_A"][:, j] = pa; S["probe_B"][:, j] = pb

    tag = f"tok{args.budget}"
    for k, v in S.items():
        np.save(f"{OUTDIR}/S_{k}_{tag}.npy", v)
    counts = np.tile(np.array([nc, nc], np.int32), (len(pairs), 1))
    np.save(f"{OUTDIR}/counts_{tag}.npy", counts)

    meta = dict(
        schema_version=SCHEMA, token_budget=args.budget,
        shape=[nq, len(pairs)], n_models=len(models), n_configs_per_cloud=nc,
        dim=dim, models=models, pairs=[f"{a}|{b}" for a, b in pairs],
        probe_C=1.0, n_folds=5, split_seed=SPLIT_SEED,
        split_a=split_a.tolist(), split_b=split_b.tolist(),
        wall_s=round(wall, 1), core_hours=round(wall * args.jobs / 3600, 2),
        per_cell_counts="constant (75, 75); stored anyway per D007/P1 F4",
        energy_note="raw is NOT comparable across queries under unnormalised "
                    "embeddings; use scale-free for anything aggregated (F3)",
        f2_note="probe_A / probe_B support the winner's-curse correction: "
                "select argmax_q on A, evaluate that q on B. Same sample size "
                "on both sides, so the gap is selection bias alone.")
    json.dump(meta, open(f"{OUTDIR}/tensor_{tag}.meta.json", "w"), indent=1)

    print(f"\nbuilt in {wall/60:.1f} min ({meta['core_hours']} core-hours)")
    for k, v in S.items():
        print(f"  S_{k:11s} {v.shape} mean {float(v.mean()):.4f} "
              f"min {float(v.min()):.4f} max {float(v.max()):.4f}")
    print(f"written: {OUTDIR}/S_*_{tag}.npy")


if __name__ == "__main__":
    main()
