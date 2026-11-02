"""D036 / S3 — open-set readout (2) raw on S_test held-out targets (D033 R §8 protocol, frozen; Call 3 (a)).

For every query in the union of the three fold arms' k <= 8 chains (chains.json -> folds), the point-to-cloud terms
A[t, m, j] / W[m, j] are recomputed from the v2 S_build / S_test cubes (d033_lib.nc_terms). Asserts:
  - terms of queries also in D033's nc_test_v2.npz equal D033's to |d| <= 1e-12;
  - re-scoring CLOUD^fold (= D033 JG^fold) for all 5 folds, k = 1..8, gives cnt_model / cnt_pair_all equal to
    D033's counts_v2.npz -> nc|model|JG|raw|<k>|...
Pooled held-out counts (each model scored with the chain of the fold it is held out in, against all 85 enrolled
models; argmin, tie -> lower index; d033_lib.counts) for CLOUD-raw^fold and CENTROID-raw^fold, k = 1..8
-> openset_counts.npz; metadata -> openset.json. Nothing is summarised (S4 is the first reader).
Usage:  OMP_NUM_THREADS=8 PYTHONPATH=.:experiments python experiments/d036_openset.py
"""
import os
import json
import time

import numpy as np

import d031_lib as L31
import d033_lib as L33
import d036_lib as L

KS = list(range(1, L.K + 1))
NEW = ("CLOUD-raw", "CENTROID-raw")
NCFG = L33.NCFG


def pooled_counts(A, W, pos, folds, arm, k, ms, ia, ib):
    n = len(ms)
    D = np.full((n * NCFG, n), np.nan)
    for F in folds:
        cols = [pos[q] for q in F["arms"][arm][:k]]
        rows = np.concatenate([np.arange(ms.index(m) * NCFG, (ms.index(m) + 1) * NCFG) for m in F["heldout"]])
        D[rows] = L33.nc_scores(A[rows], W, cols, "raw")
    assert np.isfinite(D).all()
    return L33.counts(D, n, ia, ib)


def main():
    t0 = time.time()
    ch = json.load(open(f"{L.OUT}/chains.json"))
    assert ch["frozen_before_training"] and not ch["stops"]
    assert os.popen(f"git log --oneline -- {L.OUT}/chains.json").read().strip(), "chains.json not committed"
    ms = L31.models(L.VIEW)
    ia, ib = L31.pairs(L.VIEW)
    folds = ch["folds"]
    assert sorted(m for F in folds for m in F["heldout"]) == sorted(ms)
    Q = sorted({q for F in folds for a in F["arms"].values() for q in a[:L.K]})

    cubes = L31.cubes(L.VIEW, ("build", "test"))
    A, W = L33.nc_terms(cubes["build"][1], cubes["test"][1], ms, Q)
    del cubes
    np.savez_compressed(f"{L.OUT}/nc_test.npz", A=A, W=W, U=np.array(Q), models=np.array(ms))
    print(f"[S3] terms for {len(Q)} queries ({(time.time() - t0) / 60:.1f} min)", flush=True)

    z33 = np.load("./results/D033/nc_test_v2.npz")
    assert list(z33["models"]) == ms
    U33 = [int(q) for q in z33["U"]]
    shared = [q for q in Q if q in U33]
    dA = max(float(np.abs(A[:, :, Q.index(q)] - z33["A"][:, :, U33.index(q)]).max()) for q in shared)
    dW = max(float(np.abs(W[:, Q.index(q)] - z33["W"][:, U33.index(q)]).max()) for q in shared)
    assert dA <= 1e-12 and dW <= 1e-12, (dA, dW)
    print(f"[S3] terms vs D033 ({len(shared)} shared queries): max |dA| {dA:.1e}, |dW| {dW:.1e} -> PASS", flush=True)

    pos = {q: j for j, q in enumerate(Q)}
    c33 = np.load("./results/D033/counts_v2.npz")
    for k in KS:
        c = pooled_counts(A, W, pos, folds, "CLOUD", k, ms, ia, ib)
        for x in ("cnt_model", "cnt_pair_all"):
            assert np.array_equal(c[x], c33[f"nc|model|JG|raw|{k}|{x}"]), f"CLOUD^fold k={k} {x} differs from D033"
    print("[S3] CLOUD^fold re-scored (5 folds, k = 1..8) = D033 nc|model|JG|raw counts exactly -> PASS", flush=True)

    store = {}
    for arm in NEW:
        for k in KS:
            c = pooled_counts(A, W, pos, folds, arm, k, ms, ia, ib)
            for x in ("cnt_model", "cnt_pair_all"):
                store[f"nc|{arm}|{k}|{x}"] = c[x]
    np.savez_compressed(f"{L.OUT}/openset_counts.npz", **store)
    L31.jdump(dict(schema="d036-openset-v1", readout="(2) raw, d033_lib.nc_scores(..., 'raw'); D033 R §8 protocol",
                   chains_sha256=L31.sha(f"{L.OUT}/chains.json"), queries=Q, n_queries=len(Q),
                   nc_test_sha256=L31.sha(f"{L.OUT}/nc_test.npz"),
                   reproduction=dict(shared_queries=shared, max_abs_dA=dA, max_abs_dW=dW,
                                     cloud_fold_counts_equal_d033="all 5 folds, k = 1..8, cnt_model + cnt_pair_all"),
                   arms_written=list(NEW), cloud_fold_counts="D033 counts_v2.npz nc|model|JG|raw|<k> (verbatim)",
                   s_test="written to counts only; not summarised in S3", code_commit=L31.git_head(),
                   wall_min=round((time.time() - t0) / 60, 2)), f"{L.OUT}/openset.json")
    print(f"[S3] wrote openset_counts.npz, openset.json ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
