"""D010 / S2–S4 — precompute, run the joint greedy, emit the nested k=1..8 chain.

Call 1 (approved): energy distance, scale-free.
Call 2 (approved): exact search, no approximation -- the accumulator makes it
0.02 core-hours.
Review amendment (2026-09-09): also run the k=8 MMD chain as a robustness check.

Usage:  PYTHONPATH=.:experiments python experiments/d010_select.py
"""
import os
import json
import time
import hashlib
import itertools

import numpy as np

from LLMmap.joint_statistic import energy_from_acc, energy_direct
from LLMmap.joint_greedy import joint_greedy, peak_k
from d010_lib import OUT
from d007_lib import load_corpus

K = 8
GAMMA = 0.1
POOL = "./confs/queries/pool_v1.json"


def precompute(X, models, pairs):
    """Per-query squared-distance blocks. XX2 is per MODEL (37) not per pair --
    a cloud's within-distances do not depend on its partner."""
    nq, nc, _ = X[models[0]].shape
    XX2 = np.empty((nq, len(models), nc, nc), np.float32)
    XY2 = np.empty((nq, len(pairs), nc, nc), np.float32)
    ix = {m: i for i, m in enumerate(models)}
    for q in range(nq):
        C = [np.asarray(X[m][q], np.float32) for m in models]
        n2 = [np.einsum("ij,ij->i", c, c) for c in C]
        for mi in range(len(models)):
            d = n2[mi][:, None] + n2[mi][None, :] - 2.0 * (C[mi] @ C[mi].T)
            XX2[q, mi] = np.maximum(d, 0)
        for pi, (a, b) in enumerate(pairs):
            iaa, ibb = ix[a], ix[b]
            d = n2[iaa][:, None] + n2[ibb][None, :] - 2.0 * (C[iaa] @ C[ibb].T)
            XY2[q, pi] = np.maximum(d, 0)
        if (q + 1) % 50 == 0:
            print(f"  precompute {q+1}/{nq}", flush=True)
    return XX2, XY2


def main():
    os.makedirs(OUT, exist_ok=True)
    models, X = load_corpus(pool="build")
    pairs = list(itertools.combinations(models, 2))
    ix = {m: i for i, m in enumerate(models)}
    ia = np.array([ix[a] for a, _ in pairs])
    ib = np.array([ix[b] for _, b in pairs])
    pool = json.load(open(POOL))
    print(f"{len(models)} models, {len(pairs)} pairs, "
          f"{X[models[0]].shape[0]} queries", flush=True)

    t = time.time()
    XX2, XY2 = precompute(X, models, pairs)
    t_pre = time.time() - t
    print(f"[S2] precompute {t_pre/60:.1f} min, XY2 {XY2.nbytes/1e9:.1f} GB",
          flush=True)

    # accumulator == direct, asserted rather than assumed (P1 verification)
    qs = [3, 11, 40, 77]
    acc_xy = XY2[qs, 0].sum(0)
    acc_xx = XX2[qs, ia[0]].sum(0)
    acc_yy = XX2[qs, ib[0]].sum(0)
    _, sf_acc = energy_from_acc(acc_xy, acc_xx, acc_yy)
    A = np.concatenate([np.asarray(X[pairs[0][0]][q], np.float32) for q in qs], 1)
    B = np.concatenate([np.asarray(X[pairs[0][1]][q], np.float32) for q in qs], 1)
    _, sf_dir = energy_direct(A, B)
    assert abs(sf_acc - sf_dir) < 1e-4, (sf_acc, sf_dir)
    print(f"[S2] accumulator == direct: {sf_acc:.8f} vs {sf_dir:.8f} "
          f"(|Δ| {abs(sf_acc-sf_dir):.2e})", flush=True)

    print(f"[S3] joint greedy, energy, gamma={GAMMA}:", flush=True)
    t = time.time()
    sel, trace = joint_greedy(XY2, XX2, ia, ib, K, GAMMA, "energy")
    t_e = time.time() - t
    pk, pv = peak_k(trace)
    print(f"[S3] energy chain {sel} in {t_e/60:.1f} min; objective PEAKS AT "
          f"k={pk} ({pv:.5f})", flush=True)

    print(f"[S3] joint greedy, MMD (review amendment, k={K} check):", flush=True)
    t = time.time()
    sel_m, trace_m = joint_greedy(XY2, XX2, ia, ib, K, GAMMA, "mmd")
    t_m = time.time() - t
    pk_m, pv_m = peak_k(trace_m)
    print(f"[S3] mmd chain {sel_m} in {t_m/60:.1f} min; peaks at k={pk_m}",
          flush=True)

    d008 = json.load(open("./results/D008/selection.json"))
    cvar_chain = d008["conditions"]["cvar_max"]["queries"][:K]
    out = dict(
        schema="d010-selection-v1", k=K, gamma=GAMMA, statistic="energy_scale_free",
        aggregation="set-level (joint, concatenated); NOT I4's per-query MAX",
        models=models, n_pairs=len(pairs),
        joint_energy=dict(queries=sel, trace=trace, peak_k=pk, peak_objective=pv,
                          monotone=bool(all(trace[i]["objective_cvar"] <=
                                            trace[i + 1]["objective_cvar"]
                                            for i in range(len(trace) - 1))),
                          wall_min=round(t_e / 60, 2)),
        joint_mmd_k8=dict(queries=sel_m, trace=trace_m, peak_k=pk_m,
                          peak_objective=pv_m, wall_min=round(t_m / 60, 2),
                          note="review amendment 2026-09-09; per-STEP global "
                               "bandwidth (candidates within a step share a set "
                               "size, so one sigma^2 keeps them comparable; "
                               "cross-k MMD values are not comparable and are "
                               "not plotted as a curve)"),
        overlap_with_d008_cvar=dict(
            energy=len(set(sel) & set(cvar_chain)),
            mmd=len(set(sel_m) & set(cvar_chain)),
            energy_vs_mmd=len(set(sel) & set(sel_m)),
            d008_cvar_chain=cvar_chain),
        query_texts={str(q): pool["queries"][q]["text"][:160] for q in
                     sorted(set(sel) | set(sel_m))},
        precompute_min=round(t_pre / 60, 2),
        accumulator_check=dict(acc=sf_acc, direct=sf_dir,
                               abs_diff=abs(sf_acc - sf_dir)),
        inputs=dict(pool_sha256=pool["sha256"],
                    d008_selection_sha256=hashlib.sha256(
                        open("./results/D008/selection.json", "rb").read()
                    ).hexdigest()))
    json.dump(out, open(f"{OUT}/selection.json", "w"), indent=1)

    print(f"\n[S4] energy chain overlap with D008's CVaR chain: "
          f"{out['overlap_with_d008_cvar']['energy']}/{K}; "
          f"energy vs mmd {out['overlap_with_d008_cvar']['energy_vs_mmd']}/{K}")
    print(f"[S4] objective monotone in k: {out['joint_energy']['monotone']}")
    print(f"written: {OUT}/selection.json")


if __name__ == "__main__":
    main()
