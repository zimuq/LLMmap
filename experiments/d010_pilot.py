"""D010 / S1 — the mandatory degeneracy check, plus the cost measurement Call 2 needs.

D010/S1: concatenating k queries' 1024-d embeddings gives an 8192-d joint vector
at k=8, estimated from 75 configs. High dimension / small sample is a known
failure mode for distance-based statistics. **If the statistic degrades sharply
with k, say so and stop** -- that is D010's INCONCLUSIVE branch, not a licence to
push on.

The check: does the joint statistic still separate REAL model pairs from a
same-model NULL as k grows? The null is the self-pair control this project has
used since D007/S2a and D008/S7 -- one model's 75 build configs split in half,
same query set, so the true statistic is ~0 by construction. Degeneracy would
show up as real pairs and the null converging.

Also measured, because Call 2 turns on it:
  * cost of the accumulator path vs the naive recompute-per-candidate path
  * that the two agree numerically (the accumulator is an optimisation, so it
    has to be verified rather than trusted)

Usage:  PYTHONPATH=.:experiments python experiments/d010_pilot.py
"""
import os
import json
import time
import random
import itertools

import numpy as np

from d010_lib import (per_query_sqdists, joint_energy_from_acc,
                      joint_energy_direct, concat_cloud, pair_index, OUT)
from d007_lib import load_corpus
from d008_lib import near_relative_pairs

KS = [1, 2, 4, 8, 16]
N_PAIRS = 60
N_NULL = 40
SEED = 20260910


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    random.seed(SEED)
    models, X = load_corpus(pool="build")
    nq, nc, dim = X[models[0]].shape
    pairs = pair_index(models)
    hard = near_relative_pairs(models)
    print(f"{len(models)} models, {nq} queries, {nc} configs, dim {dim}, "
          f"{len(pairs)} pairs", flush=True)

    # query sets of each size, drawn once and REUSED across k so the k-curve is
    # nested -- otherwise a change with k could be the draw, not the dimension
    order = rng.permutation(nq)[:max(KS)].tolist()
    qsets = {k: order[:k] for k in KS}

    # oversample near-relative pairs: if anything is going to lose signal as k
    # grows it is the hard pairs, and those are the ones the method exists for
    hard_ix = list(hard.keys())
    easy_ix = [i for i in range(len(pairs)) if i not in hard]
    sample = ([random.choice(hard_ix) for _ in range(N_PAIRS // 2)] +
              [random.choice(easy_ix) for _ in range(N_PAIRS // 2)])

    res = {}
    for k in KS:
        qs = qsets[k]
        t = time.time()
        real, real_hard = [], []
        for pi in sample:
            a, b = pairs[pi]
            A, B = concat_cloud(X, a, qs), concat_cloud(X, b, qs)
            _, sf = joint_energy_direct(A, B)
            real.append(sf)
            if pi in hard:
                real_hard.append(sf)
        null = []
        for _ in range(N_NULL):
            m = random.choice(models)
            C = concat_cloud(X, m, qs)
            p = rng.permutation(nc)
            h = nc // 2
            _, sf = joint_energy_direct(C[p[:h]], C[p[h:2 * h]])
            null.append(sf)
        real, null = np.array(real), np.array(null)
        rh = np.array(real_hard)

        # AUROC of real-vs-null: 1.0 = perfectly separated, 0.5 = degenerate
        allv = np.concatenate([real, null])
        lab = np.r_[np.ones(len(real)), np.zeros(len(null))]
        o = np.argsort(allv)
        ranks = np.empty(len(allv)); ranks[o] = np.arange(1, len(allv) + 1)
        n1, n0 = lab.sum(), (1 - lab).sum()
        auroc = float((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

        res[k] = dict(
            n_real=len(real), n_null=len(null),
            real_median=float(np.median(real)), real_p10=float(np.percentile(real, 10)),
            real_min=float(real.min()),
            hard_median=float(np.median(rh)), hard_min=float(rh.min()),
            null_median=float(np.median(null)), null_p95=float(np.percentile(null, 95)),
            null_max=float(null.max()),
            separation_ratio=float(np.median(real) / np.percentile(null, 95)),
            hard_separation_ratio=float(np.median(rh) / np.percentile(null, 95)),
            auroc_real_vs_null=auroc,
            frac_hard_below_null_p95=float((rh <= np.percentile(null, 95)).mean()),
            wall_s=round(time.time() - t, 1))
        r = res[k]
        print(f"[S1] k={k:2d}: real med {r['real_median']:.4f} (hard "
              f"{r['hard_median']:.4f}) vs null p95 {r['null_p95']:.4f}  "
              f"ratio {r['separation_ratio']:.2f} (hard "
              f"{r['hard_separation_ratio']:.2f})  AUROC {auroc:.4f}  "
              f"({r['wall_s']}s)", flush=True)

    # ---- accumulator correctness + cost (Call 2)
    sub_models = models[:6]
    t = time.time()
    XX2, XY2 = per_query_sqdists({m: X[m] for m in sub_models}, sub_models,
                                 queries=range(24))
    t_pre = time.time() - t
    sub_pairs = pair_index(sub_models)
    qs = [0, 3, 7, 11]
    pi = 0
    a, b = sub_pairs[pi]
    ia, ib = sub_models.index(a), sub_models.index(b)
    acc_xy = XY2[qs, pi].sum(0); acc_xx = XX2[qs, ia].sum(0)
    acc_yy = XX2[qs, ib].sum(0)
    _, sf_acc = joint_energy_from_acc(acc_xy, acc_xx, acc_yy)
    _, sf_dir = joint_energy_direct(concat_cloud(X, a, qs), concat_cloud(X, b, qs))

    # one greedy step's worth of work, both ways, on the subset
    t = time.time()
    for q in range(24):
        for p2 in range(len(sub_pairs)):
            aa, bb = sub_pairs[p2]
            joint_energy_from_acc(XY2[qs + [q], p2].sum(0),
                                  XX2[qs + [q], sub_models.index(aa)].sum(0),
                                  XX2[qs + [q], sub_models.index(bb)].sum(0))
    t_acc = time.time() - t
    t = time.time()
    for q in range(6):
        for p2 in range(len(sub_pairs)):
            aa, bb = sub_pairs[p2]
            joint_energy_direct(concat_cloud(X, aa, qs + [q]),
                                concat_cloud(X, bb, qs + [q]))
    t_dir = (time.time() - t) * 4          # scale 6 -> 24 candidates

    n_cells_step = 259 * 666
    cells_sub = 24 * len(sub_pairs)
    res["cost"] = dict(
        precompute_subset_s=round(t_pre, 1),
        precompute_full_projection_core_h=round(
            t_pre * (259 / 24) * (len(pairs) / len(sub_pairs)) / 3600, 2),
        accumulator_per_cell_ms=round(t_acc / cells_sub * 1000, 3),
        direct_per_cell_ms=round(t_dir / cells_sub * 1000, 3),
        speedup=round(t_dir / t_acc, 1),
        accumulator_step_core_h=round(t_acc / cells_sub * n_cells_step / 3600, 2),
        direct_step_core_h=round(t_dir / cells_sub * n_cells_step / 3600, 2),
        full_greedy_8_steps_accumulator_core_h=round(
            t_acc / cells_sub * n_cells_step * 8 / 3600, 2),
        agreement=dict(accumulator=sf_acc, direct=sf_dir,
                       abs_diff=abs(sf_acc - sf_dir)),
        memory_XY2_full_gb=round(259 * len(pairs) * nc * nc * 4 / 1e9, 1),
        memory_XX2_full_gb=round(259 * len(models) * nc * nc * 4 / 1e9, 2))
    c = res["cost"]
    print(f"\n[cost] accumulator {c['accumulator_per_cell_ms']:.3f} ms/cell vs "
          f"direct {c['direct_per_cell_ms']:.3f} ms/cell -> {c['speedup']}x")
    print(f"[cost] one greedy step over 259x666: "
          f"{c['accumulator_step_core_h']} core-h (accumulator) vs "
          f"{c['direct_step_core_h']} core-h (direct)")
    print(f"[cost] full k=8 greedy: {c['full_greedy_8_steps_accumulator_core_h']} "
          f"core-h; precompute {c['precompute_full_projection_core_h']} core-h; "
          f"XY2 would be {c['memory_XY2_full_gb']} GB")
    print(f"[cost] accumulator vs direct agreement: "
          f"{c['agreement']['abs_diff']:.2e}")

    ks_ok = [k for k in KS if res[k]["auroc_real_vs_null"] >= 0.95]
    res["verdict"] = dict(
        ks_checked=KS, ks_non_degenerate=ks_ok,
        degenerate=bool(len(ks_ok) < len([k for k in KS if k <= 8])),
        criterion="AUROC(real pairs vs same-model null) >= 0.95 and the "
                  "separation ratio not collapsing toward 1 as k grows",
        note="a drop that is monotone but mild is WEAKER, not degenerate; "
             "D010's INCONCLUSIVE branch needs the statistic to stop "
             "discriminating, not merely to discriminate less well")
    json.dump(res, open(f"{OUT}/degeneracy_pilot.json", "w"), indent=1)
    print(f"\n[S1] non-degenerate at k = {ks_ok}")
    print(f"written: {OUT}/degeneracy_pilot.json")


if __name__ == "__main__":
    main()
