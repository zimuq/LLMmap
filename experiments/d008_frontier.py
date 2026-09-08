"""D008 / S7 — the identifiability frontier (T2.4), a standalone deliverable.

`METHOD.md §7` lists this as a deliverable in its own right: the model pairs that
**no query in the whole pool** separates -- not the ones the selected `k` misses.
A genuine negative result that draws the method's boundary honestly.

Call 3 (approved) — three tiers instead of one percentile, because a percentile
threshold fixes the frontier's size by construction and can therefore never come
back empty:

  T1  statistically indistinguishable -- pool coverage below the 95th percentile
      of the SELF-PAIR NULL (one model against itself, its build configs split in
      half, same query). Claim: "no query separates this pair beyond what one
      model produces against itself."
  T2  operationally unidentifiable (PRIMARY) -- the pair's best query, selected on
      the build tensor, still gives < 75% 2-way accuracy on test (chance 50%).
      Claim: "no query in the pool lets us tell these two apart in practice."
  T3  relative -- the 5th percentile of pool coverage. Reported, and explicitly
      labelled as definitionally non-empty.

P1 stated in advance that T1's frontier will almost certainly be EMPTY (the null
value of scale-free energy is ~2/n ~ 0.03-0.05 while the observed pool-coverage
minimum is 0.289), and that emptiness is the finding rather than a badly chosen
threshold. Recorded here so the result cannot be reinterpreted after the fact.

Usage:  PYTHONPATH=.:experiments python experiments/d008_frontier.py
"""
import os
import json
import itertools

import numpy as np
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed

from d008_lib import (build_dq, two_way_correct, structural_diagnostics,
                      near_relative_pairs, load_tensor, OUT)
from d007_build_tensor import prepare_cubes, SPLIT_SEED

JOBS = int(os.environ.get("D008_JOBS", "68"))
T2_ACC = 0.75
T3_PCT = 5.0
SMOKE = bool(os.environ.get("D008_SMOKE"))


def self_null_worker(cache_dir, m, n_half, seed):
    """Scale-free energy of one model against ITSELF, configs split in half.

    This is the distribution of the statistic when there is genuinely no
    difference -- the only principled anchor for "cannot be separated".
    """
    X = np.load(os.path.join(cache_dir, m.replace("/", "__") + ".npy"),
                mmap_mode="r")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[1])
    ia, ib = np.sort(perm[:n_half]), np.sort(perm[n_half:2 * n_half])
    out = np.empty(X.shape[0], np.float32)
    for q in range(X.shape[0]):
        A = np.asarray(X[q][ia], np.float32); B = np.asarray(X[q][ib], np.float32)
        xy = cdist(A, B).mean(); xx = cdist(A, A).mean(); yy = cdist(B, B).mean()
        within = (xx + yy) / 2.0
        out[q] = (2 * xy - xx - yy) / within if within > 0 else np.nan
    return out


def main():
    S, meta = load_tensor()
    models = meta["models"]
    pairs = list(itertools.combinations(models, 2))
    hard = near_relative_pairs(models)
    cov = S.max(axis=0)          # I4: MAX over the WHOLE pool, not the selected k
    if SMOKE:
        pairs, cov, S = pairs[:20], cov[:20], S[:, :20]
    _, cache_dir = prepare_cubes(200)

    # ---------------- T1: the self-pair null
    null = {}
    for n_half, tag in (((37, "n37"), (25, "n25")) if not SMOKE else ((25, "n25"),)):
        r = Parallel(n_jobs=min(JOBS, len(models)), verbose=0)(
            delayed(self_null_worker)(cache_dir, m, n_half, SPLIT_SEED + i)
            for i, m in enumerate(models))
        v = np.concatenate(r)
        null[tag] = dict(n_half=n_half, n_cells=int(v.size),
                         mean=float(np.mean(v)), p50=float(np.percentile(v, 50)),
                         p95=float(np.percentile(v, 95)), max=float(v.max()))
        print(f"[T1] self-pair null at n={n_half}: mean {null[tag]['mean']:.4f}, "
              f"p95 {null[tag]['p95']:.4f}, max {null[tag]['max']:.4f} "
              f"({v.size:,} cells)", flush=True)
    # conservative: the larger p95 of the two measured sizes. Cells are 75-vs-75
    # and the null shrinks with n, so this over-states the threshold on purpose.
    theta1 = max(null[t]["p95"] for t in null)

    # ---------------- T2: operational, on test
    Dq, w2, y_ev, y_rf, _, cfg_ev = build_dq(eval_pool="test")
    qstar = S.argmax(axis=0)
    acc_star = np.empty(len(pairs)); acc_best = np.empty(len(pairs))
    for j, (a, b) in enumerate(pairs):
        ia, ib = models.index(a), models.index(b)
        ok, _ = two_way_correct(Dq[qstar[j]], y_ev, y_rf, ia, ib)
        acc_star[j] = ok.mean()
        best = 0.0
        for q in range(20 if SMOKE else S.shape[0]):
            ok, _ = two_way_correct(Dq[q], y_ev, y_rf, ia, ib)
            best = max(best, ok.mean())
        acc_best[j] = best
        if (j + 1) % 100 == 0:
            print(f"  [T2] {j+1}/{len(pairs)} pairs", flush=True)
    theta3 = float(np.percentile(cov, T3_PCT))

    def block(idx, extra=None):
        out = []
        for j in sorted(idx, key=lambda j: cov[j]):
            a, b = pairs[j]
            d = dict(pair=f"{a} | {b}", pool_coverage=round(float(cov[j]), 4),
                     best_query_index=int(qstar[j]),
                     two_way_acc_best_query=round(float(acc_star[j]), 4),
                     two_way_acc_oracle_over_pool=round(float(acc_best[j]), 4),
                     near_relative=j in hard)
            d.update(structural_diagnostics(a, b))
            out.append(d)
        return out

    t1_idx = np.where(cov < theta1)[0]
    t2_idx = np.where(acc_star < T2_ACC)[0]
    t3_idx = np.where(cov <= theta3)[0]

    out = dict(
        schema="d008-frontier-v1", n_pairs=len(pairs),
        pool_coverage=dict(
            min=float(cov.min()), p1=float(np.percentile(cov, 1)),
            p5=float(np.percentile(cov, 5)), p50=float(np.percentile(cov, 50)),
            max=float(cov.max())),
        T1=dict(theta=theta1, null=null, n_pairs=int(t1_idx.size),
                pairs=block(t1_idx),
                predicted_empty=True,
                claim="no query in the pool separates this pair beyond what one "
                      "model produces against itself"),
        T2=dict(threshold_accuracy=T2_ACC, n_pairs=int(t2_idx.size),
                pairs=block(t2_idx), primary=True,
                claim="the pair's best build-selected query still gives < 75% "
                      "2-way accuracy on test (chance 50%)",
                oracle_note="two_way_acc_oracle_over_pool is the max over all "
                            "259 queries of TEST accuracy -- selected on the "
                            "evaluation split, so it is an optimistic bound, "
                            "reported only to show how much of the gap is "
                            "selection error vs a real limit"),
        T3=dict(theta=theta3, percentile=T3_PCT, n_pairs=int(t3_idx.size),
                pairs=block(t3_idx),
                caveat="definitionally non-empty -- a percentile cannot report "
                       "the absence of a frontier"),
        hardest_10=block(np.argsort(cov)[:10]))
    json.dump(out, open(f"{OUT}/identifiability_frontier.json", "w"), indent=1)

    print(f"\n===== S7: identifiability frontier =====")
    print(f"  pool coverage: min {cov.min():.4f}  p5 {theta3:.4f}  "
          f"median {np.median(cov):.4f}")
    print(f"  T1 (theta={theta1:.4f}, self-pair null): {t1_idx.size} pairs "
          f"{'(EMPTY, as pre-registered)' if t1_idx.size == 0 else ''}")
    print(f"  T2 (best-query 2-way acc < {T2_ACC}): {t2_idx.size} pairs")
    for d in out["T2"]["pairs"][:10]:
        print(f"     {d['pair']:70s} {d['two_way_acc_best_query']:.3f} "
              f"(oracle {d['two_way_acc_oracle_over_pool']:.3f}, "
              f"near-relative={d['near_relative']})")
    print(f"  T3 (5th pct): {t3_idx.size} pairs, "
          f"{sum(1 for d in out['T3']['pairs'] if d['near_relative'])} of them "
          f"near-relative")
    print(f"written: {OUT}/identifiability_frontier.json")


if __name__ == "__main__":
    main()
