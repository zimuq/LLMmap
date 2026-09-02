"""
D004 / P1 — fair-instrument re-measurement of the hard-tail premise.

Pure re-analysis of D001's cached trace embeddings (results/D001/f_test.npy,
3536 x 384 = 68 individual traces per model x 52 models, never averaged).
No GPU, no generation, no re-extraction.

Statistics:
  M1a  A3 default -- 5-fold CV AUC of an L2-regularized linear probe,
       point cloud vs point cloud (I3-compliant). BOUNDED; kept for direct
       comparability with D001's AUC-of-Delta.
  M1b  energy distance between clouds, raw and scale-free. UNBOUNDED, so it
       cannot saturate -- this is the statistic that actually carries the
       answer, because a CVaR/mean ratio on a censored variable is
       uninterpretable (D001's failure).
  M2   52-way leave-one-out nearest neighbour over point clouds -> 52x52
       confusion -> per-pair confusion rate. Secondary: 52-class probe.
  M3   resolution audit for every statistic -- treated as a GATE on
       interpreting any ratio, not a footnote.
  M4   Llama-3-70B <-> Smaug-70B percentile under each statistic.
"""
import os
import csv
import json
import argparse
import itertools

import numpy as np
from scipy.spatial.distance import squareform, pdist
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

from d001_model_metadata import MODELS, pair_labels, le_14b

GAMMA = 0.10
SEED = 0
TARGET = ("meta-llama/Meta-Llama-3-70B-Instruct",
          "abacusai/Smaug-Llama-3-70B-Instruct")


def cvar_low(x, gamma=GAMMA):
    k = max(1, int(np.floor(gamma * x.size)))
    return float(np.sort(x)[:k].mean())


def shape_stats(x, name):
    x = np.asarray(x, dtype=float)
    uniq = np.unique(x)
    return {
        "statistic": name,
        "mean": float(x.mean()), "min": float(x.min()), "max": float(x.max()),
        "p05": float(np.percentile(x, 5)), "p10": float(np.percentile(x, 10)),
        "p25": float(np.percentile(x, 25)), "p50": float(np.percentile(x, 50)),
        "cvar_0.1": cvar_low(x),
        "M6_cvar_over_mean": cvar_low(x) / float(x.mean()) if x.mean() else None,
        # --- M3 resolution audit
        "n_distinct_values": int(uniq.size),
        "frac_at_max": float((x >= x.max() - 1e-12).mean()),
        "frac_at_min": float((x <= x.min() + 1e-12).mean()),
        "smallest_gap": float(np.diff(uniq).min()) if uniq.size > 1 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d001", default="./results/D001")
    ap.add_argument("--out", default="./results/D004")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    f = np.load(f"{args.d001}/f_test.npy")
    y = np.load(f"{args.d001}/y_test.npy")
    labels = json.load(open(f"{args.d001}/labels.json"))["label_order"]
    n = len(labels)
    assert f.shape[0] == y.shape[0] and f.shape[1] == 384 and n == 52
    idx = {c: np.where(y == c)[0] for c in range(n)}
    print(f"[M5] cache intact: f_test={f.shape}, "
          f"{min(len(v) for v in idx.values())}-{max(len(v) for v in idx.values())} traces/model",
          flush=True)

    # one full pairwise distance matrix, reused by M1b and M2
    print("[.] full 3536x3536 distance matrix...", flush=True)
    Dfull = squareform(pdist(f, metric="euclidean"))

    # ---------------- M2: 52-way LOO nearest neighbour over point clouds
    print("[M2] leave-one-out 52-way NN...", flush=True)
    Dnn = Dfull.copy()
    np.fill_diagonal(Dnn, np.inf)
    nn_pred = y[Dnn.argmin(axis=1)]
    top1 = float((nn_pred == y).mean())
    C = np.zeros((n, n), dtype=int)
    for t, p in zip(y, nn_pred):
        C[t, p] += 1
    print(f"[M2] LOO 52-way top-1 = {top1:.3%}", flush=True)

    # secondary: 52-class multiclass probe
    probe = make_pipeline(StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=2000))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    mc_acc = float(cross_val_score(probe, f, y, cv=skf, n_jobs=-1).mean())
    print(f"[M2] 52-class probe CV top-1 = {mc_acc:.3%}", flush=True)

    # ---------------- per-pair loop
    rows = []
    pairs = list(itertools.combinations(range(n), 2))
    for k, (A, B) in enumerate(pairs):
        iA, iB = idx[A], idx[B]
        X = np.vstack([f[iA], f[iB]])
        yy = np.r_[np.zeros(len(iA)), np.ones(len(iB))]

        # --- M1a: 5-fold CV AUC of a regularized linear probe (C fixed a
        # priori -- tuning per pair would leak).
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1.0, max_iter=2000))
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        auc = float(cross_val_score(clf, X, yy, cv=cv, scoring="roc_auc").mean())

        # --- M1b: energy distance (unbounded)
        dab = Dfull[np.ix_(iA, iB)].mean()
        daa = Dfull[np.ix_(iA, iA)]
        dbb = Dfull[np.ix_(iB, iB)]
        na, nb = len(iA), len(iB)
        daa_m = daa.sum() / (na * (na - 1))      # exclude self-pairs
        dbb_m = dbb.sum() / (nb * (nb - 1))
        energy = 2 * dab - daa_m - dbb_m
        within = 0.5 * (daa_m + dbb_m)
        rows.append(dict(
            a=labels[A], b=labels[B],
            m1a_probe_cv_auc=auc,
            m1b_energy=float(energy),
            m1b_energy_scalefree=float(energy / within),
            m2_confusion_rate=float((C[A, B] + C[B, A]) / (na + nb)),
            m2_separability=float(1.0 - (C[A, B] + C[B, A]) / (na + nb)),
            **pair_labels(labels[A], labels[B]),
        ))
        if (k + 1) % 200 == 0:
            print(f"    {k+1}/{len(pairs)} pairs", flush=True)

    # ---------------- distributions + M3 audit
    out = {"M5_data_provenance": {
        "reused_d001_cache": True,
        "path": f"{args.d001}/f_test.npy",
        "filesystem": "$WORK (Lustre) -- no purge policy; D004's M5 note assumed "
                      "$SCRATCH, which is where model WEIGHTS live, not this cache",
        "re_extraction_needed": False},
        "M2_loo_52way_top1": top1,
        "M2_multiclass_probe_cv_top1": mc_acc}

    stats = {}
    for key, name in [("m1a_probe_cv_auc", "M1a probe CV AUC (bounded)"),
                      ("m1b_energy_scalefree", "M1b energy distance, scale-free (UNBOUNDED)"),
                      ("m1b_energy", "M1b energy distance, raw (UNBOUNDED)"),
                      ("m2_separability", "M2 52-way separability (1 - confusion)")]:
        v = np.array([r[key] for r in rows])
        stats[key] = shape_stats(v, name)
        rel = np.array([r["structurally_related"] for r in rows], dtype=bool)
        stats[key]["mean_related"] = float(v[rel].mean())
        stats[key]["mean_unrelated"] = float(v[~rel].mean())
    out["statistics"] = stats

    # ---------------- M4
    m4 = {}
    for r in rows:
        if {r["a"], r["b"]} == set(TARGET):
            for key in ("m1a_probe_cv_auc", "m1b_energy_scalefree", "m2_separability"):
                v = np.array([q[key] for q in rows])
                m4[key] = {"value": r[key],
                           "percentile_rank": float((v < r[key]).mean() * 100)}
    m4["d001_open_set_nearest_template_percentile"] = 0.754
    out["M4_llama70b_vs_smaug"] = m4

    with open(f"{args.out}/summary.json", "w") as fh:
        json.dump(out, fh, indent=2)
    cols = list(rows[0].keys())
    for fn, key in (("pairwise_m1.csv", "m1a_probe_cv_auc"),
                    ("pairwise_m2.csv", "m2_separability")):
        with open(f"{args.out}/{fn}", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in sorted(rows, key=lambda r: r[key]):
                w.writerow(r)
    with open(f"{args.out}/resolution_audit.json", "w") as fh:
        json.dump({k: {kk: vv for kk, vv in v.items()
                       if kk in ("statistic", "n_distinct_values", "frac_at_max",
                                 "frac_at_min", "smallest_gap", "M6_cvar_over_mean")}
                   for k, v in stats.items()}, fh, indent=2)

    print(json.dumps(out, indent=2)[:3500], flush=True)


if __name__ == "__main__":
    main()
