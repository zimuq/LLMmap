"""D008 — shared loading, classification and metrics for the selection comparison.

Everything here operates on FROZEN artifacts (I6): D006's corpus embeddings and
D007's `S_energy_sf_tok200` tensor. Nothing is generated or re-embedded.

The central object is `Dq`: for each of the 259 queries, the matrix of squared
Euclidean distances from every EVAL trace to every REFERENCE trace, for that
query alone.

  reference = build split, 37 models x 75 configs = 2775 traces
  eval      = test split,  37 models x 25 configs =  925 traces

Why per-query matrices (P1/F0). Squared Euclidean distance on a concatenated
k-query trace decomposes additively over the queries:

    d^2(trace, ref) over Q_k  =  sum_{q in Q_k} ||E_q(trace) - E_q(ref)||^2

so a query set's full distance matrix is the SUM of k precomputed matrices.
That is what makes 200 random seeds, a 5-point gamma sweep and a full MMD tensor
affordable instead of subsampled.

Why the per-query normaliser (P1/F4, approved). Under that sum, a query whose
responses embed to larger vectors contributes more to every trace distance --
while selection under I4 treats the k queries symmetrically via MAX. That is
D007/F3 (raw magnitudes are not comparable across queries under unnormalised
embeddings) reappearing on the classifier side of the same pipeline. Primary
distances are therefore divided by `w2[q]`, the mean WITHIN-reference-cloud
squared distance for that query -- the exact analogue of the tensor's own
scale-free convention. Raw is kept as the sensitivity variant; which one
headlines is decided on `val`, before test is touched.
"""
import os
import csv
import json
import itertools

import numpy as np

from d007_lib import load_corpus

CORPUS = os.environ.get("D006_CORPUS", "./data/corpus_v1")
TENSOR = "./results/D007/S_energy_sf_tok200.npy"
TENSOR_META = "./results/D007/tensor_tok200.meta.json"
META_CSV = "./results/D001/model_metadata.csv"
POOL = "./confs/queries/pool_v1.json"
PAPER8 = "./confs/queries/default.json"
OUT = "./results/D008"


# --------------------------------------------------------------- frozen inputs
def load_tensor():
    meta = json.load(open(TENSOR_META))
    S = np.load(TENSOR)
    assert meta["schema_version"] == "cdqd-tensor-v1", meta["schema_version"]
    assert S.shape == (259, 666), S.shape
    return S, meta


def pair_index(models):
    """Pairs in the tensor's column order: itertools.combinations of the sorted
    model list, exactly as d007_build_tensor produced them."""
    return list(itertools.combinations(models, 2))


def near_relative_pairs(models):
    """The 65 structural near-relative pairs (Call 2, approved).

    Same rule D001/D004/D007-S2a used: same `base`, or same `lineage`. Fixed in
    D001 before any tensor existed, so it is NOT defined by the statistic the
    selection optimises -- that circularity is exactly what Call 2 rejected.
    """
    rows = {r["model"]: r for r in csv.DictReader(open(META_CSV))}
    ix = {m: i for i, m in enumerate(models)}
    out = {}
    for i, (a, b) in enumerate(pair_index(models)):
        ra, rb = rows[a], rows[b]
        same_base = bool(ra["base"]) and ra["base"] == rb["base"]
        same_lin = bool(ra["lineage"]) and ra["lineage"] == rb["lineage"]
        if same_base or same_lin:
            out[i] = dict(pair=(a, b), ab=(ix[a], ix[b]),
                          same_base=same_base, same_lineage=same_lin)
    return out


def structural_diagnostics(a, b):
    rows = {r["model"]: r for r in csv.DictReader(open(META_CSV))}
    ra, rb = rows[a], rows[b]
    return dict(
        same_base=bool(ra["base"]) and ra["base"] == rb["base"],
        same_lineage=bool(ra["lineage"]) and ra["lineage"] == rb["lineage"],
        same_org=ra["org"] == rb["org"],
        same_arch=ra["arch"] == rb["arch"],
        same_tokenizer=ra["tok"] == rb["tok"],
        params_b=(ra["params_b"], rb["params_b"]),
        variants=(ra["variant"], rb["variant"]))


# ------------------------------------------------------------- distance layer
def build_dq(eval_pool="test", ref_pool="build"):
    """Return (Dq, w2, y_eval, y_ref, models, cfg_eval).

    Dq   (n_queries, n_eval, n_ref) float32 squared Euclidean distances
    w2   (n_queries,) mean within-reference-cloud squared distance per query
    cfg_eval  config index of each eval trace (the bootstrap unit)
    """
    models, R = load_corpus(pool=ref_pool)
    _, E = load_corpus(pool=eval_pool)
    nq, nc_r, dim = R[models[0]].shape
    nc_e = E[models[0]].shape[1]

    y_ref = np.repeat(np.arange(len(models)), nc_r)
    y_eval = np.repeat(np.arange(len(models)), nc_e)
    cfg_eval = np.tile(np.arange(nc_e), len(models))

    Dq = np.empty((nq, len(models) * nc_e, len(models) * nc_r), np.float32)
    w2 = np.empty(nq, np.float64)
    for q in range(nq):
        A = np.concatenate([E[m][q] for m in models], 0).astype(np.float32)
        B = np.concatenate([R[m][q] for m in models], 0).astype(np.float32)
        d = (np.einsum("ij,ij->i", A, A)[:, None]
             + np.einsum("ij,ij->i", B, B)[None, :]
             - 2.0 * (A @ B.T))
        np.maximum(d, 0, out=d)
        Dq[q] = d
        # within-cloud scale: mean squared distance among a model's own
        # reference traces, averaged over models (the scale-free denominator)
        within = []
        for k in range(len(models)):
            C = B[k * nc_r:(k + 1) * nc_r]
            dd = (np.einsum("ij,ij->i", C, C)[:, None]
                  + np.einsum("ij,ij->i", C, C)[None, :] - 2.0 * (C @ C.T))
            np.maximum(dd, 0, out=dd)
            within.append(dd.sum() / (nc_r * (nc_r - 1)))   # exclude self-pairs
        w2[q] = float(np.mean(within))
    return Dq, w2, y_eval, y_ref, models, cfg_eval


def summed(Dq, w2, queries, normalise=True):
    """Distance matrix of a query SET: the sum of its per-query matrices (F0),
    each divided by its own scale when normalise=True (F4)."""
    out = np.zeros(Dq.shape[1:], np.float32)
    for q in queries:
        if normalise:
            out += Dq[q] / np.float32(w2[q])
        else:
            out += Dq[q]
    return out


# ------------------------------------------------------------- classification
def nn_predict(D, y_ref):
    """1-nearest-neighbour, 37-way (Call 1, approved)."""
    return y_ref[np.argmin(D, axis=1)]


def two_way_correct(D, y_eval, y_ref, a, b):
    """2-way restricted accuracy for one model pair: reference set restricted to
    the two models, over their own eval traces. Returns the boolean correctness
    vector and the eval-trace indices it refers to."""
    rm = np.where((y_ref == a) | (y_ref == b))[0]
    em = np.where((y_eval == a) | (y_eval == b))[0]
    pred = y_ref[rm][np.argmin(D[np.ix_(em, rm)], axis=1)]
    return pred == y_eval[em], em


def condition_stats(D, y_eval, y_ref, cfg_eval, hard_pairs, n_models):
    """All four METHOD.md §8 metrics for one (condition, k), plus the
    CONFIG-LEVEL count structures the bootstrap needs.

    The bootstrap unit is the eval config, not the trace: the 37 traces sharing
    a config are not independent (they share system prompt, temperature, RAG and
    CoT template). Storing per-config correct-counts makes a resample an exact
    matrix-vector product rather than a reclassification.
    """
    ncfg = int(cfg_eval.max()) + 1
    correct = nn_predict(D, y_ref) == y_eval

    cnt_total = np.bincount(cfg_eval[correct], minlength=ncfg).astype(np.float64)
    cnt_model = np.zeros((n_models, ncfg), np.float64)
    np.add.at(cnt_model, (y_eval[correct], cfg_eval[correct]), 1.0)

    keys = sorted(hard_pairs)
    cnt_pair = np.zeros((len(keys), ncfg), np.float64)
    for r, i in enumerate(keys):
        a, b = hard_pairs[i]["ab"]
        ok, em = two_way_correct(D, y_eval, y_ref, a, b)
        cnt_pair[r] = np.bincount(cfg_eval[em][ok], minlength=ncfg)

    per_model = cnt_model.sum(1) / (correct.size / n_models)
    hv = cnt_pair.sum(1) / (2 * ncfg)
    point = dict(
        mean_top1=float(correct.mean()),
        worst_class=float(per_model.min()),
        worst3_class=float(np.sort(per_model)[:3].mean()),
        hard_subset=float(hv.mean()),
        hard_subset_min=float(hv.min()),
        hard_subset_worst_pair=str(hard_pairs[keys[int(np.argmin(hv))]]["pair"]),
        per_model=per_model.round(4).tolist(),
        per_hard_pair=hv.round(4).tolist())
    return point, dict(cnt_total=cnt_total, cnt_model=cnt_model,
                       cnt_pair=cnt_pair, n_models=n_models, ncfg=ncfg)


def boot_draws(ncfg, n_boot=2000, seed=20260908):
    """Multiplicity vectors for a config-level bootstrap. The SAME draws are
    reused across conditions, which makes every comparison paired."""
    rng = np.random.default_rng(seed)
    M = np.zeros((n_boot, ncfg))
    for i in range(n_boot):
        pick = rng.integers(0, ncfg, ncfg)
        M[i] = np.bincount(pick, minlength=ncfg)
    return M


def boot_metrics(st, M):
    """Bootstrap distributions of the headline metrics, computed exactly from
    the stored per-config counts. `M` is (n_boot, ncfg) of config multiplicities.

    Each config contributes exactly one trace per model, so a model's resampled
    accuracy is its correct-count dotted with the multiplicity vector, divided by
    the number of configs drawn. No reclassification is involved.
    """
    tot = M.sum(1)                                      # = ncfg
    mean_top1 = (M @ st["cnt_total"]) / (st["n_models"] * tot)
    per_model = (st["cnt_model"] @ M.T) / tot           # (n_models, n_boot)
    worst = per_model.min(axis=0)
    worst3 = np.sort(per_model, axis=0)[:3].mean(axis=0)
    hard = ((st["cnt_pair"] @ M.T) / (2 * tot)).mean(axis=0)
    return dict(mean_top1=mean_top1, worst_class=worst,
                worst3_class=worst3, hard_subset=hard)


def ci(x):
    return dict(mean=float(np.mean(x)), lo=float(np.percentile(x, 2.5)),
                hi=float(np.percentile(x, 97.5)))


def paired_ci(a, b):
    d = a - b
    return dict(delta=float(np.mean(d)), lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)),
                p_gt_0=float((d > 0).mean()))


def queries_to_reach(curve, target):
    """Smallest k in a nested chain whose metric reaches `target`, else None."""
    for k, v in sorted(curve.items()):
        if v >= target:
            return int(k)
    return None
