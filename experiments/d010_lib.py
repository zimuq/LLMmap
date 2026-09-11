"""D010 — a set-level (joint) two-sample statistic over the frozen embeddings.

The question this exists for: `Sep(q,v,v')` is computed per query and I4
aggregates it with a MAX, so `GreedyCover` can never see whether two queries
work *together*. The joint statistic asks the other question directly — form one
point cloud per model from the CONCATENATED responses to a whole query set:

    P(Q_k, v) = { [ E(o_{q1,s}) ; ... ; E(o_{qk,s}) ] : s in S_build }   (75, k*1024)

and take a two-sample statistic between `P(Q_k,v)` and `P(Q_k,v')`. Nothing is
aggregated from per-query scores, so this does not contradict I4 — it answers a
different question (D010's Constraints say so explicitly).

THE ONE FACT THAT MAKES THE GREEDY SEARCH CHEAP. Squared Euclidean distance in
the concatenated space decomposes additively over the queries:

    || concat_a - concat_b ||^2 = sum_{q in Q}  || e_q(a) - e_q(b) ||^2

Energy distance needs the un-squared distances, so the sum has to be taken
BEFORE the square root -- but that is still an accumulator: keep running
per-pair sums of squared distances and adding a candidate query costs one add,
one sqrt and one mean. The same decomposition D008/F0 used for the classifier,
applied here to the selection statistic. It makes the exact greedy affordable,
so D010/Call 2's "propose a cheaper approximation if the naive version is too
slow" is answered by not needing one.
"""
import os
import json
import itertools

import numpy as np

from d007_lib import load_corpus

OUT = "./results/D010"
D008_OUT = "./results/D008"


def pair_index(models):
    return list(itertools.combinations(models, 2))


def per_query_sqdists(X, models, queries=None):
    """Precompute, per query, the squared-distance blocks the joint statistic
    needs: within-model (one per model) and between-model (one per pair).

    Returns
      XX2  (nq, n_models, nc, nc)   within-cloud, model v against itself
      XY2  (nq, n_pairs,  nc, nc)   between-cloud, model a against model b
    """
    qs = range(X[models[0]].shape[0]) if queries is None else queries
    nc = X[models[0]].shape[1]
    pairs = pair_index(models)
    XX2 = np.empty((len(list(qs)), len(models), nc, nc), np.float32)
    XY2 = np.empty((len(list(qs)), len(pairs), nc, nc), np.float32)
    for qi, q in enumerate(qs):
        C = [np.asarray(X[m][q], np.float32) for m in models]
        n2 = [np.einsum("ij,ij->i", c, c) for c in C]
        for mi in range(len(models)):
            d = n2[mi][:, None] + n2[mi][None, :] - 2.0 * (C[mi] @ C[mi].T)
            XX2[qi, mi] = np.maximum(d, 0)
        for pi, (a, b) in enumerate(pairs):
            ia, ib = models.index(a), models.index(b)
            d = n2[ia][:, None] + n2[ib][None, :] - 2.0 * (C[ia] @ C[ib].T)
            XY2[qi, pi] = np.maximum(d, 0)
    return XX2, XY2


def joint_energy_from_acc(xy2, xx2, yy2):
    """Scale-free energy distance from accumulated SQUARED distances.

    Scale-free (raw / mean within-cloud distance) for the same reason D007/F3
    established: raw energy scales with vector norm, and here the norm grows
    with `k` because the vectors are concatenations -- so raw values are not
    comparable across set sizes, which is precisely what a greedy over growing
    `k` would compare.
    """
    xy = np.sqrt(xy2).mean()
    xx = np.sqrt(xx2).mean()
    yy = np.sqrt(yy2).mean()
    raw = 2 * xy - xx - yy
    within = (xx + yy) / 2.0
    return float(raw), float(raw / within) if within > 0 else float("nan")


def joint_energy_direct(A, B):
    """Same statistic computed the naive way, from explicit concatenated point
    clouds. Used to verify the accumulator path, not for the search."""
    from scipy.spatial.distance import cdist
    xy = cdist(A, B).mean(); xx = cdist(A, A).mean(); yy = cdist(B, B).mean()
    raw = 2 * xy - xx - yy
    within = (xx + yy) / 2.0
    return float(raw), float(raw / within) if within > 0 else float("nan")


def concat_cloud(X, m, queries):
    """(nc, k*1024) joint vectors: one per config, not one per query."""
    return np.concatenate([np.asarray(X[m][q], np.float32) for q in queries], 1)


def cvar(x, gamma):
    x = np.asarray(x, dtype=np.float64)
    k = max(1, int(np.floor(gamma * x.size)))
    return float(np.mean(np.sort(x)[:k]))
