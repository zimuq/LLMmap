"""Set-level (joint) two-sample statistics over concatenated query responses.

`GreedyCover` scores each query independently and aggregates with a MAX (I4), so
it cannot see whether two queries work *together*. These statistics ask that
question directly: form one point cloud per model from the CONCATENATED
responses to a whole query set,

    P(Q, v) = { [ E(o_{q1,s}) ; ... ; E(o_{qk,s}) ] : s in S_build }   (n_cfg, k*1024)

and take a two-sample statistic between `P(Q,v)` and `P(Q,v')`. Nothing is
aggregated from per-query scores, so this does not contradict I4 -- it answers a
different question (D010's Constraints say so explicitly).

THE DECOMPOSITION THAT MAKES AN EXACT GREEDY AFFORDABLE. Squared Euclidean
distance in the concatenated space is additive over the queries:

    || concat_a - concat_b ||^2  =  sum_{q in Q} || e_q(a) - e_q(b) ||^2

Energy distance needs the un-squared distance, so the sum must be taken BEFORE
the square root -- but that is still an accumulator, and adding a candidate query
costs one add, one sqrt and one mean. Measured at 0.051 ms/cell against 26.2 ms
for naive recomputation, a 519x speed-up, which is why D010/Call 2's "propose a
cheaper approximation" was answered by not needing one. RBF-MMD decomposes too,
as a product: exp(-sum_q d2_q / s2) = prod_q exp(-d2_q / s2).

SCALE-FREE IS NOT OPTIONAL HERE. Raw energy distance grows with vector norm, and
here the norm grows with `k` because the vectors are concatenations -- so raw
values are not comparable across set sizes, which is exactly what a greedy over
growing `k` would be comparing. Dividing by the pooled within-cloud mean distance
removes it (the same fix D007/F3 established for the per-query tensor).
"""
import numpy as np


def energy_from_acc(acc_xy2, acc_xx2, acc_yy2):
    """Scale-free energy distance from ACCUMULATED squared distances.

    acc_*2 are (n, n) arrays of summed squared distances over the selected
    queries. Returns (raw, scale_free).
    """
    xy = np.sqrt(acc_xy2).mean()
    xx = np.sqrt(acc_xx2).mean()
    yy = np.sqrt(acc_yy2).mean()
    raw = 2.0 * xy - xx - yy
    within = (xx + yy) / 2.0
    return float(raw), float(raw / within) if within > 0 else float("nan")


def energy_direct(A, B):
    """The same statistic from explicit concatenated point clouds.

    Only used to verify the accumulator path. The accumulator is an
    optimisation, so it is checked against this rather than trusted.
    """
    from scipy.spatial.distance import cdist
    xy = cdist(A, B).mean()
    xx = cdist(A, A).mean()
    yy = cdist(B, B).mean()
    raw = 2.0 * xy - xx - yy
    within = (xx + yy) / 2.0
    return float(raw), float(raw / within) if within > 0 else float("nan")


def mmd_from_acc(acc_xy2, acc_xx2, acc_yy2, sigma2):
    """RBF-MMD^2 from accumulated squared distances, given a bandwidth.

    `sigma2` must be supplied by the caller and held constant across every
    candidate being compared. It CANNOT be held constant across different set
    sizes: the concatenated space's scale grows with `k`, so a bandwidth chosen
    at k=1 saturates at k=8. A greedy only ever compares candidates at a FIXED
    set size within one step, so a per-step global bandwidth is well defined and
    sufficient -- cross-`k` MMD values, by contrast, are not comparable and must
    not be plotted as a curve.
    """
    kxy = np.exp(-acc_xy2 / sigma2).mean()
    kxx = np.exp(-acc_xx2 / sigma2).mean()
    kyy = np.exp(-acc_yy2 / sigma2).mean()
    return float(kxx + kyy - 2.0 * kxy)


def cvar(x, gamma):
    """Mean of the worst (lowest) `gamma` fraction -- byte-identical to D004's
    `cvar_low`, D007's `cvar` and `LLMmap/greedy_cover.py`'s, in float64."""
    x = np.asarray(x, dtype=np.float64)
    k = max(1, int(np.floor(gamma * x.size)))
    return float(np.mean(np.sort(x)[:k]))
