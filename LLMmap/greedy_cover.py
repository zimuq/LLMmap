"""GreedyCover — CVaR-weighted coverage maximisation over a query pool.

`METHOD.md §5.3`. Given a separability tensor `S[q][p]` (queries x model pairs),
select `k` queries greedily so as to maximise `CVaR_gamma` of per-pair coverage.

Two things this module exists to get right:

  * **Coverage of a query SET is a MAX over its queries** (invariant I4). One
    query that separates a pair suffices. Under sum/mean, a query that is weak on
    average but *uniquely* covers a hard pair is discarded -- the exact behaviour
    CDQD is fixing. `agg="sum"` is provided ONLY as D008/S2's isolation variant,
    to measure whether the MAX-not-sum choice itself matters. It is not an
    alternative default.

  * **`gamma = 1.0` must reduce EXACTLY to mean-greedy** (`METHOD.md §6.5`): the
    mean-based baseline is a special case of this method, not a competing one.
    `mean_greedy` below is a deliberately separate, minimal implementation of
    that baseline -- not this function with a flag -- so that the reduction can
    be verified numerically rather than asserted by inspection (D008/S1).

CVaR convention is byte-identical to D004's `cvar_low` and D007's `cvar`:
the mean of the lowest `max(1, floor(gamma * n))` values. That keeps every
CVaR number in this project comparable across D's.
"""
import numpy as np


def cvar(x, gamma):
    """Mean of the worst (lowest) `gamma` fraction. `gamma=1.0` is the mean.

    Accumulates in float64 deliberately. This function SORTS before averaging,
    so on a float32 input it disagrees with `np.mean` of the same values in the
    last ulp -- summation order, not logic. That is enough to make an objective
    comparison between two candidate queries order-dependent at a tie, and it is
    what `verify_reduction`'s first assertion caught. `greedy_cover` already
    casts its tensor; casting here too makes the convention order-stable
    wherever CVaR is computed.
    """
    x = np.asarray(x, dtype=np.float64)
    k = max(1, int(np.floor(gamma * x.size)))
    return float(np.mean(np.sort(x)[:k]))


def _coverage(S, sel, agg):
    if not sel:
        return np.zeros(S.shape[1], dtype=np.float64)
    sub = S[sel]
    return sub.max(axis=0) if agg == "max" else sub.sum(axis=0)


def greedy_cover(S, k, gamma=0.1, agg="max", return_trace=False):
    """Nested chain of `k` queries maximising CVaR_gamma of per-pair coverage.

    S     (n_queries, n_pairs) separability tensor
    k     chain length; the chain is nested, so Q^(1) subset ... subset Q^(k)
    gamma CVaR level; 1.0 == mean objective
    agg   "max" (I4, the real CDQD condition) or "sum" (isolation variant only)

    Returns the list of selected query indices, in selection order. Ties are
    broken by lowest query index, deterministically; with continuous statistics
    they should not occur, and `return_trace` reports whether any did.
    """
    if agg not in ("max", "sum"):
        raise ValueError(f"agg must be 'max' or 'sum', got {agg!r}")
    S = np.asarray(S, dtype=np.float64)
    nq, npairs = S.shape
    if k > nq:
        raise ValueError(f"k={k} exceeds pool size {nq}")

    sel, trace = [], []
    cov = np.zeros(npairs)
    remaining = np.ones(nq, dtype=bool)
    for _ in range(k):
        if agg == "max":
            cand = np.maximum(cov[None, :], S)          # (nq, npairs)
        else:
            cand = cov[None, :] + S
        # CVaR of every candidate's resulting coverage, vectorised
        m = max(1, int(np.floor(gamma * npairs)))
        part = np.partition(cand, m - 1, axis=1)[:, :m]
        obj = part.mean(axis=1)
        obj[~remaining] = -np.inf
        best = float(obj.max())
        ties = int(np.sum(obj >= best - 1e-12))
        q = int(np.argmax(obj))                          # lowest index wins ties
        sel.append(q)
        remaining[q] = False
        cov = np.maximum(cov, S[q]) if agg == "max" else cov + S[q]
        trace.append(dict(k=len(sel), query=q, objective=best, n_ties=ties,
                          cvar_gamma=cvar(cov, gamma), mean_cov=float(cov.mean()),
                          min_cov=float(cov.min())))
    return (sel, trace) if return_trace else sel


def mean_greedy(S, k, agg="max"):
    """Plain mean-coverage greedy — written independently of `greedy_cover`.

    This is the `METHOD.md §6.5` baseline: *our reconstruction of mean-based
    selection*, NOT a reproduction of LLMmap's algorithm (the paper never
    instantiates one). Its only purpose beyond being that baseline is to let
    D008/S1 check numerically that `greedy_cover(..., gamma=1.0)` returns the
    identical chain.
    """
    S = np.asarray(S, dtype=np.float64)
    sel = []
    cov = np.zeros(S.shape[1])
    for _ in range(k):
        best_val, best_q = -np.inf, -1
        for q in range(S.shape[0]):
            if q in sel:
                continue
            c = np.maximum(cov, S[q]) if agg == "max" else cov + S[q]
            v = float(c.mean())
            if v > best_val + 1e-15:
                best_val, best_q = v, q
        sel.append(best_q)
        cov = np.maximum(cov, S[best_q]) if agg == "max" else cov + S[best_q]
    return sel


def verify_reduction(S, k=8, agg="max"):
    """D008/S1's required correctness check. Raises if the reduction fails."""
    x = np.asarray(S)[:, :50].ravel()[:200]
    # both sides in float64: the tensor is stored float32, and comparing a
    # float64 sorted mean against a float32 mean fails at ~1e-7 for reasons that
    # have nothing to do with the reduction being tested
    assert abs(cvar(x, 1.0) - float(np.mean(x, dtype=np.float64))) < 1e-12, \
        "CVaR(.,1.0) != mean"
    a = greedy_cover(S, k, gamma=1.0, agg=agg)
    b = mean_greedy(S, k, agg=agg)
    assert a == b, f"gamma=1.0 chain {a} != independent mean-greedy chain {b}"
    cov = np.zeros(S.shape[1])
    prev = -np.inf
    for q in a:                                   # coverage monotone in k
        cov = np.maximum(cov, S[q]) if agg == "max" else cov + S[q]
        cur = float(cov.mean())
        assert cur >= prev - 1e-12, "coverage decreased with k"
        prev = cur
    assert len(set(a)) == len(a), "duplicate query selected"
    return dict(gamma1_chain=a, mean_greedy_chain=b, identical=True, agg=agg)
