"""Greedy query-set selection under a set-level statistic (D010).

Structurally parallel to `LLMmap/greedy_cover.py`, with one deliberate and
important difference, flagged in D010/P1 as F2:

    `GreedyCover`'s objective is MONOTONE in k. Under I4, cov(p) = max_q S[q][p]
    can only rise as queries are added, so its k=8 set is always its best.

    This objective is NOT. Each added query adds 1024 dimensions to the joint
    space, and the statistic's value can fall. The search therefore records the
    per-step objective and reports where it PEAKS, rather than assuming the
    longest chain is the best one.

The chain is still built nested to k=8 so it is directly comparable with D008's
and D009's per-k numbers, but "the selected set" and "the k=8 set" are not
necessarily the same thing here, and the caller is given both.
"""
import numpy as np

from .joint_statistic import energy_from_acc, mmd_from_acc, cvar


def _stat_all_pairs(acc_xy, acc_xxm, ia, ib, add_xy, add_xxm, kind, sigma2=None):
    """Statistic for every pair, for one candidate query added to the running
    accumulators. Vectorised over pairs; `xx`/`yy` are computed once per MODEL
    (37) rather than once per pair (666), since a cloud's within-distances do
    not depend on which partner it is being compared to."""
    xy2 = acc_xy + add_xy                      # (n_pairs, nc, nc)
    xxm2 = acc_xxm + add_xxm                   # (n_models, nc, nc)
    if kind == "energy":
        xy = np.sqrt(xy2).mean(axis=(1, 2))
        xm = np.sqrt(xxm2).mean(axis=(1, 2))
        xx, yy = xm[ia], xm[ib]
        raw = 2.0 * xy - xx - yy
        within = (xx + yy) / 2.0
        return np.where(within > 0, raw / np.maximum(within, 1e-12), np.nan)
    kxy = np.exp(-xy2 / sigma2).mean(axis=(1, 2))
    km = np.exp(-xxm2 / sigma2).mean(axis=(1, 2))
    return km[ia] + km[ib] - 2.0 * kxy


def joint_greedy(XY2, XX2, ia, ib, k, gamma=0.1, kind="energy", verbose=True):
    """Nested chain of `k` queries maximising CVaR_gamma of the SET-level
    statistic over pairs.

    XY2 (nq, n_pairs, nc, nc)   per-query squared distances, cloud a vs cloud b
    XX2 (nq, n_models, nc, nc)  per-query squared distances, cloud v vs itself
    ia, ib                      model index of each pair's two members

    Returns (selected, trace). `trace[i]` records the objective after i+1
    queries, so the caller can find where it peaks (F2).
    """
    nq, n_pairs, nc, _ = XY2.shape
    n_models = XX2.shape[1]
    acc_xy = np.zeros((n_pairs, nc, nc), np.float32)
    acc_xxm = np.zeros((n_models, nc, nc), np.float32)
    sel, trace, remaining = [], [], np.ones(nq, bool)

    for step in range(k):
        # per-step global bandwidth for MMD: every candidate at this step has
        # the same set size, so one sigma^2 keeps them comparable. See
        # joint_statistic.mmd_from_acc on why it cannot be fixed across steps.
        sigma2 = None
        if kind == "mmd":
            samp = np.random.default_rng(step).choice(n_pairs,
                                                      min(64, n_pairs), False)
            probe = acc_xy[samp] + XY2[int(np.argmax(remaining)), samp]
            sigma2 = float(np.median(probe[probe > 0])) or 1.0

        best_obj, best_q, best_stats = -np.inf, -1, None
        for q in range(nq):
            if not remaining[q]:
                continue
            s = _stat_all_pairs(acc_xy, acc_xxm, ia, ib, XY2[q], XX2[q],
                                kind, sigma2)
            o = cvar(s, gamma)
            if o > best_obj:
                best_obj, best_q, best_stats = o, q, s
        sel.append(best_q)
        remaining[best_q] = False
        acc_xy += XY2[best_q]
        acc_xxm += XX2[best_q]
        trace.append(dict(k=len(sel), query=int(best_q),
                          objective_cvar=float(best_obj),
                          mean_stat=float(np.nanmean(best_stats)),
                          min_stat=float(np.nanmin(best_stats)),
                          sigma2=sigma2))
        if verbose:
            print(f"    k={len(sel)}: q={best_q}  CVaR {best_obj:.5f}  "
                  f"mean {np.nanmean(best_stats):.5f}", flush=True)
    return sel, trace


def peak_k(trace):
    """The `k` at which the objective actually peaks. Reported as a headline
    number per the D010 review's explicit request -- under a non-monotone
    objective the longest chain is not necessarily the best set."""
    o = [t["objective_cvar"] for t in trace]
    return int(np.argmax(o)) + 1, float(max(o))
