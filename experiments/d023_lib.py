"""D023 — shared selection code: CVaR_0.1-of-ADD objective, exact MILP, LP bound,
greedy. Target-free: uses D021's frozen evidence tensor only (S_build-derived).

Objective (fixed in D023): maximise the sum of the m = floor(0.1 * 666) = 66
smallest A_x(p) = sum_q x_q M2_q(p), with sum_q x_q = k. LP form (exact for
integer m):   max  m*t - sum_p u_p
              s.t. u_p >= t - A_x(p),  u_p >= 0,  sum x = k,  x in {0,1}^259.
Optimal value / m = CVaR_0.1 (same definition as LLMmap/joint_statistic.cvar).
"""
import json
import time

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix, hstack, vstack, identity

from LLMmap.joint_statistic import cvar

GAMMA = 0.1
TIME_LIMIT = 900.0


def load_evidence():
    E = np.load("./results/D021/evidence_tok200.npy").astype(np.float64)   # (259, 666)
    meta = json.load(open("./results/D021/evidence_tok200.meta.json"))
    assert meta["schema_version"] == "d021-evidence-v1"
    assert meta["primary"].startswith("evidence_tok200.npy = d64, Sigma_q full")
    assert E.shape == (259, 666)
    return E, meta


def objective(E, qs):
    """CVaR_0.1 over pairs of summed evidence for query set `qs`."""
    return cvar(E[list(qs)].sum(0), GAMMA)


def _model(E, k, extra=None):
    nq, npair = E.shape
    m = max(1, int(np.floor(GAMMA * npair)))
    # variables: x (nq), t (1), u (npair)
    c = np.concatenate([np.zeros(nq), [-float(m)], np.ones(npair)])   # minimise -(m t - sum u)
    # u_p + sum_q E[q,p] x_q - t >= 0
    A1 = hstack([csr_matrix(E.T), csr_matrix(-np.ones((npair, 1))), identity(npair, format="csr")])
    A2 = hstack([csr_matrix(np.ones((1, nq))), csr_matrix((1, 1)), csr_matrix((1, npair))])
    cons = [LinearConstraint(A1, lb=0, ub=np.inf), LinearConstraint(A2, lb=k, ub=k)]
    if extra is not None:
        cons.append(extra)
    lb = np.concatenate([np.zeros(nq), [-np.inf], np.zeros(npair)])
    ub = np.concatenate([np.ones(nq), [np.inf], np.full(npair, np.inf)])
    return c, cons, Bounds(lb, ub), m


def solve(E, k, integer=True, exclude=None, time_limit=TIME_LIMIT, mip_rel_gap=1e-6):
    """Exact MILP (integer=True) or its LP relaxation. `exclude`: a set to cut off
    (no-good cut sum_{q in S} x_q <= k-1), used to test for alternative optima.
    HiGHS via scipy.optimize.milp; deterministic (single run, fixed model)."""
    nq, npair = E.shape
    extra = None
    if exclude is not None:
        row = np.zeros(nq + 1 + npair); row[list(exclude)] = 1
        extra = LinearConstraint(csr_matrix(row), lb=-np.inf, ub=k - 1)
    c, cons, bnds, m = _model(E, k, extra)
    integ = np.concatenate([np.ones(nq), [0], np.zeros(npair)]) if integer else np.zeros(len(c))
    t0 = time.time()
    r = milp(c, constraints=cons, bounds=bnds, integrality=integ,
             options=dict(time_limit=time_limit, mip_rel_gap=mip_rel_gap, disp=False))
    wall = time.time() - t0
    out = dict(k=k, integer=integer, status=int(r.status), message=str(r.message),
               wall_s=round(wall, 2))
    if r.x is not None:
        val = -r.fun / m
        out["cvar"] = float(val)
        if integer:
            sel = sorted(int(i) for i in np.flatnonzero(r.x[:nq] > 0.5))
            out["set"] = sel
            out["cvar_recomputed"] = float(objective(E, sel))
            out["mip_gap"] = float(getattr(r, "mip_gap", np.nan))
            out["mip_dual_bound_cvar"] = (float(-r.mip_dual_bound / m)
                                          if getattr(r, "mip_dual_bound", None) is not None else None)
        else:
            out["n_fractional"] = int(((r.x[:nq] > 1e-6) & (r.x[:nq] < 1 - 1e-6)).sum())
    return out


def greedy(E, K=8):
    """Nested greedy on the same objective; ties -> lowest query id."""
    sel, trace = [], []
    for k in range(1, K + 1):
        best, bq = -np.inf, None
        for q in range(E.shape[0]):
            if q in sel:
                continue
            v = objective(E, sel + [q])
            if v > best + 1e-12:
                best, bq = v, q
        sel.append(bq)
        trace.append(dict(k=k, query=bq, cvar=float(best)))
    return sel, trace
