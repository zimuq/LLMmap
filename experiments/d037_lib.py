"""D037 — the v2 selection grid {JointGreedy, GreedyCover} x {CVaR_0.1, mean (gamma = 1)}.

  JG        LLMmap/joint_greedy.joint_greedy(..., gamma=0.1), D029 chain + counts reused verbatim
  JG-mean   the same call with gamma = 1.0                                                    NEW
  GC        LLMmap/greedy_cover.greedy_cover(S, 8, gamma=0.1, agg="max") on D029's frozen v2 tensor, reused verbatim
  GC-mean   the same call with gamma = 1.0                                                    NEW

XY2 / XX2 are rebuilt from the v2 S_build cubes with d010_select.precompute (as D029 / D036 did; nothing is stored).
JG chains are cross-checked against d024_lib.greedy_cvar on d036_lib.energy_hooks(..., "sf") (same float32 ops),
which also counts exact ties at each step's maximum. Nested k = 1..8, ties to the lowest id.
"""
import json

import numpy as np

import d031_lib as L31
import d036_lib as L36
from d024_lib import greedy_cvar

OUT = "./results/D037"
VIEW = "v2"
K = 8
NPOOL = 259
GAMMAS = {"JG": 0.1, "JG-mean": 1.0, "GC": 0.1, "GC-mean": 1.0}
GRID = ("JG", "JG-mean", "GC", "GC-mean")
CONTRASTS = {"C1": ("JG", "JG-mean"), "C2": ("JG", "GC"), "C3": ("GC", "GC-mean"), "C4": ("JG-mean", "GC-mean")}
D029_TENSOR_SHA = "8a31d7603c27d1e8b455e4b653ba3c3bc5edd4cf50582f9b43443abc73a1f223"


def d029_chains():
    a = json.load(open("./results/D029/chains.json"))["arms"]
    return dict(JG=a["JG"]["ids"][:K], GC=a["GC"]["ids"][:K], PAPER8=a["PAPER8"]["ids"][:K],
                JG_trace=a["JG"]["trace"][:K], GC_trace=a["GC"]["trace"][:K])


def tensor():
    S = np.load(L31.VIEW[VIEW]["sf"])
    assert L31.sha(L31.VIEW[VIEW]["sf"]) == D029_TENSOR_SHA, "STOP: v2 sf tensor hash differs from D029's"
    assert S.shape == (NPOOL, 3570), S.shape
    return S


def accumulators():
    from d010_select import precompute
    ms = L31.models(VIEW)
    ia, ib = L31.pairs(VIEW)
    X = L31.cubes(VIEW, ("build",))["build"][1]
    XX2, XY2 = precompute(X, ms, [(ms[a], ms[b]) for a, b in zip(ia, ib)])
    assert XY2.shape == (NPOOL, len(ia), 75, 75), XY2.shape
    return XY2, XX2, ia, ib


def jg_chain(XY2, XX2, ia, ib, gamma):
    """joint_greedy (the D's path) and greedy_cvar (tie counts); chains must be equal."""
    from LLMmap.joint_greedy import joint_greedy
    sel, trace = joint_greedy(XY2, XX2, ia, ib, K, gamma=gamma, kind="energy", verbose=False)
    sel2, trace2 = greedy_cvar(NPOOL, *L36.energy_hooks(XY2, XX2, ia, ib, "sf"), k=K, gamma=gamma)
    assert [int(q) for q in sel] == [int(q) for q in sel2], (sel, sel2)
    for a, b in zip(trace, trace2):
        assert a["objective_cvar"] == b["objective_cvar"], (a, b)
        a["n_exact_ties_at_max"] = b["n_exact_ties_at_max"]
    return [int(q) for q in sel], trace


def gc_chain(S, gamma):
    from LLMmap.greedy_cover import greedy_cover
    sel, trace = greedy_cover(S, K, gamma=gamma, agg="max", return_trace=True)
    return [int(q) for q in sel], trace


def train_order(S, ref):
    """D024 Call 2 / d031_lib.train_order, extended: a set equal to a D029 k-prefix (JG, GC, PAPER8) is trained in
    that arm's D029 order (D029 trained cands[:k], chain order), so the identical-input assert can be exact."""
    S = [int(q) for q in S]
    k = len(S)
    for tag in ("JG", "GC", "PAPER8"):
        if set(S) == set(ref[tag][:k]):
            return list(ref[tag][:k]), tag
    shared = [q for q in ref["JG"][:k] if q in S]
    return shared + sorted(set(S) - set(shared)), None
