"""D017 / S0b + S1 — why concat was not order-invariant, and picking C once.

S0 predicted concatenation would be EXACTLY order-invariant for a fitted linear
model (permuting slots permutes the learned blocks; the L2 penalty is a sum of
squares over all blocks, so both loss and penalty are permutation-equivariant and
the convex optimum is unique). Measured: 0.900541 vs 0.896216 -- a 0.0043 gap.

The prediction was wrong about the OPTIMIZER, not the objective. lbfgs stops on
max_iter/tol, and its path depends on feature order, so an unconverged fit can
differ. If that is the explanation, tightening convergence shrinks the gap toward
zero; if it does not, the objective is genuinely order-dependent and concatenation
must be rejected. This decides Call 1 on evidence.

S1 then picks the regularization ONCE on paper8 k=8's validation split and freezes
it for every condition and k -- the same discipline as the attention network's
asserted-identical hparams hash.

Usage:  PYTHONPATH=.:experiments python experiments/d017_s0b.py
"""
import os
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from d009_lib import load_query_embeddings, build_traces
from d007_lib import load_corpus

OUT = "./results/D017"
K = 8
CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def fit(Xtr, ytr, C, max_iter, tol):
    clf = LogisticRegression(C=C, max_iter=max_iter, tol=tol, solver="lbfgs")
    clf.fit(Xtr, ytr)
    return clf


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    d9 = json.load(open("./results/D009/runs.json"))
    chain = next(r for r in d9["runs"] if r["k"] == K and r["condition"] == "paper8"
                 and r["run"] == 0)["queries"]
    qe = load_query_embeddings()
    cubes = {p: load_corpus(pool=p) for p in ("build", "val", "test")}
    tr, y_tr, _, models = build_traces(chain, "build", qe, cubes["build"])
    va, y_va, _, _ = build_traces(chain, "val", qe, cubes["val"])
    te, y_te, _, _ = build_traces(chain, "test", qe, cubes["test"])
    X = lambda a: a.reshape(len(a), -1)
    rng = np.random.default_rng(0)
    perm = rng.permutation(K)

    # ---- S0b: is the order gap a convergence artifact?
    conv = []
    for mi, tol in ((3000, 1e-4), (20000, 1e-6), (60000, 1e-8)):
        t = time.time()
        a = float((fit(X(tr), y_tr, 1.0, mi, tol).predict(X(te)) == y_te).mean())
        b = float((fit(X(tr[:, perm]), y_tr, 1.0, mi, tol)
                   .predict(X(te[:, perm])) == y_te).mean())
        conv.append(dict(max_iter=mi, tol=tol, chain_order=round(a, 6),
                         permuted=round(b, 6), gap=round(abs(a - b), 6),
                         wall_s=round(time.time() - t, 1)))
        print(f"[S0b] max_iter={mi:<6} tol={tol:<7} chain {a:.6f} perm {b:.6f} "
              f"gap {abs(a-b):.6f}  ({time.time()-t:.1f}s)", flush=True)
    shrinks = conv[-1]["gap"] < conv[0]["gap"]
    print(f"[S0b] gap shrinks with tighter convergence: {shrinks} "
          f"({conv[0]['gap']:.6f} -> {conv[-1]['gap']:.6f})", flush=True)

    # ---- S1: pick C once, on VALIDATION, for concat
    sel = []
    for C in CS:
        t = time.time()
        clf = fit(X(tr), y_tr, C, 20000, 1e-6)
        av = float((clf.predict(X(va)) == y_va).mean())
        at = float((clf.predict(X(te)) == y_te).mean())
        sel.append(dict(C=C, val=round(av, 4), test=round(at, 4),
                        wall_s=round(time.time() - t, 1)))
        print(f"[S1] C={C:<6} val {av:.4f}  (test {at:.4f}, not used to choose)",
              flush=True)
    best = max(sel, key=lambda r: r["val"])
    print(f"[S1] chosen on VAL: C={best['C']} (val {best['val']:.4f})", flush=True)

    # same sweep for mean-pool, so the secondary is not handicapped by concat's C
    selm = []
    for C in CS:
        clf = fit(tr.mean(1), y_tr, C, 20000, 1e-6)
        selm.append(dict(C=C, val=round(float((clf.predict(va.mean(1)) == y_va).mean()), 4)))
    bestm = max(selm, key=lambda r: r["val"])
    print(f"[S1] mean-pool chosen on VAL: C={bestm['C']} (val {bestm['val']:.4f})",
          flush=True)

    json.dump(dict(schema="d017-s0b-s1-v1",
                   convergence_probe=dict(
                       rows=conv, gap_shrinks_with_convergence=bool(shrinks),
                       conclusion=("the S0 order gap is an lbfgs CONVERGENCE "
                                   "artifact, not an order dependency of the "
                                   "objective -- the convex optimum is "
                                   "permutation-equivariant and the gap closes as "
                                   "the solver converges"
                                   if shrinks else
                                   "gap does NOT close with convergence -- treat "
                                   "concatenation as genuinely order-dependent "
                                   "and reject it")),
                   hparam_selection=dict(
                       grid=CS, concat=sel, chosen_C_concat=best["C"],
                       meanpool=selm, chosen_C_meanpool=bestm["C"],
                       chosen_on="validation split only; test never used to pick",
                       frozen="this C is reused identically for every condition "
                              "and every k (D009/F4's fairness discipline)",
                       solver=dict(max_iter=20000, tol=1e-6)),
                   wall_s=round(time.time() - t0, 1)),
              open(f"{OUT}/hparam_selection.json", "w"), indent=1)
    print(f"\nwall {time.time()-t0:.1f}s; written: {OUT}/hparam_selection.json",
          flush=True)


if __name__ == "__main__":
    main()
