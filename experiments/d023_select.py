"""D023 / S1 (+ S-f) — selection, frozen and committed BEFORE any training.

ILP-ADD   exact MILP per k=1..8 on D021's primary tensor (d023_lib.solve), plus a
          no-good-cut re-solve to test for a tied optimum. Ordered per Call 3:
          queries shared with JointGreedy's chain first, in JointGreedy's order,
          then ascending id.
Greedy-ADD nested greedy on the same objective (ties -> lowest id), own order.
LP bounds UB_k and the certificate (7 chains x 8 k).
Identical-input k lists (Call 2): ordered list == JointGreedy's prefix.
S-f (Review amendment): re-solve on D021 variants d64_grouped and d64_halfsigma;
          Jaccard with the primary set, primary objective as a fraction of the
          variant optimum; at k=8 the number of distinct sets within 1% of the
          primary optimum (iterative no-good cuts, capped at 20).
Target-free. Writes results/D023/selection.json.
Usage:  PYTHONPATH=.:experiments python experiments/d023_select.py
"""
import json
import time

import numpy as np
from scipy.optimize import LinearConstraint

from d023_lib import load_evidence, objective, solve, greedy
from d023_s0 import existing_chains

OUT = "./results/D023"


def order_like_jg(S, jg):
    shared = [q for q in jg if q in S]
    return shared + sorted(q for q in S if q not in shared)


def near_optimal_sets(E, k, opt, primary, tol=0.01, cap=20):
    """Distinct k-sets with objective >= (1 - tol) * opt, via iterative no-good cuts."""
    from d023_lib import _model
    from scipy.optimize import milp
    from scipy.sparse import csr_matrix
    nq, npair = E.shape
    found, cuts = [sorted(primary)], []
    while len(found) < cap:
        c, cons, bnds, m = _model(E, k)
        for S in found:
            row = np.zeros(nq + 1 + npair); row[S] = 1
            cons.append(LinearConstraint(csr_matrix(row), lb=-np.inf, ub=k - 1))
        integ = np.concatenate([np.ones(nq), [0], np.zeros(npair)])
        r = milp(c, constraints=cons, bounds=bnds, integrality=integ,
                 options=dict(time_limit=900.0, mip_rel_gap=1e-6, disp=False))
        if r.x is None:
            break
        val = -r.fun / m
        if val < (1 - tol) * opt:
            break
        found.append(sorted(int(i) for i in np.flatnonzero(r.x[:nq] > 0.5)))
    return found


def main():
    t0 = time.time()
    E, meta = load_evidence()
    variants = {v: np.load(f"./results/D021/evidence_tok200_{v}.npy").astype(np.float64)
                for v in ("d64_grouped", "d64_halfsigma")}
    ch = existing_chains()
    jg = ch["JointGreedy"]

    ilp, lp, ties = {}, {}, {}
    for k in range(1, 9):
        r = solve(E, k, integer=True)
        assert r["status"] == 0 and abs(r["cvar"] - r["cvar_recomputed"]) < 1e-6, r
        r2 = solve(E, k, integer=True, exclude=r["set"])
        ilp[k] = dict(r, ordered=order_like_jg(r["set"], jg))
        ties[k] = dict(second_best=r2.get("cvar"), second_set=r2.get("set"),
                       tied=bool(r2.get("cvar") is not None and abs(r2["cvar"] - r["cvar"]) < 1e-9))
        lp[k] = solve(E, k, integer=False)
        print(f"[S1] k={k}: ILP {ilp[k]['ordered']} cvar {r['cvar']:.4f} gap {r['mip_gap']} "
              f"{r['wall_s']}s | LP {lp[k]['cvar']:.4f} | 2nd {ties[k]['second_best']}", flush=True)
    gsel, gtrace = greedy(E, 8)
    print(f"[S1] Greedy-ADD {gsel}", flush=True)

    chains = {**ch, "Greedy-ADD": gsel}
    cert = {}
    for c, q in chains.items():
        cert[c] = [dict(k=k, cvar=round(objective(E, q[:k]), 4),
                        frac_of_ub=round(objective(E, q[:k]) / lp[k]["cvar"], 4),
                        frac_of_ilp=round(objective(E, q[:k]) / ilp[k]["cvar"], 4))
                   for k in range(1, 9)]
    cert["ILP-ADD"] = [dict(k=k, cvar=round(ilp[k]["cvar"], 4),
                            frac_of_ub=round(ilp[k]["cvar"] / lp[k]["cvar"], 4),
                            frac_of_ilp=1.0) for k in range(1, 9)]

    ident = {"ILP-ADD": [k for k in range(1, 9) if ilp[k]["ordered"] == jg[:k]],
             "Greedy-ADD": [k for k in range(1, 9) if gsel[:k] == jg[:k]]}
    same_set_diff_order = {"Greedy-ADD": [k for k in range(1, 9) if gsel[:k] != jg[:k]
                                          and sorted(gsel[:k]) == sorted(jg[:k])]}
    differing = {a: [k for k in range(1, 9) if k not in ident[a]] for a in ident}
    nesting = [dict(k=k, subset_of_next=set(ilp[k]["set"]) <= set(ilp[k + 1]["set"]))
               for k in range(1, 8)]
    overlaps = {k: dict(JointGreedy=len(set(ilp[k]["set"]) & set(jg[:k])),
                        paper8=len(set(ilp[k]["set"]) & set(ch["paper8"][:k])),
                        GreedyADD=len(set(ilp[k]["set"]) & set(gsel[:k]))) for k in range(1, 9)}

    # ---- S-f: stability (target-free)
    stab = {}
    for v, Ev in variants.items():
        stab[v] = []
        for k in range(1, 9):
            rv = solve(Ev, k, integer=True)
            A, B = set(ilp[k]["set"]), set(rv["set"])
            stab[v].append(dict(k=k, variant_set=rv["set"],
                                jaccard=round(len(A & B) / len(A | B), 4),
                                primary_set_frac_of_variant_opt=round(
                                    objective(Ev, ilp[k]["set"]) / rv["cvar"], 4)))
        print(f"[S-f] {v}: Jaccard by k {[s['jaccard'] for s in stab[v]]}", flush=True)
    near = near_optimal_sets(E, 8, ilp[8]["cvar"], ilp[8]["set"])
    stab["k8_sets_within_1pct"] = dict(n=len(near), capped_at=20, sets=near,
                                       cap_reached=len(near) >= 20)
    print(f"[S-f] k=8 sets within 1% of optimum: {len(near)}{' (cap)' if len(near) >= 20 else ''}",
          flush=True)

    out = dict(schema="d023-selection-v1", frozen_before_training=True,
               evidence="results/D021/evidence_tok200.npy (d64, full Sigma, D007 halves)",
               objective="CVaR_0.1 over 666 pairs of summed M2 (m=66)",
               solver="HiGHS via scipy.optimize.milp, mip_rel_gap=1e-6, time_limit=900 s",
               ILP_ADD={k: ilp[k] for k in ilp}, ILP_ties=ties, LP_bound={k: lp[k] for k in lp},
               Greedy_ADD=dict(chain=gsel, trace=gtrace),
               train_order=dict(ILP_ADD={k: ilp[k]["ordered"] for k in ilp},
                                Greedy_ADD={k: gsel[:k] for k in range(1, 9)},
                                rule="ILP: JointGreedy's order for shared queries, then "
                                     "ascending id (Call 3); Greedy-ADD: own order"),
               identical_input_k=ident, same_set_different_order_k=same_set_diff_order,
               differing_k=differing,
               parity_quantity="mean over differing_k['ILP-ADD'] (Call 2 (b))",
               nesting=nesting, overlaps=overlaps, certificate=cert, stability=stab,
               existing_chains=ch, wall_s=round(time.time() - t0, 1))
    json.dump(out, open(f"{OUT}/selection.json", "w"), indent=1)
    print(f"[S1] identical-input k {ident}; differing {differing}; "
          f"wall {out['wall_s']} s", flush=True)


if __name__ == "__main__":
    main()
