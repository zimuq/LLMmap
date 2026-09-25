"""D017 / S2–S5 — a linear classifier on the same frozen traces, vs the attention net.

Call 1 (approved): CONCATENATION is primary, mean-pooling secondary.
  * concat is exactly order-invariant once lbfgs converges (S0b: gap 0.004324 at
    the default solver -> 0.000000 at max_iter=20000, tol=1e-6). D017's objection
    applies to a fixed-weight model, not a fitted one.
  * mean-pool is the strict special case W_i = W/k -- one weight matrix shared
    across slots, so it cannot tell which query produced which response. It is
    kept because the concat-minus-meanpool gap measures how much of the benefit
    needs per-query resolution.

S1 (approved): C = 1.0 for concat, C = 3.0 for mean-pool, both chosen on `val`
only and frozen across every condition and every k -- D009/F4's fairness rule.

F1 (approved): the fit is bit-deterministic across `random_state`, so there is no
seed axis. Uncertainty is the config-level paired bootstrap this project already
uses. Cross-architecture deltas therefore have a seed range on ONE side only and
are labelled as such, never as a symmetric test.

F2: at k=1 `coverage` and `joint_energy` select the same single query (D010/R2),
so that column is one condition, not two. Asserted, and marked in the output.

Usage:  PYTHONPATH=.:experiments python experiments/d017_linear.py
"""
import os
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from d008_lib import near_relative_pairs, boot_draws, boot_metrics, ci, paired_ci
from d009_lib import load_query_embeddings, build_traces, logit_stats
from d007_lib import load_corpus

OUT = "./results/D017"
MAX_ITER, TOL = 20000, 1e-6
C_CONCAT, C_MEANPOOL = 1.0, 3.0
N_BOOT, NCFG = 2000, 25
NAMED = ["tiiuae/Falcon3-10B-Instruct | tiiuae/Falcon3-7B-Instruct",
         "microsoft/Phi-3-medium-128k-instruct | microsoft/Phi-3-medium-4k-instruct"]
METRICS = ("mean_top1", "worst_class", "worst3_class", "hard_subset")
SMOKE = bool(os.environ.get("D017_SMOKE"))
KS = [1, 8] if SMOKE else list(range(1, 9))
if SMOKE:
    N_BOOT = 200


def chains():
    """The already-selected chains, read from the source D's (I6)."""
    d9 = json.load(open("./results/D009/runs.json"))
    d10 = json.load(open("./results/D010/metrics_by_k.json"))
    g = lambda runs, cond: next(r for r in runs if r["k"] == 8 and not r["error"]
                                and r.get("condition") == cond)["queries"]
    return {"paper8": g(d9["runs"], "paper8"),
            "coverage": g(d9["runs"], "cvar_max"),
            "joint_energy": g(d10["runs"], "joint_energy")}


def fit_predict(Xtr, ytr, Xte, C):
    clf = LogisticRegression(C=C, max_iter=MAX_ITER, tol=TOL, solver="lbfgs")
    clf.fit(Xtr, ytr)
    return clf.decision_function(Xte), int(clf.n_iter_.max())


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    ch = chains()
    qe = load_query_embeddings()
    cubes = {p: load_corpus(pool=p) for p in ("build", "val", "test")}
    models = json.load(open("./results/D008/selection.json"))["models"]
    n_models = len(models)
    hard = near_relative_pairs(models)
    keys = sorted(hard)
    names = [" | ".join(hard[i]["pair"]) for i in keys]
    M = boot_draws(NCFG, N_BOOT)
    print(f"{n_models} models, {len(hard)} structural pairs; "
          f"concat C={C_CONCAT}, meanpool C={C_MEANPOOL}, "
          f"solver max_iter={MAX_ITER} tol={TOL}", flush=True)

    POOL = {"concat": (lambda a: a.reshape(len(a), -1), C_CONCAT),
            "meanpool": (lambda a: a.mean(axis=1), C_MEANPOOL)}
    res, boots, per_pair, iters = {}, {}, {}, []
    for pname, (pool, C) in POOL.items():
        res[pname], boots[pname], per_pair[pname] = {}, {}, {}
        for cname, chain in ch.items():
            res[pname][cname], boots[pname][cname] = {}, {}
            per_pair[pname][cname] = {}
            for k in KS:
                q = chain[:k]
                tr, y_tr, _, _ = build_traces(q, "build", qe, cubes["build"])
                te, y_te, c_te, _ = build_traces(q, "test", qe, cubes["test"])
                t = time.time()
                lg, ni = fit_predict(pool(tr), y_tr, pool(te), C)
                iters.append(dict(pooling=pname, condition=cname, k=k, n_iter=ni,
                                  converged=bool(ni < MAX_ITER),
                                  wall_s=round(time.time() - t, 1)))
                pt, st = logit_stats(lg, y_te, c_te, hard, n_models)
                res[pname][cname][k] = {m: pt[m] for m in METRICS}
                res[pname][cname][k]["queries"] = q
                per_pair[pname][cname][k] = dict(zip(names, pt["per_hard_pair"]))
                b = boot_metrics(st, M)
                boots[pname][cname][k] = b
            r8 = res[pname][cname][max(KS)]
            print(f"[S2] {pname:9s} {cname:13s} k={max(KS)} mean {r8['mean_top1']:.4f} "
                  f"worst {r8['worst_class']:.4f} hard {r8['hard_subset']:.4f}",
                  flush=True)

    bad = [i for i in iters if not i["converged"]]
    print(f"[S2] all fits converged: {not bad}"
          + (f" ({len(bad)} hit max_iter)" if bad else ""), flush=True)

    # F2: coverage and joint_energy must coincide at k=1
    if 1 in KS:
        for pname in POOL:
            a = res[pname]["coverage"][1]["mean_top1"]
            b = res[pname]["joint_energy"][1]["mean_top1"]
            assert abs(a - b) < 1e-12, (pname, a, b)
        print("[F2] k=1: coverage == joint_energy exactly (one condition, not two)",
              flush=True)

    # ---- S5: ordering, per k, sign-consistency + TOST
    att = json.load(open("./results/D009/metrics_by_condition_k.json"))
    att10 = json.load(open("./results/D010/metrics_by_k.json"))
    ATT = {"paper8": lambda k: att["metrics"]["paper8"][str(k)],
           "coverage": lambda k: att["metrics"]["cvar_max"][str(k)],
           "joint_energy": lambda k: att10["metrics"]["joint_energy"][str(k)]}

    def tost(d, delta=0.02):
        return dict(equivalent=bool(d["lo"] > -delta and d["hi"] < delta),
                    ci_excludes_zero=bool(d["lo"] > 0 or d["hi"] < 0))

    order = {}
    for pname in POOL:
        order[pname] = {}
        for a, b in (("joint_energy", "coverage"), ("coverage", "paper8"),
                     ("joint_energy", "paper8")):
            rows = []
            for k in KS:
                if k == 1 and a == "joint_energy" and b == "coverage":
                    continue                       # F2: same condition
                for m in METRICS:
                    pass
                d = paired_ci(boots[pname][a][k]["mean_top1"],
                              boots[pname][b][k]["mean_top1"])
                d.update(tost(d)); d["k"] = k
                rows.append(d)
            deltas = [r["delta"] for r in rows]
            sgn = [np.sign(x) for x in deltas if abs(x) > 1e-12]
            order[pname][f"{a}_vs_{b}"] = dict(
                by_k=rows, n_k=len(rows),
                consistent_sign=bool(sgn and all(s == sgn[0] for s in sgn)),
                direction=("positive" if sgn and sgn[0] > 0 else
                           "negative" if sgn else "zero"),
                n_ci_excludes_zero=sum(1 for r in rows if r["ci_excludes_zero"]),
                n_equivalent=sum(1 for r in rows if r["equivalent"]),
                mean_delta=round(float(np.mean(deltas)), 4))

    print(f"\n===== S5: ordering under the linear classifier =====")
    for pname in POOL:
        for pair, v in order[pname].items():
            print(f"  {pname:9s} {pair:28s} mean Δ {v['mean_delta']:+.4f}  "
                  f"consistent sign {str(v['consistent_sign']):5s}  "
                  f"CI≠0 on {v['n_ci_excludes_zero']}/{v['n_k']}  "
                  f"equiv on {v['n_equivalent']}/{v['n_k']}", flush=True)

    # ---- S4: the two named pairs, linear vs attention
    d15 = {r["pair"]: r for r in json.load(
        open("./results/D015/per_pair_k8.json"))["rows"]}
    named = []
    for nm in NAMED:
        for k in ([1, 8] if 1 in KS else [max(KS)]):
            row = dict(pair=nm, k=k)
            for pname in POOL:
                for cname in ch:
                    row[f"{pname}_{cname}"] = round(
                        float(per_pair[pname][cname][k][nm]), 4)
            if k == 8 and nm in d15:
                for cname in ch:
                    row[f"attention_{cname}"] = d15[nm][cname]
            named.append(row)
    print(f"\n===== S4: the two named pairs, k=8 =====")
    for row in [r for r in named if r["k"] == 8]:
        print(f"  {row['pair'][:52]:52s}")
        for cname in ch:
            print(f"     {cname:13s} concat {row['concat_'+cname]:.3f} | "
                  f"meanpool {row['meanpool_'+cname]:.3f} | "
                  f"attention {row.get('attention_'+cname, float('nan')):.3f}",
                  flush=True)

    # ---- linear vs attention, side by side (one-sided seed range, F1)
    side = []
    for k in KS:
        for cname in ch:
            a = ATT[cname](k)
            side.append(dict(k=k, condition=cname,
                             linear_concat=res["concat"][cname][k]["mean_top1"],
                             linear_meanpool=res["meanpool"][cname][k]["mean_top1"],
                             attention=a["mean_top1"],
                             attention_seed_range=a.get("mean_top1_range"),
                             delta_concat_minus_attention=round(
                                 res["concat"][cname][k]["mean_top1"]
                                 - a["mean_top1"], 4)))
    print(f"\n===== linear (concat) vs attention network, mean top-1 =====")
    print(f"{'k':>3s} " + "".join(f"{c:>26s}" for c in ch))
    for k in KS:
        row = f"{k:>3d} "
        for cname in ch:
            s = next(x for x in side if x["k"] == k and x["condition"] == cname)
            row += f"{s['linear_concat']:.4f} vs {s['attention']:.4f} ({s['delta_concat_minus_attention']:+.3f})".rjust(26)
        print(row, flush=True)

    json.dump(dict(schema="d017-linear-v1", ks=KS, pooling=list(POOL),
                   C=dict(concat=C_CONCAT, meanpool=C_MEANPOOL),
                   solver=dict(max_iter=MAX_ITER, tol=TOL),
                   deterministic=True, n_boot=N_BOOT,
                   seed_axis="dropped -- the fit is bit-deterministic (F1); "
                             "uncertainty is the config-level paired bootstrap. "
                             "Cross-architecture deltas have a seed range on the "
                             "ATTENTION side only and are one-sided.",
                   convergence=iters, metrics=res, ordering=order,
                   linear_vs_attention=side,
                   bootstrap_ci={p: {c: {k: {m: ci(boots[p][c][k][m])
                                             for m in METRICS}
                                         for k in boots[p][c]} for c in boots[p]}
                                 for p in boots},
                   wall_min=round((time.time() - t0) / 60, 1)),
              open(f"{OUT}/linear_metrics_by_k.json", "w"), indent=1)
    json.dump(dict(schema="d017-named-v1", pairs=NAMED, rows=named,
                   attention_source="results/D015/per_pair_k8.json"),
              open(f"{OUT}/linear_named_pairs.json", "w"), indent=1)
    json.dump({p: {c: {str(k): per_pair[p][c][k] for k in per_pair[p][c]}
                   for c in per_pair[p]} for p in per_pair},
              open(f"{OUT}/linear_per_hard_pair.json", "w"), indent=1)
    print(f"\nwall {(time.time()-t0)/60:.1f} min; written: {OUT}/", flush=True)


if __name__ == "__main__":
    main()
