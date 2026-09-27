"""D020 / S0–S3 — properly paired linear-readout comparison of the four chains.

S0  refit GreedyCover / JointGreedy / H1-attention / H1-linear under both linear
    poolings, k=1..8, on D017's exact code path (C=1.0 concat / 3.0 mean-pool,
    lbfgs, max_iter=20000, tol=1e-6, no scaler). Every refit must reproduce the
    stored metrics exactly (D017 for GC/JG, D018 for the H1 chains) before its
    per-config counts are trusted. Neither D017 nor D018 saved those counts.
    Call 1 (approved): 4 reversed-slot-order k=8 concat fits measure concat's
    solver-tolerance order floor.
S1  one shared boot_draws(25, 2000) across every chain/pooling/k (paired).
S2  four deltas x 8 k x 2 poolings x 4 metrics. Resolved = paired 95% CI
    excludes 0 (deterministic fits: no seed-range condition, per D020).
    Call 2 (approved, pre-registered): PRIMARY = JointGreedy - H1-linear,
    mean_top1, k=4..8, per pooling -- resolved at a k only if the CI excludes 0
    AND the point-delta sign is the same at every k=4..8. Everything else is
    per-cell descriptive with a chance-expectation count. Identical-input
    cells are excluded; the k=2 JG/H1-linear cell is order-only.

Usage:  PYTHONPATH=.:experiments python experiments/d020_paired_linear.py
"""
import os
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from d009_lib import load_query_embeddings, build_traces, logit_stats
from d008_lib import near_relative_pairs, boot_draws, boot_metrics, ci
from d007_lib import load_corpus

OUT = "./results/D020"
MAX_ITER, TOL, NCFG, N_BOOT = 20000, 1e-6, 25, 2000
METRICS = ("mean_top1", "worst_class", "worst3_class", "hard_subset")
POOL = {"concat": (lambda a: a.reshape(len(a), -1), 1.0),
        "meanpool": (lambda a: a.mean(axis=1), 3.0)}
KS = range(1, 9)
CHAINS = ("GreedyCover", "JointGreedy", "H1-attention", "H1-linear")
DELTAS = (("JointGreedy", "H1-linear"), ("JointGreedy", "H1-attention"),
          ("GreedyCover", "H1-linear"), ("GreedyCover", "H1-attention"))
PRIMARY = ("JointGreedy", "H1-linear")
PRIMARY_KS = range(4, 9)


def chains_and_refs():
    """Chains verbatim (I6) and the stored metrics each refit must reproduce."""
    d9 = json.load(open("./results/D009/runs.json"))["runs"]
    d10 = json.load(open("./results/D010/metrics_by_k.json"))["runs"]
    g = lambda runs, cond: next(r for r in runs if r["k"] == 8 and not r["error"]
                                and r["condition"] == cond)["queries"]
    h1 = json.load(open("./results/D018/greedy_chains.json"))["chains"]
    chains = {"GreedyCover": g(d9, "cvar_max"), "JointGreedy": g(d10, "joint_energy"),
              "H1-attention": h1["H1-attention"][:8], "H1-linear": h1["H1-linear"][:8]}
    L = json.load(open("./results/D017/linear_metrics_by_k.json"))["metrics"]
    D = json.load(open("./results/D018/metrics_by_k.json"))["metrics"]
    refs = {}
    for p in POOL:
        refs[("GreedyCover", p)] = L[p]["coverage"]
        refs[("JointGreedy", p)] = L[p]["joint_energy"]
        refs[("H1-attention", p)] = D[f"H1-attention|linear_{p}"]
        refs[("H1-linear", p)] = D[f"H1-linear|linear_{p}"]
    for (c, p), ref in refs.items():   # chains must be the ones those refs used
        assert ref["8"]["queries"] == chains[c], (c, p)
    return chains, refs


def fit(q, pool, C, qe, cubes, hard, n_models):
    tr, y_tr, _, _ = build_traces(q, "build", qe, cubes["build"])
    te, y_te, c_te, _ = build_traces(q, "test", qe, cubes["test"])
    clf = LogisticRegression(C=C, max_iter=MAX_ITER, tol=TOL, solver="lbfgs")
    clf.fit(pool(tr), y_tr)
    n_it = int(clf.n_iter_.max())
    assert n_it < MAX_ITER, f"max_iter hit ({q}, C={C})"
    lg = clf.decision_function(pool(te))
    pt, st = logit_stats(lg, y_te, c_te, hard, n_models)
    return pt, st, lg.argmax(1), n_it


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    models = json.load(open("./results/D008/selection.json"))["models"]
    n_models = len(models)
    hard = near_relative_pairs(models)
    chains, refs = chains_and_refs()
    qe = load_query_embeddings()
    cubes = {p: load_corpus(pool=p) for p in ("build", "test")}
    M = boot_draws(NCFG, N_BOOT)
    n_test = n_models * NCFG

    # ---------------- S0: refit + exact reproduction
    pts, sts, boots, repro, fit_s = {}, {}, {}, {}, []
    counts = {}
    for p, (pool, C) in POOL.items():
        for c in CHAINS:
            for k in KS:
                t1 = time.time()
                pt, st, _, n_it = fit(chains[c][:k], pool, C, qe, cubes, hard, n_models)
                fit_s.append(time.time() - t1)
                ref = refs[(c, p)][str(k)]
                diff = max(abs(pt[m] - ref[m]) for m in METRICS)
                repro[f"{c}|{p}|{k}"] = dict(max_abs_diff=diff, n_iter=n_it)
                assert diff < 1e-9, (c, p, k, diff)
                pts[(c, p, k)], sts[(c, p, k)] = pt, st
                boots[(c, p, k)] = boot_metrics(st, M)
                for kk in ("cnt_total", "cnt_model", "cnt_pair"):
                    counts[f"{c}|{p}|{k}|{kk}"] = st[kk]
            print(f"[S0] {p:8s} {c:13s} k=1..8 reproduce stored exactly; "
                  f"k=8 mean_top1 {pts[(c, p, 8)]['mean_top1']:.4f}", flush=True)
    np.savez_compressed(f"{OUT}/linear_counts.npz", **counts)

    # ---------------- S0 / Call 1: concat slot-order floor at k=8
    pool, C = POOL["concat"]
    floor = {}
    for c in CHAINS:
        q = chains[c][:8]
        a_pt, _, a_pred, _ = fit(q, pool, C, qe, cubes, hard, n_models)
        b_pt, _, b_pred, _ = fit(q[::-1], pool, C, qe, cubes, hard, n_models)
        floor[c] = dict(chain_order=q, reversed_order=q[::-1],
                        mean_top1_chain=a_pt["mean_top1"],
                        mean_top1_reversed=b_pt["mean_top1"],
                        correct_count_gap=int(round(abs(a_pt["mean_top1"]
                                                        - b_pt["mean_top1"]) * n_test)),
                        predictions_differ=int((a_pred != b_pred).sum()))
        print(f"[Call1] {c:13s} reversed-order gap "
              f"{floor[c]['correct_count_gap']} traces, "
              f"{floor[c]['predictions_differ']} predictions differ", flush=True)
    floor_traces = max(f["correct_count_gap"] for f in floor.values())

    # ---------------- S1/S2: paired deltas
    def same_input(a, b, k):
        qa, qb = chains[a][:k], chains[b][:k]
        if qa == qb:
            return "identical"
        if sorted(qa) == sorted(qb):
            return "order_only"
        return None

    cells = []
    for a, b in DELTAS:
        for p in POOL:
            for k in KS:
                kind = same_input(a, b, k)
                for m in METRICS:
                    d = boots[(a, p, k)][m] - boots[(b, p, k)][m]
                    point = pts[(a, p, k)][m] - pts[(b, p, k)][m]
                    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
                    cell = dict(delta=f"{a} - {b}", pooling=p, k=k, metric=m,
                                a=pts[(a, p, k)][m], b=pts[(b, p, k)][m],
                                point_delta=point, lo=lo, hi=hi,
                                ci_excludes_0=bool(lo > 0 or hi < 0),
                                input_kind=kind or "distinct")
                    if m == "mean_top1":
                        cell["delta_traces"] = int(round(point * n_test))
                        cell["at_or_below_concat_floor"] = bool(
                            p == "concat" and abs(cell["delta_traces"]) <= floor_traces)
                    cells.append(cell)

    # primary (Call 2)
    primary = {}
    for p in POOL:
        rows = [c for c in cells if c["delta"] == " - ".join(PRIMARY)
                and c["pooling"] == p and c["metric"] == "mean_top1"
                and c["k"] in PRIMARY_KS]
        signs = {np.sign(r["point_delta"]) for r in rows}
        consistent = len(signs) == 1 and 0 not in signs
        per_k = {r["k"]: dict(point_delta=round(r["point_delta"], 4),
                              lo=round(r["lo"], 4), hi=round(r["hi"], 4),
                              delta_traces=r["delta_traces"],
                              ci_excludes_0=r["ci_excludes_0"],
                              resolved=bool(consistent and r["ci_excludes_0"]))
                 for r in rows}
        primary[p] = dict(sign_consistent_k4_8=bool(consistent),
                          n_resolved=sum(v["resolved"] for v in per_k.values()),
                          per_k=per_k)
        print(f"[S2] PRIMARY {p:8s} sign-consistent {consistent}; resolved at "
              f"{[k for k, v in per_k.items() if v['resolved']]}", flush=True)
        for k, v in per_k.items():
            print(f"     k={k}: {v['point_delta']:+.4f} [{v['lo']:+.4f}, "
                  f"{v['hi']:+.4f}] ({v['delta_traces']:+d} traces)", flush=True)

    # secondary count vs chance
    counted = [c for c in cells if c["input_kind"] == "distinct"]
    by_metric = {m: dict(n_cells=sum(1 for c in counted if c["metric"] == m),
                         n_ci_excludes_0=sum(1 for c in counted if c["metric"] == m
                                             and c["ci_excludes_0"]))
                 for m in METRICS}
    for m in by_metric:
        by_metric[m]["chance_expectation_5pct"] = round(0.05 * by_metric[m]["n_cells"], 1)
    print(f"[S2] cells (distinct inputs) by metric: {by_metric}", flush=True)
    print(f"[S2] concat order floor: {floor_traces} trace(s)", flush=True)

    json.dump(dict(
        schema="d020-paired-linear-v1", ks=list(KS), n_boot=N_BOOT,
        boot_seed=20260908, n_test_traces=n_test,
        solver=dict(C={p: POOL[p][1] for p in POOL}, max_iter=MAX_ITER, tol=TOL,
                    solver="lbfgs", scaler=None),
        chains=chains,
        resolution_rule="paired 95% CI excludes 0 (deterministic fits, no seed axis "
                        "-- D020); PRIMARY additionally requires a consistent "
                        "point-delta sign across k=4..8 (P1/Call 2)",
        primary=dict(delta=" - ".join(PRIMARY), metric="mean_top1",
                     ks=list(PRIMARY_KS), by_pooling=primary),
        secondary_counts=by_metric,
        concat_order_floor=dict(floor_traces=floor_traces, per_chain=floor,
                                note="reversed slot order, k=8, same solver "
                                     "(P1/Call 1); flag only, not a rule"),
        reproduction=dict(all_exact=True, tolerance=1e-9, cells=repro,
                          max_n_iter=max(v["n_iter"] for v in repro.values())),
        points={f"{c}|{p}|{k}": {m: pts[(c, p, k)][m] for m in METRICS}
                for (c, p, k) in pts},
        bootstrap_ci={f"{c}|{p}|{k}": {m: ci(boots[(c, p, k)][m]) for m in METRICS}
                      for (c, p, k) in boots},
        cells=cells,
        counts_file=f"{OUT}/linear_counts.npz (cnt_total/cnt_model/cnt_pair per "
                    "chain|pooling|k)",
        cost=dict(n_fits=len(fit_s) + 2 * len(CHAINS),
                  fit_s_median=round(float(np.median(fit_s)), 2),
                  fit_s_max=round(float(np.max(fit_s)), 2),
                  wall_min=round((time.time() - t0) / 60, 2))),
        open(f"{OUT}/paired_linear_comparison.json", "w"), indent=1)
    print(f"\nwall {(time.time() - t0) / 60:.1f} min; written {OUT}/", flush=True)


if __name__ == "__main__":
    main()
