"""D013 / S4–S6 — train GreedyCover's gamma grid, then compare to JointGreedy.

Trains gamma in {0.5, 0.25, 0.05} x k=1..8 x 5 seeds = 120 runs. D009's gamma=1.0
(`mean_greedy_max`) and gamma=0.1 (`cvar_max`) are READ from results/D009, never
rerun (I6), with their stored per-config counts so every comparison is paired.

Reporting follows P1's approved findings:
  F2 -- S6's validity rests on a CHECK, not on "both are now trained". The hparams
        hash is asserted equal to D009's before any cross-algorithm number is
        emitted, and recorded in the artifact.
  F3 -- at k=1 the two algorithms are the SAME algorithm (the joint statistic
        reduces to the per-query one), so their k=1 rows are one observation, not
        two. S6 asserts they agree and labels it a consistency check.
  F4 -- D012's TOST (delta = 0.02) plus its consistent-sign rule: an ordering
        among sub-1 gammas is reported only if its sign holds across k.

Usage:  PYTHONPATH=.:experiments python experiments/d013_evaluate.py
"""
import os
import json
import time
import hashlib

import numpy as np

from d009_lib import load_query_embeddings, build_traces, hparams_from_shipped
from d009_train import run_one
from d008_lib import near_relative_pairs, boot_draws, boot_metrics, ci, paired_ci
from d007_lib import load_corpus

OUT = "./results/D013"
METRICS = ("mean_top1", "worst_class", "worst3_class", "hard_subset")
DELTA = 0.02
N_RUNS = 5
NEW_GAMMAS = ("0.5", "0.25", "0.05")
REUSED = {"1.0": "mean_greedy_max", "0.1": "cvar_max"}
ALL_GAMMAS = ["1.0", "0.5", "0.25", "0.1", "0.05"]
SMOKE = bool(os.environ.get("D013_SMOKE"))
if SMOKE:
    N_RUNS, NEW_GAMMAS = 2, ("0.5",)


def tost(d, delta=DELTA):
    lo, hi = d["lo"], d["hi"]
    equivalent = (lo > -delta) and (hi < delta)
    different = (lo > 0) or (hi < 0)
    return dict(equivalent=bool(equivalent), ci_excludes_zero=bool(different),
                verdict=("NULL (equivalent)" if equivalent else
                         "different" if different else "INCONCLUSIVE"))


def consistent_sign(deltas):
    """F4's rule: an ordering counts only if its sign holds across k."""
    s = [np.sign(d) for d in deltas if abs(d) > 1e-12]
    return bool(s) and all(x == s[0] for x in s)


def main():
    os.makedirs(f"{OUT}/models", exist_ok=True)
    sel = json.load(open(f"{OUT}/gamma_selection.json"))
    d009 = json.load(open("./results/D009/runs.json"))
    d009m = json.load(open("./results/D009/metrics_by_condition_k.json"))

    # S2's guarantee, re-asserted here so S4 cannot drift from what D009 trained
    assert sel["chains"]["1.0"] == d009m["metrics"]["mean_greedy_max"]["8"]["queries"]
    assert sel["chains"]["0.1"] == d009m["metrics"]["cvar_max"]["8"]["queries"]

    models = json.load(open("./results/D008/selection.json"))["models"]
    n_models = len(models)
    hard = near_relative_pairs(models)
    qe = load_query_embeddings()
    cubes = {p: load_corpus(pool=p) for p in ("build", "val", "test")}
    ks = [1, 8] if SMOKE else list(range(1, 9))

    runs, counts = [], {}
    t_all = time.time()
    for g in NEW_GAMMAS:
        chain = sel["chains"][g]
        for k in ks:
            queries = chain[:k]
            tr_b, y_b, _, _ = build_traces(queries, "build", qe, cubes["build"])
            tr_v, y_v, _, _ = build_traces(queries, "val", qe, cubes["val"])
            tr_t, y_t, c_t, _ = build_traces(queries, "test", qe, cubes["test"])
            hp, conf = hparams_from_shipped(k, n_models)
            conf = dict(conf); conf["inference_model"] = hp
            h = hashlib.sha256(json.dumps(
                {kk: vv for kk, vv in hp.items() if kk != "num_queries"},
                sort_keys=True).encode()).hexdigest()
            assert h == d009["hparams_hash"], (h, d009["hparams_hash"])

            for r in range(N_RUNS):
                keep = f"{OUT}/models/g{g}_k{k}_r{r}.ckpt" if r == 0 else None
                res, cnt, buf = run_one(tr_b, y_b, tr_v, y_v, tr_t, y_t, c_t,
                                        hp, conf, r, hard, n_models, keep)
                res.update(gamma=g, k=k, run=r, seed=r, queries=queries)
                runs.append(res)
                if res["error"]:
                    print(f"  !! g={g} k={k} r={r}: {res['error']}", flush=True)
                else:
                    for key, arr in cnt.items():
                        if isinstance(arr, np.ndarray):
                            counts[f"{g}|{k}|{r}|{key}"] = arr
            done = [x for x in runs if x["gamma"] == g and x["k"] == k
                    and not x["error"]]
            m = np.array([x["mean_top1"] for x in done])
            print(f"[S4] gamma={g:5s} k={k} test top-1 {m.mean():.4f} "
                  f"[{m.min():.4f},{m.max():.4f}]", flush=True)

    np.savez_compressed(f"{OUT}/run_counts.npz", **counts)

    ncfg = 25
    M = boot_draws(ncfg, 2000)
    res, boots = {}, {}
    for g in NEW_GAMMAS:
        res[g], boots[g] = {}, {}
        for k in ks:
            rs = [x for x in runs if x["gamma"] == g and x["k"] == k
                  and not x["error"]]
            if not rs:
                continue
            e = {}
            for m in METRICS + ("train_acc", "val_acc"):
                v = np.array([x[m] for x in rs], float)
                e[m] = float(v.mean()); e[m + "_min"] = float(v.min())
                e[m + "_max"] = float(v.max())
                e[m + "_range"] = float(v.max() - v.min())
            e["n_runs"] = len(rs); e["queries"] = rs[0]["queries"]
            res[g][k] = e
            bs = []
            for x in rs:
                st = {kk: counts[f"{g}|{k}|{x['run']}|{kk}"]
                      for kk in ("cnt_total", "cnt_model", "cnt_pair")}
                st["n_models"] = n_models; st["ncfg"] = ncfg
                bs.append(boot_metrics(st, M))
            boots[g][k] = {m: np.mean([b[m] for b in bs], axis=0) for m in METRICS}

    # reused points: D009's stored per-config counts, same draws -> paired
    d9c = np.load("./results/D009/run_counts.npz")
    for g, cond in REUSED.items():
        res[g], boots[g] = {}, {}
        for k in ks:
            rs = sorted({n.split("|")[2] for n in d9c.files
                         if n.startswith(f"{cond}|{k}|")})
            if not rs:
                continue
            bs = []
            for r in rs:
                st = {kk: d9c[f"{cond}|{k}|{r}|{kk}"]
                      for kk in ("cnt_total", "cnt_model", "cnt_pair")}
                st["n_models"] = n_models; st["ncfg"] = ncfg
                bs.append(boot_metrics(st, M))
            boots[g][k] = {m: np.mean([b[m] for b in bs], axis=0) for m in METRICS}
            src = d009m["metrics"][cond][str(k)]
            res[g][k] = {m: src[m] for m in METRICS + ("train_acc", "val_acc")}
            for m in METRICS:
                for suf in ("_range", "_min", "_max"):
                    res[g][k][m + suf] = src.get(m + suf, float("nan"))
            res[g][k]["queries"] = sel["chains"][g][:k]
            res[g][k]["source"] = f"results/D009 {cond} (reused, I6)"

    # ---- S5: TOST for every gamma pair at every k
    comp = {}
    for i in range(len(ALL_GAMMAS)):
        for j in range(i + 1, len(ALL_GAMMAS)):
            a, b = ALL_GAMMAS[i], ALL_GAMMAS[j]
            for k in ks:
                if k not in boots.get(a, {}) or k not in boots.get(b, {}):
                    continue
                d = {}
                for m in METRICS:
                    p = paired_ci(boots[a][k][m], boots[b][k][m])
                    p["run_range"] = max(res[a][k].get(m + "_range", 0.0) or 0.0,
                                         res[b][k].get(m + "_range", 0.0) or 0.0)
                    p["exceeds_run_range"] = bool(abs(p["delta"]) > p["run_range"])
                    if m == "mean_top1":
                        p.update(tost(p))
                    d[m] = p
                comp[f"{a}_vs_{b}_k{k}"] = d

    # F4: does any sub-1 ordering hold consistently across k?
    sub1 = [g for g in ALL_GAMMAS if g != "1.0"]
    ordering = {}
    for i in range(len(sub1)):
        for j in range(i + 1, len(sub1)):
            a, b = sub1[i], sub1[j]
            ds = [comp[f"{a}_vs_{b}_k{k}"]["mean_top1"]["delta"]
                  for k in ks if f"{a}_vs_{b}_k{k}" in comp]
            ordering[f"{a}_vs_{b}"] = dict(
                deltas=[round(x, 4) for x in ds],
                consistent_sign=consistent_sign(ds),
                n_k=len(ds),
                mean_delta=round(float(np.mean(ds)), 4) if ds else None)

    # gamma=1.0 vs each sub-1: the headline
    head = {}
    for g in sub1:
        rows = []
        for k in ks:
            key = f"1.0_vs_{g}_k{k}"
            if key in comp:
                c = comp[key]["mean_top1"]
                rows.append(dict(k=k, delta=round(c["delta"], 4),
                                 lo=round(c["lo"], 4), hi=round(c["hi"], 4),
                                 excludes_zero=c["ci_excludes_zero"],
                                 exceeds_run_range=c["exceeds_run_range"]))
        head[g] = dict(by_k=rows,
                       n_excluding_zero=sum(r["excludes_zero"] for r in rows),
                       n_exceeding_range=sum(r["exceeds_run_range"] for r in rows),
                       n_k=len(rows))

    # ---- S6: cross-algorithm, now trained-to-trained
    d12 = json.load(open("./results/D012/metrics_by_gamma_k.json"))
    assert d12["hparams_hash"] == d009["hparams_hash"], "F2: hash mismatch"
    cross = dict(hparams_hash=d009["hparams_hash"],
                 hash_asserted_equal_across=["D009", "D012", "D013"],
                 note="paired only because all three share hparams hash, I2 split, "
                      "5 seeds and bootstrap draws (P1/F2)")
    shared = [g for g in ALL_GAMMAS if g in d12["metrics"]]
    for g in shared:
        rows = []
        for k in ks:
            if k not in res.get(g, {}) or str(k) not in d12["metrics"][g]:
                continue
            gc = res[g][k]["mean_top1"]
            jg = d12["metrics"][g][str(k)]["mean_top1"]
            rows.append(dict(k=k, greedycover=round(gc, 4), jointgreedy=round(jg, 4),
                             delta_jg_minus_gc=round(jg - gc, 4)))
        cross[f"gamma_{g}"] = rows
    # the gamma=1.0 penalty, both algorithms, side by side
    pen = []
    for k in ks:
        if f"1.0_vs_0.1_k{k}" not in comp or f"1.0_vs_0.1_k{k}" not in d12["comparisons"]:
            continue
        pen.append(dict(k=k,
                        greedycover=round(comp[f"1.0_vs_0.1_k{k}"]["mean_top1"]["delta"], 4),
                        jointgreedy=round(d12["comparisons"][f"1.0_vs_0.1_k{k}"]["mean_top1"]["delta"], 4)))
    cross["gamma1_penalty_by_algorithm"] = pen
    # F3: k=1 must agree -- one observation, not two
    if 1 in ks:
        k1 = {}
        for g in ("1.0", "0.1"):
            if 1 in res.get(g, {}) and str(1) in d12["metrics"].get(g, {}):
                a = res[g][1]["mean_top1"]; b = d12["metrics"][g]["1"]["mean_top1"]
                k1[g] = dict(greedycover=round(a, 4), jointgreedy=round(b, 4),
                             identical=bool(abs(a - b) < 1e-9),
                             query_gc=res[g][1]["queries"],
                             query_jg=d12["metrics"][g]["1"].get("queries"))
        cross["k1_identity_check"] = dict(
            per_gamma=k1,
            all_identical=all(v["identical"] for v in k1.values()) if k1 else None,
            note="F3: at k=1 the joint statistic reduces to the per-query one, so "
                 "the two algorithms ARE the same algorithm. These rows are ONE "
                 "observation appearing twice, not two agreeing observations -- "
                 "reported as an end-to-end consistency check.")

    band = {g: {k: {m: ci(boots[g][k][m]) for m in METRICS} for k in boots[g]}
            for g in boots}
    json.dump(dict(schema="d013-metrics-v1", gammas=ALL_GAMMAS, ks=ks,
                   n_runs_per_cell=N_RUNS, delta=DELTA,
                   chance=round(1 / n_models, 4),
                   hparams_hash=d009["hparams_hash"],
                   reused_from_d009=REUSED,
                   metrics=res, bootstrap_ci=band, comparisons=comp,
                   headline_gamma1_vs_sub1=head, sub1_ordering=ordering,
                   wall_min=round((time.time() - t_all) / 60, 1), runs=runs),
              open(f"{OUT}/metrics_by_gamma_k.json", "w"), indent=1)
    json.dump(cross, open(f"{OUT}/cross_algorithm_comparison.json", "w"), indent=1)
    json.dump(dict(schema="d013-traces-v1", traces=sel["traces"],
                   monotone=sel["monotone"],
                   jointgreedy_gamma1_trace=sel["jointgreedy_gamma1_trace"],
                   note="GreedyCover objective is MAX-aggregated (I4) and therefore "
                        "monotone in k at every gamma; JointGreedy's gamma=1.0 is "
                        "not (D012/F1)."),
              open(f"{OUT}/objective_traces.json", "w"), indent=1)

    print(f"\n===== S5: mean top-1 by gamma and k =====")
    print(f"{'k':>3s}" + "".join(f"{g:>10s}" for g in ALL_GAMMAS))
    for k in ks:
        print(f"{k:>3d}" + "".join(
            f"{res[g][k]['mean_top1']:10.4f}" if k in res.get(g, {}) else " " * 10
            for g in ALL_GAMMAS))
    print(f"\n===== headline: gamma=1.0 vs each sub-1 (mean top-1) =====")
    for g, h in head.items():
        print(f"  vs {g:5s}: CI excludes 0 on {h['n_excluding_zero']}/{h['n_k']} k, "
              f"exceeds seed range on {h['n_exceeding_range']}/{h['n_k']}")
    print(f"\n===== F4: consistent ordering among sub-1 gammas? =====")
    for k_, v in ordering.items():
        md = f"{v['mean_delta']:+.4f}" if v['mean_delta'] is not None else "  n/a "
        print(f"  {k_:14s} mean {md}  ({v['n_k']} k)  consistent sign: "
              f"{v['consistent_sign']}")
    print(f"\n===== S6: gamma=1.0 penalty, both algorithms =====")
    print(f"{'k':>3s} {'GreedyCover':>13s} {'JointGreedy':>13s}")
    for r in pen:
        print(f"{r['k']:>3d} {r['greedycover']:+13.4f} {r['jointgreedy']:+13.4f}")
    if "k1_identity_check" in cross:
        print(f"\n  F3 k=1 identity check: "
              f"{cross['k1_identity_check']['all_identical']}")
    print(f"\ntotal {(time.time()-t_all)/60:.1f} min")
    print(f"written: {OUT}/metrics_by_gamma_k.json, "
          f"{OUT}/cross_algorithm_comparison.json, {OUT}/objective_traces.json")


if __name__ == "__main__":
    main()
