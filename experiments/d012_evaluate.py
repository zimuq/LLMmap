"""D012 / S5–S6 — train the gamma grid through D009's protocol, then report.

Grid: gamma in {1.0, 0.25, 0.05} x k=1..8 x 5 seeds = 120 runs. D010's gamma=0.1
is READ from results/D010/metrics_by_k.json, never rerun (I6).

Reporting follows the review's amendments:
  * outcome decided by a pre-registered TOST equivalence test (F4), delta = 0.02
    on mean top-1 -- smaller than the smallest effect this project has resolved.
    NULL requires the CI of the difference to sit ENTIRELY inside +/-delta, which
    is positive evidence of equivalence rather than absence of evidence.
  * at k=8, gamma in {0.25, 0.1, 0.05} select the IDENTICAL set differing only in
    slot order (P1/F2), so those runs additionally measure the trained network's
    ORDER sensitivity -- reported as its own line, not folded into a gamma trend.
  * no cross-algorithm comparison against D008's gamma sweep: that sweep is
    proxy-only (P1/F3). GreedyCover-vs-JointGreedy stays proxy-to-proxy.

Usage:  PYTHONPATH=.:experiments python experiments/d012_evaluate.py
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

OUT = "./results/D012"
METRICS = ("mean_top1", "worst_class", "worst3_class", "hard_subset")
DELTA = 0.02                 # F4's pre-registered equivalence margin
N_RUNS = 5
NEW_GAMMAS = ("1.0", "0.25", "0.05")
SMOKE = bool(os.environ.get("D012_SMOKE"))
if SMOKE:
    N_RUNS, NEW_GAMMAS = 2, ("1.0",)


def tost(d, delta=DELTA):
    """Pre-registered decision for one gamma-to-gamma comparison (F4)."""
    lo, hi = d["lo"], d["hi"]
    equivalent = (lo > -delta) and (hi < delta)
    different = (lo > 0) or (hi < 0)
    return dict(equivalent=bool(equivalent), ci_excludes_zero=bool(different),
                verdict=("NULL (equivalent)" if equivalent else
                         "different" if different else "INCONCLUSIVE"))


def main():
    os.makedirs(f"{OUT}/models", exist_ok=True)
    s2 = json.load(open(f"{OUT}/s2_check.json"))
    d009 = json.load(open("./results/D009/runs.json"))
    d010sel = json.load(open("./results/D010/selection.json"))
    d010m = json.load(open("./results/D010/metrics_by_k.json"))

    # the whole grid hangs off D010's reused point -- assert it (P1 verification)
    g01 = d010sel["joint_energy"]["queries"][:8]
    assert g01 == [193, 140, 237, 114, 233, 117, 0, 16], g01
    assert s2["chains"]["0.1"] == g01

    models = json.load(open("./results/D008/selection.json"))["models"]
    n_models = len(models)
    hard = near_relative_pairs(models)
    qe = load_query_embeddings()
    cubes = {p: load_corpus(pool=p) for p in ("build", "val", "test")}
    ks = [8] if SMOKE else list(range(1, 9))

    runs, counts = [], {}
    t_all = time.time()
    for g in NEW_GAMMAS:
        chain = s2["chains"][g]
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
            print(f"[S5] gamma={g:5s} k={k} test top-1 {m.mean():.4f} "
                  f"[{m.min():.4f},{m.max():.4f}]", flush=True)

    np.savez_compressed(f"{OUT}/run_counts.npz", **counts)

    # ---- aggregate the three new gammas; read gamma=0.1 from D010
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

    # gamma=0.1: D010's stored per-config counts, same draws -> paired
    d10c = np.load("./results/D010/run_counts.npz")
    res["0.1"], boots["0.1"] = {}, {}
    for k in ks:
        pref = f"joint_energy|{k}|"
        rs = sorted({n.split("|")[2] for n in d10c.files if n.startswith(pref)})
        if not rs:
            continue
        bs = []
        for r in rs:
            st = {kk: d10c[f"joint_energy|{k}|{r}|{kk}"]
                  for kk in ("cnt_total", "cnt_model", "cnt_pair")}
            st["n_models"] = n_models; st["ncfg"] = ncfg
            bs.append(boot_metrics(st, M))
        boots["0.1"][k] = {m: np.mean([b[m] for b in bs], axis=0) for m in METRICS}
        src = d010m["metrics"]["joint_energy"][str(k)]
        res["0.1"][k] = {m: src[m] for m in METRICS + ("train_acc", "val_acc")}
        for m in METRICS:
            res["0.1"][k][m + "_range"] = src.get(m + "_range", 0.0)
            res["0.1"][k][m + "_min"] = src.get(m + "_min", float("nan"))
            res["0.1"][k][m + "_max"] = src.get(m + "_max", float("nan"))
        res["0.1"][k]["queries"] = g01[:k]
        res["0.1"][k]["source"] = "results/D010 (reused, I6)"

    # ---- F4's pre-registered decisions, every gamma pair at every k
    gs = ["1.0", "0.25", "0.1", "0.05"]
    comp = {}
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            a, b = gs[i], gs[j]
            for k in ks:
                if k not in boots.get(a, {}) or k not in boots.get(b, {}):
                    continue
                d = {}
                for m in METRICS:
                    p = paired_ci(boots[a][k][m], boots[b][k][m])
                    p["run_range"] = max(res[a][k].get(m + "_range", 0.0),
                                         res[b][k].get(m + "_range", 0.0))
                    p["exceeds_run_range"] = bool(abs(p["delta"]) > p["run_range"])
                    if m == "mean_top1":
                        p.update(tost(p))
                    d[m] = p
                comp[f"{a}_vs_{b}_k{k}"] = d

    # ---- F2: order sensitivity. At k=8, gammas 0.25/0.1/0.05 are the SAME SET.
    order = None
    if 8 in ks:
        same = [g for g in ("0.25", "0.1", "0.05") if 8 in res.get(g, {})]
        sets = {g: sorted(res[g][8]["queries"]) for g in same}
        identical = len({tuple(v) for v in sets.values()}) == 1
        vals = {g: res[g][8]["mean_top1"] for g in same}
        spread = max(vals.values()) - min(vals.values()) if vals else 0.0
        within = float(np.mean([res[g][8].get("mean_top1_range", 0.0)
                                for g in same]))
        order = dict(
            gammas=same, identical_set=bool(identical), set=sets.get(same[0]),
            mean_top1_by_gamma={g: round(v, 4) for g, v in vals.items()},
            across_order_spread=round(spread, 4),
            within_condition_seed_range=round(within, 4),
            order_matters=bool(spread > within),
            note="same 8 queries, different slot order; build_traces assigns slots "
                 "positionally, so this isolates the trained network's ORDER "
                 "sensitivity (P1/F2, review-approved)")

    band = {g: {k: {m: ci(boots[g][k][m]) for m in METRICS} for k in boots[g]}
            for g in boots}
    json.dump(dict(schema="d012-metrics-v1", gammas=gs, ks=ks,
                   n_runs_per_cell=N_RUNS, delta=DELTA,
                   chance=round(1 / n_models, 4),
                   hparams_hash=d009["hparams_hash"],
                   eval_split="test (touched once per run)",
                   metrics=res, bootstrap_ci=band, comparisons=comp,
                   order_sensitivity=order,
                   gamma_0_1_source="results/D010/metrics_by_k.json (I6, reused)",
                   wall_min=round((time.time() - t_all) / 60, 1), runs=runs),
              open(f"{OUT}/metrics_by_gamma_k.json", "w"), indent=1)
    json.dump(dict(schema="d012-traces-v1",
                   objective_traces={g: s2["selections"][g]["trace"]
                                     for g in s2["selections"]},
                   gamma_0_1_trace=d010sel["joint_energy"]["trace"],
                   peak_k={**{g: v["peak_k"] for g, v in s2["selections"].items()},
                           "0.1": d010sel["joint_energy"]["peak_k"]},
                   mechanism=s2["mechanism"], chains=s2["chains"],
                   chain_overlap_k8=s2["chain_overlap_k8"]),
              open(f"{OUT}/objective_traces.json", "w"), indent=1)

    print(f"\n===== S6: mean top-1 by gamma and k (5 runs, [min,max]) =====")
    print(f"{'k':>3s}" + "".join(f"{g:>24s}" for g in gs))
    for k in ks:
        row = f"{k:>3d}"
        for g in gs:
            e = res.get(g, {}).get(k)
            row += (f"{e['mean_top1']:.4f} [{e.get('mean_top1_min', float('nan')):.3f},"
                    f"{e.get('mean_top1_max', float('nan')):.3f}]".rjust(24)
                    if e else " " * 24)
        print(row)
    print(f"\n===== F4 equivalence test, mean top-1 (delta={DELTA}) =====")
    for key in sorted(comp):
        c = comp[key]["mean_top1"]
        print(f"  {key:22s} {c['delta']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}] "
              f"-> {c['verdict']}")
    if order:
        print(f"\n===== F2 order sensitivity (k=8, identical set) =====")
        print(f"  set identical across gammas: {order['identical_set']}")
        print(f"  mean top-1 by gamma: {order['mean_top1_by_gamma']}")
        print(f"  across-order spread {order['across_order_spread']:.4f} vs "
              f"within-condition seed range {order['within_condition_seed_range']:.4f}"
              f"  -> order matters: {order['order_matters']}")
    print(f"\ntotal {(time.time()-t_all)/60:.1f} min")
    print(f"written: {OUT}/metrics_by_gamma_k.json, {OUT}/objective_traces.json")


if __name__ == "__main__":
    main()
