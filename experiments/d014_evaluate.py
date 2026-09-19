"""D014 / S2 (Arm A) — fine-grid trained sweep, both algorithms.

Trains every DISTINCT new chain from S1 at k=1..8 x 5 seeds, through D012/D013's
exact protocol. Existing grid points are READ, never rerun (I6):
    GreedyCover  gamma in {1.0, 0.5, 0.25, 0.1, 0.05}   <- D009 + D013
    JointGreedy  gamma in {1.0, 0.25, 0.1, 0.05}        <- D012

Labelling rule, fixed in P1/Call 1 BEFORE any run and approved unchanged:
  per k -- BETTER  = bootstrap CI excludes 0 AND |delta| exceeds the 5-seed range
           EQUIV   = TOST at delta=0.02
           else UNRESOLVED
  over k=1..8 -- BENEFIT    = better at >=5 of 8 and never worse anywhere
                 NO-BENEFIT = better at <=1 of 8
                 else UNRESOLVED
  plus HARM (review amendment): worse than gamma=1.0 at >=3 of 8 k, flagged
  beside the label so a label cannot hide a negative effect.

ARM A IS DESCRIPTIVE. Per the review's amendment, S1 showed chains recur as
FAMILIES across neighbouring gamma, so a bad family produces exactly the same
label pattern as a real threshold. Arm A's labels never decide H1 vs H2 --
d014_stability.py does.

Usage:  PYTHONPATH=.:experiments python experiments/d014_evaluate.py
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

OUT = "./results/D014"
METRICS = ("mean_top1", "worst_class", "worst3_class", "hard_subset")
DELTA = 0.02
N_RUNS = 5
GRID = ["1.0", "0.75", "0.6", "0.5", "0.45", "0.4", "0.35", "0.3", "0.25",
        "0.1", "0.05"]
# where each already-trained point lives (I6: read, never rerun)
REUSE = {
    "GreedyCover": {"1.0": ("D009", "mean_greedy_max"), "0.1": ("D009", "cvar_max"),
                    "0.5": ("D013", "0.5"), "0.25": ("D013", "0.25"),
                    "0.05": ("D013", "0.05")},
    "JointGreedy": {"1.0": ("D012", "1.0"), "0.25": ("D012", "0.25"),
                    "0.1": ("D012", "0.1"), "0.05": ("D012", "0.05")},
}
SMOKE = bool(os.environ.get("D014_SMOKE"))
if SMOKE:
    N_RUNS = 2
    GRID = ["1.0", "0.6", "0.5", "0.45", "0.25"]


def tost(d, delta=DELTA):
    lo, hi = d["lo"], d["hi"]
    return dict(equivalent=bool(lo > -delta and hi < delta),
                ci_excludes_zero=bool(lo > 0 or hi < 0))


def label(per_k):
    """P1/Call 1's rule, applied to the list of per-k dicts vs gamma=1.0."""
    better = sum(1 for p in per_k if p["better"])
    worse = sum(1 for p in per_k if p["worse"])
    n = len(per_k)
    if better >= max(1, int(np.ceil(5 * n / 8))) and worse == 0:
        lab = "BENEFIT"
    elif better <= int(np.floor(1 * n / 8)):
        lab = "NO-BENEFIT"
    else:
        lab = "UNRESOLVED"
    return dict(label=lab, n_better=better, n_worse=worse, n_k=n,
                harm=bool(worse >= max(1, int(np.ceil(3 * n / 8)))))


def load_reused(alg, g, ks, n_models, M, ncfg):
    """Read an already-trained grid point plus its stored per-config counts."""
    where, key = REUSE[alg][g]
    if where == "D009":
        m = json.load(open("./results/D009/metrics_by_condition_k.json"))
        cnts = np.load("./results/D009/run_counts.npz")
        src, pref = m["metrics"][key], key
    else:
        m = json.load(open(f"./results/{where}/metrics_by_gamma_k.json"))
        cnts = np.load(f"./results/{where}/run_counts.npz")
        src, pref = m["metrics"][key], key
    res, bo = {}, {}
    for k in ks:
        if str(k) not in src:
            continue
        rs = sorted({n.split("|")[2] for n in cnts.files
                     if n.startswith(f"{pref}|{k}|")})
        if not rs:
            continue
        bs = []
        for r in rs:
            st = {kk: cnts[f"{pref}|{k}|{r}|{kk}"]
                  for kk in ("cnt_total", "cnt_model", "cnt_pair")}
            st["n_models"] = n_models; st["ncfg"] = ncfg
            bs.append(boot_metrics(st, M))
        bo[k] = {mm: np.mean([b[mm] for b in bs], axis=0) for mm in METRICS}
        e = {mm: src[str(k)][mm] for mm in METRICS}
        for mm in METRICS:
            e[mm + "_range"] = src[str(k)].get(mm + "_range", 0.0) or 0.0
        e["source"] = f"results/{where} {key} (reused, I6)"
        e["queries"] = src[str(k)].get("queries")
        res[k] = e
    return res, bo


def main():
    os.makedirs(f"{OUT}/models", exist_ok=True)
    chains = json.load(open(f"{OUT}/chain_analysis.json"))
    d009 = json.load(open("./results/D009/runs.json"))
    models = json.load(open("./results/D008/selection.json"))["models"]
    n_models = len(models)
    hard = near_relative_pairs(models)
    qe = load_query_embeddings()
    cubes = {p: load_corpus(pool=p) for p in ("build", "val", "test")}
    ks = [1, 4, 8] if SMOKE else list(range(1, 9))
    ncfg = 25
    M = boot_draws(ncfg, 2000)

    ALG = {"GreedyCover": chains["greedycover"]["chains"],
           "JointGreedy": chains["jointgreedy"]["chains"]}

    runs, counts, res, boots = [], {}, {}, {}
    t_all = time.time()
    for alg, ch in ALG.items():
        res[alg], boots[alg] = {}, {}
        for g in GRID:
            if g in REUSE[alg]:
                r, b = load_reused(alg, g, ks, n_models, M, ncfg)
                # the reused chain must equal the one S1 reselected (I6 check)
                for k in r:
                    if r[k].get("queries"):
                        assert r[k]["queries"] == ch[g][:k], (alg, g, k)
                res[alg][g], boots[alg][g] = r, b
                print(f"[S2] {alg:12s} g={g:5s} REUSED from "
                      f"{REUSE[alg][g][0]}", flush=True)
                continue
            res[alg][g], boots[alg][g] = {}, {}
            for k in ks:
                queries = ch[g][:k]
                tr_b, y_b, _, _ = build_traces(queries, "build", qe, cubes["build"])
                tr_v, y_v, _, _ = build_traces(queries, "val", qe, cubes["val"])
                tr_t, y_t, c_t, _ = build_traces(queries, "test", qe, cubes["test"])
                hp, conf = hparams_from_shipped(k, n_models)
                conf = dict(conf); conf["inference_model"] = hp
                h = hashlib.sha256(json.dumps(
                    {kk: vv for kk, vv in hp.items() if kk != "num_queries"},
                    sort_keys=True).encode()).hexdigest()
                assert h == d009["hparams_hash"], (h, d009["hparams_hash"])
                for r_ in range(N_RUNS):
                    keep = (f"{OUT}/models/{alg}_g{g}_k{k}_r{r_}.ckpt"
                            if r_ == 0 and k == 8 else None)
                    out, cnt, _ = run_one(tr_b, y_b, tr_v, y_v, tr_t, y_t, c_t,
                                          hp, conf, r_, hard, n_models, keep)
                    out.update(alg=alg, gamma=g, k=k, run=r_, queries=queries)
                    runs.append(out)
                    if out["error"]:
                        print(f"  !! {alg} g={g} k={k} r={r_}: {out['error']}",
                              flush=True)
                    else:
                        for kk, arr in cnt.items():
                            if isinstance(arr, np.ndarray):
                                counts[f"{alg}|{g}|{k}|{r_}|{kk}"] = arr
                done = [x for x in runs if x["alg"] == alg and x["gamma"] == g
                        and x["k"] == k and not x["error"]]
                e = {}
                for mm in METRICS + ("train_acc", "val_acc"):
                    v = np.array([x[mm] for x in done], float)
                    e[mm] = float(v.mean()); e[mm + "_range"] = float(v.max() - v.min())
                e["queries"] = queries
                res[alg][g][k] = e
                bs = []
                for x in done:
                    st = {kk: counts[f"{alg}|{g}|{k}|{x['run']}|{kk}"]
                          for kk in ("cnt_total", "cnt_model", "cnt_pair")}
                    st["n_models"] = n_models; st["ncfg"] = ncfg
                    bs.append(boot_metrics(st, M))
                boots[alg][g][k] = {mm: np.mean([b[mm] for b in bs], axis=0)
                                    for mm in METRICS}
            print(f"[S2] {alg:12s} g={g:5s} mean top-1 k=8 "
                  f"{res[alg][g][max(ks)]['mean_top1']:.4f}", flush=True)

    np.savez_compressed(f"{OUT}/run_counts.npz", **counts)

    # ---- Call 1's labels, vs gamma=1.0, per algorithm
    labels, detail = {}, {}
    for alg in ALG:
        labels[alg], detail[alg] = {}, {}
        for g in GRID:
            if g == "1.0":
                continue
            per_k = []
            for k in ks:
                if k not in boots[alg].get(g, {}) or k not in boots[alg]["1.0"]:
                    continue
                p = paired_ci(boots[alg][g][k]["mean_top1"],
                              boots[alg]["1.0"][k]["mean_top1"])
                p.update(tost(p))
                rr = max(res[alg][g][k].get("mean_top1_range", 0.0) or 0.0,
                         res[alg]["1.0"][k].get("mean_top1_range", 0.0) or 0.0)
                p["run_range"] = rr
                p["better"] = bool(p["lo"] > 0 and abs(p["delta"]) > rr)
                p["worse"] = bool(p["hi"] < 0 and abs(p["delta"]) > rr)
                p["k"] = k
                per_k.append(p)
            detail[alg][g] = per_k
            labels[alg][g] = label(per_k)

    json.dump(dict(schema="d014-armA-v1", grid=GRID, ks=ks, delta=DELTA,
                   n_runs_per_cell=N_RUNS,
                   hparams_hash=d009["hparams_hash"], reuse_map=REUSE,
                   labelling_rule="P1/Call 1, fixed before training: per-k better "
                                  "= CI excludes 0 AND |d| > seed range; BENEFIT "
                                  ">=5/8 better and never worse; NO-BENEFIT <=1/8; "
                                  "HARM flag = worse at >=3/8 (review amendment)",
                   arm_a_is_descriptive="Per the 2026-09-19 review amendment, S1's "
                                        "chain families make Arm A unable to "
                                        "distinguish H1 from H2. Arm B decides.",
                   metrics=res, labels=labels, per_k=detail,
                   wall_min=round((time.time() - t_all) / 60, 1), runs=runs),
              open(f"{OUT}/metrics_by_gamma_k.json", "w"), indent=1)

    print(f"\n===== Arm A: mean top-1 at k={max(ks)} =====")
    print(f"{'gamma':>6s} {'GreedyCover':>13s} {'JointGreedy':>13s}")
    for g in GRID:
        row = f"{g:>6s}"
        for alg in ALG:
            v = res[alg].get(g, {}).get(max(ks), {}).get("mean_top1")
            row += f"{v:13.4f}" if v is not None else " " * 13
        print(row)
    print(f"\n===== Arm A labels vs gamma=1.0 (DESCRIPTIVE -- Arm B decides) =====")
    print(f"{'gamma':>6s} {'GreedyCover':>26s} {'JointGreedy':>26s}")
    for g in GRID:
        if g == "1.0":
            continue
        row = f"{g:>6s}"
        for alg in ALG:
            L = labels[alg].get(g)
            s = (f"{L['label']}({L['n_better']}/{L['n_k']})"
                 + (" HARM" if L["harm"] else "")) if L else ""
            row += f"{s:>26s}"
        print(row)
    print(f"\ntotal {(time.time()-t_all)/60:.1f} min")
    print(f"written: {OUT}/metrics_by_gamma_k.json")


if __name__ == "__main__":
    main()
