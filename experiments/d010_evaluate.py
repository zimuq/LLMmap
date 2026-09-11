"""D010 / S5–S6 — evaluate the new condition(s) through D009's exact protocol,
then compare against D009's stored numbers.

D009's baselines are READ, never recomputed (I6). This script trains exactly the
conditions D010 adds:
    joint_energy   k=1..8, 5 seeds each   (the new selection criterion)
    joint_mmd      k=8,    5 seeds        (review amendment, robustness check)

Everything else -- hparams, split, trainer, seed scheme, the single touch of
S_test -- is D009's, and the hparams hash is asserted equal to D009's so
"identical training procedure" is enforced rather than intended.

Usage:  PYTHONPATH=.:experiments python experiments/d010_evaluate.py
"""
import os
import json
import time

import numpy as np

from d009_lib import (load_query_embeddings, build_traces, hparams_from_shipped,
                      logits_for, logit_stats)
from d009_train import run_one
from d008_lib import near_relative_pairs, boot_draws, boot_metrics, ci, paired_ci
from d007_lib import load_corpus
from d010_lib import OUT

METRICS = ("mean_top1", "worst_class", "worst3_class", "hard_subset")
N_RUNS = 5
SMOKE = bool(os.environ.get("D010_SMOKE"))
if SMOKE:
    N_RUNS = 2


def main():
    os.makedirs(f"{OUT}/models", exist_ok=True)
    sel = json.load(open(f"{OUT}/selection.json"))
    d009 = json.load(open("./results/D009/runs.json"))
    d009m = json.load(open("./results/D009/metrics_by_condition_k.json"))
    models = sel["models"]
    n_models = len(models)
    hard = near_relative_pairs(models)
    qe = load_query_embeddings()
    cubes = {p: load_corpus(pool=p) for p in ("build", "val", "test")}

    chain = sel["joint_energy"]["queries"]
    ks = [8] if SMOKE else list(range(1, 9))
    jobs = [("joint_energy", k, chain[:k]) for k in ks]
    jobs += [("joint_mmd", 8, sel["joint_mmd_k8"]["queries"])]

    runs, counts = [], {}
    t_all = time.time()
    for cond, k, queries in jobs:
        tr_b, y_b, _, _ = build_traces(queries, "build", qe, cubes["build"])
        tr_v, y_v, _, _ = build_traces(queries, "val", qe, cubes["val"])
        tr_t, y_t, c_t, _ = build_traces(queries, "test", qe, cubes["test"])
        hp, conf = hparams_from_shipped(k, n_models)
        conf = dict(conf); conf["inference_model"] = hp
        import hashlib
        h = hashlib.sha256(json.dumps(
            {kk: vv for kk, vv in hp.items() if kk != "num_queries"},
            sort_keys=True).encode()).hexdigest()
        assert h == d009["hparams_hash"], (
            f"hparams differ from D009's ({h[:12]} vs "
            f"{d009['hparams_hash'][:12]}) -- the comparison would not be "
            f"holding the training procedure constant")

        for r in range(N_RUNS):
            keep = f"{OUT}/models/{cond}_k{k}_r{r}.ckpt" if r == 0 else None
            res, cnt, buf = run_one(tr_b, y_b, tr_v, y_v, tr_t, y_t, c_t,
                                    hp, conf, r, hard, n_models, keep)
            res.update(condition=cond, k=k, run=r, seed=r, queries=queries)
            runs.append(res)
            if res["error"]:
                print(f"  !! {cond} k={k} r={r}: {res['error']}", flush=True)
            else:
                for key, arr in cnt.items():
                    if isinstance(arr, np.ndarray):
                        counts[f"{cond}|{k}|{r}|{key}"] = arr
        done = [x for x in runs if x["condition"] == cond and x["k"] == k
                and not x["error"]]
        m = np.array([x["mean_top1"] for x in done])
        print(f"[S5] {cond:13s} k={k} test top-1 {m.mean():.4f} "
              f"[{m.min():.4f},{m.max():.4f}]", flush=True)

    np.savez_compressed(f"{OUT}/run_counts.npz", **counts)

    # ---- aggregate, with both variance sources (D009's convention)
    ncfg = int(counts[list(counts)[0]].size) if counts else 25
    ncfg = 25
    M = boot_draws(ncfg, 2000)
    res, boots = {}, {}
    for cond, k, _ in jobs:
        rs = [x for x in runs if x["condition"] == cond and x["k"] == k
              and not x["error"]]
        if not rs:
            continue
        e = {}
        for m in METRICS + ("train_acc", "val_acc"):
            v = np.array([x[m] for x in rs], float)
            e[m] = float(v.mean()); e[m + "_min"] = float(v.min())
            e[m + "_max"] = float(v.max()); e[m + "_range"] = float(v.max() - v.min())
        e["n_runs"] = len(rs); e["queries"] = rs[0]["queries"]
        res.setdefault(cond, {})[k] = e
        bs = []
        for x in rs:
            st = {kk: counts[f"{cond}|{k}|{x['run']}|{kk}"]
                  for kk in ("cnt_total", "cnt_model", "cnt_pair")}
            st["n_models"] = n_models; st["ncfg"] = ncfg
            bs.append(boot_metrics(st, M))
        boots.setdefault(cond, {})[k] = {m: np.mean([b[m] for b in bs], axis=0)
                                         for m in METRICS}

    # ---- S6: against D009's stored numbers (read, not rerun)
    d009_boot = None   # D009 stored counts; reuse them for a paired comparison
    d009_counts = np.load("./results/D009/run_counts.npz")
    comp = {}
    for cond in res:
        for k in res[cond]:
            for base in ("cvar_max", "paper8", "mean_greedy_max", "random"):
                key = f"{base}|{k}|"
                runs_b = [n for n in d009_counts.files if n.startswith(key)
                          and n.endswith("cnt_total")]
                if not runs_b:
                    continue
                bb = []
                for n in runs_b:
                    r = n.split("|")[2]
                    st = {kk: d009_counts[f"{base}|{k}|{r}|{kk}"]
                          for kk in ("cnt_total", "cnt_model", "cnt_pair")}
                    st["n_models"] = n_models; st["ncfg"] = ncfg
                    bb.append(boot_metrics(st, M))
                bmean = {m: np.mean([b[m] for b in bb], axis=0) for m in METRICS}
                d = {}
                for m in METRICS:
                    p = paired_ci(boots[cond][k][m], bmean[m])
                    rng = max(res[cond][k][m + "_range"],
                              d009m["metrics"][base][str(k)].get(m + "_range", 0))
                    p["run_range"] = rng
                    p["resolved"] = bool(abs(p["delta"]) > rng)
                    p["baseline"] = d009m["metrics"][base][str(k)][m]
                    d[m] = p
                comp[f"{cond}_k{k}_vs_{base}"] = d

    band = {c: {k: {m: ci(boots[c][k][m]) for m in METRICS} for k in boots[c]}
            for c in boots}
    json.dump(dict(schema="d010-metrics-v1", n_runs_per_cell=N_RUNS,
                   chance=round(1 / n_models, 4),
                   hparams_hash=d009["hparams_hash"],
                   eval_split="test (touched once per run)",
                   metrics=res, bootstrap_ci=band, runs=runs),
              open(f"{OUT}/metrics_by_k.json", "w"), indent=1)
    json.dump(dict(schema="d010-comparison-v1", comparisons=comp,
                   d009_source="results/D009/metrics_by_condition_k.json",
                   note="D009's conditions are READ, never recomputed (I6); the "
                        "bootstrap uses D009's stored per-config counts with the "
                        "identical draws, so every comparison is paired"),
              open(f"{OUT}/comparison_vs_d009.json", "w"), indent=1)

    print(f"\n===== S6: joint_energy vs D009 =====")
    for k in sorted(res.get("joint_energy", {})):
        for base in ("cvar_max", "paper8"):
            c = comp.get(f"joint_energy_k{k}_vs_{base}")
            if not c:
                continue
            m = c["mean_top1"]
            print(f"  k={k} vs {base:15s} {m['delta']:+.4f} "
                  f"[{m['lo']:+.4f},{m['hi']:+.4f}] range {m['run_range']:.4f} "
                  f"{'RESOLVED' if m['resolved'] else 'not resolved'}")
    c = comp.get("joint_mmd_k8_vs_cvar_max")
    if c:
        print(f"  MMD k=8 vs cvar_max {c['mean_top1']['delta']:+.4f} "
              f"[{c['mean_top1']['lo']:+.4f},{c['mean_top1']['hi']:+.4f}]")
    print(f"\ntotal {(time.time()-t_all)/60:.1f} min")
    print(f"written: {OUT}/metrics_by_k.json, {OUT}/comparison_vs_d009.json")


if __name__ == "__main__":
    main()
