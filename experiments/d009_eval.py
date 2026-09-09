"""D009 / S5–S6 — aggregate the grid, and compare trained vs D008's proxy.

S5 uses D008's metric definitions and D008's bootstrap verbatim (same 25 test
configs, same 2,000 paired draws) so the two D's numbers are directly
comparable rather than merely similar-looking.

Two variance sources are reported side by side and never merged:
  * bootstrap CI over eval CONFIGS -- sampling noise in the test set
  * range over the 5 RUNS          -- initialisation / draw noise (P1/F1)
A delta smaller than the run range is reported as NOT RESOLVED, per P1's
pre-registered rule.

Usage:  PYTHONPATH=.:experiments python experiments/d009_eval.py
"""
import os
import json
import itertools

import numpy as np

from d008_lib import boot_draws, boot_metrics, ci, paired_ci
from d009_lib import OUT

METRICS = ("mean_top1", "worst_class", "worst3_class", "hard_subset")
TARGETS = [0.5, 0.7, 0.8, 0.9]
CONDS = ("paper8", "mean_greedy_max", "cvar_max", "random")


def q2r(curve, target):
    for k in sorted(curve):
        if curve[k] >= target:
            return int(k)
    return None


def main():
    runs = json.load(open(f"{OUT}/runs.json"))
    cnt = np.load(f"{OUT}/run_counts.npz")
    ok = [r for r in runs["runs"] if not r["error"]]
    ks = sorted({r["k"] for r in ok})
    ncfg = int(cnt[f"{ok[0]['condition']}|{ok[0]['k']}|{ok[0]['run']}|cnt_total"].size)
    M = boot_draws(ncfg, 2000)

    # ---- S5: point metrics (mean over runs) + both variance sources
    res, boots = {}, {}
    for c in CONDS:
        res[c], boots[c] = {}, {}
        for k in ks:
            rs = [r for r in ok if r["condition"] == c and r["k"] == k]
            if not rs:
                continue
            e = {}
            for m in METRICS + ("hard_subset_min", "train_acc", "val_acc"):
                v = np.array([r[m] for r in rs], float)
                e[m] = float(v.mean())
                e[m + "_min"], e[m + "_max"] = float(v.min()), float(v.max())
                e[m + "_range"] = float(v.max() - v.min())
            e["n_runs"] = len(rs)
            e["queries"] = rs[0]["queries"] if c != "random" else None
            e["overfit_gap"] = e["train_acc"] - e["val_acc"]
            res[c][k] = e
            # bootstrap on the FIRST run's counts, then averaged across runs so
            # the CI reflects config sampling at a typical run, not one draw
            bs = []
            for r in rs:
                st = {kk: cnt[f"{c}|{k}|{r['run']}|{kk}"]
                      for kk in ("cnt_total", "cnt_model", "cnt_pair")}
                st["n_models"] = len(runs["models"]); st["ncfg"] = ncfg
                bs.append(boot_metrics(st, M))
            boots[c][k] = {m: np.mean([b[m] for b in bs], axis=0) for m in METRICS}

    band = {c: {k: {m: ci(boots[c][k][m]) for m in METRICS} for k in boots[c]}
            for c in boots}

    # ---- comparisons, paired on the identical bootstrap draws
    comp = {}
    for a, b in (("cvar_max", "mean_greedy_max"), ("cvar_max", "paper8"),
                 ("mean_greedy_max", "paper8"), ("cvar_max", "random")):
        for k in ks:
            if k not in boots[a] or k not in boots[b]:
                continue
            d = {}
            for m in METRICS:
                p = paired_ci(boots[a][k][m], boots[b][k][m])
                rng = max(res[a][k][m + "_range"], res[b][k][m + "_range"])
                p["run_range"] = rng
                p["resolved"] = bool(abs(p["delta"]) > rng)
                d[m] = p
            comp[f"{a}_vs_{b}_k{k}"] = d

    q = {c: {m: {str(t): (q2r({k: res[c][k][m] for k in res[c]}, t) or f">{max(ks)}")
                 for t in TARGETS} for m in ("mean_top1", "worst_class")}
         for c in CONDS}

    # ---- S6: trained vs D008's proxy
    proxy = json.load(open("./results/D008/metrics_by_condition_k.json"))
    s6 = {}
    for c in CONDS:
        rows, tv, pv = {}, [], []
        for k in ks:
            if k not in res[c] or str(k) not in proxy["metrics"].get(c, {}):
                continue
            rows[k] = {m: dict(trained=res[c][k][m],
                               proxy=proxy["metrics"][c][str(k)][m],
                               delta=res[c][k][m] - proxy["metrics"][c][str(k)][m])
                       for m in METRICS}
            tv.append(res[c][k]["mean_top1"])
            pv.append(proxy["metrics"][c][str(k)]["mean_top1"])
        from scipy.stats import spearmanr
        s6[c] = dict(by_k=rows,
                     spearman_k_curve=float(spearmanr(tv, pv).statistic)
                     if len(tv) > 2 else None,
                     mean_delta=float(np.mean([r["mean_top1"]["delta"]
                                               for r in rows.values()])))
    # does the CVaR-over-mean-greedy gap keep its sign, trained vs proxy?
    sign = {}
    for k in ks:
        if k not in res["cvar_max"] or str(k) not in proxy["metrics"]["cvar_max"]:
            continue
        for m in METRICS:
            t = res["cvar_max"][k][m] - res["mean_greedy_max"][k][m]
            p = (proxy["metrics"]["cvar_max"][str(k)][m]
                 - proxy["metrics"]["mean_greedy_max"][str(k)][m])
            sign.setdefault(m, {})[k] = dict(trained=t, proxy=p,
                                             same_sign=bool(t * p > 0))

    json.dump(dict(schema="d009-metrics-v1", ks=ks, conditions=list(CONDS),
                   n_runs_per_cell=runs["n_runs_per_cell"],
                   chance=round(1 / len(runs["models"]), 4),
                   eval_split="test (touched once per run)",
                   hparams_hash=runs["hparams_hash"],
                   hparams_source=runs["hparams_source"],
                   metrics=res, bootstrap_ci=band, comparisons=comp,
                   queries_to_reach=q),
              open(f"{OUT}/metrics_by_condition_k.json", "w"), indent=1)
    json.dump(dict(schema="d009-proxy-comparison-v1", per_condition=s6,
                   cvar_minus_mean_greedy_sign=sign,
                   note="proxy = D008's 1-NN over frozen I5 embeddings; trained "
                        "= LLmap's own stage-2 network. Same split, same metric "
                        "definitions, same bootstrap."),
              open(f"{OUT}/proxy_vs_trained_comparison.json", "w"), indent=1)

    print(f"===== S5: test top-1, mean over {runs['n_runs_per_cell']} runs "
          f"[min,max] =====")
    print(f"{'k':>3s} " + "".join(f"{c:>26s}" for c in CONDS))
    for k in ks:
        row = f"{k:>3d} "
        for c in CONDS:
            e = res[c].get(k)
            row += (f"{e['mean_top1']:.4f} [{e['mean_top1_min']:.3f},"
                    f"{e['mean_top1_max']:.3f}]".rjust(26)) if e else " " * 26
        print(row)
    print(f"\n===== CVaR vs mean-greedy (trained) =====")
    for k in ks:
        c = comp.get(f"cvar_max_vs_mean_greedy_max_k{k}")
        if not c:
            continue
        print(f"  k={k}: mean {c['mean_top1']['delta']:+.4f} "
              f"[{c['mean_top1']['lo']:+.4f},{c['mean_top1']['hi']:+.4f}] "
              f"run-range {c['mean_top1']['run_range']:.4f} "
              f"{'RESOLVED' if c['mean_top1']['resolved'] else 'not resolved'}"
              f" | worst {c['worst_class']['delta']:+.4f} "
              f"{'R' if c['worst_class']['resolved'] else '-'}"
              f" | hard {c['hard_subset']['delta']:+.4f} "
              f"{'R' if c['hard_subset']['resolved'] else '-'}")
    print(f"\n===== S6: trained vs proxy (mean top-1) =====")
    for c in CONDS:
        print(f"  {c:16s} mean delta {s6[c]['mean_delta']:+.4f}  "
              f"k-curve Spearman {s6[c]['spearman_k_curve']}")
    print("\n  CVaR - mean-greedy, sign agreement trained vs proxy:")
    for m in METRICS:
        agree = sum(1 for k in sign[m] if sign[m][k]["same_sign"])
        print(f"    {m:14s} {agree}/{len(sign[m])} k values agree in sign")
    print(f"\n===== overfit check (S4) =====")
    for c in CONDS:
        k = max(res[c]) if res[c] else None
        if k:
            print(f"  {c:16s} k={k}: train {res[c][k]['train_acc']:.4f} "
                  f"val {res[c][k]['val_acc']:.4f} "
                  f"gap {res[c][k]['overfit_gap']:+.4f}")
    print(f"\nwritten: {OUT}/metrics_by_condition_k.json, "
          f"{OUT}/proxy_vs_trained_comparison.json")


if __name__ == "__main__":
    main()
