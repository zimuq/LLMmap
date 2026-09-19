"""D014 — re-derive Arm A's labels with the corrected reuse map. No retraining.

The first Arm A pass left JointGreedy gamma=0.1 blank. Cause: D012 itself READ
that point from D010 (I6) and therefore never stored per-config counts for it, so
`results/D012/run_counts.npz` has prefixes {1.0, 0.25, 0.05} only. The counts live
in `results/D010/run_counts.npz` under `joint_energy`. The reuse map pointed at
the wrong file for that one cell.

Everything trained in the first pass is already on disk, so this recomputes the
bootstrap and the Call-1 labels from saved counts rather than repeating 38.9
minutes of training.

Usage:  PYTHONPATH=.:experiments python experiments/d014_relabel.py
"""
import json

import numpy as np

from d008_lib import boot_draws, boot_metrics, ci, paired_ci
from d014_evaluate import METRICS, DELTA, GRID, tost, label

OUT = "./results/D014"
# corrected: JointGreedy gamma=0.1 lives in D010, not D012
SRC = {
    "GreedyCover": {
        "1.0": ("./results/D009/metrics_by_condition_k.json",
                "./results/D009/run_counts.npz", "mean_greedy_max"),
        "0.1": ("./results/D009/metrics_by_condition_k.json",
                "./results/D009/run_counts.npz", "cvar_max"),
        "0.5": ("./results/D013/metrics_by_gamma_k.json",
                "./results/D013/run_counts.npz", "0.5"),
        "0.25": ("./results/D013/metrics_by_gamma_k.json",
                 "./results/D013/run_counts.npz", "0.25"),
        "0.05": ("./results/D013/metrics_by_gamma_k.json",
                 "./results/D013/run_counts.npz", "0.05"),
    },
    "JointGreedy": {
        "1.0": ("./results/D012/metrics_by_gamma_k.json",
                "./results/D012/run_counts.npz", "1.0"),
        "0.25": ("./results/D012/metrics_by_gamma_k.json",
                 "./results/D012/run_counts.npz", "0.25"),
        "0.05": ("./results/D012/metrics_by_gamma_k.json",
                 "./results/D012/run_counts.npz", "0.05"),
        "0.1": ("./results/D010/metrics_by_k.json",
                "./results/D010/run_counts.npz", "joint_energy"),
    },
}
KS = list(range(1, 9))


def main():
    a = json.load(open(f"{OUT}/metrics_by_gamma_k.json"))
    own = np.load(f"{OUT}/run_counts.npz")
    n_models = 37
    ncfg = 25
    M = boot_draws(ncfg, 2000)

    res, boots = {}, {}
    for alg in ("GreedyCover", "JointGreedy"):
        res[alg], boots[alg] = {}, {}
        for g in GRID:
            res[alg][g], boots[alg][g] = {}, {}
            if g in SRC[alg]:
                mfile, cfile, key = SRC[alg][g]
                m = json.load(open(mfile))
                cnts = np.load(cfile)
                src = m["metrics"][key]
                for k in KS:
                    rs = sorted({n.split("|")[2] for n in cnts.files
                                 if n.startswith(f"{key}|{k}|")})
                    if not rs or str(k) not in src:
                        continue
                    bs = []
                    for r in rs:
                        st = {kk: cnts[f"{key}|{k}|{r}|{kk}"]
                              for kk in ("cnt_total", "cnt_model", "cnt_pair")}
                        st["n_models"] = n_models; st["ncfg"] = ncfg
                        bs.append(boot_metrics(st, M))
                    boots[alg][g][k] = {mm: np.mean([b[mm] for b in bs], axis=0)
                                        for mm in METRICS}
                    e = {mm: src[str(k)][mm] for mm in METRICS}
                    for mm in METRICS:
                        e[mm + "_range"] = src[str(k)].get(mm + "_range", 0.0) or 0.0
                    e["source"] = f"{mfile} [{key}] (reused, I6)"
                    res[alg][g][k] = e
            else:
                runs = [x for x in a["runs"] if x["alg"] == alg
                        and x["gamma"] == g and not x["error"]]
                for k in KS:
                    done = [x for x in runs if x["k"] == k]
                    if not done:
                        continue
                    e = {}
                    for mm in METRICS + ("train_acc", "val_acc"):
                        v = np.array([x[mm] for x in done], float)
                        e[mm] = float(v.mean())
                        e[mm + "_range"] = float(v.max() - v.min())
                    e["queries"] = done[0]["queries"]
                    res[alg][g][k] = e
                    bs = []
                    for x in done:
                        st = {kk: own[f"{alg}|{g}|{k}|{x['run']}|{kk}"]
                              for kk in ("cnt_total", "cnt_model", "cnt_pair")}
                        st["n_models"] = n_models; st["ncfg"] = ncfg
                        bs.append(boot_metrics(st, M))
                    boots[alg][g][k] = {mm: np.mean([b[mm] for b in bs], axis=0)
                                        for mm in METRICS}

    labels, detail = {}, {}
    for alg in res:
        labels[alg], detail[alg] = {}, {}
        for g in GRID:
            if g == "1.0":
                continue
            per_k = []
            for k in KS:
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

    band = {alg: {g: {k: {mm: ci(boots[alg][g][k][mm]) for mm in METRICS}
                      for k in boots[alg][g]} for g in boots[alg]} for alg in boots}
    a.update(metrics=res, labels=labels, per_k=detail, bootstrap_ci=band,
             reuse_map_corrected=SRC,
             relabel_note="JointGreedy gamma=0.1's counts live in D010, not D012 "
                          "(D012 reused that point under I6 and stored none). "
                          "Labels re-derived from saved runs; nothing retrained.")
    json.dump(a, open(f"{OUT}/metrics_by_gamma_k.json", "w"), indent=1)

    print(f"===== Arm A: mean top-1 at k=8 =====")
    print(f"{'gamma':>6s} {'GreedyCover':>13s} {'JointGreedy':>13s}")
    for g in GRID:
        row = f"{g:>6s}"
        for alg in ("GreedyCover", "JointGreedy"):
            v = res[alg].get(g, {}).get(8, {}).get("mean_top1")
            row += f"{v:13.4f}" if v is not None else f"{'--':>13s}"
        print(row)
    print(f"\n===== Arm A labels vs gamma=1.0 (DESCRIPTIVE) =====")
    print(f"{'gamma':>6s} {'GreedyCover':>26s} {'JointGreedy':>26s}")
    for g in GRID:
        if g == "1.0":
            continue
        row = f"{g:>6s}"
        for alg in ("GreedyCover", "JointGreedy"):
            L = labels[alg].get(g)
            s_ = (f"{L['label']}({L['n_better']}/{L['n_k']})"
                  + (" HARM" if L["harm"] else "")) if L and L["n_k"] else "--"
            row += f"{s_:>26s}"
        print(row)
    print(f"\nwritten: {OUT}/metrics_by_gamma_k.json")


if __name__ == "__main__":
    main()
