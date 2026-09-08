"""D008 / S5 — the gamma sweep (T2.2).

gamma in {1.0, 0.5, 0.25, 0.1, 0.05}, MAX aggregation, reported as a
mean-accuracy vs worst-class-accuracy trade-off curve. The point of the sweep,
per `TODO.md`/`METHOD.md §5.2`, is to show CVaR-coverage **contains** mean-greedy
as its gamma=1.0 special case rather than competing with it.

Also reports the chain overlap between adjacent gamma values. If the selected
sets barely move, the trade-off curve is flat by construction, and that mechanism
should be visible rather than inferred from the curve's shape.

Usage:  PYTHONPATH=.:experiments python experiments/d008_gamma.py
"""
import os
import json

import numpy as np

from d008_lib import (build_dq, summed, near_relative_pairs, condition_stats,
                      boot_draws, boot_metrics, ci, OUT)

K_HEAD = 8


def main():
    sel = json.load(open(f"{OUT}/selection.json"))
    met = json.load(open(f"{OUT}/metrics_by_condition_k.json"))
    use_norm = met["distances_normalised"]
    models = sel["models"]
    hard = near_relative_pairs(models)
    sweep = sel["gamma_sweep_chains"]

    Dq, w2, y_ev, y_rf, _, cfg_ev = build_dq(eval_pool="test")
    M = boot_draws(int(cfg_ev.max()) + 1, 2000)

    curves, bands = {}, {}
    for g, chain in sweep.items():
        curves[g], bands[g] = {}, {}
        for k in range(1, K_HEAD + 1):
            D = summed(Dq, w2, chain[:k], normalise=use_norm)
            p, st = condition_stats(D, y_ev, y_rf, cfg_ev, hard, len(models))
            curves[g][k] = {m: p[m] for m in
                            ("mean_top1", "worst_class", "worst3_class",
                             "hard_subset", "hard_subset_min")}
            b = boot_metrics(st, M)
            bands[g][k] = {m: ci(b[m]) for m in
                           ("mean_top1", "worst_class", "hard_subset")}
        print(f"[S5] gamma={g:5s} k=8  mean {curves[g][8]['mean_top1']:.4f}  "
              f"worst {curves[g][8]['worst_class']:.4f}  "
              f"hard {curves[g][8]['hard_subset']:.4f}", flush=True)

    gs = list(sweep)
    overlap = {}
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            for k in (1, 3, 8):
                overlap[f"{gs[i]}_vs_{gs[j]}_k{k}"] = len(
                    set(sweep[gs[i]][:k]) & set(sweep[gs[j]][:k]))

    out = dict(schema="d008-gamma-v1", gammas=gs, aggregation="max",
               distances_normalised=use_norm,
               chains={g: sweep[g][:K_HEAD] for g in gs},
               curves=curves, bootstrap_ci=bands, chain_overlap=overlap,
               note="gamma=1.0 IS mean-greedy (METHOD.md §6.5); the sweep exists "
                    "to show containment, not competition")
    json.dump(out, open(f"{OUT}/gamma_sweep.json", "w"), indent=1)

    print("\n===== S5: trade-off at k=8 =====")
    print(f"{'gamma':>6s} {'mean':>7s} {'worst':>7s} {'worst3':>7s} {'hard':>7s}")
    for g in gs:
        c = curves[g][8]
        print(f"{g:>6s} {c['mean_top1']:7.4f} {c['worst_class']:7.4f} "
              f"{c['worst3_class']:7.4f} {c['hard_subset']:7.4f}")
    print(f"\nchain overlap at k=8: "
          f"{ {k: v for k, v in overlap.items() if k.endswith('k8')} }")
    print(f"written: {OUT}/gamma_sweep.json")


if __name__ == "__main__":
    main()
