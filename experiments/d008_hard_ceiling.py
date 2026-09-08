"""D008 — resolution audit for the hard-subset metric (Call 2's metric).

Every condition, random draws included, scores 0.90-0.94 on hard-subset
accuracy. Before reporting "CVaR and mean-greedy do not differ on hard-subset",
this establishes whether that is *no effect* or *no headroom* -- the exact
distinction D001 got wrong, D004 made mandatory and A3 made a formal gate.

The ceiling is the oracle: for each of the 65 structural near-relative pairs,
the best 2-way TEST accuracy over all 259 queries. Selected on the evaluation
split, so it is optimistic by construction -- which is what a ceiling should be.

Usage:  PYTHONPATH=.:experiments python experiments/d008_hard_ceiling.py
"""
import json

import numpy as np

from d008_lib import (build_dq, two_way_correct, near_relative_pairs,
                      load_tensor, OUT)

def main():
    S, meta = load_tensor()
    models = meta["models"]
    hard = near_relative_pairs(models)
    Dq, w2, y_ev, y_rf, _, cfg_ev = build_dq(eval_pool="test")

    rows = []
    for i, info in sorted(hard.items()):
        a, b = info["ab"]
        accs = np.array([two_way_correct(Dq[q], y_ev, y_rf, a, b)[0].mean()
                         for q in range(S.shape[0])])
        qs = int(S[:, i].argmax())
        rows.append(dict(pair=" | ".join(info["pair"]),
                         oracle=float(accs.max()),
                         best_build_query=float(accs[qs]),
                         median_query=float(np.median(accs)),
                         floor=float(accs.min())))
    o = np.array([r["oracle"] for r in rows])
    b = np.array([r["best_build_query"] for r in rows])
    m = np.array([r["median_query"] for r in rows])
    met = json.load(open(f"{OUT}/metrics_by_condition_k.json"))
    obs = {n: met["metrics"][n]["8"]["hard_subset"] for n in
           ("paper8", "random", "mean_greedy_max", "cvar_max")}
    out = dict(schema="d008-hard-ceiling-v1", n_pairs=len(rows),
               oracle_mean=float(o.mean()), oracle_min=float(o.min()),
               best_single_build_query_mean=float(b.mean()),
               median_single_query_mean=float(m.mean()),
               observed_at_k8=obs,
               headroom_above_best_condition=float(o.mean() - max(obs.values())),
               span_random_to_best=float(max(obs.values()) - obs["random"]),
               pairs=sorted(rows, key=lambda r: r["oracle"]))
    json.dump(out, open(f"{OUT}/hard_subset_ceiling.json", "w"), indent=1)
    print(f"oracle ceiling over {len(rows)} near-relative pairs: "
          f"{o.mean():.4f} (min {o.min():.4f})")
    print(f"single best build-selected query: {b.mean():.4f}; "
          f"median query: {m.mean():.4f}")
    print(f"observed at k=8: {obs}")
    print(f"headroom above the best condition: "
          f"{o.mean() - max(obs.values()):+.4f}; "
          f"random-to-best span: {max(obs.values()) - obs['random']:+.4f}")
    for r in out["pairs"][:8]:
        print(f"   {r['pair'][:72]:72s} oracle {r['oracle']:.3f}")

if __name__ == "__main__":
    main()
