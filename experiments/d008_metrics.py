"""D008 / S4 — the four METHOD.md §8 metrics, for every condition at every k.

Order of operations matters and is pre-registered in P1:

  1. `val` first, and ONLY to fix the classifier's form -- F4's scale-normalised
     vs raw concatenation, and Call 1's secondary check (1-NN vs nearest-centroid
     vs a 37-class probe). Test is not touched until that choice is fixed.
  2. `test` once, for everything reported.

Selection reads the build-split tensor; the classifier's reference set is build;
test is held out. Nothing in the pipeline lets the evaluation split influence
which queries were chosen.

Usage:  PYTHONPATH=.:experiments python experiments/d008_metrics.py
"""
import os
import json
import itertools

import numpy as np

from d008_lib import (build_dq, summed, near_relative_pairs, condition_stats,
                      boot_draws, boot_metrics, ci, paired_ci, nn_predict,
                      queries_to_reach, OUT)

K_HEAD, K_MAX = 8, 16
N_BOOT = 2000
FORM_KS = [1, 4, 8]
TARGETS = [0.5, 0.7, 0.8, 0.9]

# SMOKE=1 exercises every code path at a fraction of the size, per the standing
# practice of smoke-testing before a fleet (docs/ENV.md, operational discipline).
if os.environ.get("D008_SMOKE"):
    K_HEAD, K_MAX, N_BOOT, FORM_KS = 2, 3, 50, [1, 2]


# ------------------------------------------------------------------ val: form
def centroid_predict(D, y_ref, n_models):
    """Nearest class-centroid in DISTANCE space (mean distance to a model's
    reference traces) -- the secondary Call 1 check, not the primary."""
    means = np.stack([D[:, y_ref == m].mean(axis=1) for m in range(n_models)], 1)
    return np.argmin(means, axis=1)


def choose_form(Dq, w2, y_eval, y_ref, cfg_eval, chains, n_models):
    """F4 + Call 1's secondary check, on `val` only.

    Decision rule, fixed in advance: pick the distance variant with the higher
    mean top-1 averaged over k in FORM_KS and over both greedy chains. Both are
    reported regardless.
    """
    res = {}
    for norm in (True, False):
        for name, chain in chains.items():
            for k in FORM_KS:
                D = summed(Dq, w2, chain[:k], normalise=norm)
                nn = float((nn_predict(D, y_ref) == y_eval).mean())
                ct = float((centroid_predict(D, y_ref, n_models) == y_eval).mean())
                res[f"{'norm' if norm else 'raw'}|{name}|k{k}"] = dict(
                    nn_top1=round(nn, 4), centroid_top1=round(ct, 4))
    def avg(pref):
        v = [r["nn_top1"] for k, r in res.items() if k.startswith(pref)]
        return float(np.mean(v))
    a, b = avg("norm|"), avg("raw|")
    return dict(detail=res, norm_mean_top1=round(a, 4), raw_mean_top1=round(b, 4),
                chosen="normalised" if a >= b else "raw",
                rule="higher val mean top-1 averaged over k in {1,4,8} and both "
                     "greedy chains; fixed before test was touched (P1/F4)"), (a >= b)


# ----------------------------------------------------------------- test: main
def evaluate(Dq, w2, queries, norm, y_eval, y_ref, cfg_eval, hard, n_models):
    D = summed(Dq, w2, queries, normalise=norm)
    return condition_stats(D, y_eval, y_ref, cfg_eval, hard, n_models)


def main():
    os.makedirs(OUT, exist_ok=True)
    sel = json.load(open(f"{OUT}/selection.json"))
    draws = json.load(open(f"{OUT}/random_draws.json"))
    models = sel["models"]
    n_models = len(models)
    hard = near_relative_pairs(models)
    chains = {n: sel["conditions"][n]["queries"]
              for n in ("mean_greedy_max", "cvar_max", "mean_greedy_sum")}

    # ---------- 1. val, form choice only
    print("[form] building val distance matrices...", flush=True)
    Dq, w2, y_ev, y_rf, _, cfg_ev = build_dq(eval_pool="val")
    form, use_norm = choose_form(Dq, w2, y_ev, y_rf, cfg_ev,
                                 {k: chains[k] for k in
                                  ("mean_greedy_max", "cvar_max")}, n_models)
    print(f"[form] normalised {form['norm_mean_top1']:.4f} vs raw "
          f"{form['raw_mean_top1']:.4f} -> using {form['chosen']}", flush=True)

    # F3's val replication: the full condition set, same code path, on val
    val_rep = {}
    for name, chain in chains.items():
        val_rep[name] = {}
        for k in range(1, K_HEAD + 1):
            p, _ = evaluate(Dq, w2, chain[:k], use_norm, y_ev, y_rf, cfg_ev,
                            hard, n_models)
            val_rep[name][k] = {m: p[m] for m in
                                ("mean_top1", "worst_class", "worst3_class",
                                 "hard_subset")}
    for k in range(1, K_HEAD + 1):
        acc = [evaluate(Dq, w2, list(s), use_norm, y_ev, y_rf, cfg_ev, hard,
                        n_models)[0] for s in
               itertools.combinations(range(K_HEAD), k)]
        val_rep.setdefault("paper8", {})[k] = {
            m: float(np.mean([a[m] for a in acc])) for m in
            ("mean_top1", "worst_class", "worst3_class", "hard_subset")}
    del Dq
    print("[form] val replication done; releasing val matrices", flush=True)

    # ---------- 2. test, everything reported
    print("[test] building test distance matrices...", flush=True)
    Dq, w2, y_ev, y_rf, _, cfg_ev = build_dq(eval_pool="test")
    M = boot_draws(int(cfg_ev.max()) + 1, N_BOOT)
    res, stats = {}, {}

    for name, chain in chains.items():
        res[name], stats[name] = {}, {}
        for k in range(1, K_MAX + 1):
            p, st = evaluate(Dq, w2, chain[:k], use_norm, y_ev, y_rf, cfg_ev,
                             hard, n_models)
            res[name][k] = p
            stats[name][k] = st
        print(f"[test] {name:17s} k={K_HEAD} mean "
              f"{res[name][K_HEAD]['mean_top1']:.4f} "
              f"worst {res[name][K_HEAD]['worst_class']:.4f} "
              f"hard {res[name][K_HEAD]['hard_subset']:.4f}", flush=True)

    # paper-8: a SET. k<8 is the mean over all C(8,k) subsets, since the file's
    # order is arbitrary and any prefix would be an artifact of it (P1, minor).
    res["paper8"], stats["paper8"] = {}, {}
    n_paper = 8 if not os.environ.get("D008_SMOKE") else K_HEAD
    for k in range(1, n_paper + 1):
        subs = list(itertools.combinations(range(n_paper), k))
        ps, sts = zip(*[evaluate(Dq, w2, list(s), use_norm, y_ev, y_rf, cfg_ev,
                                 hard, n_models) for s in subs])
        res["paper8"][k] = {m: float(np.mean([p[m] for p in ps]))
                            for m in ("mean_top1", "worst_class", "worst3_class",
                                      "hard_subset", "hard_subset_min")}
        res["paper8"][k]["n_subsets"] = len(subs)
        # at k=n_paper there is exactly one subset, so this IS the paper's set;
        # for k<n_paper the point metrics above are the subset MEAN, and only the
        # k=n_paper entry is bootstrapped (that is the comparison D008 asks for)
        stats["paper8"][k] = sts[0]
    print(f"[test] paper8            k={n_paper} mean "
          f"{res['paper8'][n_paper]['mean_top1']:.4f} "
          f"worst {res['paper8'][n_paper]['worst_class']:.4f} "
          f"hard {res['paper8'][n_paper]['hard_subset']:.4f}", flush=True)

    # random-k: 200 independent draws per k -> a band, not a point
    res["random"] = {}
    for k in range(1, K_MAX + 1):
        dr = draws[str(k)][:5] if os.environ.get("D008_SMOKE") else draws[str(k)]
        ps = [evaluate(Dq, w2, list(s), use_norm, y_ev, y_rf, cfg_ev, hard,
                       n_models)[0] for s in dr]
        res["random"][k] = {}
        for m in ("mean_top1", "worst_class", "worst3_class", "hard_subset"):
            v = np.array([p[m] for p in ps])
            res["random"][k][m] = float(v.mean())
            res["random"][k][m + "_p5"] = float(np.percentile(v, 5))
            res["random"][k][m + "_p95"] = float(np.percentile(v, 95))
        if k in (1, 8):
            print(f"[test] random k={k}: mean {res['random'][k]['mean_top1']:.4f} "
                  f"[{res['random'][k]['mean_top1_p5']:.4f}, "
                  f"{res['random'][k]['mean_top1_p95']:.4f}]", flush=True)

    # ---------- 3. paired bootstrap, CVaR vs the two baselines that matter
    boots = {n: {k: boot_metrics(stats[n][k], M) for k in stats[n]}
             for n in ("mean_greedy_max", "cvar_max", "mean_greedy_sum")}
    boots["paper8"] = {n_paper: boot_metrics(stats["paper8"][n_paper], M)}

    comp = {}
    for k in range(1, K_HEAD + 1):
        comp[f"cvar_vs_mean_k{k}"] = {
            m: paired_ci(boots["cvar_max"][k][m], boots["mean_greedy_max"][k][m])
            for m in ("mean_top1", "worst_class", "worst3_class", "hard_subset")}
    comp[f"cvar_vs_paper8_k{n_paper}"] = {
        m: paired_ci(boots["cvar_max"][n_paper][m], boots["paper8"][n_paper][m])
        for m in ("mean_top1", "worst_class", "worst3_class", "hard_subset")}
    comp[f"mean_greedy_vs_paper8_k{n_paper}"] = {
        m: paired_ci(boots["mean_greedy_max"][n_paper][m],
                     boots["paper8"][n_paper][m])
        for m in ("mean_top1", "worst_class", "worst3_class", "hard_subset")}
    comp[f"max_vs_sum_k{K_HEAD}"] = {
        m: paired_ci(boots["mean_greedy_max"][K_HEAD][m],
                     boots["mean_greedy_sum"][K_HEAD][m])
        for m in ("mean_top1", "worst_class", "worst3_class", "hard_subset")}

    band = {n: {k: {m: ci(boots[n][k][m]) for m in
                    ("mean_top1", "worst_class", "worst3_class", "hard_subset")}
                for k in boots[n]} for n in boots}

    # ---------- 4. queries needed to reach X%
    q2r = {}
    for name in ("mean_greedy_max", "cvar_max", "mean_greedy_sum", "random", "paper8"):
        q2r[name] = {}
        for m in ("mean_top1", "worst_class"):
            curve = {k: res[name][k][m] for k in res[name]}
            for t in TARGETS:
                r = queries_to_reach(curve, t)
                q2r[name][f"{m}@{t}"] = r if r is not None else (
                    ">8 (pool exhausted)" if name == "paper8" else f">{K_MAX}")

    out = dict(schema="d008-metrics-v1", n_boot=N_BOOT, k_head=K_HEAD, k_max=K_MAX,
               eval_split="test", eval_traces=int(y_ev.size),
               reference_split="build", reference_traces=int(y_rf.size),
               n_models=n_models, n_hard_pairs=len(hard),
               chance=round(1.0 / n_models, 4),
               classifier_form=form, distances_normalised=bool(use_norm),
               metrics=res, bootstrap_ci=band, comparisons=comp,
               queries_to_reach=q2r, val_replication=val_rep)
    json.dump(out, open(f"{OUT}/metrics_by_condition_k.json", "w"), indent=1)

    print(f"\n===== S4: headline (test, k={K_HEAD}) =====")
    hdr = f"{'condition':18s} {'mean':>7s} {'worst':>7s} {'worst3':>7s} {'hard':>7s}"
    print(hdr)
    for n in ("paper8", "random", "mean_greedy_sum", "mean_greedy_max", "cvar_max"):
        r = res[n][K_HEAD]
        print(f"{n:18s} {r['mean_top1']:7.4f} {r['worst_class']:7.4f} "
              f"{r['worst3_class']:7.4f} {r['hard_subset']:7.4f}")
    print("\n===== CVaR vs mean-greedy, paired bootstrap over configs =====")
    for k in range(1, K_HEAD + 1):
        c = comp[f"cvar_vs_mean_k{k}"]
        print(f"  k={k}: mean {c['mean_top1']['delta']:+.4f} "
              f"[{c['mean_top1']['lo']:+.4f},{c['mean_top1']['hi']:+.4f}]  "
              f"worst {c['worst_class']['delta']:+.4f} "
              f"[{c['worst_class']['lo']:+.4f},{c['worst_class']['hi']:+.4f}]  "
              f"hard {c['hard_subset']['delta']:+.4f} "
              f"[{c['hard_subset']['lo']:+.4f},{c['hard_subset']['hi']:+.4f}]")
    print(f"\nwritten: {OUT}/metrics_by_condition_k.json")


if __name__ == "__main__":
    main()
