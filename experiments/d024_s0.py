"""D024 / S0 — inventory and target-free setup (pre-P1).

(1) SINGLE reference config per model under the D's tier rule, for BOTH readings
    of "lowest temperature" (P1 Call 1): the literal `temperature` field, and
    greedy (do_sample=False) as temperature 0. Stop condition: a model with no
    config in any tier.
(2) 259 x 666 per-query squared-distance arrays: CENTROID, SINGLE-field,
    SINGLE-effective (all median-normalised) plus the raw-centroid sensitivity
    array. Accumulator == direct concatenation asserted for CENTROID. Health
    diagnostics per arm (medians, near-zero cells, single-query CVaR ties) --
    the D's "degenerate objective" worry, checked before any chain exists.
(3) Code-path check: the D024 greedy wrapper with D010's CLOUD statistic must
    reproduce D010's chain exactly (stop condition if not).
(4) Inputs for S2/S3 exist (D010, D020, D016, D023, D019, D021, D022).
No chain for CENTROID/SINGLE is selected here (S1, after the Review), and no
trained number is read beyond existence checks.
Writes results/D024/s0_pilot.json and results/D024/distances.npz.
Usage:  PYTHONPATH=.:experiments python experiments/d024_s0.py
"""
import os
import json
import time
import hashlib

import numpy as np

from LLMmap.joint_statistic import cvar
from d007_lib import load_corpus
from d010_select import precompute
from d024_lib import (OUT, GAMMA, K, build_prompt_confs, single_config, pairs_of,
                      point_d2, centroid_points, single_points, median_normalise,
                      greedy_cvar, additive_hooks, cloud_hooks)


def health(D, raw=None):
    """Diagnostics for a (259, 666) per-query array. `raw` = un-normalised
    squared distances (for zero counts)."""
    raw = D if raw is None else raw
    med = np.median(raw, axis=1)
    near0 = raw <= 1e-6 * med[:, None]
    single_q = np.array([cvar(np.sqrt(D[q]), GAMMA) for q in range(D.shape[0])])
    top = np.sort(single_q)[::-1]
    return dict(
        median_d2_per_query=dict(min=float(med.min()), p50=float(np.median(med)),
                                 max=float(med.max()), max_over_min=float(med.max() / med.min())),
        near_zero_cells=dict(total=int(near0.sum()),
                             queries_with_any=int(near0.any(1).sum()),
                             queries_with_ge66=int((near0.sum(1) >= 66).sum()),
                             pairs_ever=int(near0.any(0).sum())),
        single_query_cvar=dict(max=float(top[0]), second=float(top[1]),
                               n_at_max=int((single_q == top[0]).sum()),
                               n_zero=int((single_q <= 0).sum()),
                               p50=float(np.median(single_q))))


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    models, X = load_corpus(pool="build")
    ref_models = json.load(open("./results/D008/selection.json"))["models"]
    assert models == ref_models, "model order differs from D008/D009"
    pairs, ia, ib = pairs_of(models)
    nq = X[models[0]].shape[0]
    assert (nq, len(pairs)) == (259, 666)
    out = dict(schema="d024-s0-v1", models=models, n_pairs=len(pairs), n_queries=nq)

    # ---- (1) SINGLE tier
    pcs = build_prompt_confs(models)
    tiers = {}
    for reading in ("field", "effective"):
        per = {}
        for m in models:
            r = single_config(pcs[m], reading)
            assert r is not None, f"STOP: {m} has no config in any SINGLE tier"
            c, t, pc = r
            per[m] = dict(config_index=int(c), tier=int(t),
                          do_sample=pc["sampling_hparams"]["do_sample"],
                          temperature=pc["sampling_hparams"]["temperature"],
                          system_prompt=pc["system_prompt"])
        tiers[reading] = per
    tier1 = {m: sorted([(c, pc["sampling_hparams"]["do_sample"], pc["sampling_hparams"]["temperature"])
                        for c, pc in pcs[m].items()
                        if pc["system_prompt"] == "" and pc["cot_prompt"] is None
                        and pc["rag_prompt"] is None]) for m in models}
    out["single_tier"] = dict(
        readings=tiers,
        tier_counts={rd: {str(t): sum(1 for v in tiers[rd].values() if v["tier"] == t)
                          for t in (1, 2, 3)} for rd in tiers},
        sampling_of_choice={rd: sorted({(v["do_sample"], v["temperature"]) for v in tiers[rd].values()})
                            for rd in tiers},
        tier1_configs_per_model=tier1)
    print(f"[S0] SINGLE tiers: {out['single_tier']['tier_counts']}; sampling "
          f"{out['single_tier']['sampling_of_choice']}", flush=True)

    # ---- (2) distance arrays
    cen = centroid_points(X, models)                          # (37, 259, 1024)
    D_cen_raw = point_d2(cen, ia, ib)
    D_cen, med_cen = median_normalise(D_cen_raw)
    # accumulator == direct concatenation (a few queries, first/last pair)
    qs = [3, 11, 40, 77]
    for p in (0, len(pairs) - 1):
        direct = float(np.sum((np.concatenate(cen[ia[p], qs]) - np.concatenate(cen[ib[p], qs])) ** 2))
        assert abs(direct - D_cen_raw[qs, p].sum()) < 1e-6 * direct, (p, direct)
    arrays = dict(centroid=D_cen, centroid_raw=D_cen_raw, centroid_median=med_cen)
    hl = dict(centroid=health(D_cen, D_cen_raw), centroid_raw=health(D_cen_raw))
    for rd in ("field", "effective"):
        cfg = {m: tiers[rd][m]["config_index"] for m in models}
        raw = point_d2(single_points(X, models, cfg), ia, ib)
        n, med = median_normalise(raw)
        arrays[f"single_{rd}"], arrays[f"single_{rd}_raw"] = n, raw
        arrays[f"single_{rd}_median"] = med
        hl[f"single_{rd}"] = health(n, raw)
    out["health"] = hl
    for a, h in hl.items():
        print(f"[S0] {a}: {h}", flush=True)
    np.savez_compressed(f"{OUT}/distances.npz", **arrays,
                        ia=ia, ib=ib, models=np.array(models))
    out["distances_npz_sha256"] = hashlib.sha256(open(f"{OUT}/distances.npz", "rb").read()).hexdigest()

    # ---- (3) code-path check: wrapper + CLOUD statistic == D010
    d10 = json.load(open("./results/D010/selection.json"))["joint_energy"]
    t = time.time()
    XX2, XY2 = precompute(X, models, pairs)
    sel, trace = greedy_cvar(nq, *cloud_hooks(XY2, XX2, ia, ib), k=K, gamma=GAMMA)
    del XY2, XX2
    obj_diff = max(abs(a["objective_cvar"] - b["objective_cvar"]) for a, b in zip(trace, d10["trace"]))
    ok = sel == d10["queries"]
    out["code_path_check"] = dict(chain=sel, d010_chain=d10["queries"], identical=ok,
                                  max_abs_objective_diff=obj_diff, trace=trace,
                                  wall_min=round((time.time() - t) / 60, 2))
    print(f"[S0] code-path check: {sel} vs D010 {d10['queries']} -> {ok}; "
          f"max |Δobj| {obj_diff:.2e}", flush=True)

    # ---- (4) inventory
    inv = {}
    c10 = np.load("./results/D010/run_counts.npz")
    inv["D010_joint_energy_attention_seeds"] = {
        k: len([n for n in c10.files if n.startswith(f"joint_energy|{k}|") and n.endswith("cnt_total")])
        for k in range(1, 9)}
    c20 = np.load("./results/D020/linear_counts.npz")
    inv["D020_JointGreedy_linear"] = {p: all(f"JointGreedy|{p}|{k}|cnt_total" in c20.files
                                             for k in range(1, 9)) for p in ("concat", "meanpool")}
    inv["D016_k8_logits"] = {c: [os.path.exists(f"./results/D016/logits_{c}_k8_r{r}.npy")
                                 for r in range(5)] for c in ("joint_energy", "paper8", "coverage")}
    c23 = np.load("./results/D023/run_counts.npz")
    inv["D023_GreedyADD_attention_seeds"] = {
        k: len([n for n in c23.files if n.startswith(f"Greedy-ADD|{k}|") and n.endswith("cnt_total")])
        for k in range(1, 9)}
    inv["D023_GreedyADD_cnt_pair666_k8"] = sum(n.startswith("Greedy-ADD|8|") and n.endswith("cnt_pair666")
                                               for n in c23.files)
    inv["D023_JointGreedy_paper8_cnt_pair666_k8"] = {
        a: sum(n.startswith(f"{a}|8|") and n.endswith("cnt_pair666") for n in c23.files)
        for a in ("JointGreedy", "paper8")}
    l23 = np.load("./results/D023/linear_counts.npz")
    inv["D023_GreedyADD_linear"] = {p: all(f"Greedy-ADD|{p}|{k}|cnt_total" in l23.files
                                           for k in range(1, 9)) for p in ("concat", "meanpool")}
    for f in ("./results/D022/query_flags.json", "./results/D016/full_pair_trained_accuracy.json",
              "./results/D023/analysis.json", "./results/D023/selection.json"):
        inv[f] = os.path.exists(f)
    inv["D019_files"] = sorted(os.listdir("./results/D019")) if os.path.isdir("./results/D019") else None
    inv["D021_files"] = sorted(os.listdir("./results/D021")) if os.path.isdir("./results/D021") else None
    out["inventory"] = inv
    out["wall_s"] = round(time.time() - t0, 1)
    print(f"[S0] inventory {json.dumps(inv)}; wall {out['wall_s']} s", flush=True)
    json.dump(out, open(f"{OUT}/s0_pilot.json", "w"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    if not ok:
        raise SystemExit("STOP: code-path check failed (D024 S0 stop condition)")


if __name__ == "__main__":
    main()
