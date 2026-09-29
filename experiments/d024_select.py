"""D024 / S1 — selection and freeze (Review 2026-09-29: APPROVED WITH AMENDMENTS).

Chains (d024_lib.greedy_cvar + additive_hooks, k=1..8, ties -> lowest id):
  CENTROID          normalised centroid statistic               TRAINED
  SINGLE            greedy reference config (Call 1 (b))        TRAINED
  CENTROID-raw      un-normalised centroid (sensitivity)        TRAINED iff Call 4 trigger
  SINGLE-field      sampled T=0.4 reference (Call 1 sensitivity) selection only
CLOUD = D010's JointGreedy chain, verbatim (code-path check done in S0).

Call 2: canonical training order = JointGreedy-shared queries in JointGreedy's
order, then ascending id; identical-input k = identical sets.
Call 4: raw-centroid trigger = set of first k differs from CENTROID's at some
k in 3..8.
Also: overlaps (arms, paper8, Greedy-ADD, GreedyCover), and a target-free
cross-objective table (every chain scored under every statistic).
Writes results/D024/selection.json with frozen_before_training = true.
Usage:  PYTHONPATH=.:experiments python experiments/d024_select.py
"""
import json
import time
import hashlib

import numpy as np

from LLMmap.joint_statistic import cvar
from LLMmap.joint_greedy import _stat_all_pairs
from d007_lib import load_corpus
from d010_select import precompute
from d023_select import order_like_jg
from d024_lib import OUT, GAMMA, K, pairs_of, greedy_cvar, additive_hooks

KS = list(range(1, K + 1))
COMPARISONS = {"P1_CLOUD_minus_CENTROID": ("CLOUD", "CENTROID"),
               "P2_CENTROID_minus_SINGLE": ("CENTROID", "SINGLE"),
               "CLOUD_minus_SINGLE": ("CLOUD", "SINGLE"),
               "CLOUD_minus_CENTROID-raw": ("CLOUD", "CENTROID-raw"),
               "CENTROID-raw_minus_CENTROID": ("CENTROID-raw", "CENTROID")}


def main():
    t0 = time.time()
    s0 = json.load(open(f"{OUT}/s0_pilot.json"))
    npz = f"{OUT}/distances.npz"
    assert hashlib.sha256(open(npz, "rb").read()).hexdigest() == s0["distances_npz_sha256"]
    d = np.load(npz)
    arrays = {"CENTROID": d["centroid"], "SINGLE": d["single_effective"],
              "CENTROID-raw": d["centroid_raw"], "SINGLE-field": d["single_field"]}

    d23 = json.load(open("./results/D023/selection.json"))
    ex = d23["existing_chains"]
    jg = ex["JointGreedy"]
    d10 = json.load(open("./results/D010/selection.json"))["joint_energy"]["queries"]
    assert jg == d10 == s0["code_path_check"]["chain"]

    chains, traces = {"CLOUD": jg}, {}
    for arm, D in arrays.items():
        sel, tr = greedy_cvar(D.shape[0], *additive_hooks(D), k=K, gamma=GAMMA)
        chains[arm], traces[arm] = sel, tr
        print(f"[S1] {arm}: {sel}  objective {[round(t['objective_cvar'], 4) for t in tr]}  "
              f"ties {[t['n_exact_ties_at_max'] for t in tr]}", flush=True)

    # Call 4 trigger
    trig_k = [k for k in range(3, K + 1)
              if set(chains["CENTROID-raw"][:k]) != set(chains["CENTROID"][:k])]
    triggered = bool(trig_k)
    trained = ["CENTROID", "SINGLE"] + (["CENTROID-raw"] if triggered else [])
    print(f"[S1] Call 4 trigger: {triggered} (differing k in 3..8: {trig_k})", flush=True)

    # Call 2: canonical orders; CLOUD's reproduce D010 verbatim
    order = {a: {k: order_like_jg(chains[a][:k], jg) for k in KS}
             for a in ["CLOUD"] + trained}
    assert all(order["CLOUD"][k] == jg[:k] for k in KS)
    ident, differ = {}, {}
    for name, (a, b) in COMPARISONS.items():
        if a not in order or b not in order:
            continue
        ident[name] = [k for k in KS if order[a][k] == order[b][k]]
        differ[name] = [k for k in KS if k not in ident[name]]
    # identical-to-CLOUD k per trained arm (reference asserts vs D010/D020)
    ident_cloud = {a: [k for k in KS if order[a][k] == jg[:k]] for a in trained}
    print(f"[S1] identical-input k: {ident}", flush=True)

    # overlaps (k=8 sets) and per-k set overlap with CLOUD
    ref = {**{a: chains[a] for a in chains}, "paper8": ex["paper8"],
           "GreedyCover": ex["GreedyCover"], "Greedy-ADD": d23["Greedy_ADD"]["chain"]}
    names = list(ref)
    overlap_k8 = {a: {b: len(set(ref[a][:8]) & set(ref[b][:8])) for b in names} for a in names}
    overlap_cloud_by_k = {a: [len(set(chains[a][:k]) & set(jg[:k])) for k in KS] for a in chains}

    # cross-objective table (target-free): every chain under every statistic
    models, X = load_corpus(pool="build")
    pairs, ia, ib = pairs_of(models)
    assert np.array_equal(ia, d["ia"]) and np.array_equal(ib, d["ib"])
    XX2, XY2 = precompute(X, models, pairs)
    del X

    def cloud_obj(qs):
        z_xy = np.zeros(XY2.shape[1:], np.float32)
        z_xx = np.zeros(XX2.shape[1:], np.float32)
        return cvar(_stat_all_pairs(XY2[qs].sum(0), XX2[qs].sum(0), ia, ib, z_xy, z_xx,
                                    "energy"), GAMMA)

    def add_obj(D, qs):
        return cvar(np.sqrt(D[qs].sum(0)), GAMMA)

    cross = {}
    for c, qs in ref.items():
        row = {"CLOUD": [cloud_obj(list(qs[:k])) for k in KS]}
        for st, D in arrays.items():
            row[st] = [add_obj(D, list(qs[:k])) for k in KS]
        cross[c] = {st: [round(v, 6) for v in vals] for st, vals in row.items()}
    # sanity: CLOUD row on JointGreedy == D010 trace; own-statistic rows == traces
    d10t = json.load(open("./results/D010/selection.json"))["joint_energy"]["trace"]
    cl_diff = max(abs(a - b["objective_cvar"]) for a, b in zip(cross["CLOUD"]["CLOUD"], d10t))
    assert cl_diff < 1e-5, cl_diff
    for arm in arrays:
        assert max(abs(a - t["objective_cvar"]) for a, t in
                   zip(cross[arm][arm], traces[arm])) < 1e-6, arm

    out = dict(
        schema="d024-selection-v1", frozen_before_training=True,
        review="D024 Review 2026-09-29, APPROVED WITH AMENDMENTS (Calls 1-4)",
        distances_npz_sha256=s0["distances_npz_sha256"], gamma=GAMMA, k=K,
        single_reference=dict(primary="effective (greedy = temperature 0), Call 1 (b)",
                              sensitivity="field (sampled T=0.4), selection only, NOT trained",
                              configs={m: s0["single_tier"]["readings"]["effective"][m]["config_index"]
                                       for m in models}),
        chains=chains, traces=traces,
        call4_trigger=dict(rule="set of first k differs from CENTROID's at some k in 3..8",
                           differing_k=trig_k, triggered=triggered,
                           consequence=("CENTROID-raw trained in S2 as a descriptive row, no verdict "
                                        "(Review Call 4); strongest-baseline amendment applies in S3"
                                        if triggered else "not trained")),
        trained_arms=trained,
        train_order={a: {str(k): v for k, v in o.items()} for a, o in order.items()},
        order_rule="JointGreedy-shared queries in JointGreedy's order, then ascending id (Call 2)",
        identical_input_k=ident, differing_k=differ, identical_to_cloud_k=ident_cloud,
        parity_quantity="mean over each comparison's own differing k (Call 3 / D023 Call 2 (b))",
        overlaps_k8=overlap_k8, overlap_with_cloud_by_k=overlap_cloud_by_k,
        reference_chains={a: ref[a] for a in ("paper8", "GreedyCover", "Greedy-ADD")},
        cross_objective=cross, cross_objective_check=dict(cloud_vs_d010_max_abs=cl_diff),
        wall_s=round(time.time() - t0, 1))
    json.dump(out, open(f"{OUT}/selection.json", "w"), indent=1)
    print(f"[S1] overlaps k=8 with CLOUD: {overlap_k8['CLOUD']}", flush=True)
    print(f"[S1] wrote {OUT}/selection.json; wall {out['wall_s']} s", flush=True)


if __name__ == "__main__":
    main()
