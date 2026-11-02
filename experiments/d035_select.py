"""D035 / S1a — selection and freeze (Review 2026-11-01: APPROVED WITH AMENDMENTS; Calls 1-4 -> (a) / approved).

Chains (d024_lib.greedy_cvar + additive_hooks, k = 1..8, ties -> lowest id), from S0's distances.npz (sha asserted):
  CENTROID-raw, CENTROID-norm          both trained in S1b (S_val choice of CENTROID*, Call 3)
  SINGLE-raw, SINGLE-norm              both frozen here (Call 4); only CENTROID*'s scaling is trained in S2
  SINGLE-field raw / norm              selection only
  SINGLE-greedyfirst raw / norm        selection only (Call 1 (b) reading)
SINGLE = the D's rule verbatim (Call 1 (a)). Under SINGLE-norm query 253 (median d2 = 0) is excluded from the
candidates (Call 2 (a)): the greedy runs on the remaining 258 rows, ids mapped back monotonically, so the
lowest-id tie rule is unchanged.
CLOUD = D029's v2 JG chain verbatim (code-path check done in S0).

Also: canonical training order (d031_lib.train_order), identical-input / differing k per comparison, overlaps,
and the target-free cross-objective table (every chain under every statistic). No S_val / S_test is read.
Writes results/D035/chains.json with frozen_before_training = true.
Usage:  PYTHONPATH=.:experiments python experiments/d035_select.py
"""
import json
import time

import numpy as np

from LLMmap.joint_statistic import cvar
from LLMmap.joint_greedy import _stat_all_pairs
import d031_lib as L31
import d029_lib as L29
import d035_lib as L
from d010_select import precompute
from d024_lib import greedy_cvar, additive_hooks

KS = list(range(1, L.K + 1))
SELECTED = [f"{a}|{s}" for a in ("CENTROID", "SINGLE", "SINGLE-field", "SINGLE-greedyfirst") for s in L.SCALINGS]
TRAINED_CANDIDATES = ["CENTROID|raw", "CENTROID|norm", "SINGLE|raw", "SINGLE|norm"]
EXCLUDE = {"SINGLE|norm": [253]}                                   # Call 2 (a)
# every comparison P1 / P2 can become, plus the unlabelled rows (P1 §5)
COMPARISONS = {
    "P1_if_raw": ("CLOUD", "CENTROID|raw"), "P1_if_norm": ("CLOUD", "CENTROID|norm"),
    "P2_if_raw": ("CENTROID|raw", "SINGLE|raw"), "P2_if_norm": ("CENTROID|norm", "SINGLE|norm"),
    "CLOUD_minus_SINGLE_if_raw": ("CLOUD", "SINGLE|raw"), "CLOUD_minus_SINGLE_if_norm": ("CLOUD", "SINGLE|norm"),
    "CENTROID_raw_vs_norm": ("CENTROID|raw", "CENTROID|norm"),
}


def select(D, exclude=()):
    keep = np.array([q for q in range(D.shape[0]) if q not in set(exclude)])
    sel, tr = greedy_cvar(len(keep), *additive_hooks(D[keep]), k=L.K, gamma=L.GAMMA)
    sel = [int(keep[q]) for q in sel]
    for t, q in zip(tr, sel):
        t["query"] = q
    assert not set(sel) & set(exclude)
    return sel, tr


def main():
    t0 = time.time()
    s0 = json.load(open(f"{L.OUT}/s0.json"))
    npz = f"{L.OUT}/distances.npz"
    assert L31.sha(npz) == s0["distances_npz_sha256"], "distances.npz differs from S0"
    d = np.load(npz)
    ms = L.models()
    assert list(d["models"]) == ms
    ia, ib = L31.pairs("v2")
    assert np.array_equal(ia, d["ia"]) and np.array_equal(ib, d["ib"])
    arrays = {a: d[a] for a in SELECTED}
    for q in EXCLUDE["SINGLE|norm"]:
        assert s0["health"]["SINGLE|norm"]["zero_median_queries"] == [q]
    assert all(np.isfinite(D).all() for a, D in arrays.items() if a != "SINGLE|norm")
    assert np.isfinite(np.delete(arrays["SINGLE|norm"], EXCLUDE["SINGLE|norm"], 0)).all()

    d29 = json.load(open("./results/D029/chains.json"))["arms"]
    jg = L31.jg_chain("v2")
    assert jg == d29["JG"]["ids"][:L.K] == s0["code_path_check"]["chain"]
    p8 = d29["PAPER8"]["ids"][:L.K]
    assert p8 == L31.PAPER8

    chains, traces, stops = {"CLOUD": jg}, {}, []
    for arm, D in arrays.items():
        sel, tr = select(D, EXCLUDE.get(arm, ()))
        chains[arm], traces[arm] = sel, tr
        obj = [t["objective_cvar"] for t in tr]
        if len(set(obj)) == 1:
            stops.append(f"{arm}: constant objective trace")
        if tr[0]["n_exact_ties_at_max"] != 1:
            stops.append(f"{arm}: exact tie at the k = 1 maximum ({tr[0]['n_exact_ties_at_max']})")
        print(f"[S1a] {arm}: {sel}  objective {[round(o, 4) for o in obj]}  "
              f"ties {[t['n_exact_ties_at_max'] for t in tr]}", flush=True)

    # canonical training order (D024 Call 2 / D031): a function of the set
    order, tag = {}, {}
    for a in ["CLOUD"] + TRAINED_CANDIDATES:
        order[a], tag[a] = {}, {}
        for k in KS:
            order[a][k], tag[a][k] = L31.train_order(chains[a][:k], jg, p8)
    assert all(order["CLOUD"][k] == jg[:k] for k in KS)
    ident, differ = {}, {}
    for name, (a, b) in COMPARISONS.items():
        ident[name] = [k for k in KS if order[a][k] == order[b][k]]
        differ[name] = [k for k in KS if k not in ident[name]]
    ident_ref = {a: {"JG": [k for k in KS if tag[a][k] == "JG"], "PAPER8": [k for k in KS if tag[a][k] == "PAPER8"]}
                 for a in TRAINED_CANDIDATES}
    print(f"[S1a] identical-input k: {ident}", flush=True)
    print(f"[S1a] identical to D029 sets: {ident_ref}", flush=True)

    # overlaps
    mmr = json.load(open("./results/D031/chains.json"))["views"]["v2"]["arms"]["MMR|pearson|0.7"]["sel"][:L.K]
    ref = {**chains, "PAPER8": p8, "MMR|pearson|0.7 (D031, reference)": mmr}
    names = list(ref)
    overlap_k8 = {a: {b: len(set(ref[a][:8]) & set(ref[b][:8])) for b in names} for a in names}
    overlap_cloud_by_k = {a: [len(set(chains[a][:k]) & set(jg[:k])) for k in KS] for a in chains}
    single_vs = {f"{b}|{s}": [len(set(chains[f"SINGLE|{s}"][:k]) & set(chains[f"{b}|{s}"][:k])) for k in KS]
                 for b in ("SINGLE-field", "SINGLE-greedyfirst") for s in L.SCALINGS}
    collide = {q for q in range(L29.NPOOL)
               if (d["SINGLE|raw"][q] == 0).sum() >= int(np.floor(L.GAMMA * d["SINGLE|raw"].shape[1]))}
    colliding_in_chain = {a: [q for q in ref[a] if q in collide] for a in names}

    # cross-objective table (target-free): every chain under every statistic
    t = time.time()
    X, _ = L.build_cubes(ms)
    pairs = [(ms[a], ms[b]) for a, b in zip(ia, ib)]
    XX2, XY2 = precompute(X, ms, pairs)
    del X

    def cloud_obj(qs):
        z_xy = np.zeros(XY2.shape[1:], np.float32)
        z_xx = np.zeros(XX2.shape[1:], np.float32)
        return cvar(_stat_all_pairs(XY2[qs].sum(0), XX2[qs].sum(0), ia, ib, z_xy, z_xx, "energy"), L.GAMMA)

    def add_obj(D, qs):
        s = np.sqrt(D[qs].sum(0))
        return float(cvar(s, L.GAMMA)) if np.isfinite(s).all() else None

    cross = {}
    for c, qs in ref.items():
        row = {"CLOUD": [cloud_obj(list(qs[:k])) for k in KS]}
        for st, D in arrays.items():
            row[st] = [add_obj(D, list(qs[:k])) for k in KS]
        cross[c] = {st: [None if v is None else round(v, 6) for v in vals] for st, vals in row.items()}
    del XY2, XX2
    cl_diff = max(abs(a - b["objective_cvar"]) for a, b in zip(cross["CLOUD"]["CLOUD"], d29["JG"]["trace"][:L.K]))
    assert cl_diff < 1e-5, cl_diff
    own = {}
    for arm in arrays:
        own[arm] = max(abs(a - t_["objective_cvar"]) for a, t_ in zip(cross[arm][arm], traces[arm]))
        assert own[arm] < 1e-6, (arm, own[arm])
    print(f"[S1a] cross-objective table: CLOUD-vs-D029 max |d| {cl_diff:.2e}; own-row max |d| "
          f"{max(own.values()):.2e} ({(time.time() - t) / 60:.1f} min)", flush=True)

    out = dict(
        schema="d035-chains-v1", frozen_before_training=True, view=L.VIEW, models=ms,
        review="D035 Review 2026-11-01, APPROVED WITH AMENDMENTS (Calls 1-4 -> (a) / approved; A1 reporting)",
        distances_npz_sha256=s0["distances_npz_sha256"], gamma=L.GAMMA, k=L.K,
        candidate_space=dict(n=L29.NPOOL, excluded=EXCLUDE,
                             exclusion_rule="Call 2 (a): query 253 has median d2 = 0 under SINGLE (1,831 / 3,570 "
                                            "zero pairs), so SINGLE-norm is undefined for it; asserted never selected"),
        tie_rule="strict > over ascending ids (lowest id wins)",
        single_reference=dict(
            primary="effective (the D's rule verbatim: tier 1, greedy else lowest temperature, lowest config index)"
                    " -- Call 1 (a)",
            configs={m: s0["single_tier"]["readings"]["effective"][m]["config_index"] for m in ms},
            selection_only=["SINGLE-field (D024 literal field reading)", "SINGLE-greedyfirst (Call 1 (b))"]),
        chains=chains, traces=traces, stops=stops,
        trained_candidates=dict(
            S1b=["CENTROID|raw", "CENTROID|norm"],
            S2="SINGLE|<scaling of CENTROID*> (Call 4: both SINGLE scalings frozen here; the other stays "
               "selection only)"),
        train_order={a: {str(k): v for k, v in o.items()} for a, o in order.items()},
        train_order_tag={a: {str(k): v for k, v in o.items()} for a, o in tag.items()},
        order_rule="d031_lib.train_order: JG-shared queries in JG order, then ascending id; set = JG / PAPER8 "
                   "k-prefix -> that order",
        comparisons={n: list(v) for n, v in COMPARISONS.items()},
        identical_input_k=ident, differing_k=differ, identical_to_d029_sets_k=ident_ref,
        parity_quantity="mean over each comparison's own differing k",
        overlaps_k8=overlap_k8, overlap_with_cloud_by_k=overlap_cloud_by_k, single_vs_readings_by_k=single_vs,
        single_colliding_queries=sorted(collide), colliding_in_chain=colliding_in_chain,
        reference_chains={"PAPER8": p8, "MMR|pearson|0.7 (D031, reference)": mmr},
        cross_objective=cross,
        cross_objective_check=dict(cloud_vs_d029_max_abs=cl_diff, own_rows_max_abs=own),
        s_val_reads=0, s_test_reads=0, code_commit=L31.git_head(), wall_min=round((time.time() - t0) / 60, 2))
    L31.jdump(out, f"{L.OUT}/chains.json")
    print(f"[S1a] overlaps k=8 with CLOUD: {overlap_k8['CLOUD']}", flush=True)
    print(f"[S1a] colliding queries in chains: {colliding_in_chain}", flush=True)
    print(f"[S1a] wrote {L.OUT}/chains.json ({out['wall_min']} min); stops: {stops}", flush=True)


if __name__ == "__main__":
    main()
