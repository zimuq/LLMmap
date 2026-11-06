"""D037 / S1 — selection, frozen before any training or scoring (Review D037/P1: APPROVED WITH AMENDMENTS;
Calls 1, 2 -> (a); A1, A2).

  JG-mean   d037_lib.jg_chain(XY2, XX2, ia, ib, 1.0)   (joint_greedy, cross-checked with greedy_cvar)
  GC-mean   d037_lib.gc_chain(S, 1.0)                  (greedy_cover(S, 8, gamma=1.0, agg="max"))
JG, GC, PAPER8 = D029 verbatim; TOP-K[CVaR], MMR-PP[CVaR] = D031 v2; TOP-K[MEAN] = D032 v2 (reference rows).

Call 1 (a): the gamma = 1 checks that need the new chains run here, before anything is written --
LLMmap/greedy_cover.verify_reduction(S, 8, "max"), and JG-mean's CVaR_1 = float64 mean of the per-pair statistic
at every step (|d| <= 1e-12, no NaN). The degeneracy stop (non-unique k = 1 argmax, constant objective trace) too.
Any hit writes chains_stopped.json and exits non-zero (no chains.json).
Call 2 (a) / A1: new arms train in d037_lib.train_order; every D029 prefix a set matches is recorded, and S2's exact
assert targets the tag train_order returned (JG before GC before PAPER8).
A2: identical-input k = set equality; same set in a different training order is listed as order_only_k.
Existing arms use their stored training orders (D029: chain order; D031 / D032: chains.json "train").
No S_val / S_test is read. Writes results/D037/chains.json with frozen_before_training = true.
Usage:  OMP_NUM_THREADS=8 PYTHONPATH=.:experiments python experiments/d037_select.py
"""
import gc
import sys
import json
import time

import numpy as np

import d031_lib as L31
import d037_lib as L
from d032_s0 import within
from LLMmap.joint_statistic import cvar
from LLMmap.joint_greedy import _stat_all_pairs
from LLMmap.greedy_cover import verify_reduction

KS = list(range(1, L.K + 1))
TOL_MEAN = 1e-12
REFS = ("TOP-K[CVaR]", "TOP-K[MEAN]", "MMR-PP[CVaR]", "PAPER8")
VS_REF = ("PAPER8", "TOP-K[MEAN]", "TOP-K[CVaR]")


def degenerate(name, tr, key, ties):
    out = []
    if tr[0][ties] != 1:
        out.append(f"{name}: k = 1 argmax not unique ({tr[0][ties]} ties)")
    if len({t[key] for t in tr}) == 1:
        out.append(f"{name}: constant objective trace")
    return out


def jg_mean_check(XY2, XX2, ia, ib, sel, trace):
    """CVaR_1 of the selected candidate's per-pair statistic = its float64 mean = the trace objective, every step."""
    nc = XY2.shape[2]
    acc_xy, acc_xx = np.zeros((XY2.shape[1], nc, nc), np.float32), np.zeros((XX2.shape[1], nc, nc), np.float32)
    rows = []
    for q, t in zip(sel, trace):
        s = _stat_all_pairs(acc_xy, acc_xx, ia, ib, XY2[q], XX2[q], "energy")
        nan = bool(np.isnan(s).any())
        c1, mu = cvar(s, 1.0), float(np.mean(np.asarray(s, np.float64)))
        rows.append(dict(k=t["k"], query=q, cvar1=c1, mean=mu, d_cvar1_mean=abs(c1 - mu),
                         d_cvar1_trace=abs(c1 - t["objective_cvar"]), nan=nan))
        acc_xy += XY2[q]
        acc_xx += XX2[q]
    return rows


def main():
    t0 = time.time()
    s0 = json.load(open(f"{L.OUT}/s0.json"))
    assert not s0["stops"] and s0["s_val_reads"] == 0
    ref = L.d029_chains()
    ms = L31.models(L.VIEW)
    c31 = json.load(open("./results/D031/chains.json"))["views"]["v2"]["arms"]
    c32 = json.load(open("./results/D032/chains.json"))["views"]["v2"]["arms"]
    mmr = s0["mmr_reference"]
    assert c31["JG"]["sel"][:L.K] == ref["JG"]
    S = L.tensor()

    # ---------------------------------------------------------------- GC-mean
    t = time.time()
    gcm, gcm_tr = L.gc_chain(S, 1.0)
    stops = degenerate("GC-mean", gcm_tr, "objective", "n_ties")
    try:
        vr = verify_reduction(S, L.K, "max")
        assert vr["gamma1_chain"] == gcm, (vr, gcm)
    except AssertionError as e:
        vr = dict(identical=False, error=repr(e))
        stops.append(f"GC-mean verify_reduction: {e!r}")
    print(f"[S1] GC-mean: {gcm}  objective {[round(x['objective'], 5) for x in gcm_tr]}  ties "
          f"{[x['n_ties'] for x in gcm_tr]}; verify_reduction {vr.get('identical')} ({(time.time() - t) / 60:.1f} min)",
          flush=True)

    # ---------------------------------------------------------------- JG-mean
    t = time.time()
    XY2, XX2, ia, ib = L.accumulators()
    print(f"[S1] accumulators {(time.time() - t) / 60:.1f} min", flush=True)
    t = time.time()
    jgm, jgm_tr = L.jg_chain(XY2, XX2, ia, ib, 1.0)
    stops += degenerate("JG-mean", jgm_tr, "objective_cvar", "n_exact_ties_at_max")
    mc = jg_mean_check(XY2, XX2, ia, ib, jgm, jgm_tr)
    dm = max(max(r["d_cvar1_mean"], r["d_cvar1_trace"]) for r in mc)
    if dm > TOL_MEAN or any(r["nan"] for r in mc):
        stops.append(f"JG-mean objective != mean of per-pair statistic: max |d| {dm}")
    print(f"[S1] JG-mean: {jgm}  objective {[round(x['objective_cvar'], 5) for x in jgm_tr]}  ties "
          f"{[x['n_exact_ties_at_max'] for x in jgm_tr]}; CVaR_1 = mean max |d| {dm:.1e} "
          f"({(time.time() - t) / 60:.1f} min)", flush=True)
    del XY2, XX2
    gc.collect()

    chains = {"JG": ref["JG"], "JG-mean": jgm, "GC": ref["GC"], "GC-mean": gcm, "PAPER8": ref["PAPER8"],
              "TOP-K[CVaR]": c31["TOP-K"]["sel"][:L.K], "TOP-K[MEAN]": c32["TOP-K[MEAN]"]["sel"][:L.K],
              "MMR-PP[CVaR]": c31[mmr]["sel"][:L.K]}
    traces = {"JG-mean": jgm_tr, "GC-mean": gcm_tr}
    if stops:
        L31.jdump(dict(schema="d037-chains-stopped-v1", stops=stops, chains=chains, traces=traces,
                       verify_reduction=vr, jg_mean_check=mc), f"{L.OUT}/chains_stopped.json")
        print(f"[S1] STOP: {stops} -- wrote chains_stopped.json, no chains.json", flush=True)
        sys.exit(1)

    # ---------------------------------------------------------------- training orders (Call 2 (a), A1)
    order, tag, match = {}, {}, {}
    for a in ("JG-mean", "GC-mean"):
        order[a], tag[a], match[a] = {}, {}, {}
        for k in KS:
            order[a][k], tag[a][k] = L.train_order(chains[a][:k], ref)
            match[a][k] = [g for g in ("JG", "GC", "PAPER8") if set(chains[a][:k]) == set(ref[g][:k])]
    for a in ("JG", "GC", "PAPER8"):                                   # D029 trained cands[:k], chain order
        order[a] = {k: list(chains[a][:k]) for k in KS}
    for a, src in (("TOP-K[CVaR]", c31["TOP-K"]), ("TOP-K[MEAN]", c32["TOP-K[MEAN]"]), ("MMR-PP[CVaR]", c31[mmr])):
        order[a] = {k: list(src["train"][str(k)]["q"]) for k in KS}
        assert all(set(order[a][k]) == set(chains[a][:k]) for k in KS), a
    multi = {a: {k: m for k, m in match[a].items() if len(m) > 1 and k > 1} for a in match}
    d029_multi = [k for k in KS if k > 1 and set(ref["JG"][:k]) == set(ref["GC"][:k])]

    # ---------------------------------------------------------------- identical-input k (A2: set equality)
    comps = {n: v for n, v in L.CONTRASTS.items()}
    comps.update({f"{a}_vs_{r}": (a, r) for a in ("JG-mean", "GC-mean") for r in VS_REF})
    ident, order_only, differ = {}, {}, {}
    for n, (a, b) in comps.items():
        ident[n] = [k for k in KS if set(chains[a][:k]) == set(chains[b][:k])]
        order_only[n] = [k for k in ident[n] if order[a][k] != order[b][k]]
        differ[n] = [k for k in KS if k not in ident[n]]
    same_chain = {n: not differ[n] for n in comps}
    new_same = [k for k in KS if set(jgm[:k]) == set(gcm[:k])]
    assert all(order["JG-mean"][k] == order["GC-mean"][k] for k in new_same)   # same set -> same order -> bit-identical

    # ---------------------------------------------------------------- descriptive (secondary (c))
    names = list(chains)
    overlap_k8 = {a: {b: len(set(chains[a]) & set(chains[b])) for b in names} for a in names}
    overlap_by_k = {a: {b: [len(set(chains[a][:k]) & set(chains[b][:k])) for k in KS] for b in names}
                    for a in ("JG-mean", "GC-mean")}
    sims = {s: L31.similarity(S, s) for s in ("pearson", "cos")}
    wset = {a: {s: within(M, chains[a]) for s, M in sims.items()} for a in names}
    jt = [x["objective_cvar"] for x in jgm_tr]
    gt = [x["objective"] for x in gcm_tr]
    trend = {"JG-mean": dict(objective=jt, diffs=np.diff(jt).tolist(), n_decreases=int(np.sum(np.diff(jt) < 0)),
                             monotone_nonincreasing=bool(np.all(np.diff(jt) <= 0)),
                             d012_v1_pattern="objective falls as queries are added"),
             "GC-mean": dict(objective=gt, monotone_nondecreasing=bool(np.all(np.diff(gt) >= 0)),
                             note="MAX coverage: mean coverage cannot fall")}

    new = sorted(set(jgm) | set(gcm))
    d33u = {int(q) for q in json.load(open("./results/D033/chains.json"))["union"]["queries"]}
    c35 = json.load(open("./results/D035/chains.json"))["chains"]
    c36 = json.load(open("./results/D036/chains.json"))
    prior = {q for c in c35.values() for q in c} | {q for c in c36["chains"].values() for q in c[:L.K]}
    prior |= {q for F in c36["folds"] for c in F["arms"].values() for q in c}
    union = dict(new_queries=new, n_new=len(new), not_in_d033_union=sorted(set(new) - d33u),
                 not_in_d033_union_or_d035_d036_chains=sorted(set(new) - d33u - prior),
                 note="all k <= 8 queries of JG-mean and GC-mean; adding them to D034's union is a design-side "
                      "edit on resume")

    out = dict(
        schema="d037-chains-v1", frozen_before_training=True, view=L.VIEW, models=ms, n_pairs=len(ia),
        review="D037 Review 2026-11-06, APPROVED WITH AMENDMENTS (Calls 1, 2 -> (a); A1, A2)",
        s0_sha256=L31.sha(f"{L.OUT}/s0.json"), tensor_sha256=L.D029_TENSOR_SHA, gammas=L.GAMMAS, k=L.K,
        tie_rule="lowest id wins (joint_greedy / greedy_cover)", mmr_reference=mmr,
        chains=chains, traces=traces, first_query={a: chains[a][0] for a in names},
        objective_trend=trend, verify_reduction=vr, jg_mean_check=mc, jg_mean_check_max_abs_d=dm,
        train_order={a: {str(k): v for k, v in o.items()} for a, o in order.items()},
        train_order_tag={a: {str(k): v for k, v in o.items()} for a, o in tag.items()},
        d029_prefix_matches={a: {str(k): v for k, v in o.items()} for a, o in match.items()},
        multi_match_k=multi, d029_jg_gc_same_set_k=d029_multi,
        order_rule="d037_lib.train_order: set = D029 JG / GC / PAPER8 k-prefix -> that D029 order (tested in that "
                   "order; A1: S2's exact assert targets the returned tag); else JG-shared queries in JG order, "
                   "then ascending id. Existing arms: stored orders (D029 chain order, D031 / D032 'train').",
        comparisons={n: list(v) for n, v in comps.items()}, identical_input_k=ident, order_only_k=order_only,
        differing_k=differ, same_chain_all_k=same_chain, new_arms_same_set_k=new_same,
        identical_rule="set equality (A2); order_only_k = same set, different training order",
        parity_quantity="mean over each contrast's own differing k (D036)",
        overlaps_k8=overlap_k8, overlap_by_k=overlap_by_k, within_set_sim_k8=wset,
        within_set_rule="d032_s0.within on d031_lib.similarity(S_energy_sf_tok200_v2, pearson | cos)",
        d034_union=union, stops=[], s_val_reads=0, s_test_reads=0, code_commit=L31.git_head(),
        wall_min=round((time.time() - t0) / 60, 2))
    L31.jdump(out, f"{L.OUT}/chains.json")
    print(f"[S1] first queries {out['first_query']}", flush=True)
    print(f"[S1] train-order tags {out['train_order_tag']}; multi-match {multi}; D029 JG/GC same set at {d029_multi}",
          flush=True)
    print(f"[S1] identical-input k {ident}; order-only k {order_only}", flush=True)
    print(f"[S1] JG-mean objective trend {trend['JG-mean']['diffs']}", flush=True)
    print(f"[S1] overlaps k=8 JG-mean {overlap_k8['JG-mean']}; GC-mean {overlap_k8['GC-mean']}", flush=True)
    print(f"[S1] within-set {wset}", flush=True)
    print(f"[S1] D034 union: {new}; not in D033 union / D035-D036 chains: "
          f"{union['not_in_d033_union_or_d035_d036_chains']}", flush=True)
    print(f"[S1] wrote {L.OUT}/chains.json ({out['wall_min']} min)", flush=True)


if __name__ == "__main__":
    main()
