"""D036 / S1 — selection, frozen before any training or scoring (Review D036/P1: APPROVED; Calls 1-3 -> (a)).

  CLOUD-raw         d036_lib.cloud_chain(XY2, XX2, ia, ib, "raw") on all 85 models / 3,570 pairs (m = 357)
  CLOUD-raw^fold    d036_lib.fold_cloud(..., "raw") per D033 model-level fold (68 training models, 2,278 pairs)
  CENTROID-raw^fold d036_lib.fold_centroid(D035 CENTROID|raw d2, training pairs)
  CLOUD^fold        D033 JG^fold, verbatim (S0: fold_cloud(..., "sf") reproduces it)
CLOUD = D029's JG chain and CENTROID-raw = D035's chain, verbatim.

Call 1 (a): the degeneracy stop (non-unique k = 1 argmax, constant objective trace) is checked here for every new
chain; a hit writes chains_stopped.json and exits non-zero (nothing to commit as chains.json).
Also: canonical training order (d031_lib.train_order), identical-input k per comparison, overlaps, the 233-257 id
block, the target-free cross-objective table, fold descriptors (Call 2 (a) same-chain flags) and the D034 union.
No S_val / S_test is read. Writes results/D036/chains.json with frozen_before_training = true.
Usage:  OMP_NUM_THREADS=8 PYTHONPATH=.:experiments python experiments/d036_select.py
"""
import gc
import sys
import json
import time

import numpy as np

import d031_lib as L31
import d033_lib as L33
import d036_lib as L
from d010_select import precompute
from LLMmap.joint_statistic import cvar

KS = list(range(1, L.K + 1))
BLOCK = range(233, 258)                                    # the 233-257 id block (D035 Review, Chains)
COMPARISONS = {"P1": ("CLOUD-raw", "CENTROID-raw"), "P2": ("CLOUD-raw", "CLOUD"),
               "CLOUD_minus_CENTROID-raw": ("CLOUD", "CENTROID-raw"), "CLOUD-raw_vs_PAPER8": ("CLOUD-raw", "PAPER8")}
FOLD_COMPARISONS = {"O1": ("CLOUD-raw", "CENTROID-raw"), "O2": ("CLOUD-raw", "CLOUD")}


def degenerate(name, tr):
    out = []
    if tr[0]["n_exact_ties_at_max"] != 1:
        out.append(f"{name}: k = 1 argmax not unique ({tr[0]['n_exact_ties_at_max']} ties)")
    if len({t["objective_cvar"] for t in tr}) == 1:
        out.append(f"{name}: constant objective trace")
    return out


def main():
    t0 = time.time()
    s0 = json.load(open(f"{L.OUT}/s0.json"))
    assert not s0["stops"] and s0["s_val_reads"] == 0 and s0["s_test_reads"] == 0
    ms = L31.models(L.VIEW)
    ia, ib = L31.pairs(L.VIEW)
    d29 = json.load(open("./results/D029/chains.json"))["arms"]
    jg, p8 = d29["JG"]["ids"][:L.K], d29["PAPER8"]["ids"][:L.K]
    assert jg == L31.jg_chain("v2") and p8 == L31.PAPER8
    c35 = json.load(open("./results/D035/chains.json"))
    cen = c35["chains"]["CENTROID|raw"]
    mmr = json.load(open("./results/D031/chains.json"))["views"]["v2"]["arms"]["MMR|pearson|0.7"]["sel"][:L.K]
    D2 = L.centroid_raw()

    t = time.time()
    X = L31.cubes(L.VIEW, ("build",))["build"][1]
    XX2, XY2 = precompute(X, ms, [(ms[a], ms[b]) for a, b in zip(ia, ib)])
    del X
    gc.collect()
    print(f"[S1] precompute {(time.time() - t) / 60:.1f} min", flush=True)

    # ---------------------------------------------------------------- CLOUD-raw (full)
    t = time.time()
    sel, trace = L.cloud_chain(XY2, XX2, ia, ib, "raw")
    stops = degenerate("CLOUD-raw", trace)
    print(f"[S1] CLOUD-raw: {sel}  objective {[round(x['objective_cvar'], 5) for x in trace]}  "
          f"ties {[x['n_exact_ties_at_max'] for x in trace]} ({(time.time() - t) / 60:.1f} min)", flush=True)
    chains = {"CLOUD-raw": sel, "CLOUD": jg, "CENTROID-raw": cen, "PAPER8": p8}

    # ---------------------------------------------------------------- fold chains
    fold, F33 = L.d033_folds()
    folds = []
    for F in F33:
        t = time.time()
        f = F["fold"]
        tr, ho, cols = L33.train_split(L.VIEW, fold, f)
        assert tr == F["train"] and len(cols) == F["n_pairs"] == len(tr) * (len(tr) - 1) // 2
        used = set(ia[cols].tolist()) | set(ib[cols].tolist())
        assert used <= set(tr) and not used & set(ho)
        cr, cr_tr = L.fold_cloud(XY2, XX2, ia, ib, tr, cols, "raw")
        ce, ce_tr = L.fold_centroid(D2, cols)
        gc.collect()
        stops += degenerate(f"CLOUD-raw^fold{f}", cr_tr) + degenerate(f"CENTROID-raw^fold{f}", ce_tr)
        arms = {"CLOUD-raw": cr, "CENTROID-raw": ce, "CLOUD": F["arms"]["JG"]}
        folds.append(dict(
            fold=f, train=tr, heldout=F["heldout"], n_pairs=len(cols), arms=arms,
            traces={"CLOUD-raw": cr_tr, "CENTROID-raw": ce_tr}, CLOUD_objective_d033=F["jg_objective"],
            exclusion_assert=True,
            overlap_k8={a: {b: len(set(arms[a]) & set(arms[b])) for b in arms} for a in arms},
            overlap_with_full_k8={a: len(set(arms[a]) & set(chains[a])) for a in arms},
            same_set_k8={n: set(arms[a]) == set(arms[b]) for n, (a, b) in FOLD_COMPARISONS.items()},
            same_ordered_chain={n: arms[a] == arms[b] for n, (a, b) in FOLD_COMPARISONS.items()}))
        print(f"[S1] fold {f}: CLOUD-raw^fold {cr}  CENTROID-raw^fold {ce}  CLOUD^fold {arms['CLOUD']} "
              f"({(time.time() - t) / 60:.1f} min)", flush=True)
    fold_same = {n: dict(folds_same_set_k8=sum(F["same_set_k8"][n] for F in folds),
                         folds_same_ordered=sum(F["same_ordered_chain"][n] for F in folds),
                         same_chain_all_folds=all(F["same_set_k8"][n] for F in folds))
                 for n in FOLD_COMPARISONS}

    # ---------------------------------------------------------------- cross-objective table (target-free)
    def obj(qs, scale):
        if scale == "CENTROID-raw":
            return float(cvar(np.sqrt(D2[qs].sum(0)), L.GAMMA))
        z = np.zeros((XY2.shape[1], 75, 75), np.float32)
        return float(cvar(L.energy_stat(XY2[qs].sum(0), XX2[qs].sum(0), ia, ib, z, np.zeros_like(XX2[0]), scale),
                          L.GAMMA))

    ref = {**chains, "MMR|pearson|0.7 (D031, reference)": mmr}
    stats = {"CLOUD-raw": "raw", "CLOUD": "sf", "CENTROID-raw": "CENTROID-raw"}
    cross = {c: {s: [round(obj(list(qs[:k]), sc), 6) for k in KS] for s, sc in stats.items()} for c, qs in ref.items()}
    chk = dict(cloud_raw_own=max(abs(a - b["objective_cvar"]) for a, b in zip(cross["CLOUD-raw"]["CLOUD-raw"], trace)),
               cloud_vs_d029=max(abs(a - b["objective_cvar"]) for a, b in zip(cross["CLOUD"]["CLOUD"], d29["JG"]["trace"])),
               centroid_vs_d035=max(abs(a - b["objective_cvar"]) for a, b in
                                    zip(cross["CENTROID-raw"]["CENTROID-raw"], c35["traces"]["CENTROID|raw"])))
    assert max(chk.values()) < 1e-5, chk
    del XY2, XX2
    gc.collect()

    if stops:
        L31.jdump(dict(schema="d036-chains-stopped-v1", stops=stops, chains=chains, folds=folds),
                  f"{L.OUT}/chains_stopped.json")
        print(f"[S1] STOP (Call 1): {stops} -- wrote chains_stopped.json, no chains.json", flush=True)
        sys.exit(1)

    # ---------------------------------------------------------------- orders, identical inputs, descriptors
    order, tag = {}, {}
    for a in ("CLOUD-raw", "CLOUD", "CENTROID-raw", "PAPER8"):
        order[a], tag[a] = {}, {}
        for k in KS:
            order[a][k], tag[a][k] = L31.train_order(chains[a][:k], jg, p8)
    assert all(order["CLOUD"][k] == jg[:k] and order["PAPER8"][k] == p8[:k] for k in KS)
    assert all(order["CENTROID-raw"][k] == c35["train_order"]["CENTROID|raw"][str(k)] for k in KS)
    ident = {n: [k for k in KS if order[a][k] == order[b][k]] for n, (a, b) in COMPARISONS.items()}
    differ = {n: [k for k in KS if k not in ident[n]] for n in COMPARISONS}
    same_chain = {n: not differ[n] for n in COMPARISONS}            # Call 2 (a)
    names = list(ref)
    overlap_k8 = {a: {b: len(set(ref[a][:8]) & set(ref[b][:8])) for b in names} for a in names}
    overlap_by_k = {a: {b: [len(set(chains[a][:k]) & set(chains[b][:k])) for k in KS] for b in chains} for a in chains}
    block = {a: [q for q in qs if q in BLOCK] for a, qs in ref.items()}
    block_fold = {f"{a}^fold{F['fold']}": [q for q in F["arms"][a] if q in BLOCK] for F in folds for a in F["arms"]}

    new = sorted({q for q in sel} | {q for F in folds for a in ("CLOUD-raw", "CENTROID-raw") for q in F["arms"][a]})
    d33u = {int(q) for q in json.load(open("./results/D033/chains.json"))["union"]["queries"]}
    d35u = {q for c in c35["chains"].values() for q in c}
    union = dict(new_queries=new, n_new=len(new),
                 not_in_d033_union=sorted(set(new) - d33u), not_in_d033_union_or_d035_chains=sorted(set(new) - d33u - d35u),
                 note="all k <= 8 queries of CLOUD-raw (full) and CLOUD-raw^fold / CENTROID-raw^fold (5 model-level "
                      "folds); adding them to D034's generation union is a design-side edit on resume")

    out = dict(
        schema="d036-chains-v1", frozen_before_training=True, view=L.VIEW, models=ms, n_pairs=len(ia),
        review="D036 Review 2026-11-02, APPROVED (Calls 1-3 -> (a); Call 2 extended to open-set fold arms)",
        s0_sha256=L31.sha(f"{L.OUT}/s0.json"), d035_distances_sha256=s0["inventory"]["sha256"]["results/D035/distances.npz"],
        gamma=L.GAMMA, k=L.K, tie_rule="strict > over ascending ids (lowest id wins)",
        chains=chains, trace={"CLOUD-raw": trace},
        first_step={"CLOUD-raw": sel[0], "CLOUD": jg[0], "CENTROID-raw": cen[0],
                    "P2_identical_input_k1": 1 in ident["P2"]},
        train_order={a: {str(k): v for k, v in o.items()} for a, o in order.items()},
        train_order_tag={a: {str(k): v for k, v in o.items()} for a, o in tag.items()},
        order_rule="d031_lib.train_order: JG-shared queries in JG order, then ascending id; set = JG / PAPER8 "
                   "k-prefix -> that order",
        comparisons={n: list(v) for n, v in COMPARISONS.items()}, identical_input_k=ident, differing_k=differ,
        same_chain_all_k=same_chain, parity_quantity="mean over each comparison's own differing k",
        overlaps_k8=overlap_k8, overlap_by_k=overlap_by_k, block_233_257=block, block_233_257_folds=block_fold,
        cross_objective=cross, cross_objective_check=chk,
        folds=folds, fold_rule="D033 chains.json -> folds.model (family-stratified, 5 x 17), unchanged",
        fold_comparisons={n: list(v) for n, v in FOLD_COMPARISONS.items()}, fold_same_chain=fold_same,
        d034_union=union, stops=[], s_val_reads=0, s_test_reads=0, code_commit=L31.git_head(),
        wall_min=round((time.time() - t0) / 60, 2))
    L31.jdump(out, f"{L.OUT}/chains.json")
    print(f"[S1] identical-input k: {ident}; same chain all k: {same_chain}", flush=True)
    print(f"[S1] fold same-chain: {fold_same}", flush=True)
    print(f"[S1] overlaps k=8 with CLOUD-raw: {overlap_k8['CLOUD-raw']}; 233-257 block: {block}", flush=True)
    print(f"[S1] D034 union: {len(new)} new-chain queries, {len(union['not_in_d033_union_or_d035_chains'])} not in "
          f"D033 union / D035 chains", flush=True)
    print(f"[S1] wrote {L.OUT}/chains.json ({out['wall_min']} min)", flush=True)


if __name__ == "__main__":
    main()
