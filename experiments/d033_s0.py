"""D033 / S0 — families, folds (both kinds), per-fold selection on S_build, union for D034. No training, no
S_val, no S_test.

Per view: assert the frozen inputs (D031 S0's assert_inputs + D031 chains.json shas); families = N' components
(asserted = lineage groups); model-level folds (5 x 17, family-stratified) and family-level folds, seeded
sha256("D033|folds|<kind>"); per (kind, fold): OURS^fold = MMR-PP[CVaR] pearson 0.7 on the training-pair columns
of the scale-free tensor, JG^fold = joint_greedy on XY2 / XX2 restricted to training models (S_build, 75 configs).
Code-path asserts: no-holdout run reproduces D031 MMR|pearson|0.7 and the D029 / D010 JG chains exactly; no
held-out model in any selection pair or XX2 slice. Writes results/D033/chains.json (commit + push before S1).
Usage:  PYTHONPATH=.:experiments python experiments/d033_s0.py
"""
import gc
import os
import json
import time

import numpy as np

import d031_lib as L31
import d033_lib as L
from d031_s0 import assert_inputs
from d010_select import precompute


def log(s):
    print(f"[S0] {s}", flush=True)


def main():
    t0 = time.time()
    os.makedirs(L.OUT, exist_ok=True)
    c31 = json.load(open(f"{L31.OUT}/chains.json"))
    c32 = json.load(open("./results/D032/chains.json"))
    fam = L.families()
    nfam = len(set(fam.values()))
    folds = {"model": L.folds_model(fam), "family": L.folds_family(fam)}
    ms2 = L31.models("v2")
    sizes = {k: [sum(1 for m in ms2 if f[m] == j) for j in range(L.NF)] for k, f in folds.items()}
    assert sizes["model"] == [17] * 5, sizes
    fsize = {f: sum(1 for m in ms2 if fam[m] == f) for f in set(fam.values())}
    for f in set(fam.values()):                                      # spreading properties of model folds
        per = [sum(1 for m in ms2 if fam[m] == f and folds["model"][m] == j) for j in range(L.NF)]
        assert max(per) == -(-fsize[f] // L.NF), (f, per)
    sib = {k: L.sibling_enrolled(fam, folds[k], ms2) for k in L.KINDS}
    assert all(sib["model"][m] for m in ms2 if fsize[fam[m]] >= 2)
    assert not any(sib["family"].values())
    log(f"{nfam} families (= lineage groups); fold sizes v2 {sizes}; "
        f"model folds: held-out without enrolled sibling = {sorted(m for m in ms2 if not sib['model'][m])}")

    out = dict(schema="d033-chains-v1", frozen_before_training=True, plan="docs/plans/D033-P1.md",
               review="D033/P1 approved with amendments (09059ed): Calls 1-3 -> (a); A1",
               d031_chains_sha256=L31.sha(f"{L31.OUT}/chains.json"),
               families=fam, family_sizes=fsize, folds=folds,
               fold_rule=dict(model="families size-desc (rng tie-break), members rng-permuted, dealt i mod 5; "
                                    "rng = default_rng(int(sha256('D033|folds|model'), 16))",
                              family="families size-desc (rng tie-break), each to the fold with fewest models "
                                     "(lowest index on a tie); rng = sha256('D033|folds|family')"),
               views={})
    union = {}

    def add(q, src):
        union.setdefault(int(q), []).append(src)

    for view in L.VIEWS:
        V31 = c31["views"][view]
        ms = L31.models(view)
        assert ms == V31["models"]
        h = assert_inputs(view)
        assert h["sf_sha256"] == V31["inputs"]["sf_sha256"], (view, "sf tensor differs from D031")
        sf, _ = L31.tensors(view)
        ia, ib = L31.pairs(view)
        full = V31["arms"]["MMR|pearson|0.7"]["sel"]
        jg_ref = L31.jg_chain(view)

        t = time.time()
        cubes = L31.cubes(view, ("build",))
        pairs = [(ms[a], ms[b]) for a, b in zip(ia, ib)]
        XX2, XY2 = precompute(cubes["build"][1], ms, pairs)
        del cubes
        gc.collect()
        assert XY2.shape == (L31.NPOOL, len(ia), 75, 75) and XX2.shape == (L31.NPOOL, len(ms), 75, 75)
        assert XY2.dtype == XX2.dtype == np.float32
        log(f"{view}: precompute {(time.time() - t) / 60:.1f} min, XY2 {XY2.nbytes / 1e9:.1f} GB")

        # code-path: no model held out
        allm, allc = list(range(len(ms))), np.arange(len(ia))
        o_all = L.ours(sf, allc)
        assert o_all == full, (view, o_all, full)
        j_all, _ = L.jg(XY2, XX2, ia, ib, allm, allc)
        assert j_all == jg_ref[:L.K], (view, j_all, jg_ref)
        log(f"{view}: no-holdout pipeline reproduces D031 MMR|pearson|0.7 {full} and JG {jg_ref[:8]} exactly")

        V = dict(models=ms, n_pairs=len(ia), inputs=h, FULL=full, PAPER8=L31.PAPER8, JG_all=jg_ref[:L.K],
                 no_holdout_assert=dict(ours_equals_d031=True, jg_equals_ref=True), kinds={})
        for q in full:
            add(q, f"FULL|{view}")
        for kind in L.KINDS:
            fold = folds[kind]
            K = []
            for f in range(L.NF):
                t = time.time()
                tr, ho, cols = L.train_split(view, fold, f)
                used = set(ia[cols].tolist()) | set(ib[cols].tolist())
                assert used <= set(tr) and not (used & set(ho)) and len(cols) == len(tr) * (len(tr) - 1) // 2
                o = L.ours(sf, cols)
                j, jt = L.jg(XY2, XX2, ia, ib, tr, cols)
                arms = dict(OURS=o, PAPER8=list(L31.PAPER8), FULL=list(full), JG=j)
                shared = {}
                for a in L.ARMS:                        # Disclosure 6: same ordered chain -> one training
                    for b in L.ARMS[:L.ARMS.index(a)]:
                        if arms[a] == arms[b] and a not in shared:
                            shared[a] = shared.get(b, b)
                for q in o:
                    add(q, f"OURS|{view}|{kind}|f{f}")
                for q in j:
                    add(q, f"JG|{view}|{kind}|f{f}")
                K.append(dict(fold=f, train=tr, heldout=[ms[i] for i in ho], n_train=len(tr), n_heldout=len(ho),
                              n_pairs=int(len(cols)), arms=arms, jg_objective=jt, shared_training=shared,
                              ours_full_overlap=len(set(o) & set(full)),
                              ours_full_jaccard=len(set(o) & set(full)) / len(set(o) | set(full)),
                              jg_ref_overlap=len(set(j) & set(jg_ref[:L.K])),
                              exclusion_assert=True))
                log(f"{view} {kind} f{f}: train {len(tr)} held-out {len(ho)} pairs {len(cols)}  OURS {o} "
                    f"(∩FULL {K[-1]['ours_full_overlap']})  JG {j} (∩JG {K[-1]['jg_ref_overlap']})"
                    f"{'  shared ' + str(shared) if shared else ''}  ({time.time() - t:.0f} s)")
            sibv = L.sibling_enrolled(fam, fold, ms)
            V["kinds"][kind] = dict(folds=K, fold_sizes=[sum(1 for m in ms if fold[m] == j) for j in range(L.NF)],
                                    sibling_enrolled=sibv,
                                    no_sibling=sorted(m for m in ms if not sibv[m]))
        for q in L31.PAPER8:
            add(q, f"PAPER8|{view}")
        out["views"][view] = V
        del XY2, XX2
        gc.collect()

    d31 = {q for v in c31["views"].values() for a in v["arms"].values() if a.get("kind") in ("det", "ref")
           for q in a["sel"][:L.K]}
    d32 = {q for v in c32["views"].values() for a in v["arms"].values() for q in a["sel"][:L.K]}
    out["union"] = dict(
        queries=sorted(union), n=len(union),
        sources={str(q): sorted(set(s)) for q, s in sorted(union.items())},
        in_d031_det_chains=sorted(set(union) & d31), not_in_d031_det_chains=sorted(set(union) - d31),
        in_d032_chains=sorted(set(union) & d32), not_in_d031_or_d032=sorted(set(union) - d31 - d32),
        note="every k <= 8 query of OURS^fold and JG^fold (both kinds, both views), FULL and PAPER8 (both views); "
             "read by D034's generation manifest")
    out["code_commit"] = L.git_head()
    out["wall_min"] = round((time.time() - t0) / 60, 2)
    L.jdump(out, f"{L.OUT}/chains.json")
    log(f"union {len(union)} queries ({len(set(union) - d31 - d32)} not in any D031/D032 chain); "
        f"wrote {L.OUT}/chains.json ({out['wall_min']} min)")


if __name__ == "__main__":
    main()
