"""D038 / S0 — reproduction checks, both views, S_build only (target-free; run before the Review, as D037's S0).

Per view:
  1. inputs: S_build cubes only (75 configs x 259 queries per model); frozen sf tensor hash; the 20 splits.
  2. path 1 (builder ops): single-query sf energies for all 41 config sets -> energies_<view>.npz (local, gitignored).
     Full set == frozen tensor (rel <= 1e-5; exact-equality count reported). Code path: d007_lib.energy on 24 sampled
     restricted clouds == the stored half values exactly.
  3. path 2 (accumulators): the k = 1 accumulator sf statistic vs the frozen tensor, measured only (P1 Call 1).
  4. JG on full S_build (d037_lib.jg_chain) == D029 (v2) / D010 (v1) chain; trace vs stored objectives.
  5. restrict(., all) == the full blocks; y_joint(JG chain, all) == the JG trace exactly.
  6. TOP-K[CVaR], MMR* and MMR|pearson|0.3/0.5 on the frozen tensor AND on path 1's full set == D031's chains;
     GC on both == D029's v2 chain / D008's v1 cvar_max chain (same tensor hash).
  7. timing smoke for S2: JG on split 0's A, readout (2) for PAPER8 on split 0 (A -> B); values not reported.
A failed reproduction is recorded in `stops` (the D: stop for a Call); the run continues so every check is reported.
Usage:  OMP_NUM_THREADS=32 D038_WORKERS=64 PYTHONPATH=.:experiments python experiments/d038_s0.py
"""
import os
import time

import numpy as np

import d031_lib as L31
import d038_lib as L
from d007_lib import energy


def view_s0(view, stops):
    t0 = time.time()
    rec = dict()
    ms, X = L.build_cubes(view)
    F = L.frozen(view)
    ia, ib = L31.pairs(view)
    sp = L.splits()
    for a, b in sp:
        assert len(a) == L.NA and len(b) == L.NCFG - L.NA and not set(a) & set(b)
    rec.update(n_models=len(ms), n_pairs=len(ia), cubes="S_build only, 75 configs x 259 queries per model",
               frozen_sha256=L.FROZEN_SHA[view], split_A0_head=sp[0][0][:5].tolist())

    # path 1
    t = time.time()
    E = L.energies(view, ms, X, L.workers())
    np.savez(L.energies_path(view), sf=E)
    full = E[0]
    rel = np.abs(full.astype(np.float64) - F) / np.maximum(np.abs(F.astype(np.float64)), 1e-12)
    rec["path1"] = dict(wall_min=round((time.time() - t) / 60, 2), n_sets=E.shape[0],
                        full_max_rel=float(rel.max()), full_n_exact=int((full == F).sum()), n_cells=int(F.size),
                        energies_sha256=L31.sha(L.energies_path(view)))
    if not rel.max() <= 1e-5:
        stops.append(f"{view}: path-1 full-set energies differ from the frozen tensor (max rel {rel.max():.2e})")
    rng = np.random.default_rng(0)
    sets = L.config_sets()
    bad = 0
    for _ in range(24):
        q, p, s = int(rng.integers(L.NPOOL)), int(rng.integers(len(ia))), int(rng.integers(1, len(sets)))
        ix = sets[s]
        a = np.asarray(X[ms[ia[p]]][q], np.float32)[ix]
        b = np.asarray(X[ms[ib[p]]][q], np.float32)[ix]
        bad += int(np.float32(energy(a, b)[1]) != E[s, q, p])
    rec["path1"]["restricted_cloud_check"] = dict(n=24, n_mismatch=bad)
    if bad:
        stops.append(f"{view}: restricted-cloud code path differs from d007_lib.energy in {bad}/24 samples")

    # path 2
    t = time.time()
    XY2, XX2, ia, ib = L.accumulators(view, ms, X)
    rec["accumulators_min"] = round((time.time() - t) / 60, 2)
    z_xy = np.zeros(XY2.shape[1:], np.float32)
    z_xx = np.zeros(XX2.shape[1:], np.float32)
    acc1 = np.stack([L.L36.energy_stat(z_xy, z_xx, ia, ib, XY2[q], XX2[q], "sf") for q in range(L.NPOOL)])
    d = np.abs(acc1.astype(np.float64) - F)
    r = d / np.maximum(np.abs(F.astype(np.float64)), 1e-12)
    rec["path2_k1_vs_frozen"] = dict(max_abs=float(d.max()), max_rel=float(r.max()), median_rel=float(np.median(r)),
                                     n_rel_gt_1e5=int((r > 1e-5).sum()), n_cells=int(F.size),
                                     mean_signed=float((acc1.astype(np.float64) - F).mean()))
    del acc1, z_xy, z_xx

    ref = L.ref_chains(view)
    t = time.time()
    jg, tr = L.jg(XY2, XX2, ia, ib)
    rec["jg_full"] = dict(chain=jg, ref=ref["JG"], equal=jg == ref["JG"], wall_min=round((time.time() - t) / 60, 2),
                          ties=[x["n_exact_ties_at_max"] for x in tr],
                          trace_max_abs_vs_stored=float(max(abs(x["objective_cvar"] - y)
                                                            for x, y in zip(tr, ref["JG_trace"]))))
    if jg != ref["JG"]:
        stops.append(f"{view}: JG full-S_build chain {jg} != frozen {ref['JG']}")

    XY2h, XX2h = L.restrict(XY2, sets[0]), L.restrict(XX2, sets[0])
    same = bool(np.array_equal(XY2h, XY2) and np.array_equal(XX2h, XX2))
    yj, _ = L.y_joint(XY2h, XX2h, ia, ib, jg)
    yj_eq = yj == [x["objective_cvar"] for x in tr]
    rec["subblock_all"] = dict(blocks_equal=same, y_joint_equals_jg_trace=yj_eq)
    if not (same and yj_eq):
        stops.append(f"{view}: sub-block code path with C = all differs from the full path")
    del XY2h, XX2h

    rules = {}
    for src, M in (("frozen", F), ("path1_full", full)):
        rules[src] = {"TOP-K[CVaR]": L.topk(M), "MMR*": L.mmr(M, L.MMR_STAR[1]), "GC": L.gc(M),
                      **{f"MMR|pearson|{l}": L.mmr(M, l) for l in L.MMR_DESC}}
    rec["rules"] = {}
    for name in rules["frozen"]:
        ok = {src: rules[src][name][0] == ref[name] for src in rules}
        rec["rules"][name] = dict(ref=ref[name], **{src: rules[src][name][0] for src in rules}, equal=ok,
                                  ties={src: rules[src][name][1] for src in rules})
        for src, v in ok.items():
            if not v:
                stops.append(f"{view}: {name} on {src} = {rules[src][name][0]} != frozen {ref[name]}")

    # timing smoke (S2 cost)
    a0, b0 = sets[1], sets[2]
    t = time.time()
    XY2h, XX2h = L.restrict(XY2, a0), L.restrict(XX2, a0)
    L.jg(XY2h, XX2h, ia, ib)
    rec["smoke_jg_half_min"] = round((time.time() - t) / 60, 2)
    del XY2h, XX2h, XY2, XX2
    t = time.time()
    L.readout2(X, ms, ia, ib, a0, b0, {"PAPER8": L.PAPER8})
    rec["smoke_readout2_paper8_min"] = round((time.time() - t) / 60, 2)
    rec["wall_min"] = round((time.time() - t0) / 60, 2)
    print(f"[S0 {view}] {rec['wall_min']} min; path1 max rel {rec['path1']['full_max_rel']:.2e} "
          f"({rec['path1']['full_n_exact']}/{rec['path1']['n_cells']} exact); path2 k=1 max rel "
          f"{rec['path2_k1_vs_frozen']['max_rel']:.2e}; JG equal {rec['jg_full']['equal']}; "
          f"rules {[(n, v['equal']) for n, v in rec['rules'].items()]}", flush=True)
    return rec


def main():
    os.makedirs(L.OUT, exist_ok=True)
    stops = []
    out = dict(schema="d038-s0-v1", plan="docs/plans/D038-P1.md", s_val_reads=0, s_test_reads=0,
               splits=dict(R=L.R, seed0=L.SEED0, rule="default_rng(seed0 + r).permutation(75); A = sorted(first 37)",
                           A=[a.tolist() for a, _ in L.splits()], B=[b.tolist() for _, b in L.splits()]),
               views={})
    for view in L.VIEWS:
        out["views"][view] = view_s0(view, stops)
    out.update(stops=stops, code_commit=L31.git_head())
    L.jdump(out, f"{L.OUT}/s0.json")
    print(f"[S0] stops: {stops}", flush=True)


if __name__ == "__main__":
    main()
