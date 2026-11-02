"""D036 / S0 — target-free checks and inventory (run before the P1 Review, as D024's / D035's S0 were).

(1) Closed-set code path: d036_lib.energy_hooks(..., "sf") through d024_lib.greedy_cvar must reproduce D029's v2 JG
    chain with per-step |d objective| = 0.
(2) Direct computation: the accumulator CLOUD-raw statistic for a few (pair, set) cases vs d010_lib.joint_energy_direct
    (raw) on explicit concatenated float32 clouds (cdist, float64). Sets are existing chains only (D029 JG prefixes,
    PAPER8); no CLOUD-raw chain is computed here.
(3) Open-set code path: d036_lib.fold_cloud(..., "sf") must reproduce D033's JG^fold chains (5 model-level folds,
    v2) and their objective traces; d036_lib.fold_centroid on all pairs must reproduce D035's CENTROID|raw chain.
(4) Inventory: every reused D029 / D031 / D033 / D035 input present, with sha256 and required keys.
No S_val / S_test trace is loaded. Writes results/D036/s0.json.
Usage:  PYTHONPATH=.:experiments python experiments/d036_s0.py
"""
import os
import gc
import json
import time

import numpy as np

import d031_lib as L31
import d033_lib as L33
import d036_lib as L
from d010_lib import joint_energy_direct, concat_cloud
from d010_select import precompute

PAIRS_CHECK = (0, 1785, 3569)
TOL_DIRECT = 1e-3                                     # relative, raw statistic, float32 accumulators vs float64 cdist


def inventory(ms):
    inv, stops = {}, []
    files = ["results/D029/chains.json", "results/D029/counts_JG.npz", "results/D029/counts_PAPER8.npz",
             "results/D029/logits_JG.npz", "results/D029/logits_PAPER8.npz",
             "results/D035/chains.json", "results/D035/s0.json", "results/D035/distances.npz",
             "results/D035/run_counts.npz", "results/D035/linear_counts.npz", "results/D035/logits_att.npz",
             "results/D035/logits_lin.npz", "results/D035/runs.json", "results/D035/analysis.json",
             "results/D033/chains.json", "results/D033/variant_choice.json", "results/D033/counts_v2.npz",
             "results/D033/nc_test_v2.npz", "results/D033/analysis.json", "results/D027/n_prime.json"]
    inv["sha256"] = {}
    for f in files:
        if not os.path.exists(f):
            stops.append(f"missing input {f}")
            continue
        inv["sha256"][f] = L31.sha(f)
    n = len(ms)
    for arm in ("JG", "PAPER8"):
        z = np.load(f"./results/D029/counts_{arm}.npz")
        lg = np.load(f"./results/D029/logits_{arm}.npz")
        ok = (all(f"att|{k}|{r}|{c}" in z.files for k in range(1, 9) for r in range(5) for c in ("cnt_total", "cnt_model"))
              and all(f"lin|{k}|{c}" in z.files for k in range(1, 9) for c in ("cnt_total", "cnt_model"))
              and all(f"att|{k}|{r}" in lg.files for k in range(1, 9) for r in range(5))
              and all(f"lin|{k}" in lg.files for k in range(1, 9))
              and np.array_equal(lg["y"], np.repeat(np.arange(n), 25))
              and np.array_equal(lg["cfg"], np.tile(np.arange(25), n)))
        inv[f"D029_{arm}_complete"] = bool(ok)
        if not ok:
            stops.append(f"D029 {arm} counts / logits incomplete")
    att = np.load("./results/D035/run_counts.npz")
    lin = np.load("./results/D035/linear_counts.npz")
    cs = ("cnt_total", "cnt_model", "cnt_pair_all")
    ok = (all(f"CENTROID|raw|{k}|r{r}|{c}" in att.files for k in range(1, 9) for r in range(5) for c in cs)
          and all(f"CENTROID|raw|{k}|{c}" in lin.files for k in range(1, 9) for c in cs))
    inv["D035_CENTROID_raw_complete"] = bool(ok)
    if not ok:
        stops.append("D035 CENTROID|raw counts incomplete")
    vc = json.load(open("./results/D035/variant_choice.json"))
    inv["D035_CENTROID_star"] = vc["CENTROID_star"]
    z = np.load("./results/D033/counts_v2.npz")
    ok = all(f"nc|model|JG|raw|{k}|{c}" in z.files for k in range(1, 9) for c in ("cnt_model", "cnt_pair_all"))
    inv["D033_nc_JG_raw_complete"] = bool(ok)
    if not ok:
        stops.append("D033 nc|model|JG|raw counts incomplete")
    nc = np.load("./results/D033/nc_test_v2.npz")
    inv["D033_nc_test_terms"] = dict(A=list(nc["A"].shape), W=list(nc["W"].shape), U=[int(q) for q in nc["U"]])
    inv["D033_variant_v2"] = json.load(open("./results/D033/variant_choice.json"))["views"]["v2"]["choice"]
    if inv["D033_variant_v2"] != "raw":
        stops.append("D033 v2 readout (2) variant is not raw")
    inv["D027_n_prime_pairs"] = len(json.load(open("./results/D027/n_prime.json"))["pairs"])
    inv["D031_v2_boot_seed"] = L31.VIEW["v2"]["boot_seed"]
    return inv, stops


def main():
    t0 = time.time()
    os.makedirs(f"{L.OUT}/logs", exist_ok=True)
    ms = L31.models(L.VIEW)
    ia, ib = L31.pairs(L.VIEW)
    out = dict(schema="d036-s0-v1", view=L.VIEW, models=ms, n_pairs=len(ia), s_val_reads=0, s_test_reads=0)
    inv, stops = inventory(ms)
    out["inventory"] = inv
    print(f"[S0] inventory: {json.dumps({k: v for k, v in inv.items() if k != 'sha256'})}", flush=True)

    # ---- (1) closed-set code path vs D029 JG
    t = time.time()
    d29 = json.load(open("./results/D029/chains.json"))["arms"]
    jg, p8 = d29["JG"]["ids"][:L.K], d29["PAPER8"]["ids"][:L.K]
    assert jg == L31.jg_chain("v2") and p8 == L31.PAPER8
    X = L31.cubes(L.VIEW, ("build",))["build"][1]
    pairs = [(ms[a], ms[b]) for a, b in zip(ia, ib)]
    XX2, XY2 = precompute(X, ms, pairs)
    print(f"[S0] precompute {(time.time() - t) / 60:.1f} min, XY2 {XY2.nbytes / 1e9:.1f} GB", flush=True)
    sel, trace = L.cloud_chain(XY2, XX2, ia, ib, "sf")
    diffs = [abs(a["objective_cvar"] - b["objective_cvar"]) for a, b in zip(trace, d29["JG"]["trace"][:L.K])]
    ok = sel == jg and max(diffs) == 0.0
    if not ok:
        stops.append(f"closed-set code-path check failed: {sel} vs D029 {jg}, max |dobj| {max(diffs)}")
    out["code_path_closed"] = dict(chain=sel, d029_chain=jg, identical=sel == jg, per_step_abs_objective_diff=diffs,
                                   pass_=ok, wall_min=round((time.time() - t) / 60, 2))
    print(f"[S0] closed-set code path: {sel} vs D029 {jg}; per-step |dobj| {diffs} -> {'PASS' if ok else 'FAIL'}",
          flush=True)

    # ---- (2) direct computation of the raw statistic
    cases, worst = [], 0.0
    for name, qs in (("JG[:1]", jg[:1]), ("JG[:4]", jg[:4]), ("JG[:8]", jg), ("PAPER8", p8)):
        for p in PAIRS_CHECK:
            a, b = int(ia[p]), int(ib[p])
            acc = L.energy_stat(XY2[qs][:, p:p + 1].sum(0), XX2[qs].sum(0), np.array([a]), np.array([b]),
                                np.zeros((1, 75, 75), np.float32), np.zeros_like(XX2[0]), "raw")[0]
            sfa = L.energy_stat(XY2[qs][:, p:p + 1].sum(0), XX2[qs].sum(0), np.array([a]), np.array([b]),
                                np.zeros((1, 75, 75), np.float32), np.zeros_like(XX2[0]), "sf")[0]
            raw_d, sf_d = joint_energy_direct(concat_cloud(X, ms[a], qs), concat_cloud(X, ms[b], qs))
            rel = abs(float(acc) - raw_d) / abs(raw_d)
            worst = max(worst, rel)
            cases.append(dict(set=name, pair=p, a=ms[a], b=ms[b], acc_raw=float(acc), direct_raw=raw_d, rel=rel,
                              acc_sf=float(sfa), direct_sf=sf_d))
    del X
    gc.collect()
    ok = worst <= TOL_DIRECT
    if not ok:
        stops.append(f"direct-computation check: max rel err {worst:.2e} > {TOL_DIRECT}")
    out["direct_check"] = dict(tolerance_rel=TOL_DIRECT, max_rel_err=worst, cases=cases, pass_=ok)
    print(f"[S0] direct computation (raw, {len(cases)} cases): max rel err {worst:.2e} -> {'PASS' if ok else 'FAIL'}",
          flush=True)

    # ---- (3) open-set code path vs D033 JG^fold; CENTROID-raw path vs D035
    fold, F33 = L.d033_folds()
    rows = []
    for F in F33:
        t = time.time()
        f = F["fold"]
        tr, ho, cols = L33.train_split(L.VIEW, fold, f)
        assert tr == F["train"] and len(cols) == F["n_pairs"]
        s, tr_ = L.fold_cloud(XY2, XX2, ia, ib, tr, cols, "sf")
        d = [abs(x["objective_cvar"] - y) for x, y in zip(tr_, F["jg_objective"])]
        ok = s == F["arms"]["JG"] and max(d) == 0.0
        if not ok:
            stops.append(f"fold {f}: JG^fold code path failed: {s} vs D033 {F['arms']['JG']}, max |dobj| {max(d)}")
        rows.append(dict(fold=f, chain=s, d033_chain=F["arms"]["JG"], identical=s == F["arms"]["JG"],
                         max_abs_objective_diff=max(d), n_pairs=len(cols), pass_=ok,
                         wall_min=round((time.time() - t) / 60, 2)))
        print(f"[S0] fold {f}: JG^fold {s} vs D033 {F['arms']['JG']}; max |dobj| {max(d):.1e} -> "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        gc.collect()
    del XY2, XX2
    out["code_path_open"] = rows
    D2 = L.centroid_raw()
    c35 = json.load(open("./results/D035/chains.json"))
    s, tr_ = L.fold_centroid(D2, np.arange(len(ia)))
    d = [abs(x["objective_cvar"] - y["objective_cvar"]) for x, y in zip(tr_, c35["traces"]["CENTROID|raw"])]
    ok = s == c35["chains"]["CENTROID|raw"] and max(d) == 0.0
    if not ok:
        stops.append(f"CENTROID-raw path failed: {s} vs D035 {c35['chains']['CENTROID|raw']}")
    out["code_path_centroid"] = dict(chain=s, d035_chain=c35["chains"]["CENTROID|raw"], max_abs_objective_diff=max(d),
                                     pass_=ok)
    print(f"[S0] CENTROID-raw (all pairs) {s} vs D035 {c35['chains']['CENTROID|raw']} -> {'PASS' if ok else 'FAIL'}",
          flush=True)

    out["stops"] = stops
    out["code_commit"] = L31.git_head()
    out["wall_min"] = round((time.time() - t0) / 60, 2)
    L31.jdump(out, f"{L.OUT}/s0.json")
    print(f"[S0] wrote {L.OUT}/s0.json ({out['wall_min']} min); stop conditions: {stops if stops else 'none'}",
          flush=True)


if __name__ == "__main__":
    main()
