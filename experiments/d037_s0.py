"""D037 / S0 — code-path checks and inventory (run before the P1 Review, as D024's / D035's / D036's S0 were).

(1) gamma = 0.1 code paths: d037_lib.jg_chain (joint_greedy, cross-checked with greedy_cvar) reproduces D029's v2 JG
    chain k = 1..8 and gc_chain (greedy_cover) reproduces D029's GC chain, per-step |d objective| = 0.
(2) gamma = 1 reductions that need no new chain (P1 Call 1): JG -- at step 1 for every candidate and at every step of
    D029's JG chain, CVaR_1(s) equals the plain float64 mean of the per-pair statistic (|d| <= 1e-12, no NaN);
    GC -- CVaR_1(x) == mean on the tensor (verify_reduction's first assert) and on the coverage of every D029 GC
    prefix. The gamma = 1 chains themselves (verify_reduction's chain equality; JG's objective = mean at every step of
    its own chain) are S1.
(3) Inventory with sha256: D029 JG / GC / PAPER8 counts + logits + chains, the v2 sf tensor (= D029's recorded hash),
    D031 / D032 reference counts (TOP-K[CVaR], TOP-K[MEAN], MMR|pearson|0.7 = D031's v2 MMR star), D031 draws, v1
    grid sources for secondary (d).
(4) All-pair arrays for JG / GC / PAPER8 rebuilt from D029's stored logits (d031_analysis.Store.ref, which asserts the
    hard-pair columns = stored cnt_pair), plus top-1 recounts = stored cnt_total / cnt_model; both readouts, k = 1..8.
    Stored, already-read S_test logits are used for equality asserts only; nothing is summarised.
Writes results/D037/s0.json; exits 1 on any stop.
Usage:  OMP_NUM_THREADS=8 PYTHONPATH=.:experiments python experiments/d037_s0.py
"""
import os
import sys
import json
import time
import hashlib

import numpy as np

import d031_lib as L31
import d031_analysis as A
import d032_analysis as A32
import d037_lib as L
from LLMmap.joint_statistic import cvar
from LLMmap.joint_greedy import _stat_all_pairs

TOL_MEAN = 1e-12


def mean_check(s):
    assert not np.isnan(s).any(), "NaN in per-pair statistic"
    return abs(cvar(s, 1.0) - float(np.mean(np.asarray(s, np.float64))))


def inventory():
    files = ["results/D029/chains.json"] + [f"results/D029/{p}_{a}.npz" for p in ("counts", "logits")
                                            for a in ("JG", "GC", "PAPER8")]
    files += ["results/D031/chains.json", "results/D031/variant_choice.json", "results/D032/chains.json",
              "results/D027/n_prime.json", "results/D028/hard_sets_v2.json",
              "results/D010/selection.json", "results/D012/metrics_by_gamma_k.json",
              "results/D013/metrics_by_gamma_k.json", "results/D013/cross_algorithm_comparison.json",
              "results/D033/chains.json", "results/D035/chains.json", "results/D036/chains.json"]
    for pat in ("D031/counts_v2_att", "D031/counts_v2_lin", "D032/counts_v2_att", "D032/counts_v2_lin"):
        d, p = pat.split("/")
        files += sorted(f"results/{d}/{f}" for f in os.listdir(f"results/{d}") if f.startswith(p))
    sha, stops = {}, []
    for f in files:
        if os.path.exists(f):
            sha[f] = L31.sha(f)
        else:
            stops.append(f"missing input {f}")
    return sha, stops


def recount(lg, y, n):
    pred = lg.argmax(1)
    ok = (pred == y).reshape(n, A.NC)
    return ok.sum(0), ok.astype(np.int64)


def main():
    t0 = time.time()
    os.makedirs(L.OUT, exist_ok=True)
    out, stops = dict(schema="d037-s0-v1", view=L.VIEW, s_val_reads=0, s_test_reads="none new; D029 stored logits "
                      "for equality asserts only"), []
    ref = L.d029_chains()

    # (3) inventory
    out["sha256"], st = inventory()
    stops += st
    vc = json.load(open("results/D031/variant_choice.json"))["views"]["v2"]
    out["mmr_reference"] = vc["MMR_star"]
    out["tensor_sha256"] = L.D029_TENSOR_SHA
    S = L.tensor()
    print(f"[S0] inventory: {len(out['sha256'])} files, {len(st)} missing; tensor sha = D029's", flush=True)

    # (1) + (2) GC
    gc, gtr = L.gc_chain(S, 0.1)
    dg = max(abs(a["objective"] - b["objective"]) for a, b in zip(gtr, ref["GC_trace"]))
    out["gc_gamma01"] = dict(chain=gc, d029=ref["GC"], equal=gc == ref["GC"], max_abs_dobj=dg)
    if gc != ref["GC"] or dg != 0:
        stops.append(f"GC gamma=0.1 path: {gc} vs D029 {ref['GC']}, max |d obj| {dg}")
    x = np.asarray(S)[:, :50].ravel()[:200]
    dm = [abs(cvar(x, 1.0) - float(np.mean(x, dtype=np.float64)))]
    cov = np.zeros(S.shape[1])
    for q in ref["GC"]:
        cov = np.maximum(cov, S[q])
        dm.append(abs(cvar(cov, 1.0) - float(cov.mean())))
    out["gc_gamma1_reduction_no_chain"] = dict(max_abs_d=max(dm), tol=TOL_MEAN)
    if max(dm) > TOL_MEAN:
        stops.append(f"GC CVaR_1 != mean: {max(dm)}")
    print(f"[S0] GC gamma=0.1 chain {gc} = D029: {gc == ref['GC']} (max |d obj| {dg}); CVaR_1 = mean max |d| "
          f"{max(dm):.1e}", flush=True)

    # (1) + (2) JG
    t = time.time()
    XY2, XX2, ia, ib = L.accumulators()
    print(f"[S0] accumulators {(time.time() - t) / 60:.1f} min", flush=True)
    jg, jtr = L.jg_chain(XY2, XX2, ia, ib, 0.1)
    dj = max(abs(a["objective_cvar"] - b["objective_cvar"]) for a, b in zip(jtr, ref["JG_trace"]))
    out["jg_gamma01"] = dict(chain=jg, d029=ref["JG"], equal=jg == ref["JG"], max_abs_dobj=dj,
                             ties=[t_["n_exact_ties_at_max"] for t_ in jtr])
    if jg != ref["JG"] or dj != 0:
        stops.append(f"JG gamma=0.1 path: {jg} vs D029 {ref['JG']}, max |d obj| {dj}")
    print(f"[S0] JG gamma=0.1 chain {jg} = D029: {jg == ref['JG']} (max |d obj| {dj})", flush=True)
    nc = XY2.shape[2]
    acc_xy, acc_xx = np.zeros((XY2.shape[1], nc, nc), np.float32), np.zeros((XX2.shape[1], nc, nc), np.float32)
    d1 = max(mean_check(_stat_all_pairs(acc_xy, acc_xx, ia, ib, XY2[q], XX2[q], "energy")) for q in range(L.NPOOL))
    dk = []
    for q in ref["JG"]:
        dk.append(mean_check(_stat_all_pairs(acc_xy, acc_xx, ia, ib, XY2[q], XX2[q], "energy")))
        acc_xy += XY2[q]
        acc_xx += XX2[q]
    out["jg_gamma1_reduction_no_chain"] = dict(step1_all_candidates_max_abs_d=d1, d029_chain_steps_max_abs_d=max(dk),
                                               tol=TOL_MEAN, nan_free=True)
    if max(d1, max(dk)) > TOL_MEAN:
        stops.append(f"JG CVaR_1 != mean: {d1}, {max(dk)}")
    print(f"[S0] JG CVaR_1 = mean: step 1 all candidates {d1:.1e}, D029 chain steps {max(dk):.1e}", flush=True)
    del XY2, XX2

    # (4) all-pair arrays rebuilt + recounts
    st = A.Store("v2")
    n = len(L31.models(L.VIEW))
    rb = {}
    for arm in ("JG", "GC", "PAPER8"):
        z = np.load(f"./results/D029/counts_{arm}.npz")
        lg = np.load(f"./results/D029/logits_{arm}.npz")
        y = lg["y"]
        ok = True
        for k in range(1, L.K + 1):
            for ro in ("att", "lin"):
                u = st.ref(ro, arm, k)                      # asserts pair[hard cols] == stored cnt_pair
                assert u.pair is not None
                for i, r in enumerate(range(5) if ro == "att" else [None]):
                    p = f"att|{k}|{r}" if ro == "att" else f"lin|{k}"
                    tot, mod = recount(lg[p], y, n)
                    ok &= np.array_equal(tot, z[f"{p}|cnt_total"]) and np.array_equal(mod, z[f"{p}|cnt_model"])
        rb[arm] = dict(pair_all_rebuilt=True, recount_equal=bool(ok))
        if not ok:
            stops.append(f"D029 {arm} top-1 recount differs from stored counts")
    out["d029_all_pair_rebuild"] = rb
    print(f"[S0] D029 all-pair rebuild + recounts: {rb}", flush=True)

    # reference rows present (k = 1..8, both readouts, with all-pair counts)
    s31, s32 = A.Store("v2"), A32.Store32("v2")
    refs = {"TOP-K[CVaR]": (s31, "TOP-K"), "MMR-PP[CVaR]": (s31, vc["MMR_star"]), "TOP-K[MEAN]": (s32, "TOP-K[MEAN]")}
    pres = {}
    for nm, (s, key) in refs.items():
        try:
            for k in range(1, L.K + 1):
                for ro in ("att", "lin"):
                    assert s.new(ro, key, k).pair is not None
            pres[nm] = dict(key=key, complete=True)
        except (KeyError, AssertionError) as e:
            pres[nm] = dict(key=key, complete=False, error=repr(e))
            stops.append(f"reference {nm} incomplete: {e!r}")
    out["reference_rows"] = pres
    B = A.Boot("v2")
    out["boot_configs_sha256"] = hashlib.sha256(np.ascontiguousarray(B.M).tobytes()).hexdigest()
    d36 = json.load(open("results/D036/analysis.json"))["bootstrap"]["configs_sha256"]
    out["boot_equals_d036"] = out["boot_configs_sha256"] == d36
    if not out["boot_equals_d036"]:
        stops.append("D031 v2 draws differ from D036's")
    print(f"[S0] reference rows {pres}; draws = D036's: {out['boot_equals_d036']}", flush=True)

    out.update(stops=stops, code_commit=L31.git_head(), wall_min=round((time.time() - t0) / 60, 2))
    L31.jdump(out, f"{L.OUT}/s0.json")
    print(f"[S0] {'STOP: ' + '; '.join(stops) if stops else 'all checks pass'} ({out['wall_min']} min)", flush=True)
    sys.exit(1 if stops else 0)


if __name__ == "__main__":
    main()
