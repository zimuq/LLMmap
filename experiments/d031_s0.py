"""D031 / S0 — selection only (S_build objects; no training, no S_val, no S_test).

Per view: assert frozen inputs; Helm's E = the frozen raw tensor, asserted against
joint_statistic.energy_direct on 3 queries x 50 pairs (raw rel <= 1e-5; raw/within vs the scale-free
tensor <= 1e-6, the D's assert); ordered-vs-unordered SVD equivalence asserted; SVD / r_hat / GMM for
E in {raw, sf}; chains and draws for every arm (Review A1: MMR x {cos, pearson} x 3 lambda); JG
reproduced from the frozen cubes by exact list equality; training order + identical-input k.
Writes results/D031/chains.json (commit + push before S1).
Usage:  PYTHONPATH=.:experiments python experiments/d031_s0.py
"""
import os
import json
import time
import itertools

import numpy as np

import d031_lib as L
from LLMmap.joint_statistic import energy_direct
from LLMmap.joint_greedy import joint_greedy
from d010_select import precompute


def assert_inputs(view):
    h = {}
    if view == "v2":
        import d029_lib as L29
        h.update(L29.assert_inputs())
        meta = json.load(open("./results/D028/tensor_v2.json"))
        assert L.sha(L.VIEW["v2"]["raw"]) == meta["sha256"]["energy_raw"]
        assert L.sha(L.VIEW["v2"]["sf"]) == meta["sha256"]["energy_sf"]
    else:
        ms = L.models("v1")
        assert len(ms) == 37 and ms == sorted(ms)
    h["sf_sha256"] = L.sha(L.VIEW[view]["sf"])
    h["raw_sha256"] = L.sha(L.VIEW[view]["raw"])
    return h


def e_asserts(view, sf, raw, X, ms):
    ia, ib = L.pairs(view)
    P = len(ia)
    ps = np.random.default_rng(20261012).choice(P, 50, replace=False)
    worst_raw, worst_sf = 0.0, 0.0
    for q in (0, 193, 258):
        for p in ps:
            A = np.asarray(X[ms[ia[p]]][q], np.float32)
            B = np.asarray(X[ms[ib[p]]][q], np.float32)
            r, s = energy_direct(A, B)
            worst_raw = max(worst_raw, abs(r - raw[q, p]) / abs(r))
            worst_sf = max(worst_sf, abs(s - sf[q, p]))
    assert worst_raw <= 1e-5 and worst_sf <= 1e-6, (worst_raw, worst_sf)
    # ordered (n^2 columns, zero diagonal) vs unordered: same left singular vectors, sigma x sqrt(2)
    n = len(ms)
    pid = {(a, b): j for j, (a, b) in enumerate(zip(ia, ib))}
    cols = []
    for i in range(n):
        for j in range(n):
            cols.append(-1 if i == j else pid[(min(i, j), max(i, j))])
    cols = np.array(cols)
    E = np.asarray(raw, np.float64)
    Eo = np.where(cols[None, :] >= 0, E[:, np.maximum(cols, 0)], 0.0)
    Uu, su, _ = np.linalg.svd(E, full_matrices=False)
    Uo, so, _ = np.linalg.svd(Eo, full_matrices=False)
    rel_s = float(np.max(np.abs(so - np.sqrt(2) * su) / (np.sqrt(2) * su)))
    G = np.abs(Uo[:, :20].T @ Uu[:, :20])
    dev = float(np.max(np.abs(G - np.eye(20))))
    assert rel_s <= 1e-10 and dev <= 1e-8, (rel_s, dev)
    return dict(n_cells=150, raw_max_rel=worst_raw, sf_max_abs=worst_sf, ordered_sigma_rel=rel_s,
                ordered_U_first20_dev=dev, pairs_checked=ps.tolist())


def det_entry(arm, sel, jg, extra=None):
    tr = {}
    for k in range(1, L.K_MAX + 1):
        q, ident = L.train_order(sel[:k], jg)
        tr[str(k)] = dict(q=q, ident=ident)
    return dict(kind="det", arm=arm, sel=sel, train=tr,
                identical_input_k=[k for k in range(1, L.K_MAX + 1) if tr[str(k)]["ident"]], **(extra or {}))


def draw_entry(arm, view, variant, base, jg):
    ds = L.draws(arm, view, variant, base)
    trs = []
    for d in ds:
        tr = {}
        for k in L.DRAW_KS:
            if k <= len(d):
                q, ident = L.train_order(d[:k], jg)
                tr[str(k)] = dict(q=q, ident=ident)
        trs.append(tr)
    return dict(kind="draw", arm=arm, variant=variant, base_size=len(base), draws=ds, train=trs,
                ks_defined=[k for k in L.DRAW_KS if k <= len(base)],
                seed_string=f"D031|{arm}|{view}|{variant}",
                identical_inputs=[[d, k, t["ident"]] for d, tr in enumerate(trs) for k, t in tr.items() if t["ident"]])


def main():
    t0 = time.time()
    os.makedirs(L.OUT, exist_ok=True)
    flags = {r["id"]: r for r in json.load(open("./results/D022/query_flags.json"))["rows"]}
    texts = [q["text"] for q in json.load(open("./confs/queries/pool_v1.json"))["queries"]]
    out = dict(schema="d031-chains-v1", frozen_before_training=True, plan="docs/plans/D031-P1.md",
               review="D031/P1 approved with amendments A1-A3 (7c86dcd)", views={})
    for view in ("v2", "v1"):
        tv = time.time()
        ms = L.models(view)
        h = assert_inputs(view)
        sf, raw = L.tensors(view)
        print(f"[S0 {view}] inputs asserted; {len(ms)} models, tensors {sf.shape}", flush=True)
        Xb = L.cubes(view, ("build",))["build"][1]
        ea = e_asserts(view, sf, raw, Xb, ms)
        print(f"[S0 {view}] E asserts: {ea}", flush=True)

        # JG code-path reproduction (exact list equality)
        jg = L.jg_chain(view)
        ia, ib = L.pairs(view)
        prs = list(itertools.combinations(ms, 2))
        Xf = {m: Xb[m] for m in ms}
        tj = time.time()
        XX2, XY2 = precompute(Xf, ms, prs)
        rep, trace = joint_greedy(XY2, XX2, ia, ib, L.K_MAX, L.GAMMA, "energy", verbose=False)
        del XX2, XY2
        assert [int(q) for q in rep] == jg, (view, rep, jg)
        print(f"[S0 {view}] JG reproduced exactly: {jg} ({(time.time() - tj) / 60:.1f} min)", flush=True)
        del Xb, Xf

        helm = {e: L.helm(raw if e == "raw" else sf) for e in L.E_VARIANTS}
        for e, hh in helm.items():
            print(f"[S0 {view}] HELM E={e}: r_hat {hh['r_hat']} (ratio {hh['r_hat_ratio']:.3f}; runner-up s="
                  f"{hh['runner_up_s']} {hh['runner_up_ratio']:.3f}; cap20 {hh['r_hat_cap20']}), "
                  f"|Signal| {hh['n_signal']}, |Orth| {hh['n_orthogonal']}, "
                  f"GMM K {[d['K'] for d in hh['gmm_abs']]}", flush=True)

        arms = {"JG": dict(kind="ref", sel=jg), "PAPER8": dict(kind="ref", sel=L.PAPER8)}
        tk, rel = L.topk(sf)
        arms["TOP-K"] = det_entry("TOP-K", tk, jg)
        for sim in L.SIMS:
            for lam in L.LAMBDAS:
                arms[f"MMR|{sim}|{lam}"] = det_entry("MMR", L.mmr(sf, lam, sim), jg, dict(sim=sim, lam=lam))
        for e in L.E_VARIANTS:
            arms[f"HELM-RANK|{e}"] = det_entry("HELM-RANK", helm[e]["rank"][:L.K_MAX], jg,
                                               dict(E=e, label="extension of Helm et al. limitation (iii)"))
            arms[f"HELM-SIG|{e}"] = draw_entry("HELM-SIG", view, e, helm[e]["signal"], jg)
            arms[f"HELM-ORTH|{e}"] = draw_entry("HELM-ORTH", view, e, helm[e]["orthogonal"], jg)
        arms["RANDOM"] = draw_entry("RANDOM", view, "-", list(range(L.NPOOL)), jg)

        sims = {}
        for sim in L.SIMS:
            S = L.similarity(sf, sim)
            off = S[np.triu_indices(L.NPOOL, 1)]
            sims[sim] = dict(min=float(off.min()), p05=float(np.percentile(off, 5)), median=float(np.median(off)),
                             p95=float(np.percentile(off, 95)), max=float(off.max()))
        overlaps, uflags = {}, {}
        for name, a in arms.items():
            if a["kind"] in ("ref", "det"):
                s8 = a["sel"][:8]
                overlaps[name] = dict(JG=len(set(s8) & set(jg)), PAPER8=len(set(s8) & set(L.PAPER8)),
                                      TOPK=len(set(s8) & set(tk)),
                                      JG_by_k=[len(set(a["sel"][:k]) & set(jg[:k])) for k in range(1, 9)])
                uflags[name] = int(sum(bool(flags[q]["union_flag"]) for q in s8))
            else:
                d8 = [d[:8] for d in a["draws"] if len(d) >= 8]
                if d8:
                    overlaps[name] = dict(JG_mean=float(np.mean([len(set(d) & set(jg)) for d in d8])),
                                          PAPER8_mean=float(np.mean([len(set(d) & set(L.PAPER8)) for d in d8])))
                    uflags[name] = float(np.mean([sum(bool(flags[q]["union_flag"]) for q in d) for d in d8]))
        used = sorted({q for a in arms.values() for q in (a.get("sel") or [])})
        out["views"][view] = dict(
            models=ms, n_pairs=len(ia), inputs=h, e_asserts=ea, jg_reproduced=True, jg_trace=trace,
            helm=helm, single_query_cvar=rel.tolist(),
            topk_first_equals_jg_first=bool(tk[0] == jg[0]),
            sim_distribution=sims, arms=arms, overlaps_k8=overlaps, d022_union_flags_k8=uflags,
            query_texts={str(q): texts[q][:160] for q in used}, wall_min=round((time.time() - tv) / 60, 1))
        print(f"[S0 {view}] TOP-K {tk}; JG {jg}; MMR " +
              "; ".join(f"{n.split('|',1)[1]} {a['sel']}" for n, a in arms.items() if n.startswith("MMR")) +
              f"; RANK raw {arms['HELM-RANK|raw']['sel']} sf {arms['HELM-RANK|sf']['sel']}", flush=True)
        print(f"[S0 {view}] sims {sims}", flush=True)
    out["code_commit"] = L.git_head()
    out["wall_min"] = round((time.time() - t0) / 60, 1)
    L.jdump(out, f"{L.OUT}/chains.json")
    print(f"[S0] wrote {L.OUT}/chains.json ({out['wall_min']} min)", flush=True)


if __name__ == "__main__":
    main()
