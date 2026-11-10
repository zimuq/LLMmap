"""D038 / S2 — the two-halves test, one view per process (target-free). Plan: docs/plans/D038-P1.md §7 + Review A1/A2.

  7.1  S_build cubes; path 1 from energies_<view>.npz (sha == s0.json); path 2 accumulators.
  7.2  per split r: restrict the accumulators to A_r and B_r, reselect JG / MMR* / TOP-K[CVaR] / GC on each half.
       Sequential in this process (no fork after BLAS has run; D031 lin_threads_check segfault precedent).
  7.3  per directed split (r, d), sel s, meas m, every rule's chain_s and PAPER8:
         M1  held-out s_q(m): mean, mean pool rank, per step; in-sample s_q(s) alongside; non-shared JG / MMR* picks.
         M2  redundancy on m (and on s, descriptive).
         M3  Y_joint (path 2) and Y_max (path 1) for k = 1..8 on s and m (optimism, per-step gains and the A2 held-out
             level are derived from these in S3b). Assert: Y_joint(JG chain_s, s) == JG's own trace exactly.
             Chased pairs (Call 2 (a)) for JG and MMR* at steps j = 4..8: Ch_j = bottom n_p pairs (stable argsort) of
             J(c[:j-1], s); Low = bottom n_p of the pool ceiling c_p = max_q E[m][q, p].
               share_low = |Ch_j & Low| / n_p (descriptive after A1; v2 also H_all / N' shares);
               A1 (iii'): o_p = g_p(s) - g_p(m), g_p(C) = J_p(c[:j], C) - J_p(c[:j-1], C);
                          delta_j = mean o over Ch_j & Low - mean o over Ch_j \\ Low (None if either is empty);
                          delta = mean of the non-None delta_j (None if all skipped); n_skipped recorded.
         M4  readout (2) raw, references from s, probes = every m config (d038_lib.readout2): (P_all, P_tail), k = 1..8.
  7.4  plotdata_<view>.npz for the I1 tail plot (S3a).
Outputs (results/D038/): splits_<view>.json, measures_<view>.json, plotdata_<view>.npz, pairstats_<view>.npz (per-pair
J for JG / MMR* on s and m, every directed split; lets A1 be recomputed). No summary or reading is computed here.
Usage:  OMP_NUM_THREADS=32 PYTHONPATH=.:experiments python experiments/d038_splits.py v2
"""
import sys
import time
import json

import numpy as np

import d031_lib as L31
import d038_lib as L

RULES = ("JG", "MMR*", "TOP-K[CVaR]", "GC")
CHASE = ("JG", "MMR*")
JS = range(4, L.K + 1)


def hard_sets(view, ms, ia, ib):
    """v2 only: pair indices of D028 H_all (57) and D027 N' (161)."""
    if view != "v2":
        return {}
    ix = {(ms[a], ms[b]): p for p, (a, b) in enumerate(zip(ia, ib))}

    def idx(names):
        out = []
        for n in names:
            a, b = n.split(" | ")
            out.append(ix[(a, b)] if (a, b) in ix else ix[(b, a)])
        return np.array(sorted(out))
    H = idx(json.load(open("./results/D028/hard_sets_v2.json"))["H_all"])
    N = idx([p["pair"] for p in json.load(open("./results/D027/n_prime.json"))["pairs"]])
    assert (len(H), len(N)) == (57, 161)
    return {"H_all": H, "N_prime": N}


def select(XY2h, XX2h, Eh, ia, ib):
    jg, tr = L.jg(XY2h, XX2h, ia, ib)
    mm, mt = L.mmr(Eh, L.MMR_STAR[1])
    tk, tt = L.topk(Eh)
    g, gt = L.gc(Eh)
    chains = {"JG": jg, "MMR*": mm, "TOP-K[CVaR]": tk, "GC": g}
    info = {"JG": dict(trace=[t["objective_cvar"] for t in tr], n_exact_ties=[t["n_exact_ties_at_max"] for t in tr]),
            "MMR*": mt, "TOP-K[CVaR]": tt, "GC": gt}
    return chains, info


def chase(st_s, st_m, low, n_p, hs):
    """Call 2 (a) chased sets and A1's transfer statistic for one rule's chain_s; st_* = (8, P) per-prefix J."""
    lowm = np.zeros(st_s.shape[1], bool)
    lowm[low] = True
    rows, deltas = [], []
    for j in JS:
        ch = np.argsort(st_s[j - 2], kind="stable")[:n_p]
        o = (st_s[j - 1].astype(np.float64) - st_s[j - 2]) - (st_m[j - 1].astype(np.float64) - st_m[j - 2])
        inl = lowm[ch]
        dj = float(o[ch[inl]].mean() - o[ch[~inl]].mean()) if inl.any() and (~inl).any() else None
        row = dict(j=j, n_low=int(inl.sum()), share_low=float(inl.sum() / n_p), delta=dj)
        for k, v in hs.items():
            row[f"share_{k}"] = float(np.isin(ch, v).sum() / n_p)
        rows.append(row)
        if dj is not None:
            deltas.append(dj)
    return dict(steps=rows, delta=float(np.mean(deltas)) if deltas else None, n_skipped=len(JS) - len(deltas),
                share_low_mean=float(np.mean([r["share_low"] for r in rows])),
                **{f"share_{k}_mean": float(np.mean([r[f"share_{k}"] for r in rows])) for k in hs})


def measure(ch, H, E, s_i, m_i, ia, ib, n_p, hs, jg_trace):
    s_m, rank_m = L.strength(E[m_i])
    s_s, _ = L.strength(E[s_i])
    rec = dict(M1={}, M2={}, M3={}, chase={})
    stats = {}
    for name, c in ch.items():
        rec["M1"][name] = dict(mean_s_m=float(s_m[c].mean()), mean_rank_m=float(rank_m[c].mean()),
                               step_s_m=[float(s_m[q]) for q in c], mean_s_s=float(s_s[c].mean()),
                               step_s_s=[float(s_s[q]) for q in c])
        rec["M2"][name] = dict(m=L.redundancy(E[m_i], c), s=L.redundancy(E[s_i], c))
        yj_s, st_s = L.y_joint(*H[s_i], ia, ib, c)
        yj_m, st_m = L.y_joint(*H[m_i], ia, ib, c)
        rec["M3"][name] = dict(yj_s=yj_s, yj_m=yj_m, ymax_s=L.y_max(E[s_i], c), ymax_m=L.y_max(E[m_i], c))
        if name == "JG":
            assert yj_s == jg_trace, ("STOP: Y_joint(JG chain_s, s) != JG trace", yj_s, jg_trace)
        if name in CHASE:
            stats[name] = (st_s, st_m)
    jo = [q for q in ch["JG"] if q not in ch["MMR*"]]
    mo = [q for q in ch["MMR*"] if q not in ch["JG"]]
    rec["M1"]["nonshared"] = dict(n_jg_only=len(jo), n_mmr_only=len(mo),
                                  mean_s_m_jg_only=float(s_m[jo].mean()) if jo else None,
                                  mean_s_m_mmr_only=float(s_m[mo].mean()) if mo else None)
    cp = E[m_i].max(0)
    low = np.argsort(cp, kind="stable")[:n_p]
    for name, (st_s, st_m) in stats.items():
        rec["chase"][name] = chase(st_s, st_m, low, n_p, hs)
    return rec, stats


def main(view):
    t0 = time.time()
    s0 = json.load(open(f"{L.OUT}/s0.json"))
    ms, X = L.build_cubes(view)
    ia, ib = L31.pairs(view)
    P = len(ia)
    n_p = int(np.floor(L.GAMMA * P))
    ep = L.energies_path(view)
    assert L31.sha(ep) == s0["views"][view]["path1"]["energies_sha256"], f"STOP: {ep} differs from S0"
    E = np.load(ep)["sf"]
    sets = L.config_sets()
    assert E.shape == (len(sets), L.NPOOL, P)
    t = time.time()
    XY2, XX2, ia, ib = L.accumulators(view, ms, X)
    t_acc = round((time.time() - t) / 60, 2)
    hs = hard_sets(view, ms, ia, ib)
    print(f"[S2 {view}] loaded; accumulators {t_acc} min; n_p {n_p}", flush=True)

    halves, meas, pst = {}, [], {}
    t_sel = t_meas = t_ro = 0.0
    for r in range(L.R):
        H, ch = {}, {}
        for h in (1 + 2 * r, 2 + 2 * r):
            t = time.time()
            H[h] = (L.restrict(XY2, sets[h]), L.restrict(XX2, sets[h]))
            ch[h], info = select(*H[h], E[h], ia, ib)
            halves[h] = dict(set_index=h, r=r, half="AB"[h - 1 - 2 * r], chains=ch[h], info=info)
            t_sel += time.time() - t
        for d in (0, 1):
            _, _, s_i, m_i = L.directed()[2 * r + d]
            c = {**ch[s_i], "PAPER8": L.PAPER8}
            t = time.time()
            rec, stats = measure(c, H, E, s_i, m_i, ia, ib, n_p, hs, halves[s_i]["info"]["JG"]["trace"])
            t_meas += time.time() - t
            t = time.time()
            rec["M4"] = L.readout2(X, ms, ia, ib, sets[s_i], sets[m_i], c)
            t_ro += time.time() - t
            meas.append(dict(r=r, d=d, sel=s_i, meas=m_i, chains=c, **rec))
            for name, (st_s, st_m) in stats.items():
                pst[f"{r}|{d}|{name}|s"], pst[f"{r}|{d}|{name}|m"] = st_s, st_m
            if r == 0 and d == 0:
                plot_jg_a0 = (np.sort(stats["JG"][0][-1]), np.sort(stats["JG"][1][-1]))
        del H
        print(f"[S2 {view}] split {r} done {(time.time() - t0) / 60:.1f} min; JG A {ch[1 + 2 * r]['JG']} "
              f"B {ch[2 + 2 * r]['JG']}", flush=True)

    ref = L.ref_chains(view)
    _, st_full = L.y_joint(XY2, XX2, ia, ib, ref["JG"])
    np.savez_compressed(f"{L.OUT}/plotdata_{view}.npz",
                        cp_full=np.sort(E[0].max(0)), cp_A0=np.sort(E[1].max(0)), cp_B0=np.sort(E[2].max(0)),
                        jg_full=np.sort(st_full[-1]), jg_A0_on_A0=plot_jg_a0[0], jg_A0_on_B0=plot_jg_a0[1])
    np.savez_compressed(f"{L.OUT}/pairstats_{view}.npz", **pst)
    stab = [{name: dict(k8=len(set(halves[1 + 2 * r]["chains"][name]) & set(halves[2 + 2 * r]["chains"][name])),
                        k4=len(set(halves[1 + 2 * r]["chains"][name][:4]) & set(halves[2 + 2 * r]["chains"][name][:4])))
             for name in RULES} for r in range(L.R)]
    cost = dict(accumulators_min=t_acc, selection_min=round(t_sel / 60, 2), measures_min=round(t_meas / 60, 2),
                readout2_min=round(t_ro / 60, 2), wall_min=round((time.time() - t0) / 60, 2))
    base = dict(plan="docs/plans/D038-P1.md (+ Review A1 / A2)", view=view, s_val_reads=0, s_test_reads=0,
                n_pairs=P, n_p=n_p, code_commit=L31.git_head())
    L.jdump(dict(schema="d038-splits-v1", **base, R=L.R, seed0=L.SEED0,
                 rule="default_rng(seed0 + r).permutation(75); A = sorted(first 37), B = sorted(last 38)",
                 sets=[s.tolist() for s in sets], directed=L.directed(),
                 halves=[halves[h] for h in sorted(halves)], stability=stab), f"{L.OUT}/splits_{view}.json")
    L.jdump(dict(schema="d038-measures-v1", **base, hard_sets={k: v.tolist() for k, v in hs.items()},
                 splits=meas, cost=cost), f"{L.OUT}/measures_{view}.json")
    print(f"[S2 {view}] done; {cost}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
