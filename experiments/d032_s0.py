"""D032 / S0 — selection only (frozen scale-free tensors + E(q); no training, no S_val, no S_test).

Per view: assert the frozen inputs (D031 S0's assert_inputs + the shas D031 chains.json recorded; E(q) via
d009_lib.load_query_embeddings, the loader d031_train.Ctx uses for the readout); CVaR / MEAN scores; PP and
TXT similarities with their distributions; all 26 chains (k = 8, nested prefixes). Code-path asserts:
CVaR scores, TOP-K[CVaR] and the 6 MMR-PP[CVaR] chains equal D031 chains.json exactly. Training order
(d031_lib.train_order), identical-input index (any D031 deterministic chain's k-set, JG, PAPER8),
CVaR / MEAN degenerate flags, overlaps, D022 union flags, within-set similarities (secondary (d)).
Writes results/D032/chains.json (commit + push before S1).
Usage:  PYTHONPATH=.:experiments python experiments/d032_s0.py
"""
import os
import json
import time

import numpy as np
from scipy.stats import spearmanr

import d031_lib as L31
import d032_lib as L
from d031_s0 import assert_inputs
from d009_lib import load_query_embeddings, QEMB


def dist(S):
    off = S[np.triu_indices(S.shape[0], 1)]
    return dict(min=float(off.min()), p05=float(np.percentile(off, 5)), median=float(np.median(off)),
                p95=float(np.percentile(off, 95)), max=float(off.max()), p05_p95_width=float(np.percentile(off, 95) - np.percentile(off, 5)))


def within(S, sel):
    s = list(sel)
    return float(np.mean([S[a, b] for i, a in enumerate(s) for b in s[i + 1:]]))


def main():
    t0 = time.time()
    os.makedirs(L.OUT, exist_ok=True)
    c31 = json.load(open(f"{L31.OUT}/chains.json"))
    flags = {r["id"]: r for r in json.load(open("./results/D022/query_flags.json"))["rows"]}
    qe = load_query_embeddings()
    E = np.asarray(qe, np.float64)
    out = dict(schema="d032-chains-v1", frozen_before_training=True, plan="docs/plans/D032-P1.md",
               review="D032/P1 approved (04b2ea7): Call 2 -> (a), Call 3 as written",
               d031_chains_sha256=L31.sha(f"{L31.OUT}/chains.json"),
               query_embeddings=dict(path=QEMB, sha256=L31.sha(QEMB), shape=list(qe.shape),
                                     loader="d009_lib.load_query_embeddings (= d031_train.Ctx.qe, both views)"),
               views={})
    txt = {s: L.txt_similarity(E, s) for s in L.TXT_SIMS}
    for view in ("v2", "v1"):
        V31 = c31["views"][view]
        ms = L31.models(view)
        assert ms == V31["models"], view
        h = assert_inputs(view)
        assert h["sf_sha256"] == V31["inputs"]["sf_sha256"], (view, "sf tensor differs from D031")
        sf, _ = L31.tensors(view)
        ia, _ = L31.pairs(view)
        assert len(ia) == V31["n_pairs"] == L31.VIEW[view]["P"]
        jg = L31.jg_chain(view)
        assert jg == V31["arms"]["JG"]["sel"]

        rel = {c: L.rel(sf, c) for c in L.CRITS}
        assert rel["CVaR"].tolist() == V31["single_query_cvar"], "CVaR score differs from D031"
        pp = {s: L31.similarity(sf, s) for s in L.PP_SIMS}
        sims = {f"PP|{s}": pp[s] for s in L.PP_SIMS}
        sims.update({f"TXT|{s}": txt[s] for s in L.TXT_SIMS})
        for s in L.PP_SIMS:
            assert dist(pp[s])["median"] == V31["sim_distribution"][s]["median"]
        iu = np.triu_indices(L31.NPOOL, 1)
        rho = {f"{a}~{b}": float(spearmanr(sims[a][iu], sims[b][iu])[0])
               for i, a in enumerate(sims) for b in list(sims)[i + 1:]}

        sel = {}
        for c in L.CRITS:
            sel[f"TOP-K[{c}]"] = L.topk(rel[c])
            for s in L.PP_SIMS:
                for lam in L.LAMBDAS:
                    sel[f"MMR-PP[{c}]|{s}|{lam}"] = L.mmr(rel[c], pp[s], lam)
            for s in L.TXT_SIMS:
                for lam in L.LAMBDAS:
                    sel[f"MMR-TXT[{c}]|{s}|{lam}"] = L.mmr(rel[c], txt[s], lam)
        assert sorted(sel) == sorted(L.chain_names())
        sel = {n: sel[n] for n in L.chain_names()}
        repro = {}
        for n in sel:
            src = L.d031_name(n)
            if src is not None:
                assert sel[n] == V31["arms"][src]["sel"], (view, n, sel[n], V31["arms"][src]["sel"])
                repro[n] = src
        print(f"[S0 {view}] inputs asserted; CVaR scores and {len(repro)} reused chains equal D031 exactly", flush=True)

        d31det = {n: a["sel"] for n, a in V31["arms"].items() if a["kind"] == "det"}
        arms = {}
        for n, s in sel.items():
            tr = {}
            for k in range(1, L31.K_MAX + 1):
                q, ident = L31.train_order(s[:k], jg)
                src = sorted(f"D031:{m}" for m, t in d31det.items() if set(t[:k]) == set(s[:k]))
                dup = sorted(m for m, t in sel.items() if m != n and set(t[:k]) == set(s[:k]))
                tr[str(k)] = dict(q=q, ident=ident, d031_same_set=src, d032_same_set=dup)
            rule, rest = n.split("[", 1)
            crit = rest.split("]", 1)[0]
            v = n.split("|")[1:] if "|" in n else []
            arms[n] = dict(rule=rule, crit=crit, sim=v[0] if v else None, lam=float(v[1]) if v else None,
                           sel=s, reused_from_d031=repro.get(n), train=tr,
                           identical_input_k=[k for k in range(1, 9) if tr[str(k)]["ident"] or tr[str(k)]["d031_same_set"]])
        degen = {}
        for n in sel:
            if "[CVaR]" not in n:
                continue
            m = n.replace("[CVaR]", "[MEAN]")
            degen[f"{n} vs {m}"] = dict(overlap_k8=len(set(sel[n]) & set(sel[m])), equal_k8=set(sel[n]) == set(sel[m]),
                                        equal_by_k=[set(sel[n][:k]) == set(sel[m][:k]) for k in range(1, 9)])
        rank = V31["arms"]["HELM-RANK|sf"]["sel"]
        overlaps, uflags, wset = {}, {}, {}
        refs = {"JG": jg, "PAPER8": L31.PAPER8, "HELM-RANK|sf": rank}
        for n, s in list(refs.items()) + list(sel.items()):
            s8 = set(s[:8])
            o = dict(JG=len(s8 & set(jg)), PAPER8=len(s8 & set(L31.PAPER8)), HELM_RANK_sf=len(s8 & set(rank)),
                     TOPK_CVaR=len(s8 & set(sel["TOP-K[CVaR]"])), TOPK_MEAN=len(s8 & set(sel["TOP-K[MEAN]"])),
                     JG_by_k=[len(set(s[:k]) & set(jg[:k])) for k in range(1, 9)])
            if n in arms:
                cp = n.replace("[CVaR]", "[MEAN]") if "[CVaR]" in n else n.replace("[MEAN]", "[CVaR]")
                o["counterpart"] = len(s8 & set(sel[cp]))
            overlaps[n] = o
            uflags[n] = int(sum(bool(flags[q]["union_flag"]) for q in s[:8]))
            wset[n] = {sn: within(S, s[:8]) for sn, S in sims.items()}
        out["views"][view] = dict(
            models=ms, n_pairs=len(ia), inputs=h, jg=jg,
            scores={c: r.tolist() for c, r in rel.items()},
            score_rank_corr_cvar_mean=float(spearmanr(rel["CVaR"], rel["MEAN"])[0]),
            sim_distribution={n: dist(S) for n, S in sims.items()}, sim_rank_corr=rho,
            arms=arms, degenerate=degen, overlaps_k8=overlaps, d022_union_flags_k8=uflags,
            within_set_sim_k8=wset)
        for n, s in sel.items():
            print(f"[S0 {view}] {n:24s} {s}  JG∩{overlaps[n]['JG']} ident_k {arms[n]['identical_input_k']}", flush=True)
        for n, d in degen.items():
            print(f"[S0 {view}] {n}: overlap {d['overlap_k8']}/8{'  EQUAL' if d['equal_k8'] else ''}", flush=True)
        for n, d in out["views"][view]["sim_distribution"].items():
            print(f"[S0 {view}] sim {n}: " + ", ".join(f"{a} {b:.4f}" for a, b in d.items()), flush=True)
        print(f"[S0 {view}] sim rank corr {rho}", flush=True)
    out["code_commit"] = L31.git_head()
    out["wall_min"] = round((time.time() - t0) / 60, 2)
    L31.jdump(out, f"{L.OUT}/chains.json")
    print(f"[S0] wrote {L.OUT}/chains.json ({out['wall_min']} min)", flush=True)


if __name__ == "__main__":
    main()
