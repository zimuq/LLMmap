"""D038 / S1 — full-S_build descriptives (target-free; seconds, no cubes loaded). Plan: docs/plans/D038-P1.md §6.

Per view, on the frozen sf tensor (sha asserted) and the frozen chains reproduced in S0 (d038_lib.ref_chains):
  M1  per chain at k = 8: mean s_q (single-query CVaR_0.1), mean pool rank (1 = strongest of 259), per-step s_q / rank;
      JG's picks not in MMR* vs MMR*'s picks not in JG: count and mean s_q.
  M2  per chain at k = 8: within-set Pearson / cosine, effective number (sum lam)^2 / sum lam^2 of the 8 x 8 Pearson
      matrix (+ its minimum eigenvalue), each pick's maximum Pearson with earlier picks.
  (a) the M1 / M2 table includes MMR|pearson|0.3 / 0.5.
  (c) JG's and MMR*'s non-shared picks: id, pool-metadata provenance / anchor_set / source (pool_v1.json has no
      `family` field), the first 120 characters of the text, s_q and pool rank.
Writes results/D038/full.json.
Usage:  PYTHONPATH=.:experiments python experiments/d038_full.py
"""
import json

import numpy as np

import d031_lib as L31
import d038_lib as L

CHAINS = ("JG", "MMR*", "TOP-K[CVaR]", "GC", "MMR|pearson|0.3", "MMR|pearson|0.5", "PAPER8")
POOL = "confs/queries/pool_v1.json"


def view_full(view, pool):
    F = L.frozen(view)
    ref = L.ref_chains(view)
    s, rank = L.strength(F)
    rec = dict(frozen_sha256=L.FROZEN_SHA[view], n_pairs=int(F.shape[1]), chains={})
    for name in CHAINS:
        c = ref[name]
        rec["chains"][name] = dict(
            chain=c, M1=dict(mean_s=float(s[c].mean()), mean_rank=float(rank[c].mean()),
                             step_s=[float(s[q]) for q in c], step_rank=[int(rank[q]) for q in c]),
            M2=L.redundancy(F, c))
    jg, mm = ref["JG"], ref["MMR*"]
    jg_only, mm_only = [q for q in jg if q not in mm], [q for q in mm if q not in jg]
    rec["nonshared"] = dict(n_jg_only=len(jg_only), n_mmr_only=len(mm_only),
                            mean_s_jg_only=float(s[jg_only].mean()), mean_s_mmr_only=float(s[mm_only].mean()))

    def row(q):
        p = pool[q]
        return dict(id=q, provenance=p.get("provenance"), anchor_set=p.get("anchor_set"), source=p.get("source"),
                    text=p["text"][:120], s_q=float(s[q]), rank=int(rank[q]))
    rec["secondary_c"] = dict(jg_only=[row(q) for q in jg_only], mmr_only=[row(q) for q in mm_only])
    print(f"[S1 {view}] " + "; ".join(f"{n}: s {v['M1']['mean_s']:.4f} rank {v['M1']['mean_rank']:.1f} "
                                      f"eff_n {v['M2']['eff_n']:.2f}" for n, v in rec["chains"].items()), flush=True)
    return rec


def main():
    pool = json.load(open(POOL))["queries"]
    assert len(pool) == L.NPOOL
    out = dict(schema="d038-full-v1", plan="docs/plans/D038-P1.md", s_val_reads=0, s_test_reads=0,
               pool=POOL, views={v: view_full(v, pool) for v in L.VIEWS}, code_commit=L31.git_head())
    L.jdump(out, f"{L.OUT}/full.json")


if __name__ == "__main__":
    main()
