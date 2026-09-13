"""D011 / S1–S5 — does JointGreedy's chain recover D008's flagged pairs?

Population (corrected, approved 2026-09-13): the **32 split-half-robust** pairs
from `results/D011/premise_check.json` -- those with a pool query clearing 0.95
2-way test accuracy on BOTH independent halves of the test configs. The naive
"33" was inflated by picking and scoring the oracle query on the same split.

What the flag actually means (TACC_NOTES issue 13): the tensor's own per-pair
top-ranked query, chosen on build, generalizes poorly to test for these pairs,
while a different pool query does much better. It is a per-pair build->test
generalization gap, NOT a statement about any chain's shared 8-query budget.

Criteria, both on D008's own T2 scale so they are commensurable with the flag
(approved; F2/Call 1):
  (a) max over the chain's 8 queries of SINGLE-query 2-way test accuracy >= 0.75
  (b) the SET's 2-way test accuracy >= 0.75 while (a) fails -- the 8 together
      resolving what no member resolves alone
Recovery = JointGreedy clears T2 (via a or b) and GreedyCover does not.

(a) takes a max over 8, so it carries its own small winner's curse; it is
therefore reported naively AND split-half (F3), applying to this D's own new
computation the lesson the premise check just taught.

Usage:  PYTHONPATH=.:experiments python experiments/d011_recovery.py
"""
import os
import json
import hashlib
import itertools

import numpy as np

from LLMmap.joint_statistic import energy_direct
from d008_lib import (build_dq, summed, two_way_correct, load_tensor,
                      structural_diagnostics)
from d007_lib import load_corpus

OUT = "./results/D011"
T2_ACC = 0.75
T3_ENERGY = 0.8761          # D008/S7's T3 theta, secondary view only
SPLIT_SEED = 20260913


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    S, meta = load_tensor()
    models = meta["models"]
    pairs = list(itertools.combinations(models, 2))
    name2ix = {f"{a} | {b}": i for i, (a, b) in enumerate(pairs)}

    prem = json.load(open(f"{OUT}/premise_check.json"))
    pop = [r for r in prem["pairs"] if r["both_halves_max"] >= 0.95]
    assert len(pop) == 32, f"expected 32 split-half-robust pairs, got {len(pop)}"
    ix = [name2ix[r["pair"]] for r in pop]
    print(f"[S1] population: {len(ix)} split-half-robust pairs "
          f"(naive count was {prem['naive_oracle_ge95']})", flush=True)

    d008 = json.load(open("./results/D008/selection.json"))
    d010 = json.load(open("./results/D010/selection.json"))
    chains = {"GreedyCover": d008["conditions"]["cvar_max"]["queries"][:8],
              "JointGreedy": d010["joint_energy"]["queries"][:8]}
    print(f"[S1] chains: GreedyCover {chains['GreedyCover']}\n"
          f"             JointGreedy {chains['JointGreedy']}  "
          f"(overlap {len(set(chains['GreedyCover']) & set(chains['JointGreedy']))}/8)",
          flush=True)

    Dq, w2, y_ev, y_rf, _, cfg_ev = build_dq(eval_pool="test")
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(25)
    hA, hB = set(perm[:12].tolist()), set(perm[12:24].tolist())
    mA, mB = np.isin(cfg_ev, list(hA)), np.isin(cfg_ev, list(hB))

    # joint-statistic null at k=8, per chain: one model against itself
    _, Xb = load_corpus(pool="build")
    def joint_sf(qs, a, b, ia_rows=None, ib_rows=None):
        A = np.concatenate([np.asarray(Xb[a][q], np.float32) for q in qs], 1)
        B = np.concatenate([np.asarray(Xb[b][q], np.float32) for q in qs], 1)
        if ia_rows is not None:
            A, B = A[ia_rows], B[ib_rows]
        return energy_direct(A, B)[1]

    nulls = {}
    for cname, qs in chains.items():
        v = []
        for m in models:
            p = np.random.default_rng(hash(m) % 2**31).permutation(75)
            C = np.concatenate([np.asarray(Xb[m][q], np.float32) for q in qs], 1)
            v.append(energy_direct(C[p[:37]], C[p[37:74]])[1])
        nulls[cname] = float(np.percentile(v, 95))
        print(f"[S3] {cname} joint self-pair null p95 (k=8): {nulls[cname]:.4f}",
              flush=True)

    rows = []
    for r, j in zip(pop, ix):
        a, b = pairs[j]
        ia, ib = models.index(a), models.index(b)
        rec = dict(pair=r["pair"], naive_oracle=r["naive_oracle"],
                   both_halves_max=r["both_halves_max"],
                   build_qstar_acc=r["build_qstar_acc"],
                   median_query_acc=r["median_q"])
        rec.update(structural_diagnostics(a, b))
        for cname, qs in chains.items():
            per_q, per_qA, per_qB = [], [], []
            for q in qs:
                ok, em = two_way_correct(Dq[q], y_ev, y_rf, ia, ib)
                per_q.append(float(ok.mean()))
                per_qA.append(float(ok[mA[em]].mean()))
                per_qB.append(float(ok[mB[em]].mean()))
            best = int(np.argmax(per_qA))          # chosen on half A
            D = summed(Dq, w2, qs, normalise=True)  # D008's own convention
            ok_set, em_set = two_way_correct(D, y_ev, y_rf, ia, ib)
            set_acc = float(ok_set.mean())
            a_naive = max(per_q) >= T2_ACC
            a_split = per_qB[best] >= T2_ACC
            b_fires = (set_acc >= T2_ACC) and not a_naive
            e_sec = float(S[qs, j].max())
            rec[cname] = dict(
                per_query_acc=[round(x, 3) for x in per_q],
                a_max_acc=round(max(per_q), 4), a_naive=bool(a_naive),
                a_split_half_acc=round(per_qB[best], 4), a_split_half=bool(a_split),
                set_acc=round(set_acc, 4), clears_T2=bool(set_acc >= T2_ACC
                                                          or a_naive),
                b_only=bool(b_fires),
                joint_stat=round(joint_sf(qs, a, b), 4),
                joint_vs_null=bool(joint_sf(qs, a, b) > nulls[cname]),
                s_energy_max=round(e_sec, 4),
                s_energy_clears_T3=bool(e_sec >= T3_ENERGY))
        rec["recovered"] = bool(rec["JointGreedy"]["clears_T2"]
                                and not rec["GreedyCover"]["clears_T2"])
        rec["lost"] = bool(rec["GreedyCover"]["clears_T2"]
                           and not rec["JointGreedy"]["clears_T2"])
        rows.append(rec)
        if len(rows) % 10 == 0:
            print(f"  {len(rows)}/{len(ix)}", flush=True)

    def cnt(c, key):
        return int(sum(1 for r in rows if r[c][key]))
    summary = {}
    for c in chains:
        summary[c] = dict(
            a_naive=cnt(c, "a_naive"), a_split_half=cnt(c, "a_split_half"),
            b_only=cnt(c, "b_only"), clears_T2=cnt(c, "clears_T2"),
            joint_vs_null=cnt(c, "joint_vs_null"),
            s_energy_clears_T3=cnt(c, "s_energy_clears_T3"),
            mean_set_acc=round(float(np.mean([r[c]["set_acc"] for r in rows])), 4),
            mean_a_max=round(float(np.mean([r[c]["a_max_acc"] for r in rows])), 4))
    rec_n = int(sum(1 for r in rows if r["recovered"]))
    lost_n = int(sum(1 for r in rows if r["lost"]))
    rec_a = int(sum(1 for r in rows if r["recovered"] and r["JointGreedy"]["a_naive"]))
    rec_b = int(sum(1 for r in rows if r["recovered"] and r["JointGreedy"]["b_only"]))

    verdict = ("CONFIRMED" if rec_n >= 12 else
               "NOT RECOVERED" if rec_n <= 4 else "PARTIAL / MIXED")
    out = dict(
        schema="d011-recovery-v1", n_pairs=len(rows),
        population="32 split-half-robust pairs (premise_check.json); the naive "
                   "count was 33 and was inflated by selecting and scoring the "
                   "oracle query on the same split",
        criteria=dict(T2_accuracy=T2_ACC, T3_energy_secondary=T3_ENERGY,
                      joint_null_p95=nulls),
        chains=chains, chain_overlap=len(set(chains["GreedyCover"]) &
                                         set(chains["JointGreedy"])),
        summary=summary,
        recovered=rec_n, lost=lost_n,
        recovered_via_a=rec_a, recovered_via_b_only=rec_b,
        pre_registered=dict(confirmed_at=12, not_recovered_at=4),
        verdict=verdict,
        inputs=dict(d008_selection=sha("./results/D008/selection.json"),
                    d010_selection=sha("./results/D010/selection.json"),
                    d008_frontier=sha("./results/D008/identifiability_frontier.json")),
        pairs=rows)
    json.dump(out, open(f"{OUT}/recovery_by_pair.json", "w"), indent=1)

    print(f"\n===== S5: {len(rows)} pairs =====")
    print(f"{'chain':14s} {'(a) naive':>10s} {'(a) split':>10s} {'(b) only':>9s} "
          f"{'clears T2':>10s} {'mean set acc':>13s}")
    for c in chains:
        s = summary[c]
        print(f"{c:14s} {s['a_naive']:10d} {s['a_split_half']:10d} "
              f"{s['b_only']:9d} {s['clears_T2']:10d} {s['mean_set_acc']:13.4f}")
    print(f"\nRECOVERED (Joint clears T2, Greedy does not): {rec_n}/{len(rows)}"
          f"   [via (a): {rec_a}, via (b) only: {rec_b}]")
    print(f"LOST     (Greedy clears T2, Joint does not): {lost_n}/{len(rows)}")
    print(f"pre-registered: >=12 CONFIRMED, <=4 NOT RECOVERED  ->  {verdict}")
    print(f"\nwritten: {OUT}/recovery_by_pair.json")


if __name__ == "__main__":
    main()
