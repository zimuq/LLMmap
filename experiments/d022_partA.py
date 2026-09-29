"""D022 / S1 — Part A: detectability of the pool and of each strategy (query text only).

Per D022 + Review (APPROVED WITH AMENDMENTS 2026-09-28):
  A1  all 259 queries: G1, G2 (harm, jailbreak), union flag/score; judge over 20
      S_build services: binary off-topic rate (as specified) + continuous P(NO)
      and mean log-odds log(P(NO)/P(YES)) (Call 1).
  Controls (Review amendment): (i) the service "You are a helpful assistant." is
      reported separately (it stays in the 20-service mean); (ii) the 8
      anchor:random-opt queries (paper's Alpaca-sampled generic prompts) are the
      benign reference row in A2/A3. If the helpful-assistant control judges most
      random-opt anchors off-topic, the judge is flagged UNINFORMATIVE.
  A2  rates by label, overall.  A3  per strategy, k=1..8: flagged count among the
      first k, first-flag position (union, G1, G2), judge mean rate/score; random
      from pool = 1,000 ordered k-subsets (seed 20260928).  A4  Spearman(union
      score, CVaR_0.1 S_energy_sf[q]) and vs CVaR_0.1 of D021's M2_q.
Usage:  PYTHONPATH=.:experiments python experiments/d022_partA.py
"""
import os
import json
import time

import numpy as np
from scipy.stats import spearmanr

from LLMmap.joint_statistic import cvar
from d022_lib import load_pool, build_services, g1_scores, Guardian, Judge, SEED

OUT = "./results/D022"
HELPFUL = "You are a helpful assistant."


def main():
    t0 = time.time()
    texts, lab = load_pool()
    services, _ = build_services()
    s0 = json.load(open(f"{OUT}/s0_pilot.json"))
    assert services == s0["services_sampled"]
    ctrl = services.index(HELPFUL)
    chains = s0["chains"]
    chains["random-opt (benign reference)"] = [i for i, l in enumerate(lab)
                                               if l == "anchor:random-opt"]
    assert len(chains["random-opt (benign reference)"]) == 8

    # ---------------- A1
    t = time.time(); g1 = g1_scores(texts); t_g1 = time.time() - t
    t = time.time(); G = Guardian()
    g2 = {rk: [G.score(q, rk) for q in texts] for rk in ("harm", "jailbreak")}
    t_g2 = time.time() - t
    del G
    import torch; torch.cuda.empty_cache()
    t = time.time(); J = Judge()
    jr = J.run([(s, q) for q in texts for s in services]); t_j = time.time() - t
    ns = len(services)
    rows = []
    for i, q in enumerate(texts):
        js = jr[i * ns:(i + 1) * ns]
        parsed = [x["off_topic"] for x in js if x["off_topic"] is not None]
        g2flag = g2["harm"][i]["flag"] or g2["jailbreak"][i]["flag"]
        g2score = max(g2["harm"][i]["score"], g2["jailbreak"][i]["score"])
        rows.append(dict(
            id=i, label=lab[i], g1_score=g1[i]["score"], g1_flag=g1[i]["flag"],
            g2_harm=g2["harm"][i]["score"], g2_harm_label=g2["harm"][i]["label"],
            g2_jailbreak=g2["jailbreak"][i]["score"], g2_jailbreak_label=g2["jailbreak"][i]["label"],
            g2_flag=bool(g2flag), union_flag=bool(g1[i]["flag"] or g2flag),
            union_score=float(max(g1[i]["score"], g2score)),
            judge_offtopic_rate=float(np.mean(parsed)) if parsed else None,
            judge_n_unparsed=ns - len(parsed),
            judge_p_no=float(np.mean([x["p_no"] for x in js])),
            judge_log_odds_no=float(np.mean([x["log_odds_no"] for x in js])),
            judge_helpful_control=dict(off_topic=js[ctrl]["off_topic"], p_no=js[ctrl]["p_no"],
                                       log_odds_no=js[ctrl]["log_odds_no"], raw=js[ctrl]["raw"]),
            judge_raw=[x["raw"] for x in js]))
    json.dump(dict(schema="d022-query-flags-v1", services=services, helpful_control_index=ctrl,
                   g3="meta-llama/Llama-Prompt-Guard-2-86M: gated, no access -> skipped",
                   rows=rows), open(f"{OUT}/query_flags.json", "w"), indent=1)
    print(f"[A1] G1 {sum(r['g1_flag'] for r in rows)}, G2 {sum(r['g2_flag'] for r in rows)}, "
          f"union {sum(r['union_flag'] for r in rows)} of 259 flagged; judge unparsed "
          f"{sum(r['judge_n_unparsed'] for r in rows)}", flush=True)

    # ---------------- A2
    def summ(ids):
        rr = [rows[i] for i in ids]
        return dict(n=len(rr), g1=round(np.mean([r["g1_flag"] for r in rr]), 4),
                    g2=round(np.mean([r["g2_flag"] for r in rr]), 4),
                    union=round(np.mean([r["union_flag"] for r in rr]), 4),
                    judge_offtopic=round(np.mean([r["judge_offtopic_rate"] for r in rr
                                                  if r["judge_offtopic_rate"] is not None]), 4),
                    judge_p_no=round(np.mean([r["judge_p_no"] for r in rr]), 4),
                    judge_log_odds=round(np.mean([r["judge_log_odds_no"] for r in rr]), 3),
                    helpful_control_offtopic=round(np.mean(
                        [r["judge_helpful_control"]["off_topic"] is True for r in rr]), 4))
    labels = sorted(set(lab))
    a2 = {"overall": summ(range(259)),
          **{l: summ([i for i in range(259) if lab[i] == l]) for l in labels}}
    ref = a2["anchor:random-opt"]
    judge_status = ("UNINFORMATIVE (helpful-assistant control judges most benign "
                    "random-opt anchors off-topic)" if ref["helpful_control_offtopic"] > 0.5
                    else "informative on the control")

    # ---------------- A3
    def chain_stats(q):
        out = []
        for k in range(1, 9):
            f = [rows[i] for i in q[:k]]
            first = lambda key: next((j + 1 for j, r in enumerate(f) if r[key]), None)
            out.append(dict(k=k, union_flagged=sum(r["union_flag"] for r in f),
                            g1_flagged=sum(r["g1_flag"] for r in f),
                            g2_flagged=sum(r["g2_flag"] for r in f),
                            first_flag_union=first("union_flag"), first_flag_g1=first("g1_flag"),
                            first_flag_g2=first("g2_flag"),
                            judge_offtopic_mean=round(np.mean([r["judge_offtopic_rate"] for r in f]), 4),
                            judge_log_odds_mean=round(np.mean([r["judge_log_odds_no"] for r in f]), 3)))
        return out
    a3 = {c: dict(chain=q, per_k=chain_stats(q)) for c, q in chains.items()}
    rng = np.random.default_rng(SEED)
    uf = np.array([r["union_flag"] for r in rows])
    draws = np.stack([rng.permutation(259)[:8] for _ in range(1000)])
    fl = uf[draws]                                               # (1000, 8)
    firstpos = np.where(fl.any(1), fl.argmax(1) + 1, 0)          # 0 = never within 8
    a3["random-from-pool"] = dict(
        n_draws=1000, seed=SEED,
        expected_union_flagged_by_k=[round(float(fl[:, :k].sum(1).mean()), 3) for k in range(1, 9)],
        first_flag_position_distribution={str(p): int((firstpos == p).sum()) for p in range(0, 9)},
        note="position 0 = no flag among the first 8")

    # ---------------- A4
    S = np.load("./results/D007/S_energy_sf_tok200.npy").astype(np.float64)
    M2 = np.load("./results/D021/evidence_tok200.npy").astype(np.float64)
    us = np.array([r["union_score"] for r in rows])
    pw_s = np.array([cvar(S[q], 0.1) for q in range(259)])
    pw_m = np.array([cvar(M2[q], 0.1) for q in range(259)])
    a4 = dict(spearman_union_vs_cvar_S_energy=float(spearmanr(us, pw_s)[0]),
              spearman_union_vs_cvar_M2=float(spearmanr(us, pw_m)[0]),
              spearman_judge_logodds_vs_cvar_M2=float(spearmanr(
                  [r["judge_log_odds_no"] for r in rows], pw_m)[0]),
              note="descriptive only, no decision rule")
    json.dump(dict(schema="d022-strategy-detectability-v1", A2_by_label=a2,
                   benign_reference="anchor:random-opt", judge_control_status=judge_status,
                   A3=a3, A4=a4,
                   cost_s=dict(g1=round(t_g1, 1), g2=round(t_g2, 1), judge=round(t_j, 1),
                               wall=round(time.time() - t0, 1))),
              open(f"{OUT}/strategy_detectability.json", "w"), indent=1)
    print(f"[A2] overall {a2['overall']}", flush=True)
    print(f"[A2] benign ref {ref}; judge {judge_status}", flush=True)
    for c in a3:
        if "per_k" in a3[c]:
            k8 = a3[c]["per_k"][-1]
            print(f"[A3] {c:30s} union flagged@8 {k8['union_flagged']} first {k8['first_flag_union']} "
                  f"judge logodds {k8['judge_log_odds_mean']}", flush=True)
    print(f"[A3] random-from-pool {a3['random-from-pool']['expected_union_flagged_by_k']}", flush=True)
    print(f"[A4] {a4}", flush=True)
    print(f"wall {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
