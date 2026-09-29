"""D022 / S0 — inventory + cost pilot on 10 queries (pre-P1). Query text only.

Measures G1, G2 (harm, jailbreak) and judge throughput on 10 seeded queries
(judge: 10 queries x 20 services = 200 prompts), projects full cost, and checks
that every strategy's chain queries are in the pool. Writes results/D022/s0_pilot.json.
Usage:  PYTHONPATH=.:experiments python experiments/d022_s0.py
"""
import os
import json
import time

import numpy as np

from d022_lib import (load_pool, build_services, g1_scores, Guardian, Judge,
                      SEED, JUDGE_SYSTEM, JUDGE_USER, G1_ID, G2_ID, G3_ID, JUDGE_ID)

OUT = "./results/D022"


def main():
    os.makedirs(OUT, exist_ok=True)
    texts, fam = load_pool()
    from collections import Counter
    services, n_serv_all = build_services()
    d9 = json.load(open("./results/D009/runs.json"))["runs"]
    d10 = json.load(open("./results/D010/metrics_by_k.json"))["runs"]
    g = lambda runs, cond: next(r for r in runs if r["k"] == 8 and not r["error"]
                                and r["condition"] == cond)["queries"]
    h1 = json.load(open("./results/D018/greedy_chains.json"))["chains"]
    chains = {"paper8": g(d9, "paper8"), "GreedyCover": g(d9, "cvar_max"),
              "JointGreedy": g(d10, "joint_energy"), "H1-attention": h1["H1-attention"][:8],
              "H1-linear": h1["H1-linear"][:8]}
    assert all(0 <= q < 259 for c in chains.values() for q in c)
    out = dict(schema="d022-s0-v1", pool_size=len(texts), families=dict(Counter(fam)),
               chains=chains, services_available=n_serv_all, services_sampled=services,
               judge_prompt=dict(system=JUDGE_SYSTEM, user=JUDGE_USER),
               detectors=dict(G1=G1_ID, G2=G2_ID, G3=f"{G3_ID}: gated, no access -> skipped",
                              judge=JUDGE_ID))
    sample = sorted(np.random.default_rng(SEED).choice(259, 10, replace=False).tolist())
    qs = [texts[i] for i in sample]
    t = time.time(); r1 = g1_scores(qs); t1 = time.time() - t
    t = time.time(); G = Guardian(); tl2 = time.time() - t
    t = time.time(); r2 = {rk: [G.score(q, rk) for q in qs] for rk in ("harm", "jailbreak")}
    t2 = time.time() - t
    del G
    import torch; torch.cuda.empty_cache()
    t = time.time(); J = Judge(); tlj = time.time() - t
    t = time.time(); rj = J.run([(s, q) for q in qs for s in services]); tj = time.time() - t
    n_unparsed = sum(1 for x in rj if x["off_topic"] is None)
    out["pilot"] = dict(
        query_ids=sample,
        g1=r1, g2=r2,
        judge_off_topic_rate=[float(np.mean([x["off_topic"] for x in rj[i*20:(i+1)*20]
                                             if x["off_topic"] is not None])) for i in range(10)],
        judge_unparsed=n_unparsed, judge_examples=[x["raw"] for x in rj[:5]],
        seconds=dict(g1_10q=round(t1, 1), g2_load=round(tl2, 1), g2_20calls=round(t2, 1),
                     judge_load=round(tlj, 1), judge_200prompts=round(tj, 1)),
        projected_full_min=dict(g1=round(t1 / 10 * 259 / 60, 2),
                                g2=round(t2 / 10 * 259 / 60, 2),
                                judge=round(tj / 200 * 259 * 20 / 60, 1)))
    print(json.dumps(out["pilot"], indent=1)[:4000], flush=True)
    json.dump(out, open(f"{OUT}/s0_pilot.json", "w"), indent=1)


if __name__ == "__main__":
    main()
