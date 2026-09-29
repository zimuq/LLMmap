"""D022 / S2 — Part B selection: JointGreedy (gamma=0.1, scale-free energy) on the
G1-and-G2-unflagged pool. Frozen and committed BEFORE training.

Same code path as D010 (d010_select.precompute + LLMmap.joint_greedy.joint_greedy).
Code-path check (required): the same wrapper on the FULL pool must reproduce
D010's chain exactly. Stop (exit 3) if the restricted pool has < 30 queries.
Identical-prefix k (Review amendment): k where the restricted chain equals
JointGreedy's prefix -> identical inputs, listed here before training.
Target-free. Writes results/D022/restricted_chain.json.
Usage:  PYTHONPATH=.:experiments python experiments/d022_partB_select.py
"""
import sys
import json
import time
import itertools
from collections import Counter

import numpy as np

from LLMmap.joint_greedy import joint_greedy
from d010_select import precompute, K, GAMMA
from d007_lib import load_corpus

OUT = "./results/D022"
D010_CHAIN = [193, 140, 237, 114, 233, 117, 0, 16]


def main():
    t0 = time.time()
    qf = json.load(open(f"{OUT}/query_flags.json"))["rows"]
    R = [r["id"] for r in qf if not r["union_flag"]]
    comp = dict(Counter(qf[i]["label"] for i in R))
    print(f"[S2] restricted pool {len(R)} queries; composition {comp}", flush=True)
    if len(R) < 30:
        json.dump(dict(stop="restricted pool < 30", n=len(R), composition=comp),
                  open(f"{OUT}/restricted_chain.json", "w"), indent=1)
        print("[S2] STOP: restricted pool < 30 -> Call", flush=True)
        sys.exit(3)
    models, X = load_corpus(pool="build")
    pairs = list(itertools.combinations(models, 2))
    ix = {m: i for i, m in enumerate(models)}
    ia = np.array([ix[a] for a, _ in pairs]); ib = np.array([ix[b] for _, b in pairs])
    XX2, XY2 = precompute(X, models, pairs)
    full, _ = joint_greedy(XY2, XX2, ia, ib, K, GAMMA, "energy", verbose=False)
    assert list(map(int, full)) == D010_CHAIN, full
    print(f"[S2] code-path check: full pool reproduces D010 {full}", flush=True)
    selR, traceR = joint_greedy(XY2[R], XX2[R], ia, ib, K, GAMMA, "energy", verbose=False)
    chain = [int(R[i]) for i in selR]
    ident = [k for k in range(1, K + 1) if chain[:k] == D010_CHAIN[:k]]
    first_flagged_jg = next((j + 1 for j, q in enumerate(D010_CHAIN) if qf[q]["union_flag"]), None)
    json.dump(dict(schema="d022-restricted-chain-v1", frozen_before_training=True,
                   restricted_pool=R, n_restricted=len(R), composition=comp,
                   excluded_by="G1 or G2 flag (G3 skipped: gated)",
                   chain=chain, trace=traceR, code_path_check=dict(full_pool_chain=list(map(int, full)),
                                                                   reproduces_D010=True),
                   identical_input_k=ident,
                   jointgreedy_first_flagged_position=first_flagged_jg,
                   overlap=dict(JointGreedy=len(set(chain) & set(D010_CHAIN)),
                                paper8=len(set(chain) & set(range(8)))),
                   restricted_chain_judge=[dict(q=q, offtopic_rate=qf[q]["judge_offtopic_rate"],
                                                log_odds=qf[q]["judge_log_odds_no"]) for q in chain],
                   wall_s=round(time.time() - t0, 1)),
              open(f"{OUT}/restricted_chain.json", "w"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"[S2] restricted chain {chain}; identical-input k {ident}; overlap JG "
          f"{len(set(chain) & set(D010_CHAIN))}/8", flush=True)


if __name__ == "__main__":
    main()
