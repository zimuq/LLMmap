"""D008 / S0–S3 — verify the frozen inputs, prove the gamma=1.0 reduction,
and produce the selected query set for every condition.

No accuracy here: this script only decides WHICH queries each condition selects.
Scoring them is S4 (`d008_metrics.py`), which is what keeps the selection step
free of anything the evaluation could leak into.

Usage:  PYTHONPATH=.:experiments python experiments/d008_select.py
"""
import os
import json
import hashlib
import itertools

import numpy as np

from LLMmap.greedy_cover import greedy_cover, mean_greedy, verify_reduction, cvar
from d008_lib import (load_tensor, pair_index, near_relative_pairs, OUT,
                      POOL, PAPER8, CORPUS)

K_MAX = 16          # chains run to 16 for the "queries to reach X%" metric only;
K_HEAD = 8          # the headline curve is k=1..8 (D008)
GAMMAS = [1.0, 0.5, 0.25, 0.1, 0.05]
N_RANDOM_SEEDS = 200

if os.environ.get("D008_SMOKE"):
    K_MAX, K_HEAD, N_RANDOM_SEEDS = 3, 2, 5


def s0_verify():
    """Frozen-input verification. Everything asserted, nothing assumed."""
    man = json.load(open(os.path.join(CORPUS, "corpus_manifest.json")))
    assert man["status"] == "READY", man["status"]
    models = sorted(s["model"] for s in man["models"] if s["status"] == "VALIDATED")
    assert len(models) == 37, len(models)

    S, meta = load_tensor()
    sha = hashlib.sha256(open("./results/D007/S_energy_sf_tok200.npy", "rb").read()
                         ).hexdigest()
    assert meta["models"] == models, "tensor model order != manifest order"
    assert meta["pairs"][0] == f"{models[0]}|{models[1]}"

    pool = json.load(open(POOL))
    assert pool["n"] == 259 and len(pool["queries"]) == 259
    texts = [q["text"] for q in pool["queries"]]
    paper8 = json.load(open(PAPER8))
    assert texts[:8] == paper8, "the paper's 8 queries are not Q_0[0..7]"

    hp = near_relative_pairs(models)
    print(f"[S0] corpus READY 37/37 · tensor {S.shape} schema "
          f"{meta['schema_version']} · pool 259 · paper-8 == Q_0[0..7] · "
          f"{len(hp)} near-relative pairs", flush=True)
    return S, meta, models, hp, sha


def s3_conditions(S):
    """The five conditions of D008/S3, plus the gamma sweep of S5.

    Selection only. Chains run to K_MAX; the paper's 8 obviously cannot.
    """
    cond = {}

    # 1. the paper's original 8 queries -- a SET, not a chain. Its file order is
    #    arbitrary, so any k<8 prefix would be an artifact of that ordering;
    #    k<8 is scored as the mean over all C(8,k) subsets instead (P1, minor).
    cond["paper8"] = dict(kind="fixed_set", queries=list(range(8)),
                          subsets={k: [list(c) for c in
                                       itertools.combinations(range(8), k)]
                                   for k in range(1, 9)})  # scored in S4

    # 2. random-k: 200 independent uniform draws per k (NOT a nested chain --
    #    a nested random chain is a different, easier baseline)
    rng = np.random.default_rng(20260907)
    cond["random"] = dict(kind="random",
                          draws={k: [sorted(rng.choice(S.shape[0], k, replace=False)
                                            .tolist()) for _ in range(N_RANDOM_SEEDS)]
                                 for k in range(1, K_MAX + 1)})

    # 3-5. greedy conditions
    for name, gamma, agg in (("mean_greedy_max", 1.0, "max"),
                             ("cvar_max", 0.1, "max"),
                             ("mean_greedy_sum", 1.0, "sum")):
        sel, trace = greedy_cover(S, K_MAX, gamma=gamma, agg=agg, return_trace=True)
        cond[name] = dict(kind="chain", gamma=gamma, agg=agg,
                          queries=sel, trace=trace)

    # S5: the gamma sweep (mean_greedy_max and cvar_max are gamma 1.0 / 0.1)
    sweep = {}
    for g in GAMMAS:
        sel = greedy_cover(S, K_MAX, gamma=g, agg="max")
        sweep[str(g)] = sel
    return cond, sweep


def main():
    os.makedirs(OUT, exist_ok=True)
    S, meta, models, hp, sha = s0_verify()

    # ---- S1: the reduction must hold numerically, not by inspection
    red = verify_reduction(S, k=K_HEAD, agg="max")
    red_sum = verify_reduction(S, k=K_HEAD, agg="sum")
    print(f"[S1] gamma=1.0 == independent mean-greedy: max-agg OK "
          f"{red['gamma1_chain']}", flush=True)
    print(f"[S1] same under sum-agg: OK {red_sum['gamma1_chain']}", flush=True)

    cond, sweep = s3_conditions(S)
    pairs = pair_index(models)

    # coverage summaries per condition (selection-side, no accuracy)
    cov_rep = {}
    for name, c in cond.items():
        if c["kind"] != "chain":
            continue
        for k in sorted({1, min(3, K_MAX), min(8, K_MAX), K_MAX}):
            q = c["queries"][:k]
            cv = S[q].max(axis=0) if c["agg"] == "max" else S[q].sum(axis=0)
            cov_rep[f"{name}_k{k}"] = dict(
                mean=round(float(cv.mean()), 4), min=round(float(cv.min()), 4),
                cvar10=round(cvar(cv, 0.1), 4))

    overlaps = {}
    gs = [str(g) for g in GAMMAS]
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            kk = min(8, K_MAX)
            a, b = set(sweep[gs[i]][:kk]), set(sweep[gs[j]][:kk])
            overlaps[f"{gs[i]}_vs_{gs[j]}_k{kk}"] = len(a & b)

    out = dict(
        schema="d008-selection-v1",
        tensor=dict(path="results/D007/S_energy_sf_tok200.npy", sha256=sha,
                    shape=list(S.shape), schema_version=meta["schema_version"]),
        models=models, n_pairs=len(pairs),
        near_relative_pairs=len(hp),
        k_head=K_HEAD, k_max=K_MAX, gammas=GAMMAS, n_random_seeds=N_RANDOM_SEEDS,
        reduction_check=dict(max_agg=red, sum_agg=red_sum),
        conditions={k: {kk: vv for kk, vv in v.items() if kk != "draws"}
                    for k, v in cond.items()},
        gamma_sweep_chains=sweep,
        gamma_chain_overlap_at_k8=overlaps,
        coverage_by_condition=cov_rep)
    json.dump(out, open(f"{OUT}/selection.json", "w"), indent=1)
    # the random draws are bulky; keep them in their own file, seeded anyway
    json.dump(cond["random"]["draws"], open(f"{OUT}/random_draws.json", "w"))

    print("\n[S3] selected chains (first 8):")
    for name in ("mean_greedy_max", "cvar_max", "mean_greedy_sum"):
        print(f"  {name:17s} {cond[name]['queries'][:8]}")
    print(f"  paper8            {list(range(8))}")
    kk = min(8, K_MAX)
    ov = len(set(cond['mean_greedy_max']['queries'][:kk]) &
             set(cond['cvar_max']['queries'][:kk]))
    print(f"\n  mean-greedy vs CVaR overlap at k={kk}: {ov}/{kk}")
    print(f"  gamma-chain overlaps at k=8: {overlaps}")
    print(f"\n[S3] coverage (selection-side):")
    for k, v in cov_rep.items():
        print(f"  {k:24s} mean {v['mean']:.4f}  min {v['min']:.4f}  "
              f"CVaR10 {v['cvar10']:.4f}")
    print(f"\nwritten: {OUT}/selection.json")


if __name__ == "__main__":
    main()
