"""D023 / S0 — inventory and solver pilot (pre-P1, target-free).

(1) D021's primary tensor loads and its meta matches.
(2) JointGreedy's CVaR_0.1-of-ADD at k=1..8 (sanity value), plus every existing
    chain's objective.
(3) Solve the MILP at k=2 and k=8 (15-min limit each) and the LP relaxation;
    report time, status, gap, dual bound; test for an alternative optimum
    with one no-good cut.
(4) Inputs for S2/S3 exist: D010 attention counts (joint_energy, k=1..8, 5
    seeds), D020 linear counts (JointGreedy, both poolings, k=1..8), D016 k=8
    logits (joint_energy, paper8, 5 seeds each) for the 666-pair tail.
No trained number is read beyond existence checks. Writes results/D023/s0_pilot.json.
Usage:  PYTHONPATH=.:experiments python experiments/d023_s0.py
"""
import os
import json
import time

import numpy as np

from d023_lib import load_evidence, objective, solve

OUT = "./results/D023"


def existing_chains():
    d9 = json.load(open("./results/D009/runs.json"))["runs"]
    d10 = json.load(open("./results/D010/metrics_by_k.json"))["runs"]
    g = lambda runs, cond: next(r for r in runs if r["k"] == 8 and not r["error"]
                                and r["condition"] == cond)["queries"]
    h1 = json.load(open("./results/D018/greedy_chains.json"))["chains"]
    return {"JointGreedy": g(d10, "joint_energy"), "GreedyCover": g(d9, "cvar_max"),
            "paper8": g(d9, "paper8"), "H1-attention": h1["H1-attention"][:8],
            "H1-linear": h1["H1-linear"][:8]}


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    E, meta = load_evidence()
    out = dict(schema="d023-s0-v1", evidence=dict(shape=list(E.shape),
               schema=meta["schema_version"], primary=meta["primary"],
               negative_cells=int((E < 0).sum())))
    ch = existing_chains()
    out["existing_chain_objective"] = {c: [round(objective(E, q[:k]), 4) for k in range(1, 9)]
                                       for c, q in ch.items()}
    print(f"[S0] CVaR0.1(ADD) by k: {out['existing_chain_objective']}", flush=True)

    pilot = {}
    for k in (2, 8):
        ip = solve(E, k, integer=True)
        lp = solve(E, k, integer=False)
        alt = solve(E, k, integer=True, exclude=ip.get("set")) if ip.get("set") else None
        pilot[k] = dict(milp=ip, lp=lp, second_best_after_nogood_cut=alt,
                        lp_bound_over_milp=(lp["cvar"] / ip["cvar"]) if ip.get("cvar") else None,
                        jointgreedy_over_milp=out["existing_chain_objective"]["JointGreedy"][k - 1]
                        / ip["cvar"] if ip.get("cvar") else None)
        print(f"[S0] k={k}: MILP {ip['status']} {ip['message'][:40]!r} cvar {ip.get('cvar')} "
              f"set {ip.get('set')} gap {ip.get('mip_gap')} {ip['wall_s']}s | LP {lp.get('cvar')} "
              f"frac {lp.get('n_fractional')} {lp['wall_s']}s | 2nd-best {alt and alt.get('cvar')}",
              flush=True)
    out["solver_pilot"] = pilot
    out["solver"] = "HiGHS via scipy.optimize.milp (scipy " + __import__("scipy").__version__ + \
                    "), mip_rel_gap=1e-6, time_limit=900 s"

    inv = {}
    c10 = np.load("./results/D010/run_counts.npz")
    inv["D010_joint_energy_attention"] = {k: len([n for n in c10.files
                                                  if n.startswith(f"joint_energy|{k}|")
                                                  and n.endswith("cnt_total")])
                                          for k in range(1, 9)}
    c20 = np.load("./results/D020/linear_counts.npz")
    inv["D020_JointGreedy_linear"] = {p: all(f"JointGreedy|{p}|{k}|cnt_total" in c20.files
                                             for k in range(1, 9)) for p in ("concat", "meanpool")}
    inv["D016_k8_logits"] = {c: [os.path.exists(f"./results/D016/logits_{c}_k8_r{r}.npy")
                                 for r in range(5)] for c in ("joint_energy", "paper8")}
    out["inventory"] = inv
    out["wall_s"] = round(time.time() - t0, 1)
    print(f"[S0] inventory {inv}; wall {out['wall_s']} s", flush=True)
    json.dump(out, open(f"{OUT}/s0_pilot.json", "w"), indent=1)


if __name__ == "__main__":
    main()
