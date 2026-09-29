"""D022 / S3 — Part B training + evaluation of the frozen restricted chain.

Training via D023's shared functions (d023_train: D009 attention path, 5 seeds,
all checkpoints, counts incl. 666-pair; D020 linear fits). Identical-prefix k are
asserted to reproduce D010 (attention) / D020 (linear) JointGreedy counts exactly.
paper8 linear is refit (D017 saved no counts) and asserted to reproduce D017's
stored metrics exactly.

Comparisons (JG-restricted - JointGreedy, JG-restricted - paper8), per k and mean
over k; no verdict pre-registered (D022):
  attention: (i) D009/F1 reading as the D specifies (seed-averaged config
    bootstrap CI excludes 0 AND |d| > max 5-run range); (ii) hierarchical
    bootstrap on D023's approved footing (configs boot_draws(25,2000,20260928),
    seeds default_rng(20260928) jointly across arms); (iii) independent-seed
    sensitivity (second arm default_rng(20260930)).
  linear (concat, mean-pool): config bootstrap, CI excludes 0 (D020's rule).
Usage:  PYTHONPATH=.:experiments python experiments/d022_partB.py
"""
import os
import json
import time

import numpy as np

from d008_lib import boot_draws
from d023_train import Ctx, train_attention, fit_linear

OUT = "./results/D022"
NCFG, NM, NB, SEED = 25, 37, 2000, 20260928
KS = list(range(1, 9))


def pct(x):
    return dict(lo=round(float(np.percentile(x, 2.5)), 4), hi=round(float(np.percentile(x, 97.5)), 4))


def main():
    t0 = time.time()
    rc = json.load(open(f"{OUT}/restricted_chain.json"))
    assert rc["frozen_before_training"]
    chain, ident = rc["chain"], rc["identical_input_k"]
    ctx = Ctx()
    c10 = np.load("./results/D010/run_counts.npz")
    c9 = np.load("./results/D009/run_counts.npz")
    l20 = np.load("./results/D020/linear_counts.npz")
    L17 = json.load(open("./results/D017/linear_metrics_by_k.json"))["metrics"]
    os.makedirs(f"{OUT}/models", exist_ok=True)

    orders = {k: chain[:k] for k in KS}
    att, runs = train_attention(ctx, "JG-restricted", orders, f"{OUT}/models", ident, c10, "joint_energy")
    lin, lpts = fit_linear(ctx, "JG-restricted", orders, ident, l20, "JointGreedy")
    p8, p8pts = fit_linear(ctx, "paper8", {k: list(range(k)) for k in KS})
    for p in ("concat", "meanpool"):
        for k in KS:
            for m in ("mean_top1", "worst_class", "worst3_class", "hard_subset"):
                assert abs(p8pts[f"paper8|{p}|{k}"][m] - L17[p]["paper8"][str(k)][m]) < 1e-9, (p, k, m)
    print("[S3] paper8 linear refit reproduces D017 exactly", flush=True)
    np.savez_compressed(f"{OUT}/run_counts.npz", **att)
    np.savez_compressed(f"{OUT}/linear_counts.npz", **lin, **p8)

    M = boot_draws(NCFG, NB, seed=SEED); tot = M.sum(1)
    S = np.random.default_rng(SEED).integers(0, 5, (NB, 8, 5))
    S2 = np.random.default_rng(SEED + 2).integers(0, 5, (NB, 8, 5))

    def acnt(arm, k):
        src = {"JG-restricted": (att, "JG-restricted"), "JointGreedy": (c10, "joint_energy"),
               "paper8": (c9, "paper8")}[arm]
        return np.stack([src[0][f"{src[1]}|{k}|{r}|cnt_total"] for r in range(5)])

    def seed_draws(arm, k):
        return (M @ acnt(arm, k).T) / (NM * tot[:, None])                  # (NB, 5)

    def pick(x, seeds, k):
        return np.take_along_axis(x, seeds[:, k - 1, :], 1).mean(1)

    comps = {}
    for other in ("JointGreedy", "paper8"):
        per_k, dj, di = {}, [], []
        for k in KS:
            a, b = seed_draws("JG-restricted", k), seed_draws(other, k)
            pa = acnt("JG-restricted", k).sum(1) / (NM * NCFG)
            pb = acnt(other, k).sum(1) / (NM * NCFG)
            d = float(pa.mean() - pb.mean())
            old = a.mean(1) - b.mean(1)
            lo, hi = np.percentile(old, [2.5, 97.5])
            rng_ = float(max(np.ptp(pa), np.ptp(pb)))
            joint = pick(a, S, k) - pick(b, S, k)
            indep = pick(a, S, k) - pick(b, S2, k)
            dj.append(joint); di.append(indep)
            per_k[k] = dict(restricted=round(float(pa.mean()), 4), other=round(float(pb.mean()), 4),
                            delta=round(d, 4),
                            identical_input=bool(other == "JointGreedy" and k in ident),
                            d009_f1=dict(ci=[round(float(lo), 4), round(float(hi), 4)],
                                         seed_range=round(rng_, 4),
                                         resolved=bool((lo > 0 or hi < 0) and abs(d) > rng_)),
                            hierarchical_joint=pct(joint), hierarchical_independent=pct(indep))
        mean_pt = round(float(np.mean([per_k[k]["delta"] for k in KS])), 4)
        lin_rows = {}
        for p in ("concat", "meanpool"):
            rows = {}
            for k in KS:
                ca = lin[f"JG-restricted|{p}|{k}|cnt_total"]
                cb = (l20[f"JointGreedy|{p}|{k}|cnt_total"] if other == "JointGreedy"
                      else p8[f"paper8|{p}|{k}|cnt_total"])
                d = (M @ ca - M @ cb) / (NM * tot)
                rows[k] = dict(delta=round(float((ca.sum() - cb.sum()) / (NM * NCFG)), 4), **pct(d),
                               ci_excludes_0=bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0))
            lin_rows[p] = rows
        comps[f"JG-restricted_minus_{other}"] = dict(
            attention_per_k=per_k,
            attention_mean_k=dict(point=mean_pt, joint=pct(np.mean(dj, 0)),
                                  independent=pct(np.mean(di, 0))),
            linear=lin_rows)
        print(f"[S3] vs {other}: attention mean-k {mean_pt:+.4f} joint {pct(np.mean(dj, 0))}; per-k "
              f"{[per_k[k]['delta'] for k in KS]}", flush=True)
        for p in lin_rows:
            print(f"     linear {p}: {[(k, v['delta'], v['ci_excludes_0']) for k, v in lin_rows[p].items()]}",
                  flush=True)

    json.dump(dict(schema="d022-restricted-eval-v1", chain=chain, identical_input_k=ident,
                   comparisons=comps, no_verdict="Part B prices a trade-off (D022); no verdict",
                   bootstrap=dict(configs="boot_draws(25, 2000, seed=20260928)",
                                  seeds_joint="default_rng(20260928).integers(0,5,(2000,8,5))",
                                  seeds_independent_arm2="default_rng(20260930)"),
                   runs=runs, linear_points={**lpts, **p8pts},
                   wall_min=round((time.time() - t0) / 60, 1)),
              open(f"{OUT}/restricted_eval.json", "w"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"wall {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
