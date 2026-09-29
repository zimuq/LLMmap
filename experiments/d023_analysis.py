"""D023 / S3 — primary and secondary analysis (scheme approved in D023's Review).

Hierarchical bootstrap (Review, Call 1 as written in P1):
  configs  M = boot_draws(25, 2000, seed=20260928), shared over arms and k
  seeds    S = default_rng(20260928).integers(0, 5, (2000, 8, 5)), shared over
           arms, independent over k
  acc_{a,k}^(b) = (1/5) sum_j M_b . c[a,k,S_bkj] / (37 * sum M_b)
Sensitivity (Review amendment): independent seed resampling (second arm's seed
  indices from default_rng(20260930)), restricted to the differing k.
Primary (Call 2 (b)): BETTER if lo(mean over k=1..8) > 0; WORSE if hi < 0;
  PARITY if the interval of the mean over DIFFERING k lies within [-0.02, 0.02];
  else INCONCLUSIVE. The verdict string implements this full rule.
Old D009/F1 reading per k (disclosure): config-only bootstrap of seed-averaged
  draws, CI excludes 0 AND |delta| > max 5-run range.
Usage:  PYTHONPATH=.:experiments python experiments/d023_analysis.py
"""
import json

import numpy as np
from scipy.stats import spearmanr

from LLMmap.joint_statistic import cvar
from d008_lib import boot_draws
from d023_lib import load_evidence

OUT = "./results/D023"
NCFG, NM, NB, DELTA, SEED = 25, 37, 2000, 0.02, 20260928
KS = list(range(1, 9))
NAMED = ["tiiuae/Falcon3-10B-Instruct | tiiuae/Falcon3-7B-Instruct",
         "microsoft/Phi-3-medium-128k-instruct | microsoft/Phi-3-medium-4k-instruct"]


def pct(x):
    return dict(lo=round(float(np.percentile(x, 2.5)), 4), hi=round(float(np.percentile(x, 97.5)), 4))


def main():
    sel = json.load(open(f"{OUT}/selection.json"))
    diff_k = sel["differing_k"]["ILP-ADD"]
    c23 = np.load(f"{OUT}/run_counts.npz")
    c10 = np.load("./results/D010/run_counts.npz")
    l23 = np.load(f"{OUT}/linear_counts.npz")
    l20 = np.load("./results/D020/linear_counts.npz")
    M = boot_draws(NCFG, NB, seed=SEED)
    tot = M.sum(1)
    S = np.random.default_rng(SEED).integers(0, 5, (NB, 8, 5))
    S_ind = np.random.default_rng(SEED + 2).integers(0, 5, (NB, 8, 5))

    def cnt(arm, k, key="cnt_total"):
        if arm == "JointGreedy":
            return np.stack([c10[f"joint_energy|{k}|{r}|{key}"] for r in range(5)])
        return np.stack([c23[f"{arm}|{k}|{r}|{key}"] for r in range(5)])

    def acc_draws(arm, k, seeds):
        per_seed = (M @ cnt(arm, k).T) / (NM * tot[:, None])             # (NB, 5)
        return np.take_along_axis(per_seed, seeds[:, k - 1, :], 1).mean(1)

    def point(arm, k):
        return float(cnt(arm, k).sum(1).mean() / (NM * NCFG))

    def contrast(a, b, ks, seeds_a=S, seeds_b=S):
        d = np.mean([acc_draws(a, k, seeds_a) - acc_draws(b, k, seeds_b) for k in ks], 0)
        p = float(np.mean([point(a, k) - point(b, k) for k in ks]))
        return dict(point=round(p, 4), **pct(d), ks=ks)

    # ---------------- primary
    all8 = contrast("ILP-ADD", "JointGreedy", KS)
    dk = contrast("ILP-ADD", "JointGreedy", diff_k)
    if all8["lo"] > 0:
        verdict = "BETTER"
    elif all8["hi"] < 0:
        verdict = "WORSE"
    elif dk["lo"] >= -DELTA and dk["hi"] <= DELTA:
        verdict = "PARITY"
    else:
        verdict = "INCONCLUSIVE"
    ind = contrast("ILP-ADD", "JointGreedy", diff_k, S, S_ind)
    if ind["lo"] > 0:
        v_ind = "BETTER"
    elif ind["hi"] < 0:
        v_ind = "WORSE"
    elif ind["lo"] >= -DELTA and ind["hi"] <= DELTA:
        v_ind = "PARITY"
    else:
        v_ind = "INCONCLUSIVE"
    per_k = {}
    for k in KS:
        joint = acc_draws("ILP-ADD", k, S) - acc_draws("JointGreedy", k, S)
        # old D009/F1: seed-averaged config bootstrap, CI excl 0 AND |d| > max run range
        avg = lambda arm: ((M @ cnt(arm, k).T) / (NM * tot[:, None])).mean(1)
        old = avg("ILP-ADD") - avg("JointGreedy")
        rng_ = max(np.ptp(cnt(a, k).sum(1) / (NM * NCFG)) for a in ("ILP-ADD", "JointGreedy"))
        d = point("ILP-ADD", k) - point("JointGreedy", k)
        lo_o, hi_o = np.percentile(old, [2.5, 97.5])
        per_k[k] = dict(ilp=round(point("ILP-ADD", k), 4), jg=round(point("JointGreedy", k), 4),
                        delta=round(d, 4), hierarchical=pct(joint),
                        identical_input=k not in diff_k,
                        old_rule=dict(ci=[round(float(lo_o), 4), round(float(hi_o), 4)],
                                      seed_range=round(float(rng_), 4),
                                      resolved=bool((lo_o > 0 or hi_o < 0) and abs(d) > rng_)))
    print(f"[S3] VERDICT {verdict}: mean k=1..8 {all8}; differing k {dk}; independent-seed "
          f"(differing k) {ind} -> {v_ind}", flush=True)
    for k, v in per_k.items():
        print(f"     k={k}: ILP {v['ilp']} JG {v['jg']} d {v['delta']:+.4f} {v['hierarchical']} "
              f"old-rule resolved {v['old_rule']['resolved']}", flush=True)

    # ---------------- S-d decomposition
    decomp = dict(objective_effect_GreedyADD_minus_JG=contrast("Greedy-ADD", "JointGreedy", KS),
                  optimiser_effect_ILP_minus_GreedyADD=contrast("ILP-ADD", "Greedy-ADD", KS))

    # ---------------- S-a linear (config bootstrap only)
    lin = {}
    for p in ("concat", "meanpool"):
        def la(arm, k):
            c = (l20[f"JointGreedy|{p}|{k}|cnt_total"] if arm == "JointGreedy"
                 else l23[f"{arm}|{p}|{k}|cnt_total"])
            return (M @ c) / (NM * tot), float(c.sum() / (NM * NCFG))
        rows = {}
        for k in KS:
            (da, pa), (db, pb) = la("ILP-ADD", k), la("JointGreedy", k)
            rows[k] = dict(ilp=round(pa, 4), jg=round(pb, 4), delta=round(pa - pb, 4), **pct(da - db))
        d8 = np.mean([la("ILP-ADD", k)[0] - la("JointGreedy", k)[0] for k in KS], 0)
        lin[p] = dict(per_k=rows, mean_k=dict(point=round(float(np.mean([rows[k]["delta"] for k in KS])), 4),
                                               **pct(d8)))

    # ---------------- S-b tail at k=8 (666-pair trained two-logit accuracy)
    def pair_draws(arm):
        c = np.stack([c23[f"{arm}|8|{r}|cnt_pair666"] for r in range(5)])      # (5,666,25)
        per_seed = np.einsum("spc,bc->bsp", c, M) / (2.0 * tot[:, None, None])  # (NB,5,666)
        pick = np.take_along_axis(per_seed, S[:, 7, :][:, :, None], 1).mean(1)  # (NB,666)
        return pick, c.sum(2).mean(0) / (2.0 * NCFG)
    arms8 = ("ILP-ADD", "Greedy-ADD", "JointGreedy", "paper8")
    tail, pd, pp = {}, {}, {}
    pairs = json.load(open("./results/D021/evidence_tok200.meta.json"))["pairs"]
    for a in arms8:
        pd[a], pp[a] = pair_draws(a)
    cv = lambda x: np.array([cvar(row, 0.1) for row in x])
    for a in arms8:
        named = {n: round(float(pp[a][pairs.index(n.replace(" | ", "|"))]), 4) for n in NAMED}
        tail[a] = dict(cvar01=round(cvar(pp[a], 0.1), 4), worst=round(float(pp[a].min()), 4),
                       worst_pair=pairs[int(pp[a].argmin())], named=named)
    cvd = {a: cv(pd[a]) for a in arms8}
    for a in ("ILP-ADD", "Greedy-ADD", "paper8"):
        tail[a]["minus_JG"] = dict(point=round(tail[a]["cvar01"] - tail["JointGreedy"]["cvar01"], 4),
                                   **pct(cvd[a] - cvd["JointGreedy"]))

    # ---------------- S-c Goodhart
    E, _ = load_evidence()
    chains8 = {"ILP-ADD": sel["train_order"]["ILP_ADD"]["8"],
               "Greedy-ADD": sel["Greedy_ADD"]["chain"][:8],
               "JointGreedy": sel["existing_chains"]["JointGreedy"][:8],
               "paper8": sel["existing_chains"]["paper8"][:8]}
    good = {}
    for a, q in chains8.items():
        A = E[q].sum(0)
        rho = float(spearmanr(A, pp[a])[0])
        wa = set(np.lexsort((np.arange(666), A))[:66].tolist())
        wy = set(np.lexsort((np.arange(666), pp[a]))[:66].tolist())
        good[a] = dict(rho_ADD_vs_trained=round(rho, 4), worst66_overlap=len(wa & wy),
                       flag=bool(a in ("ILP-ADD", "Greedy-ADD") and rho < 0.445))

    json.dump(dict(
        schema="d023-analysis-v1", verdict=verdict,
        verdict_rule="BETTER: lo(mean k=1..8) > 0; WORSE: hi < 0; PARITY: interval of mean "
                     "over differing k within [-0.02, 0.02]; else INCONCLUSIVE (Call 2 (b))",
        bootstrap=dict(configs="boot_draws(25, 2000, seed=20260928)",
                       seeds="default_rng(20260928).integers(0,5,(2000,8,5)), joint across arms",
                       independent_sensitivity_seed_arm2="default_rng(20260930)",
                       note="5-with-replacement understates seed variance of a 5-seed mean by 4/5"),
        primary=dict(mean_k1_8=all8, mean_differing_k=dk, differing_k=diff_k,
                     independent_seed_sensitivity=dict(interval=ind, verdict_reading=v_ind),
                     schemes_agree=bool(v_ind == verdict)),
        per_k=per_k, S_a_linear=lin, S_b_tail_k8=tail, S_c_goodhart=good, S_d_decomposition=decomp,
        S_e_certificate=dict(certificate=sel["certificate"], stability=sel["stability"],
                             overlaps=sel["overlaps"], nesting=sel["nesting"])),
        open(f"{OUT}/analysis.json", "w"), indent=1)
    print(f"[S3] decomposition {decomp}", flush=True)
    print(f"[S3] linear mean-k {({p: lin[p]['mean_k'] for p in lin})}", flush=True)
    print(f"[S3] tail {tail}", flush=True)
    print(f"[S3] goodhart {good}", flush=True)


if __name__ == "__main__":
    main()
