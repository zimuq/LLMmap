"""D030 / S3 POST HOC (not pre-registered; written after the labels were read). Descriptive only.

Why: the v2 WORSE label for MET comes from E_hard over H_all, and H_all is the set of pairs on which an
LLMmap-type classifier (paper8 B0', D027) scored < 0.90 on S_val (D028 rule). A method with an
independent representation is not subject to that selection. This script puts intervals on the
unselected pair sets, ours (attention k = 8) minus each MET / ZP All cell, v2 view, D029's draws:
  - all 3,570 pairs: mean and CVaR_0.1 of two-logit accuracy;
  - N' (161 structural pairs, unselected), and its complement in H_N terms (N' minus H_N);
  - H_N and H_X separately (selected).
Writes results/D030/posthoc_v2_pairsets.json. The pre-registered labels are unchanged (Review B3).
Usage:  PYTHONPATH=.:experiments python experiments/d030_posthoc.py
"""
import json
import itertools

import numpy as np

import d029_lib as L29
from d008_lib import boot_draws
from LLMmap.joint_statistic import cvar
from d030_s3_analysis import two_logit_all, ci, r4

RES = "./results/D030"
NB, NC = 2000, 25


def main():
    models = json.load(open("./results/D027/universe_v2.json"))["models"]
    tg = L29.targets(models)
    P = list(itertools.combinations(range(85), 2))
    pidx = {f"{models[a]} | {models[b]}": j for j, (a, b) in enumerate(P)}
    nprime = [pidx[p["pair"]] for p in json.load(open("./results/D027/n_prime.json"))["pairs"]]
    hn = [pidx[p] for p in tg["strata"]["H_N"]]
    sets = dict(all3570=list(range(len(P))), N_prime=nprime, N_prime_minus_H_N=sorted(set(nprime) - set(hn)),
                H_N=hn, H_X=[pidx[p] for p in tg["strata"]["H_X"]])
    M = boot_draws(NC, NB, seed=20261005)
    S = np.random.default_rng(20261005).integers(0, 5, (NB, 24, 5))[:, 7, :]
    y, c = np.repeat(np.arange(85), NC), np.tile(np.arange(NC), 85)
    lg = np.load("./results/D029/logits_JG.npz")
    ours = np.stack([two_logit_all(lg[f"att|8|{r}"], y, c, 85, P).sum(1) for r in range(5)])     # (5, P, 25)
    per = np.einsum("rpc,bc->brp", ours, M) / (2 * NC)
    od = np.take_along_axis(per, S[:, :, None], 1).mean(1)                                      # (NB, P)
    op = ours.sum(2).mean(0) / (2 * NC)
    out = dict(schema="d030-posthoc-v1", status="POST HOC, descriptive; labels unchanged (Review B3)",
               sets={k: len(v) for k, v in sets.items()}, rows={})
    for meth, var in (("met", "e5"), ("met", "hamming"), ("zp", "512"), ("zp", "200")):
        b = np.load(f"{RES}/counts_{meth}_{var}_All_v2.npz")["cnt_pair_all"]                     # (P, 25)
        bd = (M @ b.T) / (2 * NC)
        bp = b.sum(1) / (2 * NC)
        row = {}
        for k, ix in sets.items():
            row[f"{k}|mean"] = dict(ours=r4(op[ix].mean()), base=r4(bp[ix].mean()),
                                    delta=r4(op[ix].mean() - bp[ix].mean()),
                                    ci=ci(od[:, ix].mean(1) - bd[:, ix].mean(1)))
        cv_o = np.array([cvar(x, 0.1) for x in od])
        cv_b = np.array([cvar(x, 0.1) for x in bd])
        row["all3570|cvar01"] = dict(ours=r4(cvar(op, 0.1)), base=r4(cvar(bp, 0.1)),
                                     delta=r4(cvar(op, 0.1) - cvar(bp, 0.1)), ci=ci(cv_o - cv_b))
        out["rows"][f"{meth}-{var} All"] = row
        print(meth, var, {k: (v["delta"], v["ci"]) for k, v in row.items()}, flush=True)
    json.dump(out, open(f"{RES}/posthoc_v2_pairsets.json", "w"), indent=1)


if __name__ == "__main__":
    main()
