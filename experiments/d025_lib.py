"""D025 — shared pieces: the selection wrapper, arm definitions, and the
logit-derived counts every readout stores.

Selection: D010's code path verbatim (`LLMmap.joint_greedy.joint_greedy`, scale-
free joint energy, gamma=0.1, cvar m = max(1, floor(0.1*|P|)), ties -> lowest id)
on D010's precomputed accumulators (`d010_select.precompute`). Only the pair set
P that the CVaR runs over changes between arms:
  GLOBAL   all 666 pairs
  SPEC(p)  {p}, one per H pair
  FAM(F)   all pairs among the models of family group F
"""
import json
import itertools

import numpy as np

from LLMmap.joint_greedy import joint_greedy

OUT = "./results/D025"
GAMMA, K_MAX = 0.1, 16
D010_CHAIN = [193, 140, 237, 114, 233, 117, 0, 16]


def short(pair):
    """'org/A | org/B' -> 'A|B' for arm names."""
    return "|".join(m.split("/")[1] for m in pair.split(" | "))


def arm_definitions(models):
    """{arm: dict(kind, P (pair indices in D007 order), target)} from hard_set.json."""
    hs = json.load(open(f"{OUT}/hard_set.json"))
    pairs = list(itertools.combinations(models, 2))
    pidx = {f"{a} | {b}": i for i, (a, b) in enumerate(pairs)}
    arms = {"GLOBAL": dict(kind="GLOBAL", P=list(range(len(pairs))), target=None)}
    for p, i in zip(hs["H"], hs["H_index_666"]):
        assert pidx[p] == i
        arms[f"SPEC:{short(p)}"] = dict(kind="SPEC", P=[i], target=p)
    for g, hp in zip(hs["family_groups"], hs["family_group_H_pairs"]):
        name = f"FAM:{hp[0].split(' | ')[0].split('/')[1].split('-')[0]}"
        P = sorted(pidx[f"{a} | {b}"] for a, b in itertools.combinations(g, 2))
        arms[name] = dict(kind="FAM", P=P, target=hp[0], models=g)
    return arms, pairs, hs


def select(XY2, XX2, ia, ib, P, k=K_MAX):
    """Nested chain of k queries maximising CVaR_0.1 over pair set P."""
    if len(P) == XY2.shape[1]:
        return joint_greedy(XY2, XX2, ia, ib, k, GAMMA, "energy", verbose=False)
    P = np.asarray(P)
    return joint_greedy(XY2[:, P], XX2, ia[P], ib[P], k, GAMMA, "energy", verbose=False)


# ------------------------------------------------------------------ counts
def dir_counts(lg, y, cfg, pa, pb, ncfg=25):
    """(n_pairs, 2, ncfg): two-logit restricted correct counts per config,
    direction 0 = traces of model a, 1 = traces of model b."""
    out = np.zeros((len(pa), 2, ncfg), np.int16)
    for j, (a, b) in enumerate(zip(pa, pb)):
        for d, m in enumerate((a, b)):
            sel = y == m
            # logit_stats' tie rule: argmax over [a, b] picks a on a tie
            ok = lg[sel, a] >= lg[sel, b] if d == 0 else lg[sel, b] > lg[sel, a]
            out[j, d] = np.bincount(cfg[sel][ok], minlength=ncfg)
    return out


def fam_counts(lg, y, cfg, idx, ncfg=25):
    """(len(idx), ncfg): family-restricted argmax correct (argmax over the
    family's logits only), per model of the family, per config."""
    idx = np.asarray(idx)
    out = np.zeros((len(idx), ncfg), np.int16)
    sub = lg[:, idx]
    for r, m in enumerate(idx):
        sel = y == m
        ok = idx[sub[sel].argmax(1)] == m
        out[r] = np.bincount(cfg[sel][ok], minlength=ncfg)
    return out
