"""D028 / S0 steps 1-2 — freeze the v2 hard sets and the B0' confusion matrix,
from D027's saved artefacts only (nothing is trained; S_val logits only; I2).

H_N  = D027 H' (structural, 38), with M_N / F_N = D027 M_H' / F_H' (copied,
       source file hash recorded and asserted).
H_X  = non-N' pairs with 5-seed-mean two-logit S_val accuracy < 0.90 (D027's
       population check, recomputed here from the logits and asserted equal).
       M_X = their models. No family closure (D028 ## D).
H_all = H_N | H_X.
Bands for H_X (and recomputed for H_N as a check): D027's method exactly --
config bootstrap on S_val, boot_draws(25, 2000, seed=20261002), seeds averaged
within a draw.
Confusion: 85 x 85 S_val counts (rows = true model, cols = argmax), per seed and
summed.

Usage:  PYTHONPATH=.:experiments python experiments/d028_hardsets.py
"""
import json
import hashlib
import itertools

import numpy as np

from d008_lib import boot_draws

D27 = "./results/D027"
OUT = "./results/D028"
TAU, NB, SEED = 0.90, 2000, 20261002


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def two_logit_counts(lg, y, cfg, a, b, ncfg=25):
    out = np.zeros(ncfg)
    for d, m in enumerate((a, b)):
        sel = y == m
        ok = lg[sel, a] >= lg[sel, b] if d == 0 else lg[sel, b] > lg[sel, a]
        out += np.bincount(cfg[sel][ok], minlength=ncfg)
    return out


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    hs = json.load(open(f"{D27}/hard_set_v2.json"))
    an = json.load(open(f"{D27}/analysis.json"))
    uni = json.load(open(f"{D27}/universe_v2.json"))
    models = uni["models"]
    ix = {m: i for i, m in enumerate(models)}
    L = np.load(f"{D27}/logits_b0.npz")
    y, cfg = L["y"], L["cfg"]
    seeds = [L[f"b0_r{r}"] for r in range(5)]
    M = boot_draws(25, NB, seed=SEED)
    Nset = {p["pair"] for p in json.load(open(f"{D27}/n_prime.json"))["pairs"]}

    def stats(p):
        a, b = (ix[m] for m in p.split(" | "))
        C = np.stack([two_logit_counts(lg, y, cfg, a, b) for lg in seeds])
        acc = float(C.sum(1).mean() / 50.0)
        bt = (M @ C.mean(0)) / (2.0 * M.sum(1))
        lo, hi = np.percentile(bt, [2.5, 97.5])
        band = "stably_hard" if hi < TAU else ("stably_easy" if lo > TAU else "uncertain")
        return dict(pair=p, acc=round(acc, 4), ci=[round(float(lo), 4), round(float(hi), 4)], band=band)

    # H_X: recompute over every non-N' pair
    allX = []
    for a, b in itertools.combinations(models, 2):
        p = f"{a} | {b}"
        if p not in Nset:
            s = stats(p)
            if s["acc"] < TAU:
                allX.append(s)
    allX.sort(key=lambda r: r["acc"])
    d27x = sorted(r["pair"] for r in an["population_check_nonN_below_tau"])
    assert sorted(r["pair"] for r in allX) == d27x, "H_X recomputation != D027 population check"
    H_X = [r["pair"] for r in allX]
    M_X = sorted({m for p in H_X for m in p.split(" | ")})
    HN = [stats(p) for p in hs["H"]]
    d27band = {r["pair"]: r["band"] for r in an["pair_table"]}
    assert all(r["band"] == d27band[r["pair"]] for r in HN), "H_N band recomputation != D027"

    out = dict(
        schema="d028-hard-sets-v2", frozen="S0, before any v2 pool generation",
        rule=("hard = paper8 B0' (D027: 85-way, D009 attention protocol, seeds 0-4) two-logit "
              "accuracy on S_val < 0.90, applied to ALL 3,570 pairs; strata H_N (structural, "
              "N' = same base or same lineage) and H_X (non-structural)"),
        decision="human, 2026-10-03 (D028 ## D: 'hard = structural hard + unrelated hard'; "
                 "revision of DECISIONS.md D7 for the v2 universe)",
        sources=dict(d027_hard_set_v2_sha256=sha(f"{D27}/hard_set_v2.json"),
                     d027_analysis_sha256=sha(f"{D27}/analysis.json"),
                     d027_logits_b0_sha256=sha(f"{D27}/logits_b0.npz"),
                     universe_v2_sha256=sha(f"{D27}/universe_v2.json")),
        bands_method=f"config bootstrap on S_val, boot_draws(25, {NB}, seed={SEED}), seeds averaged within a draw",
        H_N=hs["H"], M_N=hs["M_H"], F_N=hs["F_H"], family_groups_N=hs["family_groups"],
        H_N_rows=HN, H_X=H_X, M_X=M_X, H_X_rows=allX, H_all=hs["H"] + H_X,
        counts=dict(H_N=len(hs["H"]), H_X=len(H_X), H_all=len(hs["H"]) + len(H_X), M_X=len(M_X),
                    H_X_bands={b: sum(r["band"] == b for r in allX) for b in ("stably_hard", "uncertain", "stably_easy")},
                    H_N_bands={b: sum(r["band"] == b for r in HN) for b in ("stably_hard", "uncertain", "stably_easy")}))
    json.dump(out, open(f"{OUT}/hard_sets_v2.json", "w"), indent=1)

    nm = len(models)
    per = []
    for lg in seeds:
        Cm = np.zeros((nm, nm), int)
        np.add.at(Cm, (y, lg.argmax(1)), 1)
        per.append(Cm)
    json.dump(dict(schema="d028-b0prime-confusion-sval", models=models, rows="true model", cols="argmax prediction",
                   n_per_row_per_seed=25, summed=np.sum(per, 0).tolist(), per_seed=[c.tolist() for c in per],
                   source="results/D027/logits_b0.npz (S_val)"),
              open(f"{OUT}/b0prime_confusion_sval.json", "w"))
    print(f"H_N {len(hs['H'])} H_X {len(H_X)} H_all {len(out['H_all'])} M_X {len(M_X)}; bands X {out['counts']['H_X_bands']} "
          f"N {out['counts']['H_N_bands']}")


if __name__ == "__main__":
    main()
