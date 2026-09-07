"""
D007 / S2a — does `S_probe` saturate at the REAL per-cell sample size?

The S6b pilot measured probe AUC with 600 points per cloud and found 0% of pairs
at ceiling, which is part of the evidence A3 was decided on. The tensor's actual
cells are **75 vs 75 in 1024 dimensions** — an order of magnitude fewer points,
and firmly p >> n. That is a different regime, and whether a bounded statistic
saturates there is exactly the question D004's resolution-audit-as-gate exists to
force. Answering it costs ~1 minute; discovering it after building the full
tensor costs the tensor.

Reports, over ~500 cells spanning easy and near-relative pairs:
  * distribution of S_probe, fraction at/near ceiling and floor, distinct values
  * the same for S_energy (scale-free), which has no ceiling by construction
  * a self-pair control: a model against ITSELF must give AUC ~ 0.5. If it does
    not, the instrument is overfitting and every cell is suspect -- this is the
    single most informative check available at n=75.

Usage:  PYTHONPATH=. python experiments/d007_s2a_pilot.py
"""
import csv
import json
import random
import itertools

import numpy as np

from d007_lib import load_corpus, probe_auc, energy

OUT = "./results/D007/s2a_pilot.json"
META = "./results/D001/model_metadata.csv"
N_CELLS = 500
N_SELF = 40


def near_relative(models):
    rows = {r["model"]: r for r in csv.DictReader(open(META))}
    out = set()
    for a, b in itertools.combinations(models, 2):
        ra, rb = rows.get(a, {}), rows.get(b, {})
        if (ra.get("base") and ra["base"] == rb.get("base")) or \
           (ra.get("lineage") and ra["lineage"] == rb.get("lineage")):
            out.add((a, b))
    return out


def main():
    import os
    os.makedirs("./results/D007", exist_ok=True)
    random.seed(0)
    models, X = load_corpus(pool="build")
    nq = X[models[0]].shape[0]
    print(f"{len(models)} models, {nq} queries, "
          f"{X[models[0]].shape[1]} configs/cloud, dim {X[models[0]].shape[2]}",
          flush=True)

    kin = near_relative(models)
    allp = list(itertools.combinations(models, 2))
    hard = [p for p in allp if p in kin]
    easy = [p for p in allp if p not in kin]
    print(f"pairs: {len(allp)} total, {len(hard)} near-relative", flush=True)

    # deliberately over-sample near-relative pairs: if anything is going to
    # NOT saturate it is those, and if even they saturate the gate has fired
    cells = ([(random.choice(hard), random.randrange(nq)) for _ in range(N_CELLS // 2)] +
             [(random.choice(easy), random.randrange(nq)) for _ in range(N_CELLS // 2)])

    rec = []
    for (a, b), q in cells:
        A, B = X[a][q], X[b][q]
        auc = probe_auc(A, B)
        raw, sf = energy(A, B)
        rec.append(dict(pair=f"{a}|{b}", q=q, near_relative=(a, b) in kin,
                        auc=auc, energy_raw=raw, energy_sf=sf))
        if len(rec) % 100 == 0:
            print(f"  {len(rec)}/{len(cells)}", flush=True)

    # ---- self-pair control: split ONE model's 75 configs in half and separate
    # the halves. True AUC is 0.5; anything systematically above it is the
    # probe fitting noise, and that would inflate every cell in the tensor.
    self_auc = []
    for _ in range(N_SELF):
        m = random.choice(models)
        q = random.randrange(nq)
        C = X[m][q]
        idx = np.random.default_rng(len(self_auc)).permutation(len(C))
        h = len(C) // 2
        self_auc.append(probe_auc(C[idx[:h]], C[idx[h:2 * h]]))

    auc = np.array([r["auc"] for r in rec])
    sf = np.array([r["energy_sf"] for r in rec])
    sa = np.array(self_auc)
    hard_auc = np.array([r["auc"] for r in rec if r["near_relative"]])

    out = dict(
        n_cells=len(rec), n_per_cloud=int(X[models[0]].shape[1]),
        dim=int(X[models[0]].shape[2]),
        probe=dict(mean=round(float(auc.mean()), 4), median=round(float(np.median(auc)), 4),
                   p10=round(float(np.percentile(auc, 10)), 4),
                   min=round(float(auc.min()), 4), max=round(float(auc.max()), 4),
                   frac_ge_0999=round(float((auc >= 0.999).mean()), 4),
                   frac_ge_099=round(float((auc >= 0.99).mean()), 4),
                   frac_le_055=round(float((auc <= 0.55).mean()), 4),
                   distinct=int(len(np.unique(np.round(auc, 4))))),
        probe_near_relative=dict(n=len(hard_auc),
                                 mean=round(float(hard_auc.mean()), 4),
                                 frac_ge_099=round(float((hard_auc >= 0.99).mean()), 4)),
        energy_scale_free=dict(mean=round(float(sf.mean()), 5),
                               p10=round(float(np.percentile(sf, 10)), 5),
                               min=round(float(sf.min()), 5), max=round(float(sf.max()), 5)),
        self_pair_control=dict(
            n=len(sa), mean=round(float(sa.mean()), 4),
            p95=round(float(np.percentile(sa, 95)), 4), max=round(float(sa.max()), 4),
            note="true AUC is 0.5 by construction; a mean materially above that "
                 "means the probe fits noise at n=75 and every cell is inflated"),
        cells=rec[:50])

    sat = out["probe"]["frac_ge_099"] > 0.20
    infl = out["self_pair_control"]["mean"] > 0.60
    out["verdict"] = dict(
        probe_saturating=bool(sat), probe_inflated_at_n75=bool(infl),
        conclusion=("S_probe SATURATES at n=75 -- A3's gate fires, S_energy is "
                    "authoritative for interpretation" if sat else
                    "S_probe has room to show spread at n=75") +
                   ("; AND the self-pair control is inflated, so the probe is "
                    "fitting noise -- report this as an instrument limitation"
                    if infl else "; self-pair control is near 0.5 as it should be"))
    json.dump(out, open(OUT, "w"), indent=1)

    print(f"\nS_probe   mean {out['probe']['mean']}  p10 {out['probe']['p10']}  "
          f">=0.99 {out['probe']['frac_ge_099']:.1%}  >=0.999 {out['probe']['frac_ge_0999']:.1%}")
    print(f"  near-relative cells: mean {out['probe_near_relative']['mean']}  "
          f">=0.99 {out['probe_near_relative']['frac_ge_099']:.1%}")
    print(f"S_energy  scale-free mean {out['energy_scale_free']['mean']}  "
          f"range [{out['energy_scale_free']['min']}, {out['energy_scale_free']['max']}]")
    print(f"SELF-PAIR mean {out['self_pair_control']['mean']} (should be ~0.5), "
          f"p95 {out['self_pair_control']['p95']}")
    print(f"\nVERDICT: {out['verdict']['conclusion']}")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
