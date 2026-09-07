"""
D007 / S4–S6 — resolution audit, T1.6 hard-tail check, and the F1 budget choice.

Per-pair coverage is `max_q S[q][p]` (I4: MAX over queries, never sum or mean).

Three things are computed for that coverage, and the ORDER matters:

  S4  RESOLUTION AUDIT, AS A GATE. Fraction of cells and -- more importantly --
      of PAIRS at/near ceiling, for both statistics. Per A3, a CVaR/mean ratio
      near 1 may NOT be reported as "no tail" while this audit shows censoring.
      This is the discipline D001 lacked and D004 made mandatory.

  F2  WINNER'S-CURSE CORRECTION. `max_q` over 259 noisy per-cell estimates is
      upward-biased. Correction: select `q* = argmax_q S_probe_A[q][p]` on config
      half A, then evaluate that query on the disjoint half B. Halves are equal
      size, so `max_A - B[q*]` isolates SELECTION bias rather than confounding it
      with sample size. T1.6's criterion is applied to the corrected estimate;
      the naive one is reported alongside. If they disagree on the verdict, that
      disagreement IS the finding and gets escalated -- design side said so
      explicitly when approving.

  S5  T1.6: 10th percentile of per-pair coverage > 0.9 => TAIL STILL ABSENT.

F1 additionally compares analysis budgets 100 vs 200 over the full 666 pairs,
replacing D006/R2's provisional 15-pair result (which was measured on embeddings
later shown to discard signal).

Usage:  PYTHONPATH=.:experiments python experiments/d007_analyze.py
"""
import os
import json
import itertools

import numpy as np

OUT = "./results/D007"
CEIL_PROBE = 0.99
GAMMA = 0.10


def load(tag):
    meta = json.load(open(f"{OUT}/tensor_{tag}.meta.json"))
    S = {k: np.load(f"{OUT}/S_{k}_{tag}.npy")
         for k in ("probe", "energy_raw", "energy_sf", "probe_A", "probe_B")}
    return meta, S


def cvar(x, g=GAMMA):
    """Mean of the worst (lowest-coverage) g fraction -- the hard tail."""
    k = max(1, int(np.floor(g * len(x))))
    return float(np.mean(np.sort(x)[:k]))


def coverage(S, meta):
    """Per-pair coverage under I4's MAX, naive and F2-corrected."""
    naive = S["probe"].max(axis=0)
    qstar = S["probe_A"].argmax(axis=0)
    corrected = S["probe_B"][qstar, np.arange(S["probe_B"].shape[1])]
    max_a = S["probe_A"].max(axis=0)
    energy_cov = S["energy_sf"].max(axis=0)
    return dict(naive=naive, corrected=corrected, max_a=max_a,
                bias=max_a - corrected, energy=energy_cov, qstar=qstar)


def audit(S, cov):
    p = S["probe"]
    return dict(
        probe_cells_ge_ceiling=round(float((p >= CEIL_PROBE).mean()), 4),
        probe_cells_ge_0999=round(float((p >= 0.999).mean()), 4),
        probe_cells_distinct=int(len(np.unique(np.round(p, 4)))),
        probe_PAIRS_at_ceiling_naive=round(float((cov["naive"] >= CEIL_PROBE).mean()), 4),
        probe_PAIRS_at_ceiling_corrected=round(float((cov["corrected"] >= CEIL_PROBE).mean()), 4),
        energy_sf_cells_min=round(float(S["energy_sf"].min()), 5),
        energy_sf_cells_max=round(float(S["energy_sf"].max()), 5),
        energy_sf_distinct=int(len(np.unique(np.round(S["energy_sf"], 4)))),
        energy_PAIRS_range=[round(float(cov["energy"].min()), 5),
                            round(float(cov["energy"].max()), 5)],
        note="A bounded statistic pinned near its ceiling measures the ceiling, "
             "not the population (D001's failure; D004's mandatory check). "
             "Energy distance is unbounded and cannot saturate this way.")


def tail(x, label):
    return {f"{label}_mean": round(float(np.mean(x)), 4),
            f"{label}_p10": round(float(np.percentile(x, 10)), 4),
            f"{label}_p50": round(float(np.percentile(x, 50)), 4),
            f"{label}_min": round(float(np.min(x)), 4),
            f"{label}_cvar10": round(cvar(x), 4),
            f"{label}_cvar_over_mean": round(cvar(x) / float(np.mean(x)), 4)}


def main():
    tags = [t for t in ("tok200", "tok100")
            if os.path.exists(f"{OUT}/tensor_{t}.meta.json")]
    assert tags, "no tensor built yet"
    res = {}

    for tag in tags:
        meta, S = load(tag)
        cov = coverage(S, meta)
        a = audit(S, cov)
        r = dict(shape=meta["shape"], audit=a)
        for lab, x in (("naive", cov["naive"]), ("corrected", cov["corrected"]),
                       ("energy", cov["energy"])):
            r.update(tail(x, lab))
        r["f2_bias"] = dict(
            mean=round(float(cov["bias"].mean()), 4),
            median=round(float(np.median(cov["bias"])), 4),
            p90=round(float(np.percentile(cov["bias"], 90)), 4),
            frac_positive=round(float((cov["bias"] > 0).mean()), 4),
            note="max_A - B[q*], equal sample sizes, so this is selection bias")
        # T1.6, applied to the corrected estimate (approved), naive reported too
        r["t16"] = dict(
            criterion="10th percentile of per-pair coverage > 0.9 => TAIL STILL ABSENT",
            naive_p10=r["naive_p10"], naive_fires=bool(r["naive_p10"] > 0.9),
            corrected_p10=r["corrected_p10"], corrected_fires=bool(r["corrected_p10"] > 0.9),
            disagree=bool((r["naive_p10"] > 0.9) != (r["corrected_p10"] > 0.9)))
        res[tag] = r

    # ---- F1: which analysis budget?
    if "tok100" in res and "tok200" in res:
        _, S1 = load("tok100"); _, S2 = load("tok200")
        c1, c2 = coverage(S1, None), coverage(S2, None)
        d_cell = S1["probe"] - S2["probe"]
        d_cov = c1["corrected"] - c2["corrected"]
        de = c1["energy"] - c2["energy"]
        res["F1_budget_comparison"] = dict(
            n_pairs=int(len(d_cov)), n_cells=int(d_cell.size),
            cell_probe_mean_100=round(float(S1["probe"].mean()), 4),
            cell_probe_mean_200=round(float(S2["probe"].mean()), 4),
            cell_mean_delta_100_minus_200=round(float(d_cell.mean()), 5),
            cells_where_100_higher=round(float((d_cell > 0).mean()), 4),
            corrected_coverage_delta=round(float(d_cov.mean()), 5),
            pairs_where_100_higher=round(float((d_cov > 0).mean()), 4),
            energy_coverage_delta=round(float(de.mean()), 5),
            pairs_where_100_higher_energy=round(float((de > 0).mean()), 4),
            supersedes="D006/R2's provisional 15-pair result, measured on "
                       "normalised embeddings later shown to discard signal")

    json.dump(res, open(f"{OUT}/analysis.json", "w"), indent=1)

    for tag in tags:
        r = res[tag]
        print(f"\n===== {tag} =====")
        a = r["audit"]
        print(f"S4 audit: cells>=0.99 {a['probe_cells_ge_ceiling']:.1%}  "
              f"PAIRS at ceiling naive {a['probe_PAIRS_at_ceiling_naive']:.1%} / "
              f"corrected {a['probe_PAIRS_at_ceiling_corrected']:.1%}")
        print(f"          energy_sf range {a['energy_PAIRS_range']} (unbounded)")
        for lab in ("naive", "corrected", "energy"):
            print(f"  {lab:9s} mean {r[lab+'_mean']:.4f}  p10 {r[lab+'_p10']:.4f}  "
                  f"CVaR10 {r[lab+'_cvar10']:.4f}  CVaR/mean {r[lab+'_cvar_over_mean']:.4f}")
        b = r["f2_bias"]
        print(f"  F2 selection bias: mean {b['mean']:+.4f}, positive on {b['frac_positive']:.1%} of pairs")
        t = r["t16"]
        print(f"  T1.6  naive p10 {t['naive_p10']:.4f} -> fires={t['naive_fires']}   "
              f"corrected p10 {t['corrected_p10']:.4f} -> fires={t['corrected_fires']}"
              + ("   *** DISAGREE ***" if t["disagree"] else ""))

    if "F1_budget_comparison" in res:
        f = res["F1_budget_comparison"]
        print(f"\n===== F1: analysis budget =====")
        print(f"  cell probe AUC: 100tok {f['cell_probe_mean_100']:.4f} vs "
              f"200tok {f['cell_probe_mean_200']:.4f} "
              f"(100 higher on {f['cells_where_100_higher']:.1%} of {f['n_cells']:,} cells)")
        print(f"  corrected coverage delta {f['corrected_coverage_delta']:+.5f} "
              f"(100 higher on {f['pairs_where_100_higher']:.1%} of pairs)")
        print(f"  energy coverage delta    {f['energy_coverage_delta']:+.5f} "
              f"(100 higher on {f['pairs_where_100_higher_energy']:.1%} of pairs)")
    print(f"\nwritten: {OUT}/analysis.json")


if __name__ == "__main__":
    main()
