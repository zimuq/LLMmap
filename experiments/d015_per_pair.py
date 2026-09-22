"""D015 / S0–S6 — per-pair decomposition of `hard_subset` over the 65 structural pairs.

Read-only over already-frozen runs (I6). Nothing is trained, selected, embedded or
generated. Zero GPU; runs on a login node in seconds.

WHAT THIS CAN AND CANNOT ANSWER (P1/F3, approved). The aggregate being decomposed
is +0.41 pp (joint energy - coverage at k=8) over 65 pairs x 2 models x 25 configs
= 3,250 trace decisions -- about 13 trace flips in total, 0.21 per pair. The
smallest step any single pair can show is one trace of 50 = 2.00 pp. So "most
pairs improved a little" is arithmetically unobservable: whatever appears is a few
pairs at +/-2 pp and the rest at exactly zero, forced by the denominator. The
answerable half is "does any pair move BACKWARD" -- the redistribution an aggregate
hides -- and that is what the moved-backward table reports.

WHERE THE NUMBERS COME FROM (P1/F1, approved). D012 and D013 store runs only for
the gammas they TRAINED; the ones they reused under I6 have none of their own:
    JointGreedy gamma=0.1   -> results/D010 (`joint_energy`)
    GreedyCover gamma=1.0   -> results/D009 (`mean_greedy_max`)
    GreedyCover gamma=0.1   -> results/D009 (`cvar_max`)
D014's 13 further (algorithm, gamma) cells are included per Call 1 (approved):
already-frozen, zero new compute, and they take S3's curve from 4-5 gamma values
to 11. Every cell is labelled with its source D.

Usage:  PYTHONPATH=.:experiments python experiments/d015_per_pair.py
"""
import os
import json
import time
import itertools

import numpy as np

from d008_lib import near_relative_pairs, boot_draws, structural_diagnostics

OUT = "./results/D015"
N_BOOT = 2000
NCFG = 25
NAMED = ["tiiuae/Falcon3-10B-Instruct | tiiuae/Falcon3-7B-Instruct",
         "microsoft/Phi-3-medium-128k-instruct | microsoft/Phi-3-medium-4k-instruct"]

# (source file, counts file, prefix-in-counts) for every (algorithm, gamma) cell
SRC = {
    ("GreedyCover", "1.0"): ("D009", "mean_greedy_max"),
    ("GreedyCover", "0.1"): ("D009", "cvar_max"),
    ("GreedyCover", "0.5"): ("D013", "0.5"),
    ("GreedyCover", "0.25"): ("D013", "0.25"),
    ("GreedyCover", "0.05"): ("D013", "0.05"),
    ("JointGreedy", "0.1"): ("D010", "joint_energy"),
    ("JointGreedy", "1.0"): ("D012", "1.0"),
    ("JointGreedy", "0.25"): ("D012", "0.25"),
    ("JointGreedy", "0.05"): ("D012", "0.05"),
}
for g in ("0.75", "0.6", "0.45", "0.4", "0.35", "0.3"):
    SRC[("GreedyCover", g)] = ("D014", f"GreedyCover|{g}")
for g in ("0.75", "0.6", "0.5", "0.45", "0.4", "0.35", "0.3"):
    SRC[("JointGreedy", g)] = ("D014", f"JointGreedy|{g}")

_CACHE = {}


def counts_for(where, prefix, k):
    """(n_seeds, 65, 25) correct-counts for one (condition, k), from stored runs."""
    key = (where, "counts")
    if key not in _CACHE:
        _CACHE[key] = np.load(f"./results/{where}/run_counts.npz")
    c = _CACHE[key]
    names = sorted(n for n in c.files
                   if n.startswith(f"{prefix}|{k}|") and n.endswith("cnt_pair"))
    if not names:
        return None
    return np.stack([c[n] for n in names])


def pair_acc(cp, M):
    """Per-pair accuracy and bootstrap draws from correct-counts.

    cp: (n_seeds, 65, ncfg). Each config contributes 2 traces to a pair (one per
    model), so a pair's accuracy is its count sum over 2*n_configs. Bootstrap
    resamples CONFIGS with the shared multiplicity vectors, then averages the
    per-seed draws -- D009's own convention, applied per pair instead of averaged
    over pairs (P1/F2, approved).
    """
    ncfg = cp.shape[2]
    mean = cp.sum(axis=2).mean(axis=0) / (2.0 * ncfg)        # (65,)
    tot = M.sum(1)                                            # (n_boot,)
    draws = np.mean([(cp[s] @ M.T) / (2.0 * tot) for s in range(cp.shape[0])],
                    axis=0)                                   # (65, n_boot)
    return mean, draws


def ci_of(x):
    return dict(lo=round(float(np.percentile(x, 2.5)), 4),
                hi=round(float(np.percentile(x, 97.5)), 4))


def delta(a_draws, b_draws, a_mean, b_mean, i):
    d = a_draws[i] - b_draws[i]
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return dict(delta=round(float(a_mean[i] - b_mean[i]), 4),
                lo=round(lo, 4), hi=round(hi, 4),
                resolved=bool(lo > 0 or hi < 0),
                sign=("+" if a_mean[i] > b_mean[i] else
                      "-" if a_mean[i] < b_mean[i] else "0"))


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    models = json.load(open("./results/D008/selection.json"))["models"]
    hard = near_relative_pairs(models)
    keys = sorted(hard)
    names = [" | ".join(hard[i]["pair"]) for i in keys]
    assert len(keys) == 65, len(keys)
    M = boot_draws(NCFG, N_BOOT)

    # ---------------- S0: indexing check
    s0 = dict(n_pairs=len(keys),
              model_list_stored_in={"D009": True, "D010": False,
                                    "D012": False, "D013": False, "D014": False},
              method="verified by construction + shape, not by stored metadata "
                     "(P1/S0): all sources derive the index from "
                     "near_relative_pairs(models) with models read from "
                     "results/D008/selection.json, row order sorted(hard_pairs)",
              shapes={})
    for where in ("D009", "D010", "D012", "D013", "D014"):
        c = np.load(f"./results/{where}/run_counts.npz")
        cp = [n for n in c.files if n.endswith("cnt_pair")]
        shp = c[cp[0]].shape
        s0["shapes"][where] = dict(n_cnt_pair=len(cp), shape=list(shp))
        assert shp == (65, NCFG), (where, shp)
    inv = ", ".join(f"{w}:{v['n_cnt_pair']}" for w, v in s0["shapes"].items())
    print(f"[S0] 65 pairs; cnt_pair (65,{NCFG}) in all five sources ({inv})",
          flush=True)

    # ---------------- S1/S2: the 65 x 3 tables
    METH = {"paper8": ("D009", "paper8"),
            "coverage": ("D009", "cvar_max"),
            "joint_energy": ("D010", "joint_energy")}
    tables = {}
    for k in (8, 1):
        mean, draws = {}, {}
        for m, (where, pref) in METH.items():
            cp = counts_for(where, pref, k)
            assert cp is not None, (m, k)
            mean[m], draws[m] = pair_acc(cp, M)
        rows = []
        for i, (key, nm) in enumerate(zip(keys, names)):
            a, b = hard[key]["pair"]
            r = dict(pair=nm, index=int(key))
            for m in METH:
                r[m] = round(float(mean[m][i]), 4)
                r[m + "_ci"] = ci_of(draws[m][i])
            r["d_joint_minus_coverage"] = delta(draws["joint_energy"],
                                                draws["coverage"],
                                                mean["joint_energy"],
                                                mean["coverage"], i)
            r["d_coverage_minus_paper8"] = delta(draws["coverage"], draws["paper8"],
                                                 mean["coverage"], mean["paper8"], i)
            r.update(structural_diagnostics(a, b))
            rows.append(r)
        rows.sort(key=lambda r: r["paper8"])
        tables[k] = rows
        agg = {m: round(float(mean[m].mean()), 4) for m in METH}
        n_zero = sum(1 for r in rows if r["d_joint_minus_coverage"]["delta"] == 0)
        print(f"[S{1 if k==8 else 2}] k={k}: aggregate {agg} | "
              f"{n_zero}/65 pairs have joint-coverage delta exactly 0", flush=True)
        json.dump(dict(schema="d015-per-pair-v1", k=k, n_pairs=65,
                       sort="ascending paper8 accuracy -- a DISPLAY convenience, "
                            "not a hardness claim (D015/S1)",
                       chance=0.5, n_boot=N_BOOT,
                       resolution_floor="one trace of 50 = 2.00 pp; no pair can "
                                        "move less (P1/F3)",
                       aggregate=agg, rows=rows),
                  open(f"{OUT}/per_pair_k{k}.json", "w"), indent=1)

    # ---------------- S3: the two named pairs across every stored (gamma, k)
    idx = {nm: i for i, nm in enumerate(names)}
    missing = [n for n in NAMED if n not in idx]
    assert not missing, missing
    long = []
    for (alg, g), (where, pref) in sorted(SRC.items()):
        for k in range(1, 9):
            cp = counts_for(where, pref, k)
            if cp is None:
                continue
            mean, draws = pair_acc(cp, M)
            for nm in NAMED:
                i = idx[nm]
                long.append(dict(pair=nm, algorithm=alg, gamma=float(g), k=k,
                                 mean=round(float(mean[i]), 4),
                                 **ci_of(draws[i]), n_seeds=int(cp.shape[0]),
                                 source_D=where))
    gam = {a: sorted({r["gamma"] for r in long if r["algorithm"] == a}, reverse=True)
           for a in ("GreedyCover", "JointGreedy")}
    print(f"[S3] {len(long)} rows; gammas GreedyCover {gam['GreedyCover']}", flush=True)
    print(f"     gammas JointGreedy {gam['JointGreedy']}", flush=True)
    json.dump(dict(schema="d015-named-gamma-k-v1", pairs=NAMED,
                   gammas_available=gam, n_rows=len(long),
                   note="every cell is an already-frozen trained run; D014's cells "
                        "included per Call 1 (approved 2026-09-22). Nothing "
                        "interpolated or newly computed.",
                   rows=long), open(f"{OUT}/per_pair_gamma_named.json", "w"), indent=1)

    # ---------------- S6: presentation views
    r8 = tables[8]
    hardest = r8[:8]
    ranks = {nm: next(i + 1 for i, r in enumerate(r8) if r["pair"] == nm)
             for nm in NAMED}
    # rank under each method independently (F4's corroboration uses paper8's order)
    per_method_rank = {}
    for m in METH:
        order = sorted(r8, key=lambda r: r[m])
        per_method_rank[m] = {nm: next(i + 1 for i, r in enumerate(order)
                                       if r["pair"] == nm) for nm in NAMED}
    json.dump(dict(schema="d015-hardest8-v1",
                   sorted_by="paper8 accuracy at k=8 (display sort)",
                   named_pair_ranks_of_65=ranks,
                   named_pair_rank_by_method=per_method_rank,
                   rows=[{kk: v for kk, v in r.items()
                          if kk in ("pair", "paper8", "coverage", "joint_energy",
                                    "d_joint_minus_coverage",
                                    "d_coverage_minus_paper8", "same_lineage",
                                    "same_base")} for r in hardest]),
              open(f"{OUT}/summary_hardest8.json", "w"), indent=1)
    print(f"[S6] named-pair ranks of 65 (by paper8): {ranks}", flush=True)
    print(f"     rank by method: {per_method_rank}", flush=True)

    import csv
    meta = {r["model"]: r for r in csv.DictReader(
        open("./results/D001/model_metadata.csv"))}
    fam = {}
    for r in r8:
        a, b = r["pair"].split(" | ")
        la, lb = meta[a]["lineage"], meta[b]["lineage"]
        tag = la if la == lb else f"{la}/{lb}"
        fam.setdefault(tag, []).append(r)
    roll = [dict(lineage=t, n_pairs=len(v),
                 paper8=round(float(np.mean([x["paper8"] for x in v])), 4),
                 coverage=round(float(np.mean([x["coverage"] for x in v])), 4),
                 joint_energy=round(float(np.mean([x["joint_energy"] for x in v])), 4))
            for t, v in sorted(fam.items(), key=lambda kv: -len(kv[1]))]
    json.dump(dict(schema="d015-family-rollup-v1", k=8, n_families=len(roll),
                   weighting="unweighted mean of member pairs within a family; "
                             "each family is one row regardless of size. A DISPLAY "
                             "choice, not a claim that hard_subset should weight "
                             "families equally (D015/S6).",
                   rows=roll), open(f"{OUT}/summary_family_rollup.json", "w"),
              indent=1)
    print(f"[S6] {len(roll)} lineage families; largest "
          f"{roll[0]['lineage']} ({roll[0]['n_pairs']} pairs)", flush=True)

    back = []
    for k in (8, 1):
        for r in tables[k]:
            for d in ("d_joint_minus_coverage", "d_coverage_minus_paper8"):
                if r[d]["delta"] < 0 and r[d]["resolved"]:
                    back.append(dict(k=k, pair=r["pair"], contrast=d, **r[d]))
    json.dump(dict(schema="d015-moved-backward-v1",
                   criterion="delta < 0 AND bootstrap CI excludes zero",
                   n_rows=len(back),
                   result="none" if not back else f"{len(back)} resolved-negative",
                   rows=back), open(f"{OUT}/summary_moved_backward.json", "w"),
              indent=1)
    print(f"[S6] moved-backward (resolved negative): "
          f"{len(back) if back else 'none'}", flush=True)
    if back:
        for r in back[:10]:
            print(f"     k={r['k']} {r['pair'][:56]:56s} {r['contrast'][2:20]:18s} "
                  f"{r['delta']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}]", flush=True)

    # unresolved-negative too, for the redistribution question (descriptive)
    softback = {k: sum(1 for r in tables[k]
                       if r["d_joint_minus_coverage"]["delta"] < 0) for k in (8, 1)}
    softfwd = {k: sum(1 for r in tables[k]
                      if r["d_joint_minus_coverage"]["delta"] > 0) for k in (8, 1)}
    print(f"\n[S1/S2] joint vs coverage, direction counts (unresolved included):")
    for k in (8, 1):
        print(f"   k={k}: {softfwd[k]} pairs up, {softback[k]} down, "
              f"{65-softfwd[k]-softback[k]} exactly flat")

    json.dump(dict(schema="d015-s0-v1", **s0,
                   direction_counts=dict(up=softfwd, down=softback),
                   wall_s=round(time.time() - t0, 1)),
              open(f"{OUT}/s0_index_check.json", "w"), indent=1)
    print(f"\nwall {time.time()-t0:.1f}s; written: {OUT}/", flush=True)


if __name__ == "__main__":
    main()
