"""D018 / P1-F2 — reuse check for the 65 structural pairs, before any comparison.

Recomputes the k=8 per-pair table for the three reused conditions (paper8,
GreedyCover = D009 `cvar_max`, JointGreedy = D010 `joint_energy`) from their
stored run counts with D015's own `pair_acc`, and asserts it EXACTLY equals
results/D015/per_pair_k8.json: same 65 pairs, same indices, same means, same
bootstrap CIs, same deltas. Also asserts the per-pair means average to the
`hard_subset` stored in D009/D010's metrics files, and that the reused chains
equal D008/D010's selections.

Read-only (I6). Zero GPU; runs on a login node in seconds.

Usage:  PYTHONPATH=.:experiments python experiments/d018_f2_check.py
"""
import json

import numpy as np

from d008_lib import near_relative_pairs, boot_draws
from d015_per_pair import counts_for, pair_acc, ci_of, delta, N_BOOT, NCFG

OUT = "./results/D018"
METH = {"paper8": ("D009", "paper8"),
        "coverage": ("D009", "cvar_max"),
        "joint_energy": ("D010", "joint_energy")}


def main():
    models = json.load(open("./results/D008/selection.json"))["models"]
    hard = near_relative_pairs(models)
    keys = sorted(hard)
    names = [" | ".join(hard[i]["pair"]) for i in keys]
    d15 = json.load(open("./results/D015/per_pair_k8.json"))
    rows = {r["pair"]: r for r in d15["rows"]}
    M = boot_draws(NCFG, N_BOOT)
    fails = []

    # --- pair set and index
    assert len(keys) == 65 == d15["n_pairs"] == len(rows)
    idx_ok = all(nm in rows and rows[nm]["index"] == int(k)
                 for k, nm in zip(keys, names))
    if not idx_ok:
        fails.append("pair set / index differs from D015")

    # --- per-pair numbers
    mean, draws = {}, {}
    for m, (where, pref) in METH.items():
        mean[m], draws[m] = pair_acc(counts_for(where, pref, 8), M)
    n_cells = 0
    for i, nm in enumerate(names):
        r = rows[nm]
        for m in METH:
            n_cells += 2
            if round(float(mean[m][i]), 4) != r[m]:
                fails.append(f"{nm} {m} mean {mean[m][i]:.4f} != {r[m]}")
            if ci_of(draws[m][i]) != r[m + "_ci"]:
                fails.append(f"{nm} {m} ci {ci_of(draws[m][i])} != {r[m + '_ci']}")
        for key, a, b in (("d_joint_minus_coverage", "joint_energy", "coverage"),
                          ("d_coverage_minus_paper8", "coverage", "paper8")):
            n_cells += 1
            d = delta(draws[a], draws[b], mean[a], mean[b], i)
            if d != r[key]:
                fails.append(f"{nm} {key} {d} != {r[key]}")
    agg = {m: round(float(mean[m].mean()), 4) for m in METH}
    if agg != d15["aggregate"]:
        fails.append(f"aggregate {agg} != {d15['aggregate']}")

    # --- per-pair means average to the stored hard_subset (D009/D010 metrics)
    d9m = json.load(open("./results/D009/metrics_by_condition_k.json"))["metrics"]
    d10m = json.load(open("./results/D010/metrics_by_k.json"))["metrics"]
    stored_hs = {"paper8": d9m["paper8"]["8"]["hard_subset"],
                 "coverage": d9m["cvar_max"]["8"]["hard_subset"],
                 "joint_energy": d10m["joint_energy"]["8"]["hard_subset"]}
    hs_diff = {m: round(abs(float(mean[m].mean()) - stored_hs[m]), 6) for m in METH}
    for m, dv in hs_diff.items():
        if dv > 1e-4:
            fails.append(f"hard_subset {m}: per-pair mean vs stored differs by {dv}")

    # --- reused chains (P1/F2)
    gc = json.load(open("./results/D008/selection.json"))["conditions"]["cvar_max"]["queries"][:8]
    jg = json.load(open("./results/D010/selection.json"))["joint_energy"]["queries"][:8]
    chain_ok = dict(GreedyCover=d9m["cvar_max"]["8"]["queries"] == gc,
                    JointGreedy=d10m["joint_energy"]["8"].get("queries", jg) == jg)
    for c, ok in chain_ok.items():
        if not ok:
            fails.append(f"{c} chain differs between selection and metrics file")

    verdict = "IDENTICAL" if not fails else "MISMATCH"
    out = dict(schema="d018-f2-check-v1", verdict=verdict, n_pairs=len(keys),
               pair_index_match=idx_ok, n_cells_compared=n_cells,
               comparison="exact equality at D015's stored precision (4 dp) of "
                          "means, 95% CIs and paired deltas; bootstrap seed "
                          "20260908, 2,000 draws, same as D015",
               aggregate_recomputed=agg, aggregate_d015=d15["aggregate"],
               hard_subset_stored=stored_hs, hard_subset_abs_diff=hs_diff,
               chains=dict(GreedyCover=gc, JointGreedy=jg), chain_match=chain_ok,
               failures=fails)
    json.dump(out, open(f"{OUT}/f2_reuse_check.json", "w"), indent=1)
    print(f"[F2] {verdict}: 65 pairs, {n_cells} cells vs D015; aggregate {agg}; "
          f"hard_subset diff {hs_diff}; chains {chain_ok}", flush=True)
    for f in fails[:20]:
        print("   ", f)
    assert not fails, f"{len(fails)} mismatches"


if __name__ == "__main__":
    main()
