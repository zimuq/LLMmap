"""D025 — POST HOC, descriptive (not pre-registered): every arm's accuracy on each
H pair minus GLOBAL@8, at k=8 and k=16, with the same bootstrap as d025_analysis
(linear config-only; attention hierarchical, S[:, k-1] slots). Written after S3
showed FAM:Phi beating SPEC:Phi on the Phi-3-medium pair. numpy only.
Usage:  python experiments/d025_posthoc.py
"""
import json
import itertools

import numpy as np

OUT = "./results/D025"
NCFG, NB, SEED = 25, 2000, 20260929


def boot_draws(ncfg, n_boot, seed):          # == d008_lib.boot_draws (asserted below)
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, ncfg, ncfg), minlength=ncfg) for _ in range(n_boot)])


def safe(arm):
    return arm.replace(":", "_").replace("|", "__").replace("/", "_")


sel = json.load(open(f"{OUT}/selection.json"))
hs = json.load(open(f"{OUT}/hard_set.json"))
models = json.load(open("./results/D008/selection.json"))["models"]
pairs = [f"{a} | {b}" for a, b in itertools.combinations(models, 2)]
alias = sel["fam_equals_spec"]
arms = list(sel["arms"])
C = {a: np.load(f"{OUT}/counts_{safe(alias.get(a, a))}.npz") for a in arms}
M = boot_draws(NCFG, NB, SEED)
tot = M.sum(1)
S = np.random.default_rng(SEED).integers(0, 5, (NB, 16, 5))


def acc(arm, r, k, i):
    if r == "lin":
        c = C[arm][f"lin|test|{k}|dir666"][i].sum(0).astype(float)
        return c.sum() / (2 * NCFG), (M @ c) / (2 * tot)
    c = np.stack([C[arm][f"att|{k}|{s}|dir666"][i].sum(0) for s in range(5)]).astype(float)
    per = (M @ c.T) / (2 * tot[:, None])
    return c.sum(1).mean() / (2 * NCFG), np.take_along_axis(per, S[:, k - 1], 1).mean(1)


out = {}
for p in hs["H"]:
    i = pairs.index(p)
    out[p] = {}
    for r in ("lin", "att"):
        g, gd = acc("GLOBAL", r, 8, i)
        row = {}
        for a in arms:
            if a == "GLOBAL":
                continue
            for k in (8, 16):
                v, d = acc(a, r, k, i)
                lo, hi = np.percentile(d - gd, [2.5, 97.5])
                row[f"{a}@{k}"] = dict(acc=round(float(v), 4), minus_global_k8=round(float(v - g), 4),
                                       ci=[round(float(lo), 4), round(float(hi), 4)])
        out[p][r] = dict(global_k8=round(float(g), 4), arms=row)

# the boot_draws re-implementation must match the one d025_analysis used
a = json.load(open(f"{OUT}/analysis.json"))
phi = [p for p in hs["H"] if "Phi-3-medium" in p][0]
sp = [x for x in arms if x.startswith("SPEC:Phi")][0]
assert out[phi]["lin"]["arms"][f"{sp}@8"]["ci"] == a["labels"]["lin"]["per_pair"][phi]["a"]["ci"]
assert out[phi]["att"]["arms"][f"{sp}@8"]["ci"] == a["labels"]["att"]["per_pair"][phi]["a"]["ci"]
json.dump(dict(schema="d025-posthoc-v1", label="POST HOC, descriptive; not pre-registered", rows=out),
          open(f"{OUT}/posthoc_all_arms_on_H.json", "w"), indent=1)
for r in ("lin", "att"):
    print(r, {k: (v["acc"], v["minus_global_k8"], v["ci"]) for k, v in out[phi][r]["arms"].items()
              if k.startswith(("FAM:Phi", "SPEC:Phi"))})
