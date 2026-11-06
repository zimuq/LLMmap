"""D036 R2 addendum (Review D036/R2, closing): descriptive open-set CLOUD^fold - CENTROID-raw^fold, readout (2) raw,
k = 8 and per k = 1..8 (Delta_all, Delta_tail with intervals), no label. Same draws and stored counts as O1 / O2:
d033_analysis.Boot("v2"); CLOUD^fold = D033 counts_v2.npz nc|model|JG|raw|<k>, CENTROID-raw^fold = D036
openset_counts.npz. Consistency asserts, every k: levels equal
D036 analysis.json's O2.b / O1.b exactly; Delta equals O1 - O2 to 1.5e-4 (three independent 4-dp roundings).
Writes results/D036/addendum.json.
Usage:  OMP_NUM_THREADS=8 PYTHONPATH=.:experiments python experiments/d036_addendum.py
"""
import json

import numpy as np

import d031_lib as L31
import d033_analysis as OA

OUT = "./results/D036"


def main():
    BO = OA.Boot("v2")
    oc = np.load(f"{OUT}/openset_counts.npz")
    c33 = np.load("./results/D033/counts_v2.npz")
    cloud = lambda k: OA.Unit(c33[f"nc|model|JG|raw|{k}|cnt_model"][None], c33[f"nc|model|JG|raw|{k}|cnt_pair_all"][None])
    cen = lambda k: OA.Unit(oc[f"nc|CENTROID-raw|{k}|cnt_model"][None], oc[f"nc|CENTROID-raw|{k}|cnt_pair_all"][None])
    rows = {}
    for k in range(1, 9):
        c = OA.contrast(cloud(k), cen(k), BO)
        c.pop("label")                                                  # descriptive: no label (Review)
        rows[k] = c
    a = json.load(open(f"{OUT}/analysis.json"))["open_set"]["contrasts"]
    for k in range(1, 9):                                               # levels: same r4 of the same point, exact
        for e in ("all", "tail"):
            r, o1, o2 = rows[k][f"d_{e}"], a["O1"]["per_k"][str(k)][f"d_{e}"], a["O2"]["per_k"][str(k)][f"d_{e}"]
            assert (r["a"], r["b"]) == (o2["b"], o1["b"]), (k, e, r, o1, o2)
            # each delta is rounded to 4 dp on its own: |r4(x - y) - (r4(x - z) - r4(y - z))| <= 1.5e-4
            assert abs(r["delta"] - (o1["delta"] - o2["delta"])) <= 1.5e-4 + 1e-12, (k, e, r, o1, o2)
    L31.jdump(dict(schema="d036-addendum-v1", contrast="CLOUD^fold - CENTROID-raw^fold, readout (2) raw",
                   draws="d033_analysis.Boot('v2'), configs only, folds fixed (= O1 / O2)", label=None,
                   k8=rows[8], per_k=rows, code_commit=L31.git_head()), f"{OUT}/addendum.json")
    print(f"[D036 addendum] wrote {OUT}/addendum.json", flush=True)


if __name__ == "__main__":
    main()
