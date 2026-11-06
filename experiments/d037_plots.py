"""D037 / S3a — plot inputs and the I1 hard-tail plot only. Computes NO contrast, interval or label (the D's S3 order:
the tail plot is viewed before S3b, d037_analysis.py, is submitted).

  data   (llmmap-gpu env, numpy only) writes results/D037/plotdata.npz:
           tail|<arm>          sorted all-pair two-logit accuracy at k = 8 (attention, 5-seed mean), every arm
           curve|<ro>|<arm>    85-way top-1 point estimate, k = 1..8, attention (5-seed mean) and linear
  draw   (llmmap env, which has matplotlib) draws from plotdata.npz:
           tail_k8.png         I1 plot: full curve and bottom decile, the four grid arms + reference rows + PAPER8
           curves.png          85-way top-1 vs k, attention and linear
Prints nothing about S_test beyond "written". Every panel is labelled "corpus v2".
Usage:  PYTHONPATH=.:experiments python experiments/d037_plots.py data
        /work/11280/zimuq1/vista/miniconda3/envs/llmmap/bin/python experiments/d037_plots.py draw
"""
import sys
import json

import numpy as np

OUT = "./results/D037"
ARMS8 = ("JG", "JG-mean", "GC", "GC-mean", "TOP-K[CVaR]", "TOP-K[MEAN]", "MMR-PP[CVaR]", "PAPER8")
STYLE = {"JG": ("#1f77b4", "-"), "JG-mean": ("#1f77b4", ":"), "GC": ("#d62728", "-"), "GC-mean": ("#d62728", ":"),
         "TOP-K[CVaR]": ("#2ca02c", "-."), "TOP-K[MEAN]": ("#2ca02c", (0, (1, 3))), "MMR-PP[CVaR]": ("#9467bd", "-."),
         "PAPER8": ("#444444", "--")}
KS = range(1, 9)


def data():
    import d037_lib as L
    from d031_analysis import Arm, Boot
    ch = json.load(open(f"{OUT}/chains.json"))
    rj = json.load(open(f"{OUT}/runs.json"))
    assert rj["chains_sha256"] == L.L31.sha(f"{OUT}/chains.json")
    D, B = L.Data(ch["mmr_reference"]), Boot("v2")
    np.savez_compressed(f"{OUT}/plotdata.npz",
                        **{f"tail|{a}": Arm(a, D.unit("att", a, 8)).sorted_curve() for a in ARMS8},
                        **{f"curve|{ro}|{a}": np.array([D.unit(ro, a, k).e_all(B)[1] for k in KS])
                           for ro in ("att", "lin") for a in ARMS8})
    print(f"[S3a] wrote {OUT}/plotdata.npz (tail plot + curve inputs; no contrast computed)", flush=True)


def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(f"{OUT}/plotdata.npz")
    n = len(z["tail|JG"])
    x = np.arange(1, n + 1) / n
    m = int(np.floor(0.1 * n))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, sl, ttl in ((ax[0], slice(None), f"all {n:,} pairs"), (ax[1], slice(0, m), f"bottom decile ({m} pairs)")):
        for arm in ARMS8:
            v = z[f"tail|{arm}"]
            c, ls = STYLE[arm]
            a.plot(x[sl], v[sl], lw=1.3, color=c, ls=ls, label=f"{arm}; CVaR0.1 {v[:m].mean():.3f}, mean {v.mean():.3f}")
        a.set(xlabel="pair quantile", ylabel="two-logit accuracy (both directions)", title=ttl)
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.suptitle("corpus v2 — attention (5-seed mean), k = 8, S_test")
    fig.tight_layout()
    fig.savefig(f"{OUT}/tail_k8.png", dpi=130)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for a, ro in zip(ax, ("att", "lin")):
        for arm in ARMS8:
            c, ls = STYLE[arm]
            a.plot(list(KS), z[f"curve|{ro}|{arm}"], marker="o", ms=3, lw=1.2, color=c, ls=ls, label=arm)
        a.set(xlabel="k", ylabel="85-way top-1 (S_test)",
              title=f"corpus v2 — {'attention (5-seed mean)' if ro == 'att' else 'linear (D017 concat)'}")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/curves.png", dpi=130)
    print(f"[S3a] drew {OUT}/tail_k8.png, {OUT}/curves.png", flush=True)


if __name__ == "__main__":
    {"data": data, "draw": draw}[sys.argv[1]]()
