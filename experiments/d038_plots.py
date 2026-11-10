"""D038 / S3a + secondary (b) — plots only; no reading is computed here.

  tail      (S3a, the I1 check; viewed before S3b) from plotdata_<view>.npz -> tail_<view>.png: sorted single-query pool
            ceiling c_p and sorted JG k = 8 joint sf statistic, full S_build vs split 0's halves; full curve and
            bottom decile.
  optimism  (after S3b) from analysis.json -> optimism_<view>.png: per-step Y_joint optimism (median and 2.5-97.5 %
            band over the 40 directed splits) for JG, MMR*, TOP-K[CVaR], GC, PAPER8.
Usage:  /work/11280/zimuq1/vista/miniconda3/envs/llmmap/bin/python experiments/d038_plots.py tail|optimism
"""
import sys
import json

import numpy as np

OUT = "./results/D038"
VIEWS = ("v2", "v1")
STYLE = {"JG": ("#1f77b4", "-"), "MMR*": ("#9467bd", "-."), "TOP-K[CVaR]": ("#2ca02c", "-."), "GC": ("#d62728", "-"),
         "PAPER8": ("#444444", "--")}


def plt_():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def tail():
    plt = plt_()
    for view in VIEWS:
        z = np.load(f"{OUT}/plotdata_{view}.npz")
        n = len(z["cp_full"])
        x = np.arange(1, n + 1) / n
        m = int(np.floor(0.1 * n))
        rows = ((("cp_full", "full S_build (75 configs)", "#000000", "-"), ("cp_A0", "A_0 (37)", "#1f77b4", "--"),
                 ("cp_B0", "B_0 (38)", "#d62728", ":")),
                (("jg_full", "frozen JG chain on full S_build", "#000000", "-"),
                 ("jg_A0_on_A0", "JG chain_A0 on A_0 (selection)", "#1f77b4", "--"),
                 ("jg_A0_on_B0", "JG chain_A0 on B_0 (held out)", "#d62728", ":")))
        ylab = ("pool ceiling c_p = max over 259 queries of single-query sf", "joint sf statistic of the k = 8 set")
        fig, ax = plt.subplots(2, 2, figsize=(13, 9))
        for i, curves in enumerate(rows):
            for jx, (sl, ttl) in enumerate(((slice(None), f"all {n:,} pairs"), (slice(0, m), f"bottom decile ({m})"))):
                a = ax[i, jx]
                for key, lab, c, ls in curves:
                    v = z[key]
                    a.plot(x[sl], v[sl], color=c, ls=ls, lw=1.3,
                           label=f"{lab}; CVaR0.1 {v[:m].mean():.3f}, min {v[0]:.3f}, mean {v.mean():.3f}")
                a.set(xlabel="pair quantile", ylabel=ylab[i], title=ttl)
                a.grid(alpha=0.3)
            ax[i, 0].legend(fontsize=7)
        fig.suptitle(f"corpus {view} — S_build only: pair-distribution hard tail (I1)")
        fig.tight_layout()
        fig.savefig(f"{OUT}/tail_{view}.png", dpi=130)
        print(f"[S3a] wrote {OUT}/tail_{view}.png", flush=True)


def optimism():
    plt = plt_()
    an = json.load(open(f"{OUT}/analysis.json"))
    for view in VIEWS:
        cur = an["views"][view]["descriptive"]["per_step_optimism_yj"]
        fig, ax = plt.subplots(figsize=(7.5, 5))
        js = np.arange(1, 9)
        for rule, (c, ls) in STYLE.items():
            med = [s["median"] for s in cur[rule]]
            lo, hi = [s["band"][0] for s in cur[rule]], [s["band"][1] for s in cur[rule]]
            ax.plot(js, med, color=c, ls=ls, marker="o", ms=3, label=rule)
            ax.fill_between(js, lo, hi, color=c, alpha=0.12)
        ax.axhline(0, color="#888888", lw=0.8)
        ax.set(xlabel="step j", ylabel="gain_j(sel) − gain_j(meas) on Y_joint",
               title=f"corpus {view} — per-step optimism, median and 2.5–97.5 % band over 40 directed splits")
        ax.title.set_fontsize(9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{OUT}/optimism_{view}.png", dpi=130)
        print(f"[S3b] wrote {OUT}/optimism_{view}.png", flush=True)


if __name__ == "__main__":
    {"tail": tail, "optimism": optimism}[sys.argv[1]]()
