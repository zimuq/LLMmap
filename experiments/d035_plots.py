"""D035 — figures (llmmap env, which has matplotlib; llmmap-gpu does not), from results/D035/plotdata.npz.
  tail_k8.png  I1 hard-tail plot: sorted all-pair two-logit accuracy at k = 8 (attention, 5-seed mean) for
               CLOUD, CENTROID-raw, CENTROID-norm, SINGLE, with PAPER8 as reference; drawn before labels are read
  curves.png   85-way top-1 vs k, attention and linear
Every panel is labelled "corpus v2".
Usage:  python experiments/d035_plots.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "./results/D035"
COL = {"PAPER8": "#444444", "CLOUD": "#1f77b4", "CENTROID|raw": "#2ca02c", "CENTROID|norm": "#9467bd",
       "SINGLE|raw": "#d62728", "SINGLE|norm": "#d62728"}


def main():
    z = np.load(f"{OUT}/plotdata.npz")
    arms = [a for a in COL if f"tail|{a}" in z]
    n = len(z["tail|CLOUD"])
    x = np.arange(1, n + 1) / n
    m = int(np.floor(0.1 * n))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, sl, ttl in ((ax[0], slice(None), f"all {n:,} pairs"), (ax[1], slice(0, m), f"bottom decile ({m} pairs)")):
        for arm in arms:
            v = z[f"tail|{arm}"]
            a.plot(x[sl], v[sl], lw=1.3, color=COL[arm], ls="--" if arm == "PAPER8" else "-",
                   label=f"{arm}; CVaR0.1 {v[:m].mean():.3f}, mean {v.mean():.3f}")
        a.set(xlabel="pair quantile", ylabel="two-logit accuracy (both directions)",
              title=f"corpus v2 — attention, k = 8 — {ttl}")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/tail_k8.png", dpi=130)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for a, ro in zip(ax, ("att", "lin")):
        for arm in arms:
            y = z[f"curve|{ro}|{arm}"]
            a.plot(range(1, len(y) + 1), y, marker="o", ms=3, lw=1.2, color=COL[arm],
                   ls="--" if arm == "PAPER8" else "-", label=arm)
        a.set(xlabel="k", ylabel="85-way top-1 (S_test)",
              title=f"corpus v2 — {'attention (5-seed mean)' if ro == 'att' else 'linear (D017 concat)'}")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/curves.png", dpi=130)


if __name__ == "__main__":
    main()
