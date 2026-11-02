"""D036 — figures (llmmap env, which has matplotlib; llmmap-gpu does not), from results/D036/plotdata.npz.
  tail_k8.png        I1 hard-tail plot, closed-set: sorted all-pair two-logit accuracy at k = 8 (attention, 5-seed
                     mean) for CLOUD-raw, CENTROID-raw, CLOUD, PAPER8; drawn before labels are read
  tail_openset.png   I1 hard-tail plot, open-set: sorted all-pair two-candidate accuracy of held-out S_test targets,
                     readout (2) raw, k = 8, for the three fold arms; drawn before labels are read
  curves.png         85-way top-1 vs k, attention and linear
  openset_curves.png held-out O_all and O_tail vs k, readout (2) raw
Every panel is labelled "corpus v2".
Usage:  python experiments/d036_plots.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "./results/D036"
COL = {"PAPER8": "#444444", "CLOUD": "#1f77b4", "CLOUD-raw": "#ff7f0e", "CENTROID-raw": "#2ca02c"}


def tail(z, prefix, arms, fname, title):
    n = len(z[f"{prefix}|{arms[0]}"])
    x = np.arange(1, n + 1) / n
    m = int(np.floor(0.1 * n))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, sl, ttl in ((ax[0], slice(None), f"all {n:,} pairs"), (ax[1], slice(0, m), f"bottom decile ({m} pairs)")):
        for arm in arms:
            v = z[f"{prefix}|{arm}"]
            a.plot(x[sl], v[sl], lw=1.3, color=COL[arm], ls="--" if arm == "PAPER8" else "-",
                   label=f"{arm}; CVaR0.1 {v[:m].mean():.3f}, mean {v.mean():.3f}")
        a.set(xlabel="pair quantile", ylabel="two-candidate accuracy (both directions)", title=f"{title} — {ttl}")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=130)


def main():
    z = np.load(f"{OUT}/plotdata.npz")
    closed = [a for a in COL if f"tail|{a}" in z]
    fold = [a for a in COL if f"otail|{a}" in z]
    tail(z, "tail", closed, "tail_k8.png", "corpus v2 — attention, k = 8")
    tail(z, "otail", fold, "tail_openset.png", "corpus v2 — open-set (2) raw, held-out, k = 8 (fold arms)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for a, ro in zip(ax, ("att", "lin")):
        for arm in closed:
            y = z[f"curve|{ro}|{arm}"]
            a.plot(range(1, len(y) + 1), y, marker="o", ms=3, lw=1.2, color=COL[arm],
                   ls="--" if arm == "PAPER8" else "-", label=arm)
        a.set(xlabel="k", ylabel="85-way top-1 (S_test)",
              title=f"corpus v2 — {'attention (5-seed mean)' if ro == 'att' else 'linear (D017 concat)'}")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/curves.png", dpi=130)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, e in zip(ax, ("all", "tail")):
        for arm in fold:
            y = z[f"ocurve|{e}|{arm}"]
            a.plot(range(1, len(y) + 1), y, marker="o", ms=3, lw=1.2, color=COL[arm], label=f"{arm}^fold")
        a.set(xlabel="k", ylabel=f"held-out O_{e}", title=f"corpus v2 — open-set (2) raw, O_{e}")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/openset_curves.png", dpi=130)


if __name__ == "__main__":
    main()
