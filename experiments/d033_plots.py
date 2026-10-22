"""D033 / S3 plots from results/D033/plotdata.npz (llmmap env, matplotlib).

tail_<view>.png  secondary (f), the mandatory hard-tail plot (I1): sorted all-pair two-candidate accuracy of
                 HELD-OUT targets (model-level folds, k = 8), per arm, readouts (1) and (2).
curves.png       secondary (c): O_all and O_tail vs k = 1-8 under (2), four arms, both views.
Usage:  .../envs/llmmap/bin/python experiments/d033_plots.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "./results/D033"
ARMS = ("OURS", "PAPER8", "FULL", "JG")
NAME = {"OURS": "OURS$^{fold}$", "PAPER8": "PAPER8", "FULL": "FULL (seen all)", "JG": "JG$^{fold}$"}
RO = {"enc": "(1) open-set encoder, Euclidean", "nc": "(2) nearest point cloud"}


def main():
    z = np.load(f"{RES}/plotdata.npz")
    for view in ("v2", "v1"):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for a, ro in zip(ax, ("enc", "nc")):
            for arm in ARMS:
                p = z[f"{view}|{ro}|{arm}|pair_acc_sorted"]
                n10 = max(1, int(np.floor(0.1 * len(p))))
                a.plot(np.arange(len(p)) / len(p), p, label=f"{NAME[arm]} (CVaR$_{{0.1}}$ {p[:n10].mean():.3f})")
            a.axvline(0.1, color="grey", lw=0.8, ls=":")
            a.set_title(f"{view}: {RO[ro]}")
            a.set_xlabel("pair quantile (sorted by two-candidate accuracy)")
            a.legend(fontsize=8)
        ax[0].set_ylabel("held-out pair accuracy")
        fig.tight_layout()
        fig.savefig(f"{RES}/tail_{view}.png", dpi=130)
        plt.close(fig)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for j, view in enumerate(("v2", "v1")):
        for i, (col, nm) in enumerate(((1, "O_all"), (2, "O_tail"))):
            a = ax[i, j]
            for arm in ARMS:
                d = z[f"{view}|curve|{arm}"]
                a.plot(d[:, 0], d[:, col], marker="o", ms=3, label=NAME[arm])
            a.set_title(f"{view}: {nm} vs k, readout (2), held-out targets")
            a.set_xlabel("k")
            a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{RES}/curves.png", dpi=130)
    print("wrote tail_{v2,v1}.png, curves.png")


if __name__ == "__main__":
    main()
