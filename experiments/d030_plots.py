"""D030 / S3 (e) — the mandatory hard-tail plot (CLAUDE.md, I1): two-logit accuracy over all pairs,
sorted, for ours (attention, k = 8), PAPER8 and each method's B*. One panel per view.
Reads results/D030/tail_{v2,v1}.npz (d030_s3_analysis.py); writes results/D030/tail_{v2,v1}.png.
Usage:  python experiments/d030_plots.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = "./results/D030"
TITLE = {"v2": "corpus v2, 85-way: all 3,570 pairs", "v1": "corpus v1 view, 37-way: all 666 pairs"}


def main():
    for v in ("v2", "v1"):
        z = np.load(f"{RES}/tail_{v}.npz")
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for k in z.files:
            x = z[k]
            q = np.arange(1, len(x) + 1) / len(x)
            for a in ax:
                a.plot(q, x, label=k, lw=1.4)
        ax[1].set_xlim(0, 0.1)
        ax[1].set_title("worst 10% (CVaR0.1 region)")
        ax[0].set_title(TITLE[v])
        for a in ax:
            a.set_xlabel("pair quantile (sorted by two-logit accuracy)")
            a.set_ylabel("two-logit accuracy")
            a.axhline(0.5, color="grey", lw=0.6, ls=":")
            a.grid(alpha=0.3)
        ax[0].legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(f"{RES}/tail_{v}.png", dpi=130)
        print(f"wrote {RES}/tail_{v}.png")


if __name__ == "__main__":
    main()
