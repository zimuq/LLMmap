"""D032 / S3 plots from results/D032/plotdata.npz (d032_analysis.py):
  tail_{v2,v1}.png  mandatory hard-tail plot (I1): sorted two-logit accuracy over ALL pairs, attention k = 8,
                    every arm (starred arms + JG / PAPER8 / GC / RANDOM coloured, non-starred variants thin grey);
                    right panel = worst 10% (the CVaR_0.1 region).
  curves.png        E_all and E_tail vs k, both readouts, per view (starred arms + reference rows).
Usage:  python experiments/d032_plots.py   (needs matplotlib: the llmmap env)
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = "./results/D032"
TITLE = {"v2": "corpus v2, 85-way, all 3,570 pairs", "v1": "corpus v1 view, 37-way, all 666 pairs"}
REFS = ("JG", "PAPER8", "GC", "RANDOM")


def main():
    z = np.load(f"{RES}/plotdata.npz")
    vc = json.load(open(f"{RES}/variant_choice.json"))
    for v in ("v2", "v1"):
        stars = set(vc["views"][v]["stars"].values())
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for k in [k for k in z.files if k.startswith(f"tail|{v}|")]:
            nm = k.split("|", 2)[2]
            x = z[k]
            q = np.arange(1, len(x) + 1) / len(x)
            hi = nm in stars or nm in REFS
            for a in ax:
                if hi:
                    a.plot(q, x, label=nm, lw=2.2 if nm == "JG" else 1.3, ls="--" if "[MEAN]" in nm else "-")
                else:
                    a.plot(q, x, color="0.75", lw=0.6, zorder=0)
        ax[1].set_xlim(0, 0.1)
        ax[1].set_title("worst 10% (CVaR0.1 region)")
        ax[0].set_title(f"{TITLE[v]} — attention, k = 8 (grey: non-starred variants)")
        for a in ax:
            a.set_xlabel("pair quantile (sorted by two-logit accuracy)")
            a.set_ylabel("two-logit accuracy")
            a.axhline(0.5, color="grey", lw=0.6, ls=":")
            a.grid(alpha=0.3)
        ax[0].legend(fontsize=6.5, loc="lower right")
        fig.tight_layout()
        fig.savefig(f"{RES}/tail_{v}.png", dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(2, 4, figsize=(21, 9))
    for i, v in enumerate(("v2", "v1")):
        stars = set(vc["views"][v]["stars"].values())
        for j, (ro, col, lab) in enumerate((("att", 1, "E_all"), ("att", 2, "E_tail"), ("lin", 1, "E_all"),
                                            ("lin", 2, "E_tail"))):
            a = ax[i, j]
            for k in [k for k in z.files if k.startswith(f"curve|{v}|{ro}|")]:
                nm = k.split("|", 3)[3]
                if nm not in stars and nm not in REFS:
                    continue
                d = z[k]
                a.plot(d[:, 0], d[:, col], "o" if len(d) < 4 else "-", label=nm, ms=4,
                       lw=2.2 if nm == "JG" else 1.0, ls="--" if "[MEAN]" in nm and len(d) >= 4 else None)
            a.set_title(f"{v} {ro} {lab}")
            a.set_xlabel("k")
            a.grid(alpha=0.3)
        ax[i, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(f"{RES}/curves.png", dpi=110)
    print("wrote tail_{v2,v1}.png, curves.png")


if __name__ == "__main__":
    main()
