"""D031 / S4 plots from results/D031/plotdata.npz (d031_analysis.py):
  tail_{v2,v1}.png  mandatory hard-tail plot (I1): sorted two-logit accuracy over ALL pairs, attention k = 8,
                    JG, PAPER8, GC and the starred variants (draw arms: mean of per-draw sorted curves);
                    right panel = worst 10% (the CVaR_0.1 region).
  curves.png        E_all and E_tail vs k, every arm, both readouts, per view.
  budget.png        target-side completions (log) vs E_all: MET cells at n = 8..250 and our JG curve.
Usage:  python experiments/d031_plots.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = "./results/D031"
TITLE = {"v2": "corpus v2, 85-way, all 3,570 pairs", "v1": "corpus v1 view, 37-way, all 666 pairs"}


def main():
    z = np.load(f"{RES}/plotdata.npz")
    for v in ("v2", "v1"):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for k in [k for k in z.files if k.startswith(f"tail|{v}|")]:
            x = z[k]
            q = np.arange(1, len(x) + 1) / len(x)
            lw = 2.2 if k.endswith("|JG") else 1.2
            for a in ax:
                a.plot(q, x, label=k.split("|", 2)[2], lw=lw)
        ax[1].set_xlim(0, 0.1)
        ax[1].set_title("worst 10% (CVaR0.1 region)")
        ax[0].set_title(f"{TITLE[v]} — attention, k = 8")
        for a in ax:
            a.set_xlabel("pair quantile (sorted by two-logit accuracy)")
            a.set_ylabel("two-logit accuracy")
            a.axhline(0.5, color="grey", lw=0.6, ls=":")
            a.grid(alpha=0.3)
        ax[0].legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(f"{RES}/tail_{v}.png", dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(2, 4, figsize=(20, 9))
    for i, v in enumerate(("v2", "v1")):
        for j, (ro, col, lab) in enumerate((("att", 1, "E_all"), ("att", 2, "E_tail"), ("lin", 1, "E_all"),
                                            ("lin", 2, "E_tail"))):
            a = ax[i, j]
            for k in [k for k in z.files if k.startswith(f"curve|{v}|{ro}|")]:
                d = z[k]
                nm = k.split("|", 3)[3]
                style = "o" if len(d) < 4 else "-"
                a.plot(d[:, 0], d[:, col], style, label=nm, lw=2.2 if nm == "JG" else 1.0, ms=4)
            a.set_title(f"{v} {ro} {lab}")
            a.set_xlabel("k")
            a.grid(alpha=0.3)
        ax[i, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(f"{RES}/curves.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, v in enumerate(("v2", "v1")):
        a = ax[i]
        for k in [k for k in z.files if k.startswith(f"budget|{v}|")]:
            d = z[k]
            nm = k.split("|", 2)[2]
            if nm == "ours":
                a.plot(d[:, 0], d[:, 1], "k-o", lw=2.2, ms=4, label="JG (attention), k queries")
            else:
                a.plot(d[:, 0], d[:, 1], "-s", ms=4, label=f"MET-{nm}, n completions")
        a.set_xscale("log")
        a.set_xlabel("target-side completions")
        a.set_ylabel("E_all (top-1)")
        a.set_title(TITLE[v].split(",")[0] + f", {'85' if v == 'v2' else '37'}-way")
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{RES}/budget.png", dpi=130)
    print("wrote tail_{v2,v1}.png, curves.png, budget.png")


if __name__ == "__main__":
    main()
