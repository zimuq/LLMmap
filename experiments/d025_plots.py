"""D025 / S4 figures from results/D025/analysis.json (numpy + matplotlib only).
Run under the `llmmap` env (llmmap-gpu has no matplotlib; llmmap has no sklearn,
so the analysis itself runs under llmmap-gpu).
  curves.png        per H pair: GLOBAL / SPEC / FAM two-logit accuracy vs k, both readouts
  tail_k8_k16.png   mandatory hard-tail plot: GLOBAL worst-100 of 666 pairs, k=8 vs 16
Usage:  python experiments/d025_plots.py
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "./results/D025"
a = json.load(open(f"{OUT}/analysis.json"))
H = list(a["plot_arms"])
KS = list(range(1, 17))

fig, axs = plt.subplots(1, len(H), figsize=(4.2 * len(H), 3.8), sharey=True)
for ax, p in zip(np.atleast_1d(axs), H):
    for arm, col in zip(a["plot_arms"][p], ("k", "C3", "C0")):
        for r, ls in (("lin", "-"), ("att", "--")):
            ax.plot(KS, a["curves"][r][p][arm], ls, c=col, label=f"{arm.split(':')[0]} {r}")
    ax.axhline(0.95, c="gray", lw=0.6)
    ax.set_title(p.replace(" | ", "\n").replace("-Instruct", ""), fontsize=8)
    ax.set_xlabel("k")
np.atleast_1d(axs)[0].set_ylabel("two-logit accuracy (S_test)")
np.atleast_1d(axs)[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/curves.png", dpi=120)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
t = a["secondary_d_global"]["tail"]
for j, r in enumerate(("lin", "att")):
    for k in (8, 16):
        ax[j].plot(np.arange(1, 101), t[f"{r}|{k}"]["sorted_worst100"], label=f"GLOBAL k={k}")
    ax[j].axvline(66, ls=":", c="gray")
    ax[j].set(title=f"{'linear concat' if r == 'lin' else 'attention'}: worst 100 of 666 pairs",
              xlabel="pair rank (worst first)", ylabel="two-logit accuracy (S_test)")
    ax[j].legend()
fig.tight_layout(); fig.savefig(f"{OUT}/tail_k8_k16.png", dpi=120)
print("wrote curves.png, tail_k8_k16.png")
