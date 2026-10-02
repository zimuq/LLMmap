"""D026 — curves.png from analysis.json (run under the `llmmap` env: numpy+matplotlib)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

a = json.load(open("./results/D026/analysis.json"))
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
style = {"POOL": "C0", "NEW": "C2", "POOL+NEW": "C1", "POOL+NEW+F3": "C3"}
for ax, ro in zip(axes, ("concat", "meanpool")):
    for name, col in style.items():
        c = a["sets"][ro][name]["curve"]
        k = list(range(1, len(c) + 1))
        ax.plot(k, [r["acc_test"] for r in c], "-o", color=col, ms=4,
                label=name + (" (deployment-dependent)" if name.endswith("F3") else ""),
                ls="--" if name.endswith("F3") else "-")
        ax.fill_between(k, [r["ci_test"][0] for r in c], [r["ci_test"][1] for r in c],
                        color=col, alpha=0.12)
    ax.axvline(4, color="grey", lw=0.8, ls=":")
    ax.axhline(0.95, color="grey", lw=0.8, ls=":")
    ax.set_title(f"{ro} (C={a['readouts'][ro]})  primary: {a['primary'][ro]['label']}")
    ax.set_xlabel("k (S_val-greedy, nested)")
axes[0].set_ylabel("two-way S_test accuracy (Phi-3-medium 128k vs 4k)")
axes[0].legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig("./results/D026/curves.png", dpi=130)
print("wrote results/D026/curves.png")
