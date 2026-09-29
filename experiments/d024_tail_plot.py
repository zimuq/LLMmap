"""D024 — mandatory hard-tail plot (CLAUDE.md): sorted 666-pair trained two-logit
accuracy at k=8, per arm (5-seed mean over the 25 test configs). Descriptive.
Usage:  PYTHONPATH=.:experiments python experiments/d024_tail_plot.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

c24 = np.load("./results/D024/run_counts.npz")
c23 = np.load("./results/D023/run_counts.npz")
src = {"SINGLE": (c24, "SINGLE"), "CENTROID": (c24, "CENTROID"),
       "CENTROID-raw": (c24, "CENTROID-raw"), "CLOUD": (c23, "JointGreedy"),
       "paper8": (c23, "paper8")}
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
summary = {}
for name, (f, key) in src.items():
    acc = np.stack([f[f"{key}|8|{r}|cnt_pair666"] for r in range(5)]).sum(2).mean(0) / 50.0
    s = np.sort(acc)
    summary[name] = dict(n_below_0p9=int((s < 0.9).sum()), n_below_0p8=int((s < 0.8).sum()),
                         worst10=[round(float(x), 3) for x in s[:10]])
    ax[0].plot(np.arange(1, 667), s, label=name)
    ax[1].plot(np.arange(1, 101), s[:100], label=name)
ax[0].set(xlabel="pair rank (worst first)", ylabel="trained 2-logit accuracy", title="all 666 pairs, k=8")
ax[1].set(xlabel="pair rank (worst first)", title="hard tail: worst 100")
ax[1].axvline(66, ls=":", c="gray"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig("./results/D024/tail_k8.png", dpi=120)
json.dump(summary, open("./results/D024/tail_k8_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
