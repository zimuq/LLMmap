"""D027 — mandatory hard-tail plot of N' pair accuracies (llmmap env: numpy + matplotlib)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

a = json.load(open("./results/D027/analysis.json"))
t = a["pair_table"]                                 # sorted ascending by acc
x = range(len(t))
fig, ax = plt.subplots(figsize=(11, 4))
for i, r in zip(x, t):
    new = bool(r["candidate_members"])
    ax.plot([i, i], r["ci"], color="C3" if new else "C0", lw=0.8, alpha=0.5)
    ax.plot(i, r["acc"], "o", ms=3.5, color="C3" if new else "C0")
ax.axhline(0.90, color="grey", ls=":", lw=1)
ax.plot([], [], "o", color="C3", label="pair with >=1 new model")
ax.plot([], [], "o", color="C0", label="v1-only pair")
ax.set_xlabel(f"N' pairs, sorted by acc_B0' (n={len(t)})")
ax.set_ylabel("two-logit acc, S_val, 5-seed mean (95% config bootstrap)")
ax.set_title(f"D027 B0' (paper8, 85-way) — |H'|={a['n_H']} (<0.90), |H'_new|={a['n_H_new']}; label {a['label']}")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig("./results/D027/tail_nprime.png", dpi=130)
print("wrote results/D027/tail_nprime.png")
