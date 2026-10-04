"""D028 / S3 — mandatory hard-tail plot of the frozen v2 tensor (llmmap env).

Per pair: coverage of the whole pool = max over the 259 queries of
S_energy_sf_tok200_v2 (I4: MAX, never sum/mean). Distribution over the 3,570
pairs, sorted, with H_N and H_X marked; CVaR_0.1 / mean of that distribution
reported (the I1 failure check, METHOD 6.4). -> results/D028/tail_v2.png and
results/D028/tail_v2.json.

Usage:  python experiments/d028_plots.py
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "./results/D028"
meta = json.load(open(f"{OUT}/tensor_v2.json"))
S = np.load(f"{OUT}/S_energy_sf_tok200_v2.npy")
pairs = [p.replace("|", " | ") for p in meta["pairs"]]
hs = json.load(open(f"{OUT}/hard_sets_v2.json"))
HN, HX = set(hs["H_N"]), set(hs["H_X"])
cov = S.max(0)
order = np.argsort(cov)
m = max(1, int(np.floor(0.1 * len(cov))))
cvar = float(np.sort(cov)[:m].mean())
summary = dict(n_pairs=len(cov), mean=float(cov.mean()), cvar_0_1=cvar, cvar_over_mean=cvar / float(cov.mean()),
               H_N_rank_median=float(np.median([int(np.where(order == pairs.index(p))[0][0]) for p in HN])),
               H_X_rank_median=float(np.median([int(np.where(order == pairs.index(p))[0][0]) for p in HX])),
               H_all_in_bottom_decile=sum(int(np.where(order == pairs.index(p))[0][0]) < m for p in HN | HX))
json.dump(summary, open(f"{OUT}/tail_v2.json", "w"), indent=1)

fig, ax = plt.subplots(figsize=(11, 4))
x = np.arange(len(cov))
ax.plot(x, cov[order], color="0.6", lw=1, label="all pairs")
for name, H, col in (("H_N (structural)", HN, "C3"), ("H_X (non-structural)", HX, "C0")):
    xi = [int(np.where(order == pairs.index(p))[0][0]) for p in H]
    ax.scatter(xi, cov[[pairs.index(p) for p in H]], s=14, color=col, label=name, zorder=3)
ax.axvline(m, color="k", ls=":", lw=0.8)
ax.set_yscale("log")
ax.set_xlabel(f"pairs sorted by pool coverage (max over 259 queries), n={len(cov)}; dotted = bottom decile")
ax.set_ylabel("S_energy_sf_tok200_v2 coverage")
ax.set_title(f"D028 v2 tensor hard tail — CVaR0.1/mean = {summary['cvar_over_mean']:.3f}")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/tail_v2.png", dpi=130)
print(summary)
