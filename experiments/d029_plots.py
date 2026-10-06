"""D029 — figures (llmmap env, which has matplotlib; llmmap-gpu does not).
  tail_chains_k8.png  from tail_k8.npz (S0): per-pair I4 coverage tail, k=8
  curves.png          from analysis.json (S2): E_all / E_hard vs k, both readouts
Every panel is labelled "corpus v2".
Usage:  python experiments/d029_plots.py [tail|curves]
"""
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "./results/D029"
COL = {"PAPER8": "#444444", "JG": "#1f77b4", "JG+ML": "#2ca02c", "GC": "#ff7f0e", "JG8_ML16": "#d62728"}


def tail():
    t = np.load(f"{OUT}/tail_k8.npz")
    n = len(t["all259"])
    x = np.arange(1, n + 1) / n
    m = int(np.floor(0.1 * n))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, sl, ttl in ((ax[0], slice(None), "all 3,570 pairs"), (ax[1], slice(0, m), f"bottom decile ({m} pairs)")):
        a.plot(x[sl], np.sort(t["all259"])[sl], "k--", lw=1, label="pool max, all 259 (D028)")
        for name in ("PAPER8", "JG", "GC", "JG+ML"):
            v = np.sort(t[name])
            a.plot(x[sl], v[sl], lw=1.3, color=COL[name],
                   label=f"{name} k=8 (pool part); CVaR0.1/mean {v[:m].mean() / v.mean():.3f}")
        a.set(xlabel="pair quantile", ylabel="I4 coverage, S_energy_sf_tok200_v2", title=f"corpus v2 — {ttl}")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/tail_chains_k8.png", dpi=130)


def curves():
    cur = json.load(open(f"{OUT}/analysis.json"))["secondary"]["c_curves"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for i, ro in enumerate(("att", "lin")):
        for j, e in enumerate(("E_all", "E_hard")):
            a = ax[i, j]
            for arm in COL:
                ks = sorted(int(key.split("|")[2]) for key in cur if key.startswith(f"{ro}|{arm}|"))
                y = np.array([cur[f"{ro}|{arm}|{k}"][e]["point"] for k in ks])
                lo = np.array([cur[f"{ro}|{arm}|{k}"][e]["ci"][0] for k in ks])
                hi = np.array([cur[f"{ro}|{arm}|{k}"][e]["ci"][1] for k in ks])
                a.errorbar(ks, y, yerr=[y - lo, hi - y], marker="o", ms=3, capsize=2, lw=1.2,
                           color=COL[arm], label=arm)
            a.set_title(f"corpus v2 — {'attention' if ro == 'att' else 'linear'} — "
                        f"{'E_all (85-way top-1)' if e == 'E_all' else 'E_hard (H_all, 57 pairs, two-way)'}",
                        fontsize=10)
            a.grid(alpha=0.3)
            a.set_xlabel("k")
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/curves.png", dpi=130)


if __name__ == "__main__":
    {"tail": tail, "curves": curves}[sys.argv[1]]()
