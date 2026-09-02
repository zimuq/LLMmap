"""
D001 / P1 — S3..S6 analysis, incorporating the Review's amendments.

Runs off the .npy cached by d001_extract.py; cheap and re-runnable.

Amendments applied (docs/D001-Review-P1-v2.md):
  A1  primary verdict = Mann-Whitney related-vs-unrelated over ALL pairs with a
      model-level bootstrap CI, plus the M6 ratio. Decile enrichment is retained
      as a DESCRIPTIVE SECONDARY only.
  A2  new M6 = CVaR_0.1(separability) / mean(separability), all pairs and <=14B.
  A3  interpretation limit carried into the output (open-set nearest-template,
      NOT the paper's closed-set softmax).
  Judgment calls: Phi-3-medium@14B INCLUDED (37/9/6 primary, 35/11/6
      sensitivity); MoE by TOTAL params.
"""
import os
import csv
import json
import argparse
import itertools

import numpy as np
from scipy.stats import mannwhitneyu

from d001_model_metadata import MODELS, pair_labels, le_14b

RNG = np.random.default_rng(0)
GAMMA = 0.10  # CVaR tail (C1 default)


# ----------------------------------------------------------------- statistics
def auc_from_scores(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) + 0.5 P(a == b). a: scores of class-A traces, b: class-B."""
    gt = np.greater.outer(a, b).sum()
    eq = np.equal.outer(a, b).sum()
    return float((gt + 0.5 * eq) / (a.size * b.size))


def cvar_low(x: np.ndarray, gamma: float = GAMMA) -> float:
    """Mean of the worst (lowest) gamma-fraction of separability values."""
    k = max(1, int(np.floor(gamma * x.size)))
    return float(np.sort(x)[:k].mean())


def dominance(x: np.ndarray, y: np.ndarray) -> float:
    """P(x < y) + 0.5 P(x == y) -- 'how reliably x is harder than y'."""
    return auc_from_scores(y, x)  # symmetric reuse


# ----------------------------------------------------------------------- core
def build_pairs(D, y, labels):
    """Per-pair separability. Returns list of dict rows."""
    n = len(labels)
    idx = {c: np.where(y == c)[0] for c in range(n)}
    rows = []
    for A, B in itertools.combinations(range(n), 2):
        iA, iB = idx[A], idx[B]
        # Delta = d(trace, T_B) - d(trace, T_A); large positive => looks like A
        dA = D[iA, B] - D[iA, A]
        dB = D[iB, B] - D[iB, A]
        auc = auc_from_scores(dA, dB)
        # nearest-of-two balanced error
        err = 0.5 * (np.mean(dA < 0) + np.mean(dB > 0))
        rows.append(dict(a=labels[A], b=labels[B], i=A, j=B,
                         auc=auc, confusion_rate=float(err)))
    return rows


def annotate(rows, DB):
    for r in rows:
        r.update(pair_labels(r["a"], r["b"]))
        r["centroid_dist"] = float(np.linalg.norm(DB[r["i"]] - DB[r["j"]]))
    return rows


def summarize(rows, tag, out):
    sep = np.array([r["auc"] for r in rows])
    rel = np.array([r["structurally_related"] for r in rows], dtype=bool)
    near = np.array([r["near_relative"] for r in rows], dtype=bool)

    res = {
        "tag": tag,
        "n_pairs": int(sep.size),
        "n_structurally_related": int(rel.sum()),
        "n_near_relative": int(near.sum()),
        # M2
        "mean": float(sep.mean()),
        "p05": float(np.percentile(sep, 5)),
        "p10": float(np.percentile(sep, 10)),
        "p25": float(np.percentile(sep, 25)),
        "p50": float(np.percentile(sep, 50)),
        "min": float(sep.min()),
        "max": float(sep.max()),
        # M6 (A2) -- the headline premise number
        "cvar_0.1": cvar_low(sep),
        "M6_cvar_over_mean": cvar_low(sep) / float(sep.mean()),
    }

    # --- A1 primary test: related vs unrelated across the FULL distribution
    if rel.sum() >= 3 and (~rel).sum() >= 3:
        u = mannwhitneyu(sep[rel], sep[~rel], alternative="less")
        res["primary_mannwhitney_p"] = float(u.pvalue)
        res["primary_effect_dominance"] = dominance(sep[rel], sep[~rel])
        res["mean_sep_related"] = float(sep[rel].mean())
        res["mean_sep_unrelated"] = float(sep[~rel].mean())
    else:
        res["primary_mannwhitney_p"] = None
        res["primary_effect_dominance"] = None

    # --- decile enrichment: DESCRIPTIVE SECONDARY ONLY (A1)
    k = max(1, int(np.floor(0.10 * sep.size)))
    hard_idx = np.argsort(sep)[:k]
    hard = np.zeros(sep.size, dtype=bool)
    hard[hard_idx] = True
    for lab in ("same_base", "adjacent_version", "near_relative",
                "structurally_related", "same_org", "same_arch", "same_tokenizer"):
        v = np.array([r[lab] for r in rows], dtype=bool)
        base_rate = v.mean()
        in_decile = v[hard].mean()
        res[f"decile_{lab}_baserate"] = float(base_rate)
        res[f"decile_{lab}_observed"] = float(in_decile)
        res[f"decile_{lab}_enrichment"] = float(in_decile / base_rate) if base_rate > 0 else None

    out[tag] = res
    return res, hard


def model_bootstrap(rows, labels, B=2000):
    """Model-level bootstrap: resample models, recompute the primary statistics.

    Pairs share models and are not independent, so a pair-level bootstrap would
    understate the CI (P1 S4).
    """
    # Index by NAME, not by the full-52 index carried on each row -- otherwise a
    # subset (e.g. <=14B) resamples in subset space but looks up full-space keys.
    n = len(labels)
    pos = {m: i for i, m in enumerate(labels)}
    A = np.full((n, n), np.nan)
    R = np.zeros((n, n), dtype=bool)
    for r in rows:
        i, j = pos[r["a"]], pos[r["b"]]
        A[i, j] = A[j, i] = r["auc"]
        R[i, j] = R[j, i] = r["structurally_related"]

    ii, jj = np.triu_indices(n, 1)
    dom, ratio = [], []
    for _ in range(B):
        s = RNG.integers(0, n, size=n)
        # drop pairs where the same original model was drawn twice
        keep = s[ii] != s[jj]
        aucs = A[s[ii], s[jj]][keep]
        rels = R[s[ii], s[jj]][keep]
        if aucs.size < 20:
            continue
        ratio.append(cvar_low(aucs) / aucs.mean())
        if rels.sum() >= 3 and (~rels).sum() >= 3:
            dom.append(dominance(aucs[rels], aucs[~rels]))

    def ci(v):
        if len(v) < 50:
            return None
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    return {"dominance_ci95": ci(dom), "M6_ratio_ci95": ci(ratio),
            "n_boot": len(ratio)}


def permutation_enrichment(rows, n_perm=10000):
    """D001's original FALSIFIES criterion, kept as a secondary check."""
    sep = np.array([r["auc"] for r in rows])
    v = np.array([r["structurally_related"] for r in rows], dtype=bool)
    k = max(1, int(np.floor(0.10 * sep.size)))
    obs = v[np.argsort(sep)[:k]].mean()
    null = np.empty(n_perm)
    for t in range(n_perm):
        null[t] = v[RNG.permutation(sep.size)[:k]].mean()
    return {"observed_rate_in_decile": float(obs),
            "null_mean": float(null.mean()),
            "p_value": float((null >= obs).mean())}


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./results/D001")
    args = ap.parse_args()
    d = args.dir

    D = np.load(f"{d}/D.npy")
    y = np.load(f"{d}/y_test.npy")
    DB = np.load(f"{d}/DB.npy")
    labels = json.load(open(f"{d}/labels.json"))["label_order"]
    assert len(labels) == D.shape[1] == 52, "unexpected model count"
    assert set(labels) == set(MODELS), "metadata/label mismatch"

    rows = annotate(build_pairs(D, y, labels), DB)
    out = {}

    # ---- all pairs (M1/M2/M3/M6)
    res_all, hard_all = summarize(rows, "all_52", out)
    out["all_52"]["bootstrap"] = model_bootstrap(rows, labels)
    out["all_52"]["permutation_decile"] = permutation_enrichment(rows)

    # ---- M4: the motivating case
    sep = np.array([r["auc"] for r in rows])
    tgt = ("meta-llama/Meta-Llama-3-70B-Instruct", "abacusai/Smaug-Llama-3-70B-Instruct")
    for r in rows:
        if {r["a"], r["b"]} == set(tgt):
            out["M4_llama70b_vs_smaug"] = {
                "auc": r["auc"],
                "confusion_rate": r["confusion_rate"],
                "percentile_rank": float((sep < r["auc"]).mean() * 100),
                "centroid_dist": r["centroid_dist"],
                "note": ("Open-set nearest-template measure. NOT comparable to the "
                         "paper's 84% closed-set softmax number (Review A3)."),
            }

    # ---- M5: <=14B subset (primary 37/9/6; sensitivity 35/11/6)
    for boundary, tag in ((True, "le14b_inclusive"), (False, "le14b_strict")):
        keep = {m for m in labels if le_14b(m, boundary)}
        sub = [r for r in rows if r["a"] in keep and r["b"] in keep]
        res, _ = summarize(sub, tag, out)
        out[tag]["n_models"] = len(keep)
        if tag == "le14b_inclusive":
            out[tag]["bootstrap"] = model_bootstrap(sub, sorted(keep))
            # I1 check: >=3 near-relative pairs in the universe
            nr = [r for r in sub if r["near_relative"]]
            out[tag]["I1_near_relative_pairs"] = len(nr)
            out[tag]["I1_satisfied"] = len(nr) >= 3
            nr_sorted = sorted(nr, key=lambda r: r["auc"])
            out[tag]["I1_hardest_near_relative_pairs"] = [
                {"a": r["a"], "b": r["b"], "auc": r["auc"],
                 "percentile_rank_in_subset": float(
                     (np.array([q["auc"] for q in sub]) < r["auc"]).mean() * 100)}
                for r in nr_sorted[:10]
            ]

    # ---- correlation: centroid distance vs measured separability (evidences F1)
    cd = np.array([r["centroid_dist"] for r in rows])
    out["centroid_vs_separability_pearson"] = float(np.corrcoef(cd, sep)[0, 1])

    out["_interpretation_limits"] = [
        "Separability = AUC of Delta=d(trace,T_B)-d(trace,T_A) under the OPEN-SET "
        "nearest-template classifier. The paper's 84% for Meta-Llama-3-70B comes "
        "from the CLOSED-SET softmax classifier. Different systems; M4 is not a "
        "reproduction of that number. (Review A3)",
        "Hardness is measured under the paper's existing 8 queries, not hardness "
        "in principle (that is T2.4).",
        "This is LLMmap0.2, which the README states is not a one-to-one conversion "
        "of the paper's implementation.",
        "Templates are collapsed centroids (intra-model spread discarded); this "
        "metric is scoped to D001 and is NOT a decision on A3.",
    ]

    os.makedirs(d, exist_ok=True)
    with open(f"{d}/summary.json", "w") as fh:
        json.dump(out, fh, indent=2)

    cols = ["a", "b", "auc", "confusion_rate", "centroid_dist", "same_base",
            "adjacent_version", "near_relative", "structurally_related",
            "same_family", "same_org", "same_arch", "same_tokenizer",
            "relation_inferred"]
    with open(f"{d}/pairwise.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["auc"]):
            w.writerow(r)
    with open(f"{d}/hard_pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["auc"])[:len(rows) // 10]:
            w.writerow(r)
    keep = {m for m in labels if le_14b(m, True)}
    with open(f"{d}/subset_14b.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted([r for r in rows if r["a"] in keep and r["b"] in keep],
                        key=lambda r: r["auc"]):
            w.writerow(r)
    with open(f"{d}/model_metadata.csv", "w", newline="") as fh:
        mc = ["model", "org", "lineage", "variant", "base", "base_prov", "arch",
              "tok", "params_b", "active_b", "proprietary"]
        w = csv.DictWriter(fh, fieldnames=mc)
        w.writeheader()
        for m in labels:
            w.writerow({"model": m, **{k: MODELS[m].get(k) for k in mc[1:]}})

    # ---- M2 plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(sep, bins=40, color="#4C78A8", edgecolor="white")
        ax[0].axvline(res_all["cvar_0.1"], color="#E45756", ls="--",
                      label=f"CVaR$_{{0.1}}$={res_all['cvar_0.1']:.3f}")
        ax[0].axvline(res_all["mean"], color="#54A24B", ls="-",
                      label=f"mean={res_all['mean']:.3f}")
        ax[0].set_xlabel("pairwise separability (AUC)"); ax[0].set_ylabel("# pairs")
        ax[0].set_title(f"All {len(rows)} pairs — M6 ratio="
                        f"{res_all['M6_cvar_over_mean']:.3f}")
        ax[0].legend(fontsize=8)
        rel = np.array([r["structurally_related"] for r in rows], dtype=bool)
        ax[1].hist(sep[~rel], bins=30, alpha=.65, density=True, label="unrelated")
        ax[1].hist(sep[rel], bins=30, alpha=.65, density=True, label="structurally related")
        ax[1].set_xlabel("pairwise separability (AUC)"); ax[1].set_ylabel("density")
        ax[1].set_title("Primary test (A1): related vs unrelated")
        ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{d}/distribution.png", dpi=150)
        print("[S3] wrote distribution.png")
    except ImportError:
        print("[S3] matplotlib missing -- skipped distribution.png")

    print(json.dumps({k: v for k, v in out.items()
                      if k in ("all_52", "le14b_inclusive", "M4_llama70b_vs_smaug",
                               "centroid_vs_separability_pearson")}, indent=2)[:4000])


if __name__ == "__main__":
    main()
