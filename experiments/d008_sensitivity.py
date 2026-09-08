"""D008 / S6 + S6b — is the answer an artifact of the statistic, or of one draw
of the build configs?

S6 (T2.3) — statistic sensitivity. `S_probe` is NOT one of the alternatives:
D007/R1 established it is unusable at this tensor's scale (51.8-98.2% of pairs
ceiling-pinned), and D008 says so twice. The alternative is RBF-MMD, a genuinely
different unbounded two-sample statistic.

  **Global bandwidth, not the per-cell median heuristic (P1/F2, approved).** The
  textbook median heuristic makes the kernel's scale cell-dependent, and I4
  aggregates with a MAX *across queries* -- the exact operation D007/F3 showed is
  invalid for a statistic whose scale varies by query. One sigma^2 for all
  172,494 cells. The per-cell variant is computed too, as a diagnostic only.

S6b (P1/F1, approved) — selection stability. D008's CONFIRMED criterion leans on
k=1-3, where the verdict rests on WHICH SINGLE QUERY greedy picks out of 259
noisy cells (D007/R8). Split the 75 build configs into the same two halves D007's
F2 used, rebuild `S_energy` on each, run both objectives on each, and report both
the chain overlap and each chain's test accuracy. Only the selection differs; the
classifier's reference set stays the full build split.

Usage:  PYTHONPATH=.:experiments python experiments/d008_sensitivity.py
"""
import os
import json
import time
import itertools

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from joblib import Parallel, delayed

from LLMmap.greedy_cover import greedy_cover
from d008_lib import (build_dq, summed, near_relative_pairs, condition_stats,
                      load_tensor, OUT)
from d007_build_tensor import prepare_cubes, SPLIT_SEED

JOBS = int(os.environ.get("D008_JOBS", "68"))
K = 8
BW_CELLS = 200
SMOKE = bool(os.environ.get("D008_SMOKE"))
if SMOKE:
    K, BW_CELLS = 2, 10


def _cube(cache_dir, m):
    return np.load(os.path.join(cache_dir, m.replace("/", "__") + ".npy"),
                   mmap_mode="r")


def global_bandwidth(cache_dir, models, nq, seed=20260908):
    """One sigma^2 for every cell: median pairwise squared distance over a large
    random sample of cells (F2)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(BW_CELLS):
        a, b = rng.choice(len(models), 2, replace=False)
        q = int(rng.integers(nq))
        A = np.asarray(_cube(cache_dir, models[a])[q], np.float32)
        B = np.asarray(_cube(cache_dir, models[b])[q], np.float32)
        vals.append(cdist(A, B, "sqeuclidean").ravel())
    return float(np.median(np.concatenate(vals)))


def mmd_worker(cache_dir, a, b, sigma2):
    A_ = _cube(cache_dir, a); B_ = _cube(cache_dir, b)
    nq = A_.shape[0]
    g = np.empty(nq, np.float32); loc = np.empty(nq, np.float32)
    for q in range(nq):
        A = np.asarray(A_[q], np.float32); B = np.asarray(B_[q], np.float32)
        dxx = cdist(A, A, "sqeuclidean"); dyy = cdist(B, B, "sqeuclidean")
        dxy = cdist(A, B, "sqeuclidean")
        g[q] = (np.exp(-dxx / sigma2).mean() + np.exp(-dyy / sigma2).mean()
                - 2 * np.exp(-dxy / sigma2).mean())
        s = np.median(np.concatenate([dxx.ravel(), dyy.ravel(), dxy.ravel()]))
        s = s if s > 0 else 1.0
        loc[q] = (np.exp(-dxx / s).mean() + np.exp(-dyy / s).mean()
                  - 2 * np.exp(-dxy / s).mean())
    return g, loc


def energy_subset_worker(cache_dir, a, b, idx):
    """Scale-free energy distance on a CONFIG SUBSET (S6b)."""
    A_ = _cube(cache_dir, a); B_ = _cube(cache_dir, b)
    nq = A_.shape[0]
    out = np.empty(nq, np.float32)
    for q in range(nq):
        A = np.asarray(A_[q][idx], np.float32); B = np.asarray(B_[q][idx], np.float32)
        xy = cdist(A, B).mean(); xx = cdist(A, A).mean(); yy = cdist(B, B).mean()
        raw = 2 * xy - xx - yy
        within = (xx + yy) / 2.0
        out[q] = raw / within if within > 0 else np.nan
    return out


def assemble(res, nq, npairs, which=0):
    S = np.empty((nq, npairs), np.float32)
    for j, r in enumerate(res):
        S[:, j] = r[which] if isinstance(r, tuple) else r
    return S


def main():
    os.makedirs(OUT, exist_ok=True)
    S_e, meta = load_tensor()
    models = meta["models"]
    pairs = list(itertools.combinations(models, 2))
    if SMOKE:
        pairs = pairs[:20]
    hard = near_relative_pairs(models)
    met = json.load(open(f"{OUT}/metrics_by_condition_k.json"))
    sel = json.load(open(f"{OUT}/selection.json"))
    use_norm = met["distances_normalised"]

    _, cache_dir = prepare_cubes(200)
    nq = S_e.shape[0]

    # ---------------- S6: MMD tensor, global bandwidth
    sigma2 = global_bandwidth(cache_dir, models, nq)
    print(f"[S6] global bandwidth sigma^2 = {sigma2:.2f} "
          f"(median over {BW_CELLS} sampled cells)", flush=True)
    t = time.time()
    res = Parallel(n_jobs=JOBS, verbose=1, batch_size=1)(
        delayed(mmd_worker)(cache_dir, a, b, sigma2) for a, b in pairs)
    S_m = assemble(res, nq, len(pairs), 0)
    S_mloc = assemble(res, nq, len(pairs), 1)
    S_e = S_e[:, :len(pairs)] if SMOKE else S_e
    print(f"[S6] MMD tensor built in {(time.time()-t)/60:.1f} min", flush=True)
    np.save(f"{OUT}/S_mmd_global_tok200.npy", S_m)

    # resolution audit -- the same discipline D004/D007 made mandatory. A kernel
    # statistic with a global bandwidth can saturate toward 0 or 2 instead of a
    # probe's ceiling, and that would disqualify it as a check just as surely.
    audit = dict(
        sigma2=sigma2,
        cells_min=float(S_m.min()), cells_max=float(S_m.max()),
        distinct=int(len(np.unique(np.round(S_m, 5)))),
        frac_below_1e3=float((S_m < 1e-3).mean()),
        frac_above_1p9=float((S_m > 1.9).mean()),
        energy_distinct=int(len(np.unique(np.round(S_e, 4)))))
    print(f"[S6] MMD audit: range [{audit['cells_min']:.4f}, "
          f"{audit['cells_max']:.4f}], {audit['distinct']:,} distinct "
          f"(energy: {audit['energy_distinct']:,}), "
          f"{audit['frac_below_1e3']:.1%} below 1e-3", flush=True)

    rho = spearmanr(S_e.ravel(), S_m.ravel()).statistic
    rho_loc = spearmanr(S_m.ravel(), S_mloc.ravel()).statistic
    cov_rho = spearmanr(S_e.max(0), S_m.max(0)).statistic

    # chains under MMD, and what they score
    Dq, w2, y_ev, y_rf, _, cfg_ev = build_dq(eval_pool="test")
    chains = {}
    for name, S, g in (("energy_cvar", S_e, 0.1), ("energy_mean", S_e, 1.0),
                       ("mmd_cvar", S_m, 0.1), ("mmd_mean", S_m, 1.0)):
        chains[name] = greedy_cover(S, K, gamma=g, agg="max")
    acc = {}
    for name, ch in chains.items():
        D = summed(Dq, w2, ch[:K], normalise=use_norm)
        p, _ = condition_stats(D, y_ev, y_rf, cfg_ev, hard, len(models))
        acc[name] = {m: p[m] for m in ("mean_top1", "worst_class",
                                       "worst3_class", "hard_subset")}
        print(f"[S6] {name:12s} {ch}  mean {p['mean_top1']:.4f} "
              f"worst {p['worst_class']:.4f} hard {p['hard_subset']:.4f}",
              flush=True)

    s6 = dict(
        audit=audit,
        spearman_cells_energy_vs_mmd=float(rho),
        spearman_cells_mmd_global_vs_local_bandwidth=float(rho_loc),
        spearman_pair_coverage_energy_vs_mmd=float(cov_rho),
        chains={k: v for k, v in chains.items()},
        chain_overlap=dict(
            cvar_k8=len(set(chains["energy_cvar"]) & set(chains["mmd_cvar"])),
            mean_k8=len(set(chains["energy_mean"]) & set(chains["mmd_mean"]))),
        accuracy=acc,
        ranking_preserved=bool(
            (acc["energy_cvar"]["worst_class"] >= acc["energy_mean"]["worst_class"])
            == (acc["mmd_cvar"]["worst_class"] >= acc["mmd_mean"]["worst_class"])),
        falsification="if the selected sets or the relative accuracy ranking "
                      "change materially under MMD, S3-S5 are partly an artifact "
                      "of the statistic and must be reported as such (D008/S6)")

    # ---------------- S6b: selection stability across build-config halves
    rng = np.random.default_rng(SPLIT_SEED)          # D007's F2 halves, reused
    perm = rng.permutation(75)
    halves = dict(A=np.sort(perm[:37]), B=np.sort(perm[37:74]))
    stab = dict(split_seed=SPLIT_SEED,
                halves={k: v.tolist() for k, v in halves.items()}, chains={},
                accuracy={})
    for hname, idx in halves.items():
        t = time.time()
        r = Parallel(n_jobs=JOBS, verbose=0, batch_size=1)(
            delayed(energy_subset_worker)(cache_dir, a, b, idx) for a, b in pairs)
        S_h = assemble(r, nq, len(pairs))
        np.save(f"{OUT}/S_energy_half{hname}_tok200.npy", S_h)
        for g, lab in ((0.1, "cvar"), (1.0, "mean")):
            ch = greedy_cover(S_h, K, gamma=g, agg="max")
            stab["chains"][f"{lab}_half{hname}"] = ch
            D = summed(Dq, w2, ch, normalise=use_norm)
            p, _ = condition_stats(D, y_ev, y_rf, cfg_ev, hard, len(models))
            stab["accuracy"][f"{lab}_half{hname}"] = {
                m: p[m] for m in ("mean_top1", "worst_class", "hard_subset")}
        print(f"[S6b] half {hname} done in {(time.time()-t)/60:.1f} min: "
              f"cvar {stab['chains']['cvar_half'+hname]}", flush=True)

    for lab in ("cvar", "mean"):
        a, b = stab["chains"][f"{lab}_halfA"], stab["chains"][f"{lab}_halfB"]
        stab[f"{lab}_overlap_k8"] = len(set(a) & set(b))
        stab[f"{lab}_first_query_same"] = bool(a[0] == b[0])
        stab[f"{lab}_overlap_k3"] = len(set(a[:3]) & set(b[:3]))
    stab["full_build_chains"] = {
        "cvar": sel["conditions"]["cvar_max"]["queries"][:K],
        "mean": sel["conditions"]["mean_greedy_max"]["queries"][:K]}
    stab["note"] = ("only SELECTION uses a half; the classifier's reference set "
                    "is always the full build split, so this isolates selection "
                    "variance from reference-set size")

    json.dump(dict(schema="d008-sensitivity-v1", S6=s6, S6b=stab),
              open(f"{OUT}/statistic_sensitivity.json", "w"), indent=1)
    print(f"\n[S6b] chain overlap across halves: cvar "
          f"{stab['cvar_overlap_k8']}/8 (k=3: {stab['cvar_overlap_k3']}/3, "
          f"same first query: {stab['cvar_first_query_same']}), mean "
          f"{stab['mean_overlap_k8']}/8")
    print(f"written: {OUT}/statistic_sensitivity.json")


if __name__ == "__main__":
    main()
