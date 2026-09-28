"""D021 / S0 — inventory, incumbent footing, the delta check, and two target-free pilots.

Run BEFORE P1 is reviewed, as D018's S0 cost pilot was: everything here either
touches no target at all, or touches the target only through the two INCUMBENT
predictors (MAX, JOINT-E), never through ADD / JOINT-G. The uncertainty method
was fixed and committed (docs/plans/D021-P1.md, commit c54e4d1) before this ran.

  (1) target inventory: ties at y = 1.0 per chain, the non-ceiling subset (min
      over chains < 0.99), pair-index alignment D016 <-> D007 tensor
  (2) MAX and JOINT-E for the three k=8 chains at k=1..8 -- asserted against the
      objectives D008 (MAX) and D010 (JOINT-E) stored for their own chains
  (3) LOMO-jackknife half-widths of pooled rho(JOINT-E) and rho(MAX) -- D021's
      delta check (stop for a Call if JOINT-E's exceeds 0.05)
  (4) Sigma_q cross-fit pilot on 8 queries: M^2 with Sigma from full S_build vs
      Sigma fit per half. NO target involved.
  (5) cost: per-query time for PCA + Ledoit-Wolf + 666-pair M^2

Read-only. Writes results/D021/s0_inventory.json only.
Usage:  PYTHONPATH=.:experiments python experiments/d021_s0.py
"""
import os
import json
import time

import numpy as np
from scipy.stats import spearmanr, t as tdist
from sklearn.covariance import ledoit_wolf

from LLMmap.joint_statistic import cvar

OUT = "./results/D021"
EMB = "./data/corpus_v1/embeddings"
CHAINS = ("paper8", "coverage", "joint_energy")
DELTA, D_PCA = 0.05, 64


def load_meta():
    m = json.load(open("./results/D007/tensor_tok200.meta.json"))
    return m["models"], m["pairs"], m["split_a"], m["split_b"]


def load_build(models, queries):
    """{model: (len(queries), 75, 1024)} S_build, tok200, unnormalised -- same
    arrays load_corpus('build') returns, restricted to `queries`."""
    X = {}
    for m in models:
        slug = m.replace("/", "__")
        a = np.load(f"{EMB}/{slug}.npy", mmap_mode="r")
        rows = json.load(open(f"{EMB}/{slug}.index.json"))["rows"]
        pos = {(r["query_index"], r["config"]): i for i, r in enumerate(rows)
               if r["pool"] == "build"}
        cfgs = sorted({c for (_, c) in pos})
        assert len(cfgs) == 75
        X[m] = np.stack([np.stack([a[pos[(q, c)]] for c in cfgs]) for q in queries]
                        ).astype(np.float32)
    return X


def chains_k8():
    d9 = json.load(open("./results/D009/runs.json"))["runs"]
    d10 = json.load(open("./results/D010/metrics_by_k.json"))["runs"]
    g = lambda runs, cond: next(r for r in runs if r["k"] == 8 and not r["error"]
                                and r["condition"] == cond)["queries"]
    return {"paper8": g(d9, "paper8"), "coverage": g(d9, "cvar_max"),
            "joint_energy": g(d10, "joint_energy")}


def jackknife(stat_fn, ia, ib, n_models=37):
    """LOMO jackknife as pre-registered: returns (T_hat, half_width, lo, hi)."""
    full = stat_fn(np.ones(len(ia), bool))
    reps = np.array([stat_fn((ia != m) & (ib != m)) for m in range(n_models)])
    se = np.sqrt((n_models - 1) / n_models * ((reps - reps.mean()) ** 2).sum())
    hw = float(tdist.ppf(0.975, n_models - 1) * se)
    return float(full), hw, float(full - hw), float(full + hw)


def m2_query(Xq, ia, ib, sa, sb, d=D_PCA, sigma="full"):
    """Cross-fitted Mahalanobis^2 for one query, all pairs. Xq: (37, 75, 1024)."""
    n_m, n_c, dim = Xq.shape
    flat = Xq.reshape(-1, dim).astype(np.float64)
    mu = flat.mean(0)
    _, _, vt = np.linalg.svd(flat - mu, full_matrices=False)
    Z = ((flat - mu) @ vt[:d].T).reshape(n_m, n_c, d)
    if sigma == "full":
        R = (Z - Z.mean(1, keepdims=True)).reshape(-1, d)
        S, shrink = ledoit_wolf(R, assume_centered=True)
    else:   # per half: pooled within-model covariance fit inside each half
        Ss, sh = [], []
        for idx in (sa, sb):
            Zh = Z[:, idx]
            s_, k_ = ledoit_wolf((Zh - Zh.mean(1, keepdims=True)).reshape(-1, d),
                                 assume_centered=True)
            Ss.append(s_); sh.append(k_)
        S, shrink = (Ss[0] + Ss[1]) / 2, float(np.mean(sh))
    P = np.linalg.inv(S)
    mA, mB = Z[:, sa].mean(1), Z[:, sb].mean(1)
    dA, dB = mA[ia] - mA[ib], mB[ia] - mB[ib]
    return np.einsum("pi,ij,pj->p", dA, P, dB), float(shrink)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    models, pairs, sa, sb = load_meta()
    ia = np.array([models.index(p.split("|")[0]) for p in pairs])
    ib = np.array([models.index(p.split("|")[1]) for p in pairs])
    out = dict(schema="d021-s0-v1",
               uncertainty_method="LOMO jackknife, 37 folds, t_36 -- fixed in "
                                  "commit c54e4d1 before this script ran")

    # ---------------- (1) target inventory
    d16 = sorted(json.load(open("./results/D016/full_pair_trained_accuracy.json"))
                 ["rows"], key=lambda r: r["index"])
    assert [r["pair"].replace(" | ", "|") for r in d16] == pairs
    y = {c: np.array([r[c] for r in d16]) for c in CHAINS}
    ymin = np.min(np.stack([y[c] for c in CHAINS]), axis=0)
    nonceil = ymin < 0.99
    out["target"] = dict(
        n_pairs=len(pairs), split_a=len(sa), split_b=len(sb),
        configs_in_neither_half=sorted(set(range(75)) - set(sa) - set(sb)),
        ties_at_1={c: int((y[c] == 1.0).sum()) for c in CHAINS},
        n_distinct_values={c: int(len(np.unique(y[c]))) for c in CHAINS},
        n_nonceiling_min_lt_0_99=int(nonceil.sum()),
        n_nonceiling_structural=int(sum(1 for r, n in zip(d16, nonceil)
                                        if n and r["structural"])))
    print(f"[S0-1] {out['target']}", flush=True)

    # ---------------- (2) incumbents, asserted against stored objectives
    chains = chains_k8()
    S = np.load("./results/D007/S_energy_sf_tok200.npy").astype(np.float64)  # (259, 666), stored float32
    allq = sorted({q for c in chains.values() for q in c})
    t1 = time.time()
    X = load_build(models, allq)
    t_load = time.time() - t1
    qpos = {q: i for i, q in enumerate(allq)}
    pred = {"MAX": {}, "JOINT-E": {}}
    checks = {}
    for c, Q in chains.items():
        pred["MAX"][c], pred["JOINT-E"][c] = {}, {}
        acc_xy = np.zeros((len(pairs), 75, 75), np.float64)
        acc_xx = np.zeros((len(models), 75, 75), np.float64)
        for k in range(1, 9):
            q = Q[k - 1]
            pred["MAX"][c][k] = S[Q[:k]].max(0)
            V = np.stack([X[m][qpos[q]] for m in models]).astype(np.float64)
            nrm = (V ** 2).sum(-1)
            G = np.einsum("mid,njd->mnij", V, V)               # (37,37,75,75)
            D2 = np.maximum(nrm[:, None, :, None] + nrm[None, :, None, :] - 2 * G, 0)
            acc_xx += D2[np.arange(len(models)), np.arange(len(models))]
            acc_xy += D2[ia, ib]
            xy = np.sqrt(acc_xy).mean((1, 2))
            xm = np.sqrt(acc_xx).mean((1, 2))
            pred["JOINT-E"][c][k] = (2 * xy - xm[ia] - xm[ib]) / ((xm[ia] + xm[ib]) / 2)
        checks[c] = dict(queries=Q)
    # D008 stored MAX for coverage's own chain; D010 stored JOINT-E for joint_energy's
    tr8 = json.load(open("./results/D008/selection.json"))["conditions"]["cvar_max"]["trace"]
    tr10 = json.load(open("./results/D010/selection.json"))["joint_energy"]["trace"]
    dmax = max(abs(cvar(pred["MAX"]["coverage"][t["k"]], 0.1) - t["cvar_gamma"])
               + abs(pred["MAX"]["coverage"][t["k"]].mean() - t["mean_cov"])
               for t in tr8 if t["k"] <= 8)
    dje = max(abs(cvar(pred["JOINT-E"]["joint_energy"][t["k"]], 0.1) - t["objective_cvar"])
              + abs(pred["JOINT-E"]["joint_energy"][t["k"]].mean() - t["mean_stat"])
              for t in tr10 if t["k"] <= 8)
    out["incumbent_reproduction"] = dict(
        MAX_vs_D008_trace_max_abs=dmax, JOINT_E_vs_D010_trace_max_abs=dje,
        note="D010 accumulated in float32, this in float64: agreement to ~1e-5 "
             "is the expected ceiling")
    print(f"[S0-2] MAX vs D008 trace {dmax:.2e}; JOINT-E vs D010 trace {dje:.2e}",
          flush=True)
    assert dmax < 1e-6 and dje < 1e-4   # float32 storage (MAX) / float32 accumulation (D010)

    # ---------------- (3) the delta check (incumbents only)
    def pooled(name, mask):
        return np.mean([spearmanr(pred[name][c][8][mask], y[c][mask])[0]
                        for c in CHAINS])
    dcheck = {}
    for name in ("JOINT-E", "MAX"):
        for sub, base in (("all_666", np.ones(len(pairs), bool)), ("nonceiling", nonceil)):
            T, hw, lo, hi = jackknife(lambda msk: pooled(name, msk & base), ia, ib)
            per = {c: float(spearmanr(pred[name][c][8][base], y[c][base])[0])
                   for c in CHAINS}
            dcheck[f"{name}|{sub}"] = dict(pooled_rho=round(T, 4), half_width=round(hw, 4),
                                           lo=round(lo, 4), hi=round(hi, 4),
                                           per_chain=per)
    # paired incumbent difference: the resolution a PAIRED delta-rho actually has
    # (the gate is judged on paired delta-rho). Incumbents only -- no ADD.
    for sub, base in (("all_666", np.ones(len(pairs), bool)), ("nonceiling", nonceil)):
        T, hw, lo, hi = jackknife(lambda msk: pooled("JOINT-E", msk & base)
                                  - pooled("MAX", msk & base), ia, ib)
        dcheck[f"JOINT-E_minus_MAX|{sub}"] = dict(pooled_delta_rho=round(T, 4),
                                                  half_width=round(hw, 4),
                                                  lo=round(lo, 4), hi=round(hi, 4))
    out["delta_check"] = dict(delta=DELTA, results=dcheck,
                              joint_e_halfwidth_exceeds_delta=bool(
                                  dcheck["JOINT-E|all_666"]["half_width"] > DELTA))
    for k_, v in dcheck.items():
        print(f"[S0-3] {k_:28s} {v.get('pooled_rho', v.get('pooled_delta_rho')):+.4f} +/- {v['half_width']:.4f} "
              f"per-chain {({c: round(r, 3) for c, r in v.get('per_chain', {}).items()})}",
              flush=True)

    # ---------------- (4) Sigma cross-fit pilot, NO target
    pilot, t_q = {}, []
    for q in chains["joint_energy"]:
        Xq = np.stack([X[m][qpos[q]] for m in models])
        t2 = time.time()
        a, sh_a = m2_query(Xq, ia, ib, sa, sb, sigma="full")
        t_q.append(time.time() - t2)
        b, sh_b = m2_query(Xq, ia, ib, sa, sb, sigma="half")
        rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-9)
        pilot[int(q)] = dict(spearman_full_vs_half=float(spearmanr(a, b)[0]),
                             median_rel_diff=float(np.median(rel)),
                             p95_rel_diff=float(np.percentile(rel, 95)),
                             mean_ratio_half_over_full=float(np.mean(b) / np.mean(a)),
                             n_negative_full=int((a < 0).sum()),
                             n_negative_half=int((b < 0).sum()),
                             shrinkage_full=sh_a, shrinkage_half=sh_b,
                             m2_full_median=float(np.median(a)))
        print(f"[S0-4] q={q}: rho(full,half) {pilot[int(q)]['spearman_full_vs_half']:.5f} "
              f"median rel diff {pilot[int(q)]['median_rel_diff']:.4f} "
              f"neg {pilot[int(q)]['n_negative_full']}/{pilot[int(q)]['n_negative_half']}",
              flush=True)
    out["sigma_crossfit_pilot"] = dict(queries=chains["joint_energy"], d=D_PCA,
                                       per_query=pilot, target_touched=False)

    # ---------------- (5) cost
    out["cost"] = dict(load_s_for_queries=round(t_load, 1), n_queries_loaded=len(allq),
                       m2_per_query_s=round(float(np.median(t_q)), 2),
                       projected_s1_primary_min=round(259 * float(np.median(t_q)) / 60, 1),
                       projected_load_all_259_min=round(t_load / len(allq) * 259 / 60, 1),
                       wall_s=round(time.time() - t0, 1))
    print(f"[S0-5] {out['cost']}", flush=True)
    json.dump(out, open(f"{OUT}/s0_inventory.json", "w"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    main()
