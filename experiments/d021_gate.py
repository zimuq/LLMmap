"""D021 / A1 + S1–S3 — the additive-evidence feasibility gate.

Review (APPROVED WITH AMENDMENTS, 2026-09-28):
  A1  FIRST, target-free: a system-prompt-grouped split of S_build. Every system
      prompt (32 build prompts + "none") is assigned GLOBALLY to half A or B
      (seed 20260928, LPT balancing of total configs), so no prompt appears in
      both halves in ANY model. Configs are drawn per model (D006), so per-model
      half sizes vary; reported. M^2 (d=64) under D007's halves vs the grouped
      halves, per query: if the median Spearman over the 666 pairs is < 0.99,
      STOP before the target is loaded (exit 3).
  S1  evidence tensor M^2_q(p), 259 x 666: primary d=64, Sigma_q from full
      S_build, D007 halves; variants d=32/128, per-half Sigma, per-pair Sigma,
      grouped halves.
  S2  MAX / JOINT-E / ADD / JOINT-G for the 3 chains at k=1..8; J == A at k=1.
  S3  primary gate (pre-registered rule, LOMO jackknife, delta=0.05), realised
      half-widths + reachability (Call 1 amendment), secondaries (a)-(e).

Everything before `load_target()` is target-free by construction.
Usage:  PYTHONPATH=.:experiments python experiments/d021_gate.py
"""
import os
import sys
import json
import glob
import time

import numpy as np
from scipy.stats import spearmanr, norm
from sklearn.covariance import ledoit_wolf

from d021_s0 import load_meta, chains_k8, jackknife

OUT = "./results/D021"
EMB = "./data/corpus_v1/embeddings"
CORPUS = "./data/corpus_v1"
CHAINS = ("paper8", "coverage", "joint_energy")
DELTA, D_PRI, DS = 0.05, 64, (32, 64, 128)
SEED_SPLIT, SEED_MC, N_MC = 20260928, 20260928, 2000


# ------------------------------------------------------------------ loading
def load_all(models):
    """(37, 259, 75, 1024) float32 S_build, and each model's per-config system
    prompt text (build pool, config_index order)."""
    X, prompts, templ = [], [], []
    for m in models:
        slug = m.replace("/", "__")
        a = np.load(f"{EMB}/{slug}.npy", mmap_mode="r")
        rows = json.load(open(f"{EMB}/{slug}.index.json"))["rows"]
        pos = {(r["query_index"], r["config"]): i for i, r in enumerate(rows)
               if r["pool"] == "build"}
        idx = np.array([pos[(q, c)] for q in range(259) for c in range(75)])
        X.append(np.asarray(a[idx], dtype=np.float32).reshape(259, 75, -1))
        p, t = {}, {}
        with open(f"{CORPUS}/{slug}.jsonl") as fh:
            for line in fh:
                r = json.loads(line)
                if r["dataset"] == "build":
                    pc = r["prompt_conf"]
                    p[r["config_index"]] = pc["system_prompt"] or "<none>"
                    t[r["config_index"]] = dict(cot=pc["cot_prompt"], rag=pc["rag_prompt"])
        assert sorted(p) == list(range(75))
        prompts.append([p[c] for c in range(75)])
        templ.append([t[c] for c in range(75)])
    return np.stack(X), prompts, templ


def grouped_split(prompts):
    """Global prompt -> half, LPT on total config counts, seeded tie order."""
    from collections import Counter
    cnt = Counter(t for pm in prompts for t in pm)
    groups = sorted(cnt)
    rng = np.random.default_rng(SEED_SPLIT)
    order = rng.permutation(len(groups))
    ranked = sorted(range(len(groups)), key=lambda i: (-cnt[groups[i]], order[i]))
    half, tot = {}, [0, 0]
    for i in ranked:
        h = 0 if tot[0] <= tot[1] else 1
        half[groups[i]] = h
        tot[h] += cnt[groups[i]]
    A = [[c for c in range(75) if half[pm[c]] == 0] for pm in prompts]
    B = [[c for c in range(75) if half[pm[c]] == 1] for pm in prompts]
    return A, B, dict(n_groups=len(groups), totals=tot,
                      per_model_A=[len(a) for a in A], per_model_B=[len(b) for b in B],
                      min_half=int(min(min(len(a) for a in A), min(len(b) for b in B))),
                      assignment={g: half[g] for g in groups})


# ------------------------------------------------------------------ evidence
def pca_scores(Xq, dmax=128):
    n_m, n_c, dim = Xq.shape
    flat = Xq.reshape(-1, dim).astype(np.float64)
    mu = flat.mean(0)
    _, _, vt = np.linalg.svd(flat - mu, full_matrices=False)
    return ((flat - mu) @ vt[:dmax].T).reshape(n_m, n_c, dmax)


def lw_within(Z, halves=None):
    """Pooled within-model Ledoit-Wolf covariance. halves: per-model index lists
    to fit per half and average (the 'per-half Sigma' variant)."""
    d = Z.shape[-1]
    if halves is None:
        R = (Z - Z.mean(1, keepdims=True)).reshape(-1, d)
        return ledoit_wolf(R, assume_centered=True)
    Ss, sh = [], []
    for H in halves:
        R = np.concatenate([Z[v, H[v]] - Z[v, H[v]].mean(0) for v in range(Z.shape[0])])
        s, k = ledoit_wolf(R, assume_centered=True)
        Ss.append(s); sh.append(k)
    return (Ss[0] + Ss[1]) / 2, float(np.mean(sh))


def half_means(Z, H):
    return np.stack([Z[v, H[v]].mean(0) for v in range(Z.shape[0])])


def m2(Z, P, HA, HB, ia, ib):
    mA, mB = half_means(Z, HA), half_means(Z, HB)
    return np.einsum("pi,ij,pj->p", mA[ia] - mA[ib], P, mB[ia] - mB[ib])


def m2_pairsigma(Z, HA, HB, ia, ib):
    d = Z.shape[-1]
    R = Z - Z.mean(1, keepdims=True)
    mA, mB = half_means(Z, HA), half_means(Z, HB)
    out = np.empty(len(ia))
    for p, (a, b) in enumerate(zip(ia, ib)):
        S, _ = ledoit_wolf(np.concatenate([R[a], R[b]]), assume_centered=True)
        out[p] = (mA[a] - mA[b]) @ np.linalg.solve(S, mB[a] - mB[b])
    return out


# ------------------------------------------------------------------ targets (after A1 only)
def load_target(pairs):
    d16 = sorted(json.load(open("./results/D016/full_pair_trained_accuracy.json"))["rows"],
                 key=lambda r: r["index"])
    assert [r["pair"].replace(" | ", "|") for r in d16] == pairs
    return {c: np.array([r[c] for r in d16]) for c in CHAINS}


def trained_curves():
    d9 = json.load(open("./results/D009/metrics_by_condition_k.json"))["metrics"]
    d10 = json.load(open("./results/D010/metrics_by_k.json"))["metrics"]
    d17 = json.load(open("./results/D017/linear_metrics_by_k.json"))["metrics"]
    d18 = json.load(open("./results/D018/metrics_by_k.json"))["metrics"]
    ks = [str(k) for k in range(1, 9)]
    att = {"paper8": d9["paper8"], "coverage": d9["cvar_max"],
           "joint_energy": d10["joint_energy"],
           "H1-attention": d18["H1-attention|attention"],
           "H1-linear": d18["H1-linear|attention"]}
    out = {"attention": {c: [att[c][k]["mean_top1"] for k in ks] for c in att}}
    for pool in ("concat", "meanpool"):
        out[f"linear_{pool}"] = {
            **{c: [d17[pool][c][k]["mean_top1"] for k in ks]
               for c in ("paper8", "coverage", "joint_energy")},
            **{c: [d18[f"{c}|linear_{pool}"][k]["mean_top1"] for k in ks]
               for c in ("H1-attention", "H1-linear")}}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    models, pairs, sa, sb = load_meta()
    ia = np.array([models.index(p.split("|")[0]) for p in pairs])
    ib = np.array([models.index(p.split("|")[1]) for p in pairs])
    n_m = len(models)
    HA7, HB7 = [sa] * n_m, [sb] * n_m                  # D007's halves (same indices per model)

    X, prompts, templ = load_all(models)
    t_load = time.time() - t0
    GA, GB, gmeta = grouped_split(prompts)
    print(f"[load] {t_load/60:.1f} min; grouped split: {gmeta['n_groups']} prompt groups, "
          f"totals {gmeta['totals']}, per-model half min {gmeta['min_half']}", flush=True)
    # CoT/RAG overlap disclosure (not grouped, per the Review)
    cotrag = {}
    for kind in ("cot", "rag"):
        inA = {t for v in range(n_m) for c in GA[v] if (t := templ[v][c][kind])}
        inB = {t for v in range(n_m) for c in GB[v] if (t := templ[v][c][kind])}
        cotrag[kind] = dict(n_templates=len(inA | inB), in_both_halves=len(inA & inB))
    gmeta["cot_rag_overlap_not_grouped"] = cotrag
    print(f"[A1] CoT/RAG templates in both grouped halves (not grouped, disclosed): {cotrag}",
          flush=True)
    t1 = time.time()

    # ---------------- S1: evidence (target-free)
    E = {f"d{d}": np.empty((259, len(pairs))) for d in DS}
    E["d64_halfsigma"] = np.empty((259, len(pairs)))
    E["d64_pairsigma"] = np.empty((259, len(pairs)))
    E["d64_grouped"] = np.empty((259, len(pairs)))
    shrink = {k: [] for k in ("d32", "d64", "d128", "d64_halfsigma")}
    keep = {}                                          # per-query d=64 pieces for S2/S3(d)
    need = set()
    for c in chains_k8().values():
        need |= set(c)
    h1 = json.load(open("./results/D018/greedy_chains.json"))["chains"]
    for c in h1.values():
        need |= set(c[:8])
    for q in range(259):
        Z128 = pca_scores(X[:, q])
        for d in DS:
            Z = Z128[..., :d]
            S, k_ = lw_within(Z)
            P = np.linalg.inv(S)
            E[f"d{d}"][q] = m2(Z, P, HA7, HB7, ia, ib)
            shrink[f"d{d}"].append(float(k_))
            if d == D_PRI:
                E["d64_grouped"][q] = m2(Z, P, GA, GB, ia, ib)
                Sh, kh = lw_within(Z, halves=(HA7, HB7))
                E["d64_halfsigma"][q] = m2(Z, np.linalg.inv(Sh), HA7, HB7, ia, ib)
                shrink["d64_halfsigma"].append(kh)
                E["d64_pairsigma"][q] = m2_pairsigma(Z, HA7, HB7, ia, ib)
                if q in need:
                    keep[q] = dict(Z=Z.copy(), S=S, P=P)          # float64: J == A exactly
        if q % 50 == 0:
            print(f"[S1] q={q} ({(time.time()-t1)/60:.1f} min)", flush=True)
    for k_, v in E.items():
        np.save(f"{OUT}/evidence_tok200{'' if k_ == 'd64' else '_' + k_}.npy", v.astype(np.float32))
    meta = dict(schema_version="d021-evidence-v1", shape=[259, len(pairs)],
                primary="evidence_tok200.npy = d64, Sigma_q full S_build (LW), D007 halves",
                variants={k_: f"evidence_tok200_{k_}.npy" for k_ in E if k_ != "d64"},
                pca="per query, SVD on all 2,775 S_build points (37 x 75), top-d",
                shrinkage="sklearn ledoit_wolf on pooled within-model residuals, "
                          "assume_centered=True",
                shrinkage_per_query={k_: v for k_, v in shrink.items()},
                split_seed_d007=20260907, split_a=sa, split_b=sb,
                config_49="in neither D007 half: enters PCA and Sigma only",
                grouped_split=dict(seed=SEED_SPLIT, **{k_: v for k_, v in gmeta.items()
                                                       if k_ != "assignment"}),
                n_configs_per_model=75, n_models=n_m, token_budget=200,
                negative_cells={k_: int((v < 0).sum()) for k_, v in E.items()},
                pairs=pairs)
    json.dump(meta, open(f"{OUT}/evidence_tok200.meta.json", "w"), indent=1)
    print(f"[S1] {(time.time()-t1)/60:.1f} min; negative cells "
          f"{meta['negative_cells']}", flush=True)

    # ---------------- A1: prompt-overlap check (target-free), gates everything after
    rq = np.array([spearmanr(E["d64"][q], E["d64_grouped"][q])[0] for q in range(259)])
    rel = np.abs(E["d64"] - E["d64_grouped"]) / np.maximum(np.abs(E["d64"]), 1e-9)
    a1 = dict(rule="median per-query Spearman >= 0.99 -> D007 halves stay primary; "
                   "else stop for a Call before any target correlation",
              median_spearman=float(np.median(rq)), min_spearman=float(rq.min()),
              p05_spearman=float(np.percentile(rq, 5)),
              n_queries_below_0_99=int((rq < 0.99).sum()),
              median_rel_diff=float(np.median(rel)),
              mean_ratio_grouped_over_d007=float(E["d64_grouped"].mean() / E["d64"].mean()),
              negative_cells=dict(d007=int((E["d64"] < 0).sum()),
                                  grouped=int((E["d64_grouped"] < 0).sum())),
              grouped_split={k_: v for k_, v in gmeta.items()}, target_touched=False)
    a1["pass"] = bool(a1["median_spearman"] >= 0.99)
    json.dump(a1, open(f"{OUT}/a1_prompt_split_check.json", "w"), indent=1)
    print(f"[A1] median Spearman {a1['median_spearman']:.4f} (min {a1['min_spearman']:.4f}, "
          f"{a1['n_queries_below_0_99']} q < 0.99); median rel diff "
          f"{a1['median_rel_diff']:.4f}; grouped/D007 mean ratio "
          f"{a1['mean_ratio_grouped_over_d007']:.3f} -> "
          f"{'PASS' if a1['pass'] else 'STOP'}", flush=True)
    if not a1["pass"]:
        print("[A1] STOP: target not loaded. Call required.", flush=True)
        sys.exit(3)

    # ---------------- S2: predictors (target still not loaded)
    chains = chains_k8()
    S_tensor = np.load("./results/D007/S_energy_sf_tok200.npy").astype(np.float64)
    pred = {n: {c: {} for c in CHAINS} for n in ("MAX", "JOINT-E", "ADD", "JOINT-G",
                                                   "ADD_grouped")}
    ja_k1 = {}
    for c, Q in chains.items():
        acc_xy = np.zeros((len(pairs), 75, 75)); acc_xx = np.zeros((n_m, 75, 75))
        for k in range(1, 9):
            q = Q[k - 1]
            pred["MAX"][c][k] = S_tensor[Q[:k]].max(0)
            V = X[:, q].astype(np.float64)
            nrm = (V ** 2).sum(-1)
            D2 = np.maximum(nrm[:, None, :, None] + nrm[None, :, None, :]
                            - 2 * np.einsum("mid,njd->mnij", V, V), 0)
            acc_xx += D2[np.arange(n_m), np.arange(n_m)]
            acc_xy += D2[ia, ib]
            xy = np.sqrt(acc_xy).mean((1, 2)); xm = np.sqrt(acc_xx).mean((1, 2))
            pred["JOINT-E"][c][k] = (2 * xy - xm[ia] - xm[ib]) / ((xm[ia] + xm[ib]) / 2)
            pred["ADD"][c][k] = E["d64"][Q[:k]].sum(0)
            pred["ADD_grouped"][c][k] = E["d64_grouped"][Q[:k]].sum(0)
            Zc = np.concatenate([keep[qq]["Z"] for qq in Q[:k]], axis=-1)
            Sj, _ = lw_within(Zc)
            pred["JOINT-G"][c][k] = m2(Zc, np.linalg.inv(Sj), HA7, HB7, ia, ib)
        d1 = np.abs(pred["JOINT-G"][c][1] - pred["ADD"][c][1]).max()
        ja_k1[c] = float(d1 / np.abs(pred["ADD"][c][1]).max())
        assert ja_k1[c] < 1e-8, (c, ja_k1[c])
    print(f"[S2] J == A at k=1 (max rel diff) {ja_k1}", flush=True)
    np.savez_compressed(f"{OUT}/predictors.npz",
                        **{f"{n}|{c}|{k}": v for n in pred for c in pred[n]
                           for k, v in pred[n][c].items()})
    json.dump(dict(schema="d021-predictors-v1", file="predictors.npz",
                   keys="'{predictor}|{chain}|{k}' -> (666,) in D007 pair order",
                   predictors=list(pred), chains=chains, ks=list(range(1, 9)),
                   J_equals_A_at_k1_max_rel_diff=ja_k1,
                   notes=dict(MAX="max_q S_energy_sf_tok200 (D007), what GreedyCover optimises",
                              **{"JOINT-E": "scale-free joint energy, accumulator path, S_build "
                                            "75 configs (D010)"},
                              ADD="signed sum of M^2_q, d=64 (primary)",
                              **{"JOINT-G": "full LW covariance on concatenated 64k-d PCA "
                                            "scores, cross-fitted, dependence check only"},
                              ADD_grouped="ADD under the A1 grouped halves (sensitivity)")),
              open(f"{OUT}/predictors.json", "w"), indent=1)

    # ---------------- S3: primary (target loaded HERE, first time)
    y = load_target(pairs)
    ymin = np.min(np.stack([y[c] for c in CHAINS]), 0)
    nonceil = ymin < 0.99
    allp = np.ones(len(pairs), bool)

    def rho(name, c, mask, k=8, src=pred):
        return spearmanr(src[name][c][k][mask], y[c][mask])[0]

    def pooled(name, mask, src=pred):
        return float(np.mean([rho(name, c, mask, src=src) for c in CHAINS]))

    def jk(fn, base):
        T, hw, lo, hi = jackknife(lambda msk: fn(msk & base), ia, ib)
        return dict(value=round(T, 4), half_width=round(hw, 4), lo=round(lo, 4), hi=round(hi, 4))

    def block(base, src=pred, add="ADD"):
        r = {n: jk(lambda m, n=n: pooled(n, m, src), base) for n in ("MAX", "JOINT-E", add)}
        r[f"d_{add}_minus_JOINT-E"] = jk(lambda m: pooled(add, m, src) - pooled("JOINT-E", m, src), base)
        r[f"d_{add}_minus_MAX"] = jk(lambda m: pooled(add, m, src) - pooled("MAX", m, src), base)
        r["per_chain"] = {c: {n: round(rho(n, c, base, src=src), 4)
                              for n in ("MAX", "JOINT-E", add)} for c in CHAINS}
        return r

    prim = block(allp)
    dj, dm = prim["d_ADD_minus_JOINT-E"], prim["d_ADD_minus_MAX"]
    go = dj["lo"] > -DELTA and dm["lo"] > 0
    nogo = dj["hi"] < -DELTA
    verdict = "GO" if go else ("NO-GO" if nogo else "INCONCLUSIVE")
    signs = {c: dict(ADD_minus_JOINT_E=round(prim["per_chain"][c]["ADD"]
                                             - prim["per_chain"][c]["JOINT-E"], 4),
                     ADD_minus_MAX=round(prim["per_chain"][c]["ADD"]
                                         - prim["per_chain"][c]["MAX"], 4)) for c in CHAINS}
    consistent = {k_: len({np.sign(signs[c][k_]) for c in CHAINS}) == 1
                  for k_ in ("ADD_minus_JOINT_E", "ADD_minus_MAX")}
    reach = dict(realised_half_width_ADD_minus_JOINT_E=dj["half_width"],
                 realised_half_width_ADD_minus_MAX=dm["half_width"],
                 GO_first_leg_needs_point_above=round(-DELTA + dj["half_width"], 4),
                 NO_GO_needs_point_below=round(-DELTA - dj["half_width"], 4),
                 GO_second_leg_needs_point_above=round(dm["half_width"], 4),
                 p1_estimate=dict(h=0.09, GO=0.04, NO_GO=-0.14))
    print(f"[S3] VERDICT {verdict}: d(ADD-JOINT-E) {dj['value']:+.4f} [{dj['lo']:+.4f}, "
          f"{dj['hi']:+.4f}]; d(ADD-MAX) {dm['value']:+.4f} [{dm['lo']:+.4f}, {dm['hi']:+.4f}]; "
          f"per-chain {signs}; consistent {consistent}", flush=True)

    # (a) dependence
    dep = {}
    for c in CHAINS:
        A8, J8 = pred["ADD"][c][8], pred["JOINT-G"][c][8]
        ok = A8 > 0
        ratio = J8[ok] / A8[ok]
        dep[c] = dict(J_over_A_quantiles={p: round(float(np.percentile(ratio, p)), 4)
                                          for p in (5, 25, 50, 75, 95)},
                      n_A_nonpositive=int((~ok).sum()))
    dep["rho_JOINT_G"] = jk(lambda m: pooled("JOINT-G", m), allp)
    dep["d_JOINT_G_minus_ADD"] = jk(lambda m: pooled("JOINT-G", m) - pooled("ADD", m), allp)
    # (b) non-ceiling
    nc = block(nonceil)
    # (c) calibration, under both splits
    calib = {}
    for lab, name in (("d007", "ADD"), ("grouped", "ADD_grouped")):
        pe = np.concatenate([norm.cdf(-np.sqrt(np.maximum(pred[name][c][8], 0)) / 2)
                             for c in CHAINS])
        oe = np.concatenate([1 - y[c] for c in CHAINS])
        edges = np.unique(np.percentile(pe, np.linspace(0, 100, 11)))
        b = np.clip(np.digitize(pe, edges[1:-1]), 0, len(edges) - 2)
        calib[lab] = [dict(bin=int(i), n=int((b == i).sum()),
                           pred_err=round(float(pe[b == i].mean()), 4),
                           obs_err=round(float(oe[b == i].mean()), 4))
                      for i in range(len(edges) - 1) if (b == i).any()]
        calib[lab + "_overall"] = dict(pred=round(float(pe.mean()), 4), obs=round(float(oe.mean()), 4))
    # (d) k-curve: cross-fitted block-diagonal Gaussian Monte Carlo
    allchains = {**chains, "H1-attention": h1["H1-attention"][:8], "H1-linear": h1["H1-linear"][:8]}
    kc = {}
    for lab, (HA, HB) in (("d007", (HA7, HB7)), ("grouped", (GA, GB))):
        kc[lab] = {}
        for c, Q in allchains.items():
            rng = np.random.default_rng(SEED_MC)
            dist = np.zeros((n_m * N_MC, n_m))
            truth = np.repeat(np.arange(n_m), N_MC)
            accs = []
            for q in Q:
                Z, S, P = keep[q]["Z"], keep[q]["S"], keep[q]["P"]
                mA, mB = half_means(Z, HA), half_means(Z, HB)
                L, W = np.linalg.cholesky(S), np.linalg.cholesky(P)
                z = mB[truth] + rng.standard_normal((n_m * N_MC, Z.shape[-1])) @ L.T
                zw, mw = z @ W, mA @ W
                dist += (zw ** 2).sum(1)[:, None] + (mw ** 2).sum(1)[None] - 2 * zw @ mw.T
                accs.append(float((dist.argmin(1) == truth).mean()))
            kc[lab][c] = accs
    tr = trained_curves()
    kagree = {}
    for lab in kc:
        for ro, curves in tr.items():
            for scope, cs in (("primary_3", CHAINS), ("all_5", list(allchains))):
                p_ = np.concatenate([kc[lab][c] for c in cs])
                t_ = np.concatenate([curves[c] for c in cs])
                kagree[f"{lab}|{ro}|{scope}"] = dict(
                    spearman=round(float(spearmanr(p_, t_)[0]), 4),
                    mean_abs_gap=round(float(np.abs(p_ - t_).mean()), 4),
                    mean_signed_gap_pred_minus_trained=round(float((p_ - t_).mean()), 4),
                    n_points=int(len(p_)))
    # (e) sensitivity (reported, never chosen by fit)
    sens = {}
    for vname in ("d32", "d128", "d64_halfsigma", "d64_pairsigma", "d64_grouped"):
        src = {"MAX": pred["MAX"], "JOINT-E": pred["JOINT-E"],
               "ADD": {c: {8: E[vname][chains[c]].sum(0)} for c in CHAINS}}
        r = block(allp, src=src)
        sens[vname] = {k_: r[k_] for k_ in ("ADD", "d_ADD_minus_JOINT-E", "d_ADD_minus_MAX")}

    json.dump(dict(
        schema="d021-gate-v1", delta=DELTA, verdict=verdict,
        verdict_rule="GO: lo(d ADD-JOINT-E) > -delta AND lo(d ADD-MAX) > 0; NO-GO: "
                     "hi(d ADD-JOINT-E) < -delta; else INCONCLUSIVE (full pre-registered rule)",
        uncertainty="LOMO jackknife, 37 folds, t_36 (commit c54e4d1)",
        exploratory_targets="D016 S_test-derived, repeatedly inspected; not confirmatory",
        primary=prim, per_chain_deltas=signs, sign_consistent_across_chains=consistent,
        reachability=reach, a1=a1, secondary=dict(
            a_dependence=dep, b_nonceiling=nc, c_calibration=calib,
            d_kcurve=dict(predicted=kc, trained=tr, agreement=kagree, n_mc=N_MC, seed=SEED_MC),
            e_sensitivity=sens),
        subset_sizes=dict(all=int(allp.sum()), nonceiling=int(nonceil.sum())),
        cost=dict(load_min=round(t_load / 60, 1), wall_min=round((time.time() - t0) / 60, 1))),
        open(f"{OUT}/gate_analysis.json", "w"), indent=1,
        default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(f"[S3] non-ceiling d(ADD-JOINT-E) {nc['d_ADD_minus_JOINT-E']}; "
          f"JOINT-G {dep['rho_JOINT_G']}; calib overall {calib['d007_overall']}", flush=True)
    for k_, v in kagree.items():
        print(f"[S3d] {k_:40s} {v}", flush=True)
    for k_, v in sens.items():
        print(f"[S3e] {k_:14s} d(ADD-JOINT-E) {v['d_ADD_minus_JOINT-E']}", flush=True)
    print(f"\nwall {(time.time()-t0)/60:.1f} min; written {OUT}/", flush=True)


if __name__ == "__main__":
    main()
