"""D024 / S3 — primary and secondary analysis (D024 Review 2026-09-29, Calls 1-4 + amendment).

Hierarchical bootstrap, ONE set of draws shared by all arms (Call 3; D023's scheme):
  configs  M = boot_draws(25, 2000, seed=20260928)
  seeds    S = default_rng(20260928).integers(0, 5, (2000, 8, 5)), shared over arms,
           independent over k
  acc_{a,k}^(b) = (1/5) sum_j M_b . c[a,k,S_bkj] / (37 * sum M_b)
Verdicts (P1 = CLOUD - CENTROID, P2 = CENTROID - SINGLE, each on its own; no
combined verdict): BETTER lo(mean k=1..8) > 0; WORSE hi < 0; PARITY interval of
the mean over THAT comparison's differing k within [-0.02, 0.02]; else
INCONCLUSIVE. Independent-seed sensitivity: arm 2 seeds from default_rng(20260930),
differing k. CLOUD - SINGLE reported without verdict.
Amendment (Call 4): if CENTROID-raw was trained and its mean-over-k mean top-1
exceeds CENTROID's, CLOUD - CENTROID-raw is reported with the same interval,
descriptively, and flagged for the headline.
Secondaries: (a) linear, config bootstrap; (b) D021-style predictor footing;
(c) k=8 666-pair tail, named pairs, D019 hard-model sets; (d) chains.
Usage:  PYTHONPATH=.:experiments python experiments/d024_analysis.py
"""
import json

import numpy as np
from scipy.stats import spearmanr

from LLMmap.joint_statistic import cvar
from d008_lib import boot_draws
from d019_hard_model_recall import model_sets
from d021_s0 import jackknife

OUT = "./results/D024"
NCFG, NM, NB, DELTA, SEED = 25, 37, 2000, 0.02, 20260928
KS = list(range(1, 9))
NAMED = ["tiiuae/Falcon3-10B-Instruct | tiiuae/Falcon3-7B-Instruct",
         "microsoft/Phi-3-medium-128k-instruct | microsoft/Phi-3-medium-4k-instruct"]
D21_CHAINS = ("paper8", "coverage", "joint_energy")


def pct(x):
    return dict(lo=round(float(np.percentile(x, 2.5)), 4), hi=round(float(np.percentile(x, 97.5)), 4))


def verdict_of(all8, dk):
    if all8["lo"] > 0:
        return "BETTER"
    if all8["hi"] < 0:
        return "WORSE"
    if dk is not None and dk["lo"] >= -DELTA and dk["hi"] <= DELTA:
        return "PARITY"
    return "INCONCLUSIVE"


def main():
    sel = json.load(open(f"{OUT}/selection.json"))
    trained = sel["trained_arms"]
    c24 = np.load(f"{OUT}/run_counts.npz")
    l24 = np.load(f"{OUT}/linear_counts.npz")
    c10 = np.load("./results/D010/run_counts.npz")
    c23 = np.load("./results/D023/run_counts.npz")
    l20 = np.load("./results/D020/linear_counts.npz")
    l23 = np.load("./results/D023/linear_counts.npz")
    M = boot_draws(NCFG, NB, seed=SEED)
    tot = M.sum(1)
    S = np.random.default_rng(SEED).integers(0, 5, (NB, 8, 5))
    S_ind = np.random.default_rng(SEED + 2).integers(0, 5, (NB, 8, 5))

    def cnt(arm, k, key="cnt_total"):
        if arm == "CLOUD":
            return np.stack([c10[f"joint_energy|{k}|{r}|{key}"] for r in range(5)])
        if arm == "Greedy-ADD":
            return np.stack([c23[f"Greedy-ADD|{k}|{r}|{key}"] for r in range(5)])
        return np.stack([c24[f"{arm}|{k}|{r}|{key}"] for r in range(5)])

    def acc_draws(arm, k, seeds):
        per_seed = (M @ cnt(arm, k).T) / (NM * tot[:, None])             # (NB, 5)
        return np.take_along_axis(per_seed, seeds[:, k - 1, :], 1).mean(1)

    def point(arm, k):
        return float(cnt(arm, k).sum(1).mean() / (NM * NCFG))

    def seed_range(arm, k):
        return float(np.ptp(cnt(arm, k).sum(1) / (NM * NCFG)))

    def contrast(a, b, ks, seeds_a=S, seeds_b=S):
        if not ks:
            return None
        d = np.mean([acc_draws(a, k, seeds_a) - acc_draws(b, k, seeds_b) for k in ks], 0)
        p = float(np.mean([point(a, k) - point(b, k) for k in ks]))
        return dict(point=round(p, 4), **pct(d), ks=ks)

    def per_k_table(a, b, diff_k):
        rows = {}
        for k in KS:
            joint = acc_draws(a, k, S) - acc_draws(b, k, S)
            avg = lambda arm: ((M @ cnt(arm, k).T) / (NM * tot[:, None])).mean(1)
            old = avg(a) - avg(b)
            rng_ = max(seed_range(a, k), seed_range(b, k))
            d = point(a, k) - point(b, k)
            lo_o, hi_o = np.percentile(old, [2.5, 97.5])
            rows[k] = dict(a=round(point(a, k), 4), b=round(point(b, k), 4), delta=round(d, 4),
                           hierarchical=pct(joint), identical_input=k not in diff_k,
                           old_rule=dict(ci=[round(float(lo_o), 4), round(float(hi_o), 4)],
                                         seed_range=round(rng_, 4),
                                         resolved=bool((lo_o > 0 or hi_o < 0) and abs(d) > rng_)))
        return rows

    def compare(name, a, b, with_verdict):
        diff_k = sel["differing_k"][name]
        all8 = contrast(a, b, KS)
        dk = contrast(a, b, diff_k)
        ind = contrast(a, b, diff_k, S, S_ind)
        out = dict(a=a, b=b, mean_k1_8=all8, mean_differing_k=dk, differing_k=diff_k,
                   per_k=per_k_table(a, b, diff_k))
        if not diff_k:
            out["verdict"] = "IDENTICAL CHAINS"
            return out
        if with_verdict:
            v = verdict_of(all8, dk)
            v_ind = verdict_of(ind, ind)
            out.update(verdict=v, independent_seed_sensitivity=dict(interval=ind, verdict_reading=v_ind),
                       schemes_agree=bool(v == v_ind))
        else:
            out["independent_seed_sensitivity"] = dict(interval=ind)
        return out

    prim = {"P1_CLOUD_minus_CENTROID": compare("P1_CLOUD_minus_CENTROID", "CLOUD", "CENTROID", True),
            "P2_CENTROID_minus_SINGLE": compare("P2_CENTROID_minus_SINGLE", "CENTROID", "SINGLE", True),
            "CLOUD_minus_SINGLE": compare("CLOUD_minus_SINGLE", "CLOUD", "SINGLE", False)}
    for n, v in prim.items():
        print(f"[S3] {n}: {v.get('verdict', '(no verdict)')} mean k1-8 {v['mean_k1_8']}; "
              f"differing k {v['mean_differing_k']}; ind {v.get('independent_seed_sensitivity')}",
              flush=True)
        for k, r in v["per_k"].items():
            print(f"     k={k}: {r['a']} vs {r['b']} d {r['delta']:+.4f} {r['hierarchical']} "
                  f"ident {r['identical_input']} old-rule {r['old_rule']['resolved']}", flush=True)

    # ---- Call 4 amendment: strongest centroid baseline
    strongest = None
    if "CENTROID-raw" in trained:
        m_raw = float(np.mean([point("CENTROID-raw", k) for k in KS]))
        m_cen = float(np.mean([point("CENTROID", k) for k in KS]))
        raw_vs_cen = compare("CENTROID-raw_minus_CENTROID", "CENTROID-raw", "CENTROID", False)
        strongest = dict(mean_top1_over_k=dict(raw=round(m_raw, 4), normalised=round(m_cen, 4)),
                         raw_exceeds_normalised=bool(m_raw > m_cen),
                         raw_minus_normalised=raw_vs_cen)
        if m_raw > m_cen:
            strongest["CLOUD_minus_CENTROID-raw"] = compare("CLOUD_minus_CENTROID-raw", "CLOUD",
                                                            "CENTROID-raw", False)
            strongest["HEADLINE"] = ("raw centroid trains better than normalised: the novelty claim is "
                                     "only as strong as the weaker of CLOUD-CENTROID and CLOUD-raw")
        print(f"[S3] strongest-baseline: {strongest['mean_top1_over_k']} "
              f"{strongest.get('CLOUD_minus_CENTROID-raw', {}).get('mean_k1_8')}", flush=True)

    # ---- ladder (descriptive)
    ladder_arms = ["SINGLE", "CENTROID"] + (["CENTROID-raw"] if "CENTROID-raw" in trained else []) \
        + ["CLOUD", "Greedy-ADD"]
    ladder = {a: {k: dict(mean=round(point(a, k), 4),
                          seeds=[round(float(x), 4) for x in cnt(a, k).sum(1) / (NM * NCFG)])
                  for k in KS} for a in ladder_arms}
    for a in ladder_arms:
        ladder[a]["mean_over_k"] = round(float(np.mean([point(a, k) for k in KS])), 4)
    print(f"[S3] ladder mean over k: { {a: ladder[a]['mean_over_k'] for a in ladder_arms} }", flush=True)

    # ---- (a) linear, config bootstrap only
    def lin_cnt(arm, p, k):
        if arm == "CLOUD":
            return l20[f"JointGreedy|{p}|{k}|cnt_total"]
        if arm == "Greedy-ADD":
            return l23[f"Greedy-ADD|{p}|{k}|cnt_total"]
        return l24[f"{arm}|{p}|{k}|cnt_total"]

    lin_pairs = [("CLOUD", "CENTROID"), ("CENTROID", "SINGLE"), ("CLOUD", "SINGLE")]
    if "CENTROID-raw" in trained:
        lin_pairs += [("CLOUD", "CENTROID-raw"), ("CENTROID-raw", "CENTROID")]
    lin = {}
    for p in ("concat", "meanpool"):
        la = lambda arm, k: ((M @ lin_cnt(arm, p, k)) / (NM * tot),
                             float(lin_cnt(arm, p, k).sum() / (NM * NCFG)))
        lin[p] = {"points": {a: {k: round(la(a, k)[1], 4) for k in KS} for a in ladder_arms}}
        for a, b in lin_pairs:
            rows = {}
            for k in KS:
                (da, pa), (db, pb) = la(a, k), la(b, k)
                rows[k] = dict(delta=round(pa - pb, 4), **pct(da - db))
            d8 = np.mean([la(a, k)[0] - la(b, k)[0] for k in KS], 0)
            lin[p][f"{a}_minus_{b}"] = dict(
                per_k=rows, mean_k=dict(point=round(float(np.mean([rows[k]["delta"] for k in KS])), 4),
                                        **pct(d8)))
        print(f"[S3] linear {p}: " + "; ".join(f"{n} {v['mean_k']}" for n, v in lin[p].items()
                                                if n != "points"), flush=True)

    # ---- (b) predictor footing, D021-style
    dist = np.load(f"{OUT}/distances.npz")
    ia, ib = dist["ia"], dist["ib"]
    pairs = json.load(open("./results/D021/evidence_tok200.meta.json"))["pairs"]
    d16 = sorted(json.load(open("./results/D016/full_pair_trained_accuracy.json"))["rows"],
                 key=lambda r: r["index"])
    assert [r["pair"].replace(" | ", "|") for r in d16] == pairs
    y = {c: np.array([r[c] for r in d16]) for c in D21_CHAINS}
    d21 = np.load("./results/D021/predictors.npz")
    d21_chain_q = json.load(open("./results/D021/predictors.json"))["chains"]
    pred = {n: {c: d21[f"{n}|{c}|8"] for c in D21_CHAINS} for n in ("MAX", "JOINT-E", "ADD")}
    for n, arr in (("CENTROID", dist["centroid"]), ("SINGLE", dist["single_effective"]),
                   ("CENTROID-raw", dist["centroid_raw"])):
        pred[n] = {c: np.sqrt(arr[d21_chain_q[c][:8]].sum(0)) for c in D21_CHAINS}

    def pooled(n, mask):
        return float(np.mean([spearmanr(pred[n][c][mask], y[c][mask])[0] for c in D21_CHAINS]))

    def jk(fn):
        T, hw, lo, hi = jackknife(fn, ia, ib)
        return dict(value=round(T, 4), half_width=round(hw, 4), lo=round(lo, 4), hi=round(hi, 4))

    g21 = json.load(open("./results/D021/gate_analysis.json"))["primary"]
    foot = {n: jk(lambda m, n=n: pooled(n, m)) for n in pred}
    for n in ("MAX", "JOINT-E", "ADD"):
        assert abs(foot[n]["value"] - g21[n]["value"]) < 1e-3, (n, foot[n], g21[n])
    foot["d_JOINT-E_minus_CENTROID"] = jk(lambda m: pooled("JOINT-E", m) - pooled("CENTROID", m))
    foot["d_CENTROID_minus_SINGLE"] = jk(lambda m: pooled("CENTROID", m) - pooled("SINGLE", m))
    foot["d_ADD_minus_CENTROID"] = jk(lambda m: pooled("ADD", m) - pooled("CENTROID", m))
    foot["per_chain"] = {c: {n: round(float(spearmanr(pred[n][c], y[c])[0]), 4) for n in pred}
                         for c in D21_CHAINS}
    foot["d021_values_reproduced"] = True
    print(f"[S3] (b) pooled rho: { {n: foot[n]['value'] for n in pred} }; "
          f"JOINT-E-CENTROID {foot['d_JOINT-E_minus_CENTROID']}; "
          f"CENTROID-SINGLE {foot['d_CENTROID_minus_SINGLE']}", flush=True)

    # ---- (c) hard cases at k=8
    def pair_counts(arm):
        if arm in ("CLOUD", "paper8"):
            key = "JointGreedy" if arm == "CLOUD" else "paper8"
            return np.stack([c23[f"{key}|8|{r}|cnt_pair666"] for r in range(5)])
        if arm == "Greedy-ADD":
            return np.stack([c23[f"Greedy-ADD|8|{r}|cnt_pair666"] for r in range(5)])
        return np.stack([c24[f"{arm}|8|{r}|cnt_pair666"] for r in range(5)])

    def pair_draws(arm):
        c = pair_counts(arm)                                                     # (5,666,25)
        per_seed = np.einsum("spc,bc->bsp", c, M) / (2.0 * tot[:, None, None])   # (NB,5,666)
        pick = np.take_along_axis(per_seed, S[:, 7, :][:, :, None], 1).mean(1)   # (NB,666)
        return pick, c.sum(2).mean(0) / (2.0 * NCFG)

    arms8 = ladder_arms + ["paper8"]
    pd, pp = {}, {}
    for a in arms8:
        pd[a], pp[a] = pair_draws(a)
    cv = lambda x: np.array([cvar(row, 0.1) for row in x])
    cvd = {a: cv(pd[a]) for a in arms8}
    tail = {}
    for a in arms8:
        tail[a] = dict(cvar01=round(cvar(pp[a], 0.1), 4), worst=round(float(pp[a].min()), 4),
                       worst_pair=pairs[int(pp[a].argmin())],
                       named={n: round(float(pp[a][pairs.index(n.replace(" | ", "|"))]), 4) for n in NAMED})
    tail_diffs = {}
    for a, b in [("CLOUD", "CENTROID"), ("CENTROID", "SINGLE"), ("CLOUD", "SINGLE")] + \
            ([("CLOUD", "CENTROID-raw")] if "CENTROID-raw" in trained else []):
        named = {}
        for n in NAMED:
            j = pairs.index(n.replace(" | ", "|"))
            named[n] = dict(point=round(float(pp[a][j] - pp[b][j]), 4), **pct(pd[a][:, j] - pd[b][:, j]))
        tail_diffs[f"{a}_minus_{b}"] = dict(
            cvar01=dict(point=round(tail[a]["cvar01"] - tail[b]["cvar01"], 4), **pct(cvd[a] - cvd[b])),
            named=named)
    print(f"[S3] (c) tail: { {a: (tail[a]['cvar01'], tail[a]['worst']) for a in arms8} }", flush=True)

    models = json.load(open("./results/D008/selection.json"))["models"]
    sets, _ = model_sets(models)
    hard_sets = {}
    for sname, ms in sets.items():
        idx = [models.index(m) for m in ms]
        row = {}
        for a in ladder_arms:
            cm = cnt(a, 8, "cnt_model")                                          # (5,37,25)
            per_seed = cm[:, idx, :].mean((1, 2))
            row[a] = dict(mean=round(float(per_seed.mean()), 4),
                          seed_range=[round(float(per_seed.min()), 4), round(float(per_seed.max()), 4)])
        hard_sets[sname] = row
    # sanity: CLOUD per-model means == cnt_total means
    assert abs(cnt("CLOUD", 8, "cnt_model").mean() - point("CLOUD", 8)) < 1e-9
    print(f"[S3] (c) hard-model sets: {hard_sets}", flush=True)

    # ---- (d) chains
    flags = {r["id"]: r for r in json.load(open("./results/D022/query_flags.json"))["rows"]}
    chains = {**sel["chains"], **sel["reference_chains"]}
    comp = {}
    for a, q in chains.items():
        q8 = q[:8]
        fam = {}
        for x in q8:
            lab = flags[x]["label"]
            fam[lab] = fam.get(lab, 0) + 1
        comp[a] = dict(chain=q8, labels=fam, union_flags=int(sum(bool(flags[x]["union_flag"]) for x in q8)),
                       first_flag_position=next((i + 1 for i, x in enumerate(q8) if flags[x]["union_flag"]), None))
    print(f"[S3] (d) flags: { {a: comp[a]['union_flags'] for a in comp} }", flush=True)

    json.dump(dict(
        schema="d024-analysis-v1",
        verdicts={"P1_CLOUD_minus_CENTROID": prim["P1_CLOUD_minus_CENTROID"].get("verdict"),
                  "P2_CENTROID_minus_SINGLE": prim["P2_CENTROID_minus_SINGLE"].get("verdict")},
        verdict_rule="BETTER: lo(mean k=1..8) > 0; WORSE: hi < 0; PARITY: interval of mean over the "
                     "comparison's differing k within [-0.02, 0.02]; else INCONCLUSIVE. P1 and P2 are "
                     "separate questions; no combined verdict (Review Call 3)",
        bootstrap=dict(configs="boot_draws(25, 2000, seed=20260928)",
                       seeds="default_rng(20260928).integers(0,5,(2000,8,5)), shared by all arms",
                       independent_sensitivity_seed_arm2="default_rng(20260930)",
                       note="5-with-replacement understates seed variance of a 5-seed mean by 4/5"),
        primary=prim, strongest_baseline_amendment=strongest, ladder=ladder,
        secondary_a_linear=lin, secondary_b_predictor_footing=foot,
        secondary_c_tail_k8=dict(per_arm=tail, differences=tail_diffs, hard_model_sets=hard_sets),
        secondary_d_chains=dict(composition=comp, overlaps_k8=sel["overlaps_k8"],
                                overlap_with_cloud_by_k=sel["overlap_with_cloud_by_k"])),
        open(f"{OUT}/analysis.json", "w"), indent=1)
    print(f"[S3] wrote {OUT}/analysis.json", flush=True)


if __name__ == "__main__":
    main()
