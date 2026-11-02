"""D035 / S3 — the single S_test read (Review D035/P1: Calls 1-4 -> (a) / approved; A1; disclosures D1-D5).

Order (P1 §5-§6): plot inputs for the I1 hard-tail plot (written first; d035_plots.py draws tail_k8.png before
labels are read), then the WORSE scan, the primary labels P1 = CLOUD - CENTROID*, P2 = CENTROID* - SINGLE, the
k = 8 labelled secondaries (D031 label()), D1 / A1 flags, then secondaries (a)-(e) and the D024 side-by-side.

Bootstrap = D031 v2 Boot (configs boot_draws(25, 2000, 20261005); attention seeds slot k - 1 of
default_rng(20261005).integers(0, 5, (2000, 24, 5)), shared by every arm; linear: configs only).
Independent-seed row: the second arm's seeds from default_rng(20261101) (same shape).
CLOUD / PAPER8 cells = D029's counts and logits verbatim (d031_analysis.Store.ref).
Usage:  PYTHONPATH=.:experiments python experiments/d035_analysis.py
"""
import copy
import json
import time
import itertools

import numpy as np

import d031_lib as L31
import d031_analysis as A
import d029_lib as L29
import d035_lib as L
from d008_lib import boot_draws
from LLMmap.joint_statistic import cvar

KS = list(range(1, L.K + 1))
NB, NC, DELTA = A.NB, A.NC, A.DELTA
IND_SEED = 20261101


def pct(x):
    return dict(lo=A.r4(np.percentile(x, 2.5)), hi=A.r4(np.percentile(x, 97.5)))


class Data:
    def __init__(self, ch, arms):
        self.n = 85
        self.store = A.Store("v2")                      # only .ref() and .hcols() are used (D029 cells)
        self.att = dict(np.load(f"{L.OUT}/run_counts.npz"))
        self.lin = dict(np.load(f"{L.OUT}/linear_counts.npz"))
        self.arms = arms
        self.cache = {}

    def unit(self, ro, arm, k):
        key = (ro, arm, k)
        if key not in self.cache:
            if arm in ("CLOUD", "PAPER8"):
                u = self.store.ref(ro, "JG" if arm == "CLOUD" else "PAPER8", k)
            elif ro == "att":
                g = lambda c: np.stack([self.att[f"{arm}|{k}|r{r}|{c}"] for r in range(5)])
                u = A.Unit(g("cnt_total"), g("cnt_model"), g("cnt_pair_all"), k)
            else:
                z, b = self.lin, f"{arm}|{k}"
                u = A.Unit(z[f"{b}|cnt_total"][None], z[f"{b}|cnt_model"][None], z[f"{b}|cnt_pair_all"][None], k)
            self.cache[key] = u
        return self.cache[key]


def verdict(all8, dk, lin_point):
    if all8["lo"] > 0 and lin_point > 0:
        return "BETTER"
    if all8["hi"] < 0:
        return "WORSE"
    if dk is not None and dk["lo"] >= -DELTA and dk["hi"] <= DELTA:
        return "PARITY"
    return "INCONCLUSIVE"


def main():
    t0 = time.time()
    ch = json.load(open(f"{L.OUT}/chains.json"))
    vc = json.load(open(f"{L.OUT}/variant_choice.json"))
    rj = json.load(open(f"{L.OUT}/runs.json"))
    assert vc["chains_sha256"] == L31.sha(f"{L.OUT}/chains.json")
    star, other, single = vc["CENTROID_star"], vc["other_centroid"], vc["single_trained"]
    sc = star.split("|")[1]
    trained = ["CLOUD", "CENTROID|raw", "CENTROID|norm", single]
    D = Data(ch, trained)
    ms = ch["models"]
    B = A.Boot("v2")
    assert B.S.shape == (NB, 24, 5)
    Bi = copy.copy(B)
    Bi.S = np.random.default_rng(IND_SEED).integers(0, 5, (NB, 24, 5))
    ordr = lambda a: {k: (ch["chains"]["CLOUD"][:k] if a == "CLOUD" else ch["train_order"][a][str(k)]) for k in KS}
    orders = {a: ordr(a) for a in trained}
    differ = lambda a, b: [k for k in KS if orders[a][k] != orders[b][k]]

    # ---------------------------------------------------------------- (b, I1) tail-plot inputs, before labels
    pc = {a: D.unit("att", a, 8).sorted_curve() for a in trained + ["PAPER8"]}
    np.savez_compressed(f"{L.OUT}/plotdata.npz", **{f"tail|{a}": v for a, v in pc.items()},
                        **{f"curve|{ro}|{a}": np.array([D.unit(ro, a, k).e_all(B)[1] for k in KS])
                           for ro in ("att", "lin") for a in trained + ["PAPER8"]})
    print("[S3] plotdata.npz written (I1 tail plot inputs)", flush=True)

    # ---------------------------------------------------------------- primary machinery
    def e_all_k(ro, a, k, Bx=B):
        return D.unit(ro, a, k).e_all(Bx)

    def mean_k(ro, a, b, ks, Bb=B):
        if not ks:
            return None
        d = np.mean([e_all_k(ro, a, k)[0] - e_all_k(ro, b, k, Bb)[0] for k in ks], 0)
        p = float(np.mean([e_all_k(ro, a, k)[1] - e_all_k(ro, b, k)[1] for k in ks]))
        return dict(point=A.r4(p), **pct(d), ks=ks)

    def seed_pts(a, k):
        u = D.unit("att", a, k)
        return u.tot.sum(1) / (u.n * NC)

    def per_k(a, b):
        rows = {}
        dk = differ(a, b)
        for k in KS:
            (da, pa), (db, pb) = e_all_k("att", a, k), e_all_k("att", b, k)
            ua, ub = D.unit("att", a, k), D.unit("att", b, k)
            avg = lambda u: ((B.M @ u.tot.T) / (u.n * B.sM[:, None])).mean(1)
            old = avg(ua) - avg(ub)
            rng_ = max(float(np.ptp(seed_pts(a, k))), float(np.ptp(seed_pts(b, k))))
            lo_o, hi_o = np.percentile(old, [2.5, 97.5])
            (la, lpa), (lb, lpb) = e_all_k("lin", a, k), e_all_k("lin", b, k)
            rows[k] = dict(a=A.r4(pa), b=A.r4(pb), delta=A.r4(pa - pb), hierarchical=pct(da - db),
                           identical_input=k not in dk, linear=dict(delta=A.r4(lpa - lpb), **pct(la - lb)),
                           old_rule=dict(ci=[A.r4(lo_o), A.r4(hi_o)], seed_range=A.r4(rng_),
                                         resolved=bool((lo_o > 0 or hi_o < 0) and abs(pa - pb) > rng_)))
        return rows

    def k8(a, b):
        ea = {ro: D.unit(ro, a, 8) for ro in ("att", "lin")}
        eb = {ro: D.unit(ro, b, 8) for ro in ("att", "lin")}
        ad = lambda x: (x[0], x[1], None)
        out = {}
        for ro in ("att", "lin"):
            out[ro] = dict(E_all=A.contrast(ad(ea[ro].e_all(B)), ad(eb[ro].e_all(B))),
                           E_tail=A.contrast(ad(ea[ro].e_tail(B)), ad(eb[ro].e_tail(B))))
        out["label"] = A.label(out["att"]["E_all"], out["att"]["E_tail"],
                               out["lin"]["E_all"]["delta"], out["lin"]["E_tail"]["delta"])
        return out

    def compare(a, b, with_verdict):
        dk = differ(a, b)
        all8 = mean_k("att", a, b, KS)
        dmk = mean_k("att", a, b, dk)
        ind8 = mean_k("att", a, b, KS, Bi)
        indd = mean_k("att", a, b, dk, Bi)
        lin8 = mean_k("lin", a, b, KS)
        out = dict(a=a, b=b, differing_k=dk, mean_k1_8=all8, mean_differing_k=dmk, linear_mean_k1_8=lin8,
                   independent_seed=dict(mean_k1_8=ind8, mean_differing_k=indd), per_k=per_k(a, b), k8=k8(a, b))
        if not dk:
            out["verdict"] = "IDENTICAL CHAINS"
        elif with_verdict:
            out["verdict"] = verdict(all8, dmk, lin8["point"])
            out["independent_seed"]["verdict"] = verdict(ind8, indd, lin8["point"])
            out["schemes_agree"] = out["verdict"] == out["independent_seed"]["verdict"]
        return out

    # ---------------------------------------------------------------- WORSE scan (k = 8, richer first)
    scan_pairs = [("CLOUD", "CENTROID|raw"), ("CLOUD", "CENTROID|norm"), ("CLOUD", single),
                  ("CENTROID|raw", single), ("CENTROID|norm", single), (star, other)]
    scan = {f"{a} - {b}": k8(a, b) for a, b in scan_pairs}
    worse = [n for n, v in scan.items() if v["label"] == "WORSE"]
    print(f"[S3] WORSE scan (k = 8): {worse or 'none'}", flush=True)

    prim = {"P1": compare("CLOUD", star, True), "P2": compare(star, single, True)}
    rows = {"CLOUD - SINGLE": compare("CLOUD", single, False), f"CLOUD - {other}": compare("CLOUD", other, False),
            f"{star} - {other}": compare(star, other, False)}
    for n, v in {**prim, **rows}.items():
        print(f"[S3] {n} ({v['a']} - {v['b']}): {v.get('verdict', '(no verdict)')}; mean k1-8 {v['mean_k1_8']}; "
              f"differing k {v['mean_differing_k']}; linear {v['linear_mean_k1_8']}; ind {v['independent_seed']}; "
              f"k=8 label {v['k8']['label']}", flush=True)
        for k, r in v["per_k"].items():
            print(f"     k={k}: {r['a']} vs {r['b']} d {r['delta']:+.4f} {r['hierarchical']} ident {r['identical_input']}"
                  f" lin {r['linear']} old-rule {r['old_rule']['resolved']}", flush=True)

    # ---------------------------------------------------------------- D1 / A1
    mo = lambda a: float(np.mean([e_all_k("att", a, k)[1] for k in KS]))
    d1 = dict(mean_top1_over_k={a: A.r4(mo(a)) for a in trained}, other_exceeds_star=bool(mo(other) > mo(star)))
    if d1["other_exceeds_star"]:
        d1["CLOUD_minus_other"] = rows[f"CLOUD - {other}"]["mean_k1_8"]
        d1["HEADLINE"] = (f"{other} trains better than CENTROID* = {star} on S_test: the fingerprint claim is no "
                          f"stronger than the weaker of CLOUD - {star} and CLOUD - {other}")
    a1 = dict(linear_prefers=vc["linear_diagnostic"]["prefers"], attention_choice=star,
              agrees=vc["linear_diagnostic"]["agrees_with_attention"])
    if not a1["agrees"]:
        a1["HEADLINE"] = f"the linear S_val diagnostic prefers {vc['linear_diagnostic']['prefers']}, not {star}"
    print(f"[S3] D1 {d1['mean_top1_over_k']} other>star {d1['other_exceeds_star']}; A1 {a1}", flush=True)

    # ---------------------------------------------------------------- (b) hard cases at k = 8
    pid = {f"{a} | {b}": j for j, (a, b) in enumerate(itertools.combinations(ms, 2))}
    tg = L29.targets(ms)
    npr = [pid[p["pair"]] for p in json.load(open("./results/D027/n_prime.json"))["pairs"]]
    assert len(npr) == 161
    rest = sorted(set(range(len(pid))) - set(npr))
    sets = dict(H_all=[pid[p] for p in tg["strata"]["H_all"]], stably_hard=[pid[p] for p in tg["strata"]["stably_hard"]],
                N_prime=npr, not_N_prime=rest)
    hard = {}
    for a in trained + ["PAPER8"]:
        u = D.unit("att", a, 8)
        pa = u.pair.sum(2).mean(0) / (2 * NC)
        worst = np.argsort(pa, kind="stable")[:int(np.floor(L.GAMMA * len(pa)))]
        hard[a] = dict({s: A.r4(u.e_hard(B, c)[1]) for s, c in sets.items()},
                       E_tail=A.r4(cvar(pa, L.GAMMA)), N_prime_share_of_worst_decile=A.r4(np.isin(worst, npr).mean()))
    hard_d = {}
    for a, b in [("CLOUD", star), (star, single), ("CLOUD", single), ("CLOUD", other)]:
        hard_d[f"{a} - {b}"] = {s: A.contrast((*D.unit("att", a, 8).e_hard(B, c), None),
                                              (*D.unit("att", b, 8).e_hard(B, c), None)) for s, c in sets.items()}
    print(f"[S3] (b) hard k=8: {hard}", flush=True)

    # ---------------------------------------------------------------- (e) Call 1 split (64 greedy / 21 sampled refs)
    eff = json.load(open(f"{L.OUT}/s0.json"))["single_tier"]["readings"]["effective"]
    g_idx = [i for i, m in enumerate(ms) if not eff[m]["do_sample"]]
    s_idx = [i for i, m in enumerate(ms) if eff[m]["do_sample"]]
    assert (len(g_idx), len(s_idx)) == (64, 21)
    split = {}
    for nm, idx in (("greedy_ref_64", g_idx), ("sampled_ref_21", s_idx)):
        per = {k: (D.unit("att", single, k).model_set(B, idx), D.unit("att", star, k).model_set(B, idx)) for k in KS}
        d8 = per[8][0][0] - per[8][1][0]
        dm = np.mean([per[k][0][0] - per[k][1][0] for k in KS], 0)
        split[nm] = dict(n=len(idx),
                         k8=dict(single=A.r4(per[8][0][1]), star=A.r4(per[8][1][1]),
                                 delta=A.r4(per[8][0][1] - per[8][1][1]), **pct(d8)),
                         mean_k=dict(delta=A.r4(np.mean([per[k][0][1] - per[k][1][1] for k in KS])), **pct(dm)))
    print(f"[S3] (e) SINGLE - CENTROID* by reference kind: {split}", flush=True)

    # ---------------------------------------------------------------- D024 side-by-side (v1)
    d24 = json.load(open("./results/D024/analysis.json"))
    if sc == "norm":
        side = dict(P1=d24["primary"]["P1_CLOUD_minus_CENTROID"]["mean_k1_8"],
                    P2=d24["primary"]["P2_CENTROID_minus_SINGLE"]["mean_k1_8"], scaling="matching (normalised)")
    else:
        c24 = np.load("./results/D024/run_counts.npz")
        M24 = boot_draws(25, NB, seed=20260928)
        S24 = np.random.default_rng(20260928).integers(0, 5, (NB, 8, 5))
        cnt = lambda arm, k: np.stack([c24[f"{arm}|{k}|{r}|cnt_total"] for r in range(5)])
        acc = lambda arm, k: np.take_along_axis((M24 @ cnt(arm, k).T) / (37 * M24.sum(1)[:, None]),
                                                S24[:, k - 1, :], 1).mean(1)
        pt = lambda arm, k: float(cnt(arm, k).sum(1).mean() / (37 * 25))
        dd = np.mean([acc("CENTROID-raw", k) - acc("SINGLE", k) for k in KS], 0)
        side = dict(P1=d24["strongest_baseline_amendment"]["CLOUD_minus_CENTROID-raw"]["mean_k1_8"],
                    P2=dict(point=A.r4(np.mean([pt("CENTROID-raw", k) - pt("SINGLE", k) for k in KS])), **pct(dd),
                            note="v1 CENTROID-raw - SINGLE(normalised): mixes scalings (D024 trained SINGLE only "
                                 "normalised); D024 draws, seed 20260928"), scaling="raw (P1 matching; P2 mixed)")
    print(f"[S3] D024 side-by-side ({side['scaling']}): P1 {side['P1']}; P2 {side['P2']}", flush=True)

    out = dict(
        schema="d035-analysis-v1", centroid_star=star, other_centroid=other, single=single,
        verdicts={n: v["verdict"] for n, v in prim.items()},
        verdict_rule="BETTER lo(mean k1-8) > 0 and linear mean-k point > 0; WORSE hi < 0; PARITY interval of the "
                     "mean over that comparison's differing k within +-0.02; else INCONCLUSIVE",
        bootstrap=dict(configs="boot_draws(25, 2000, 20261005)",
                       seeds="default_rng(20261005).integers(0, 5, (2000, 24, 5)), slot k - 1, shared by all arms",
                       independent_seed_arm2=f"default_rng({IND_SEED})", linear="configs only",
                       note="5-with-replacement understates seed variance of a 5-seed mean by 4/5 (D4)"),
        worse_scan=dict(order=[f"{a} - {b}" for a, b in scan_pairs], worse=worse, rows=scan),
        primary=prim, unlabelled=rows, D1_strongest_baseline=d1, A1_linear_val=a1,
        secondary_b_hard_k8=dict(per_arm=hard, differences=hard_d, n={s: len(c) for s, c in sets.items()}),
        secondary_c_chains=dict(chains=ch["chains"], overlaps_k8=ch["overlaps_k8"],
                                overlap_with_cloud_by_k=ch["overlap_with_cloud_by_k"],
                                single_vs_readings_by_k=ch["single_vs_readings_by_k"],
                                cross_objective=ch["cross_objective"]),
        secondary_d_collisions=dict(colliding_queries=ch["single_colliding_queries"],
                                    in_chain=ch["colliding_in_chain"]),
        secondary_e_reference_split=split, d024_side_by_side=side,
        reused_cells="CLOUD / PAPER8 = D029 counts + logits (I6); new arms trained here under D029's hparams hash",
        code_commit=L31.git_head(), wall_min=round((time.time() - t0) / 60, 1))
    L31.jdump(out, f"{L.OUT}/analysis.json")
    print(f"[S3] verdicts {out['verdicts']}; wrote analysis.json ({out['wall_min']} min)", flush=True)


if __name__ == "__main__":
    main()
