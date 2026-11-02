"""D036 / S4 — the single S_test read (Review D036/P1: APPROVED; Calls 1-3 -> (a), Call 2 extended; D1-D5).

Order (P1 §7): plot inputs for the I1 hard-tail plots (closed-set k = 8 and open-set held-out k = 8; written first,
d036_plots.py draws them before labels are read), then the WORSE scan, P1 = CLOUD-raw - CENTROID-raw,
P2 = CLOUD-raw - CLOUD, the k = 8 labelled secondaries (D031 label()), the triangle assert, the open-set O1 / O2
(D033 label()), then secondaries (a)-(e).

Closed-set bootstrap = D031 v2 Boot (configs boot_draws(25, 2000, 20261005); attention seeds slot k - 1 of
default_rng(20261005).integers(0, 5, (2000, 24, 5)), shared by every arm; linear: configs only). Independent-seed
row: the second arm's seeds from default_rng(20261101) (D035's, D3). Open-set: d033_analysis.Boot("v2") (configs
only; folds fixed). CLOUD / PAPER8 = D029 counts verbatim, CENTROID-raw = D035 counts verbatim, CLOUD^fold = D033
counts verbatim.
Usage:  PYTHONPATH=.:experiments python experiments/d036_analysis.py
"""
import copy
import json
import time
import hashlib
import itertools

import numpy as np

import d031_lib as L31
import d031_analysis as A
import d033_analysis as OA
import d029_lib as L29
import d036_lib as L
from LLMmap.joint_statistic import cvar

KS = list(range(1, L.K + 1))
NB, NC, DELTA = A.NB, A.NC, A.DELTA
IND_SEED = 20261101
ARMS = ("CLOUD-raw", "CENTROID-raw", "CLOUD", "PAPER8")
FOLD_ARMS = ("CLOUD-raw", "CENTROID-raw", "CLOUD")


def pct(x):
    return dict(lo=A.r4(np.percentile(x, 2.5)), hi=A.r4(np.percentile(x, 97.5)))


def sha_arr(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


class Data:
    def __init__(self):
        self.store = A.Store("v2")                      # .ref() only (D029 cells)
        self.src = {"CLOUD-raw": (dict(np.load(f"{L.OUT}/run_counts.npz")), dict(np.load(f"{L.OUT}/linear_counts.npz")),
                                  "CLOUD-raw"),
                    "CENTROID-raw": (dict(np.load("./results/D035/run_counts.npz")),
                                     dict(np.load("./results/D035/linear_counts.npz")), "CENTROID|raw")}
        self.cache = {}

    def unit(self, ro, arm, k):
        key = (ro, arm, k)
        if key not in self.cache:
            if arm in ("CLOUD", "PAPER8"):
                u = self.store.ref(ro, "JG" if arm == "CLOUD" else "PAPER8", k)
            else:
                att, lin, nm = self.src[arm]
                if ro == "att":
                    g = lambda c: np.stack([att[f"{nm}|{k}|r{r}|{c}"] for r in range(5)])
                    u = A.Unit(g("cnt_total"), g("cnt_model"), g("cnt_pair_all"), k)
                else:
                    b = f"{nm}|{k}"
                    u = A.Unit(lin[f"{b}|cnt_total"][None], lin[f"{b}|cnt_model"][None], lin[f"{b}|cnt_pair_all"][None], k)
            self.cache[key] = u
        return self.cache[key]


def verdict(all8, dk, lin_point):
    """The D's table with the symmetric linear clause (DECISIONS 2026-11-02, D5)."""
    if all8["lo"] > 0 and lin_point > 0:
        return "BETTER"
    if all8["hi"] < 0 and lin_point < 0:
        return "WORSE"
    if dk is not None and dk["lo"] >= -DELTA and dk["hi"] <= DELTA:
        return "PARITY"
    if all8["lo"] > 0:
        return "INCONCLUSIVE (attention BETTER, linear disagrees)"
    if all8["hi"] < 0:
        return "INCONCLUSIVE (attention WORSE, linear disagrees)"
    return "INCONCLUSIVE"


def main():
    t0 = time.time()
    ch = json.load(open(f"{L.OUT}/chains.json"))
    rj = json.load(open(f"{L.OUT}/runs.json"))
    oj = json.load(open(f"{L.OUT}/openset.json"))
    sha_ch = L31.sha(f"{L.OUT}/chains.json")
    assert rj["chains_sha256"] == sha_ch == oj["chains_sha256"]
    ms = ch["models"]
    D = Data()
    B = A.Boot("v2")
    assert B.S.shape == (NB, 24, 5)
    Bi = copy.copy(B)
    Bi.S = np.random.default_rng(IND_SEED).integers(0, 5, (NB, 24, 5))
    orders = {a: {k: ch["train_order"][a][str(k)] for k in KS} for a in ARMS}
    differ = lambda a, b: [k for k in KS if orders[a][k] != orders[b][k]]

    # ---------------------------------------------------------------- open-set units
    BO = OA.Boot("v2")
    assert np.array_equal(BO.M, B.M), "open-set config draws differ from D031's"
    oc = dict(np.load(f"{L.OUT}/openset_counts.npz"))
    c33 = np.load("./results/D033/counts_v2.npz")
    ocache = {}

    def ounit(arm, k):
        if (arm, k) not in ocache:
            if arm == "CLOUD":
                b = f"nc|model|JG|raw|{k}"
                ocache[(arm, k)] = OA.Unit(c33[f"{b}|cnt_model"][None], c33[f"{b}|cnt_pair_all"][None])
            else:
                b = f"nc|{arm}|{k}"
                ocache[(arm, k)] = OA.Unit(oc[f"{b}|cnt_model"][None], oc[f"{b}|cnt_pair_all"][None])
        return ocache[(arm, k)]

    # ---------------------------------------------------------------- (b, I1) tail-plot inputs, before labels
    pc = {a: A.Arm(a, D.unit("att", a, 8)).sorted_curve() for a in ARMS}
    po = {a: np.sort(ounit(a, 8).stats(BO)["pa"][1]) for a in FOLD_ARMS}
    np.savez_compressed(f"{L.OUT}/plotdata.npz", **{f"tail|{a}": v for a, v in pc.items()},
                        **{f"otail|{a}": v for a, v in po.items()},
                        **{f"curve|{ro}|{a}": np.array([D.unit(ro, a, k).e_all(B)[1] for k in KS])
                           for ro in ("att", "lin") for a in ARMS},
                        **{f"ocurve|{e}|{a}": np.array([ounit(a, k).stats(BO)[e][1] for k in KS])
                           for e in ("all", "tail") for a in FOLD_ARMS})
    print("[S4] plotdata.npz written (I1 tail plot inputs, closed- and open-set)", flush=True)

    # ---------------------------------------------------------------- closed-set machinery
    def e_all_k(ro, a, k, Bx=B):
        return D.unit(ro, a, k).e_all(Bx)

    def mean_vec(ro, a, b, ks, Bb=B):
        d = np.mean([e_all_k(ro, a, k)[0] - e_all_k(ro, b, k, Bb)[0] for k in ks], 0)
        p = float(np.mean([e_all_k(ro, a, k)[1] - e_all_k(ro, b, k)[1] for k in ks]))
        return d, p

    def mean_k(ro, a, b, ks, Bb=B):
        if not ks:
            return None
        d, p = mean_vec(ro, a, b, ks, Bb)
        return dict(point=A.r4(p), **pct(d), ks=ks)

    def per_k(a, b):
        rows, dk = {}, differ(a, b)
        for k in KS:
            (da, pa), (db, pb) = e_all_k("att", a, k), e_all_k("att", b, k)
            (la, lpa), (lb, lpb) = e_all_k("lin", a, k), e_all_k("lin", b, k)
            rows[k] = dict(a=A.r4(pa), b=A.r4(pb), delta=A.r4(pa - pb), hierarchical=pct(da - db),
                           identical_input=k not in dk, linear=dict(a=A.r4(lpa), b=A.r4(lpb), delta=A.r4(lpa - lpb),
                                                                    **pct(la - lb)))
        return rows

    def k8(a, b):
        ad = lambda x: (x[0], x[1], None)
        out = {}
        for ro in ("att", "lin"):
            ua, ub = D.unit(ro, a, 8), D.unit(ro, b, 8)
            out[ro] = dict(E_all=A.contrast(ad(ua.e_all(B)), ad(ub.e_all(B))),
                           E_tail=A.contrast(ad(ua.e_tail(B)), ad(ub.e_tail(B))))
        out["label"] = A.label(out["att"]["E_all"], out["att"]["E_tail"],
                               out["lin"]["E_all"]["delta"], out["lin"]["E_tail"]["delta"])
        return out

    def compare(a, b, with_verdict):
        dk = differ(a, b)
        all8, dmk = mean_k("att", a, b, KS), mean_k("att", a, b, dk)
        ind8, indd = mean_k("att", a, b, KS, Bi), mean_k("att", a, b, dk, Bi)
        lin8 = mean_k("lin", a, b, KS)
        out = dict(a=a, b=b, differing_k=dk, mean_k1_8=all8, mean_differing_k=dmk, linear_mean_k1_8=lin8,
                   independent_seed=dict(mean_k1_8=ind8, mean_differing_k=indd), per_k=per_k(a, b), k8=k8(a, b))
        if not dk:
            out["verdict"] = "SAME CHAIN"                                        # Call 2 (a)
        elif with_verdict:
            out["verdict"] = verdict(all8, dmk, lin8["point"])
            out["independent_seed"]["verdict"] = verdict(ind8, indd, lin8["point"])
            out["schemes_agree"] = out["verdict"] == out["independent_seed"]["verdict"]
        return out

    # ---------------------------------------------------------------- WORSE scan (every ordered contrast)
    scan = {}
    for a, b in itertools.permutations(ARMS, 2):
        v = compare(a, b, True)
        scan[f"{a} - {b}"] = dict(mean_k_verdict=v.get("verdict"), k8_label=v["k8"]["label"],
                                  mean_k1_8=v["mean_k1_8"], linear_mean_k1_8=v["linear_mean_k1_8"])
    worse = [n for n, v in scan.items() if v["mean_k_verdict"] == "WORSE" or v["k8_label"] == "WORSE"
             or (v["mean_k_verdict"] or "").startswith("INCONCLUSIVE (attention WORSE")]
    print(f"[S4] WORSE scan: {worse or 'none'}", flush=True)

    prim = {"P1": compare("CLOUD-raw", "CENTROID-raw", True), "P2": compare("CLOUD-raw", "CLOUD", True)}
    rows = {"CLOUD - CENTROID-raw (D035 P1 restated)": compare("CLOUD", "CENTROID-raw", False),
            "CLOUD-raw - PAPER8": compare("CLOUD-raw", "PAPER8", False)}
    for n, v in {**prim, **rows}.items():
        print(f"[S4] {n} ({v['a']} - {v['b']}): {v.get('verdict', '(no verdict)')}; mean k1-8 {v['mean_k1_8']}; "
              f"differing k {v['mean_differing_k']}; linear {v['linear_mean_k1_8']}; ind {v['independent_seed']}; "
              f"k=8 label {v['k8']['label']}", flush=True)
        for k, r in v["per_k"].items():
            print(f"     k={k}: {r['a']} vs {r['b']} d {r['delta']:+.4f} {r['hierarchical']} ident {r['identical_input']}"
                  f" lin {r['linear']}", flush=True)

    # ---------------------------------------------------------------- (d) triangle
    d1, p1 = mean_vec("att", "CLOUD-raw", "CENTROID-raw", KS)
    d2, p2 = mean_vec("att", "CLOUD-raw", "CLOUD", KS)
    d3, p3 = mean_vec("att", "CLOUD", "CENTROID-raw", KS)
    tri_max = float(np.abs((d1 - d2) - d3).max())
    l1, lp1 = mean_vec("lin", "CLOUD-raw", "CENTROID-raw", KS)
    l2, lp2 = mean_vec("lin", "CLOUD-raw", "CLOUD", KS)
    l3, lp3 = mean_vec("lin", "CLOUD", "CENTROID-raw", KS)
    tri_lin = float(np.abs((l1 - l2) - l3).max())
    pub = json.load(open("./results/D035/analysis.json"))["primary"]["P1"]["mean_k1_8"]
    mine = dict(point=A.r4(p3), **pct(d3))
    tri_ok = tri_max <= 1e-12 and tri_lin <= 1e-12 and all(mine[x] == pub[x] for x in ("point", "lo", "hi"))
    assert tri_ok, (tri_max, tri_lin, mine, pub)
    tri = dict(max_abs_replicate_diff_att=tri_max, max_abs_replicate_diff_lin=tri_lin,
               P1_minus_P2_point=A.r4(p1 - p2), CLOUD_minus_CENTROID_raw=mine, D035_published_P1=pub, pass_=tri_ok)
    print(f"[S4] (d) triangle: max |(P1 - P2) - D035 P1| per replicate att {tri_max:.1e}, lin {tri_lin:.1e}; "
          f"{mine} vs D035 {pub} -> PASS", flush=True)

    # ---------------------------------------------------------------- open-set O1 / O2 (D033 label)
    lv = {a: OA.level(ounit(a, 8), BO) for a in FOLD_ARMS}
    d33 = json.load(open("./results/D033/analysis.json"))["views"]["v2"]["levels"]["nc|model|JG"]
    assert all(lv["CLOUD"][x] == d33[x] for x in ("O_all", "O_tail")), (lv["CLOUD"], d33)
    fs = ch["fold_same_chain"]
    opn = {}
    for n, (a, b) in ch["fold_comparisons"].items():
        c = OA.contrast(ounit(a, 8), ounit(b, 8), BO)
        if fs[n]["same_chain_all_folds"]:
            c["label"] = "SAME CHAIN"                                            # Call 2 (a), extended
        c["folds_differing_k8"] = 5 - fs[n]["folds_same_set_k8"]
        c["per_k"] = {k: OA.contrast(ounit(a, k), ounit(b, k), BO) for k in KS}
        opn[n] = dict(a=a, b=b, **c)
        print(f"[S4] open-set {n} ({a} - {b}) k=8: {c['label']}; O_all {c['d_all']}; O_tail {c['d_tail']}; "
              f"folds differing {c['folds_differing_k8']}", flush=True)
    print(f"[S4] open-set levels k=8: {lv} (CLOUD^fold = D033 published)", flush=True)

    # ---------------------------------------------------------------- (b) hard cases at k = 8
    pid = {f"{a} | {b}": j for j, (a, b) in enumerate(itertools.combinations(ms, 2))}
    tg = L29.targets(ms)
    npr = [pid[p["pair"]] for p in json.load(open("./results/D027/n_prime.json"))["pairs"]]
    assert len(npr) == 161
    rest = sorted(set(range(len(pid))) - set(npr))
    sets = dict(H_all=[pid[p] for p in tg["strata"]["H_all"]], stably_hard=[pid[p] for p in tg["strata"]["stably_hard"]],
                N_prime=npr, not_N_prime=rest)
    hard = {}
    for a in ARMS:
        u = D.unit("att", a, 8)
        pa = u.pair.sum(2).mean(0) / (2 * NC)
        worst = np.argsort(pa, kind="stable")[:int(np.floor(L.GAMMA * len(pa)))]
        hard[a] = dict({s: A.r4(u.e_hard(B, c)[1]) for s, c in sets.items()},
                       E_tail=A.r4(cvar(pa, L.GAMMA)), N_prime_share_of_worst_decile=A.r4(np.isin(worst, npr).mean()))
    hard_d = {n: {s: A.contrast((*D.unit("att", a, 8).e_hard(B, c), None), (*D.unit("att", b, 8).e_hard(B, c), None))
                  for s, c in sets.items()}
              for n, (a, b) in (("P1", ("CLOUD-raw", "CENTROID-raw")), ("P2", ("CLOUD-raw", "CLOUD")))}
    ohard = {a: {s: A.r4(ounit(a, 8).sub(BO, c)[1]) for s, c in sets.items()} for a in FOLD_ARMS}
    ohard_d = {n: {s: dict(delta=A.r4(ounit(a, 8).sub(BO, c)[1] - ounit(b, 8).sub(BO, c)[1]),
                           ci=A.ci(ounit(a, 8).sub(BO, c)[0] - ounit(b, 8).sub(BO, c)[0])) for s, c in sets.items()}
               for n, (a, b) in ch["fold_comparisons"].items()}
    print(f"[S4] (b) hard k=8 closed: {hard}; open: {ohard}", flush=True)

    out = dict(
        schema="d036-analysis-v1", chains_sha256=sha_ch,
        verdicts={n: v["verdict"] for n, v in prim.items()},
        open_set_labels={n: v["label"] for n, v in opn.items()},
        verdict_rule="BETTER lo(mean k1-8) > 0 and linear mean-k point > 0; WORSE hi < 0 and linear mean-k point < 0; "
                     "PARITY interval of the mean over that comparison's differing k within +-0.02; else "
                     "INCONCLUSIVE (noted when attention alone resolves); identical sets at every k -> SAME CHAIN",
        bootstrap=dict(configs="boot_draws(25, 2000, 20261005)", configs_sha256=sha_arr(B.M),
                       seeds="default_rng(20261005).integers(0, 5, (2000, 24, 5)), slot k - 1, shared by all arms",
                       independent_seed_arm2=f"default_rng({IND_SEED})", linear="configs only",
                       open_set="d033_analysis.Boot('v2'): configs only (asserted = D031 draws), folds fixed",
                       open_set_configs_sha256=sha_arr(BO.M)),
        worse_scan=dict(worse=worse, rows=scan), primary=prim, unlabelled=rows, triangle=tri,
        open_set=dict(levels_k8=lv, contrasts=opn, cloud_level_equals_d033=True, fold_same_chain=fs),
        secondary_b_hard_k8=dict(per_arm=hard, differences=hard_d, open_set_per_arm=ohard, open_set_differences=ohard_d,
                                 n={s: len(c) for s, c in sets.items()}),
        secondary_c_chains=dict(chains=ch["chains"], first_step=ch["first_step"], overlaps_k8=ch["overlaps_k8"],
                                block_233_257=ch["block_233_257"], cross_objective=ch["cross_objective"],
                                fold_chains={F["fold"]: F["arms"] for F in ch["folds"]}, d034_union=ch["d034_union"]),
        reused_cells="CLOUD / PAPER8 = D029, CENTROID-raw = D035, CLOUD^fold = D033 (I6); CLOUD-raw trained here",
        code_commit=L31.git_head(), wall_min=round((time.time() - t0) / 60, 1))
    L31.jdump(out, f"{L.OUT}/analysis.json")
    print(f"[S4] verdicts {out['verdicts']}; open-set {out['open_set_labels']}; wrote analysis.json "
          f"({out['wall_min']} min)", flush=True)


if __name__ == "__main__":
    main()
