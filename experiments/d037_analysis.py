"""D037 / S3b — labels and secondaries (Review D037/P1: APPROVED WITH AMENDMENTS; A1, A2; D3 cross-check kept).
Submitted only after the S3a tail plot (results/D037/tail_k8.png) has been viewed.

Bootstrap: d031_analysis.Boot("v2") (configs boot_draws(25, 2000, 20261005); attention seeds slot k - 1 of
default_rng(20261005), shared by every arm; linear: configs only); independent-seed row: the second arm's seeds from
default_rng(20261101) (D035 / D036). Cells: JG / GC / PAPER8 = D029 (all-pair from stored logits), TOP-K[CVaR] /
MMR-PP[CVaR] = D031, TOP-K[MEAN] = D032, JG-mean / GC-mean = results/D037 (d037_lib.Data).

  primary k = 8      C1 = JG - JG-mean, C2 = JG - GC ("seen (D029 / D031)"), label37
  WORSE scan         56 ordered contrasts among the 8 arms at k = 8, label37; D031-rule (asymmetric) hits listed
  secondary          C3 = GC - GC-mean, C4 = JG-mean - GC-mean (label37); mean over k = 1..8, D036's verdict table,
                     PARITY on the differing k (A2: set equality); independent-seed rows for C1, C2
  A2                 order-only rows: same set, different training order (C2 k = 3 expected)
  descriptive        interaction C1 - C3 at k = 8; (a) curves + queries to reach JG's k = 4 / 8 E_all; (b) hard sets;
                     (c) chains; (d) v1 side-by-side (D012 / D013, different protocol era)
Writes results/D037/analysis.json.
Usage:  PYTHONPATH=.:experiments python experiments/d037_analysis.py
"""
import copy
import json
import time
import hashlib
import itertools

import numpy as np

import d031_lib as L31
import d031_analysis as A
import d029_lib as L29
import d037_lib as L
from LLMmap.joint_statistic import cvar

KS = list(range(1, L.K + 1))
NB, NC, DELTA = A.NB, A.NC, A.DELTA
IND_SEED = 20261101
ARMS8 = L.ARMS8
SEEN = "seen (D029 / D031)"


def pct(x):
    return dict(lo=A.r4(np.percentile(x, 2.5)), hi=A.r4(np.percentile(x, 97.5)))


def sha_arr(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def label37(d_all, d_tail, lin_all, lin_tail):
    """The D's table: D031 BETTER; PARITY; WORSE with the symmetric linear clause on that endpoint."""
    w_all, w_tail = d_all["ci"][1] < 0, d_tail["ci"][1] < 0
    if (w_all and lin_all < 0) or (w_tail and lin_tail < 0):
        return "WORSE"
    if d_all["ci"][0] > 0 and d_tail["ci"][0] > 0 and lin_all > 0 and lin_tail > 0:
        return "BETTER"
    if all(-DELTA <= x <= DELTA for x in d_all["ci"] + d_tail["ci"]):
        return "PARITY"
    if d_all["ci"][0] > 0 and d_tail["ci"][0] > 0:
        return "INCONCLUSIVE (attention BETTER, linear disagrees)"
    if w_all or w_tail:
        return "INCONCLUSIVE (attention WORSE, linear disagrees)"
    return "INCONCLUSIVE"


def verdict(all8, dk, lin_point):
    """D036's mean-over-k table (symmetric linear clause)."""
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
    sha_ch = L31.sha(f"{L.OUT}/chains.json")
    assert rj["chains_sha256"] == sha_ch
    ms, chains = ch["models"], ch["chains"]
    D = L.Data(ch["mmr_reference"])
    B = A.Boot("v2")
    assert B.S.shape == (NB, 24, 5)
    s0 = json.load(open(f"{L.OUT}/s0.json"))
    assert sha_arr(B.M) == s0["boot_configs_sha256"]
    Bi = copy.copy(B)
    Bi.S = np.random.default_rng(IND_SEED).integers(0, 5, (NB, 24, 5))
    same_set = lambda a, b, k: set(chains[a][:k]) == set(chains[b][:k])
    differ = lambda a, b: [k for k in KS if not same_set(a, b, k)]
    xcheck = dict(n=0, asym_d031_worse=[])

    # ---------------------------------------------------------------- k = 8 cells
    def k8(a, b):
        ad = lambda x: (x[0], x[1], None)
        out = {}
        for ro in ("att", "lin"):
            ua, ub = D.unit(ro, a, 8), D.unit(ro, b, 8)
            out[ro] = dict(E_all=A.contrast(ad(ua.e_all(B)), ad(ub.e_all(B))),
                           E_tail=A.contrast(ad(ua.e_tail(B)), ad(ub.e_tail(B))))
        args = (out["att"]["E_all"], out["att"]["E_tail"], out["lin"]["E_all"]["delta"], out["lin"]["E_tail"]["delta"])
        out["label"], out["label_d031_rule"] = label37(*args), A.label(*args)
        xcheck["n"] += 1                                                    # D3: label37 = D031 label off the
        if out["label_d031_rule"] == "WORSE" and out["label"] != "WORSE":  # asymmetric-WORSE cells
            xcheck["asym_d031_worse"].append(f"{a} - {b}")
        else:
            assert out["label"].split(" (")[0] == out["label_d031_rule"], (a, b, out)
        out["same_set_k8"] = same_set(a, b, 8)
        if {a, b} == {"JG", "GC"}:
            out["note"] = SEEN
        return out

    # ---------------------------------------------------------------- mean over k (D036 table)
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
        rows = {}
        for k in KS:
            (da, pa), (db, pb) = e_all_k("att", a, k), e_all_k("att", b, k)
            (la, lpa), (lb, lpb) = e_all_k("lin", a, k), e_all_k("lin", b, k)
            rows[k] = dict(a=A.r4(pa), b=A.r4(pb), delta=A.r4(pa - pb), hierarchical=pct(da - db),
                           identical_input=same_set(a, b, k),
                           linear=dict(a=A.r4(lpa), b=A.r4(lpb), delta=A.r4(lpa - lpb), **pct(la - lb)))
        return rows

    def mean_over_k(a, b, ind):
        dk = differ(a, b)
        all8, dmk, lin8 = mean_k("att", a, b, KS), mean_k("att", a, b, dk), mean_k("lin", a, b, KS)
        out = dict(differing_k=dk, mean_k1_8=all8, mean_differing_k=dmk, linear_mean_k1_8=lin8)
        out["verdict"] = "SAME CHAIN" if not dk else verdict(all8, dmk, lin8["point"])
        if ind:
            ind8, indd = mean_k("att", a, b, KS, Bi), mean_k("att", a, b, dk, Bi)
            out["independent_seed"] = dict(mean_k1_8=ind8, mean_differing_k=indd,
                                           verdict="SAME CHAIN" if not dk else verdict(ind8, indd, lin8["point"]))
            out["schemes_agree"] = out["verdict"] == out["independent_seed"]["verdict"]
        return out

    # ---------------------------------------------------------------- C1-C4
    con = {}
    for n, (a, b) in L.CONTRASTS.items():
        con[n] = dict(a=a, b=b, role="primary" if n in ("C1", "C2") else "labelled secondary",
                      k8=k8(a, b), mean_over_k=mean_over_k(a, b, n in ("C1", "C2")), per_k=per_k(a, b))
        if n == "C2":
            con[n]["note"] = SEEN
        v = con[n]
        print(f"[S3b] {n} ({a} - {b}){' [' + SEEN + ']' if n == 'C2' else ''}: k=8 label {v['k8']['label']} "
              f"(D031 rule {v['k8']['label_d031_rule']}); att E_all {v['k8']['att']['E_all']} E_tail "
              f"{v['k8']['att']['E_tail']}; lin E_all {v['k8']['lin']['E_all']['delta']} E_tail "
              f"{v['k8']['lin']['E_tail']['delta']}", flush=True)
        m = v["mean_over_k"]
        print(f"      mean k1-8: {m['verdict']}; {m['mean_k1_8']}; differing k {m['mean_differing_k']}; linear "
              f"{m['linear_mean_k1_8']}" + (f"; independent seed {m['independent_seed']}" if "independent_seed" in m
                                            else ""), flush=True)

    # ---------------------------------------------------------------- WORSE scan (56 ordered contrasts, k = 8)
    scan = {}
    for a, b in itertools.permutations(ARMS8, 2):
        c = con.get(next((n for n, v in L.CONTRASTS.items() if v == (a, b)), None), {}).get("k8") or k8(a, b)
        scan[f"{a} - {b}"] = dict(label=c["label"], label_d031_rule=c["label_d031_rule"],
                                  E_all=c["att"]["E_all"], E_tail=c["att"]["E_tail"],
                                  lin=dict(E_all=c["lin"]["E_all"]["delta"], E_tail=c["lin"]["E_tail"]["delta"]),
                                  **({"note": SEEN} if {a, b} == {"JG", "GC"} else {}))
    assert len(scan) == 56
    worse = [n for n, v in scan.items() if v["label"] == "WORSE"]
    worse_d031 = [n for n, v in scan.items() if v["label_d031_rule"] == "WORSE"]
    print(f"[S3b] WORSE scan (label37): {worse or 'none'}", flush=True)
    print(f"[S3b] WORSE under D031's asymmetric rule: {worse_d031 or 'none'}", flush=True)

    # ---------------------------------------------------------------- A2 order-only rows
    order_only = {}
    for n, ks in ch["order_only_k"].items():
        a, b = ch["comparisons"][n]
        for k in ks:
            r = {}
            for ro in ("att", "lin"):
                (da, pa), (db, pb) = e_all_k(ro, a, k), e_all_k(ro, b, k)
                ea, eb = D.unit(ro, a, k).e_tail(B), D.unit(ro, b, k).e_tail(B)
                r[ro] = dict(E_all=dict(delta=A.r4(pa - pb), **pct(da - db)),
                             E_tail=dict(delta=A.r4(ea[1] - eb[1]), **pct(ea[0] - eb[0])))
            order_only[f"{n} k={k}"] = dict(a=a, b=b, k=k, set=sorted(chains[a][:k]),
                                            order_a=ch["train_order"].get(a, {}).get(str(k)),
                                            order_b=ch["train_order"].get(b, {}).get(str(k)), **r)
    print(f"[S3b] A2 order-only rows: {order_only}", flush=True)

    # ---------------------------------------------------------------- interaction C1 - C3 at k = 8
    inter = {}
    for ro in ("att", "lin"):
        u = {a: D.unit(ro, a, 8) for a in L.GRID}
        r = {}
        for e, fn in (("E_all", lambda x: x.e_all(B)), ("E_tail", lambda x: x.e_tail(B))):
            v = {a: fn(x) for a, x in u.items()}
            d = (v["JG"][0] - v["JG-mean"][0]) - (v["GC"][0] - v["GC-mean"][0])
            p = (v["JG"][1] - v["JG-mean"][1]) - (v["GC"][1] - v["GC-mean"][1])
            r[e] = dict(point=A.r4(p), **pct(d))
        inter[ro] = r
    print(f"[S3b] interaction (C1 - C3) k=8, no label: {inter}", flush=True)

    # ---------------------------------------------------------------- (a) curves + queries to reach
    curves = {ro: {a: dict(E_all=[A.r4(D.unit(ro, a, k).e_all(B)[1]) for k in KS],
                           E_tail=[A.r4(D.unit(ro, a, k).e_tail(B)[1]) for k in KS]) for a in ARMS8}
              for ro in ("att", "lin")}
    reach = {}
    for ro in ("att", "lin"):
        reach[ro] = {}
        for kt in (4, 8):
            tgt = D.unit(ro, "JG", kt).e_all(B)[1]
            reach[ro][f"JG_k{kt}"] = {a: next((k for k in KS if D.unit(ro, a, k).e_all(B)[1] >= tgt), None)
                                      for a in ARMS8}
    print(f"[S3b] (a) queries to reach JG's k=4 / 8 E_all: {reach}", flush=True)

    # ---------------------------------------------------------------- (b) hard sets at k = 8
    pid = {f"{a} | {b}": j for j, (a, b) in enumerate(itertools.combinations(ms, 2))}
    tg = L29.targets(ms)
    npr = [pid[p["pair"]] for p in json.load(open("./results/D027/n_prime.json"))["pairs"]]
    assert len(npr) == 161
    sets = dict(H_all=[pid[p] for p in tg["strata"]["H_all"]], stably_hard=[pid[p] for p in tg["strata"]["stably_hard"]],
                N_prime=npr, not_N_prime=sorted(set(range(len(pid))) - set(npr)))
    assert (len(sets["H_all"]), len(sets["stably_hard"])) == (57, 26)
    hard = {}
    for a in ARMS8:
        u = D.unit("att", a, 8)
        pa = u.pair.sum(2).mean(0) / (2 * NC)
        hard[a] = dict({s: A.r4(u.e_hard(B, c)[1]) for s, c in sets.items()}, E_tail=A.r4(cvar(pa, L31.GAMMA)))
    hard_d = {n: {s: A.contrast((*D.unit("att", a, 8).e_hard(B, c), None), (*D.unit("att", b, 8).e_hard(B, c), None))
                  for s, c in sets.items()} for n, (a, b) in L.CONTRASTS.items()}
    print(f"[S3b] (b) hard k=8: {hard}", flush=True)

    # ---------------------------------------------------------------- (d) v1 side-by-side + 2x2 table
    m12 = json.load(open("./results/D012/metrics_by_gamma_k.json"))["metrics"]
    m13 = json.load(open("./results/D013/metrics_by_gamma_k.json"))["metrics"]
    v1 = {"JG": m12["0.1"]["8"]["mean_top1"], "JG-mean": m12["1.0"]["8"]["mean_top1"],
          "GC": m13["0.1"]["8"]["mean_top1"], "GC-mean": m13["1.0"]["8"]["mean_top1"]}
    grid = {a: {ro: dict(E_all=A.r4(D.unit(ro, a, 8).e_all(B)[1]), E_tail=A.r4(D.unit(ro, a, 8).e_tail(B)[1]))
                for ro in ("att", "lin")} for a in L.GRID}
    print(f"[S3b] 2x2 k=8: {grid}; v1 (D012 / D013, different protocol era) mean top-1: {v1}", flush=True)

    out = dict(
        schema="d037-analysis-v1", chains_sha256=sha_ch,
        labels_k8={n: v["k8"]["label"] + (f" [{SEEN}]" if n == "C2" else "") for n, v in con.items()},
        mean_over_k_verdicts={n: v["mean_over_k"]["verdict"] for n, v in con.items()},
        label_rule="label37: BETTER lo(all) > 0, lo(tail) > 0, linear all > 0, tail > 0; WORSE (hi(all) < 0 and "
                   "linear all < 0) or (hi(tail) < 0 and linear tail < 0); PARITY both attention intervals within "
                   "+-0.02; else INCONCLUSIVE (annotated when attention alone resolves)",
        label_crosscheck=dict(cells=xcheck["n"], equal_to_d031_label_except=xcheck["asym_d031_worse"]),
        mean_over_k_rule="D036: BETTER lo(mean k1-8) > 0 and linear point > 0; WORSE hi < 0 and linear point < 0; "
                         "PARITY interval of the mean over differing k (set equality, A2) within +-0.02; SAME CHAIN",
        bootstrap=dict(configs="boot_draws(25, 2000, 20261005)", configs_sha256=sha_arr(B.M),
                       seeds="default_rng(20261005).integers(0, 5, (2000, 24, 5)), slot k - 1, shared by all arms",
                       independent_seed_arm2=f"default_rng({IND_SEED})", linear="configs only"),
        contrasts=con, worse_scan=dict(worse=worse, worse_d031_asymmetric_rule=worse_d031, rows=scan),
        order_only_rows=order_only, interaction_C1_minus_C3_k8=inter,
        secondary_a=dict(curves=curves, queries_to_reach=reach,
                         rule="smallest k with point E_all >= JG's at k = 4 / 8, same readout (D032)"),
        secondary_b_hard_k8=dict(per_arm=hard, differences=hard_d, n={s: len(c) for s, c in sets.items()}),
        secondary_c_chains=dict(chains=chains, first_query=ch["first_query"], objective_trend=ch["objective_trend"],
                                overlaps_k8=ch["overlaps_k8"], within_set_sim_k8=ch["within_set_sim_k8"],
                                d034_union=ch["d034_union"], multi_match_k=ch["multi_match_k"],
                                train_order_tag=ch["train_order_tag"]),
        grid_2x2_k8=grid,
        secondary_d_v1=dict(mean_top1_k8=v1, label="v1, D012 / D013, different protocol era",
                            sources=["results/D012/metrics_by_gamma_k.json (JG, gamma 0.1 / 1.0)",
                                     "results/D013/metrics_by_gamma_k.json (GC, gamma 0.1 / 1.0)"],
                            note="as published; never pooled with v2, no interval"),
        reused_cells="JG / GC / PAPER8 = D029, TOP-K[CVaR] / MMR-PP[CVaR] = D031, TOP-K[MEAN] = D032 (I6); "
                     "JG-mean / GC-mean trained here",
        code_commit=L31.git_head(), wall_min=round((time.time() - t0) / 60, 1))
    L31.jdump(out, f"{L.OUT}/analysis.json")
    print(f"[S3b] labels k=8 {out['labels_k8']}; mean-over-k {out['mean_over_k_verdicts']}; wrote analysis.json "
          f"({out['wall_min']} min)", flush=True)


if __name__ == "__main__":
    main()
