"""D038 / S3b — paired summaries and the pre-registered readings (after the S3a tail plots were viewed). Plan:
docs/plans/D038-P1.md §8 + Review A1 / A2. Reads results/D038/measures_<view>.json (S2) only; no S_val / S_test.

Summary rule, every paired quantity d over the 40 directed splits: median, band = 2.5-97.5 percentile (np.percentile
linear), n(d > 0), n(d < 0), n(d = 0), n undefined (A1 only). "Consistent" = >= 36 of 40 with the stated sign; zeros
and undefined values never count toward it. Bands are not confidence intervals (splits share configs).
  anchor (M4)    P_all(MMR*) - P_all(JG), k = 8; v2 median <= 0 -> readings "S_build-internal only".
  H-strength     mean s_q(m): MMR* - JG; v2 n(>0) >= 36; v1 control |median_v1| < |median_v2|.
  H-redundancy   eff_n(m): JG - MMR*; v2 n(<0) >= 36.
  H-chase        (i) Y_joint optimism JG - MMR*, (ii) JG - TOP-K[CVaR], both n(>0) >= 36 at k = 8;
                 (iii') A1 delta(JG) - delta(MMR*), n(>0) >= 36; not evaluable if > 4 splits undefined (then H-chase is
                 decided on (i), (ii) and labelled partial); v1 control median_v1(i) < median_v2(i).
  A2             held-out Y_joint(m) JG - MMR* at k = 8: (alpha) n(>0) >= 36, (beta) n(<0) >= 36, (gamma) neither.
                 Also vs TOP-K[CVaR] and the Y_max versions (descriptive).
Writes results/D038/analysis.json. No BETTER / WORSE label anywhere (the D).
Usage:  PYTHONPATH=.:experiments python experiments/d038_analysis.py
"""
import json

import numpy as np

import d031_lib as L31

OUT = "./results/D038"
VIEWS = ("v2", "v1")
RULES = ("JG", "MMR*", "TOP-K[CVaR]", "GC", "PAPER8")
CONS = 36
K8 = 7


def summ(v):
    d = np.array([x for x in v if x is not None], np.float64)
    out = dict(n=len(v), n_undef=len(v) - len(d), n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()),
               n_zero=int((d == 0).sum()))
    if len(d):
        out.update(median=float(np.median(d)), band=[float(x) for x in np.percentile(d, [2.5, 97.5])])
    else:
        out.update(median=None, band=None)
    return out


def sub(a, b):
    return None if a is None or b is None else a - b


def view_analysis(view):
    M = json.load(open(f"{OUT}/measures_{view}.json"))
    S = M["splits"]
    assert len(S) == 40 and M["s_val_reads"] == 0 and M["s_test_reads"] == 0
    sp = json.load(open(f"{OUT}/splits_{view}.json"))

    def per(f):
        return [f(x) for x in S]

    def opt(x, rule, y="yj", k=K8):
        return x["M3"][rule][f"{y}_s"][k] - x["M3"][rule][f"{y}_m"][k]

    def ho(x, rule, y="yj", k=K8):
        return x["M3"][rule][f"{y}_m"][k]

    def pall(x, rule, k=K8):
        return x["M4"][rule][k][0]

    def ptail(x, rule, k=K8):
        return x["M4"][rule][k][1]

    rec = {}
    # ---- anchor
    anchor = summ(per(lambda x: pall(x, "MMR*") - pall(x, "JG")))
    rec["anchor"] = dict(d_P_all_MMR_minus_JG_k8=anchor,
                         d_P_tail_MMR_minus_JG_k8=summ(per(lambda x: ptail(x, "MMR*") - ptail(x, "JG"))))

    # ---- readings
    hs = summ(per(lambda x: x["M1"]["MMR*"]["mean_s_m"] - x["M1"]["JG"]["mean_s_m"]))
    hr = summ(per(lambda x: x["M2"]["JG"]["m"]["eff_n"] - x["M2"]["MMR*"]["m"]["eff_n"]))
    c1 = summ(per(lambda x: opt(x, "JG") - opt(x, "MMR*")))
    c2 = summ(per(lambda x: opt(x, "JG") - opt(x, "TOP-K[CVaR]")))
    c3 = summ(per(lambda x: sub(x["chase"]["JG"]["delta"], x["chase"]["MMR*"]["delta"])))
    c3_eval = c3["n_undef"] <= 4
    rec["readings"] = {
        "H-strength": dict(d=hs, consistent=hs["n_pos"] >= CONS),
        "H-redundancy": dict(d=hr, consistent=hr["n_neg"] >= CONS),
        "H-chase": dict(i=dict(d=c1, consistent=c1["n_pos"] >= CONS),
                        ii=dict(d=c2, consistent=c2["n_pos"] >= CONS),
                        iii_prime=dict(d=c3, evaluable=c3_eval, consistent=c3_eval and c3["n_pos"] >= CONS))}

    # ---- A2
    a2 = summ(per(lambda x: ho(x, "JG") - ho(x, "MMR*")))
    case = "alpha" if a2["n_pos"] >= CONS else "beta" if a2["n_neg"] >= CONS else "gamma"
    rec["A2"] = dict(case=case, d_ho_yj_JG_minus_MMR=a2,
                     d_ho_yj_JG_minus_TOPK=summ(per(lambda x: ho(x, "JG") - ho(x, "TOP-K[CVaR]"))),
                     d_ho_ymax_JG_minus_MMR=summ(per(lambda x: ho(x, "JG", "ymax") - ho(x, "MMR*", "ymax"))),
                     d_ho_ymax_JG_minus_TOPK=summ(per(lambda x: ho(x, "JG", "ymax") - ho(x, "TOP-K[CVaR]", "ymax"))),
                     d_ho_yj_JG_minus_GC=summ(per(lambda x: ho(x, "JG") - ho(x, "GC"))))

    # ---- descriptive
    D = {}
    D["levels"] = {r: dict(opt_yj=summ(per(lambda x: opt(x, r))), opt_ymax=summ(per(lambda x: opt(x, r, "ymax"))),
                           yj_s=summ(per(lambda x: x["M3"][r]["yj_s"][K8])), yj_m=summ(per(lambda x: ho(x, r))),
                           ymax_s=summ(per(lambda x: x["M3"][r]["ymax_s"][K8])),
                           ymax_m=summ(per(lambda x: ho(x, r, "ymax"))),
                           P_all=summ(per(lambda x: pall(x, r))), P_tail=summ(per(lambda x: ptail(x, r))),
                           mean_s_m=summ(per(lambda x: x["M1"][r]["mean_s_m"])),
                           mean_s_s=summ(per(lambda x: x["M1"][r]["mean_s_s"])),
                           mean_rank_m=summ(per(lambda x: x["M1"][r]["mean_rank_m"])),
                           eff_n_m=summ(per(lambda x: x["M2"][r]["m"]["eff_n"])),
                           within_pearson_m=summ(per(lambda x: x["M2"][r]["m"]["within_pearson"])),
                           within_cos_m=summ(per(lambda x: x["M2"][r]["m"]["within_cos"])))
                    for r in RULES}
    D["paper8_optimism_by_direction"] = {
        f"d{d}": dict(opt_yj=summ([opt(x, "PAPER8") for x in S if x["d"] == d]),
                      opt_ymax=summ([opt(x, "PAPER8", "ymax") for x in S if x["d"] == d])) for d in (0, 1)}
    D["optimism_ymax"] = dict(JG_minus_MMR=summ(per(lambda x: opt(x, "JG", "ymax") - opt(x, "MMR*", "ymax"))),
                              JG_minus_TOPK=summ(per(lambda x: opt(x, "JG", "ymax") - opt(x, "TOP-K[CVaR]", "ymax"))))
    D["gc_rows"] = dict(opt_yj_JG_minus_GC=summ(per(lambda x: opt(x, "JG") - opt(x, "GC"))),
                        opt_ymax_GC_minus_JG=summ(per(lambda x: opt(x, "GC", "ymax") - opt(x, "JG", "ymax"))),
                        opt_ymax_GC_minus_MMR=summ(per(lambda x: opt(x, "GC", "ymax") - opt(x, "MMR*", "ymax"))))
    D["P_all_TOPK_minus_MMR_k8"] = summ(per(lambda x: pall(x, "TOP-K[CVaR]") - pall(x, "MMR*")))
    D["P_all_TOPK_minus_JG_k8"] = summ(per(lambda x: pall(x, "TOP-K[CVaR]") - pall(x, "JG")))
    D["P_all_GC_minus_JG_k8"] = summ(per(lambda x: pall(x, "GC") - pall(x, "JG")))
    D["P_all_curve"] = {r: [summ(per(lambda x: pall(x, r, k)))["median"] for k in range(8)] for r in RULES}
    D["anchor_by_k"] = [summ(per(lambda x: pall(x, "MMR*", k) - pall(x, "JG", k))) for k in range(8)]
    D["nonshared"] = dict(
        d_mean_s_m_mmr_only_minus_jg_only=summ(per(lambda x: sub(x["M1"]["nonshared"]["mean_s_m_mmr_only"],
                                                                 x["M1"]["nonshared"]["mean_s_m_jg_only"]))),
        n_jg_only=summ(per(lambda x: x["M1"]["nonshared"]["n_jg_only"])))
    D["within_m_JG_minus_MMR"] = dict(
        pearson=summ(per(lambda x: x["M2"]["JG"]["m"]["within_pearson"] - x["M2"]["MMR*"]["m"]["within_pearson"])),
        cos=summ(per(lambda x: x["M2"]["JG"]["m"]["within_cos"] - x["M2"]["MMR*"]["m"]["within_cos"])))
    D["stability"] = {r: dict(k8=[s[r]["k8"] for s in sp["stability"]], k4=[s[r]["k4"] for s in sp["stability"]],
                              k8_median=float(np.median([s[r]["k8"] for s in sp["stability"]])),
                              k4_median=float(np.median([s[r]["k4"] for s in sp["stability"]])))
                      for r in RULES[:4]}
    hk = [k for k in ("H_all", "N_prime") if k in M["hard_sets"]]
    D["chase_shares"] = dict(
        label="expected >= reference by construction (Review A1); descriptive only",
        share_low_JG_minus_MMR=summ(per(lambda x: x["chase"]["JG"]["share_low_mean"]
                                        - x["chase"]["MMR*"]["share_low_mean"])),
        **{f"share_{k}_JG_minus_MMR": summ(per(lambda x: x["chase"]["JG"][f"share_{k}_mean"]
                                                - x["chase"]["MMR*"][f"share_{k}_mean"])) for k in hk},
        base_rates={k: len(M["hard_sets"][k]) / M["n_pairs"] for k in hk},
        per_step={r: [dict(j=j + 4, share_low=summ(per(lambda x: x["chase"][r]["steps"][j]["share_low"])),
                           delta=summ(per(lambda x: x["chase"][r]["steps"][j]["delta"])))
                      for j in range(5)] for r in ("JG", "MMR*")},
        delta={r: summ(per(lambda x: x["chase"][r]["delta"])) for r in ("JG", "MMR*")},
        n_skipped={r: per(lambda x: x["chase"][r]["n_skipped"]) for r in ("JG", "MMR*")})

    def gain(x, r, C, j):
        y = x["M3"][r][f"yj_{C}"]
        return y[j] - (y[j - 1] if j else 0.0)
    D["per_step_optimism_yj"] = {r: [summ(per(lambda x: gain(x, r, "s", j) - gain(x, r, "m", j))) for j in range(8)]
                                 for r in RULES}
    rec["descriptive"] = D
    rec["cost"] = M["cost"]
    return rec


def main():
    out = dict(schema="d038-analysis-v1", plan="docs/plans/D038-P1.md (+ Review A1 / A2)", s_val_reads=0,
               s_test_reads=0, consistent_rule=f">= {CONS} of 40", views={v: view_analysis(v) for v in VIEWS})
    v2, v1 = out["views"]["v2"], out["views"]["v1"]
    anchor_ok = v2["anchor"]["d_P_all_MMR_minus_JG_k8"]["median"] > 0
    hs, hc = v2["readings"]["H-strength"], v2["readings"]["H-chase"]
    hs_ctrl = abs(v1["readings"]["H-strength"]["d"]["median"]) < abs(hs["d"]["median"])
    hc_ctrl = v1["readings"]["H-chase"]["i"]["d"]["median"] < hc["i"]["d"]["median"]
    if hc["iii_prime"]["evaluable"]:
        chase = "supported" if hc["i"]["consistent"] and hc["ii"]["consistent"] and hc["iii_prime"]["consistent"] \
            else "not supported"
    else:
        chase = "supported (partial)" if hc["i"]["consistent"] and hc["ii"]["consistent"] \
            else "not supported (partial)"
    out["verdicts"] = dict(
        anchor_reproduces_test_ordering=anchor_ok,
        scope="" if anchor_ok else "S_build-internal only, does not reproduce the test ordering",
        H_strength=dict(supported_v2=hs["consistent"], v1_control_holds=hs_ctrl),
        H_redundancy=dict(supported_v2=v2["readings"]["H-redundancy"]["consistent"]),
        H_chase=dict(status_v2=chase, iii_prime_evaluable=hc["iii_prime"]["evaluable"], v1_control_holds=hc_ctrl),
        A2_case_v2=v2["A2"]["case"], A2_case_v1=v1["A2"]["case"])
    out["code_commit"] = L31.git_head()
    L31.jdump(out, f"{OUT}/analysis.json")
    print(json.dumps(out["verdicts"], indent=1), flush=True)


if __name__ == "__main__":
    main()
