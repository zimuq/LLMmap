"""D032 / S3 — the single S_test read of the new cells (P1 §S3). D031's analysis objects imported verbatim.

Order: 1. WORSE scan (every Q1-Q3 cell, starred and non-starred, attention k = 8); 2. Q1 (criterion);
3. Q2 (redundancy space); 4. Q3 (redundancy term); 5. secondary (a) curves + queries-to-reach, (b) overlaps /
D022 flags (from S0), (c) near-relative mechanism, (d) within-set similarities (from S0), (e) tail-plot data.
Bootstrap = d031_analysis.Boot (D031 S4's replicates, same seeds). Labels = d031_analysis.label (delta 0.02;
BETTER needs the linear sign to agree on both endpoints).
Cells read from: D031 counts (Store: TOP-K[CVaR], MMR-PP[CVaR] x 6, RANDOM, JG / PAPER8 / GC as D031 S4 reads
them) and D032 counts (the 19 new chains). Writes results/D032/analysis.json and plotdata.npz.
Usage:  PYTHONPATH=.:experiments python experiments/d032_analysis.py
"""
import glob
import json
import time
import itertools

import numpy as np

import d031_lib as L31
import d032_lib as L
from d031_analysis import Boot, Arm, Store, contrast, label, r4, ci, NC, NB
from LLMmap.joint_statistic import cvar

REFS = ("JG", "PAPER8", "GC")


class Store32(Store):
    def __init__(self, view):
        self.view, self.n = view, L31.VIEW[view]["n"]
        self.lin, self.att = {}, {}
        for f in sorted(glob.glob(f"{L.OUT}/counts_{view}_lin_*of*.npz")):
            self.lin.update(dict(np.load(f)))
        for f in sorted(glob.glob(f"{L.OUT}/counts_{view}_att_*of*.npz")):
            self.att.update(dict(np.load(f)))


def nr_cols(view):
    ms = L31.models(view)
    if view == "v1":
        from d008_lib import near_relative_pairs
        return sorted(near_relative_pairs(ms))
    pid = {f"{a} | {b}": j for j, (a, b) in enumerate(itertools.combinations(ms, 2))}
    return sorted(pid[p["pair"]] for p in json.load(open("./results/D027/n_prime.json"))["pairs"])


class View:
    def __init__(self, view, a31):
        self.view, self.B, self.a31 = view, Boot(view), a31
        self.s31, self.s32 = Store(view), Store32(view)
        self._arm, self._ev = {}, {}

    def arm(self, ro, name, k=8):
        kk = (ro, name, k)
        if kk not in self._arm:
            if name in REFS:
                X = Arm(name, self.s31.ref(ro, name, k))
            elif name == "RANDOM":
                X = self.s31.arm(ro, "RANDOM", k, self.a31["RANDOM"])
            elif L.d031_name(name):
                X = Arm(name, self.s31.new(ro, L.d031_name(name), k))
            else:
                X = Arm(name, self.s32.new(ro, name, k))
            self._arm[kk] = X
        return self._arm[kk]

    def ev(self, ro, name, what, k=8):
        kk = (ro, name, what, k)
        if kk not in self._ev:
            X = self.arm(ro, name, k)
            self._ev[kk] = X.e_all(self.B) if what == "all" else X.e_tail(self.B)
        return self._ev[kk]

    def cell(self, a, b, **tags):
        row = {ro: {f"d_{e}": contrast(self.ev(ro, a, e), self.ev(ro, b, e)) for e in ("all", "tail")}
               for ro in ("att", "lin")}
        la, lt = row["lin"]["d_all"]["delta"], row["lin"]["d_tail"]["delta"]
        return dict(a=a, b=b, label=label(row["att"]["d_all"], row["att"]["d_tail"], la, lt),
                    linear_as_label=label(row["lin"]["d_all"], row["lin"]["d_tail"], la, lt),
                    worse=row["att"]["d_all"]["ci"][1] < 0 or row["att"]["d_tail"]["ci"][1] < 0, **tags, **row)


def q1_verdict(lab):
    if any(l == "WORSE" for v in lab.values() for l in v.values()):
        return "WORSE (headline)"
    ok = {v: sum(l == "BETTER" for l in lab[v].values()) >= 2 for v in lab}
    if all(ok.values()):
        return "SUPPORTED"
    one = [v for v, o in ok.items() if o]
    return f"SUPPORTED-{one[0]}" if one else "NOT SUPPORTED"


def q2_verdict(lab):
    if any(l == "WORSE" for l in lab.values()):
        return "WORSE (headline)"
    if all(l == "BETTER" for l in lab.values()):
        return "SUPPORTED"
    one = [v for v, l in lab.items() if l == "BETTER"]
    if one:
        return f"SUPPORTED-in-{one[0]}"
    if all(l == "PARITY" for l in lab.values()):
        return "space does not matter (PARITY in both views)"
    return "NOT SUPPORTED"


def main():
    t0 = time.time()
    ch = json.load(open(f"{L.OUT}/chains.json"))
    vc = json.load(open(f"{L.OUT}/variant_choice.json"))
    c31 = json.load(open(f"{L31.OUT}/chains.json"))
    s4 = json.load(open(f"{L31.OUT}/analysis.json"))
    out = dict(schema="d032-analysis-v1", review="D032/P1 approved (04b2ea7)",
               bootstrap=s4["bootstrap"], label_rule="d031_analysis.label, delta = 0.02; BETTER also needs linear "
               "point deltas > 0 on both endpoints", worse_scan={}, q1={}, q2={}, q3={}, secondary={},
               honesty_note="TOP-K[CVaR] and MMR-PP[CVaR] S_test were read in D031 and D032 was designed after "
                            "seeing them; every MEAN and TXT cell is a fresh S_test read; MMR-PP[CVaR] - TOP-K[CVaR] "
                            "duplicates D031 S6 ('seen'); confirmatory evidence only from a later fresh-S_test D.")
    plot = {}
    q1_lab, q2_lab = {}, {}
    for view in ("v2", "v1"):
        tv = time.time()
        V = View(view, c31["views"][view]["arms"])
        S0 = ch["views"][view]
        star = vc["views"][view]["stars"]
        # reused cells equal D031 S4's levels exactly (same counts, same replicates)
        for nm, src in (("TOP-K[CVaR]", "TOP-K"), (star["MMR-PP[CVaR]"], L.d031_name(star["MMR-PP[CVaR]"])), ("JG", "JG")):
            for ro in ("att", "lin"):
                c = s4["curves"][view][f"{ro}|{src}"]["8"]
                assert r4(V.ev(ro, nm, "all")[1]) == c["E_all"] and r4(V.ev(ro, nm, "tail")[1]) == c["E_tail"], (view, nm, ro)

        # 1. WORSE scan
        scan = []
        for r in ("TOP-K", "MMR-PP", "MMR-TXT"):
            prs = [(f"TOP-K[CVaR]", f"TOP-K[MEAN]", True)] if r == "TOP-K" else \
                [(star[f"{r}[CVaR]"], star[f"{r}[MEAN]"], True)] + \
                [(v, v.replace("[CVaR]", "[MEAN]"), False) for v in L.variants(r, "CVaR")]
            scan += [V.cell(a, b, q="Q1", rule=r, starred=s, kind="star-star" if s else "variant-matched")
                     for a, b, s in prs]
        for c in L.CRITS:
            pp, tx = star[f"MMR-PP[{c}]"], star[f"MMR-TXT[{c}]"]
            scan.append(V.cell(pp, tx, q="Q2", crit=c, starred=True, kind="star-star"))
            scan += [V.cell(v, tx, q="Q2", crit=c, starred=False, kind="PP variant vs TXT*") for v in L.variants("MMR-PP", c) if v != pp]
            scan += [V.cell(pp, v, q="Q2", crit=c, starred=False, kind="PP* vs TXT variant") for v in L.variants("MMR-TXT", c) if v != tx]
            for x in ("MMR-PP", "MMR-TXT"):
                scan += [V.cell(v, f"TOP-K[{c}]", q="Q3", crit=c, rule=x, starred=v == star[f"{x}[{c}]"],
                                kind="variant - TOP-K", seen=(x == "MMR-PP" and c == "CVaR"))
                         for v in L.variants(x, c)]
        out["worse_scan"][view] = dict(
            rule="hi(delta) < 0 on E_all or E_tail (attention, k = 8); delta = first - second arm",
            n_cells=len(scan), worse=[f"{r['q']}: {r['a']} - {r['b']}" for r in scan if r["worse"]], rows=scan)

        # 2. Q1
        q1 = {}
        for r in ("TOP-K", "MMR-PP", "MMR-TXT"):
            q1[r] = next(x for x in scan if x["q"] == "Q1" and x["rule"] == r and x["starred"])
        q1_lab[view] = {r: x["label"] for r, x in q1.items()}
        out["q1"][view] = dict(cells=q1, labels=q1_lab[view], n_better=sum(l == "BETTER" for l in q1_lab[view].values()),
                               degenerate={r: S0["degenerate"].get(f"{x['a']} vs {x['b']}") for r, x in q1.items()})
        # 3. Q2
        q2 = {c: next(x for x in scan if x["q"] == "Q2" and x["crit"] == c and x["starred"]) for c in L.CRITS}
        q2_lab[view] = q2["CVaR"]["label"]
        out["q2"][view] = dict(cells=q2, labels={c: x["label"] for c, x in q2.items()}, primary="CVaR")
        # 4. Q3 (starred cells)
        q3 = {f"{x}[{c}]": next(r for r in scan if r["q"] == "Q3" and r["crit"] == c and r["rule"] == x and r["starred"])
              for c in L.CRITS for x in ("MMR-PP", "MMR-TXT")}
        out["q3"][view] = dict(cells=q3, labels={n: x["label"] for n, x in q3.items()},
                               seen=["MMR-PP[CVaR]"])
        for name, blk in (("Q1", q1), ("Q2", q2), ("Q3", q3)):
            print(f"[S3 {view}] {name}: " + "; ".join(
                f"{k} {x['a']} - {x['b']}: att all {x['att']['d_all']['delta']:+.4f} {x['att']['d_all']['ci']} "
                f"tail {x['att']['d_tail']['delta']:+.4f} {x['att']['d_tail']['ci']} -> {x['label']} "
                f"(lin {x['lin']['d_all']['delta']:+.4f}/{x['lin']['d_tail']['delta']:+.4f}; lin-as-label "
                f"{x['linear_as_label']})" for k, x in blk.items()), flush=True)
        print(f"[S3 {view}] WORSE cells: {out['worse_scan'][view]['worse']}", flush=True)

        # 5a. curves + reach
        stars6 = list(dict.fromkeys(star.values()))
        curves = {}
        for ro in ("att", "lin"):
            names = list(REFS) + (stars6 if ro == "att" else L.chain_names())
            for nm in names:
                pts = {}
                for k in range(1, 9):
                    u = V.arm(ro, nm, k).units[0]
                    ea = u.e_all(V.B)
                    pts[k] = dict(E_all=r4(ea[1]), E_all_ci=ci(ea[0]))
                    if u.pair is not None:
                        et = u.e_tail(V.B)
                        pts[k].update(E_tail=r4(et[1]), E_tail_ci=ci(et[0]))
                curves[f"{ro}|{nm}"] = pts
            for k in ((2, 4, 8) if ro == "lin" else (8,)):
                ea, et = V.ev(ro, "RANDOM", "all", k), V.ev(ro, "RANDOM", "tail", k)
                curves.setdefault(f"{ro}|RANDOM", {})[k] = dict(E_all=r4(ea[1]), E_all_ci=ci(ea[0]),
                                                                E_tail=r4(et[1]), E_tail_ci=ci(et[0]))
        for nm in L.chain_names():                                      # non-starred attention: k = 8 only
            if nm not in stars6:
                ea, et = V.ev("att", nm, "all"), V.ev("att", nm, "tail")
                curves[f"att|{nm}"] = {8: dict(E_all=r4(ea[1]), E_all_ci=ci(ea[0]), E_tail=r4(et[1]), E_tail_ci=ci(et[0]))}
        reach = {}
        for ro in ("att", "lin"):
            tgt = {k: curves[f"{ro}|JG"][k]["E_all"] for k in (4, 8)}
            for kk, pts in curves.items():
                if kk.startswith(ro + "|") and len(pts) == 8:
                    reach[kk] = {f"JG_k{k}": next((j for j in sorted(pts) if pts[j]["E_all"] >= t), None) for k, t in tgt.items()}
        # 5c. mechanism: per-pair gain CVaR - MEAN, near-relative vs other; near-relative share of worst decile
        nr = nr_cols(view)
        P = L31.VIEW[view]["P"]
        oth = sorted(set(range(P)) - set(nr))
        nw = max(1, int(np.floor(L31.GAMMA * P)))
        mech = dict(near_relative=dict(source="results/D027/n_prime.json (N', structural)" if view == "v2"
                                       else "d008_lib.near_relative_pairs (structural)", n=len(nr), n_other=len(oth)),
                    worst_decile_size=nw, base_share=r4(len(nr) / P), gain={}, worst_decile_share={})
        for ro in ("att", "lin"):
            for x, cell in q1.items():
                (da, pa), (db, pb) = (V.arm(ro, cell["a"]).units[0].pair_acc(V.B), V.arm(ro, cell["b"]).units[0].pair_acc(V.B))
                g, gp = da - db, pa - pb
                gn, go = g[:, nr].mean(1), g[:, oth].mean(1)
                mech["gain"][f"{ro}|{x}"] = dict(a=cell["a"], b=cell["b"],
                                                 near_relative=dict(point=r4(gp[nr].mean()), ci=ci(gn)),
                                                 other=dict(point=r4(gp[oth].mean()), ci=ci(go)),
                                                 nr_minus_other=dict(point=r4(gp[nr].mean() - gp[oth].mean()), ci=ci(gn - go)))
            for nm in list(REFS) + L.chain_names():
                u = V.arm(ro, nm).units[0]
                if u.pair is None:
                    continue
                p = u.pair.sum(2).mean(0) / (2 * NC)
                w = np.lexsort((np.arange(P), p))[:nw]
                mech["worst_decile_share"][f"{ro}|{nm}"] = r4(np.isin(w, nr).mean())
        out["secondary"][view] = dict(
            curves=curves, reach=dict(rule="smallest k with point E_all >= JG's point E_all at k (same readout)", **reach),
            overlaps_k8=S0["overlaps_k8"], d022_union_flags_k8=S0["d022_union_flags_k8"],
            degenerate=S0["degenerate"], within_set_sim_k8=S0["within_set_sim_k8"],
            sim_distribution=S0["sim_distribution"], sim_rank_corr=S0["sim_rank_corr"], mechanism=mech)
        # 5e. tail plot data (attention, k = 8, every arm)
        for nm in list(REFS) + L.chain_names() + ["RANDOM"]:
            plot[f"tail|{view}|{nm}"] = V.arm("att", nm).sorted_curve()
        for kk, pts in curves.items():
            ks = sorted(pts)
            plot[f"curve|{view}|{kk}"] = np.array([[k, pts[k]["E_all"], pts[k].get("E_tail") or np.nan] for k in ks])
        print(f"[S3 {view}] done ({(time.time() - tv) / 60:.1f} min)", flush=True)
    out["q1"]["verdict"] = q1_verdict(q1_lab)
    out["q2"]["verdict"] = q2_verdict(q2_lab)
    print(f"[S3] Q1 verdict: {out['q1']['verdict']}; Q2 verdict: {out['q2']['verdict']}", flush=True)
    out["wall_s"] = round(time.time() - t0)
    L31.jdump(out, f"{L.OUT}/analysis.json")
    np.savez_compressed(f"{L.OUT}/plotdata.npz", **plot)
    print(f"[S3] wrote analysis.json, plotdata.npz ({out['wall_s']} s)", flush=True)


if __name__ == "__main__":
    main()
