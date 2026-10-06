"""D029 / S2 — the single S_test read. CORPUS V2 throughout.

Order (D029 ## D, P1 S2, Review A1):
  0. determinism stop: JG+ML counts == JG counts wherever their inputs are identical
  1. GUARDRAIL SCAN FIRST: every non-PAPER8 arm x trained k x readout vs PAPER8 at
     min(k, 8). Breach cell: lo(d_all) <= -0.02; kind = "point" if d_all <= -0.02
     else "ni_not_shown"; coincides_with_gain if point d_hard > 0 or d_all > 0.
  2. primary label (attention, k=8, JG vs PAPER8), plus the same table under linear
  3. P1-P3 (linear; attention alongside)
  4. secondary (a)-(f)
Bootstrap: configs M = boot_draws(25, 2000, seed=20261005), shared by everything;
attention seeds S = default_rng(20261005).integers(0, 5, (2000, 24, 5)), arm at k
uses S[:, k-1] (joint across arms at equal k). Linear: configs only.
Writes results/D029/{analysis.json, per_pair_hall.csv}; curves.png by d029_plots.py.
Usage:  PYTHONPATH=.:experiments python experiments/d029_analysis.py
"""
import csv
import json
import time

import numpy as np

import d029_lib as L
from d008_lib import boot_draws

NB, NCFG, NM = 2000, 25, 85
ARMS = ("PAPER8", "JG", "JG+ML", "GC", "JG8_ML16")
REFK = 8
DELTA = 0.02


def safe(arm):
    return arm.replace("+", "p")


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


class Data:
    def __init__(self):
        self.ch = json.load(open(f"{L.OUT}/chains.json"))
        self.models = self.ch["models"]
        self.tg = L.targets(self.models)
        self.H = self.tg["H"]
        self.hix = {p: j for j, p in enumerate(self.H)}
        self.C = {a: dict(np.load(f"{L.OUT}/counts_{safe(a)}.npz")) for a in ARMS}
        self.M = boot_draws(NCFG, NB, seed=L.SEED)
        self.S = np.random.default_rng(L.SEED).integers(0, 5, (NB, 24, 5))
        self.sM = self.M.sum(1)

    def ks(self, arm):
        return self.ch["arms"][arm]["k_train"]

    # ---- per-seed (or single) arrays -> (draws, point)
    def _att(self, arm, k, key):
        return np.stack([self.C[arm][f"att|{k}|{r}|{key}"] for r in range(5)])

    def _pick(self, per_seed, k):
        """per_seed (NB, 5, ...) -> resample seeds with slot k-1, mean."""
        idx = self.S[:, k - 1, :]
        sh = (NB, 5) + (1,) * (per_seed.ndim - 2)
        return np.take_along_axis(per_seed, idx.reshape(sh), 1).mean(1)

    def pair_acc(self, arm, k, ro, cols):
        """(NB, n) draws and (n,) point of two-way accuracy for H pairs `cols`."""
        if ro == "lin":
            d = self.C[arm][f"lin|{k}|dirH"][cols].sum(1).astype(float)      # (n, 25)
            return (self.M @ d.T) / (2 * self.sM[:, None]), d.sum(1) / (2 * NCFG)
        d = self._att(arm, k, "dirH")[:, cols].sum(2).astype(float)          # (5, n, 25)
        per = np.einsum("spc,bc->bsp", d, self.M) / (2 * self.sM[:, None, None])
        return self._pick(per, k), d.sum(2).mean(0) / (2 * NCFG)

    def model_recall(self, arm, k, ro, idx):
        if ro == "lin":
            d = self.C[arm][f"lin|{k}|cnt_model"][idx].astype(float)
            return (self.M @ d.T) / self.sM[:, None], d.sum(1) / NCFG
        d = self._att(arm, k, "cnt_model")[:, idx].astype(float)
        per = np.einsum("spc,bc->bsp", d, self.M) / self.sM[:, None, None]
        return self._pick(per, k), d.sum(2).mean(0) / NCFG

    def E_pairs(self, arm, k, ro, pairs):
        dr, pt = self.pair_acc(arm, k, ro, [self.hix[p] for p in pairs])
        return dr.mean(1), float(pt.mean())

    def E_models(self, arm, k, ro, idx):
        dr, pt = self.model_recall(arm, k, ro, list(idx))
        return dr.mean(1), float(pt.mean())

    def E_all(self, arm, k, ro):
        return self.E_models(arm, k, ro, range(NM))

    def E_hard(self, arm, k, ro):
        return self.E_pairs(arm, k, ro, self.H)


def contrast(fa, fb):
    (da, pa), (db, pb) = fa, fb
    d = da - db
    return dict(a=round(pa, 4), b=round(pb, 4), delta=round(pa - pb, 4),
                ci=[round(x, 4) for x in ci(d)])


def summ(f):
    d, p = f
    return dict(point=round(p, 4), ci=[round(x, 4) for x in ci(d)])


def main():
    t0 = time.time()
    D = Data()
    out = dict(schema="d029-analysis-v1", corpus="v2 (85 models, 85-way; every number here is corpus v2)",
               bootstrap=dict(configs=f"boot_draws(25, 2000, seed={L.SEED})",
                              seeds=f"default_rng({L.SEED}).integers(0,5,(2000,24,5)); arm at k uses slot k-1",
                              note="5-with-replacement understates a 5-seed mean's seed variance by 4/5"))

    # ---- 0. determinism stop
    jg, jm = D.ch["arms"]["JG"]["cands"], D.ch["arms"]["JG+ML"]["cands"]
    same = [k for k in D.ks("JG") if jg[:k] == jm[:k]]
    for k in same:
        for key in D.C["JG"]:
            if key.split("|")[0] == "lin" and key.split("|")[1] == str(k) or \
               key.split("|")[0] == "att" and key.split("|")[1] == str(k):
                assert np.array_equal(D.C["JG"][key], D.C["JG+ML"][key]), f"STOP: determinism {key}"
    out["determinism_identical_input_k"] = same
    print(f"[S2] determinism: JG+ML == JG exactly at k {same}", flush=True)

    # ---- 1. guardrail scan
    scan, breaches = [], []
    for arm in ("JG", "JG+ML", "GC", "JG8_ML16"):
        for k in D.ks(arm):
            for ro in ("att", "lin"):
                kr = min(k, REFK)
                ca = contrast(D.E_all(arm, k, ro), D.E_all("PAPER8", kr, ro))
                chd = contrast(D.E_hard(arm, k, ro), D.E_hard("PAPER8", kr, ro))
                cell = dict(arm=arm, k=k, readout=ro, paper8_k=kr, unequal_budget=k != kr,
                            d_all=ca["delta"], d_all_ci=ca["ci"], d_hard=chd["delta"], d_hard_ci=chd["ci"],
                            E_all=ca["a"], E_all_paper8=ca["b"], E_hard=chd["a"], E_hard_paper8=chd["b"])
                cell["breach"] = ca["ci"][0] <= -DELTA
                if cell["breach"]:
                    cell["kind"] = "point" if ca["delta"] <= -DELTA else "ni_not_shown"
                    cell["coincides_with_gain"] = chd["delta"] > 0 or ca["delta"] > 0
                    breaches.append(cell)
                scan.append(cell)
    out["guardrail"] = dict(rule="lo(d_all) <= -0.02 vs PAPER8 at min(k,8); kind point / ni_not_shown (Review A1)",
                            n_cells=len(scan), n_breach=len(breaches), breaches=breaches, scan=scan)
    print(f"[S2] GUARDRAIL: {len(breaches)} breach cells of {len(scan)}", flush=True)
    for b in breaches:
        print(f"   BREACH {b['arm']} k={b['k']} {b['readout']} d_all {b['d_all']:+.4f} {b['d_all_ci']} "
              f"kind={b['kind']} gain={b['coincides_with_gain']} d_hard {b['d_hard']:+.4f}", flush=True)

    # ---- 2. primary label
    def label(ro, other):
        ca = contrast(D.E_all("JG", 8, ro), D.E_all("PAPER8", 8, ro))
        ch = contrast(D.E_hard("JG", 8, ro), D.E_hard("PAPER8", 8, ro))
        oh = contrast(D.E_hard("JG", 8, other), D.E_hard("PAPER8", 8, other))
        if ca["ci"][0] <= -DELTA:
            lab = "GUARDRAIL BREACH"
        elif ch["ci"][0] > 0 and ca["ci"][0] > -DELTA and oh["delta"] > 0:
            lab = "CONFIRMED"
        elif ch["ci"][1] < DELTA:
            lab = "NOT CONFIRMED"
        else:
            lab = "INCONCLUSIVE"
        kind = None if ca["ci"][0] > -DELTA else ("point" if ca["delta"] <= -DELTA else "ni_not_shown")
        return dict(label=lab, E_all=ca, E_hard=ch, other_readout_d_hard=oh["delta"], guardrail_kind=kind)

    out["primary"] = dict(readout="attention (primary)", k=8, contrast="JG - PAPER8", **label("att", "lin"))
    out["primary_under_linear"] = dict(readout="linear (reported, not the label)", **label("lin", "att"))
    print(f"[S2] PRIMARY (attention, k=8): {out['primary']['label']}  d_hard {out['primary']['E_hard']['delta']:+.4f} "
          f"{out['primary']['E_hard']['ci']}  d_all {out['primary']['E_all']['delta']:+.4f} "
          f"{out['primary']['E_all']['ci']}  linear d_hard {out['primary']['other_readout_d_hard']:+.4f}", flush=True)
    print(f"[S2] same table under linear: {out['primary_under_linear']['label']}", flush=True)

    # ---- 3. ammunition P1-P3
    ml_by16 = any(c >= L.NPOOL for c in jm[:16])
    amm = {}
    for ro in ("lin", "att"):
        a = D.E_pairs("JG", 16, ro, D.tg["T1"])
        m = D.E_pairs("JG+ML", 16, ro, D.tg["T1"])
        b = (m[0] - a[0], m[1] - a[1])
        hi_a, lo_b = ci(a[0])[1], ci(b[0])[0]
        if a[1] >= 0.90:
            l1 = "REJECTED"
        elif hi_a < 0.90 and b[1] >= 0.05 and lo_b > 0:
            l1 = "SUPPORTED"
        elif hi_a < 0.90:
            l1 = "POOL-LIMITED ONLY"
        else:
            l1 = "INCONCLUSIVE"
        comp = {f"vs_JG_k{kk}": contrast(D.E_pairs("JG8_ML16", 24, ro, D.tg["T1"]), D.E_pairs("JG", kk, ro, D.tg["T1"]))
                for kk in (8, 16)}
        per1 = {p: dict(JG16=round(float(D.pair_acc("JG", 16, ro, [D.hix[p]])[1][0]), 4),
                        JGML16=round(float(D.pair_acc("JG+ML", 16, ro, [D.hix[p]])[1][0]), 4),
                        JG8_ML16=round(float(D.pair_acc("JG8_ML16", 24, ro, [D.hix[p]])[1][0]), 4),
                        PAPER8=round(float(D.pair_acc("PAPER8", 8, ro, [D.hix[p]])[1][0]), 4)) for p in D.tg["T1"]}

        def thr(f):
            d, p = f
            h = ci(d)[1]
            return dict(point=round(p, 4), ci=[round(x, 4) for x in ci(d)],
                        label="SUPPORTED" if p >= 0.90 else ("REJECTED" if h < 0.90 else "INCONCLUSIVE"))
        amm[ro] = dict(
            P1=dict(label=l1, a=summ(a), b=dict(point=round(b[1], 4), ci=[round(x, 4) for x in ci(b[0])]),
                    ml_selected_by_k16=ml_by16, b_identically_zero=not ml_by16, companion=comp, per_pair=per1),
            P2=thr(D.E_pairs("JG", 16, ro, D.tg["T2"])),
            P3=thr(D.E_pairs("JG", 8, ro, D.tg["T3"])))
        for P, kk, arm in (("P2", 16, "JG"), ("P3", 8, "JG")):
            T = D.tg["T2"] if P == "P2" else D.tg["T3"]
            amm[ro][P]["per_pair"] = {p: dict(JG=round(float(D.pair_acc(arm, kk, ro, [D.hix[p]])[1][0]), 4),
                                              PAPER8=round(float(D.pair_acc("PAPER8", 8, ro, [D.hix[p]])[1][0]), 4))
                                      for p in T}
    out["ammunition"] = dict(primary_readout="lin", T1=D.tg["T1"], T2=D.tg["T2"], T3=D.tg["T3"], **amm)
    for ro in ("lin", "att"):
        print(f"[S2] {ro}: P1 {amm[ro]['P1']['label']} (a {amm[ro]['P1']['a']}, b {amm[ro]['P1']['b']}); "
              f"P2 {amm[ro]['P2']['label']} {amm[ro]['P2']['point']}; P3 {amm[ro]['P3']['label']} "
              f"{amm[ro]['P3']['point']}", flush=True)

    # ---- 4. secondary
    sec = {}
    # (a) strata
    st = {}
    for ro in ("att", "lin"):
        for name, pairs in D.tg["strata"].items():
            for arm, k in (("JG", 8), ("JG", 16), ("JG+ML", 8), ("JG+ML", 16), ("GC", 8)):
                st[f"{ro}|{name}|{arm}|{k}"] = dict(n=len(pairs), **contrast(D.E_pairs(arm, k, ro, pairs),
                                                                             D.E_pairs("PAPER8", 8, ro, pairs)))
        for name in ("F_N", "M_X", "M_N"):
            idx = D.tg["model_sets"][name]
            for arm, k in (("JG", 8), ("JG", 16), ("JG+ML", 16), ("GC", 8)):
                st[f"{ro}|recall_{name}|{arm}|{k}"] = dict(n=len(idx), **contrast(D.E_models(arm, k, ro, idx),
                                                                                  D.E_models("PAPER8", 8, ro, idx)))
    sec["a_strata_vs_PAPER8_k8"] = st
    # (b) blocks
    bl = {}
    for ro in ("att", "lin"):
        for name, idx in D.tg["blocks"].items():
            for arm, k in (("JG", 8), ("JG", 16), ("JG+ML", 16), ("GC", 8)):
                bl[f"{ro}|{name}|{arm}|{k}"] = dict(n=len(idx), **contrast(D.E_models(arm, k, ro, idx),
                                                                           D.E_models("PAPER8", 8, ro, idx)))
    sec["b_blocks_vs_PAPER8_k8"] = dict(note="v1_37 includes the 3 regenerated gemma (new text; configs replay "
                                             "corpus_v1, whose S_test was read since D009)", cells=bl)
    # (c) curves
    cur = {}
    for ro in ("att", "lin"):
        for arm in ARMS:
            for k in D.ks(arm):
                cur[f"{ro}|{arm}|{k}"] = dict(E_all=summ(D.E_all(arm, k, ro)), E_hard=summ(D.E_hard(arm, k, ro)))
    sec["c_curves"] = cur
    # (d) GC vs PAPER8 / JG
    gcd = {}
    for ro in ("att", "lin"):
        for k in range(1, 9):
            for ref in ("PAPER8", "JG"):
                gcd[f"{ro}|{k}|GC-{ref}"] = dict(all=contrast(D.E_all("GC", k, ro), D.E_all(ref, k, ro)),
                                                 hard=contrast(D.E_hard("GC", k, ro), D.E_hard(ref, k, ro)))
    sec["d_GC"] = gcd
    # (e) per-pair H_all
    cols = [("PAPER8", 8), ("JG", 8), ("JG", 16), ("JG+ML", 8), ("JG+ML", 16), ("GC", 8)]
    rows = []
    for p in D.H:
        r = dict(pair=p, stratum="H_N" if p in D.tg["strata"]["H_N"] else "H_X", band=D.tg["bands"][p],
                 mode_mixed=p in D.tg["mode_mixed"], fresh_only=p in D.tg["strata"]["fresh_only"],
                 T=("T1" if p in D.tg["T1"] else "T2" if p in D.tg["T2"] else "T3" if p in D.tg["T3"] else ""))
        for ro in ("att", "lin"):
            for arm, k in cols:
                r[f"{ro}_{arm}_k{k}"] = round(float(D.pair_acc(arm, k, ro, [D.hix[p]])[1][0]), 4)
        rows.append(r)
    with open(f"{L.OUT}/per_pair_hall.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sec["e_per_pair"] = "per_pair_hall.csv"
    # (f) tiny-aya confusion
    ta = D.tg["tiny_aya"]
    conf = {}
    for arm, ks in (("PAPER8", (8,)), ("JG", (8, 16)), ("JG+ML", (8, 16))):
        lz = np.load(f"{L.OUT}/logits_{safe(arm)}.npz")
        y = lz["y"]
        sel = np.isin(y, ta)
        for k in ks:
            for ro, keys in (("lin", [f"lin|{k}"]), ("att", [f"att|{k}|{r}" for r in range(5)])):
                n = w_ = o_ = 0
                for key in keys:
                    pr = lz[key][sel].argmax(1)
                    err = pr != y[sel]
                    n += int(sel.sum()); w_ += int((err & np.isin(pr, ta)).sum()); o_ += int((err & ~np.isin(pr, ta)).sum())
                conf[f"{ro}|{arm}|{k}"] = dict(traces=n, errors=w_ + o_, within_family=w_, out_of_family=o_,
                                               within_share=round(w_ / max(1, w_ + o_), 4))
    sec["f_tiny_aya_confusion"] = conf
    out["secondary"] = sec
    out["wall_min"] = round((time.time() - t0) / 60, 1)
    json.dump(out, open(f"{L.OUT}/analysis.json", "w"), indent=1)

    print(f"[S2] done in {out['wall_min']} min", flush=True)


if __name__ == "__main__":
    main()
