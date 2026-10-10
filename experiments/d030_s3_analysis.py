"""D030 / P2 S3 — the single S_test read (Review D030/P2: B1-B3).

Order (P2 §2): 1. WORSE scan over every MET / ZP cell; 2. B* per method x view (highest E_all point
among {Gen, All} x {primary, variant}; native excluded); 3. labels (WORSE > BETTER > PARITY >
INCONCLUSIVE; BETTER needs the linear readout's point-estimate sign on both endpoints);
4. opposite-failure breakdown; 5. secondary (a)-(e), Top-3 / MRR, tie rates, double-BOS;
6. exact McNemar (ours linear vs each MET / ZP cell; tied baseline traces excluded), Holm.

Comparators, read verbatim (I6):
  v2  ours att  results/D029/counts_JG.npz att|8|r|*      linear lin|8|*     PAPER8 counts_PAPER8.npz
      bootstrap M = boot_draws(25, 2000, seed=20261005), seeds default_rng(20261005).integers(0,5,(2000,24,5))[:, 7]
  v1  ours att  E_all D010 joint_energy|8|r|cnt_model, E_hard D023 JointGreedy|8|r|cnt_pair666
      linear    E_all D020 JointGreedy|concat|8|cnt_model, E_hard Call 1 (call1_linear_v1.npz)
      PAPER8    E_all D009 paper8|8|r|cnt_model, E_hard D023 paper8|8|r|cnt_pair666 (attention only)
      bootstrap M = boot_draws(25, 2000, seed=20260928), seeds default_rng(20260928).integers(0,5,(2000,8,5))[:, 7]
Baselines have no seed dimension: only their configs are resampled, with the view's M.
Writes results/D030/analysis.json and results/D030/tail_{v2,v1}.npz (plotted by d030_plots.py).
Usage:  PYTHONPATH=.:experiments python experiments/d030_s3_analysis.py
"""
import json
import time
import itertools

import numpy as np
from scipy.stats import binomtest

import d029_lib as L29
from d008_lib import boot_draws, near_relative_pairs
from LLMmap.joint_statistic import cvar

RES = "./results/D030"
NB, NC, DELTA = 2000, 25, 0.02
SEEDS = {"v2": 20261005, "v1": 20260928}
METHODS = {"met": ("hamming", "e5"), "zp": ("512", "200")}
NAME = {("met", "hamming"): "MET-Hamming", ("met", "e5"): "MET-e5", ("zp", "512"): "ZP-512",
        ("zp", "200"): "ZP-200", ("os", "cos"): "LLMMAP-OS cos", ("os", "euc"): "LLMMAP-OS euc"}
LIBS = ("Gen", "All")
TYPES = ("none", "sys", "RAG", "CoT")


def r4(x):
    return round(float(x), 4)


def ci(x):
    return [r4(np.percentile(x, 2.5)), r4(np.percentile(x, 97.5))]


class Boot:
    def __init__(self, v):
        self.M = boot_draws(NC, NB, seed=SEEDS[v])
        n_slot = 24 if v == "v2" else 8
        self.S = np.random.default_rng(SEEDS[v]).integers(0, 5, (NB, n_slot, 5))[:, 7, :]
        self.sM = self.M.sum(1)

    def pick(self, per):
        """per (NB, R, ...) -> (NB, ...): seeds resampled with slot 7 when R == 5, else R == 1."""
        if per.shape[1] == 1:
            return per[:, 0]
        sh = (NB, 5) + (1,) * (per.ndim - 2)
        return np.take_along_axis(per, self.S.reshape(sh), 1).mean(1)


class Arm:
    """model (R, n, 25) top-1 credit; pairs (R, P, 25) two-logit counts (both directions) for the view's
    E_hard set (v2: H_all 57; v1: 666); R = 5 seeds or 1."""

    def __init__(self, name, model, pairs, view):
        self.name, self.view = name, view
        self.model = np.asarray(model, float)
        self.pairs = None if pairs is None else np.asarray(pairs, float)

    def E_all(self, B, idx=None):
        c = self.model if idx is None else self.model[:, idx]
        per = (B.M @ c.sum(1).T)[:, :, None] / (c.shape[1] * B.sM[:, None, None])    # (NB, R, 1)
        return B.pick(per)[:, 0], float(c.sum((1, 2)).mean() / (c.shape[1] * NC))

    def pair_acc(self, B):
        per = np.einsum("rpc,bc->brp", self.pairs, B.M) / (2 * B.sM[:, None, None])
        return B.pick(per), self.pairs.sum(2).mean(0) / (2 * NC)

    def E_hard(self, B):
        d, p = self.pair_acc(B)
        if self.view == "v2":
            return d.mean(1), float(p.mean())
        return np.array([cvar(x, 0.1) for x in d]), cvar(p, 0.1)


def contrast(a, b):
    (da, pa), (db, pb) = a, b
    return dict(a=r4(pa), b=r4(pb), delta=r4(pa - pb), ci=ci(da - db))


def label(d_all, d_hard, lin_all, lin_hard):
    if d_all["ci"][1] < 0 or d_hard["ci"][1] < 0:
        return "WORSE"
    if d_all["ci"][0] > 0 and d_hard["ci"][0] > 0 and lin_all > 0 and lin_hard > 0:
        return "BETTER"
    if all(-DELTA <= x <= DELTA for x in d_all["ci"] + d_hard["ci"]):
        return "PARITY"
    return "INCONCLUSIVE"


def two_logit_all(lg, y, cfg, n, pairs):
    """(P, 2, 25) two-logit counts for logits lg; dir 0 = traces of a (tie -> a, D025 rule)."""
    out = np.zeros((len(pairs), 2, NC))
    for j, (a, b) in enumerate(pairs):
        for d, m in enumerate((a, b)):
            sel = y == m
            ok = lg[sel, a] >= lg[sel, b] if d == 0 else lg[sel, b] > lg[sel, a]
            out[j, d] = np.bincount(cfg[sel][ok], minlength=NC)
    return out


def rank_from_logits(lg, y, n):
    lg = lg.reshape(n, NC, n)
    st = lg[np.arange(n), :, np.arange(n)]
    return 1 + (lg > st[:, :, None]).sum(2) + 0.5 * ((lg == st[:, :, None]).sum(2) - 1)


def top3_mrr(rank):
    rank = np.asarray(rank, float)
    return dict(top3=r4((rank <= 3).mean()), mrr=r4((1 / rank).mean()))


def main():
    t0 = time.time()
    uni = json.load(open("./results/D027/universe_v2.json"))
    V = {"v2": uni["models"], "v1": sorted(uni["v1"])}
    man = json.load(open("./confs/baselines/baseline_ext_v1.json"))["models"]
    tg = L29.targets(V["v2"])
    P = {v: list(itertools.combinations(range(len(V[v])), 2)) for v in V}
    pidx2 = {f"{V['v2'][a]} | {V['v2'][b]}": j for j, (a, b) in enumerate(P["v2"])}
    H = [pidx2[p] for p in tg["H"]]
    sets2 = {k: [pidx2[p] for p in tg["strata"][k]] for k in ("H_all", "H_N", "H_X")}
    sets2["N_prime"] = [pidx2[p["pair"]] for p in json.load(open("./results/D027/n_prime.json"))["pairs"]]
    assert len(sets2["N_prime"]) == 161
    struct1 = sorted(near_relative_pairs(V["v1"]))
    sets1 = {"structural65": struct1, "all666": list(range(666))}
    B = {v: Boot(v) for v in V}
    out = dict(schema="d030-analysis-v1", written=time.strftime("%Y-%m-%dT%H:%M:%S"),
               bootstrap={v: dict(configs=f"boot_draws(25, 2000, seed={SEEDS[v]})",
                                  seeds=f"default_rng({SEEDS[v]}).integers(0,5,(2000,{24 if v == 'v2' else 8},5))[:, 7]",
                                  baselines="configs only (no seed dimension)") for v in V})

    # ------------------------------------------------ comparators
    j29 = dict(np.load("./results/D029/counts_JG.npz"))
    p29 = dict(np.load("./results/D029/counts_PAPER8.npz"))
    c10 = np.load("./results/D010/run_counts.npz")
    c23 = np.load("./results/D023/run_counts.npz")
    c09 = np.load("./results/D009/run_counts.npz")
    l20 = np.load("./results/D020/linear_counts.npz")
    call1 = json.load(open(f"{RES}/call1_linear_v1.json"))
    st = lambda d, f: np.stack([d[f.format(r)] for r in range(5)])  # noqa: E731
    ours = {
        "v2": dict(att=Arm("OURS att", st(j29, "att|8|{}|cnt_model"), st(j29, "att|8|{}|dirH").sum(2), "v2"),
                   lin=Arm("OURS lin", j29["lin|8|cnt_model"][None], j29["lin|8|dirH"].sum(1)[None], "v2"),
                   p8=Arm("PAPER8 att", st(p29, "att|8|{}|cnt_model"), st(p29, "att|8|{}|dirH").sum(2), "v2"),
                   p8lin=Arm("PAPER8 lin", p29["lin|8|cnt_model"][None], p29["lin|8|dirH"].sum(1)[None], "v2")),
        "v1": dict(att=Arm("OURS att", st(c10, "joint_energy|8|{}|cnt_model"), st(c23, "JointGreedy|8|{}|cnt_pair666"), "v1"),
                   lin=Arm("OURS lin", l20["JointGreedy|concat|8|cnt_model"][None],
                           np.load(f"{RES}/call1_linear_v1.npz")["cnt_pair666"][None] if call1["all_equal"] else None, "v1"),
                   p8=Arm("PAPER8 att", st(c09, "paper8|8|{}|cnt_model"), st(c23, "paper8|8|{}|cnt_pair666"), "v1"))}
    for v in V:
        for k, a in ours[v].items():
            assert a.model.shape[1:] == (len(V[v]), NC), (v, k, a.model.shape)

    # ------------------------------------------------ baselines
    base = {}
    for v in V:
        for meth, vars_ in list(METHODS.items()) + [("os", ("cos", "euc"))]:
            for var in vars_:
                for lib in LIBS:
                    c = dict(np.load(f"{RES}/counts_{meth}_{var}_{lib}_{v}.npz"))
                    hard = c["dirH"].sum(1) if v == "v2" else c["cnt_pair666"]
                    a = Arm(f"{NAME[(meth, var)]} {lib}", c["cnt_model"][None], hard[None], v)
                    a.c = c
                    base[(v, meth, var, lib)] = a

    E = {}
    for v in V:
        for k, a in list(ours[v].items()):
            E[(v, "ours", k)] = (a.E_all(B[v]), a.E_hard(B[v]) if a.pairs is not None else None)
        for key, a in base.items():
            if key[0] == v:
                E[key] = (a.E_all(B[v]), a.E_hard(B[v]))

    def cell_row(v, meth, var, lib):
        (ea, eh) = E[(v, meth, var, lib)]
        oa, oh = E[(v, "ours", "att")]
        return dict(view=v, method=meth, variant=var, lib=lib, name=f"{NAME[(meth, var)]} {lib}",
                    E_all=r4(ea[1]), E_all_ci=ci(ea[0]), E_hard=r4(eh[1]), E_hard_ci=ci(eh[0]),
                    d_all=contrast(oa, ea), d_hard=contrast(oh, eh))

    # ------------------------------------------------ 1. WORSE scan (every MET / ZP cell)
    scan, worse = [], []
    for v in V:
        for meth, vars_ in METHODS.items():
            for var in vars_:
                for lib in LIBS:
                    r = cell_row(v, meth, var, lib)
                    r["worse"] = r["d_all"]["ci"][1] < 0 or r["d_hard"]["ci"][1] < 0
                    scan.append(r)
                    if r["worse"]:
                        worse.append(r)
    out["worse_scan"] = dict(rule="hi(delta) < 0 on E_all or E_hard, delta = OURS(att, k=8) - cell; every "
                                  "MET / ZP variant x library cell, both views", n_cells=len(scan),
                             n_worse=len(worse), worse=worse, cells=scan)
    print(f"[S3] WORSE scan: {len(worse)} of {len(scan)} cells", flush=True)
    for r in worse:
        print(f"   WORSE {r['view']} {r['name']}: d_all {r['d_all']}, d_hard {r['d_hard']}", flush=True)

    # ------------------------------------------------ 2-3. B* and labels
    labels = {}
    for v in V:
        oa, oh = E[(v, "ours", "att")]
        la, lh = E[(v, "ours", "lin")]
        for meth, vars_ in METHODS.items():
            cells = [(var, lib) for var in vars_ for lib in LIBS]
            pts = {c: E[(v, meth) + c][0][1] for c in cells}
            best = max(cells, key=lambda c: (pts[c], -cells.index(c)))
            ea, eh = E[(v, meth) + best]
            d_all, d_hard = contrast(oa, ea), contrast(oh, eh)
            lin_all = la[1] - ea[1]
            if lh is not None:
                lin_hard, lin_hard_src = lh[1] - eh[1], "Call 1 asserted reproduction (cnt_pair666, CVaR)" \
                    if v == "v1" else "D029 lin|8 dirH"
            else:   # Call 1 fallback: 65-pair structural mean
                b65 = base[(v, meth) + best].c["cnt_pair"].sum(1).mean() / (2 * NC)
                o65 = l20["JointGreedy|concat|8|cnt_pair"].sum(1).mean() / (2 * NC)
                lin_hard, lin_hard_src = o65 - b65, "FALLBACK 65-pair structural proxy (Call 1)"
            lab = label(d_all, d_hard, lin_all, lin_hard)
            prim = METHODS[meth][0]
            labels[f"{meth}|{v}"] = dict(
                view=v, method=meth, label=lab, B_star=dict(variant=best[0], lib=best[1], name=f"{NAME[(meth, best[0])]} {best[1]}"),
                E_all_points={f"{a}|{b}": r4(p) for (a, b), p in pts.items()},
                d_all=d_all, d_hard=d_hard, linear_sign=dict(d_all=r4(lin_all), d_hard=r4(lin_hard), hard_source=lin_hard_src),
                vs_primary={lib: dict(d_all=contrast(oa, E[(v, meth, prim, lib)][0]), d_hard=contrast(oh, E[(v, meth, prim, lib)][1]))
                            for lib in LIBS},
                b_star_optimism="B* is the max over 4 cells chosen on S_test; its E_all is biased upward, so delta "
                                "against B* is conservative for our claim (Review B2)")
            print(f"[S3] LABEL {meth} {v}: {lab}  B*={labels[f'{meth}|{v}']['B_star']['name']}  d_all {d_all}  "
                  f"d_hard {d_hard}  lin {r4(lin_all)}/{r4(lin_hard)}", flush=True)
    out["labels"] = labels
    out["ours"] = {v: {k: dict(E_all=r4(E[(v, 'ours', k)][0][1]), E_all_ci=ci(E[(v, 'ours', k)][0][0]),
                               **(dict(E_hard=r4(E[(v, 'ours', k)][1][1]), E_hard_ci=ci(E[(v, 'ours', k)][1][0]))
                                  if E[(v, 'ours', k)][1] is not None else {}))
                       for k in ours[v]} for v in V}

    # ------------------------------------------------ ours per-pair over all pairs (logits) for tail / N'
    y2, c2 = np.repeat(np.arange(85), NC), np.tile(np.arange(NC), 85)
    lg29 = np.load("./results/D029/logits_JG.npz")
    lgp8 = np.load("./results/D029/logits_PAPER8.npz")
    dirA = {"att": np.stack([two_logit_all(lg29[f"att|8|{r}"], y2, c2, 85, P["v2"]) for r in range(5)]),
            "lin": two_logit_all(lg29["lin|8"], y2, c2, 85, P["v2"])[None],
            "p8": np.stack([two_logit_all(lgp8[f"att|8|{r}"], y2, c2, 85, P["v2"]) for r in range(5)])}
    assert np.array_equal(dirA["att"][:, H], st(j29, "att|8|{}|dirH")), "logit-derived dirH != stored (att)"
    assert np.array_equal(dirA["lin"][0][H], j29["lin|8|dirH"]), "logit-derived dirH != stored (lin)"
    acc2 = {k: d.sum(2).sum(2).mean(0) / (2 * NC) for k, d in dirA.items()}          # (3570,)
    y1, c1 = np.repeat(np.arange(37), NC), np.tile(np.arange(NC), 37)
    lg16 = {r: np.load(f"./results/D016/logits_joint_energy_k8_r{r}.npy") for r in range(5)}
    lg16p = {r: np.load(f"./results/D016/logits_paper8_k8_r{r}.npy") for r in range(5)}
    for r in range(5):   # trace order check against the stored counts
        cm = np.zeros((37, NC))
        ok = lg16[r].argmax(1) == y1
        np.add.at(cm, (y1[ok], c1[ok]), 1.0)
        assert np.array_equal(cm, c10[f"joint_energy|8|{r}|cnt_model"]), f"D016 logits order r={r}"
    acc1 = {"att": st(c23, "JointGreedy|8|{}|cnt_pair666").sum(2).mean(0) / (2 * NC),
            "p8": st(c23, "paper8|8|{}|cnt_pair666").sum(2).mean(0) / (2 * NC)}
    if call1["all_equal"]:
        acc1["lin"] = np.load(f"{RES}/call1_linear_v1.npz")["cnt_pair666"].sum(1) / (2 * NC)

    # ------------------------------------------------ 4. opposite-failure breakdown
    opp = {}
    for v in V:
        sets = sets2 if v == "v2" else sets1
        acc_o = acc2 if v == "v2" else acc1
        o_ratio = E[(v, "ours", "att")][1][1] / E[(v, "ours", "att")][0][1]
        rows = {}
        for k in ("att", "lin", "p8"):
            if k in acc_o:
                rows[f"ours:{k}"] = {s: r4(acc_o[k][ix].mean()) for s, ix in sets.items()}
        for meth, vars_ in list(METHODS.items()) + [("os", ("cos", "euc"))]:
            for var in vars_:
                for lib in LIBS:
                    a = base[(v, meth, var, lib)]
                    pa = a.c["cnt_pair_all"].sum(1) / (2 * NC)
                    ea, eh = E[(v, meth, var, lib)]
                    r = {s: r4(pa[ix].mean()) for s, ix in sets.items()}
                    r.update(E_all=r4(ea[1]), E_hard=r4(eh[1]), hard_over_all=r4(eh[1] / ea[1]) if ea[1] else None,
                             hard_over_all_rel_ours=r4((eh[1] / ea[1]) / o_ratio) if ea[1] else None,
                             tie_rate=r4(a.c["tie"].mean()))
                    if "native_top1" in a.c:
                        nt = a.c["native_top1"]
                        r.update(native_top1=r4(nt.mean()), native_minus_E_all=r4(nt.mean() - ea[1]),
                                 native_tie_rate=r4(a.c["native_tie"].mean()), native_rank_mean=r4(a.c["native_rank"].mean()))
                    rows[f"{NAME[(meth, var)]} {lib}"] = r
        opp[v] = dict(ours_hard_over_all=r4(o_ratio), sets={s: len(ix) for s, ix in sets.items()}, rows=rows)
    out["opposite_failure"] = opp

    # ------------------------------------------------ 5. secondary
    sec = {}
    # (a) representation / truncation, paired
    pa_ = {}
    for v in V:
        for meth, (prim, var) in METHODS.items():
            for lib in LIBS:
                pa_[f"{meth}|{lib}|{v}"] = dict(
                    contrast=f"{NAME[(meth, prim)]} - {NAME[(meth, var)]} ({lib})",
                    d_all=contrast(E[(v, meth, prim, lib)][0], E[(v, meth, var, lib)][0]),
                    d_hard=contrast(E[(v, meth, prim, lib)][1], E[(v, meth, var, lib)][1]))
    sec["a_representation"] = pa_
    # (b) blocks (v2)
    blocks = tg["blocks"]
    bl = {}
    for name_, a in [("OURS att", ours["v2"]["att"]), ("OURS lin", ours["v2"]["lin"]), ("PAPER8 att", ours["v2"]["p8"])] + \
            [(a.name, a) for (v, *_), a in base.items() if v == "v2"]:
        bl[name_] = {b: dict(zip(("E_all", "ci"), (r4(a.E_all(B["v2"], ix)[1]), ci(a.E_all(B["v2"], ix)[0]))))
                     for b, ix in blocks.items()}
    sec["b_blocks_v2"] = dict(blocks={b: len(ix) for b, ix in blocks.items()}, rows=bl)
    # (c) config type of the slot
    ct = {}
    for v in V:
        typ = np.array([man[m]["test_types"] for m in V[v]])                       # (n, 25)
        share = {t: r4((typ == t).mean()) for t in TYPES}
        rows = {}
        arms = [(f"OURS {k}", a) for k, a in ours[v].items()] + [(a.name, a) for (vv, *_), a in base.items() if vv == v]
        for nm, a in arms:
            cm = a.model.mean(0)
            rows[nm] = {t: (r4(cm[typ == t].mean()) if (typ == t).any() else None) for t in TYPES}
        ct[v] = dict(share_of_slots=share, rows=rows)
    sec["c_config_type"] = ct
    # (d) LLMMAP-OS (descriptive), split by encoder-training overlap (A4: v1 37 in-sample, v2-new 48 out)
    s0f = json.load(open(f"{RES}/s0_freeze.json"))["llmmap_os"]
    ov = set(s0f["overlap_with_universe"])
    dd = {}
    for v in V:
        ins = [i for i, m in enumerate(V[v]) if m in ov]
        oos = [i for i, m in enumerate(V[v]) if m not in ov]
        for var in ("cos", "euc"):
            for lib in LIBS:
                a = base[(v, "os", var, lib)]
                ea, eh = E[(v, "os", var, lib)]
                r = dict(E_all=r4(ea[1]), E_all_ci=ci(ea[0]), E_hard=r4(eh[1]), E_hard_ci=ci(eh[0]),
                         d_all_ours_minus=contrast(E[(v, "ours", "att")][0], ea),
                         d_hard_ours_minus=contrast(E[(v, "ours", "att")][1], eh),
                         in_sample=dict(n=len(ins), E_all=r4(a.E_all(B[v], ins)[1]) if ins else None),
                         out_of_sample=dict(n=len(oos), E_all=r4(a.E_all(B[v], oos)[1]) if oos else None),
                         tie_rate=r4(a.c["tie"].mean()), **top3_mrr(a.c["rank"]))
                dd[f"{var}|{lib}|{v}"] = r
    sec["d_llmmap_os"] = dict(overlap_rule="released encoder's training models (37, all v1) = in-sample", rows=dd)
    # Top-3 / MRR and tie rates, every cell + ours (from logits)
    tm = {}
    for v in V:
        rows = {}
        if v == "v2":
            for k, key in (("OURS att", "att|8|{}"), ("PAPER8 att", "att|8|{}")):
                src = lg29 if k.startswith("OURS") else lgp8
                rk = np.stack([rank_from_logits(src[key.format(r)], y2, 85) for r in range(5)])
                rows[k] = dict(top3=r4(np.mean([(x <= 3).mean() for x in rk])), mrr=r4(np.mean([(1 / x).mean() for x in rk])))
            rows["OURS lin"] = top3_mrr(rank_from_logits(lg29["lin|8"], y2, 85))
        else:
            for k, src in (("OURS att", lg16), ("PAPER8 att", lg16p)):
                rk = np.stack([rank_from_logits(src[r], y1, 37) for r in range(5)])
                rows[k] = dict(top3=r4(np.mean([(x <= 3).mean() for x in rk])), mrr=r4(np.mean([(1 / x).mean() for x in rk])))
            if call1["all_equal"]:
                rows["OURS lin"] = top3_mrr(rank_from_logits(np.load(f"{RES}/call1_linear_v1.npz")["logits"], y1, 37))
        for (vv, meth, var, lib), a in base.items():
            if vv == v:
                rows[a.name] = dict(**top3_mrr(a.c["rank"]), top1=r4(a.c["cnt_model"].mean()), tie_rate=r4(a.c["tie"].mean()))
        tm[v] = rows
    sec["top3_mrr_ties"] = tm
    sec["dblbos_control"] = json.load(open(f"{RES}/dblbos_control.json"))
    out["secondary"] = sec

    # (e) tail arrays
    for v in V:
        acc_o = acc2 if v == "v2" else acc1
        arrs = {"OURS att": np.sort(acc_o["att"]), "PAPER8 att": np.sort(acc_o["p8"])}
        for meth in METHODS:
            bs = labels[f"{meth}|{v}"]["B_star"]
            arrs[f"{meth} B* ({bs['name']})"] = np.sort(base[(v, meth, bs["variant"], bs["lib"])].c["cnt_pair_all"].sum(1) / (2 * NC))
        np.savez(f"{RES}/tail_{v}.npz", **arrs)
        out.setdefault("tail", {})[v] = {k: dict(min=r4(x.min()), p10=r4(np.percentile(x, 10)), median=r4(np.median(x)),
                                                 cvar01=r4(cvar(x, 0.1)), n_below_090=int((x < 0.9).sum()), n=len(x))
                                         for k, x in arrs.items()}

    # ------------------------------------------------ 6. McNemar (ours linear vs each MET / ZP cell), Holm
    mc = []
    for v in V:
        o = ours[v]["lin"].model[0]                                             # (n, 25) in {0, 1}
        assert set(np.unique(o)) <= {0.0, 1.0}
        for meth, vars_ in METHODS.items():
            for var in vars_:
                for lib in LIBS:
                    a = base[(v, meth, var, lib)]
                    keep = a.c["tie"] == 0
                    bcorr = a.c["cnt_model"] == 1.0
                    b_ = int(((o == 1) & ~bcorr & keep).sum())
                    c_ = int(((o == 0) & bcorr & keep).sum())
                    p = binomtest(b_, b_ + c_, 0.5).pvalue if b_ + c_ else 1.0
                    mc.append(dict(view=v, cell=f"{NAME[(meth, var)]} {lib}", ours_only=b_, baseline_only=c_,
                                   n_traces=int(keep.sum()), n_excluded_ties=int((~keep).sum()), p=float(p)))
    order = np.argsort([m["p"] for m in mc])
    k = len(mc)
    run = 0.0
    for rnk, i in enumerate(order):
        run = max(run, min(1.0, (k - rnk) * mc[i]["p"]))
        mc[i]["p_holm"] = float(run)
    out["mcnemar"] = dict(rule="exact two-sided McNemar on per-trace top-1, ours = linear readout k=8; baseline traces "
                               "with a top-1 tie excluded (Call 7); Holm over all tests in the table", n_tests=k, tests=mc)
    out["wall_s"] = round(time.time() - t0, 1)
    json.dump(out, open(f"{RES}/analysis.json", "w"), indent=1)
    print(f"[S3] analysis.json written ({out['wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
