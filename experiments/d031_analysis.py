"""D031 / S4 — the single S_test read (Review D031/P1: A1-A3, Calls 2-9).

Order (P1 §S4): 1. WORSE scan; 2. labels (JG vs HELM*, TOP-K, MMR*, RANDOM; attention, k = 8; linear sign
clause); 3. draw-arm companions; 4. secondary (a)-(d) + H_all (v2) + D019 model sets (v1) + GC reference
row (A3); 5. budget section. Writes results/D031/analysis.json and plot inputs results/D031/plotdata.npz.

Bootstrap per view (P1 §0): configs M = boot_draws(25, 2000, seed_view); attention seeds slot k-1 of
default_rng(seed_view).integers(0, 5, (2000, n_slot, 5)), shared across arms; linear: configs only.
Draw arms (Review Call 5 (a)): every replicate also resamples draws with replacement (default_rng(20261012):
one index vector per replicate for 20-draw sets, one for 10-draw sets). Review A2: a draw arm's E_tail =
mean over draws of each draw's CVaR_0.1 (the CVaR of the draw-mean pair accuracies is a secondary).
Seeds within one set: pair accuracies are averaged over the resampled seeds before CVaR (D029/D030).
Usage:  PYTHONPATH=.:experiments python experiments/d031_analysis.py
"""
import glob
import json
import time
import itertools

import numpy as np

import d031_lib as L
from d008_lib import boot_draws
from LLMmap.joint_statistic import cvar

NB, NC, DELTA = 2000, 25, 0.02
RES = L.OUT


def r4(x):
    return None if x is None else round(float(x), 4)


def ci(x):
    return [r4(np.percentile(x, 2.5)), r4(np.percentile(x, 97.5))]


def cvar_rows(X, g=L.GAMMA):
    k = max(1, int(np.floor(g * X.shape[1])))
    return np.partition(X, k - 1, axis=1)[:, :k].mean(1)


class Boot:
    def __init__(self, view):
        v = L.VIEW[view]
        self.M = boot_draws(NC, NB, seed=v["boot_seed"])
        self.sM = self.M.sum(1)
        self.S = np.random.default_rng(v["boot_seed"]).integers(0, 5, (NB, v["n_slot"], 5))
        rd = np.random.default_rng(L.BOOT_DRAW_SEED)
        self.D = {20: rd.integers(0, 20, (NB, 20)), 10: rd.integers(0, 10, (NB, 10))}


class Unit:
    """One trained set: tot (R, 25), mod (R, n, 25), pair (R, P, 25) or None; R = 5 seeds or 1."""

    def __init__(self, tot, mod, pair, k):
        self.tot = np.asarray(tot, float)
        self.mod = np.asarray(mod, float)
        self.pair = None if pair is None else np.asarray(pair, float)
        self.k, self.R, self.n = k, self.tot.shape[0], self.mod.shape[1]

    def _pick(self, B, per):                     # per (NB, R, ...) -> (NB, ...)
        if self.R == 1:
            return per[:, 0]
        idx = B.S[:, self.k - 1, :]
        return np.take_along_axis(per, idx.reshape((NB, 5) + (1,) * (per.ndim - 2)), 1).mean(1)

    def e_all(self, B):
        per = (B.M @ self.tot.T) / (self.n * B.sM[:, None])
        return self._pick(B, per), float(self.tot.sum() / (self.R * self.n * NC))

    def pair_acc(self, B, cols=None):
        p = self.pair if cols is None else self.pair[:, cols]
        per = np.einsum("rpc,bc->brp", p, B.M) / (2 * B.sM[:, None, None])
        return self._pick(B, per), p.sum(2).mean(0) / (2 * NC)

    def e_tail(self, B):
        d, p = self.pair_acc(B)
        return cvar_rows(d), cvar(p, L.GAMMA)

    def e_hard(self, B, cols):
        d, p = self.pair_acc(B, cols)
        return d.mean(1), float(p.mean())

    def model_set(self, B, idx):
        m = self.mod[:, idx].sum(1)                                                    # (R, 25)
        per = (B.M @ m.T) / (len(idx) * B.sM[:, None])
        return self._pick(B, per), float(m.sum() / (self.R * len(idx) * NC))


class Arm:
    """A deterministic set (one Unit) or a draw arm (list of Units, resampled per Review Call 5 (a))."""

    def __init__(self, name, units):
        self.name = name
        self.units = units if isinstance(units, list) else [units]
        self.draw = isinstance(units, list)

    def _agg(self, B, fn):
        vals = [fn(u, B) for u in self.units]
        bd = np.stack([v[0] for v in vals], 1)                                         # (NB, D)
        pt = np.array([v[1] for v in vals])
        if not self.draw:
            return bd[:, 0], float(pt[0]), None
        D = B.D[len(self.units)]
        return np.take_along_axis(bd, D, 1).mean(1), float(pt.mean()), pt

    def e_all(self, B):
        return self._agg(B, lambda u, B: u.e_all(B))

    def e_tail(self, B):                                                               # A2: mean of per-draw CVaR
        return self._agg(B, lambda u, B: u.e_tail(B))

    def e_hard(self, B, cols):
        return self._agg(B, lambda u, B: u.e_hard(B, cols))

    def model_set(self, B, idx):
        return self._agg(B, lambda u, B: u.model_set(B, idx))

    def tail_of_mean(self, B):
        """Secondary for draw arms: CVaR of the draw-mean pair accuracies (the D's original reading)."""
        D = B.D[len(self.units)]
        acc, pts = 0, []
        for j, u in enumerate(self.units):
            d, p = u.pair_acc(B)
            w = (D == j).sum(1)[:, None]
            acc = acc + w * d
            pts.append(p)
        return cvar_rows(acc / D.shape[1]), cvar(np.mean(pts, 0), L.GAMMA)

    def sorted_curve(self):
        return np.mean([np.sort(u.pair.sum(2).mean(0) / (2 * NC)) for u in self.units], 0)


def contrast(a, b):
    (da, pa, _), (db, pb, _) = a, b
    return dict(a=r4(pa), b=r4(pb), delta=r4(pa - pb), ci=ci(da - db))


def label(d_all, d_tail, lin_all, lin_tail):
    if d_all["ci"][1] < 0 or d_tail["ci"][1] < 0:
        return "WORSE"
    if d_all["ci"][0] > 0 and d_tail["ci"][0] > 0 and lin_all > 0 and lin_tail > 0:
        return "BETTER"
    if all(-DELTA <= x <= DELTA for x in d_all["ci"] + d_tail["ci"]):
        return "PARITY"
    return "INCONCLUSIVE"


# ------------------------------------------------------------------ loaders
def pair_from_logits(lg, n):
    G = lg.reshape(n, NC, n)
    ia, ib = np.array(list(itertools.combinations(range(n), 2))).T
    return (G[ia, :, ia] >= G[ia, :, ib]).astype(np.uint8) + (G[ib, :, ib] > G[ib, :, ia]).astype(np.uint8)


class Store:
    def __init__(self, view):
        self.view, self.n = view, L.VIEW[view]["n"]
        self.lin = {}
        for f in sorted(glob.glob(f"{RES}/counts_{view}_lin*.npz")):
            self.lin.update(dict(np.load(f)))
        self.att = {}
        for f in sorted(glob.glob(f"{RES}/counts_{view}_att_*of*.npz")):
            self.att.update(dict(np.load(f)))

    def new(self, readout, name, k, d=None):
        base = f"{name}|{k}" + (f"|d{d}" if d is not None else "")
        if readout == "lin":
            z = self.lin
            return Unit(z[f"{base}|cnt_total"][None], z[f"{base}|cnt_model"][None], z[f"{base}|cnt_pair_all"][None], k)
        z = self.att
        g = lambda c: np.stack([z[f"{base}|r{r}|{c}"] for r in range(5)])
        return Unit(g("cnt_total"), g("cnt_model"), g("cnt_pair_all"), k)

    def arm(self, readout, name, k, ch_arm):
        if ch_arm["kind"] == "det":
            return Arm(name, self.new(readout, name, k))
        nd = L.N_DRAWS if readout == "lin" else L.N_DRAWS_ATT
        return Arm(name, [self.new(readout, name, k, d) for d in range(nd)])

    def ref(self, readout, name, k):
        """JG / PAPER8 / GC reference rows, read verbatim (I6)."""
        n, v = self.n, self.view
        if v == "v2":
            fn = {"JG": "JG", "PAPER8": "PAPER8", "GC": "GC"}[name]
            z = np.load(f"./results/D029/counts_{fn}.npz")
            lg = np.load(f"./results/D029/logits_{fn}.npz")
            assert np.array_equal(lg["y"], np.repeat(np.arange(n), NC)) and np.array_equal(lg["cfg"], np.tile(np.arange(NC), n))
            if readout == "att":
                rs = range(5)
                tot = [z[f"att|{k}|{r}|cnt_total"] for r in rs]
                mod = [z[f"att|{k}|{r}|cnt_model"] for r in rs]
                pair = [pair_from_logits(lg[f"att|{k}|{r}"], n) for r in rs]
                for r in rs:
                    assert np.array_equal(pair[r][self.hcols()].astype(float), z[f"att|{k}|{r}|cnt_pair"])
            else:
                tot, mod = [z[f"lin|{k}|cnt_total"]], [z[f"lin|{k}|cnt_model"]]
                pair = [pair_from_logits(lg[f"lin|{k}"], n)]
                assert np.array_equal(pair[0][self.hcols()].astype(float), z[f"lin|{k}|cnt_pair"])
            return Unit(tot, mod, pair, k)
        if readout == "lin":
            if name in ("JG", "PAPER8"):
                return self.new("lin", name, k)
            z = np.load("./results/D020/linear_counts.npz")                           # GC: no all-pair counts
            return Unit(z[f"GreedyCover|concat|{k}|cnt_total"][None], z[f"GreedyCover|concat|{k}|cnt_model"][None], None, k)
        rs = range(5)
        if name == "JG":
            a, p = np.load("./results/D010/run_counts.npz"), "joint_energy"
            z23 = np.load("./results/D023/run_counts.npz")
            pair = [z23[f"JointGreedy|{k}|{r}|cnt_pair666"] for r in rs]
            for r in rs:
                assert np.array_equal(z23[f"JointGreedy|{k}|{r}|cnt_total"], a[f"{p}|{k}|{r}|cnt_total"])
        elif name == "PAPER8":
            a, p = np.load("./results/D009/run_counts.npz"), "paper8"
            z23 = np.load("./results/D023/run_counts.npz")
            pair = [z23[f"paper8|{k}|{r}|cnt_pair666"] for r in rs]
            for r in rs:
                assert np.array_equal(z23[f"paper8|{k}|{r}|cnt_total"], a[f"{p}|{k}|{r}|cnt_total"])
        else:                                                                           # GC = D009 cvar_max (= D013 gamma 0.1)
            a, p = np.load("./results/D009/run_counts.npz"), "cvar_max"
            pair = None
            if k == 8:
                pair = []
                for r in rs:
                    lg = np.load(f"./results/D016/logits_coverage_k8_r{r}.npy")
                    pc = pair_from_logits(lg, n)
                    assert float((lg.argmax(1) == np.repeat(np.arange(n), NC)).sum()) == a[f"{p}|8|{r}|cnt_total"].sum()
                    pair.append(pc)
        return Unit([a[f"{p}|{k}|{r}|cnt_total"] for r in rs], [a[f"{p}|{k}|{r}|cnt_model"] for r in rs], pair, k)

    def hcols(self):
        import d029_lib as L29
        ms = L.models("v2")
        tg = L29.targets(ms)
        pid = {f"{a} | {b}": j for j, (a, b) in enumerate(itertools.combinations(ms, 2))}
        return [pid[p] for p in tg["H"]]


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    ch = json.load(open(f"{RES}/chains.json"))
    vc = json.load(open(f"{RES}/variant_choice.json"))
    out = dict(schema="d031-analysis-v1", review="D031/P1 approved with amendments A1-A3",
               bootstrap=dict(n=NB, v2_seed=20261005, v1_seed=20260928, draw_seed=L.BOOT_DRAW_SEED,
                              note="draw arms: configs + seeds + draws resampled (Call 5 a); E_tail of a draw arm = "
                                   "mean of per-draw CVaR (A2); seeds averaged before CVaR within a set; 5-seed "
                                   "resampling understates seed variance by 4/5"),
               worse_scan={}, labels={}, linear_labels={}, companions={}, curves={}, reach={}, ordering={},
               secondary={}, gc={}, helm={}, budget={})
    plot = {}
    for view in ("v2", "v1"):
        B = Boot(view)
        st = Store(view)
        A = ch["views"][view]["arms"]
        V = vc["views"][view]
        star = {"HELM*": V["HELM_star"], "TOP-K": "TOP-K", "MMR*": V["MMR_star"], "RANDOM": "RANDOM"}
        variants = [n for n in A if n not in ("JG", "PAPER8") and not n.startswith("HELM-ORTH")]
        arms = {ro: {n: st.arm(ro, n, 8, A[n]) for n in variants} for ro in ("att", "lin")}
        refs = {ro: {n: Arm(n, st.ref(ro, n, 8)) for n in ("JG", "PAPER8", "GC")} for ro in ("att", "lin")}
        cache = {}

        def ev(ro, nm, arm, what):
            kk = (ro, nm, what)
            if kk not in cache:
                cache[kk] = arm.e_all(B) if what == "all" else arm.e_tail(B)
            return cache[kk]

        JG = {ro: refs[ro]["JG"] for ro in ("att", "lin")}
        # 1. WORSE scan + 2. labels
        scan, labels, lin_labels = [], {}, {}
        for nm in variants:
            row = {}
            for ro in ("att", "lin"):
                X = arms[ro][nm]
                row[ro] = dict(d_all=contrast(ev(ro, "JG", JG[ro], "all"), ev(ro, nm, X, "all")),
                               d_tail=contrast(ev(ro, "JG", JG[ro], "tail"), ev(ro, nm, X, "tail")))
            worse = row["att"]["d_all"]["ci"][1] < 0 or row["att"]["d_tail"]["ci"][1] < 0
            scan.append(dict(arm=nm, starred=[s for s, v in star.items() if v == nm], worse=worse, **row))
        out["worse_scan"][view] = dict(rule="hi(delta) < 0 on E_all or E_tail, delta = JG(att, k=8) - X(att, k=8)",
                                       n_cells=len(scan), worse=[r["arm"] for r in scan if r["worse"]], rows=scan)
        by = {r["arm"]: r for r in scan}
        for s, nm in star.items():
            r = by[nm]
            labels[s] = dict(variant=nm, label=label(r["att"]["d_all"], r["att"]["d_tail"],
                                                    r["lin"]["d_all"]["delta"], r["lin"]["d_tail"]["delta"]),
                             att=r["att"], lin_points=dict(d_all=r["lin"]["d_all"]["delta"],
                                                           d_tail=r["lin"]["d_tail"]["delta"]))
            lin_labels[s] = label(r["lin"]["d_all"], r["lin"]["d_tail"], r["lin"]["d_all"]["delta"],
                                  r["lin"]["d_tail"]["delta"])
        out["labels"][view] = labels
        out["linear_labels"][view] = dict(note="same table with linear as the label readout (reported, not the label)",
                                          **lin_labels)
        print(f"[S4 {view}] labels: " + ", ".join(f"{s} ({v['variant']}) {v['label']}" for s, v in labels.items()),
              flush=True)

        # 3. draw companions
        comp = {}
        for nm in ("RANDOM", "HELM-SIG|raw", "HELM-SIG|sf"):
            c = {}
            for ro in ("att", "lin"):
                X = arms[ro][nm]
                ea, et = ev(ro, nm, X, "all"), ev(ro, nm, X, "tail")
                ja, jt = ev(ro, "JG", JG[ro], "all")[1], ev(ro, "JG", JG[ro], "tail")[1]
                tm = X.tail_of_mean(B)
                c[ro] = dict(n_draws=len(X.units), per_draw_E_all=[r4(x) for x in ea[2]],
                             per_draw_E_tail=[r4(x) for x in et[2]],
                             sd_E_all=r4(np.std(ea[2], ddof=1)), sd_E_tail=r4(np.std(et[2], ddof=1)),
                             jg_percentile_E_all=r4(np.mean(ea[2] < ja) + 0.5 * np.mean(ea[2] == ja)),
                             jg_percentile_E_tail=r4(np.mean(et[2] < jt) + 0.5 * np.mean(et[2] == jt)),
                             E_tail_mean_of_cvar=dict(point=r4(et[1]), ci=ci(et[0])),
                             E_tail_cvar_of_mean=dict(point=r4(tm[1]), ci=ci(tm[0]),
                                                      d_vs_JG=dict(delta=r4(jt - tm[1]),
                                                                   ci=ci(ev(ro, "JG", JG[ro], "tail")[0] - tm[0]))))
            comp[nm] = c
        out["companions"][view] = comp

        # 4a. curves (det arms k = 1..8 both readouts; draw arms lin k 2/4/8, att k 8)
        curves = {}
        det = ["JG", "PAPER8", "GC"] + [n for n in variants if A[n]["kind"] == "det"]
        for ro in ("att", "lin"):
            for nm in det:
                pts = {}
                for k in range(1, 9):
                    u = st.ref(ro, nm, k) if nm in ("JG", "PAPER8", "GC") else st.new(ro, nm, k)
                    ea = u.e_all(B)
                    pts[k] = dict(E_all=r4(ea[1]), E_all_ci=ci(ea[0]))
                    if u.pair is not None:
                        et = u.e_tail(B)
                        pts[k].update(E_tail=r4(et[1]), E_tail_ci=ci(et[0]))
                curves[f"{ro}|{nm}"] = pts
            for nm in [n for n in variants if A[n]["kind"] == "draw"]:
                ks = (2, 4, 8) if ro == "lin" else (8,)
                pts = {}
                for k in ks:
                    X = st.arm(ro, nm, k, A[nm])
                    ea, et = X.e_all(B), X.e_tail(B)
                    pts[k] = dict(E_all=r4(ea[1]), E_all_ci=ci(ea[0]), E_tail=r4(et[1]), E_tail_ci=ci(et[0]))
                curves[f"{ro}|{nm}"] = pts
        out["curves"][view] = curves
        reach = {}
        for ro in ("att", "lin"):
            tgt = {k: curves[f"{ro}|JG"][k]["E_all"] for k in (4, 8)}
            for key_, pts in curves.items():
                if not key_.startswith(ro + "|"):
                    continue
                reach[key_] = {f"JG_k{k}": next((kk for kk in sorted(pts) if pts[kk]["E_all"] >= t), None)
                               for k, t in tgt.items()}
        out["reach"][view] = dict(rule="smallest trained k with point E_all >= JG's point E_all at k (same readout)",
                                  **reach)

        # 4c. Helm diagnostics + ordering check (linear)
        H = ch["views"][view]["helm"]
        out["helm"][view] = {e: dict(r_hat=H[e]["r_hat"], r_hat_cap20=H[e]["r_hat_cap20"], tail_flag=H[e]["tail_flag"],
                                     ratio=r4(H[e]["r_hat_ratio"]), runner_up_s=H[e]["runner_up_s"],
                                     runner_up_ratio=r4(H[e]["runner_up_ratio"]),
                                     spectrum_top10=[r4(x) for x in H[e]["spectrum"][:10]],
                                     gmm=[{k: v for k, v in d.items() if k != "zero_set"} for d in H[e]["gmm_abs"]],
                                     gmm_signed_diag=H[e]["gmm_signed_diag"], n_signal=H[e]["n_signal"],
                                     n_orthogonal=H[e]["n_orthogonal"],
                                     degenerate=H[e]["degenerate_sig_equals_random"]) for e in L.E_VARIANTS}
        order = {}
        for e in L.E_VARIANTS:
            for k in (2, 4, 8):
                X = {s: st.arm("lin", f"{s}|{e}" if s != "RANDOM" else "RANDOM", k,
                               A[f"{s}|{e}" if s != "RANDOM" else "RANDOM"]) for s in ("HELM-SIG", "HELM-ORTH", "RANDOM")}
                R = {s: (x.e_all(B), x.e_tail(B)) for s, x in X.items()}
                order[f"{e}|k{k}"] = dict(
                    **{s: dict(E_all=r4(v[0][1]), E_all_ci=ci(v[0][0]), E_tail=r4(v[1][1]), E_tail_ci=ci(v[1][0]))
                       for s, v in R.items()},
                    signal_minus_orth=dict(all=contrast(R["HELM-SIG"][0], R["HELM-ORTH"][0]),
                                           tail=contrast(R["HELM-SIG"][1], R["HELM-ORTH"][1])),
                    signal_minus_uniform=dict(all=contrast(R["HELM-SIG"][0], R["RANDOM"][0]),
                                              tail=contrast(R["HELM-SIG"][1], R["RANDOM"][1])))
        out["ordering"][view] = order

        # 4. secondary: H_all (v2) / D019 sets (v1); GC reference row (A3)
        sec = {}
        names = ["PAPER8", "GC"] + list(dict.fromkeys(star.values())) + [V["HELM_SIG_star"], V["HELM_RANK_star"]]
        names = list(dict.fromkeys(names))
        if view == "v2":
            hc = st.hcols()
            for ro in ("att", "lin"):
                j = JG[ro].e_hard(B, hc)
                for nm in names:
                    X = refs[ro][nm] if nm in refs[ro] else arms[ro][nm]
                    sec[f"{ro}|H_all|{nm}"] = contrast(j, X.e_hard(B, hc))
        else:
            ms = L.models("v1")
            sets = json.load(open("./results/D019/hard_model_recall.json"))["model_sets"]
            for sname, mem in sets.items():
                idx = [ms.index(m) for m in mem]
                for ro in ("att", "lin"):
                    j = JG[ro].model_set(B, idx)
                    for nm in names:
                        X = refs[ro][nm] if nm in refs[ro] else arms[ro][nm]
                        sec[f"{ro}|{sname}|{nm}"] = contrast(j, X.model_set(B, idx))
        out["secondary"][view] = sec
        gc = {}
        for ro in ("att", "lin"):
            G = refs[ro]["GC"]
            gc[ro] = dict(E_all=contrast(ev(ro, "JG", JG[ro], "all"), G.e_all(B)),
                          vs_TOPK_E_all=contrast(G.e_all(B), ev(ro, "TOP-K", arms[ro]["TOP-K"], "all")))
            if G.units[0].pair is not None:
                gc[ro]["E_tail"] = contrast(ev(ro, "JG", JG[ro], "tail"), G.e_tail(B))
                gc[ro]["vs_TOPK_E_tail"] = contrast(G.e_tail(B), ev(ro, "TOP-K", arms[ro]["TOP-K"], "tail"))
        out["gc"][view] = dict(note="GreedyCover reference row (Review A3): v2 D029 GC; v1 D009 cvar_max (= D013 "
                                    "gamma 0.1), all-pair from D016 logits at k=8 only (attention); v1 linear D020, "
                                    "no all-pair counts -> no E_tail", **gc)
        # 4d. tail plot data (attention, k = 8)
        for nm in ["JG", "PAPER8", "GC"] + list(dict.fromkeys(list(star.values()) +
                                                              [V["HELM_SIG_star"], V["HELM_RANK_star"]])):
            X = refs["att"][nm] if nm in refs["att"] else arms["att"][nm]
            plot[f"tail|{view}|{nm}"] = X.sorted_curve()
        for key_, pts in curves.items():
            ks = sorted(pts)
            plot[f"curve|{view}|{key_}"] = np.array([[k, pts[k]["E_all"], pts[k].get("E_tail") or np.nan] for k in ks])
        # 5. budget
        out["budget"][view] = budget(view, B, JG, plot)
        print(f"[S4 {view}] done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    out["wall_s"] = round(time.time() - t0)
    L.jdump(out, f"{RES}/analysis.json")
    np.savez_compressed(f"{RES}/plotdata.npz", **plot)
    print(f"[S4] wrote analysis.json, plotdata.npz ({out['wall_s']} s)", flush=True)


def budget(view, B, JG, plot):
    n = L.VIEW[view]["n"]
    BS = f"{RES}/budget_scores"
    ms = L.models(view)
    # MET* = highest D030 S_test E_all at n = 250 among the 4 cells (the D's rule; favours MET)
    e250 = {}
    for var in ("hamming", "e5"):
        for lib in ("Gen", "All"):
            z = np.load(f"./results/D030/counts_met_{var}_{lib}_{view}.npz")
            e250[f"{var}|{lib}"] = float(z["cnt_total"].sum() / (n * NC))
    star = max(e250, key=lambda c: e250[c])
    cells = {}
    for c in e250:
        var, lib = c.split("|")
        z = np.load(f"{RES}/budget_met_{var}_{lib}_{view}.npz")
        rows = {}
        for nn in (8, 25, 50, 100, 250):
            cm = z[f"n{nn}|cnt_model_draws"]
            p2 = np.load(f"{BS}/pair2x_{var}_{lib}_{view}_n{nn}.npy")
            us = [Unit(cm[d].sum(0)[None], cm[d][None], (p2[d] / 2.0)[None], 8) for d in range(len(cm))]
            X = Arm(f"MET-{c}-n{nn}", us if len(us) > 1 else us[0])
            ea, et = X.e_all(B), X.e_tail(B)
            rows[nn] = dict(E_all=r4(ea[1]), E_all_ci=ci(ea[0]), E_tail=r4(et[1]), E_tail_ci=ci(et[0]),
                            tie_rate=r4(z[f"n{nn}|tie_rate_draws"].mean()), draws=len(us),
                            sd_E_all=r4(np.std(ea[2], ddof=1)) if ea[2] is not None else None, _x=X)
        cells[c] = rows
    assert abs(cells[star][250]["E_all"] - r4(e250[star])) < 1e-4
    ja, jt = JG["att"].e_all(B), JG["att"].e_tail(B)
    X8 = cells[star][8]["_x"]
    d_all = contrast((ja[0], ja[1], None), X8.e_all(B))
    d_tail = contrast((jt[0], jt[1], None), X8.e_tail(B))
    la = JG["lin"].e_all(B)[1] - X8.e_all(B)[1]
    lt = JG["lin"].e_tail(B)[1] - X8.e_tail(B)[1]
    reach = next((nn for nn in (8, 25, 50, 100, 250) if cells[star][nn]["E_all"] >= r4(ja[1])), None)
    for c, rows in cells.items():
        plot[f"budget|{view}|{c}"] = np.array([[nn, rows[nn]["E_all"], rows[nn]["E_tail"]] for nn in sorted(rows)])
        for r in rows.values():
            del r["_x"]
    # our curve: v2 D029 JG k 1-8, 12, 16; v1 D010 k 1-8 + D025 GLOBAL 9-16 (attention)
    ours = []
    if view == "v2":
        z = np.load("./results/D029/counts_JG.npz")
        for k in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16):
            ours.append([k, np.mean([z[f"att|{k}|{r}|cnt_total"].sum() for r in range(5)]) / (n * NC)])
    else:
        z10, z25 = np.load("./results/D010/run_counts.npz"), np.load("./results/D025/counts_GLOBAL.npz")
        for k in range(1, 17):
            if k <= 8:
                v10 = [z10[f"joint_energy|{k}|{r}|cnt_total"].sum() for r in range(5)]
                v25 = [z25[f"att|{k}|{r}|cnt_total"].sum() for r in range(5)]
                assert v10 == v25, ("D025 GLOBAL != D010", k)
                ours.append([k, np.mean(v10) / (n * NC)])
            else:
                ours.append([k, np.mean([z25[f"att|{k}|{r}|cnt_total"].sum() for r in range(5)]) / (n * NC)])
    plot[f"budget|{view}|ours"] = np.array(ours)
    return dict(MET_star=star, MET_star_rule="highest D030 S_test E_all at n = 250 (favours MET)",
                e250={c: r4(v) for c, v in e250.items()},
                label=label(d_all, d_tail, la, lt), d_all=d_all, d_tail=d_tail,
                lin_points=dict(d_all=r4(la), d_tail=r4(lt)),
                smallest_n_MET_star_reaches_JG_k8=reach, JG_k8_att_E_all=r4(ja[1]), cells=cells,
                ours_curve=[[int(k), r4(v)] for k, v in ours])


if __name__ == "__main__":
    main()
