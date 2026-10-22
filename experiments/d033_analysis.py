"""D033 / S3 — the single S_test read (Review D033/P1: Calls 1-3 -> (a), A1).

Order (P1 §6): 1. assemble pooled held-out counts per (view, kind, readout, arm[, seed]) from S2's stored distances /
point-to-cloud terms, and write counts_<view>.npz; 2. tail-plot data (secondary (f), drawn before labels are read);
3. WORSE scan over the 4 primary cells, then over every labelled secondary contrast; 4. labels + claim rule;
5. secondary (a)-(g) + A1 encoder diagnostics. Writes analysis.json and plotdata.npz.

Readouts: "enc" = (1) Euclidean (Call 1 (a); cosine is a secondary row), "nc" = (2) with each view's S1 variant.
Pooled held-out: every model is held out in exactly one fold of a kind; its S_test targets are scored with that
fold's chain (and encoder) against ALL enrolled models. O_all = top-1; O_tail = CVaR_0.1 over all P pairs of the
both-direction two-candidate accuracy (D031's E_tail). Bootstrap per view: D031's config draws M (25 S_test configs,
2,000 replicates); (1) seeds: default_rng(sha256("D033|seeds|<view>")).integers(0, 3, (2000, 3)), shared by arms
and folds; folds fixed. Label table: the D's (delta 0.02, no linear clause).
Usage:  PYTHONPATH=.:experiments python experiments/d033_analysis.py
"""
import json
import time

import numpy as np

import d031_lib as L31
import d033_lib as L
from d008_lib import boot_draws, near_relative_pairs
from d031_analysis import cvar_rows, ci, r4, NB, NC, DELTA
from LLMmap.joint_statistic import cvar

RES = L.OUT
PRIMARY = [("OURS", "PAPER8")]


def label(d_all, d_tail):
    if d_all["ci"][1] < 0 or d_tail["ci"][1] < 0:
        return "WORSE"
    if d_all["ci"][0] > 0 and d_tail["ci"][0] > 0:
        return "BETTER"
    if all(-DELTA <= x <= DELTA for x in d_all["ci"] + d_tail["ci"]):
        return "PARITY"
    return "INCONCLUSIVE"


class Boot:
    def __init__(self, view):
        self.M = boot_draws(NC, NB, seed=L31.VIEW[view]["boot_seed"])
        self.sM = self.M.sum(1)
        self.S3 = L.rng_for(f"D033|seeds|{view}").integers(0, 3, (NB, 3))


class Unit:
    """cm (R, n, 25) held-out top-1 counts, cp (R, P, 25) both-direction pair counts; R = 3 seeds or 1."""

    def __init__(self, cm, cp):
        self.cm, self.cp = np.asarray(cm, float), np.asarray(cp, float)
        self.R, self.n = self.cm.shape[0], self.cm.shape[1]
        self._cache = None

    def _pick(self, B, per):
        if self.R == 1:
            return per[:, 0]
        assert self.R == 3
        idx = B.S3.reshape((NB, 3) + (1,) * (per.ndim - 2))
        return np.take_along_axis(per, idx, 1).mean(1)

    def stats(self, B):
        if self._cache is None:
            tot = self.cm.sum(1)
            per = (B.M @ tot.T) / (self.n * B.sM[:, None])
            pacc = np.einsum("rpc,bc->brp", self.cp, B.M) / (2 * B.sM[:, None, None])
            pa = self._pick(B, pacc)
            pa_pt = self.cp.sum(2).mean(0) / (2 * NC)
            self._cache = dict(all=(self._pick(B, per), float(tot.sum() / (self.R * self.n * NC))),
                               tail=(cvar_rows(pa), float(cvar(pa_pt, L31.GAMMA))), pa=(pa, pa_pt))
        return self._cache

    def sub(self, B, cols):
        pa, pt = self.stats(B)["pa"]
        return pa[:, cols].mean(1), float(pt[cols].mean())


def contrast(ua, ub, B):
    out = {}
    for e in ("all", "tail"):
        (da, pa), (db, pb) = ua.stats(B)[e], ub.stats(B)[e]
        out[f"d_{e}"] = dict(a=r4(pa), b=r4(pb), delta=r4(pa - pb), ci=ci(da - db))
    out["label"] = label(out["d_all"], out["d_tail"])
    return out


def level(u, B):
    s = u.stats(B)
    return dict(O_all=r4(s["all"][1]), O_all_ci=ci(s["all"][0]), O_tail=r4(s["tail"][1]), O_tail_ci=ci(s["tail"][0]))


# ------------------------------------------------------------------ assembling pooled held-out units
class View:
    def __init__(self, view, ch, vc):
        self.view, self.ch = view, ch
        self.ms = L31.models(view)
        self.n = len(self.ms)
        self.ia, self.ib = L31.pairs(view)
        self.P = len(self.ia)
        self.var = vc["views"][view]["choice"]
        self.B = Boot(view)
        z = np.load(f"{RES}/nc_test_{view}.npz")
        self.A, self.W, self.U = z["A"], z["W"], [int(q) for q in z["U"]]
        assert list(z["models"]) == self.ms
        self.units, self.store, self.seen = {}, {}, {}

    def folds(self, kind):
        return self.ch["views"][self.view]["kinds"][kind]["folds"]

    def rows(self, models):
        return np.concatenate([np.arange(self.ms.index(m) * NC, (self.ms.index(m) + 1) * NC) for m in models])

    def nc(self, kind, arm, variant, k=L.K):
        key = ("nc", kind, arm, variant, k)
        if key not in self.units:
            pos = {q: j for j, q in enumerate(self.U)}
            D = np.full((self.n * NC, self.n), np.nan)
            seen = np.full((L.NF, self.n, NC), np.nan)
            for F in self.folds(kind):
                cols = [pos[q] for q in F["arms"][arm][:k]]
                S = L.nc_scores(self.A, self.W, cols, variant)
                r = self.rows(F["heldout"])
                D[r] = S[r]
                ok = (np.argmin(S, 1) == np.repeat(np.arange(self.n), NC)).reshape(self.n, NC).astype(float)
                tr = F["train"]
                seen[F["fold"], tr] = ok[tr]
            assert np.isfinite(D).all()
            c = L.counts(D, self.n, self.ia, self.ib)
            self._keep(key, c, seen)
            self.units[key] = Unit(c["cnt_model"][None], c["cnt_pair_all"][None])
        return self.units[key]

    def enc(self, kind, arm, metric="euc"):
        key = ("enc", kind, arm, metric)
        if key not in self.units:
            seeds = L.SEEDS if kind == "model" else (0,)
            cms, cps, seens = [], [], []
            for s in seeds:
                D = np.full((self.n * NC, self.n), np.nan)
                seen = np.full((L.NF, self.n, NC), np.nan)
                for F in self.folds(kind):
                    src = F["shared_training"].get(arm, arm)
                    nm = f"{self.view}_{kind}_f{F['fold']}_{src}_s{s}"
                    z = np.load(f"{RES}/feat/{nm}.npz")
                    run = json.load(open(f"{RES}/runs/{nm}.json"))
                    assert run["queries"] == F["arms"][arm] and run["train_models"] == [self.ms[i] for i in F["train"]]
                    S = z[f"D_test_{metric}"].astype(np.float64)
                    assert np.array_equal(z["y_test"], np.repeat(np.arange(self.n), NC))
                    assert np.array_equal(z["c_test"], np.tile(np.arange(NC), self.n))
                    r = self.rows(F["heldout"])
                    D[r] = S[r]
                    ok = (np.argmin(S, 1) == z["y_test"]).reshape(self.n, NC).astype(float)
                    seen[F["fold"], F["train"]] = ok[F["train"]]
                assert np.isfinite(D).all()
                c = L.counts(D, self.n, self.ia, self.ib)
                self._keep(key + (s,), c, seen)
                cms.append(c["cnt_model"])
                cps.append(c["cnt_pair_all"])
                seens.append(seen)
            self.seen[key] = np.stack(seens)
            self.units[key] = Unit(np.stack(cms), np.stack(cps))
        return self.units[key]

    def _keep(self, key, c, seen):
        p = "|".join(map(str, key))
        self.store[f"{p}|cnt_model"] = c["cnt_model"]
        self.store[f"{p}|cnt_pair_all"] = c["cnt_pair_all"]
        self.store[f"{p}|cnt_model_seen"] = seen
        if key[0] == "nc":
            self.seen[key] = seen[None]

    def unit(self, readout, kind, arm, alt=False):
        if readout == "enc":
            return self.enc(kind, arm, "cos" if alt else "euc")
        return self.nc(kind, arm, ("raw" if self.var == "sf" else "sf") if alt else self.var)

    def seen_level(self, readout, kind, arm):
        self.unit(readout, kind, arm)
        key = ("enc", kind, arm, "euc") if readout == "enc" else ("nc", kind, arm, self.var, L.K)
        s = self.seen[key]                                    # (R, NF, n, NC), NaN where held out
        return r4(np.nanmean(s))


# ------------------------------------------------------------------ references (no new run)
def ceiling(view):
    d = json.load(open("./results/D031/posthoc_contrasts.json"))["views"][view]["cells"]["MMR* - PAPER8"]
    return {f"{ro}|{arm}": dict(O_all=d[ro]["d_all"][ab], O_tail=d[ro]["d_tail"][ab])
            for ro in ("att", "lin") for arm, ab in (("FULL", "a"), ("PAPER8", "b"))}


def llmmap_os(view, ms, ia, ib, B):
    z = np.load(f"./data/baseline_ext_v1/derived/os_{view}.npz")
    assert list(z["models"]) == ms
    T = z["template"]
    X = z["test"].reshape(-1, T.shape[1]).astype(np.float64)
    from scipy.spatial.distance import cdist
    ref = json.load(open("./results/D030/analysis.json"))["secondary"]["d_llmmap_os"]["rows"]
    out = {}
    for metric, nm in (("cosine", "cos"), ("euclidean", "euc")):
        c = L.counts(cdist(X, T, metric=metric), len(ms), ia, ib)
        u = Unit(c["cnt_model"][None], c["cnt_pair_all"][None])
        lv = level(u, B)
        assert abs(lv["O_all"] - ref[f"{nm}|All|{view}"]["E_all"]) < 1e-4, (view, nm, lv, ref[f"{nm}|All|{view}"])
        out[nm] = dict(lv, d030_E_all=ref[f"{nm}|All|{view}"]["E_all"],
                       in_sample=ref[f"{nm}|All|{view}"]["in_sample"], out_of_sample=ref[f"{nm}|All|{view}"]["out_of_sample"])
    return out


def nprime_cols(view, ms, ia, ib):
    if view == "v1":
        return sorted(near_relative_pairs(ms))
    pid = {(ms[a], ms[b]): j for j, (a, b) in enumerate(zip(ia, ib))}
    cols = []
    for p in json.load(open("./results/D027/n_prime.json"))["pairs"]:
        a, b = p["pair"].split(" | ")
        cols.append(pid[(a, b)] if (a, b) in pid else pid[(b, a)])
    return sorted(cols)


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    ch = json.load(open(f"{RES}/chains.json"))
    vc = json.load(open(f"{RES}/variant_choice.json"))
    out = dict(schema="d033-analysis-v1", chains_sha256=L31.sha(f"{RES}/chains.json"),
               variant_choice_sha256=L31.sha(f"{RES}/variant_choice.json"), bootstrap=dict(
                   NB=NB, configs="d008_lib.boot_draws(25, 2000, seed_view) (D031)",
                   seeds="default_rng(sha256('D033|seeds|<view>')).integers(0, 3, (2000, 3)), shared by arms and folds"),
               readouts=dict(enc="(1) LLMmap open-set encoder, Euclidean (Call 1 (a))",
                             nc="(2) nearest point cloud, S1 variant per view"),
               views={})
    plot = {}
    worse_primary, worse_secondary = [], []
    for view in L.VIEWS:
        V = View(view, ch, vc)
        B = V.B
        R = dict(variant=V.var, levels={}, seen={}, primary={}, secondary={})
        # 1-2. levels and tail-plot data first
        for kind in L.KINDS:
            for ro in ("enc", "nc"):
                for arm in L.ARMS:
                    u = V.unit(ro, kind, arm)
                    R["levels"][f"{ro}|{kind}|{arm}"] = level(u, B)
                    R["seen"][f"{ro}|{kind}|{arm}"] = dict(seen_O_all=V.seen_level(ro, kind, arm),
                                                           heldout_O_all=R["levels"][f"{ro}|{kind}|{arm}"]["O_all"])
                    if kind == "model":
                        plot[f"{view}|{ro}|{arm}|pair_acc_sorted"] = np.sort(u.stats(B)["pa"][1])
                for arm in L.ARMS:
                    alt = V.unit(ro, kind, arm, alt=True)
                    R["levels"][f"{ro}-alt|{kind}|{arm}"] = level(alt, B)
        # 3-4. primary cells
        for ro in ("enc", "nc"):
            c = contrast(V.unit(ro, "model", "OURS"), V.unit(ro, "model", "PAPER8"), B)
            R["primary"][ro] = c
            if c["label"] == "WORSE":
                worse_primary.append(f"{view} {ro} OURS^fold - PAPER8")
            print(f"[S3 {view}] PRIMARY {ro}: OURS^fold - PAPER8  all {c['d_all']['delta']:+.4f} {c['d_all']['ci']} "
                  f"tail {c['d_tail']['delta']:+.4f} {c['d_tail']['ci']} -> {c['label']}", flush=True)
        # secondary labelled contrasts
        sec = {}
        for ro in ("enc", "nc"):
            for kind in L.KINDS:
                pairs = [("FULL", "OURS"), ("JG", "PAPER8"), ("OURS", "JG")]
                if kind == "family":
                    pairs = [("OURS", "PAPER8")] + pairs
                for a, b in pairs:
                    sec[f"{ro}|{kind}|{a} - {b}"] = contrast(V.unit(ro, kind, a), V.unit(ro, kind, b), B)
                sec[f"{ro}-alt|{kind}|OURS - PAPER8"] = contrast(V.unit(ro, kind, "OURS", True),
                                                                  V.unit(ro, kind, "PAPER8", True), B)
        for k_, c in sec.items():
            if c["label"] == "WORSE":
                worse_secondary.append(f"{view} {k_}")
        R["secondary"]["labelled"] = sec
        for k_ in ("enc|model|FULL - OURS", "nc|model|FULL - OURS", "enc-alt|model|OURS - PAPER8",
                   "nc-alt|model|OURS - PAPER8", "enc|family|OURS - PAPER8", "nc|family|OURS - PAPER8"):
            c = sec[k_]
            print(f"[S3 {view}] {k_}: all {c['d_all']['delta']:+.4f} {c['d_all']['ci']} tail "
                  f"{c['d_tail']['delta']:+.4f} {c['d_tail']['ci']} -> {c['label']}", flush=True)
        # (b) references
        R["secondary"]["ceiling_closed_set_D031"] = ceiling(view)
        R["secondary"]["llmmap_os_D030"] = llmmap_os(view, V.ms, V.ia, V.ib, B)
        # (c) curves under (2), all four arms, model-level folds
        cur = {}
        for arm in L.ARMS:
            pts = []
            for k in range(1, L.K + 1):
                u = V.nc("model", arm, V.var, k)
                s = u.stats(B)
                pts.append([k, s["all"][1], s["tail"][1]])
            cur[arm] = pts
            plot[f"{view}|curve|{arm}"] = np.array(pts)
        R["secondary"]["curves_nc"] = cur
        # (d) near-relatives
        npc = nprime_cols(view, V.ms, V.ia, V.ib)
        other = np.setdiff1d(np.arange(V.P), npc)
        nr = {}
        for ro in ("enc", "nc"):
            for arm in L.ARMS:
                u = V.unit(ro, "model", arm)
                (dn, pn), (do, po) = u.sub(B, npc), u.sub(B, other)
                pa = u.stats(B)["pa"][1]
                worst = np.argsort(pa, kind="stable")[:max(1, int(np.floor(0.1 * V.P)))]
                cm = u.cm.mean(0)
                sibm = V.ch["views"][view]["kinds"]["model"]["sibling_enrolled"]
                ws = [i for i, m in enumerate(V.ms) if sibm[m]]
                wo = [i for i, m in enumerate(V.ms) if not sibm[m]]
                nr[f"{ro}|{arm}"] = dict(
                    nprime_pair_acc=r4(pn), other_pair_acc=r4(po), diff=r4(pn - po), diff_ci=ci(dn - do),
                    worst_decile_nprime_share=r4(np.isin(worst, npc).mean()), nprime_share_all=r4(len(npc) / V.P),
                    top1_sibling_enrolled=r4(cm[ws].sum() / (len(ws) * NC)) if ws else None,
                    top1_no_sibling=r4(cm[wo].sum() / (len(wo) * NC)) if wo else None,
                    n_sibling_enrolled=len(ws), n_no_sibling=len(wo))
        R["secondary"]["near_relatives"] = dict(n_nprime=len(npc), rows=nr)
        out["views"][view] = R
        np.savez_compressed(f"{RES}/counts_{view}.npz", **V.store)
        print(f"[S3 {view}] done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    v2e = out["views"]["v2"]["primary"]["enc"]["label"]
    claim = "SUPPORTED" if (v2e == "BETTER" and not worse_primary) else "NOT SUPPORTED"
    out["worse_scan"] = dict(primary=worse_primary, secondary=worse_secondary,
                             scope="claim rule reads the 4 primary cells only (P1 §6, Review Disclosure 9)")
    out["claim"] = dict(rule="BETTER under (1) in v2 and no WORSE in the 4 primary OURS^fold - PAPER8 cells",
                        v2_enc_label=v2e, verdict=claim)
    # A1 encoder diagnostics
    import glob
    runs = [json.load(open(p)) for p in sorted(glob.glob(f"{RES}/runs/*.json"))]
    a1 = {}
    for r in runs:
        a1.setdefault(f"{r['view']}|{r['kind']}|{r['arm']}", []).append(r)
    out["a1_encoder"] = {k: dict(n=len(v), best_epoch=sorted(x["best_epoch"] for x in v),
                                 epochs_run=sorted(x["epochs_run"] for x in v),
                                 s_val_pair_acc_at_best=[r4(x["s_val_pair_acc_at_best"]) for x in v],
                                 share_best_epoch_le2=r4(np.mean([x["best_epoch"] <= 2 for x in v])),
                                 no_improvement=sum(x["no_improvement_after_first_epoch"] for x in v),
                                 train_min=r4(np.mean([x["wall_train_min"] for x in v])))
                         for k, v in sorted(a1.items())}
    out["union"] = dict(n=ch["union"]["n"], not_in_d031_or_d032=ch["union"]["not_in_d031_or_d032"])
    out["wall_s"] = round(time.time() - t0, 1)
    L.jdump(out, f"{RES}/analysis.json")
    np.savez_compressed(f"{RES}/plotdata.npz", **plot)
    print(f"[S3] WORSE primary {worse_primary}; secondary {worse_secondary}", flush=True)
    print(f"[S3] claim: {claim} (v2 enc {v2e})", flush=True)


if __name__ == "__main__":
    main()
