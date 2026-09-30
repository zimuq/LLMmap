"""D025 / S3 — labels and secondaries (D025 Review 2026-09-29, APPROVED WITH AMENDMENTS).

Quantities per H pair p, readout r:  a(p) = acc_SPEC(p)(p,8) - acc_GLOBAL(p,8)
                                      b(p) = acc_SPEC(p)(p,16) - acc_GLOBAL(p,8)
Labels (M = 0.05): CEILING_r(p) if fixed pre-training (selection.json); else
  BUDGET-LIMITED  a >= +0.05 and lo(a) > 0
  POOL-LIMITED    hi(b) < +0.05
  CONFLICT        both;   INCONCLUSIVE  neither.
Pooled label per readout over its non-CEILING pairs (linear = primary).
Bootstrap (Review Call 3): linear configs M = boot_draws(25, 2000, seed=20260929),
shared everywhere; attention hierarchical, same M, seed slots
S = default_rng(20260929).integers(0, 5, (2000, 16, 5)), arm X at k uses S[:, k-1]
(joint within k, independent across k). 5-with-replacement understates seed
variance by 4/5 (disclosed).
Secondaries: A1 k-to-0.95; A2 stably-hard co-report; (a) family-restricted recall
FAM vs GLOBAL; (b) directions; (c) cross-specialist placebo; (d) GLOBAL k=9..16 +
tail plot; (e) S_val linear; (f) stably-hard pooled; (g) chains / by-products.
Usage:  PYTHONPATH=.:experiments python experiments/d025_analysis.py
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from d008_lib import boot_draws
from d025_lib import OUT, K_MAX
from d025_train import safe

NCFG, NB, SEED, MARGIN = 25, 2000, 20260929, 0.05
KS = list(range(1, K_MAX + 1))


def pct(x):
    return [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]


def label(a_pt, a_ci, b_ci):
    bud = a_pt >= MARGIN and a_ci[0] > 0
    pool = b_ci[1] < MARGIN
    return "CONFLICT" if bud and pool else "BUDGET-LIMITED" if bud else \
        "POOL-LIMITED" if pool else "INCONCLUSIVE"


def main():
    sel = json.load(open(f"{OUT}/selection.json"))
    hs = json.load(open(f"{OUT}/hard_set.json"))
    models = json.load(open("./results/D008/selection.json"))["models"]
    import itertools
    pairs = [f"{a} | {b}" for a, b in itertools.combinations(models, 2)]
    H = hs["H"]
    hidx = {p: pairs.index(p) for p in H}
    ceil = sel["ceiling_and_attribution"]["pairs"]
    arms = dict(sel["arms"])
    alias = sel["fam_equals_spec"]                      # FAM arm -> SPEC arm
    src = {a: alias.get(a, a) for a in arms}
    C = {a: np.load(f"{OUT}/counts_{safe(a)}.npz") for a in sel["trained_arms"]}
    spec_of = {arms[a]["target"]: a for a in arms if arms[a]["kind"] == "SPEC"}
    fam_of = {arms[a]["target"]: a for a in arms if arms[a]["kind"] == "FAM"}
    M = boot_draws(NCFG, NB, seed=SEED)
    tot = M.sum(1)
    S = np.random.default_rng(SEED).integers(0, 5, (NB, K_MAX, 5))

    def get(arm, key):
        return C[src[arm]][key]

    # ---------- per-pair accuracy: point + draws
    def pair_cnt(arm, r, k, i, split="test", seed=None):
        if r == "lin":
            return get(arm, f"lin|{split}|{k}|dir666")[i].sum(0).astype(float)          # (25,)
        return np.stack([get(arm, f"att|{k}|{s}|dir666")[i].sum(0) for s in range(5)]).astype(float)

    def pair_acc(arm, r, k, i, split="test"):
        c = pair_cnt(arm, r, k, i, split)
        if r == "lin":
            return float(c.sum() / (2 * NCFG)), (M @ c) / (2 * tot)
        per_seed = (M @ c.T) / (2 * tot[:, None])
        return float(c.sum(1).mean() / (2 * NCFG)), np.take_along_axis(per_seed, S[:, k - 1], 1).mean(1)

    # ---------- identical-input k vs GLOBAL: counts must match exactly
    ident_ok = {}
    for a, ks in sel["identical_input_k_vs_GLOBAL"].items():
        for k in ks:
            for key in ("dir666", "cnt_total"):
                assert np.array_equal(C[a][f"lin|test|{k}|{key}"], C["GLOBAL"][f"lin|test|{k}|{key}"]), (a, k)
                for s in range(5):
                    assert np.array_equal(C[a][f"att|{k}|{s}|{key}"], C["GLOBAL"][f"att|{k}|{s}|{key}"]), (a, k, s)
        ident_ok[a] = ks

    # ---------- CEILING check: retrained GLOBAL k=8 == the pre-training values
    for p in H:
        lv, _ = pair_acc("GLOBAL", "lin", 8, hidx[p])
        av, _ = pair_acc("GLOBAL", "att", 8, hidx[p])
        assert abs(lv - ceil[p]["linear_global_k8"]) < 1e-4 and abs(av - ceil[p]["attention_global_k8"]) < 1e-4, p

    # ---------- primary + secondary labels
    labels, curves = {}, {}
    for r in ("lin", "att"):
        per, draws_a, draws_b = {}, {}, {}
        for p in H:
            i, sp = hidx[p], spec_of[p]
            g8, g8d = pair_acc("GLOBAL", r, 8, i)
            s8, s8d = pair_acc(sp, r, 8, i)
            s16, s16d = pair_acc(sp, r, 16, i)
            a_pt, b_pt = s8 - g8, s16 - g8
            a_ci, b_ci = pct(s8d - g8d), pct(s16d - g8d)
            is_ceil = ceil[p]["ceiling_linear" if r == "lin" else "ceiling_attention"]
            lab = "CEILING" if is_ceil else label(a_pt, a_ci, b_ci)
            per[p] = dict(global_k8=round(g8, 4), spec_k8=round(s8, 4), spec_k16=round(s16, 4),
                          a=dict(point=round(a_pt, 4), ci=a_ci), b=dict(point=round(b_pt, 4), ci=b_ci),
                          ceiling=bool(is_ceil), label=lab,
                          label_if_no_ceiling_rule=label(a_pt, a_ci, b_ci),
                          attribution=ceil[p]["attribution"])
            if r == "att" and ceil[p]["attribution"] == "READOUT-LIMITED" and lab == "BUDGET-LIMITED":
                per[p]["note"] = ("READOUT-LIMITED pair: read as 'dedicated queries partly compensate "
                                  "for the attention network', not budget scarcity (Review Call 1)")
            if not is_ceil:
                draws_a[p], draws_b[p] = s8d - g8d, s16d - g8d
        pooled = None
        if draws_a:
            pa = np.mean(list(draws_a.values()), 0)
            pb = np.mean(list(draws_b.values()), 0)
            a_pt = float(np.mean([per[p]["a"]["point"] for p in draws_a]))
            b_pt = float(np.mean([per[p]["b"]["point"] for p in draws_a]))
            pooled = dict(pairs=list(draws_a), a=dict(point=round(a_pt, 4), ci=pct(pa)),
                          b=dict(point=round(b_pt, 4), ci=pct(pb)), label=label(a_pt, pct(pa), pct(pb)))
        labels[r] = dict(per_pair=per, pooled=pooled)
        # per-k curves for every arm on every H pair
        curves[r] = {p: {a: [round(pair_acc(a, r, k, hidx[p])[0], 4) for k in KS] for a in arms} for p in H}
    disagree = {p: (labels["lin"]["per_pair"][p]["label"], labels["att"]["per_pair"][p]["label"])
                for p in H if labels["lin"]["per_pair"][p]["label"] != labels["att"]["per_pair"][p]["label"]}
    for r in labels:
        print(f"[S3] {r}: " + "; ".join(f"{p.split(' | ')[0].split('/')[1]}: {v['label']} a {v['a']} b {v['b']}"
                                        for p, v in labels[r]["per_pair"].items())
              + f"\n     pooled {labels[r]['pooled']}", flush=True)

    # ---------- A1: smallest k with acc >= 0.95
    k95 = {r: {p: {a: next((k for k, v in zip(KS, curves[r][p][a]) if v >= 0.95), None) for a in arms}
               for p in H} for r in curves}

    # ---------- (a) family-restricted recall, FAM vs GLOBAL
    def fam_rec(arm, r, k, g):
        if r == "lin":
            c = get(arm, f"lin|test|{k}|fam|{g}").astype(float)                           # (|F|,25)
            return float(c.sum() / c.size), (M @ c.sum(0)) / (c.shape[0] * tot)
        c = np.stack([get(arm, f"att|{k}|{s}|fam|{g}").sum(0) for s in range(5)]).astype(float)
        nF = get(arm, f"att|{k}|0|fam|{g}").shape[0]
        per_seed = (M @ c.T) / (nF * tot[:, None])
        return float(c.sum(1).mean() / (nF * NCFG)), np.take_along_axis(per_seed, S[:, k - 1], 1).mean(1)

    fam = {}
    for g, a in arms.items():
        if a["kind"] != "FAM":
            continue
        fam[g] = {"models": a["models"], "H_pair": a["target"], "alias_of": alias.get(g)}
        for r in ("lin", "att"):
            g8, g8d = fam_rec("GLOBAL", r, 8, g)
            f8, f8d = fam_rec(g, r, 8, g)
            f16, f16d = fam_rec(g, r, 16, g)
            is_ceil = (1 - g8) < MARGIN
            a_ci, b_ci = pct(f8d - g8d), pct(f16d - g8d)
            fam[g][r] = dict(global_k8=round(g8, 4), fam_k8=round(f8, 4), fam_k16=round(f16, 4),
                             a=dict(point=round(f8 - g8, 4), ci=a_ci), b=dict(point=round(f16 - g8, 4), ci=b_ci),
                             ceiling=bool(is_ceil),
                             label_descriptive="CEILING" if is_ceil else label(f8 - g8, a_ci, b_ci),
                             curve_fam=[round(fam_rec(g, r, k, g)[0], 4) for k in KS],
                             curve_global=[round(fam_rec("GLOBAL", r, k, g)[0], 4) for k in KS])
    print(f"[S3] (a) families: " + "; ".join(f"{g} lin {v['lin']['label_descriptive']} a {v['lin']['a']} | "
                                            f"att {v['att']['label_descriptive']} a {v['att']['a']}"
                                            for g, v in fam.items()), flush=True)

    # ---------- (b) directions (k=8, 16) for GLOBAL / SPEC / FAM on each H pair
    def dir_acc(arm, r, k, i):
        if r == "lin":
            c = get(arm, f"lin|test|{k}|dir666")[i]
            return [round(float(x), 4) for x in c.sum(1) / NCFG]
        c = np.stack([get(arm, f"att|{k}|{s}|dir666")[i] for s in range(5)])
        return [round(float(x), 4) for x in c.sum(2).mean(0) / NCFG]
    directions = {r: {p: {a: {k: dir_acc(a, r, k, hidx[p]) for k in (8, 16)}
                          for a in ("GLOBAL", spec_of[p], fam_of[p])} for p in H} for r in ("lin", "att")}

    # ---------- (c) cross-specialist placebo
    placebo = {r: {} for r in ("lin", "att")}
    for r in placebo:
        for p in H:
            i = hidx[p]
            g8, g8d = pair_acc("GLOBAL", r, 8, i)
            row = {}
            for q in H:
                if q == p:
                    continue
                for k in (8, 16):
                    v, d = pair_acc(spec_of[q], r, k, i)
                    row[f"{spec_of[q]}@{k}"] = dict(acc=round(v, 4), minus_global_k8=round(v - g8, 4),
                                                   ci=pct(d - g8d))
            placebo[r][p] = row

    # ---------- (d) GLOBAL k=1..16 + A2 co-report
    sets = dict(M_H=hs["M_H"], F_H=hs["F_H"],
                stably_hard_models=sorted({m for p in hs["H_stably_hard"] for m in p.split(" | ")}),
                stably_hard_families=sorted({m for g, hp in zip(hs["family_groups"], hs["family_group_H_pairs"])
                                             if hp[0] in hs["H_stably_hard"] for m in g}))
    glob = {}
    for r in ("lin", "att"):
        rows = {}
        for k in KS:
            if r == "lin":
                cm = [get("GLOBAL", f"lin|test|{k}|cnt_model")]
            else:
                cm = [get("GLOBAL", f"att|{k}|{s}|cnt_model") for s in range(5)]
            per_seed = {n: [float(c[[models.index(m) for m in ms]].mean()) for c in cm] for n, ms in sets.items()}
            top1 = [float(c.mean()) for c in cm]
            rows[k] = dict(mean_top1=round(float(np.mean(top1)), 4),
                           **{n: round(float(np.mean(v)), 4) for n, v in per_seed.items()},
                           seed_range_top1=[round(min(top1), 4), round(max(top1), 4)] if r == "att" else None)
        glob[r] = rows
    print(f"[S3] (d) GLOBAL k=8 vs 16: " + "; ".join(f"{r}: {glob[r][8]} -> {glob[r][16]}" for r in glob), flush=True)

    # tail plot: 666-pair two-logit accuracy, GLOBAL k=8 vs 16
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    tail = {}
    for j, r in enumerate(("lin", "att")):
        for k in (8, 16):
            if r == "lin":
                acc = get("GLOBAL", f"lin|test|{k}|dir666").sum((1, 2)) / (2 * NCFG)
            else:
                acc = np.mean([get("GLOBAL", f"att|{k}|{s}|dir666").sum((1, 2)) for s in range(5)], 0) / (2 * NCFG)
            s_ = np.sort(acc)
            tail[f"{r}|{k}"] = dict(worst10=[round(float(x), 3) for x in s_[:10]],
                                    n_below_0p9=int((s_ < 0.9).sum()),
                                    cvar01=round(float(s_[:66].mean()), 4),
                                    worst_pair=pairs[int(np.argmin(acc))])
            ax[j].plot(np.arange(1, 101), s_[:100], label=f"GLOBAL k={k}")
        ax[j].axvline(66, ls=":", c="gray")
        ax[j].set(title=f"{'linear concat' if r == 'lin' else 'attention'}: worst 100 of 666 pairs",
                  xlabel="pair rank (worst first)", ylabel="two-logit accuracy (S_test)")
        ax[j].legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/tail_k8_k16.png", dpi=120)

    # curves figure: one panel per H pair, both readouts
    fig, axs = plt.subplots(1, len(H), figsize=(4.2 * len(H), 3.8), sharey=True)
    for ax_, p in zip(np.atleast_1d(axs), H):
        for a, col in (("GLOBAL", "k"), (spec_of[p], "C3"), (fam_of[p], "C0")):
            for r, ls in (("lin", "-"), ("att", "--")):
                ax_.plot(KS, curves[r][p][a], ls, c=col, label=f"{a.split(':')[0]} {r}")
        ax_.axhline(0.95, c="gray", lw=0.6)
        ax_.set(title=p.replace(" | ", "\n").replace("-Instruct", ""), xlabel="k")
        ax_.title.set_fontsize(8)
    np.atleast_1d(axs)[0].set_ylabel("two-logit accuracy (S_test)")
    np.atleast_1d(axs)[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/curves.png", dpi=120)

    # ---------- (e) S_val linear
    sval = {}
    for p in H:
        i, sp = hidx[p], spec_of[p]
        g8, g8d = pair_acc("GLOBAL", "lin", 8, i, "val")
        s8, s8d = pair_acc(sp, "lin", 8, i, "val")
        s16, s16d = pair_acc(sp, "lin", 16, i, "val")
        sval[p] = dict(global_k8=round(g8, 4), a=round(s8 - g8, 4), b=round(s16 - g8, 4),
                       a_ci=pct(s8d - g8d), b_ci=pct(s16d - g8d))

    # ---------- (f) stably-hard pooled (ignores CEILING, reported raw; plus the ceiling-respecting set)
    stab = {}
    for r in ("lin", "att"):
        da, db, pa_, pb_ = [], [], [], []
        for p in hs["H_stably_hard"]:
            i, sp = hidx[p], spec_of[p]
            g8, g8d = pair_acc("GLOBAL", r, 8, i)
            s8, s8d = pair_acc(sp, r, 8, i)
            s16, s16d = pair_acc(sp, r, 16, i)
            da.append(s8d - g8d); db.append(s16d - g8d); pa_.append(s8 - g8); pb_.append(s16 - g8)
        stab[r] = dict(pairs=hs["H_stably_hard"], a=dict(point=round(float(np.mean(pa_)), 4), ci=pct(np.mean(da, 0))),
                       b=dict(point=round(float(np.mean(pb_)), 4), ci=pct(np.mean(db, 0))),
                       note="raw pooled over stably-hard pairs regardless of CEILING")

    out = dict(
        schema="d025-analysis-v1",
        primary=dict(readout="linear concat (Review Call 1: stays primary)",
                     population=sel["ceiling_and_attribution"]["primary_linear_population"],
                     pooled_label=labels["lin"]["pooled"] and labels["lin"]["pooled"]["label"]),
        labels=labels, readout_label_disagreements=disagree,
        attribution={p: ceil[p] for p in H},
        bootstrap=dict(configs=f"boot_draws(25, {NB}, seed={SEED}) shared everywhere",
                       attention_seeds=f"default_rng({SEED}).integers(0,5,({NB},16,5)); arm at k uses S[:,k-1]",
                       note="5-with-replacement understates seed variance of a 5-seed mean by 4/5"),
        A1_k_to_095=k95, A2_sets=sets, secondary_a_family=fam, secondary_b_directions=directions,
        secondary_c_placebo=placebo, secondary_d_global=dict(by_k=glob, tail=tail),
        secondary_e_sval_linear=sval, secondary_f_stably_hard=stab,
        secondary_g=dict(identical_greedy_answers=hs["descriptive"]["byte_identical_greedy_answers_per_H_pair"],
                         peak_k=sel["peak_k"], overlap_with_GLOBAL_by_k=sel["overlap_with_GLOBAL_by_k"],
                         d022_union_flags=sel["d022_union_flags"], chains=sel["chains"]),
        identical_input_k_checked=ident_ok, curves=curves)
    json.dump(out, open(f"{OUT}/analysis.json", "w"), indent=1)
    print(f"[S3] A1 k-to-0.95: {k95}\n[S3] (e) S_val {sval}\n[S3] (f) {stab}\n[S3] tail {tail}\n"
          f"[S3] disagreements {disagree}\n[S3] wrote {OUT}/analysis.json", flush=True)


if __name__ == "__main__":
    main()
