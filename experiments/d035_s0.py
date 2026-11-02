"""D035 / S0 — inventory, target-free (run before the P1 Review, as D024's S0 was).

(1) SINGLE reference config per model under D024's tier rule, both readings ("effective": greedy = temperature 0,
    the D's SINGLE; "field": the literal temperature field, SINGLE-field), plus P1 Call 1's "greedy_first" reading
    (SINGLE-greedyfirst, selection only). Stop condition: a model with no config in any tier.
(2) 259 x 3,570 per-query squared-distance arrays from S_build: CENTROID, SINGLE (effective), SINGLE-field, each
    raw and median-normalised. Accumulator == direct concatenation asserted for CENTROID. Health rows per array
    (median spread, exact-zero / near-zero cells, the k = 1 objective's argmax uniqueness). Stop conditions: a
    query whose median pair distance is 0 (normalisation undefined), a non-unique k = 1 argmax.
(3) Code-path check: d024_lib.greedy_cvar + cloud_hooks (D010 / D029 accumulators) must reproduce D029's v2 JG
    chain k = 1..8 with per-step objective difference 0.
(4) Inventory: D029 CLOUD / PAPER8 counts and logits (attention seeds 0-4, linear, k = 1..8), D031's v2 bootstrap
    seed, D027 N', D029 hard set, D024's v1 results.
No chain of a new arm is selected here (S1, after the Review), and no S_val / S_test trace is loaded.
Writes results/D035/s0.json and results/D035/distances.npz (gitignored; sha in s0.json).
Usage:  PYTHONPATH=.:experiments python experiments/d035_s0.py
"""
import os
import json
import time

import numpy as np

import d031_lib as L31
import d029_lib as L29
import d035_lib as L
from LLMmap.joint_statistic import cvar
from d010_select import precompute
from d024_lib import greedy_cvar, cloud_hooks


def health(D, raw, ms, ia, ib, top=12):
    """Diagnostics for one (259, P) array; `raw` = its un-normalised squared distances."""
    med = np.median(raw, axis=1)
    zero = raw == 0
    near0 = raw <= 1e-6 * np.maximum(med[:, None], 1e-300)
    k1 = np.array([cvar(np.sqrt(D[q]), L.GAMMA) if np.isfinite(D[q]).all() else np.nan for q in range(D.shape[0])])
    best = np.nanmax(k1)
    per_pair = zero.sum(0)
    worst = np.argsort(-per_pair, kind="stable")[:top]
    return dict(
        median_d2_per_query=dict(min=float(med.min()), p50=float(np.median(med)), max=float(med.max()),
                                 max_over_min=float(med.max() / med.min()) if med.min() > 0 else None,
                                 n_zero_median=int((med == 0).sum())),
        exact_zero_cells=dict(total=int(zero.sum()), queries_with_any=int(zero.any(1).sum()),
                              queries_with_ge_m=int((zero.sum(1) >= 357).sum()), pairs_ever=int(zero.any(0).sum()),
                              max_per_pair=int(per_pair.max()), max_per_query=int(zero.sum(1).max())),
        near_zero_cells=int(near0.sum()),
        k1_objective=dict(argmax=int(np.nanargmax(k1)), max=float(best), n_at_max=int((k1 == best).sum()),
                          second=float(np.sort(k1[np.isfinite(k1)])[-2]), n_zero=int((k1 <= 0).sum()),
                          n_nan=int(np.isnan(k1).sum())),
        most_colliding_pairs=[dict(pair=f"{ms[ia[p]]} | {ms[ib[p]]}", zero_queries=int(per_pair[p]))
                              for p in worst if per_pair[p] > 0])


def main():
    t0 = time.time()
    os.makedirs(L.OUT, exist_ok=True)
    stops = []
    ms = L.models()
    n = len(ms)
    ia, ib = L31.pairs(L.VIEW)
    P = len(ia)
    assert (n, P) == (85, 3570)
    out = dict(schema="d035-s0-v1", view=L.VIEW, models=ms, n_pairs=P, n_queries=L29.NPOOL,
               cvar_m=max(1, int(np.floor(L.GAMMA * P))), code_commit=L31.git_head())
    X, axis = L.build_cubes(ms)
    out["config_axis_is_0_74"] = sum(axis[m] == list(range(75)) for m in ms)
    print(f"[S0] S_build cubes loaded ({(time.time() - t0) / 60:.1f} min); config axis 0..74 for "
          f"{out['config_axis_is_0_74']} / {n} models", flush=True)

    # ---- (1) SINGLE tiers
    pcs = L.build_prompt_confs(ms, axis)
    tiers = {}
    for rd in ("effective", "field"):
        ch = L.single_choice(pcs, ms, rd)
        miss = [m for m in ms if ch[m] is None]
        if miss:
            stops.append(f"SINGLE ({rd}): no config in any tier for {miss}")
        tiers[rd] = {m: dict(config_index=int(c), tier=int(t), do_sample=pc["sampling_hparams"]["do_sample"],
                             temperature=pc["sampling_hparams"]["temperature"], system_prompt=pc["system_prompt"])
                     for m, (c, t, pc) in ((m, ch[m]) for m in ms if ch[m] is not None)}

    def plain(pc, sys_):
        return pc["cot_prompt"] is None and pc["rag_prompt"] is None and (sys_ is None or pc["system_prompt"] == sys_)

    gf = L.single_choice_greedy_first(pcs, ms)
    tiers["greedy_first"] = {m: dict(config_index=int(c), tier=int(t), do_sample=pc["sampling_hparams"]["do_sample"],
                                     temperature=pc["sampling_hparams"]["temperature"],
                                     system_prompt=pc["system_prompt"]) for m, (c, t, pc) in gf.items()}
    out["single_greedy_first_differs"] = sorted(m for m in ms if tiers["greedy_first"][m]["config_index"]
                                                != tiers["effective"][m]["config_index"])
    tier_sets = {m: dict(
        t1=sorted([c, pc["sampling_hparams"]["do_sample"], pc["sampling_hparams"]["temperature"]]
                  for c, pc in pcs[m].items() if plain(pc, "")),
        t2=sorted([c, pc["sampling_hparams"]["do_sample"], pc["sampling_hparams"]["temperature"]]
                  for c, pc in pcs[m].items() if plain(pc, "You are a helpful assistant.")),
        t3_n=sum(1 for pc in pcs[m].values() if plain(pc, None)),
        greedy_n=sum(1 for pc in pcs[m].values() if not pc["sampling_hparams"]["do_sample"]),
        plain_greedy=sorted([c, pc["system_prompt"][:60]] for c, pc in pcs[m].items()
                            if plain(pc, None) and not pc["sampling_hparams"]["do_sample"]))
        for m in ms}
    out["single_tier"] = dict(
        readings=tiers,
        tier_counts={rd: {str(t): sum(1 for v in tiers[rd].values() if v["tier"] == t) for t in (1, 2, 3)}
                     for rd in tiers},
        sampling_of_choice={rd: sorted({(v["do_sample"], v["temperature"]) for v in tiers[rd].values()},
                                       key=str) for rd in tiers},
        tier1_size_hist={str(s): sum(1 for m in ms if len(tier_sets[m]["t1"]) == s)
                         for s in sorted({len(v["t1"]) for v in tier_sets.values()})},
        tier1_with_greedy=sum(1 for m in ms if any(not ds for _, ds, _ in tier_sets[m]["t1"])),
        tier_sets=tier_sets,
        same_config_index_all_models={rd: len({v["config_index"] for v in tiers[rd].values()}) == 1 for rd in tiers})
    print(f"[S0] SINGLE tiers {out['single_tier']['tier_counts']}; chosen sampling "
          f"{out['single_tier']['sampling_of_choice']}; tier-1 size histogram {out['single_tier']['tier1_size_hist']}; "
          f"tier 1 has a greedy config for {out['single_tier']['tier1_with_greedy']} / {n}", flush=True)

    # ---- (2) distance arrays
    t = time.time()
    cen = L.centroid_points(X, ms)
    raw = {"CENTROID": L.d2_chunked(cen, ia, ib)}
    qs = [3, 11, 40, 77, 193]
    for p in (0, P // 2, P - 1):
        direct = float(np.sum((np.concatenate(cen[ia[p], qs]) - np.concatenate(cen[ib[p], qs])) ** 2))
        assert abs(direct - raw["CENTROID"][qs, p].sum()) < 1e-9 * direct, (p, direct)
    del cen
    for rd, arm in (("effective", "SINGLE"), ("field", "SINGLE-field"), ("greedy_first", "SINGLE-greedyfirst")):
        if len(tiers[rd]) == n:
            cfg = {m: tiers[rd][m]["config_index"] for m in ms}
            raw[arm] = L.d2_chunked(L.single_points(X, ms, axis, cfg), ia, ib)
    arrays, hl = {}, {}
    for arm, R in raw.items():
        N, med = L.normalise(R)
        arrays[f"{arm}|raw"], arrays[f"{arm}|norm"], arrays[f"{arm}|median"] = R, N, med
        for sc, D in (("raw", R), ("norm", N)):
            h = health(D, R, ms, ia, ib)
            hl[f"{arm}|{sc}"] = h
            h["zero_median_queries"] = [int(q) for q in np.where(med == 0)[0]]
            if h["median_d2_per_query"]["n_zero_median"] and sc == "norm":
                stops.append(f"{arm}|norm: {h['median_d2_per_query']['n_zero_median']} queries with median d2 = 0")
            if h["k1_objective"]["n_at_max"] != 1:
                stops.append(f"{arm}|{sc}: k = 1 argmax not unique ({h['k1_objective']['n_at_max']} queries)")
            print(f"[S0] {arm}|{sc}: median d2 {h['median_d2_per_query']}; zero cells {h['exact_zero_cells']}; "
                  f"k=1 {h['k1_objective']}", flush=True)
    if "SINGLE" in raw and "SINGLE-field" in raw:
        out["single_effective_vs_field_identical_points"] = bool(np.array_equal(raw["SINGLE"], raw["SINGLE-field"]))
    out["health"] = hl
    np.savez_compressed(f"{L.OUT}/distances.npz", **arrays, ia=ia, ib=ib, models=np.array(ms))
    out["distances_npz_sha256"] = L31.sha(f"{L.OUT}/distances.npz")
    out["distances_wall_min"] = round((time.time() - t) / 60, 2)

    # ---- (3) code-path check vs D029 JG
    t = time.time()
    d29 = json.load(open("./results/D029/chains.json"))["arms"]["JG"]
    pairs = [(ms[a], ms[b]) for a, b in zip(ia, ib)]
    XX2, XY2 = precompute(X, ms, pairs)
    del X
    sel, trace = greedy_cvar(L29.NPOOL, *cloud_hooks(XY2, XX2, ia, ib), k=L.K, gamma=L.GAMMA)
    del XY2, XX2
    diffs = [abs(a["objective_cvar"] - b["objective_cvar"]) for a, b in zip(trace, d29["trace"][:L.K])]
    ok = sel == d29["ids"][:L.K] and max(diffs) == 0.0
    if not ok:
        stops.append(f"code-path check failed: {sel} vs D029 {d29['ids'][:L.K]}, max |dobj| {max(diffs)}")
    out["code_path_check"] = dict(chain=sel, d029_chain=d29["ids"][:L.K], identical=sel == d29["ids"][:L.K],
                                  per_step_abs_objective_diff=diffs, trace=trace,
                                  wall_min=round((time.time() - t) / 60, 2))
    print(f"[S0] code-path check: {sel} vs D029 {d29['ids'][:L.K]}; per-step |dobj| {diffs} -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)

    # ---- (4) inventory
    inv = {}
    for arm in ("JG", "PAPER8"):
        z = np.load(f"./results/D029/counts_{arm}.npz")
        lg = np.load(f"./results/D029/logits_{arm}.npz")
        inv[f"D029_{arm}"] = dict(
            att_counts=all(f"att|{k}|{r}|{c}" in z.files for k in range(1, 9) for r in range(5)
                           for c in ("cnt_total", "cnt_model")),
            lin_counts=all(f"lin|{k}|{c}" in z.files for k in range(1, 9) for c in ("cnt_total", "cnt_model")),
            att_logits=all(f"att|{k}|{r}" in lg.files for k in range(1, 9) for r in range(5)),
            lin_logits=all(f"lin|{k}" in lg.files for k in range(1, 9)),
            logits_order_ok=bool(np.array_equal(lg["y"], np.repeat(np.arange(n), 25))
                                 and np.array_equal(lg["cfg"], np.tile(np.arange(25), n))))
    tg = L29.targets(ms)
    inv["D029_hard"] = {k: len(v) for k, v in tg["strata"].items()}
    inv["D027_n_prime_pairs"] = len(json.load(open("./results/D027/n_prime.json"))["pairs"])
    inv["D031_v2_boot_seed"] = L31.VIEW["v2"]["boot_seed"]
    inv["D031_v2_n_slot"] = L31.VIEW["v2"]["n_slot"]
    inv["D024_files"] = {f: os.path.exists(f"./results/D024/{f}") for f in ("analysis.json", "selection.json",
                                                                           "run_counts.npz", "linear_counts.npz")}
    out["inventory"] = inv
    print(f"[S0] inventory {json.dumps(inv)}", flush=True)
    out["stops"] = stops
    out["wall_min"] = round((time.time() - t0) / 60, 2)
    L31.jdump(out, f"{L.OUT}/s0.json")
    print(f"[S0] wrote {L.OUT}/s0.json ({out['wall_min']} min); stop conditions: {stops if stops else 'none'}",
          flush=True)


if __name__ == "__main__":
    main()
