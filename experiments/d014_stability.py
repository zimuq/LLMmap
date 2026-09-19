"""D014 / S3 (Arm B) — chain stability under build-config resampling. THIS DECIDES.

Per the 2026-09-19 review amendment, Arm A cannot separate H1 from H2: S1 showed
chains recur as FAMILIES across neighbouring gamma, so an unlucky family produces
exactly the label pattern a real threshold would. Only perturbing the data the
chains are built from separates them, which is what this does.

For each algorithm and gamma, draw R=30 half-subsamples of the BUILD configs
WITHOUT replacement (with-replacement would duplicate points and bias energy
distance), re-select the chain using ONLY that subsample, and score it with
D008's 1-NN proxy against the FULL build reference. Val and test are never seen
by selection (I2).

DECISION RULE, fixed before any run (review amendment), paired over the 30
resamples, at both k=4 and k=8:

    d_r = Q_r(gamma=0.25) - Q_r(gamma=0.5)

  SYSTEMATIC (-> H1)      CI excludes 0 AND mean d > 0.02, at BOTH k
  CHAIN-SPECIFIC (-> H2)  CI within +/-0.02, at BOTH k
  else UNRESOLVED

POSITIVE CONTROL: the paired contrast Q_r(0.25) - Q_r(1.0) must itself be
SYSTEMATIC at both k. If Arm B cannot reproduce the known gamma=1.0-vs-0.25 gap,
it has no power and is reported as uninformative -- whatever the main contrast
says.

Also trains a 20-run sanity check (review amendment): the proxy-best and
proxy-worst resampled chain at gamma=0.5 and gamma=0.25, GreedyCover, k=8, 5
seeds each. D009 validated the proxy ACROSS conditions; it was never tested for
ranking chains WITHIN a gamma, which is exactly what Arm B asks of it. If the
trained ordering disagrees in sign at either gamma, Arm B is reported as
proxy-level and unconfirmed.

Usage:  PYTHONPATH=.:experiments python experiments/d014_stability.py
"""
import os
import json
import time
import hashlib
import itertools

import numpy as np

from LLMmap.greedy_cover import greedy_cover
from LLMmap.joint_greedy import joint_greedy
from d008_lib import (load_tensor, build_dq, summed, near_relative_pairs,
                      condition_stats, nn_predict)
from d009_lib import load_query_embeddings, build_traces, hparams_from_shipped
from d009_train import run_one
from d007_lib import load_corpus
from d010_select import precompute

OUT = "./results/D014"
R = 30
GAMMAS = ["1.0", "0.75", "0.5", "0.35", "0.25", "0.1"]
KS = [4, 8]
SEED = 20260919
SMOKE = bool(os.environ.get("D014_SMOKE"))
if SMOKE:
    R, GAMMAS = 4, ["1.0", "0.5", "0.25"]


def proxy_quality(Dq, w2, queries, y_ev, y_rf):
    """D008's 1-NN proxy: mean top-1 over the eval pool, full build reference."""
    D = summed(Dq, w2, queries, normalise=True)
    return float((nn_predict(D, y_rf) == y_ev).mean())


def main():
    os.makedirs(f"{OUT}/models", exist_ok=True)
    S, meta = load_tensor()   # frozen full-data tensor, for the percentile report
    models = meta["models"]
    pairs = list(itertools.combinations(models, 2))
    ix = {m: i for i, m in enumerate(models)}
    ia = np.array([ix[a] for a, _ in pairs]); ib = np.array([ix[b] for _, b in pairs])
    hard = near_relative_pairs(models)

    # per-query cubes, so a subsample is a column slice
    _, X = load_corpus(pool="build")
    nq, nc, dim = X[models[0]].shape
    print(f"{len(models)} models, {nq} queries, {nc} build configs", flush=True)

    # the proxy's evaluation: build-split chains scored on the VAL split, so the
    # ranking never touches test (I2) and Arm B stays independent of Arm A's
    # reported numbers
    Dq, w2, y_ev, y_rf, _, cfg_ev = build_dq(eval_pool="val")

    rng = np.random.default_rng(SEED)
    subs = [np.sort(rng.choice(nc, nc // 2, replace=False)) for _ in range(R)]

    qual = {a: {g: {k: [] for k in KS} for g in GAMMAS}
            for a in ("GreedyCover", "JointGreedy")}
    chains_by = {a: {g: [] for g in GAMMAS}
                 for a in ("GreedyCover", "JointGreedy")}

    t0 = time.time()
    for r, idx in enumerate(subs):
        Xs = {m: X[m][:, idx, :] for m in models}
        # One precompute serves BOTH algorithms. The smoke run rebuilt the
        # per-query energy tensor in a separate 172k-cell Python loop -- ~4 min
        # per resample, ~2 h for R=30. It is redundant: scale-free energy is
        # exactly what the accumulator blocks already encode at a single query,
        # so Ssub falls out of XX2/XY2 with two sqrt-means and no extra work.
        XX2, XY2 = precompute(Xs, models, pairs)
        xy = np.sqrt(XY2).mean(axis=(2, 3))            # (nq, n_pairs)
        xm = np.sqrt(XX2).mean(axis=(2, 3))            # (nq, n_models)
        xx, yy = xm[:, ia], xm[:, ib]
        within = (xx + yy) / 2.0
        Ssub = np.where(within > 0, (2 * xy - xx - yy) / np.maximum(within, 1e-12),
                        np.nan).astype(np.float32)
        for g in GAMMAS:
            gc = greedy_cover(Ssub, max(KS), gamma=float(g), agg="max")
            jg = joint_greedy(XY2, XX2, ia, ib, max(KS), float(g), "energy",
                              verbose=False)[0]
            chains_by["GreedyCover"][g].append(gc)
            chains_by["JointGreedy"][g].append(jg)
            for k in KS:
                qual["GreedyCover"][g][k].append(proxy_quality(Dq, w2, gc[:k],
                                                               y_ev, y_rf))
                qual["JointGreedy"][g][k].append(proxy_quality(Dq, w2, jg[:k],
                                                               y_ev, y_rf))
        print(f"  resample {r+1}/{R}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    # ---- the pre-registered decision rule
    def paired(alg, ga, gb, k, n_boot=2000, seed=7):
        a = np.array(qual[alg][ga][k]); b = np.array(qual[alg][gb][k])
        d = a - b
        rg = np.random.default_rng(seed)
        bs = np.array([d[rg.integers(0, len(d), len(d))].mean()
                       for _ in range(n_boot)])
        return dict(mean=float(d.mean()), lo=float(np.percentile(bs, 2.5)),
                    hi=float(np.percentile(bs, 97.5)),
                    median_a=float(np.median(a)), median_b=float(np.median(b)),
                    iqr_a=[float(np.percentile(a, 25)), float(np.percentile(a, 75))],
                    iqr_b=[float(np.percentile(b, 25)), float(np.percentile(b, 75))])

    def classify(res_by_k):
        sysd = all(r["lo"] > 0 and r["mean"] > 0.02 for r in res_by_k.values())
        chain = all(r["lo"] > -0.02 and r["hi"] < 0.02 for r in res_by_k.values())
        return ("SYSTEMATIC (-> H1)" if sysd else
                "CHAIN-SPECIFIC (-> H2)" if chain else "UNRESOLVED")

    verdict = {}
    for alg in qual:
        main_c = {k: paired(alg, "0.25", "0.5", k) for k in KS}
        ctrl_c = {k: paired(alg, "0.25", "1.0", k) for k in KS}
        cls = classify(main_c)
        ctrl = classify(ctrl_c)
        powered = ctrl == "SYSTEMATIC (-> H1)"
        verdict[alg] = dict(
            main_contrast="Q(0.25) - Q(0.5)", by_k=main_c, classification=cls,
            positive_control="Q(0.25) - Q(1.0)", control_by_k=ctrl_c,
            control_classification=ctrl, has_power=bool(powered),
            final=(cls if powered else "UNINFORMATIVE (control failed)"))
        print(f"\n[ArmB] {alg}: main {cls}", flush=True)
        for k in KS:
            m = main_c[k]
            print(f"    k={k}: d={m['mean']:+.4f} [{m['lo']:+.4f},{m['hi']:+.4f}] "
                  f"| median 0.25={m['median_a']:.4f} vs 0.5={m['median_b']:.4f}",
                  flush=True)
        print(f"  control {ctrl} -> powered: {powered}", flush=True)

    # descriptive: where does the frozen full-data chain sit in its own distribution
    frozen = json.load(open(f"{OUT}/chain_analysis.json"))
    pct = {}
    for alg, key in (("GreedyCover", "greedycover"), ("JointGreedy", "jointgreedy")):
        pct[alg] = {}
        for g in GAMMAS:
            fq = {k: proxy_quality(Dq, w2, frozen[key]["chains"][g][:k], y_ev, y_rf)
                  for k in KS}
            pct[alg][g] = {str(k): dict(
                frozen_quality=round(fq[k], 4),
                percentile=round(float((np.array(qual[alg][g][k]) < fq[k]).mean()
                                       * 100), 1)) for k in KS}

    dist = {alg: {g: {str(k): dict(
        median=round(float(np.median(qual[alg][g][k])), 4),
        iqr=[round(float(np.percentile(qual[alg][g][k], 25)), 4),
             round(float(np.percentile(qual[alg][g][k], 75)), 4)],
        rng=[round(float(np.min(qual[alg][g][k])), 4),
             round(float(np.max(qual[alg][g][k])), 4)],
        values=[round(float(v), 4) for v in qual[alg][g][k]])
        for k in KS} for g in GAMMAS} for alg in qual}

    # ---- the 20-run trained sanity check (review amendment)
    print("\n[check] training proxy-best/worst resampled chains "
          "(GreedyCover, k=8)", flush=True)
    d009 = json.load(open("./results/D009/runs.json"))
    qe = load_query_embeddings()
    cubes = {p: load_corpus(pool=p) for p in ("build", "val", "test")}
    sanity = {}
    for g in ("0.5", "0.25"):
        q8 = np.array(qual["GreedyCover"][g][8])
        picks = {"proxy_best": int(q8.argmax()), "proxy_worst": int(q8.argmin())}
        sanity[g] = dict(proxy=dict(best=float(q8.max()), worst=float(q8.min())))
        for name, r_ix in picks.items():
            chain = chains_by["GreedyCover"][g][r_ix][:8]
            tr_b, y_b, _, _ = build_traces(chain, "build", qe, cubes["build"])
            tr_v, y_v, _, _ = build_traces(chain, "val", qe, cubes["val"])
            tr_t, y_t, c_t, _ = build_traces(chain, "test", qe, cubes["test"])
            hp, conf = hparams_from_shipped(8, len(models))
            conf = dict(conf); conf["inference_model"] = hp
            h = hashlib.sha256(json.dumps(
                {kk: vv for kk, vv in hp.items() if kk != "num_queries"},
                sort_keys=True).encode()).hexdigest()
            assert h == d009["hparams_hash"]
            accs = []
            for r_ in range(5 if not SMOKE else 2):
                out, _, _ = run_one(tr_b, y_b, tr_v, y_v, tr_t, y_t, c_t, hp,
                                    conf, r_, hard, len(models), None)
                if not out["error"]:
                    accs.append(out["mean_top1"])
            sanity[g][name] = dict(resample=r_ix, chain=chain,
                                   trained_mean_top1=round(float(np.mean(accs)), 4),
                                   runs=len(accs))
        tb = sanity[g]["proxy_best"]["trained_mean_top1"]
        tw = sanity[g]["proxy_worst"]["trained_mean_top1"]
        sanity[g]["ordering_agrees"] = bool(tb >= tw)
        print(f"  g={g}: proxy best {q8.max():.4f} -> trained {tb:.4f} | "
              f"proxy worst {q8.min():.4f} -> trained {tw:.4f} | "
              f"agrees: {sanity[g]['ordering_agrees']}", flush=True)
    proxy_confirmed = all(v["ordering_agrees"] for v in sanity.values())

    json.dump(dict(schema="d014-armB-v1", R=R, gammas=GAMMAS, ks=KS, seed=SEED,
                   sampling="half of 75 build configs, WITHOUT replacement",
                   proxy="D008 1-NN, full build reference, scored on VAL (I2: "
                         "test never touched by Arm B)",
                   decision_rule="fixed before any run (2026-09-19 review): "
                                 "SYSTEMATIC = CI excludes 0 and mean d > 0.02 at "
                                 "both k; CHAIN-SPECIFIC = CI within +/-0.02 at "
                                 "both k; else UNRESOLVED. Control Q(0.25)-Q(1.0) "
                                 "must be SYSTEMATIC or Arm B is uninformative.",
                   verdict=verdict, distributions=dist,
                   frozen_chain_percentile=pct,
                   proxy_sanity_check=dict(
                       detail=sanity, proxy_ranking_confirmed=proxy_confirmed,
                       note="D009 validated the proxy ACROSS conditions; this "
                            "tests it for ranking chains WITHIN a gamma, which is "
                            "what Arm B asks of it. If it fails, Arm B is "
                            "proxy-level and unconfirmed."),
                   wall_min=round((time.time() - t0) / 60, 1)),
              open(f"{OUT}/chain_stability.json", "w"), indent=1)

    print(f"\n===== Arm B verdicts (THESE DECIDE) =====")
    for alg, v in verdict.items():
        print(f"  {alg:12s} {v['final']}")
    print(f"  proxy within-gamma ranking confirmed: {proxy_confirmed}")
    print(f"\ntotal {(time.time()-t0)/60:.1f} min")
    print(f"written: {OUT}/chain_stability.json")


if __name__ == "__main__":
    main()
