"""D018 / S2–S5 — evaluate the H.1-guided chains and compare against ours.

S2  each new chain at k=1..8 under this project's standard protocol: the attention
    network with 5 seeds (D009's convention), the linear classifier under BOTH
    poolings with its deterministic fit plus the 2,000-draw config bootstrap
    (D017's convention). S_test touched exactly once per chain per k (I2).
S3  compared against GreedyCover / JointGreedy, whose numbers are READ, never
    rerun (I6), with the reuse assertions P1/F2 requires -- the trap that bit
    D014's first pass and D016's.
S4  the 65-pair hard_subset breakdown at k=8, D015's machinery unchanged.
S5  chain overlap and the three-way verdict under P1/F3's tightened PARITY rule.

P1/F3, pre-registered: TOST's delta=0.02 sits INSIDE the attention network's own
0.024-0.081 seed range, so PARITY is reported only when the TOST interval is within
+/-0.02 AND |delta| is below the seed range. Otherwise INCONCLUSIVE.

P1/F1, pre-registered: the attention arm IS Algorithm H.1; the linear arm is NOT
(D017 showed that classifier is better at every k). They are reported separately
and never averaged into one "beats H.1" verdict.

Usage:  PYTHONPATH=.:experiments python experiments/d018_evaluate.py
"""
import os
import csv
import json
import time
import hashlib

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from d009_lib import (load_query_embeddings, build_traces, hparams_from_shipped,
                      logit_stats)
from d009_train import run_one
from d008_lib import near_relative_pairs, boot_draws, boot_metrics, ci, paired_ci
from d007_lib import load_corpus

OUT = "./results/D018"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
METRICS = ("mean_top1", "worst_class", "worst3_class", "hard_subset")
N_SEEDS, N_BOOT, NCFG, DELTA = 5, 2000, 25, 0.02
MAX_ITER, TOL = 20000, 1e-6
SMOKE = bool(os.environ.get("D018_SMOKE"))
KS = [1, 8] if SMOKE else list(range(1, 9))
if SMOKE:
    N_SEEDS, N_BOOT = 2, 200


def ours():
    """GreedyCover / JointGreedy chains and stored metrics -- read, never rerun."""
    d8 = json.load(open("./results/D008/selection.json"))
    d10 = json.load(open("./results/D010/selection.json"))
    d9m = json.load(open("./results/D009/metrics_by_condition_k.json"))
    d10m = json.load(open("./results/D010/metrics_by_k.json"))
    gc = d8["conditions"]["cvar_max"]["queries"][:8]
    jg = d10["joint_energy"]["queries"][:8]
    # P1/F2: assert the reuse, since D012/D013 store only the gammas they trained
    assert d9m["metrics"]["cvar_max"]["8"]["queries"] == gc, "GreedyCover drift"
    d15 = {r["pair"]: r for r in json.load(
        open("./results/D015/per_pair_k8.json"))["rows"]}
    assert abs(d9m["metrics"]["cvar_max"]["8"]["mean_top1"] - 0.8519) < 1e-3
    assert abs(d10m["metrics"]["joint_energy"]["8"]["mean_top1"] - 0.8757) < 1e-3
    return dict(GreedyCover=dict(chain=gc, metrics=d9m["metrics"]["cvar_max"]),
                JointGreedy=dict(chain=jg,
                                 metrics=d10m["metrics"]["joint_energy"])), d15


def main():
    os.makedirs(f"{OUT}/models", exist_ok=True)
    t0 = time.time()
    qe = load_query_embeddings()
    cubes = {p: load_corpus(pool=p) for p in ("build", "val", "test")}
    models = json.load(open("./results/D008/selection.json"))["models"]
    n_models = len(models)
    hard = near_relative_pairs(models)
    keys = sorted(hard)
    pnames = [" | ".join(hard[i]["pair"]) for i in keys]
    hash_ref = json.load(open("./results/D009/runs.json"))["hparams_hash"]
    M = boot_draws(NCFG, N_BOOT)
    theirs, d15 = ours()

    chains, search = {}, {}
    for arm in ("attention", "linear"):
        f = f"{OUT}/greedy_{arm}.json"
        assert os.path.exists(f), f"missing {f} -- run d018_search.py --arm {arm}"
        search[f"H1-{arm}"] = json.load(open(f))
        chains[f"H1-{arm}"] = search[f"H1-{arm}"]["chain"][:8]
    # S1 deliverable: both chains + per-step trace in one file
    json.dump(dict(schema="d018-chains-v1", chains=chains, search=search),
              open(f"{OUT}/greedy_chains.json", "w"), indent=1)
    # S4 alignment: our 65-pair index must be D015's (P1/F2, d018_f2_check.py)
    assert set(pnames) == set(d15) and len(pnames) == 65, "pair index drift vs D015"
    print(f"[S2] H1-attention {chains['H1-attention']}", flush=True)
    print(f"[S2] H1-linear    {chains['H1-linear']}", flush=True)

    res, boots, per_pair = {}, {}, {}
    for cname, chain in chains.items():
        # --- attention-network evaluation, 5 seeds (D009's convention)
        key = f"{cname}|attention"
        res[key], boots[key], per_pair[key] = {}, {}, {}
        for k in KS:
            q = chain[:k]
            tr_b, y_b, _, _ = build_traces(q, "build", qe, cubes["build"])
            tr_v, y_v, _, _ = build_traces(q, "val", qe, cubes["val"])
            tr_t, y_t, c_t, _ = build_traces(q, "test", qe, cubes["test"])
            hp, conf = hparams_from_shipped(k, n_models)
            conf = dict(conf); conf["inference_model"] = hp
            h = hashlib.sha256(json.dumps(
                {kk: vv for kk, vv in hp.items() if kk != "num_queries"},
                sort_keys=True).encode()).hexdigest()
            assert h == hash_ref
            pts, sts = [], []
            for r in range(N_SEEDS):
                keep = f"{OUT}/models/{cname}_k{k}_r{r}.ckpt" if r == 0 and k == 8 else None
                o, cnt, _ = run_one(tr_b, y_b, tr_v, y_v, tr_t, y_t, c_t, hp, conf,
                                    r, hard, n_models, keep)
                assert not o["error"], o["error"]
                pts.append(o); sts.append(cnt)
            e = {}
            for m in METRICS:
                v = np.array([p[m] for p in pts], float)
                e[m] = float(v.mean()); e[m + "_range"] = float(v.max() - v.min())
            e["queries"] = q
            res[key][k] = e
            boots[key][k] = {m: np.mean([boot_metrics(s, M)[m] for s in sts], axis=0)
                             for m in METRICS}
            per_pair[key][k] = dict(zip(pnames, np.mean(
                [p["per_hard_pair"] for p in pts], axis=0).tolist()))
        print(f"[S2] {key:24s} k={max(KS)} mean {res[key][max(KS)]['mean_top1']:.4f}",
              flush=True)

        # --- linear evaluation, both poolings (D017's convention)
        for pool_name, pool_fn, C in (("concat", lambda a: a.reshape(len(a), -1), 1.0),
                                      ("meanpool", lambda a: a.mean(axis=1), 3.0)):
            key = f"{cname}|linear_{pool_name}"
            res[key], boots[key], per_pair[key] = {}, {}, {}
            for k in KS:
                q = chain[:k]
                tr, y_tr, _, _ = build_traces(q, "build", qe, cubes["build"])
                te, y_te, c_te, _ = build_traces(q, "test", qe, cubes["test"])
                clf = LogisticRegression(C=C, max_iter=MAX_ITER, tol=TOL,
                                         solver="lbfgs")
                clf.fit(pool_fn(tr), y_tr)
                assert int(clf.n_iter_.max()) < MAX_ITER
                pt, st = logit_stats(clf.decision_function(pool_fn(te)), y_te,
                                     c_te, hard, n_models)
                res[key][k] = {m: pt[m] for m in METRICS}
                res[key][k]["queries"] = q
                boots[key][k] = boot_metrics(st, M)
                per_pair[key][k] = dict(zip(pnames, pt["per_hard_pair"]))
            print(f"[S2] {key:24s} k={max(KS)} "
                  f"mean {res[key][max(KS)]['mean_top1']:.4f}", flush=True)

    # ---- S3/S5: the head-to-head, attention readout only (F1)
    def verdict(d, seed_range):
        """P1/F3: PARITY needs TOST within +/-delta AND |d| below the seed range."""
        within = (d["lo"] > -DELTA) and (d["hi"] < DELTA)
        below = abs(d["delta"]) < seed_range
        if d["lo"] > 0:
            return "OURS BETTER (resolved)"
        if d["hi"] < 0:
            return "H1 BETTER (resolved)"
        return "PARITY" if (within and below) else "INCONCLUSIVE"

    comp = {}
    for ourname, o in theirs.items():
        for k in KS:
            a = boots["H1-attention|attention"][k]["mean_top1"]
            # rebuild ours' bootstrap from its stored counts (I6, read-only)
            src = ("./results/D009/run_counts.npz" if ourname == "GreedyCover"
                   else "./results/D010/run_counts.npz")
            pref = "cvar_max" if ourname == "GreedyCover" else "joint_energy"
            cnts = np.load(src)
            rs = sorted({n.split("|")[2] for n in cnts.files
                         if n.startswith(f"{pref}|{k}|")})
            bb = []
            for r in rs:
                st = {kk: cnts[f"{pref}|{k}|{r}|{kk}"]
                      for kk in ("cnt_total", "cnt_model", "cnt_pair")}
                st["n_models"] = n_models; st["ncfg"] = NCFG
                bb.append(boot_metrics(st, M))
            ob = np.mean([b["mean_top1"] for b in bb], axis=0)
            d = paired_ci(ob, a)                     # ours - H1
            sr = max(o["metrics"][str(k)].get("mean_top1_range", 0.0) or 0.0,
                     res["H1-attention|attention"][k]["mean_top1_range"])
            d["seed_range"] = sr
            d["ours"] = o["metrics"][str(k)]["mean_top1"]
            d["h1"] = res["H1-attention|attention"][k]["mean_top1"]
            d["verdict"] = verdict(d, sr)
            comp[f"{ourname}_vs_H1attention_k{k}"] = d

    print(f"\n===== S5: ours vs Algorithm H.1 (attention arm), mean top-1 =====")
    print(f"{'k':>3s} {'GreedyCover':>34s} {'JointGreedy':>34s}")
    for k in KS:
        row = f"{k:>3d}"
        for ourname in ("GreedyCover", "JointGreedy"):
            c = comp[f"{ourname}_vs_H1attention_k{k}"]
            row += f"{c['ours']:.4f} vs {c['h1']:.4f} {c['delta']:+.4f} {c['verdict'][:9]}".rjust(34)
        print(row, flush=True)

    ov = {f"H1-{a}_vs_{b}": len(set(chains[f'H1-{a}']) & set(theirs[b]["chain"]))
          for a in ("attention", "linear") for b in theirs}
    ov["H1attention_vs_H1linear"] = len(set(chains["H1-attention"])
                                        & set(chains["H1-linear"]))
    print(f"\n[S5] chain overlap (of 8): {ov}", flush=True)

    json.dump(dict(schema="d018-metrics-v1", ks=KS, chains=chains,
                   n_seeds_attention=N_SEEDS, n_boot=N_BOOT, delta=DELTA,
                   parity_rule="TOST within +/-0.02 AND |delta| below the seed "
                               "range (P1/F3, pre-registered)",
                   arm_framing="H1-attention IS Algorithm H.1; H1-linear is NOT "
                               "(P1/F1) -- reported separately, never averaged",
                   metrics=res, comparisons=comp, chain_overlap=ov,
                   bootstrap_ci={c: {k: {m: ci(boots[c][k][m]) for m in METRICS}
                                     for k in boots[c]} for c in boots},
                   wall_min=round((time.time() - t0) / 60, 1)),
              open(f"{OUT}/metrics_by_k.json", "w"), indent=1)

    # ---- S4: hard-pair view at k=8
    hp8 = {c: per_pair[c][max(KS)] for c in per_pair}
    # reference columns for all 65 pairs, read verbatim from D015 (I6; verified
    # IDENTICAL to a recompute from D009/D010 counts by d018_f2_check.py)
    for nm in pnames:
        hp8.setdefault("_ours_reference", {})[nm] = dict(
            GreedyCover=d15[nm]["coverage"], JointGreedy=d15[nm]["joint_energy"],
            paper8=d15[nm]["paper8"])
    json.dump(dict(schema="d018-hard-pair-v1", k=max(KS), n_pairs=len(pnames),
                   resolution_caveat="one trace of 50 = 2.00 pp; no pair can move "
                                     "less (D015/R1)",
                   per_pair=hp8), open(f"{OUT}/per_hard_pair.json", "w"), indent=1)
    print(f"\nwall {(time.time()-t0)/60:.1f} min; written: {OUT}/", flush=True)


if __name__ == "__main__":
    main()
