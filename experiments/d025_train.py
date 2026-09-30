"""D025 / S2 — train and read out ONE frozen arm (sharded: one job per arm).

Attention (secondary readout): D009 protocol (d009_train.run_one, hparams from
confs/default.json, hparams hash == D009's, S_val early stopping, S_test once),
seeds 0-4, k=1..16 (chain's own order). ALL checkpoints saved; each reloaded and
mean_top1 asserted. GLOBAL k<=8: cnt_total/cnt_model/cnt_pair must reproduce
D010's stored counts EXACTLY (Review Call 2; stop condition).
Linear (primary readout): D017 concat, C=1.0, lbfgs, tol 1e-6, max_iter 20000,
fit on S_build; logits on S_test and S_val. GLOBAL k<=8: test counts must equal
D020's JointGreedy concat counts and per_hard_pair D017's values (stop condition).
Stored per (readout, [split,] k, [seed]): logits, cnt_total, cnt_model, cnt_pair
(65 N pairs), dir666 (666 x 2 x 25 directional two-logit), fam|<group> (|F| x 25
family-restricted argmax, every family group).
Usage:  PYTHONPATH=.:experiments python experiments/d025_train.py <ARM>
"""
import os
import sys
import json
import time
import hashlib
import itertools

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from d007_lib import load_corpus
from d008_lib import near_relative_pairs
from d009_lib import load_query_embeddings, build_traces, hparams_from_shipped, logit_stats
from d009_train import run_one
from d020_paired_linear import POOL, MAX_ITER, TOL
from d023_train import reload_logits
from d025_lib import OUT, K_MAX, dir_counts, fam_counts

N_SEEDS = 5
CNT = ("cnt_total", "cnt_model", "cnt_pair")


def safe(arm):
    return arm.replace(":", "_").replace("|", "__").replace("/", "_")


def all_counts(lg, y, c, hard, nm, pa, pb, fams):
    pt, st = logit_stats(lg, y, c, hard, nm)
    out = {k: st[k] for k in CNT}
    out["dir666"] = dir_counts(lg, y, c, pa, pb)
    for g, idx in fams.items():
        out[f"fam|{g}"] = fam_counts(lg, y, c, idx)
    # directional counts must add up to the combined two-logit counts on N
    keys = sorted(hard)
    assert np.array_equal(out["dir666"][keys].sum(1), st["cnt_pair"]), "dir666 != cnt_pair"
    return pt, out


def main():
    arm = sys.argv[1]
    t0 = time.time()
    sel = json.load(open(f"{OUT}/selection.json"))
    assert sel["frozen_before_training"] and arm in sel["trained_arms"], arm
    chain = sel["chains"][arm]
    models = json.load(open("./results/D008/selection.json"))["models"]
    nm = len(models)
    hard = near_relative_pairs(models)
    pairs = list(itertools.combinations(models, 2))
    pa = np.array([models.index(a) for a, _ in pairs])
    pb = np.array([models.index(b) for _, b in pairs])
    fams = {n: [models.index(m) for m in a["models"]] for n, a in sel["arms"].items()
            if a["kind"] == "FAM"}
    qe = load_query_embeddings()
    cubes = {p: load_corpus(pool=p) for p in ("build", "val", "test")}
    hash_ref = json.load(open("./results/D009/runs.json"))["hparams_hash"]
    is_global = arm == "GLOBAL"
    ref_att = np.load("./results/D010/run_counts.npz") if is_global else None
    ref_lin = np.load("./results/D020/linear_counts.npz") if is_global else None
    d17 = json.load(open("./results/D017/linear_per_hard_pair.json"))["concat"]["joint_energy"] \
        if is_global else None
    pool_fn, C = POOL["concat"]
    ckdir = f"{OUT}/models/{safe(arm)}"
    os.makedirs(ckdir, exist_ok=True)
    store, runs, lin_pts = {}, [], {}
    names = [" | ".join(hard[i]["pair"]) for i in sorted(hard)]

    for k in range(1, K_MAX + 1):
        q = chain[:k]
        tr_b, y_b, _, _ = build_traces(q, "build", qe, cubes["build"])
        tr_v, y_v, c_v, _ = build_traces(q, "val", qe, cubes["val"])
        tr_t, y_t, c_t, _ = build_traces(q, "test", qe, cubes["test"])

        # ---------------- linear (primary)
        t1 = time.time()
        clf = LogisticRegression(C=C, max_iter=MAX_ITER, tol=TOL, solver="lbfgs")
        clf.fit(pool_fn(tr_b), y_b)
        n_it = int(clf.n_iter_.max())
        assert n_it < MAX_ITER, f"max_iter hit k={k}"
        for split, tr, y, c in (("test", tr_t, y_t, c_t), ("val", tr_v, y_v, c_v)):
            lg = clf.decision_function(pool_fn(tr)).astype(np.float32)
            pt, cn = all_counts(lg, y, c, hard, nm, pa, pb, fams)
            store[f"lin|{split}|{k}|logits"] = lg
            for kk, vv in cn.items():
                store[f"lin|{split}|{k}|{kk}"] = vv
            lin_pts[f"{split}|{k}"] = dict(mean_top1=pt["mean_top1"], n_iter=n_it,
                                           per_hard_pair=pt["per_hard_pair"])
            if is_global and k <= 8 and split == "test":
                for kk in CNT:
                    assert np.array_equal(cn[kk], ref_lin[f"JointGreedy|concat|{k}|{kk}"]), \
                        f"STOP: GLOBAL linear k={k} {kk} != D020"
                ref = d17[str(k)]
                assert all(abs(pt["per_hard_pair"][j] - ref[n]) < 1e-9 for j, n in enumerate(names)), \
                    f"STOP: GLOBAL linear k={k} per_hard_pair != D017"
        print(f"[S2] {arm} k={k} linear test top1 {lin_pts[f'test|{k}']['mean_top1']:.4f} "
              f"({time.time() - t1:.0f}s, {n_it} it)", flush=True)

        # ---------------- attention (secondary)
        hp, conf = hparams_from_shipped(k, nm)
        conf = dict(conf); conf["inference_model"] = hp
        h = hashlib.sha256(json.dumps({kk: vv for kk, vv in hp.items() if kk != "num_queries"},
                                      sort_keys=True).encode()).hexdigest()
        assert h == hash_ref
        for r in range(N_SEEDS):
            path = f"{ckdir}/k{k}_r{r}.ckpt"
            o, st, _ = run_one(tr_b, y_b, tr_v, y_v, tr_t, y_t, c_t, hp, conf, r, hard, nm, path)
            assert not o["error"], o["error"]
            lg = reload_logits(path, tr_t, nm, k)
            acc = float((lg.argmax(1) == y_t).mean())
            assert abs(acc - o["mean_top1"]) < 1e-9, (arm, k, r, acc, o["mean_top1"])
            pt, cn = all_counts(lg, y_t, c_t, hard, nm, pa, pb, fams)
            for kk in CNT:
                assert np.array_equal(cn[kk], st[kk]), (arm, k, r, kk)
                if is_global and k <= 8:
                    assert np.array_equal(cn[kk], ref_att[f"joint_energy|{k}|{r}|{kk}"]), \
                        f"STOP: GLOBAL attention k={k} r={r} {kk} != D010"
            store[f"att|{k}|{r}|logits"] = lg.astype(np.float32)
            for kk, vv in cn.items():
                store[f"att|{k}|{r}|{kk}"] = vv
            runs.append(dict(arm=arm, k=k, run=r, queries=q,
                             **{kk: vv for kk, vv in o.items() if kk != "best_ckpt"}))
        print(f"[S2] {arm} k={k} attention mean top1 "
              f"{np.mean([x['mean_top1'] for x in runs if x['k'] == k]):.4f}"
              f"{' (== D010 counts)' if is_global and k <= 8 else ''} "
              f"[{(time.time() - t0) / 60:.1f} min]", flush=True)

    np.savez_compressed(f"{OUT}/counts_{safe(arm)}.npz", **store)
    json.dump(dict(schema="d025-runs-v1", arm=arm, chain=chain, hparams_hash=hash_ref, runs=runs,
                   linear_points=lin_pts, global_reproduction_asserted=is_global,
                   wall_min=round((time.time() - t0) / 60, 1)),
              open(f"{OUT}/runs_{safe(arm)}.json", "w"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"[S2] {arm} done, wall {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
