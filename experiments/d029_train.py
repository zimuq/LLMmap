"""D029 / S1 — train and read out ONE frozen arm of results/D029/chains.json.

Attention: D009 protocol (d009_train.run_one unchanged; S_build fit, S_val early
stopping, seeds 0-4), hparams from confs/default.json with num_classes=85,
hash (num_classes/num_queries excluded) == D027's. Every checkpoint saved and
reloaded; the reloaded S_test top-1 is asserted equal to run_one's SILENTLY.
Linear: D017 concat (d020_paired_linear.POOL["concat"], C=1.0, lbfgs, tol 1e-6,
max_iter 20000), fit on S_build, decision function on S_test.

S_test discipline (P1 Call 6): S_test logits and counts are WRITTEN, never
printed, logged or summarised here. Logs and runs_{arm}.json carry S_val /
train numbers only. S2 is the first reader.

Stored per (readout, k, [seed]): logits (gitignored npz), cnt_total (25),
cnt_model (85 x 25), cnt_pair (57 x 25, H_all), dirH (57 x 2 x 25), fam|<g>.
Usage:  PYTHONPATH=.:experiments python experiments/d029_train.py <ARM>
"""
import os
import sys
import json
import time
import hashlib

import numpy as np
from sklearn.linear_model import LogisticRegression

import d029_lib as L
from d009_lib import build_traces, hparams_from_shipped, logit_stats
from d009_train import run_one
from d020_paired_linear import POOL, MAX_ITER, TOL
from d023_train import reload_logits
from d025_lib import dir_counts, fam_counts

N_SEEDS = 5
TEST_KEYS = ("mean_top1", "worst_class", "worst3_class", "hard_subset", "hard_subset_min",
             "per_model", "per_hard_pair")


def safe(arm):
    return arm.replace("+", "p")


def hp_hash(hp):
    return hashlib.sha256(json.dumps({k: v for k, v in hp.items() if k not in ("num_queries", "num_classes")},
                                     sort_keys=True).encode()).hexdigest()


def counts(lg, y, c, hard, nm, pa, pb, fams):
    _, st = logit_stats(lg, y, c, hard, nm)
    out = dict(cnt_total=st["cnt_total"], cnt_model=st["cnt_model"], cnt_pair=st["cnt_pair"])
    out["dirH"] = dir_counts(lg, y, c, pa, pb)
    assert np.array_equal(out["dirH"].sum(1), st["cnt_pair"]), "dirH != cnt_pair"
    for g, idx in fams.items():
        out[f"fam|{g}"] = fam_counts(lg, y, c, idx)
    return out


def main():
    arm = sys.argv[1]
    t0 = time.time()
    ch = json.load(open(f"{L.OUT}/chains.json"))
    assert ch["frozen_before_training"] and arm in ch["arms"], arm
    a = ch["arms"][arm]
    cands, ks = a["cands"], a["k_train"]
    models = L.universe()
    assert models == ch["models"]
    nm = len(models)
    tg = L.targets(models)
    pa, pb = L.hard_ab(models, tg["H"])
    hard = {j: dict(ab=(int(x), int(y))) for j, (x, y) in enumerate(zip(pa, pb))}
    qe = L.query_embeddings_275()
    cubes = L.load_v2(models)
    print(f"[S1] {arm}: {nm} models, cands {a['ids']}, k {ks}; cubes loaded "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    pool_fn, C = POOL["concat"]
    ckdir = f"{L.OUT}/models/{safe(arm)}"
    os.makedirs(ckdir, exist_ok=True)
    store, logits, runs, lin = {}, {}, [], {}

    for k in ks:
        q = cands[:k]
        tr_b, y_b, _, _ = build_traces(q, "build", qe, cubes["build"])
        tr_v, y_v, _, _ = build_traces(q, "val", qe, cubes["val"])
        tr_t, y_t, c_t, _ = build_traces(q, "test", qe, cubes["test"])
        assert c_t.max() == 24 and len(y_t) == nm * 25

        # ---------------- linear
        t1 = time.time()
        clf = LogisticRegression(C=C, max_iter=MAX_ITER, tol=TOL, solver="lbfgs")
        clf.fit(pool_fn(tr_b), y_b)
        n_it = int(clf.n_iter_.max())
        assert n_it < MAX_ITER, f"max_iter hit k={k}"
        tr_acc = float((clf.predict(pool_fn(tr_b)) == y_b).mean())
        lg = clf.decision_function(pool_fn(tr_t)).astype(np.float32)
        logits[f"lin|{k}"] = lg
        for kk, vv in counts(lg, y_t, c_t, hard, nm, pa, pb, tg["fams"]).items():
            store[f"lin|{k}|{kk}"] = vv
        lin[str(k)] = dict(n_iter=n_it, train_acc=tr_acc, wall_s=round(time.time() - t1, 1))
        print(f"[S1] {arm} k={k} linear fit: train {tr_acc:.4f}, {n_it} it, {time.time() - t1:.0f}s", flush=True)

        # ---------------- attention
        hp, conf = hparams_from_shipped(k, nm)
        conf = dict(conf); conf["inference_model"] = hp
        assert hp_hash(hp) == L.D027_HP_HASH, "hparams differ from D027/D009 beyond classes/queries"
        for r in range(N_SEEDS):
            path = f"{ckdir}/k{k}_r{r}.ckpt"
            o, st, buf = run_one(tr_b, y_b, tr_v, y_v, tr_t, y_t, c_t, hp, conf, r, hard, nm, path)
            assert not o["error"], o["error"] + buf.getvalue()[-800:]
            lg = reload_logits(path, tr_t, nm, k)
            assert abs(float((lg.argmax(1) == y_t).mean()) - o["mean_top1"]) < 1e-9, "reload mismatch"
            cn = counts(lg, y_t, c_t, hard, nm, pa, pb, tg["fams"])
            for kk in ("cnt_total", "cnt_model", "cnt_pair"):
                assert np.array_equal(cn[kk], st[kk]), (arm, k, r, kk)
            logits[f"att|{k}|{r}"] = lg.astype(np.float32)
            for kk, vv in cn.items():
                store[f"att|{k}|{r}|{kk}"] = vv
            runs.append(dict(k=k, seed=r, queries=q, **{kk: vv for kk, vv in o.items() if kk not in TEST_KEYS}))
        vals = [x["val_acc"] for x in runs if x["k"] == k]
        print(f"[S1] {arm} k={k} attention S_val top-1 mean {np.mean(vals):.4f} "
              f"[{min(vals):.4f}, {max(vals):.4f}]  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    np.savez_compressed(f"{L.OUT}/counts_{safe(arm)}.npz", **store)
    np.savez_compressed(f"{L.OUT}/logits_{safe(arm)}.npz", y=y_t, cfg=c_t, **logits)
    json.dump(dict(schema="d029-runs-v1", corpus="v2", arm=arm, cands=cands, ids=a["ids"], k=ks,
                   hparams_hash_excl_classes_queries=L.D027_HP_HASH, attention_runs=runs, linear=lin,
                   s_test="written to counts/logits only; not summarised in S1",
                   wall_min=round((time.time() - t0) / 60, 1), node=os.uname().nodename),
              open(f"{L.OUT}/runs_{safe(arm)}.json", "w"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"[S1] {arm} done, wall {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
