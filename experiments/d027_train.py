"""D027 / S2 — B0' (paper8, k=8, D009 attention protocol, seeds 0-4) on the
85-model v2 universe, plus the linear (D017 concat, C=1.0) readout.

* Universe order: sorted(37 v1 + S0 survivors) (results/D027/survivors.txt).
* Traces: d009_lib.build_traces over the screen embeddings (E(query) from
  corpus_v1's _queries.npy -- the paper8 texts are pool ids 0..7, asserted).
* Training: d009_train.run_one unchanged; S_build trains, S_val early-stops
  (the accepted D025 optimism). There is no S_test: run_one's "test" slot is
  S_val, so its per-pair numbers are S_val numbers. Hyper-parameters are
  D009's shipped block, asserted equal to D009's hash with num_classes and
  num_queries excluded (the only two fields that legitimately differ).
* Every checkpoint saved; S_val logits saved (gitignored) for all readouts.
Writes results/D027/b0_runs.json, results/D027/logits_b0.npz (gitignored),
results/D027/logits_linear.npz (gitignored).

Usage:  PYTHONPATH=.:experiments python experiments/d027_train.py
"""
import os
import json
import time
import hashlib

import numpy as np
from sklearn.linear_model import LogisticRegression

from d009_lib import load_query_embeddings, build_traces, hparams_from_shipped
from d009_train import run_one
from d023_train import reload_logits

RES = "./results/D027"
SCR = "./data/corpus_v2_screen"
PAPER8, K, N_SEEDS = list(range(8)), 8, 5


def universe():
    v1 = sorted(s["model"] for s in json.load(open("./data/corpus_v1/corpus_manifest.json"))["models"])
    surv = [l.strip() for l in open(f"{RES}/survivors.txt") if l.strip()]
    models = sorted(v1 + surv)
    assert len(models) == 85, len(models)
    return models


def load_screen(models):
    """{pool: (models, X)} with X[m] (8, ncfg, 1024) float32, configs sorted."""
    out = {}
    for pool in ("build", "val"):
        X = {}
        for m in models:
            s = m.replace("/", "__")
            a = np.load(f"{SCR}/embeddings/{s}.npy")
            rows = json.load(open(f"{SCR}/embeddings/{s}.index.json"))["rows"]
            sel = [(i, r["query_index"], r["config"]) for i, r in enumerate(rows) if r["pool"] == pool]
            cfgs = sorted({c for _, _, c in sel})
            ci = {c: j for j, c in enumerate(cfgs)}
            cube = np.empty((8, len(cfgs), a.shape[1]), np.float32)
            for i, q, c in sel:
                cube[q, ci[c]] = a[i]
            X[m] = cube
        n = {X[m].shape[1] for m in models}
        assert n == {75 if pool == "build" else 25}, (pool, n)
        out[pool] = (models, X)
    return out


def hp_hash(hp):
    return hashlib.sha256(json.dumps({k: v for k, v in hp.items() if k not in ("num_queries", "num_classes")},
                                     sort_keys=True).encode()).hexdigest()


def main():
    t0 = time.time()
    models = universe()
    nm = len(models)
    q0 = json.load(open("./confs/queries/pool_v1.json"))
    assert q0["sha256"] == "0a6d098c7b540bf7fc4e111a9d11f58eb4b3ddad80f37ba1f10d6677775ae0e8"
    qe = load_query_embeddings()
    cubes = load_screen(models)
    tr_b, y_b, c_b, _ = build_traces(PAPER8, "build", qe, cubes["build"])
    tr_v, y_v, c_v, _ = build_traces(PAPER8, "val", qe, cubes["val"])
    print(f"[S2] {nm} models; build {tr_b.shape}, val {tr_v.shape}", flush=True)

    hp, conf = hparams_from_shipped(K, nm)
    conf = dict(conf); conf["inference_model"] = hp
    ref_hp, _ = hparams_from_shipped(K, 37)
    assert hp_hash(hp) == hp_hash(ref_hp), "hyper-parameters differ from D009's beyond num_classes"
    os.makedirs(f"{RES}/models", exist_ok=True)
    runs, logits = [], {}
    for r in range(N_SEEDS):
        path = f"{RES}/models/b0_paper8_k8_r{r}.ckpt"
        # logit_stats needs a non-empty pair dict; per-pair numbers are computed in S3 from logits
        res, _, buf = run_one(tr_b, y_b, tr_v, y_v, tr_v, y_v, c_v, hp, conf, r,
                              {0: dict(ab=(0, 1))}, nm, path)
        assert not res["error"], res["error"] + buf.getvalue()[-800:]
        lv = reload_logits(path, tr_v, nm, K)
        assert abs(float((lv.argmax(1) == y_v).mean()) - res["mean_top1"]) < 1e-9, "reload mismatch"
        logits[f"b0_r{r}"] = lv.astype(np.float32)
        runs.append(dict(seed=r, wall_s=res["wall_s"], train_acc=res["train_acc"], val_acc=res["val_acc"],
                         val_mean_top1=res["mean_top1"], best_ckpt=res["best_ckpt"], ckpt=path))
        print(f"[S2] B0' seed {r}: S_val top-1 {res['mean_top1']:.4f} (train {res['train_acc']:.4f}) "
              f"{res['wall_s']} s", flush=True)

    # linear readout (D017 concat: E(query)+E(response) per slot, flattened)
    t = time.time()
    clf = LogisticRegression(C=1.0, max_iter=20000, tol=1e-6, solver="lbfgs")
    clf.fit(tr_b.reshape(len(tr_b), -1), y_b)
    ll = clf.decision_function(tr_v.reshape(len(tr_v), -1)).astype(np.float32)
    lin = dict(n_iter=int(clf.n_iter_.max()), converged=bool(clf.n_iter_.max() < 20000),
               val_mean_top1=float((ll.argmax(1) == y_v).mean()), wall_s=round(time.time() - t, 1))
    print(f"[S2] linear: S_val top-1 {lin['val_mean_top1']:.4f}, converged {lin['converged']}", flush=True)

    np.savez_compressed(f"{RES}/logits_b0.npz", y=y_v, cfg=c_v, **logits)
    np.savez_compressed(f"{RES}/logits_linear.npz", y=y_v, cfg=c_v, linear=ll)
    json.dump(dict(schema="d027-b0-runs-v1", models=models, queries=PAPER8, k=K,
                   hparams_hash_excl_classes_queries=hp_hash(hp),
                   protocol="d009_train.run_one; S_build train, S_val early stop + readout; no S_test",
                   runs=runs, linear=lin, wall_s=round(time.time() - t0, 1)),
              open(f"{RES}/b0_runs.json", "w"), indent=1)


if __name__ == "__main__":
    main()
