"""D018 / S0 — the mandatory cost pilot. Nothing else runs until this is reviewed.

Algorithm H.1's real search is 2,044 candidate evaluations per arm:
    step k evaluates (260-k) candidates at set size k, for k=1..8
    sum_{k=1..8} (260-k) = 2044

Cost per evaluation is NOT constant in k, and differs by arm in opposite
directions, which is the thing this pilot exists to measure rather than assume:
  * attention network -- roughly k-flat (fixed architecture, 22 steps/epoch)
  * linear + concat   -- grows with k (features = k * 2048), and D017 needed
                         max_iter=20000/tol=1e-6 for its order-invariance result,
                         which cost 32 s at k=8. A search does not need bit-exact
                         order invariance, only a reliable argmax, so the looser
                         default is timed too -- that is a real fidelity/cost
                         choice, surfaced rather than taken silently.
  * linear + meanpool -- k-flat (always 2048 features)

Usage:  PYTHONPATH=.:experiments python experiments/d018_cost_pilot.py
"""
import os
import io
import json
import time
import shutil
import hashlib
import tempfile
import contextlib

import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.linear_model import LogisticRegression

from LLMmap.trainer import train_model
from d009_lib import (load_query_embeddings, build_traces, make_loader,
                      hparams_from_shipped, logits_for)
from d007_lib import load_corpus

OUT = "./results/D018"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
KS_PROBE = [1, 4, 8]
N_CAND = 3                      # candidates timed per (arm, k)
POOL_N = 259
STEPS = {k: POOL_N + 1 - k for k in range(1, 9)}   # candidates at each greedy step
TOTAL_EVALS = sum(STEPS.values())


def project(cost_by_k):
    """Full-search cost, given measured per-evaluation cost at each k."""
    ks = sorted(cost_by_k)
    tot = 0.0
    for k, n in STEPS.items():
        # interpolate/extrapolate linearly in k from the probed points
        c = float(np.interp(k, ks, [cost_by_k[x] for x in ks]))
        tot += n * c
    return tot


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    qe = load_query_embeddings()
    cubes = {p: load_corpus(pool=p) for p in ("build", "val", "test")}
    models = json.load(open("./results/D008/selection.json"))["models"]
    n_models = len(models)
    d9 = json.load(open("./results/D009/runs.json"))
    hash_ref = d9["hparams_hash"]
    pool = json.load(open("./confs/queries/pool_v1.json"))
    rng = np.random.default_rng(20260926)
    print(f"pool {pool['n']} queries; greedy search = {TOTAL_EVALS} candidate "
          f"evaluations per arm ({STEPS})", flush=True)

    res = {"n_evals_per_arm": TOTAL_EVALS, "candidates_per_step": STEPS, "arms": {}}

    # ---- arm A: attention network
    att = {}
    for k in KS_PROBE:
        ts = []
        for _ in range(N_CAND):
            q = rng.choice(POOL_N, k, replace=False).tolist()
            tr, y_tr, _, _ = build_traces(q, "build", qe, cubes["build"])
            va, y_va, _, _ = build_traces(q, "val", qe, cubes["val"])
            hp, conf = hparams_from_shipped(k, n_models)
            conf = dict(conf); conf["inference_model"] = hp
            h = hashlib.sha256(json.dumps(
                {kk: vv for kk, vv in hp.items() if kk != "num_queries"},
                sort_keys=True).encode()).hexdigest()
            assert h == hash_ref
            pl.seed_everything(0, workers=True)
            tmp = tempfile.mkdtemp(prefix="d018_")
            t = time.time()
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                _, net = train_model(tmp, siamese=False,
                                     loader_train=make_loader(tr, y_tr,
                                                              conf["batch_size"], True),
                                     loader_test=make_loader(va, y_va,
                                                             conf["batch_size"], False),
                                     conf=conf)
            _ = (logits_for(net, va, DEV).argmax(1) == y_va).mean()
            ts.append(time.time() - t)
            shutil.rmtree(tmp, ignore_errors=True)
        att[k] = float(np.median(ts))
        print(f"[A] attention  k={k}: {att[k]:.1f} s/eval  (n={N_CAND})", flush=True)
    res["arms"]["attention"] = dict(per_eval_s=att,
                                    projected_total_h=round(project(att)/3600, 2))

    # ---- arms B/C: linear, both poolings, both solver settings
    for pname, pool_fn in (("concat", lambda a: a.reshape(len(a), -1)),
                           ("meanpool", lambda a: a.mean(axis=1))):
        for tag, (mi, tol) in (("tight", (20000, 1e-6)), ("loose", (3000, 1e-4))):
            cost = {}
            for k in KS_PROBE:
                ts = []
                for _ in range(N_CAND):
                    q = rng.choice(POOL_N, k, replace=False).tolist()
                    tr, y_tr, _, _ = build_traces(q, "build", qe, cubes["build"])
                    va, y_va, _, _ = build_traces(q, "val", qe, cubes["val"])
                    t = time.time()
                    clf = LogisticRegression(C=1.0 if pname == "concat" else 3.0,
                                             max_iter=mi, tol=tol, solver="lbfgs")
                    clf.fit(pool_fn(tr), y_tr)
                    _ = (clf.predict(pool_fn(va)) == y_va).mean()
                    ts.append(time.time() - t)
                cost[k] = float(np.median(ts))
            key = f"linear_{pname}_{tag}"
            res["arms"][key] = dict(per_eval_s=cost, solver=dict(max_iter=mi, tol=tol),
                                    projected_total_h=round(project(cost)/3600, 2))
            print(f"[{'B' if pname=='concat' else 'C'}] {key:24s} "
                  + "  ".join(f"k={k}:{cost[k]:.1f}s" for k in KS_PROBE)
                  + f"  -> {res['arms'][key]['projected_total_h']} h", flush=True)

    print(f"\n===== S0 projections, full 2,044-evaluation search per arm =====")
    for k, v in sorted(res["arms"].items(), key=lambda kv: kv[1]["projected_total_h"]):
        print(f"  {k:26s} {v['projected_total_h']:6.2f} GPU-hours", flush=True)
    res["wall_s"] = round(time.time() - t0, 1)
    res["note"] = ("per-evaluation cost interpolated linearly in k from probed "
                   "points k=1,4,8 and weighted by the real candidate count at "
                   "each greedy step; N_CAND=3 medians per cell")
    json.dump(res, open(f"{OUT}/cost_pilot.json", "w"), indent=1)
    print(f"\npilot wall {res['wall_s']}s; written: {OUT}/cost_pilot.json", flush=True)


if __name__ == "__main__":
    main()
