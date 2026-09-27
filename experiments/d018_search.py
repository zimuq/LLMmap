"""D018 / S1 — Algorithm H.1's real greedy search, on our 259-query pool.

    at step k, for every remaining candidate q:
        f <- train(current_best + {q}, S_build)
        A <- eval(f, S_val)                    # Call 1: S_val, never S_test
    q* <- argmax A

Two arms, run independently because the argmax depends on which classifier
produces A (D018's own point -- these are two chains, not one chain scored twice):

  attention  -- the paper's own network. THIS ARM IS ALGORITHM H.1.
  linear     -- same procedure, a classifier the paper never used and which D017
                showed is better at every k. NOT Algorithm H.1; a robustness check
                on whether the search procedure or the classifier does the work
                (P1/F1, approved). Mean-pool at the TIGHT solver (Call 3): concat
                costs 2.6x for an argmax, and loosening the solver instead would
                risk selecting on the 0.004 non-convergence wobble D017 measured.

One seed during the search (Call 2) -- Algorithm H.1's "372 training runs" implies
one run per candidate, so this is more faithful than 5, not a compromise. Each
step's WINNING MARGIN over the runner-up is recorded, because P1/F3 pre-registered
that those margins are often smaller than the attention network's own seed range:
the chain is noisy, and that is a property of H.1 as specified.

Usage:  PYTHONPATH=.:experiments python experiments/d018_search.py --arm attention
        PYTHONPATH=.:experiments python experiments/d018_search.py --arm linear
"""
import os
import io
import json
import time
import shutil
import hashlib
import argparse
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
K_MAX, SEED = 8, 0
MAX_ITER, TOL, C_MEANPOOL = 20000, 1e-6, 3.0


def eval_attention(q, qe, cubes, n_models, hash_ref):
    tr, y_tr, _, _ = build_traces(q, "build", qe, cubes["build"])
    va, y_va, _, _ = build_traces(q, "val", qe, cubes["val"])
    hp, conf = hparams_from_shipped(len(q), n_models)
    conf = dict(conf); conf["inference_model"] = hp
    h = hashlib.sha256(json.dumps({k: v for k, v in hp.items()
                                   if k != "num_queries"},
                                  sort_keys=True).encode()).hexdigest()
    assert h == hash_ref, (h, hash_ref)
    pl.seed_everything(SEED, workers=True)
    tmp = tempfile.mkdtemp(prefix="d018_")
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            _, net = train_model(tmp, siamese=False,
                                 loader_train=make_loader(tr, y_tr,
                                                          conf["batch_size"], True),
                                 loader_test=make_loader(va, y_va,
                                                         conf["batch_size"], False),
                                 conf=conf)
        return float((logits_for(net, va, DEV).argmax(1) == y_va).mean())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def eval_linear(q, qe, cubes, n_models, hash_ref):
    tr, y_tr, _, _ = build_traces(q, "build", qe, cubes["build"])
    va, y_va, _, _ = build_traces(q, "val", qe, cubes["val"])
    clf = LogisticRegression(C=C_MEANPOOL, max_iter=MAX_ITER, tol=TOL,
                             solver="lbfgs")
    clf.fit(tr.mean(axis=1), y_tr)
    assert int(clf.n_iter_.max()) < MAX_ITER, "linear fit hit max_iter"
    return float((clf.predict(va.mean(axis=1)) == y_va).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["attention", "linear"], required=True)
    ap.add_argument("--kmax", type=int, default=K_MAX)
    ap.add_argument("--pool-limit", type=int, default=None,
                    help="SMOKE ONLY: restrict the candidate pool")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    qe = load_query_embeddings()
    cubes = {p: load_corpus(pool=p) for p in ("build", "val")}
    models = json.load(open("./results/D008/selection.json"))["models"]
    n_models = len(models)
    hash_ref = json.load(open("./results/D009/runs.json"))["hparams_hash"]
    n_pool = qe.shape[0] if args.pool_limit is None else args.pool_limit
    evaluate = eval_attention if args.arm == "attention" else eval_linear
    print(f"[S1] arm={args.arm}  pool={n_pool}  k=1..{args.kmax}  seed={SEED}  "
          f"argmax on S_val (Call 1)", flush=True)

    ckpt = f"{OUT}/greedy_{args.arm}_partial.json"
    chain, trace = [], []
    if os.path.exists(ckpt):                       # resume, so a wall-clock
        st = json.load(open(ckpt))                 # limit never loses the prefix
        chain, trace = st["chain"], st["trace"]
        print(f"[S1] resuming from k={len(chain)}: {chain}", flush=True)

    t0 = time.time()
    for k in range(len(chain) + 1, args.kmax + 1):
        cands = [q for q in range(n_pool) if q not in chain]
        scores = {}
        t = time.time()
        for i, q in enumerate(cands):
            scores[q] = evaluate(chain + [q], qe, cubes, n_models, hash_ref)
            if (i + 1) % 50 == 0:
                print(f"    k={k} {i+1}/{len(cands)}  "
                      f"({(time.time()-t)/60:.1f} min)", flush=True)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        best, best_a = ranked[0]
        runner, runner_a = ranked[1]
        chain.append(int(best))
        trace.append(dict(k=k, chosen=int(best), val_acc=round(best_a, 6),
                          runner_up=int(runner), runner_up_acc=round(runner_a, 6),
                          margin=round(best_a - runner_a, 6),
                          n_candidates=len(cands),
                          n_tied_with_winner=int(sum(1 for _, a in ranked
                                                     if a >= best_a - 1e-12)),
                          wall_min=round((time.time() - t) / 60, 2)))
        json.dump(dict(arm=args.arm, chain=chain, trace=trace),
                  open(ckpt, "w"), indent=1)
        print(f"[S1] k={k}: q={best} val {best_a:.4f}  margin {best_a-runner_a:+.4f}"
              f"  (runner-up q={runner})  [{(time.time()-t)/60:.1f} min]", flush=True)

    marg = [t["margin"] for t in trace]
    out = dict(schema="d018-search-v1", arm=args.arm, chain=chain, trace=trace,
               seed=SEED, selection_split="S_val (Call 1; S_test never touched)",
               classifier=("attention network -- THIS ARM IS ALGORITHM H.1"
                           if args.arm == "attention" else
                           "linear, mean-pool, C=3.0, max_iter=20000 tol=1e-6 "
                           "(Call 3) -- NOT Algorithm H.1; robustness check (F1)"),
               margins=dict(values=marg, median=float(np.median(marg)),
                            min=float(np.min(marg)), max=float(np.max(marg)),
                            note="P1/F3: the attention network's own 5-seed range "
                                 "is 0.024-0.081 on mean top-1, so a margin below "
                                 "that is not a resolved preference. The chain is "
                                 "noisy by construction -- a property of Algorithm "
                                 "H.1 as specified, not of this implementation."),
               wall_min=round((time.time() - t0) / 60, 1))
    json.dump(out, open(f"{OUT}/greedy_{args.arm}.json", "w"), indent=1)
    print(f"\n[S1] {args.arm} chain: {chain}")
    print(f"[S1] margins: median {np.median(marg):.4f}  range "
          f"[{np.min(marg):.4f}, {np.max(marg):.4f}]  "
          f"({sum(1 for m in marg if m < 0.024)}/{len(marg)} below the attention "
          f"network's own seed floor)")
    print(f"[S1] wall {out['wall_min']} min; written {OUT}/greedy_{args.arm}.json")


if __name__ == "__main__":
    main()
