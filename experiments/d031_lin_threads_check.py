"""D031 / S2 diagnostic (after the v1 linear identical-input assert failed at JG k = 2 under 8 BLAS threads).

Hypothesis: the lbfgs fit is bit-deterministic only at a fixed BLAS thread count (reduction order); D020
(and D030 Call 1, which reproduced D020 bit-exactly) ran in the main process with OMP_NUM_THREADS=16,
d031_train's linear pool with 8. Run 1 (job 1067103, both thread counts in one process) showed 8 threads
!= D020 at k = 2 and 8, then segfaulted forking the second pool after OpenMP was live in the parent; so
each configuration now runs in its own process:
  main T   fit in the main process under threadpool_limits(T)   (D020 / D030 Call 1 replica at T = 16)
  pool T   fit in a d031_train fork-pool worker with N_THR = T    (the production path)
Refits v1 JG k in {2, 8} (D020 code path, D010 chain); exits 1 unless cnt_total / cnt_model equal D020's
stored counts. Equality checks only; no S_test metric is printed. Appends to results/D031/lin_threads_check.json.
Usage:  PYTHONPATH=.:experiments python experiments/d031_lin_threads_check.py {main|pool} T
"""
import os
import sys
import json

import numpy as np

import d031_train as T
import d031_lib as L

PATH = f"{L.OUT}/lin_threads_check.json"


def _w(job):
    o, n_it = T.lin_fit(job["q"], ("test",))
    lg, y, c = o["test"]
    return job["k"], T.counts(T.CTX, lg, y, c), n_it


def main(mode, t):
    T.CTX = T.Ctx("v1", ("build", "test"))
    T.N_THR, T.N_WORK = t, 2
    z = np.load("./results/D020/linear_counts.npz")
    jg = L.jg_chain("v1")
    jobs = [dict(k=k, q=jg[:k]) for k in (2, 8)]
    res = map(_w, jobs) if mode == "main" else T.pool_map(_w, jobs)
    out = json.load(open(PATH)) if os.path.exists(PATH) else {}
    ok = True
    for k, cn, n_it in res:
        eq = {f: bool(np.array_equal(cn[f], z[f"JointGreedy|concat|{k}|{f}"])) for f in ("cnt_total", "cnt_model")}
        ok &= all(eq.values())
        out[f"k{k}|{mode}|threads{t}"] = dict(equal_to_D020=eq, n_iter=n_it, omp=os.environ.get("OMP_NUM_THREADS"))
        print(f"k{k} {mode} threads{t}: equal_to_D020 {eq} n_iter {n_it}", flush=True)
    L.jdump(out, PATH)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
