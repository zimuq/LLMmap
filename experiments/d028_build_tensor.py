"""D028 / S3 — the frozen v2 tensor (I6), with D007's statistics UNCHANGED.

`d007_build_tensor.py` cannot be run as-is on 85 models: its loader,
`d007_lib.load_corpus`, hard-asserts 37 models and reads corpus_v1. This
wrapper supplies only an 85-model S_build cube loader over data/corpus_v2
(identical cube layout: (259, 75, 1024) float32, configs in sorted index
order). It imports `pair_worker` (probe AUC, energy distance raw and
scale-free, split-half probes) and uses D007's SPLIT_SEED. Equivalence is
proved, not assumed: for every pair of two REUSED v1 models, each of the five
v2 tensors must equal D007's tok200 tensor (reproduction max |diff| reported;
a Call if it is not ~0).

Usage:  PYTHONPATH=.:experiments python experiments/d028_build_tensor.py [--jobs 68]
"""
import os
import json
import time
import hashlib
import argparse
import itertools

import numpy as np
from joblib import Parallel, delayed

from d007_build_tensor import pair_worker, SPLIT_SEED

OUT = "./results/D028"
CACHE = os.environ.get("D028_CACHE", "/tmp/d028_cubes")
KEYS = ("probe", "energy_raw", "energy_sf", "probe_A", "probe_B")


def prepare(models):
    os.makedirs(CACHE, exist_ok=True)
    for m in models:
        s = m.replace("/", "__")
        dst = f"{CACHE}/{s}.npy"
        if os.path.exists(dst):
            continue
        a = np.load(f"./data/corpus_v2/embeddings/{s}.npy", mmap_mode="r")
        rows = json.load(open(f"./data/corpus_v2/embeddings/{s}.index.json"))["rows"]
        sel = [(i, r["query_index"], r["config"]) for i, r in enumerate(rows) if r["pool"] == "build"]
        cfgs = sorted({c for _, _, c in sel})
        assert len(cfgs) == 75
        ci = {c: j for j, c in enumerate(cfgs)}
        cube = np.empty((259, 75, a.shape[1]), np.float32)
        for i, q, c in sel:
            cube[q, ci[c]] = a[i]
        np.save(dst, cube)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=68)
    a = ap.parse_args()
    models = json.load(open("./results/D027/universe_v2.json"))["models"]
    t0 = time.time()
    prepare(models)
    pairs = list(itertools.combinations(models, 2))
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(75)
    sa, sb = np.sort(perm[:37]), np.sort(perm[37:74])
    res = Parallel(n_jobs=a.jobs, verbose=5, batch_size=1)(
        delayed(pair_worker)(CACHE, x, y, sa, sb) for x, y in pairs)
    S = {k: np.empty((259, len(pairs)), np.float32) for k in KEYS}
    for j, r in enumerate(res):
        for k, v in zip(KEYS, r):
            S[k][:, j] = v
    shas = {}
    for k, v in S.items():
        p = f"{OUT}/S_{k}_tok200_v2.npy"
        np.save(p, v)
        shas[k] = hashlib.sha256(open(p, "rb").read()).hexdigest()

    # reproduction vs D007 on pairs of two reused v1 models
    reuse = set(json.load(open(f"{OUT}/reuse.json")))
    d7 = json.load(open("./results/D007/tensor_tok200.meta.json"))
    d7idx = {p: i for i, p in enumerate(d7["pairs"])}
    repro = {}
    for k in KEYS:
        old = np.load(f"./results/D007/S_{k}_tok200.npy")
        diffs = []
        for j, (x, y) in enumerate(pairs):
            if x in reuse and y in reuse:
                diffs.append(float(np.abs(S[k][:, j] - old[:, d7idx[f"{x}|{y}"]]).max()))
        repro[k] = dict(n_pairs=len(diffs), max_abs_diff=max(diffs))
    tol = 1e-6                                     # Review Call 2 threshold, fixed before the build
    bad = {k: v for k, v in repro.items() if not v["max_abs_diff"] <= tol}
    meta = dict(schema="cdqd-tensor-v2", reproduction_tolerance=tol, reproduction_ok=not bad, token_budget=200, shape=[259, len(pairs)], n_models=len(models),
                models=models, pairs=[f"{x}|{y}" for x, y in pairs], split_seed=SPLIT_SEED,
                split_a=sa.tolist(), split_b=sb.tolist(), sha256=shas, frozen_tensor="S_energy_sf_tok200_v2 "
                "(and the other four, same build)", reproduction_vs_d007=repro,
                code="experiments/d028_build_tensor.py (loader) + d007_build_tensor.pair_worker (unchanged)",
                wall_s=round(time.time() - t0, 1), jobs=a.jobs)
    json.dump(meta, open(f"{OUT}/tensor_v2.json", "w"), indent=1)
    print(f"tensor built: {len(pairs)} pairs, {meta['wall_s']} s; reproduction {repro}")
    if bad:
        offenders = {}
        for k in bad:
            old = np.load(f"./results/D007/S_{k}_tok200.npy")
            offenders[k] = [f"{x}|{y}" for j, (x, y) in enumerate(pairs) if x in reuse and y in reuse
                            and float(np.abs(S[k][:, j] - old[:, d7idx[f"{x}|{y}"]]).max()) > tol][:50]
        json.dump(offenders, open(f"{OUT}/tensor_repro_offenders.json", "w"), indent=1)
        raise SystemExit(f"CALL: reproduction above {tol} on {sorted(bad)} -- see tensor_repro_offenders.json")


if __name__ == "__main__":
    main()
