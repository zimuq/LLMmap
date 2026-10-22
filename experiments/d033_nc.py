"""D033 / S1 + S2(2) — readout (2), nearest point cloud (training-free), on frozen fold chains (chains.json).

  val     S1: per view, per-query point-to-cloud terms for S_val targets over the view's union queries; code-path
          assert against the frozen raw-energy tensor; raw vs scale-free by O_all of HELD-OUT S_val targets
          (model-level folds, OURS^fold, k = 8; tie -> raw) -> variant_choice.json. No S_test trace is loaded.
  test    S2: the same terms for S_test targets -> nc_test_<view>.npz (gitignored). Written, never summarised.

Terms: A[t, m, j] = mean_i ||x_{t,q_j} - Y_{m,q_j,i}||, W[m, j] = V-statistic within-cloud mean distance of m's
75 S_build points (d033_lib.nc_terms). Score(t, m) = sum over the chain of 2A - W (raw) or (2A - W) / W (sf).
Usage:  PYTHONPATH=.:experiments python experiments/d033_nc.py val|test
"""
import sys
import json
import time

import numpy as np
from scipy.spatial.distance import cdist

import d031_lib as L31
import d033_lib as L

PAIRS_CHECK = lambda P: (0, P // 2, P - 1)
Q_CHECK = (0, 64, 128, 193, 258)


def view_queries(ch, view):
    return sorted({int(q) for q, src in ch["union"]["sources"].items() if any(f"|{view}" in s for s in src)})


def heldout_D(A, W, U, ch, view, kind, arm, variant, k, n):
    """(n*25, n) scores; each target row uses the chain of the fold its model is held out in."""
    ms = L31.models(view)
    pos = {q: j for j, q in enumerate(U)}
    D = np.empty((A.shape[0], n))
    nct = A.shape[0] // n
    for F in ch["views"][view]["kinds"][kind]["folds"]:
        cols = [pos[q] for q in F["arms"][arm][:k]]
        rows = np.concatenate([np.arange(ms.index(m) * nct, (ms.index(m) + 1) * nct) for m in F["heldout"]])
        D[rows] = L.nc_scores(A[rows], W, cols, variant)
    return D


def code_path_assert(view, cubes_b):
    _, raw = L31.tensors(view)
    ms = L31.models(view)
    ia, ib = L31.pairs(view)
    worst = 0.0
    for p in PAIRS_CHECK(len(ia)):
        a, b = ms[ia[p]], ms[ib[p]]
        for q in Q_CHECK:
            X = np.asarray(cubes_b[a][q], np.float64)
            Y = np.asarray(cubes_b[b][q], np.float64)
            two = {a: cubes_b[a], b: cubes_b[b]}
            A, W = L.nc_terms(two, two, [a, b], [q])
            # rows 0..74 = a's S_build points; column 1 = model b; W[0], W[1] = within of a, b
            e = 2.0 * A[:75, 1, 0].mean() - W[0, 0] - W[1, 0]
            ref = float(raw[q, p])
            direct = 2 * cdist(X, Y).mean() - cdist(X, X).mean() - cdist(Y, Y).mean()
            rel = abs(e - ref) / abs(ref)
            worst = max(worst, rel)
            assert rel <= 1e-5 and abs(e - direct) <= 1e-9, (view, a, b, q, e, ref, direct)
    return dict(pairs=[int(p) for p in PAIRS_CHECK(len(ia))], queries=list(Q_CHECK), max_rel_err_vs_tensor=worst)


def cmd_val():
    t0 = time.time()
    assert L.committed(f"{L.OUT}/chains.json"), "chains.json must be committed and pushed before S1"
    ch = json.load(open(f"{L.OUT}/chains.json"))
    out = dict(schema="d033-variant-choice-v1", chains_sha256=L31.sha(f"{L.OUT}/chains.json"),
               rule="O_all of held-out S_val targets, model-level folds, OURS^fold, k = 8; higher wins; tie -> raw",
               s_test_reads=0, views={})
    for view in L.VIEWS:
        ms = L31.models(view)
        n = len(ms)
        U = view_queries(ch, view)
        cubes = L31.cubes(view, ("build", "val"))
        cp = code_path_assert(view, cubes["build"][1])
        print(f"[S1 {view}] code-path: point-to-cloud terms reproduce the frozen raw tensor "
              f"(max rel err {cp['max_rel_err_vs_tensor']:.2e})", flush=True)
        A, W = L.nc_terms(cubes["build"][1], cubes["val"][1], ms, U)
        del cubes
        np.savez_compressed(f"{L.OUT}/nc_val_{view}.npz", A=A, W=W, U=np.array(U), models=np.array(ms))
        ia, ib = L31.pairs(view)
        acc = {}
        for v in L.NC_VARIANTS:
            D = heldout_D(A, W, U, ch, view, "model", "OURS", v, L.K, n)
            acc[v] = float(L.counts(D, n, ia, ib)["cnt_model"].mean())
        choice = "sf" if acc["sf"] > acc["raw"] else "raw"
        out["views"][view] = dict(s_val_O_all=acc, choice=choice, n_union_queries=len(U), code_path=cp)
        print(f"[S1 {view}] S_val held-out O_all (OURS^fold, k=8): raw {acc['raw']:.4f}  sf {acc['sf']:.4f} "
              f"-> {choice}", flush=True)
    out["code_commit"] = L.git_head()
    out["wall_min"] = round((time.time() - t0) / 60, 2)
    L.jdump(out, f"{L.OUT}/variant_choice.json")
    print(f"[S1] wrote {L.OUT}/variant_choice.json ({out['wall_min']} min)", flush=True)


def cmd_test():
    t0 = time.time()
    assert L.committed(f"{L.OUT}/variant_choice.json"), "variant_choice.json must be committed before S2"
    ch = json.load(open(f"{L.OUT}/chains.json"))
    for view in L.VIEWS:
        ms = L31.models(view)
        U = view_queries(ch, view)
        cubes = L31.cubes(view, ("build", "test"))
        A, W = L.nc_terms(cubes["build"][1], cubes["test"][1], ms, U)
        del cubes
        np.savez_compressed(f"{L.OUT}/nc_test_{view}.npz", A=A, W=W, U=np.array(U), models=np.array(ms))
        print(f"[S2 nc {view}] wrote nc_test_{view}.npz ({len(U)} queries; not summarised)", flush=True)
    print(f"[S2 nc] done ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    {"val": cmd_val, "test": cmd_test}[sys.argv[1]]()
