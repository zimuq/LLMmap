"""D037 / S2 — closed-set training of JG-mean and GC-mean only, on the frozen chains (results/D037/chains.json,
committed). JG, GC, PAPER8 (D029), TOP-K[CVaR], MMR-PP[CVaR] (D031) and TOP-K[MEAN] (D032) are read verbatim in S3.

Attention: D009 / D029 / D031 / D035 / D036 protocol (att_conf, run_one), seeds 0-4, k = 1..8, chains.json ->
train_order, S_val for early stopping only, best checkpoint reloaded on S_test (asserted = run_one).
Linear: D017 concat, 16 fits through d035_train.pmap (fork BEFORE d009_train / d023_train are imported) at 32 BLAS
threads, D029's setting (D036 Call 4).

Identical-input asserts (P1 §5; Review A1: against the tag train_order returned, never another D029 arm):
  tag JG / PAPER8 / GC -> d031_train.check_ident against D029 counts_<tag>.npz (attention every seed, linear);
  tag GC additionally   -> logits equal D029 logits_GC.npz (att|k|r, lin|k);
  JG-mean and GC-mean with the same set at k (same set -> same order) -> every count array and logits bit-identical.
A failed assert stops the job before anything is written.
S_test discipline: counts and logits are WRITTEN, never printed or summarised (S3b is the first reader).
Usage:  OMP_NUM_THREADS=32 D031_WORKERS=2 D031_THREADS=32 PYTHONPATH=.:experiments python experiments/d037_train.py
"""
import os
import json
import time

import numpy as np

import d031_lib as L31
import d031_train as T
import d037_lib as L
from d009_lib import build_traces
from d035_train import pmap, att_conf

ARMS = ("JG-mean", "GC-mean")
N_SEEDS = 5
KS = list(range(1, L.K + 1))
CKDIR = f"{L.OUT}/models"


def ck_path(arm, k, r):
    return f"{CKDIR}/{arm}_k{k}_r{r}.ckpt"


def load_chains():
    ch = json.load(open(f"{L.OUT}/chains.json"))
    assert ch["frozen_before_training"] and not ch["stops"], ch["stops"]
    r = os.popen(f"git log --oneline -- {L.OUT}/chains.json").read().strip()
    assert r, "chains.json is not committed -- no training before it is"
    return ch


def assert_gc_logits(lg, key, ref):
    assert np.array_equal(lg.astype(np.float32), ref[key].astype(np.float32)), f"logits differ from D029 GC {key}"


def assert_same(store, logits, a, b):
    """Every count array and the logits of cell a equal cell b's (same set -> same order -> same input)."""
    pa = [c for c in store if c.startswith(a + "|")]
    assert pa
    for c in pa:
        assert np.array_equal(store[c], store[b + c[len(a):]]), f"{c} != {b} (same set, same order)"
    assert np.array_equal(logits[a], logits[b]), f"logits {a} != {b}"


def main():
    t0 = time.time()
    ch = load_chains()
    orders = {a: {k: (ch["train_order"][a][str(k)], ch["train_order_tag"][a][str(k)]) for k in KS} for a in ARMS}
    same = ch["new_arms_same_set_k"]
    assert all(orders["JG-mean"][k][0] == orders["GC-mean"][k][0] for k in same)
    gcl = np.load("./results/D029/logits_GC.npz")
    ctx = T.Ctx("v2", ("build", "val", "test"))
    assert ctx.models == ch["models"]
    T.CTX = ctx

    # ---------------------------------------------------------------- linear (forked, before CUDA)
    jobs = [dict(key=f"{a}|{k}", name=a, k=k, q=q, ident=tag) for a in ARMS for k, (q, tag) in orders[a].items()]
    lstore, llog, lmeta = {}, {}, {}
    for key, cn, lg, m in pmap(T._lin_worker, jobs):                # check_ident(tag) runs inside the worker
        a, k = key.rsplit("|", 1)
        if orders[a][int(k)][1] == "GC":
            assert_gc_logits(lg, f"lin|{k}", gcl)
        for c, v in cn.items():
            lstore[f"{key}|{c}"] = v
        llog[key] = lg
        lmeta[key] = m
        print(f"[S2 lin] {key} done ({m['n_iter']} it, {m['wall_s']}s)"
              f"{' (identical input: asserts passed)' if m['ident'] else ''}", flush=True)
    for k in same:
        assert_same(lstore, llog, f"JG-mean|{k}", f"GC-mean|{k}")

    # ---------------------------------------------------------------- attention
    from d009_train import run_one                  # after the forked linear fits (d035_train.pmap)
    from d023_train import reload_logits
    os.makedirs(CKDIR, exist_ok=True)
    store, logits, runs = {}, {}, []
    for k in KS:
        hp, conf, h = att_conf(ctx, k)
        for a in ARMS:
            q, tag = orders[a][k]
            tr_b, y_b, _, _ = build_traces(q, "build", ctx.qe, ctx.cubes["build"])
            tr_v, y_v, _, _ = build_traces(q, "val", ctx.qe, ctx.cubes["val"])
            tr_t, y_t, c_t, _ = build_traces(q, "test", ctx.qe, ctx.cubes["test"])
            for r in range(N_SEEDS):
                key, path = f"{a}|{k}|r{r}", ck_path(a, k, r)
                o, st, buf = run_one(tr_b, y_b, tr_v, y_v, tr_t, y_t, c_t, hp, conf, r, ctx.hard, ctx.n, path)
                assert not o["error"], o["error"] + buf.getvalue()[-800:]
                lg = reload_logits(path, tr_t, ctx.n, k)
                assert abs(float((lg.argmax(1) == y_t).mean()) - o["mean_top1"]) < 1e-9, "reload mismatch"
                cn = T.counts(ctx, lg, y_t, c_t)
                for c in ("cnt_total", "cnt_model", "cnt_pair"):
                    assert np.array_equal(cn[c], st[c]), (key, c)
                lg = lg.astype(np.float32)
                if tag:
                    T.check_ident("v2", "att", tag, k, cn, r)
                if tag == "GC":
                    assert_gc_logits(lg, f"att|{k}|{r}", gcl)
                for c, v in cn.items():
                    store[f"{key}|{c}"] = v
                logits[key] = lg
                runs.append(dict(key=key, arm=a, k=k, seed=r, queries=q, ident=tag, ckpt=path, hp_hash=h,
                                 **{x: y for x, y in o.items() if x in ("wall_s", "best_ckpt", "train_acc",
                                                                        "val_acc")}))
            print(f"[S2 att] {a} k={k} done{f' (identical input to D029 {tag}: reproduced exactly)' if tag else ''} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        if k in same:
            for r in range(N_SEEDS):
                assert_same(store, logits, f"JG-mean|{k}|r{r}", f"GC-mean|{k}|r{r}")
            print(f"[S2 att] k={k}: JG-mean = GC-mean set, bit-identical", flush=True)

    np.savez_compressed(f"{L.OUT}/linear_counts.npz", **lstore)
    np.savez_compressed(f"{L.OUT}/logits_lin.npz", **llog)
    np.savez_compressed(f"{L.OUT}/run_counts.npz", **store)
    np.savez_compressed(f"{L.OUT}/logits_att.npz", **logits)
    L31.jdump(dict(schema="d037-runs-v1", arms=list(ARMS), chains_sha256=L31.sha(f"{L.OUT}/chains.json"), runs=runs,
                   linear=lmeta, identical_to_d029={a: {str(k): t for k, (_, t) in o.items() if t}
                                                    for a, o in orders.items()},
                   new_arms_same_set_k=same, linear_threads=T.N_THR,
                   s_test="written to counts/logits only; not summarised in S2", code_commit=L31.git_head(),
                   wall_min=round((time.time() - t0) / 60, 1), node=os.uname().nodename), f"{L.OUT}/runs.json")
    print(f"[S2] done: {len(runs)} attention runs, {len(lmeta)} linear fits, wall {(time.time() - t0) / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
