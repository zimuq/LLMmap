"""D036 / S2 — closed-set training of CLOUD-raw only, on the frozen chains (results/D036/chains.json, committed).

Attention: D009 / D029 / D031 / D035 protocol (hparams_from_shipped, run_one), seeds 0-4, k = 1..8, canonical order
(chains.json -> train_order), S_val for early stopping only, best checkpoint reloaded on S_test (asserted = run_one).
Linear: D017 concat, 8 fits through d035_train.pmap (fork BEFORE d009_train / d023_train are imported) at 32 BLAS
threads, D029's setting (Review D036/R1, Call 4 -> (a)). CENTROID-raw (D035), CLOUD / PAPER8 (D029) are not
retrained for any label input.
A1 (same Review): CENTROID-raw linear refit at the same 32 threads, k = 1..8, stored as CENTROID|raw@32|<k>|... --
a P1 linear thread-sensitivity row only; the label uses D035's stored 8-thread counts.

Identical-input asserts (P1 §5, Call 4 (a)): a set equal to a JG / PAPER8 k-prefix -> d031_train.check_ident against
D029 (attention every seed, linear); a set equal to D035's CENTROID-raw set -> attention: every count array and the
logits equal D035's; linear: cnt_total / cnt_model / cnt_pair equal D035's (D035 fit at 8 threads), the logits and
cnt_pair_all difference is recorded in runs.json (not asserted). The A1 refit at k = 1 must equal CLOUD-raw's k = 1
fit bit for bit (same input, same threads). A failed assert stops the job before anything is written.
S_test discipline: counts and logits are WRITTEN, never printed or summarised (S4 is the first reader).
Usage:  OMP_NUM_THREADS=32 D031_WORKERS=2 D031_THREADS=32 PYTHONPATH=.:experiments python experiments/d036_train.py
"""
import os
import json
import time

import numpy as np

import d031_lib as L31
import d031_train as T
import d036_lib as L
from d009_lib import build_traces
from d035_train import pmap, att_conf

ARM = "CLOUD-raw"
A1 = "CENTROID|raw@32"
LIN_D035 = ("cnt_total", "cnt_model", "cnt_pair")       # Call 4 (a): D035's linear fits ran at 8 threads
N_SEEDS = 5
KS = list(range(1, L.K + 1))
CKDIR = f"{L.OUT}/models"


def ck_path(k, r):
    return f"{CKDIR}/{ARM}_k{k}_r{r}.ckpt"


def load_chains():
    ch = json.load(open(f"{L.OUT}/chains.json"))
    assert ch["frozen_before_training"] and not ch["stops"], ch["stops"]
    r = os.popen(f"git log --oneline -- {L.OUT}/chains.json").read().strip()
    assert r, "chains.json is not committed -- no training before it is"
    return ch


def same_as_centroid(ch, k):
    return ch["train_order"][ARM][str(k)] == ch["train_order"]["CENTROID-raw"][str(k)]


def assert_d035(cn, lg, key35, ref, keys=("cnt_total", "cnt_model", "cnt_pair", "cnt_pair_all"), logits=True):
    """Bit-identical counts (`keys`) and, if `logits`, logits vs D035's CENTROID|raw cell. Returns the disclosure
    (max |d logit|, number of differing cnt_pair_all entries) for the parts not asserted."""
    z = ref["counts"]
    for c in keys:
        assert np.array_equal(cn[c], z[f"{key35}|{c}"]), f"{key35} {c} differs from D035"
    d = np.abs(lg.astype(np.float32) - ref["logits"][key35].astype(np.float32))
    if logits:
        assert not d.any(), f"{key35} logits differ"
    return dict(asserted=list(keys) + (["logits"] if logits else []), max_abs_dlogit=float(d.max()),
                n_cnt_pair_all_differ=int((cn["cnt_pair_all"] != z[f"{key35}|cnt_pair_all"]).sum()))


def main():
    t0 = time.time()
    ch = load_chains()
    orders = {k: (ch["train_order"][ARM][str(k)], ch["train_order_tag"][ARM][str(k)]) for k in KS}
    cen_k = [k for k in KS if same_as_centroid(ch, k)]
    d35 = dict(att=dict(counts=np.load("./results/D035/run_counts.npz"), logits=np.load("./results/D035/logits_att.npz")),
               lin=dict(counts=np.load("./results/D035/linear_counts.npz"), logits=np.load("./results/D035/logits_lin.npz")))
    ctx = T.Ctx("v2", ("build", "val", "test"))
    assert ctx.models == ch["models"]
    T.CTX = ctx

    # ---------------------------------------------------------------- linear (forked, before CUDA)
    jobs = [dict(key=f"{ARM}|{k}", name=ARM, k=k, q=q, ident=tag) for k, (q, tag) in orders.items()]
    jobs += [dict(key=f"{A1}|{k}", name=A1, k=k, q=ch["train_order"]["CENTROID-raw"][str(k)],
                  ident=ch["train_order_tag"]["CENTROID-raw"][str(k)]) for k in KS]
    lstore, llog, lmeta = {}, {}, {}
    for key, cn, lg, m in pmap(T._lin_worker, jobs):
        k = int(key.rsplit("|", 1)[1])
        if key.startswith(ARM) and k in cen_k:
            m["ident_d035"] = assert_d035(cn, lg, f"CENTROID|raw|{k}", d35["lin"], LIN_D035, logits=False)
        if key.startswith(A1) and k in cen_k:
            for c, v in cn.items():
                assert np.array_equal(v, lstore[f"{ARM}|{k}|{c}"]), f"{key} {c} != {ARM}|{k} (same input, threads)"
            assert np.array_equal(lg, llog[f"{ARM}|{k}"]), f"{key} logits != {ARM}|{k}"
            m["ident_d035"] = assert_d035(cn, lg, f"CENTROID|raw|{k}", d35["lin"], LIN_D035, logits=False)
        elif key.startswith(A1):                    # same input as D035, other thread count: disclosed only
            m["vs_d035"] = assert_d035(cn, lg, f"CENTROID|raw|{k}", d35["lin"], (), logits=False)
        for c, v in cn.items():
            lstore[f"{key}|{c}"] = v
        llog[key] = lg
        lmeta[key] = m
        print(f"[S2 lin] {key} done ({m['n_iter']} it, {m['wall_s']}s)"
              f"{' (identical input: asserts passed)' if m['ident'] or k in cen_k else ''}", flush=True)

    # ---------------------------------------------------------------- attention
    from d009_train import run_one                  # after the forked linear fits (d035_train.pmap)
    from d023_train import reload_logits
    os.makedirs(CKDIR, exist_ok=True)
    store, logits, runs = {}, {}, []
    for k, (q, tag) in orders.items():
        tr_b, y_b, _, _ = build_traces(q, "build", ctx.qe, ctx.cubes["build"])
        tr_v, y_v, _, _ = build_traces(q, "val", ctx.qe, ctx.cubes["val"])
        tr_t, y_t, c_t, _ = build_traces(q, "test", ctx.qe, ctx.cubes["test"])
        hp, conf, h = att_conf(ctx, k)
        for r in range(N_SEEDS):
            key, path = f"{ARM}|{k}|r{r}", ck_path(k, r)
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
            if k in cen_k:
                assert_d035(cn, lg, f"CENTROID|raw|{k}|r{r}", d35["att"])
            for c, v in cn.items():
                store[f"{key}|{c}"] = v
            logits[key] = lg
            runs.append(dict(key=key, arm=ARM, k=k, seed=r, queries=q, ident=tag, ident_d035=k in cen_k, ckpt=path,
                             hp_hash=h, **{a: b for a, b in o.items() if a in ("wall_s", "best_ckpt", "train_acc",
                                                                                "val_acc")}))
        print(f"[S2 att] {ARM} k={k} done{' (identical input: reproduced exactly)' if tag or k in cen_k else ''} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    np.savez_compressed(f"{L.OUT}/linear_counts.npz", **lstore)
    np.savez_compressed(f"{L.OUT}/logits_lin.npz", **llog)
    np.savez_compressed(f"{L.OUT}/run_counts.npz", **store)
    np.savez_compressed(f"{L.OUT}/logits_att.npz", **logits)
    L31.jdump(dict(schema="d036-runs-v1", arm=ARM, chains_sha256=L31.sha(f"{L.OUT}/chains.json"), runs=runs,
                   linear=lmeta, identical_to_d029_k={str(k): t for k, (_, t) in orders.items() if t},
                   identical_to_d035_centroid_k=cen_k, linear_threads=T.N_THR,
                   a1=dict(key=A1, note="CENTROID-raw linear refit at the CLOUD-raw thread count (Review D036/R1 A1); "
                                        "sensitivity row only, the label uses D035's stored 8-thread counts"),
                   s_test="written to counts/logits only; not summarised in S2", code_commit=L31.git_head(),
                   wall_min=round((time.time() - t0) / 60, 1), node=os.uname().nodename), f"{L.OUT}/runs.json")
    print(f"[S2] done: {len(runs)} attention runs, {len(lmeta)} linear fits, wall {(time.time() - t0) / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
