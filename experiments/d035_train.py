"""D035 / S1b + S2 — training on the frozen chains (results/D035/chains.json).

  val    S1b (Call 3 (a)): attention (D009 / D029 protocol, seeds 0-4, k = 1..8, canonical order) for CENTROID|raw
         and CENTROID|norm with ONLY the build and val cubes loaded; run_one's test slot receives the S_val traces.
         Every best checkpoint is kept and reloaded on S_val (asserted = run_one's val_acc). CENTROID* = the variant
         with the higher mean over k of the 5-seed mean S_val top-1 (exact tie -> norm). Linear (D017 concat)
         S_val top-1 per k for both variants = diagnostic only (Review A1). -> variant_choice.json, runs_val.json.
  test   S2 (variant_choice.json committed; asserted): CENTROID|raw / |norm = reload the S1b checkpoints on S_test
         (no retraining); SINGLE|<scaling of CENTROID*> = run_one with S_test, seeds 0-4; linear fits for all
         three. -> run_counts.npz, linear_counts.npz, logits_*.npz (gitignored), runs.json.

Identical-input asserts: a set equal to D029's JG / PAPER8 k-prefix reproduces D029's S_val accuracies (val) and
counts (test) exactly (d031_train.check_ident); identical sets between two new arms give bit-identical logits.
S_test discipline: counts and logits are WRITTEN, never printed or summarised (S3 is the first reader).
Usage:  PYTHONPATH=.:experiments python experiments/d035_train.py val|test
"""
import os
import sys
import json
import time

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import d031_lib as L31
import d031_train as T
import d035_lib as L
from d009_lib import build_traces, hparams_from_shipped

N_SEEDS = 5
KS = list(range(1, L.K + 1))
CENTROIDS = ("CENTROID|raw", "CENTROID|norm")
CKDIR = f"{L.OUT}/models"


def ck_path(arm, k, r):
    return f"{CKDIR}/{arm.replace('|', '-')}_k{k}_r{r}.ckpt"


def load_chains():
    ch = json.load(open(f"{L.OUT}/chains.json"))
    assert ch["frozen_before_training"] and not ch["stops"], ch["stops"]
    r = os.popen(f"git log --oneline -- {L.OUT}/chains.json").read().strip()
    assert r, "chains.json is not committed -- no training before it is"
    return ch


def orders(ch, arm):
    return {k: (ch["train_order"][arm][str(k)], ch["train_order_tag"][arm][str(k)]) for k in KS}


def d029_val_acc():
    out = {}
    for tag in ("JG", "PAPER8"):
        for r in json.load(open(f"./results/D029/runs_{tag}.json"))["attention_runs"]:
            out[(tag, r["k"], r["seed"])] = r["val_acc"]
    return out


def att_conf(ctx, k):
    hp, conf = hparams_from_shipped(k, ctx.n)
    conf = dict(conf); conf["inference_model"] = hp
    return hp, conf, T.hp_check("v2", hp)


def pmap(fn, jobs):
    """d031_train.pool_map, but a dead worker raises (BrokenProcessPool) instead of hanging the job. Linear fits
    fork BEFORE d009_train / d023_train are imported: their module-level torch.cuda.is_available() initialises
    CUDA, and lbfgs segfaulted in every forked worker after it (S1b job 1115748)."""
    assert "d009_train" not in sys.modules and "d023_train" not in sys.modules, "fork after CUDA init"
    with ProcessPoolExecutor(T.N_WORK, mp_context=mp.get_context("fork")) as ex:
        for r in ex.map(fn, jobs):
            yield r


# ------------------------------------------------------------------ S1b
def lin_val(ch):
    """Linear S_val diagnostic (forked workers; run before torch touches CUDA)."""
    jobs = [dict(key=f"{a}|{k}", q=orders(ch, a)[k][0]) for a in CENTROIDS for k in KS]
    out = {}
    for key, acc, n_it, w in pmap(T._val_worker, jobs):
        out[key] = dict(val_top1=acc, n_iter=n_it, wall_s=w)
        print(f"[S1b lin] {key}: S_val top-1 {acc:.4f} ({n_it} it, {w}s)", flush=True)
    return out


def cmd_val():
    t0 = time.time()
    ch = load_chains()
    ctx = T.Ctx("v2", ("build", "val"))
    assert ctx.models == ch["models"]
    T.CTX = ctx
    lin = lin_val(ch)
    from d009_train import run_one                  # after the forked linear fits (see pmap)
    from d023_train import reload_logits
    ref = d029_val_acc()
    os.makedirs(CKDIR, exist_ok=True)
    runs, vlog, seen = [], {}, {}
    for arm in CENTROIDS:
        for k, (q, tag) in orders(ch, arm).items():
            tr_b, y_b, _, _ = build_traces(q, "build", ctx.qe, ctx.cubes["build"])
            tr_v, y_v, c_v, _ = build_traces(q, "val", ctx.qe, ctx.cubes["val"])
            hp, conf, h = att_conf(ctx, k)
            accs = []
            for r in range(N_SEEDS):
                path = ck_path(arm, k, r)
                o, _, buf = run_one(tr_b, y_b, tr_v, y_v, tr_v, y_v, c_v, hp, conf, r, ctx.hard, ctx.n, path)
                assert not o["error"], o["error"] + buf.getvalue()[-800:]
                lg = reload_logits(path, tr_v, ctx.n, k).astype(np.float32)
                acc = float((lg.argmax(1) == y_v).mean())
                assert abs(acc - o["val_acc"]) < 1e-9 and abs(acc - o["mean_top1"]) < 1e-9, (arm, k, r, acc, o)
                if tag:
                    assert o["val_acc"] == ref[(tag, k, r)], f"{arm} k={k} r={r}: S_val {o['val_acc']} != D029 {tag}"
                sk = (tuple(q), r)
                if sk in seen:
                    assert np.array_equal(lg, seen[sk][1]), f"{arm} k={k} r={r}: logits differ from {seen[sk][0]}"
                else:
                    seen[sk] = (f"{arm}|{k}", lg)
                vlog[f"{arm}|{k}|r{r}"] = lg
                accs.append(acc)
                runs.append(dict(arm=arm, k=k, seed=r, queries=q, ident=tag, hp_hash=h, ckpt=path,
                                 **{a: b for a, b in o.items() if a in ("wall_s", "best_ckpt", "train_acc", "val_acc")}))
            print(f"[S1b att] {arm} k={k}: S_val top-1 {np.round(accs, 4).tolist()} mean {np.mean(accs):.4f}"
                  f"{f' (= D029 {tag}, exact)' if tag else ''} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    att = {a: [float(np.mean([x["val_acc"] for x in runs if x["arm"] == a and x["k"] == k])) for k in KS]
           for a in CENTROIDS}
    score = {a: float(np.mean(v)) for a, v in att.items()}
    star = "CENTROID|raw" if score["CENTROID|raw"] > score["CENTROID|norm"] else "CENTROID|norm"
    lin_k = {a: [lin[f"{a}|{k}"]["val_top1"] for k in KS] for a in CENTROIDS}
    lin_score = {a: float(np.mean(v)) for a, v in lin_k.items()}
    lin_pref = ("CENTROID|raw" if lin_score["CENTROID|raw"] > lin_score["CENTROID|norm"] else
                "CENTROID|norm" if lin_score["CENTROID|norm"] > lin_score["CENTROID|raw"] else "tie")
    res = dict(
        schema="d035-variant-choice-v1", s_test_reads=0, chains_sha256=L31.sha(f"{L.OUT}/chains.json"),
        rule="CENTROID* = higher mean over k = 1..8 of the 5-seed mean S_val attention top-1 (reloaded best "
             "checkpoints); exact tie -> CENTROID|norm (Call 3 (a))",
        attention_val_top1_by_k=att, attention_score=score, CENTROID_star=star,
        other_centroid=[a for a in CENTROIDS if a != star][0],
        single_trained=f"SINGLE|{star.split('|')[1]}", single_selection_only=f"SINGLE|{'raw' if star.endswith('norm') else 'norm'}",
        linear_diagnostic=dict(val_top1_by_k=lin_k, score=lin_score, prefers=lin_pref,
                               agrees_with_attention=lin_pref == star, per_fit=lin,
                               note="diagnostic only; not used for the choice. Review A1: if it prefers the other "
                                    "variant, R's headline says so"),
        identical_to_d029_asserted=[f"{x['arm']}|{x['k']}|r{x['seed']}" for x in runs if x["ident"]],
        code_commit=L31.git_head(), wall_min=round((time.time() - t0) / 60, 1), node=os.uname().nodename)
    L31.jdump(res, f"{L.OUT}/variant_choice.json")
    L31.jdump(dict(schema="d035-runs-val-v1", runs=runs, s_test_reads=0), f"{L.OUT}/runs_val.json")
    np.savez_compressed(f"{L.OUT}/logits_val.npz", **vlog)
    print(f"[S1b] attention score {score} -> CENTROID* = {star}; linear diagnostic {lin_score} prefers {lin_pref}",
          flush=True)
    print(f"[S1b] wrote variant_choice.json ({res['wall_min']} min)", flush=True)


# ------------------------------------------------------------------ S2
def assert_choice_committed():
    vc = json.load(open(f"{L.OUT}/variant_choice.json"))
    assert vc["s_test_reads"] == 0 and vc["chains_sha256"] == L31.sha(f"{L.OUT}/chains.json")
    r = os.popen(f"git log --oneline -- {L.OUT}/variant_choice.json").read().strip()
    assert r, "variant_choice.json is not committed -- no S_test computation before it is"
    return vc


def lin_test(ch, arms):
    jobs = []
    for a in arms:
        for k, (q, tag) in orders(ch, a).items():
            jobs.append(dict(key=f"{a}|{k}", name=a, k=k, q=q, ident=tag))
    store, logits, meta = {}, {}, {}
    for key, cn, lg, m in pmap(T._lin_worker, jobs):
        for c, v in cn.items():
            store[f"{key}|{c}"] = v
        logits[key] = lg
        meta[key] = m
        print(f"[S2 lin] {key} done ({m['n_iter']} it, {m['wall_s']}s)"
              f"{' (identical input: reproduced exactly)' if m['ident'] else ''}", flush=True)
    return store, logits, meta


def cmd_test():
    t0 = time.time()
    vc = assert_choice_committed()
    ch = load_chains()
    single = vc["single_trained"]
    arms = list(CENTROIDS) + [single]
    ctx = T.Ctx("v2", ("build", "val", "test"))
    assert ctx.models == ch["models"]
    T.CTX = ctx
    lstore, llog, lmeta = lin_test(ch, arms)
    np.savez_compressed(f"{L.OUT}/linear_counts.npz", **lstore)
    np.savez_compressed(f"{L.OUT}/logits_lin.npz", **llog)
    from d009_train import run_one                  # after the forked linear fits (see pmap)
    from d023_train import reload_logits
    vref = json.load(open(f"{L.OUT}/runs_val.json"))["runs"]
    vref = {(x["arm"], x["k"], x["seed"]): x for x in vref}
    store, logits, runs, seen = {}, {}, [], {}
    for arm in arms:
        for k, (q, tag) in orders(ch, arm).items():
            tr_t, y_t, c_t, _ = build_traces(q, "test", ctx.qe, ctx.cubes["test"])
            if arm == single:
                tr_b, y_b, _, _ = build_traces(q, "build", ctx.qe, ctx.cubes["build"])
                tr_v, y_v, _, _ = build_traces(q, "val", ctx.qe, ctx.cubes["val"])
                hp, conf, h = att_conf(ctx, k)
            for r in range(N_SEEDS):
                key = f"{arm}|{k}|r{r}"
                path = ck_path(arm, k, r)
                if arm == single:
                    o, st, buf = run_one(tr_b, y_b, tr_v, y_v, tr_t, y_t, c_t, hp, conf, r, ctx.hard, ctx.n, path)
                    assert not o["error"], o["error"] + buf.getvalue()[-800:]
                    lg = reload_logits(path, tr_t, ctx.n, k)
                    assert abs(float((lg.argmax(1) == y_t).mean()) - o["mean_top1"]) < 1e-9, "reload mismatch"
                    cn = T.counts(ctx, lg, y_t, c_t)
                    for c in ("cnt_total", "cnt_model", "cnt_pair"):
                        assert np.array_equal(cn[c], st[c]), (key, c)
                    meta = dict(source="trained (S2)", hp_hash=h,
                                **{a: b for a, b in o.items() if a in ("wall_s", "best_ckpt", "train_acc", "val_acc")})
                else:
                    v = vref[(arm, k, r)]
                    assert v["ckpt"] == path and v["queries"] == q
                    lg = reload_logits(path, tr_t, ctx.n, k)
                    cn = T.counts(ctx, lg, y_t, c_t)
                    meta = dict(source="S1b checkpoint reloaded (Call 3 (a))", hp_hash=v["hp_hash"],
                                **{a: v[a] for a in ("wall_s", "best_ckpt", "train_acc", "val_acc")})
                lg = lg.astype(np.float32)
                if tag:
                    T.check_ident("v2", "att", tag, k, cn, r)
                sk = (tuple(q), r)
                if sk in seen:
                    assert np.array_equal(lg, seen[sk][1]), f"{key}: logits differ from {seen[sk][0]}"
                else:
                    seen[sk] = (key, lg)
                for c, v in cn.items():
                    store[f"{key}|{c}"] = v
                logits[key] = lg
                runs.append(dict(key=key, arm=arm, k=k, seed=r, queries=q, ident=tag, ckpt=path, **meta))
            print(f"[S2 att] {arm} k={k} done{' (identical input: reproduced exactly)' if tag else ''} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    np.savez_compressed(f"{L.OUT}/run_counts.npz", **store)
    np.savez_compressed(f"{L.OUT}/logits_att.npz", **logits)
    L31.jdump(dict(schema="d035-runs-v1", centroid_star=vc["CENTROID_star"], single=single, arms=arms, runs=runs,
                   linear=lmeta, identical_input_asserted=sorted({r["key"] for r in runs if r["ident"]} |
                                                                 {k for k, m in lmeta.items() if m["ident"]}),
                   s_test="written to counts/logits only; not summarised in S2", code_commit=L31.git_head(),
                   wall_min=round((time.time() - t0) / 60, 1), node=os.uname().nodename), f"{L.OUT}/runs.json")
    print(f"[S2] done: {len(runs)} attention runs, {len(lmeta)} linear fits, wall {(time.time() - t0) / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    {"val": cmd_val, "test": cmd_test}[sys.argv[1]]()
