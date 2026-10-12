"""D031 / S1 + S2 — readouts on frozen chains (results/D031/chains.json).

  val                 S1: linear (D017 concat) fit on S_build, top-1 on S_val at k = 8 for HELM's 4 variants
                      (SIG = mean of its 20 draws) and MMR's 6 (Review A1) -> variant_choice.json. No S_test
                      trace is built.
  lin <VIEW>          S2: every linear job of the view (all variants, all draws, HELM-ORTH included); v1 also
                      re-derives JG (k 1-8, asserted vs D020) and fits PAPER8 (asserted vs D017).
  att <VIEW> <i> <n>  S2: shard i of n of the view's attention jobs (D009 protocol, seeds 0-4).

S_test discipline: counts and logits are WRITTEN, never printed or summarised (S4 is the first reader).
Identical-input sets reproduce the stored JG / PAPER8 counts exactly (asserted; the job stops otherwise).
Counts per (job[, seed]): cnt_total (25), cnt_model (n x 25), cnt_pair (v2 H_all 57 / v1 structural 65),
dirH (v2, 57 x 2 x 25), cnt_pair_all (P x 25, uint8, both directions summed; tie -> first member).
Usage:  PYTHONPATH=.:experiments python experiments/d031_train.py val|lin|att ...
"""
import os
import sys
import json
import time
import hashlib
import multiprocessing as mp

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import d031_lib as L
from d009_lib import build_traces, hparams_from_shipped, logit_stats, load_query_embeddings
from d020_paired_linear import POOL, MAX_ITER, TOL

N_SEEDS = 5
N_WORK = int(os.environ.get("D031_WORKERS", "8"))
N_THR = int(os.environ.get("D031_THREADS", "8"))
POOL_FN, C = POOL["concat"]
CTX = None


class Ctx:
    def __init__(self, view, splits):
        self.view = view
        self.models = L.models(view)
        self.n = len(self.models)
        self.ia, self.ib = L.pairs(view)
        self.qe = load_query_embeddings()
        self.cubes = L.cubes(view, splits)
        if view == "v2":
            import d029_lib as L29
            tg = L29.targets(self.models)
            self.pa, self.pb = L29.hard_ab(self.models, tg["H"])
            self.hard = {j: dict(ab=(int(a), int(b))) for j, (a, b) in enumerate(zip(self.pa, self.pb))}
            pid = {(int(a), int(b)): j for j, (a, b) in enumerate(zip(self.ia, self.ib))}
            self.hard_cols = [pid[(int(a), int(b))] for a, b in zip(self.pa, self.pb)]
        else:
            from d008_lib import near_relative_pairs
            self.hard = near_relative_pairs(self.models)
            self.hard_cols = sorted(self.hard)


def hp_check(view, hp):
    if view == "v2":
        import d029_lib as L29
        h = hashlib.sha256(json.dumps({k: v for k, v in hp.items() if k not in ("num_queries", "num_classes")},
                                      sort_keys=True).encode()).hexdigest()
        assert h == L29.D027_HP_HASH, "hparams differ from D027/D009"
    else:
        h = hashlib.sha256(json.dumps({k: v for k, v in hp.items() if k != "num_queries"},
                                      sort_keys=True).encode()).hexdigest()
        assert h == json.load(open("./results/D009/runs.json"))["hparams_hash"], "hparams differ from D009"
    return h


def counts(ctx, lg, y, c):
    _, st = logit_stats(lg, y, c, ctx.hard, ctx.n)
    out = dict(cnt_total=st["cnt_total"], cnt_model=st["cnt_model"], cnt_pair=st["cnt_pair"])
    assert np.array_equal(y, np.repeat(np.arange(ctx.n), 25)) and np.array_equal(c, np.tile(np.arange(25), ctx.n))
    G = lg.reshape(ctx.n, 25, ctx.n)
    ia, ib = ctx.ia, ctx.ib
    d0 = G[ia, :, ia] >= G[ia, :, ib]             # traces of a; tie -> a (logit_stats / D023 rule)
    d1 = G[ib, :, ib] > G[ib, :, ia]
    out["cnt_pair_all"] = (d0.astype(np.uint8) + d1.astype(np.uint8))
    assert np.array_equal(out["cnt_pair_all"][ctx.hard_cols].astype(np.float64), st["cnt_pair"]), "pair_all != cnt_pair"
    if ctx.view == "v2":
        from d025_lib import dir_counts
        out["dirH"] = dir_counts(lg, y, c, ctx.pa, ctx.pb)
    return out


# ------------------------------------------------------------------ references for identical-input asserts
def ref_counts(view, readout, ident, k, seed=None):
    if view == "v2":
        z = np.load(f"./results/D029/counts_{ident}.npz")
        p = f"att|{k}|{seed}|" if readout == "att" else f"lin|{k}|"
        return {c: z[p + c] for c in ("cnt_total", "cnt_model")}
    if readout == "att":
        if ident == "JG":
            z, p = np.load("./results/D010/run_counts.npz"), f"joint_energy|{k}|{seed}|"
        else:
            z, p = np.load("./results/D009/run_counts.npz"), f"paper8|{k}|{seed}|"
        return {c: z[p + c] for c in ("cnt_total", "cnt_model")}
    if ident == "JG":
        z = np.load("./results/D020/linear_counts.npz")
        return {c: z[f"JointGreedy|concat|{k}|{c}"] for c in ("cnt_total", "cnt_model")}
    m = json.load(open("./results/D017/linear_metrics_by_k.json"))["metrics"]["concat"]["paper8"][str(k)]
    return dict(mean_top1=m["mean_top1"])


def check_ident(view, readout, ident, k, cn, seed=None):
    ref = ref_counts(view, readout, ident, k, seed)
    if "mean_top1" in ref:
        got = cn["cnt_total"].sum() / (L.VIEW[view]["n"] * 25)
        assert abs(got - ref["mean_top1"]) < 1e-12, (view, readout, ident, k, got, ref)
        return
    for c, v in ref.items():
        assert np.array_equal(cn[c], v), f"identical-input {view} {readout} {ident} k={k} r={seed} {c} differs"


# ------------------------------------------------------------------ linear
def lin_fit(q, splits):
    ctx = CTX
    tr_b, y_b, _, _ = build_traces(q, "build", ctx.qe, ctx.cubes["build"])
    with threadpool_limits(limits=N_THR):
        clf = LogisticRegression(C=C, max_iter=MAX_ITER, tol=TOL, solver="lbfgs").fit(POOL_FN(tr_b), y_b)
        n_it = int(clf.n_iter_.max())
        assert n_it < MAX_ITER, f"max_iter hit {q}"
        out = {}
        for s in splits:
            tr, y, c, _ = build_traces(q, s, ctx.qe, ctx.cubes[s])
            out[s] = (clf.decision_function(POOL_FN(tr)).astype(np.float32), y, c)
    return out, n_it


def _val_worker(job):
    t = time.time()
    out, n_it = lin_fit(job["q"], ("val",))
    lg, y, _ = out["val"]
    return job["key"], float((lg.argmax(1) == y).mean()), n_it, round(time.time() - t, 1)


def _lin_worker(job):
    t = time.time()
    out, n_it = lin_fit(job["q"], ("test",))
    lg, y, c = out["test"]
    cn = counts(CTX, lg, y, c)
    if job["ident"]:
        check_ident(CTX.view, "lin", job["ident"], job["k"], cn)
    return job["key"], cn, lg, dict(n_iter=n_it, wall_s=round(time.time() - t, 1), q=job["q"], ident=job["ident"])


def pool_map(fn, jobs):
    with mp.get_context("fork").Pool(N_WORK) as pool:
        for r in pool.imap_unordered(fn, jobs):
            yield r


def cmd_val():
    global CTX
    t0 = time.time()
    ch = json.load(open(f"{L.OUT}/chains.json"))
    assert ch["frozen_before_training"]
    res = dict(schema="d031-variant-choice-v1", readout="linear (D017 concat), fit S_build, top-1 S_val, k = 8",
               s_test_reads=0, chains_sha256=L.sha(f"{L.OUT}/chains.json"), views={})
    for view in ("v2", "v1"):
        CTX = Ctx(view, ("build", "val"))
        A = ch["views"][view]["arms"]
        jobs = []
        for name, a in A.items():
            if not (name.startswith("HELM-SIG") or name.startswith("HELM-RANK") or name.startswith("MMR")):
                continue
            if a["kind"] == "det":
                jobs.append(dict(key=name, q=a["train"]["8"]["q"]))
            else:
                jobs += [dict(key=f"{name}|d{d}", q=tr["8"]["q"]) for d, tr in enumerate(a["train"]) if "8" in tr]
        vals = {}
        for k, acc, n_it, w in pool_map(_val_worker, jobs):
            vals[k] = dict(val_top1=acc, n_iter=n_it, wall_s=w)
            print(f"[S1 {view}] {k}: S_val top-1 {acc:.4f} ({n_it} it, {w}s)", flush=True)
        V = {}
        for e in L.E_VARIANTS:
            V[f"HELM-RANK|{e}"] = vals[f"HELM-RANK|{e}"]["val_top1"]
            dr = [vals[f"HELM-SIG|{e}|d{d}"]["val_top1"] for d in range(L.N_DRAWS) if f"HELM-SIG|{e}|d{d}" in vals]
            V[f"HELM-SIG|{e}"] = float(np.mean(dr)) if len(dr) == L.N_DRAWS else None   # None = ineligible (Call 3)
        for sim in L.SIMS:
            for lam in L.LAMBDAS:
                V[f"MMR|{sim}|{lam}"] = vals[f"MMR|{sim}|{lam}"]["val_top1"]

        def pick(order):
            best, bv = None, -1.0
            for n in order:
                if V[n] is not None and V[n] > bv:
                    best, bv = n, V[n]
            return best
        helm_order = ["HELM-SIG|raw", "HELM-RANK|raw", "HELM-SIG|sf", "HELM-RANK|sf"]
        mmr_order = [f"MMR|{s}|{l}" for s in L.SIMS for l in L.LAMBDAS]
        res["views"][view] = dict(
            values=V, per_job=vals, helm_tie_order=helm_order, mmr_tie_order=mmr_order,
            HELM_star=pick(helm_order), MMR_star=pick(mmr_order),
            HELM_SIG_star=pick(["HELM-SIG|raw", "HELM-SIG|sf"]), HELM_RANK_star=pick(["HELM-RANK|raw", "HELM-RANK|sf"]),
            ineligible=[n for n, v in V.items() if v is None])
        print(f"[S1 {view}] HELM* = {res['views'][view]['HELM_star']}, MMR* = {res['views'][view]['MMR_star']}",
              flush=True)
    res["wall_min"] = round((time.time() - t0) / 60, 1)
    res["code_commit"] = L.git_head()
    L.jdump(res, f"{L.OUT}/variant_choice.json")
    print(f"[S1] wrote variant_choice.json ({res['wall_min']} min)", flush=True)


def assert_choice_committed():
    ch = json.load(open(f"{L.OUT}/variant_choice.json"))
    assert ch["s_test_reads"] == 0
    r = os.popen(f"git log --oneline -- {L.OUT}/variant_choice.json").read().strip()
    assert r, "variant_choice.json is not committed -- no S_test computation before it is"
    return ch


def cmd_lin(view):
    global CTX
    t0 = time.time()
    assert_choice_committed()
    ch = json.load(open(f"{L.OUT}/chains.json"))
    CTX = Ctx(view, ("build", "test"))
    jobs = [dict(j, key=L.key(j)) for j in L.jobs(ch, view, "lin")]
    if view == "v1":                         # v1 references (P1 Call 7): JG re-derived, PAPER8 fitted
        for name, chain in (("JG", L.jg_chain("v1")), ("PAPER8", L.PAPER8)):
            for k in range(1, L.K_MAX + 1):
                j = dict(name=name, k=k, draw=None, q=chain[:k], ident=name)
                jobs.append(dict(j, key=L.key(j)))
    print(f"[S2 lin {view}] {len(jobs)} fits, {N_WORK} workers x {N_THR} threads", flush=True)
    store, logits, meta = {}, {}, {}
    for i, (k, cn, lg, m) in enumerate(pool_map(_lin_worker, jobs)):
        for c, v in cn.items():
            store[f"{k}|{c}"] = v
        logits[k] = lg
        meta[k] = m
        if (i + 1) % 20 == 0:
            print(f"[S2 lin {view}] {i + 1}/{len(jobs)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    if view == "v1":                          # JG k = 8 all-pair vs D030 Call 1 (same D020 code path)
        z = np.load("./results/D030/call1_linear_v1.npz")
        assert np.array_equal(store["JG|8|cnt_pair_all"].astype(np.float64), z["cnt_pair666"]), "v1 JG lin k8 vs D030"
    np.savez_compressed(f"{L.OUT}/counts_{view}_lin.npz", **store)
    np.savez_compressed(f"{L.OUT}/logits_{view}_lin.npz", **logits)
    L.jdump(dict(schema="d031-runs-v1", view=view, readout="linear concat", fits=meta,
                 identical_input_asserted=sorted(k for k, m in meta.items() if m["ident"]),
                 s_test="written to counts/logits only; not summarised in S2",
                 wall_min=round((time.time() - t0) / 60, 1), node=os.uname().nodename),
            f"{L.OUT}/runs_{view}_lin.json")
    print(f"[S2 lin {view}] done, {len(jobs)} fits, wall {(time.time() - t0) / 60:.1f} min", flush=True)


def cmd_att(view, shard, nshard):
    from d009_train import run_one
    from d023_train import reload_logits
    t0 = time.time()
    assert_choice_committed()
    ch = json.load(open(f"{L.OUT}/chains.json"))
    ctx = Ctx(view, ("build", "val", "test"))
    jobs = L.jobs(ch, view, "att")[shard::nshard]
    print(f"[S2 att {view} {shard}/{nshard}] {len(jobs)} sets x {N_SEEDS} seeds", flush=True)
    ckdir = f"{L.OUT}/models/{view}"
    os.makedirs(ckdir, exist_ok=True)
    store, logits, runs = {}, {}, []
    for i, j in enumerate(jobs):
        q, k = j["q"], j["k"]
        tr_b, y_b, _, _ = build_traces(q, "build", ctx.qe, ctx.cubes["build"])
        tr_v, y_v, _, _ = build_traces(q, "val", ctx.qe, ctx.cubes["val"])
        tr_t, y_t, c_t, _ = build_traces(q, "test", ctx.qe, ctx.cubes["test"])
        hp, conf = hparams_from_shipped(k, ctx.n)
        conf = dict(conf); conf["inference_model"] = hp
        h = hp_check(view, hp)
        vals = []
        for r in range(N_SEEDS):
            kk = L.key(j, r)
            path = f"{ckdir}/{kk.replace('|', '_')}.ckpt"
            o, st, buf = run_one(tr_b, y_b, tr_v, y_v, tr_t, y_t, c_t, hp, conf, r, ctx.hard, ctx.n, path)
            assert not o["error"], o["error"] + buf.getvalue()[-800:]
            lg = reload_logits(path, tr_t, ctx.n, k)
            assert abs(float((lg.argmax(1) == y_t).mean()) - o["mean_top1"]) < 1e-9, "reload mismatch"
            cn = counts(ctx, lg, y_t, c_t)
            for c in ("cnt_total", "cnt_model", "cnt_pair"):
                assert np.array_equal(cn[c], st[c]), (kk, c)
            if j["ident"]:
                check_ident(view, "att", j["ident"], k, cn, r)
            for c, v in cn.items():
                store[f"{kk}|{c}"] = v
            logits[kk] = lg.astype(np.float32)
            vals.append(o["val_acc"])
            runs.append(dict(key=kk, name=j["name"], k=k, draw=j["draw"], seed=r, queries=q, ident=j["ident"],
                             hp_hash=h, **{a: b for a, b in o.items() if a in ("wall_s", "best_ckpt", "train_acc",
                                                                                "val_acc", "error")}))
        print(f"[S2 att {view}] {i + 1}/{len(jobs)} {L.key(j)}: S_val top-1 mean {np.mean(vals):.4f}"
              f"{' (identical input: reproduced exactly)' if j['ident'] else ''} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    tag = f"{view}_att_{shard}of{nshard}"
    np.savez_compressed(f"{L.OUT}/counts_{tag}.npz", **store)
    np.savez_compressed(f"{L.OUT}/logits_{tag}.npz", **logits)
    L.jdump(dict(schema="d031-runs-v1", view=view, readout="attention (D009)", shard=[shard, nshard], runs=runs,
                 s_test="written to counts/logits only; not summarised in S2",
                 wall_min=round((time.time() - t0) / 60, 1), node=os.uname().nodename), f"{L.OUT}/runs_{tag}.json")
    print(f"[S2 att {view} {shard}/{nshard}] done, wall {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "val":
        cmd_val()
    elif cmd == "lin":
        cmd_lin(sys.argv[2])
    elif cmd == "att":
        cmd_att(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
