"""D032 / S1 + S2 — readouts on the frozen chains (results/D032/chains.json). D031 training code imported
verbatim (d031_train: Ctx, lin_fit, counts, pool_map, check_ident, hp_check; D009 run_one), never edited.

  val <VIEW>          S1: linear (D017 concat) fit S_build, top-1 S_val at k = 8 for the 18 MMR variants of
                      MMR-PP[MEAN], MMR-TXT[CVaR], MMR-TXT[MEAN]; plus (diagnostic, Call 2 (a)) D031's 6
                      MMR-PP[CVaR] chains re-scored under this process's threads. -> results/D032/val_<VIEW>.json
  choose              S1: stars from val_v2.json + val_v1.json -> variant_choice.json (s_test_reads 0)
  lin <VIEW> <i> <n>  S2: shard i of n of the view's linear fits (19 new chains x k 1-8) + the TOP-K[CVaR] k = 8
                      thread-match refit (asserted equal to D031's stored counts; the job stops otherwise)
  att <VIEW> <i> <n>  S2: shard i of n of the attention jobs (starred new arms k 1-8, non-starred k = 8; seeds 0-4)

Threads are set by the caller (D031_THREADS / D031_WORKERS / OMP_NUM_THREADS): v2 8, v1 16 (Call 2 (a)).
S_test discipline: counts and logits are written, never printed or summarised (S3 is the first reader).
Identical-input cells (S0 index: same k-set as a D031 deterministic chain, JG or PAPER8) must reproduce the
stored counts exactly (all count fields vs D031; JG / PAPER8 also vs D029 / D020 / D017 / D010 / D009).
Usage:  PYTHONPATH=.:experiments python experiments/d032_train.py val|choose|lin|att ...
"""
import os
import sys
import glob
import json
import time

import numpy as np

import d031_lib as L31
import d031_train as T
import d032_lib as L
from d009_lib import build_traces, hparams_from_shipped

FIELDS = ("cnt_total", "cnt_model", "cnt_pair", "cnt_pair_all", "dirH")


def chains():
    ch = json.load(open(f"{L.OUT}/chains.json"))
    assert ch["frozen_before_training"]
    assert os.popen(f"git log --oneline -- {L.OUT}/chains.json").read().strip(), "chains.json not committed"
    return ch


def assert_choice_committed():
    vc = json.load(open(f"{L.OUT}/variant_choice.json"))
    assert vc["s_test_reads"] == 0
    assert os.popen(f"git log --oneline -- {L.OUT}/variant_choice.json").read().strip(), \
        "variant_choice.json is not committed -- no S_test computation before it is"
    return vc


def threads_tag():
    return dict(threads=T.N_THR, workers=T.N_WORK, omp=os.environ.get("OMP_NUM_THREADS"))


# ------------------------------------------------------------------ S1
def _val_worker(job):
    t = time.time()
    out, n_it = T.lin_fit(job["q"], ("val",))
    lg, y, _ = out["val"]
    return job["key"], float((lg.argmax(1) == y).mean()), n_it, round(time.time() - t, 1)


def cmd_val(view):
    t0 = time.time()
    ch = chains()
    A = ch["views"][view]["arms"]
    c31 = json.load(open(f"{L31.OUT}/chains.json"))["views"][view]["arms"]
    T.CTX = T.Ctx(view, ("build", "val"))
    jobs = [dict(key=n, q=A[n]["train"]["8"]["q"]) for r, c in L.CELLS for n in L.variants(r, c)]
    diag = [dict(key=f"D031:{L.d031_name(n)}", q=c31[L.d031_name(n)]["train"]["8"]["q"]) for n in L.variants("MMR-PP", "CVaR")]
    for d, n in zip(diag, L.variants("MMR-PP", "CVaR")):
        assert d["q"] == A[n]["train"]["8"]["q"]
    print(f"[S1 {view}] {len(jobs)} + {len(diag)} diagnostic fits, {threads_tag()}", flush=True)
    vals = {}
    for k, acc, n_it, w in T.pool_map(_val_worker, jobs + diag):
        vals[k] = dict(val_top1=acc, n_iter=n_it, wall_s=w)
        print(f"[S1 {view}] {k}: S_val top-1 {acc:.4f} ({n_it} it, {w}s)", flush=True)
    L31.jdump(dict(view=view, per_job=vals, **threads_tag(), chains_sha256=L31.sha(f"{L.OUT}/chains.json"),
                   wall_min=round((time.time() - t0) / 60, 1), node=os.uname().nodename), f"{L.OUT}/val_{view}.json")


def pick(order, V):
    best, bv = None, -1.0
    for n in order:                                      # strict >: first in the D031 tie order wins
        if V[n] > bv:
            best, bv = n, V[n]
    return best


def cmd_choose():
    ch = chains()
    v31 = json.load(open(f"{L31.OUT}/variant_choice.json"))
    res = dict(schema="d032-variant-choice-v1", readout="linear (D017 concat), fit S_build, top-1 S_val, k = 8",
               rule="highest S_val E_all; ties: cos (uncentred) first, then lower lambda (D031 order)",
               s_test_reads=0, chains_sha256=L31.sha(f"{L.OUT}/chains.json"), views={})
    for view in ("v2", "v1"):
        vj = json.load(open(f"{L.OUT}/val_{view}.json"))
        V = {k: v["val_top1"] for k, v in vj["per_job"].items()}
        stars = {"TOP-K[CVaR]": "TOP-K[CVaR]", "TOP-K[MEAN]": "TOP-K[MEAN]"}
        d31 = v31["views"][view]["MMR_star"]
        stars["MMR-PP[CVaR]"] = "MMR-PP[CVaR]|" + d31.split("|", 1)[1]                    # D031 choice, verbatim
        for r, c in L.CELLS:
            stars[f"{r}[{c}]"] = pick(L.variants(r, c), V)
        order31 = L.variants("MMR-PP", "CVaR")
        Vd = {n: V[f"D031:{L.d031_name(n)}"] for n in order31}
        rescore = pick(order31, Vd)
        D31vals = {n: v31["views"][view]["values"][L.d031_name(n)] for n in order31}
        res["views"][view] = dict(
            stars=stars, values={n: V[n] for r, c in L.CELLS for n in L.variants(r, c)},
            threads=dict(threads=vj["threads"], workers=vj["workers"], omp=vj["omp"]),
            d031_mmr_pp_cvar_diagnostic=dict(
                note="Call 2 (a): D031's 6 MMR-PP[CVaR] chains re-scored on S_val under D032 S1 threads; "
                     "the star is NOT changed (fixed before D031 read S_test)",
                d031_values=D31vals, rescored_values=Vd,
                equal_to_d031={n: Vd[n] == D31vals[n] for n in order31},
                rescored_max=rescore, d031_star=stars["MMR-PP[CVaR]"],
                still_max=rescore == stars["MMR-PP[CVaR]"]))
        print(f"[S1 {view}] stars {stars}; D031 MMR-PP[CVaR] rescored max {rescore} "
              f"(still max: {rescore == stars['MMR-PP[CVaR]']})", flush=True)
    res["code_commit"] = L31.git_head()
    L31.jdump(res, f"{L.OUT}/variant_choice.json")
    print("[S1] wrote variant_choice.json", flush=True)


# ------------------------------------------------------------------ S2 references
class Ref31:
    """D031 stored counts of one view (both readouts), for the identical-input asserts."""

    def __init__(self, view):
        self.z = {}
        for ro in ("lin", "att"):
            d = {}
            for f in sorted(glob.glob(f"{L31.OUT}/counts_{view}_{ro}_*of*.npz")):
                d.update(dict(np.load(f)))
            self.z[ro] = d

    def check(self, ro, src, k, cn, seed=None):
        base = f"{src}|{k}" + (f"|r{seed}" if seed is not None else "")
        n = 0
        for c in FIELDS:
            if c in cn:
                ref = self.z[ro][f"{base}|{c}"]
                assert np.array_equal(cn[c], ref), f"identical-input {ro} {base} {c} differs from D031"
                n += 1
        return n


def ident_list(t):
    """Sources a trained cell must reproduce: D031 deterministic chains + JG / PAPER8."""
    src = [s.split(":", 1)[1] for s in t["d031_same_set"]]
    if t["ident"] and t["ident"] not in src:
        src.append(t["ident"])
    return src


def lin_jobs(ch, view):
    A = ch["views"][view]["arms"]
    refit = [dict(key="REFIT|TOP-K[CVaR]|8", name="TOP-K[CVaR]", k=8, q=A["TOP-K[CVaR]"]["train"]["8"]["q"],
                  src=["TOP-K"], ident=None)]
    jobs = []
    for n in L.NEW:
        for k in range(1, L31.K_MAX + 1):
            t = A[n]["train"][str(k)]
            jobs.append(dict(key=f"{n}|{k}", name=n, k=k, q=t["q"], src=ident_list(t), ident=t["ident"]))
    return refit + [j for j in jobs if j["src"]] + [j for j in jobs if not j["src"]]   # asserts first


def att_jobs(ch, vc, view):
    A = ch["views"][view]["arms"]
    stars = set(vc["views"][view]["stars"].values())
    out = []
    for n in L.NEW:
        for k in (range(1, L31.K_MAX + 1) if n in stars else (8,)):
            t = A[n]["train"][str(k)]
            out.append(dict(name=n, k=k, q=t["q"], src=ident_list(t), ident=t["ident"]))
    return [j for j in out if j["src"]] + [j for j in out if not j["src"]]


REF = None


def _check(ro, j, cn, seed=None):
    """Assert one trained cell against every source in its S0 index; returns what was asserted."""
    done = []
    for s in j["src"]:
        n0 = len(done)
        if s in ("JG", "PAPER8"):
            T.check_ident(REF.view, ro, s, j["k"], cn, seed)                   # D029 / D020 / D017 / D010 / D009
            done.append(f"{s}:stored")
        if f"{s}|{j['k']}" + (f"|r{seed}" if seed is not None else "") + "|cnt_total" in REF.z[ro]:
            REF.check(ro, s, j["k"], cn, seed)                                  # D031 counts, every field
            done.append(f"D031:{s}")
        assert len(done) > n0, ("no reference found", ro, j["name"], j["k"], s)
    return done


def _lin_worker(job):
    t = time.time()
    out, n_it = T.lin_fit(job["q"], ("test",))
    lg, y, c = out["test"]
    cn = T.counts(T.CTX, lg, y, c)
    done = _check("lin", job, cn) if job["src"] else []
    return job["key"], cn, lg, dict(n_iter=n_it, wall_s=round(time.time() - t, 1), q=job["q"], asserted=done)


def cmd_lin(view, shard, nshard):
    global REF
    t0 = time.time()
    ch = chains()
    assert_choice_committed()
    REF = Ref31(view)
    REF.view = view
    T.CTX = T.Ctx(view, ("build", "test"))
    jobs = lin_jobs(ch, view)[shard::nshard]
    tag = f"{view}_lin_{shard}of{nshard}"
    print(f"[S2 lin {view} {shard}/{nshard}] {len(jobs)} fits ({sum(bool(j['src']) for j in jobs)} identical-input "
          f"first), {threads_tag()}", flush=True)
    store, logits, meta = {}, {}, {}
    for i, (k, cn, lg, m) in enumerate(T.pool_map(_lin_worker, jobs)):
        for c, v in cn.items():
            store[f"{k}|{c}"] = v
        logits[k] = lg
        meta[k] = m
        if m["asserted"]:
            print(f"[S2 lin {view}] {k}: identical input reproduced exactly ({', '.join(m['asserted'])})", flush=True)
        if (i + 1) % 20 == 0:
            print(f"[S2 lin {view}] {i + 1}/{len(jobs)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    np.savez_compressed(f"{L.OUT}/counts_{tag}.npz", **store)
    np.savez_compressed(f"{L.OUT}/logits_{tag}.npz", **logits)
    L31.jdump(dict(schema="d032-runs-v1", view=view, readout="linear concat", shard=[shard, nshard], fits=meta,
                   **threads_tag(), identical_input_asserted=sorted(k for k, m in meta.items() if m["asserted"]),
                   s_test="written to counts/logits only; not summarised in S2",
                   wall_min=round((time.time() - t0) / 60, 1), node=os.uname().nodename), f"{L.OUT}/runs_{tag}.json")
    print(f"[S2 lin {view} {shard}/{nshard}] done, {len(jobs)} fits, wall {(time.time() - t0) / 60:.1f} min", flush=True)


def cmd_att(view, shard, nshard):
    global REF
    from d009_train import run_one
    from d023_train import reload_logits
    t0 = time.time()
    ch = chains()
    vc = assert_choice_committed()
    REF = Ref31(view)
    REF.view = view
    ctx = T.Ctx(view, ("build", "val", "test"))
    T.CTX = ctx
    jobs = att_jobs(ch, vc, view)[shard::nshard]
    print(f"[S2 att {view} {shard}/{nshard}] {len(jobs)} sets x {T.N_SEEDS} seeds", flush=True)
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
        h = T.hp_check(view, hp)
        vals, done = [], []
        for r in range(T.N_SEEDS):
            kk = f"{j['name']}|{k}|r{r}"
            path = f"{ckdir}/{kk.replace('|', '_')}.ckpt"
            o, st, buf = run_one(tr_b, y_b, tr_v, y_v, tr_t, y_t, c_t, hp, conf, r, ctx.hard, ctx.n, path)
            assert not o["error"], o["error"] + buf.getvalue()[-800:]
            lg = reload_logits(path, tr_t, ctx.n, k)
            assert abs(float((lg.argmax(1) == y_t).mean()) - o["mean_top1"]) < 1e-9, "reload mismatch"
            cn = T.counts(ctx, lg, y_t, c_t)
            for c in ("cnt_total", "cnt_model", "cnt_pair"):
                assert np.array_equal(cn[c], st[c]), (kk, c)
            if j["src"]:
                done = _check("att", j, cn, r)
            for c, v in cn.items():
                store[f"{kk}|{c}"] = v
            logits[kk] = lg.astype(np.float32)
            vals.append(o["val_acc"])
            runs.append(dict(key=kk, name=j["name"], k=k, seed=r, queries=q, asserted=done, hp_hash=h,
                             **{a: b for a, b in o.items() if a in ("wall_s", "best_ckpt", "train_acc", "val_acc", "error")}))
        print(f"[S2 att {view}] {i + 1}/{len(jobs)} {j['name']}|{k}: S_val top-1 mean {np.mean(vals):.4f}"
              f"{' (identical input reproduced exactly: ' + ', '.join(done) + ')' if done else ''} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    tag = f"{view}_att_{shard}of{nshard}"
    np.savez_compressed(f"{L.OUT}/counts_{tag}.npz", **store)
    np.savez_compressed(f"{L.OUT}/logits_{tag}.npz", **logits)
    L31.jdump(dict(schema="d032-runs-v1", view=view, readout="attention (D009)", shard=[shard, nshard], runs=runs,
                   s_test="written to counts/logits only; not summarised in S2",
                   wall_min=round((time.time() - t0) / 60, 1), node=os.uname().nodename), f"{L.OUT}/runs_{tag}.json")
    print(f"[S2 att {view} {shard}/{nshard}] done, wall {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "val":
        cmd_val(sys.argv[2])
    elif cmd == "choose":
        cmd_choose()
    elif cmd == "lin":
        cmd_lin(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "att":
        cmd_att(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
