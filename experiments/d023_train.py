"""D023 / S2 — train and evaluate frozen chains (also imported by D022 Part B).

Attention: D009's code path exactly as D018/D019 ran it (d009_train.run_one,
hparams_from_shipped, hparams hash == D009's, S_val early stopping, S_test once),
seeds 0-4, ALL checkpoints saved. Each checkpoint is reloaded, its test logits
recomputed, mean_top1 asserted equal to the run's, and counts stored:
cnt_total (25), cnt_model (37x25), cnt_pair (65x25, structural), cnt_pair666
(666x25, D016's two-logit loop, D007 pair order).
Linear: D020's `fit` (C=1.0 concat / 3.0 mean-pool, lbfgs, tol 1e-6, max_iter
20000), counts as D020.
Identical-input k (ordered list == reference chain's prefix): counts asserted
EXACTLY equal to the reference's stored counts (D010 attention / D020 linear);
on failure the job stops (D023 Review, Call 3).
Usage:  PYTHONPATH=.:experiments python experiments/d023_train.py
"""
import os
import json
import time
import hashlib

import numpy as np
import torch

from LLMmap.inference_model_archs import InferenceModelLLMmap
from d009_lib import load_query_embeddings, build_traces, hparams_from_shipped, logits_for, logit_stats
from d009_train import run_one
from d008_lib import near_relative_pairs, pair_index
from d007_lib import load_corpus
from d020_paired_linear import fit, POOL

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_SEEDS = 5
CNT = ("cnt_total", "cnt_model", "cnt_pair")


class Ctx:
    def __init__(self):
        self.models = json.load(open("./results/D008/selection.json"))["models"]
        self.n_models = len(self.models)
        self.hard = near_relative_pairs(self.models)
        pairs = pair_index(self.models)
        self.pa = np.array([self.models.index(a) for a, _ in pairs])
        self.pb = np.array([self.models.index(b) for _, b in pairs])
        self.qe = load_query_embeddings()
        self.cubes = {p: load_corpus(pool=p) for p in ("build", "val", "test")}
        self.hash_ref = json.load(open("./results/D009/runs.json"))["hparams_hash"]


def pair666_counts(lg, y, cfg, pa, pb, ncfg=25):
    out = np.zeros((len(pa), ncfg))
    for j, (a, b) in enumerate(zip(pa, pb)):
        m = (y == a) | (y == b)
        p2 = np.where(lg[m][:, [a, b]].argmax(1) == 0, a, b)
        ok = p2 == y[m]
        out[j] = np.bincount(cfg[m][ok], minlength=ncfg)
    return out


def reload_logits(path, tr_t, n_models, k):
    hp, _ = hparams_from_shipped(k, n_models)
    net = InferenceModelLLMmap(hp)
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    net.load_state_dict({kk[len("net."):] if kk.startswith("net.") else kk: vv
                         for kk, vv in sd.items()})
    return logits_for(net, tr_t, DEV)


def train_attention(ctx, arm, orders, ckpt_dir, ident=None, ref_counts=None, ref_prefix=None):
    """orders: {k: ordered query list}. ident: list of k to assert against
    ref_counts[f'{ref_prefix}|{k}|{r}|cnt_*'] exactly."""
    counts, runs = {}, []
    for k, q in sorted(orders.items()):
        tr_b, y_b, _, _ = build_traces(q, "build", ctx.qe, ctx.cubes["build"])
        tr_v, y_v, _, _ = build_traces(q, "val", ctx.qe, ctx.cubes["val"])
        tr_t, y_t, c_t, _ = build_traces(q, "test", ctx.qe, ctx.cubes["test"])
        hp, conf = hparams_from_shipped(k, ctx.n_models)
        conf = dict(conf); conf["inference_model"] = hp
        h = hashlib.sha256(json.dumps({kk: vv for kk, vv in hp.items() if kk != "num_queries"},
                                      sort_keys=True).encode()).hexdigest()
        assert h == ctx.hash_ref
        for r in range(N_SEEDS):
            path = f"{ckpt_dir}/{arm}_k{k}_r{r}.ckpt"
            o, st, _ = run_one(tr_b, y_b, tr_v, y_v, tr_t, y_t, c_t, hp, conf, r,
                               ctx.hard, ctx.n_models, path)
            assert not o["error"], o["error"]
            lg = reload_logits(path, tr_t, ctx.n_models, k)
            acc = float((lg.argmax(1) == y_t).mean())
            assert abs(acc - o["mean_top1"]) < 1e-9, (arm, k, r, acc, o["mean_top1"])
            for c in CNT:
                counts[f"{arm}|{k}|{r}|{c}"] = st[c]
            counts[f"{arm}|{k}|{r}|cnt_pair666"] = pair666_counts(lg, y_t, c_t, ctx.pa, ctx.pb)
            if ident and k in ident:
                for c in CNT:
                    ref = ref_counts[f"{ref_prefix}|{k}|{r}|{c}"]
                    assert np.array_equal(st[c], ref), f"identical-input k={k} r={r} {c} differs"
            runs.append(dict(arm=arm, k=k, run=r, seed=r, queries=q,
                             **{kk: vv for kk, vv in o.items() if kk not in ("best_ckpt",)}))
        print(f"[S2] {arm} k={k} {q}: mean_top1 "
              f"{np.mean([x['mean_top1'] for x in runs if x['k'] == k]):.4f}"
              f"{' (identical input: reproduces reference exactly)' if ident and k in ident else ''}",
              flush=True)
    return counts, runs


def fit_linear(ctx, arm, orders, ident=None, ref_counts=None, ref_prefix=None):
    counts, pts = {}, {}
    for p, (pool, C) in POOL.items():
        for k, q in sorted(orders.items()):
            pt, st, _, n_it = fit(q, pool, C, ctx.qe, ctx.cubes, ctx.hard, ctx.n_models)
            for c in CNT:
                counts[f"{arm}|{p}|{k}|{c}"] = st[c]
                if ident and k in ident:
                    assert np.array_equal(st[c], ref_counts[f"{ref_prefix}|{p}|{k}|{c}"]), \
                        f"linear identical-input {p} k={k} {c} differs"
            pts[f"{arm}|{p}|{k}"] = {m: pt[m] for m in ("mean_top1", "worst_class",
                                                          "worst3_class", "hard_subset")}
    return counts, pts


def baseline_666(ctx, runs_meta):
    """cnt_pair666 for existing k=8 runs from D016's cached test logits."""
    _, y_t, c_t, _ = build_traces(list(range(8)), "test", ctx.qe, ctx.cubes["test"])
    out = {}
    for name, (cond, stored) in runs_meta.items():
        for r in range(N_SEEDS):
            lg = np.load(f"./results/D016/logits_{cond}_k8_r{r}.npy")
            assert abs(float((lg.argmax(1) == y_t).mean()) - stored[r]) < 1e-9, (name, r)
            out[f"{name}|8|{r}|cnt_pair666"] = pair666_counts(lg, y_t, c_t, ctx.pa, ctx.pb)
    return out


def main():
    OUT = "./results/D023"
    t0 = time.time()
    sel = json.load(open(f"{OUT}/selection.json"))
    assert sel["frozen_before_training"]
    ctx = Ctx()
    ref_att = np.load("./results/D010/run_counts.npz")
    ref_lin = np.load("./results/D020/linear_counts.npz")
    arms = {"ILP-ADD": {int(k): v for k, v in sel["train_order"]["ILP_ADD"].items()},
            "Greedy-ADD": {int(k): v for k, v in sel["train_order"]["Greedy_ADD"].items()}}
    os.makedirs(f"{OUT}/models", exist_ok=True)
    counts, runs, lin_counts, lin_pts = {}, [], {}, {}
    for arm, orders in arms.items():
        ident = sel["identical_input_k"][arm]
        c, r = train_attention(ctx, arm, orders, f"{OUT}/models", ident, ref_att, "joint_energy")
        counts.update(c); runs += r
        lc, lp = fit_linear(ctx, arm, orders, ident, ref_lin, "JointGreedy")
        lin_counts.update(lc); lin_pts.update(lp)
        print(f"[S2] {arm} linear done", flush=True)
    d9 = json.load(open("./results/D009/runs.json"))["runs"]
    d10 = json.load(open("./results/D010/metrics_by_k.json"))["runs"]
    st = lambda runs_, cond: [r["mean_top1"] for r in sorted(
        [x for x in runs_ if x["k"] == 8 and x["condition"] == cond and not x["error"]],
        key=lambda x: x["run"])]
    counts.update(baseline_666(ctx, {"JointGreedy": ("joint_energy", st(d10, "joint_energy")),
                                     "paper8": ("paper8", st(d9, "paper8"))}))
    np.savez_compressed(f"{OUT}/run_counts.npz", **counts)
    np.savez_compressed(f"{OUT}/linear_counts.npz", **lin_counts)
    json.dump(dict(schema="d023-runs-v1", hparams_hash=ctx.hash_ref, runs=runs,
                   linear_points=lin_pts, wall_min=round((time.time() - t0) / 60, 1)),
              open(f"{OUT}/runs.json", "w"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"wall {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
