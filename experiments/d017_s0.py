"""D017 / S0 — pooling, determinism, and three facts that decide Call 1.

D017 flags pooling as where a wrong default quietly biases everything. It proposes
mean-pooling and asks that concatenation be considered and rejected-or-accepted
with a reason. The reasons turn out to be measurable rather than arguable, so this
measures them:

  A. Is E(query) usable by a linear model at all? Within one condition every trace
     answers the SAME k queries, so E(q_i) is identical across samples -- a linear
     model absorbs it into the bias. If so the 2048-d trace is effectively 1024-d
     here, under ANY pooling, and that is worth knowing before interpreting
     anything.
  B. Does concatenation reintroduce the slot-order dependency D017 worries about?
     For a LINEAR model, f(x) = sum_i W_i v_i + b, so permuting slots permutes the
     learned blocks and leaves the fitted function unchanged. Prediction: accuracy
     is exactly invariant. Measured, not asserted.
  C. Mean-pool vs concatenation. Mean-pooling is the strict special case W_i = W/k
     for all i -- it forces every slot to share one weight matrix, i.e. it cannot
     tell which query produced which response. Concatenation can. The gap between
     them therefore measures how much of the selection benefit needs per-query
     resolution, which is exactly D017's theoretical question in miniature.

  D. Determinism: is the fit stochastic at all? If not, "5 seeds" has nothing to
     measure and the config bootstrap is the only real uncertainty source.

Usage:  PYTHONPATH=.:experiments python experiments/d017_s0.py
"""
import os
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from d009_lib import load_query_embeddings, build_traces
from d007_lib import load_corpus

OUT = "./results/D017"
K = 8


def fit_eval(Xtr, ytr, Xte, yte, C=1.0, seed=0):
    # sklearn >= 1.9 removed `multi_class`; lbfgs on a >2-class problem is
    # multinomial by default, which is what this needs.
    clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs",
                             random_state=seed)
    clf.fit(Xtr, ytr)
    return float((clf.predict(Xte) == yte).mean()), clf


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    d9 = json.load(open("./results/D009/runs.json"))
    chain = next(r for r in d9["runs"] if r["k"] == K and r["condition"] == "paper8"
                 and r["run"] == 0)["queries"]
    qe = load_query_embeddings()
    cubes = {p: load_corpus(pool=p) for p in ("build", "val", "test")}
    tr, y_tr, _, models = build_traces(chain, "build", qe, cubes["build"])
    te, y_te, _, _ = build_traces(chain, "test", qe, cubes["test"])
    print(f"traces {tr.shape} -> {te.shape}, {len(models)} models", flush=True)
    res = {}

    # ---- A. is E(query) constant across traces?
    qpart = tr[:, :, :1024]
    const = bool(np.allclose(qpart, qpart[0:1], atol=0))
    res["A_query_half_constant"] = dict(
        constant_across_traces=const,
        note="within one condition every trace answers the SAME k queries, so a "
             "linear model absorbs E(query) into the bias. The 2048-d trace is "
             "effectively 1024-d for ANY linear pooling. The attention network "
             "CAN use it (to modulate how each slot is read); concatenation gets "
             "that same capability via separate weight blocks, mean-pooling does "
             "not.")
    print(f"[A] E(query) identical across all traces: {const}", flush=True)

    cat_tr = tr.reshape(len(tr), -1); cat_te = te.reshape(len(te), -1)
    rsp_tr = tr[:, :, 1024:].reshape(len(tr), -1)
    rsp_te = te[:, :, 1024:].reshape(len(te), -1)
    mp_tr = tr.mean(axis=1); mp_te = te.mean(axis=1)

    # ---- D. determinism
    t = time.time(); a1, c1 = fit_eval(cat_tr, y_tr, cat_te, y_te, seed=0)
    t_fit = time.time() - t
    _, c2 = fit_eval(cat_tr, y_tr, cat_te, y_te, seed=7)
    ident = bool(np.array_equal(c1.coef_, c2.coef_))
    res["D_determinism"] = dict(
        fit_wall_s=round(t_fit, 2), acc=round(a1, 4),
        coef_bit_identical_across_seeds=ident,
        max_coef_diff=float(np.abs(c1.coef_ - c2.coef_).max()),
        implication=("deterministic -- `5 seeds` has nothing to measure; the "
                     "config-level bootstrap is the only real uncertainty source"
                     if ident else "stochastic -- keep 5 seeds"))
    print(f"[D] fit {t_fit:.2f}s, acc {a1:.4f}; two seeds bit-identical: {ident}",
          flush=True)

    # ---- B. slot-order invariance of concatenation
    rng = np.random.default_rng(0)
    perm = rng.permutation(K)
    a_perm, _ = fit_eval(tr[:, perm].reshape(len(tr), -1), y_tr,
                         te[:, perm].reshape(len(te), -1), y_te)
    res["B_concat_order_invariance"] = dict(
        chain_order_acc=round(a1, 6), permuted_order_acc=round(a_perm, 6),
        identical=bool(abs(a1 - a_perm) < 1e-12), permutation=perm.tolist(),
        note="for a linear model f(x)=sum_i W_i v_i + b, permuting slots permutes "
             "the learned blocks; the fitted function is unchanged. D017's concern "
             "that concatenation 'would reintroduce a slot-order dependency' holds "
             "for a FIXED-weight model, not a fitted one.")
    print(f"[B] concat under chain order {a1:.6f} vs permuted {a_perm:.6f} -> "
          f"identical: {res['B_concat_order_invariance']['identical']}", flush=True)

    # ---- A': does dropping the (constant) query half change anything?
    a_rsp, _ = fit_eval(rsp_tr, y_tr, rsp_te, y_te)
    res["A_response_only_equals_full"] = dict(
        full_2048k=round(a1, 6), response_only_1024k=round(a_rsp, 6),
        identical=bool(abs(a1 - a_rsp) < 1e-9))
    print(f"[A'] full 2048k {a1:.6f} vs response-only 1024k {a_rsp:.6f}",
          flush=True)

    # ---- C. mean-pool vs concatenation
    t = time.time(); a_mp, _ = fit_eval(mp_tr, y_tr, mp_te, y_te)
    res["C_pooling"] = dict(
        concat_acc=round(a1, 4), concat_features=int(cat_tr.shape[1]),
        concat_fit_s=round(t_fit, 2),
        meanpool_acc=round(a_mp, 4), meanpool_features=int(mp_tr.shape[1]),
        meanpool_fit_s=round(time.time() - t, 2),
        gap=round(a1 - a_mp, 4),
        note="mean-pooling is the strict special case W_i = W/k for all i: one "
             "shared weight matrix across slots, so it cannot tell which query "
             "produced which response. Concatenation can. The gap measures how "
             "much of the signal needs per-query resolution.")
    print(f"[C] concat {a1:.4f} ({cat_tr.shape[1]} feats) vs mean-pool {a_mp:.4f} "
          f"({mp_tr.shape[1]} feats) -> gap {a1-a_mp:+.4f}", flush=True)

    # attention network's own number for the same condition/chain
    att = next(r for r in d9["runs"] if r["k"] == K and r["condition"] == "paper8"
               and r["run"] == 0)["mean_top1"]
    res["reference_attention_network"] = dict(paper8_k8_seed0=att)
    print(f"[ref] attention network, same chain: {att:.4f}", flush=True)

    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(f"{OUT}/pooling_and_determinism_check.json", "w"), indent=1)
    print(f"\nwall {res['wall_s']}s; written: {OUT}/"
          f"pooling_and_determinism_check.json", flush=True)


if __name__ == "__main__":
    main()
