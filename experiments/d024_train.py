"""D024 / S2 — train and evaluate the frozen CENTROID / SINGLE (/ CENTROID-raw) chains.

Attention: d023_train.train_attention (D009 code path, hparams hash == D009's,
S_val early stopping, S_test once, seeds 0-4, all checkpoints saved, reload +
mean_top1 assert, cnt_total/cnt_model/cnt_pair/cnt_pair666).
Linear: d023_train.fit_linear (D020's fit, C=1.0 concat / 3.0 mean-pool).
Order: the frozen canonical training lists in selection.json (Review Call 2).
Stop conditions (Review Call 2):
  - identical-to-CLOUD k: counts EXACTLY equal to D010 (`joint_energy`) and
    D020 (`JointGreedy`) -- asserted inside the D023 helpers;
  - identical-input k between two new arms (CENTROID<->SINGLE, and
    CENTROID-raw<->CENTROID if trained): bit-identical counts, asserted here.
Usage:  PYTHONPATH=.:experiments python experiments/d024_train.py
"""
import os
import json
import time

import numpy as np

from d023_train import Ctx, train_attention, fit_linear, CNT
from d024_lib import OUT

PAIRS_NEW = {"P2_CENTROID_minus_SINGLE": ("CENTROID", "SINGLE"),
             "CENTROID-raw_minus_CENTROID": ("CENTROID-raw", "CENTROID")}


def main():
    t0 = time.time()
    sel = json.load(open(f"{OUT}/selection.json"))
    assert sel["frozen_before_training"]
    arms = sel["trained_arms"]
    ctx = Ctx()
    assert ctx.models == json.load(open(f"{OUT}/s0_pilot.json"))["models"]
    ref_att = np.load("./results/D010/run_counts.npz")
    ref_lin = np.load("./results/D020/linear_counts.npz")
    os.makedirs(f"{OUT}/models", exist_ok=True)
    counts, runs, lin_counts, lin_pts = {}, [], {}, {}
    for arm in arms:
        orders = {int(k): v for k, v in sel["train_order"][arm].items()}
        ident = sel["identical_to_cloud_k"][arm]
        c, r = train_attention(ctx, arm, orders, f"{OUT}/models", ident, ref_att, "joint_energy")
        counts.update(c); runs += r
        lc, lp = fit_linear(ctx, arm, orders, ident, ref_lin, "JointGreedy")
        lin_counts.update(lc); lin_pts.update(lp)
        print(f"[S2] {arm} done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    # bit-identity between new arms on identical-input k (stop condition)
    checked = {}
    for name, (a, b) in PAIRS_NEW.items():
        if a not in arms or b not in arms:
            continue
        ks = sel["identical_input_k"][name]
        for k in ks:
            for r in range(5):
                for c in CNT + ("cnt_pair666",):
                    assert np.array_equal(counts[f"{a}|{k}|{r}|{c}"], counts[f"{b}|{k}|{r}|{c}"]), \
                        f"STOP: {name} identical-input k={k} r={r} {c} differs"
            for p in ("concat", "meanpool"):
                for c in CNT:
                    assert np.array_equal(lin_counts[f"{a}|{p}|{k}|{c}"], lin_counts[f"{b}|{p}|{k}|{c}"]), \
                        f"STOP: {name} linear identical-input {p} k={k} {c} differs"
        checked[name] = ks
    print(f"[S2] new-arm bit-identity checks passed on {checked}; identical-to-CLOUD k "
          f"{ {a: sel['identical_to_cloud_k'][a] for a in arms} } reproduced D010/D020 exactly",
          flush=True)

    np.savez_compressed(f"{OUT}/run_counts.npz", **counts)
    np.savez_compressed(f"{OUT}/linear_counts.npz", **lin_counts)
    json.dump(dict(schema="d024-runs-v1", hparams_hash=ctx.hash_ref, arms=arms, runs=runs,
                   linear_points=lin_pts, new_arm_identity_checked=checked,
                   identical_to_cloud_checked={a: sel["identical_to_cloud_k"][a] for a in arms},
                   wall_min=round((time.time() - t0) / 60, 1)),
              open(f"{OUT}/runs.json", "w"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"wall {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
