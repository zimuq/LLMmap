"""D019 / S0–S3 — hard-model true 37-way top-1 recall for D018's two H1 chains.

S0  checkpoint availability. D018 kept only seed 0 per chain at k=8
    (`if r == 0 and k == 8`), so 2 of 10 exist; seeds 1-4 were never written.
S1  retrain 5 seeds x 2 chains at k=8 on D018's exact code path, keeping
    per_model (P1/Call 2, approved). Asserts: 5-seed mean_top1 AND range equal
    D018's stored values; the saved seed-0 checkpoints reload to a per_model
    bit-identical to the retrained seed 0; hparams hash equals D009's.
    Both chains are read by the ATTENTION network (P1/Call 1, approved).
S2  mean per_model restricted to the four already-fixed model sets. Lists used
    verbatim (P1/Call 3, approved); worst16-paper8 regenerated procedurally
    (n=21 asserted). Each list's construction is re-checked for the audit trail.
S3  the 5-condition x 4-set table, comparators read from stored runs (I6),
    filtered on `condition` (D010's runs also hold `joint_mmd` at k=8).
    Point estimates only -- descriptive, no significance test (D019).

Usage:  PYTHONPATH=.:experiments python experiments/d019_hard_model_recall.py
"""
import os
import json
import time
import hashlib

import numpy as np
import torch

from LLMmap.inference_model_archs import InferenceModelLLMmap
from d009_lib import (load_query_embeddings, build_traces, hparams_from_shipped,
                      logits_for, logit_stats)
from d009_train import run_one
from d008_lib import near_relative_pairs
from d007_lib import load_corpus

OUT = "./results/D019"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K, N_SEEDS = 8, 5

# ---- the four model sets, verbatim from D019 `## D` (P1/Call 3)
WORST6 = ["HuggingFaceH4/zephyr-7b-beta", "meta-llama/Llama-3.2-3B-Instruct",
          "meta-llama/Meta-Llama-3.1-8B-Instruct",
          "microsoft/Phi-3-medium-128k-instruct", "microsoft/Phi-3-medium-4k-instruct",
          "microsoft/Phi-3-mini-128k-instruct", "mistralai/Mistral-7B-Instruct-v0.2",
          "mistralai/Mistral-7B-Instruct-v0.3", "tiiuae/Falcon3-10B-Instruct",
          "tiiuae/Falcon3-7B-Instruct"]
CONSENSUS6 = ["microsoft/Phi-3-medium-128k-instruct",
              "microsoft/Phi-3-medium-4k-instruct",
              "mistralai/Mistral-7B-Instruct-v0.2",
              "mistralai/Mistral-7B-Instruct-v0.3",
              "tiiuae/Falcon3-10B-Instruct", "tiiuae/Falcon3-7B-Instruct"]
NATURALGAP5 = ["microsoft/Phi-3-medium-128k-instruct",
               "microsoft/Phi-3-medium-4k-instruct",
               "microsoft/Phi-3-mini-4k-instruct",
               "tiiuae/Falcon3-10B-Instruct", "tiiuae/Falcon3-7B-Instruct",
               "mistralai/Mistral-7B-Instruct-v0.2",
               "mistralai/Mistral-7B-Instruct-v0.3"]


def union(rows):
    return sorted({m for r in rows for m in r["pair"].split(" | ")})


def model_sets(models):
    d15 = json.load(open("./results/D015/per_pair_k8.json"))["rows"]
    p8 = sorted(d15, key=lambda r: (r["paper8"], r["index"]))
    sets = dict(worst6_paper8=sorted(WORST6), worst16_paper8=union(p8[:16]),
                consensus6=sorted(CONSENSUS6), naturalgap5=sorted(NATURALGAP5))
    want = dict(worst6_paper8=10, worst16_paper8=21, consensus6=6, naturalgap5=7)
    for s, n in want.items():
        assert len(sets[s]) == n, f"{s}: {len(sets[s])} != {n}"
        assert all(m in models for m in sets[s]), f"{s}: unknown model"

    # audit trail only: does each list's stated construction reproduce it?
    stable = {c: {r["pair"] for r in sorted(d15, key=lambda r: r[c])[:6]}
              for c in ("paper8", "coverage", "joint_energy")}
    by_idx = {c: {r["pair"] for r in sorted(d15, key=lambda r: (r[c], r["index"]))[:6]}
              for c in ("paper8", "coverage", "joint_energy")}
    cons_file = union([r for r in d15 if r["pair"] in set.intersection(*stable.values())])
    cons_idx = union([r for r in d15 if r["pair"] in set.intersection(*by_idx.values())])
    full = json.load(open("./results/D016/full_pair_trained_accuracy.json"))["rows"]
    mn = sorted(full, key=lambda r: (r["min_over_methods"], r["index"]))
    v = [r["min_over_methods"] for r in mn[:8]]
    audit = dict(
        worst6_paper8=dict(rule="paper8 ascending, pair-index tie-break, 6 pairs",
                           reproduces=union(p8[:6]) == sorted(WORST6)),
        worst16_paper8=dict(rule="paper8 ascending, pair-index tie-break, 16 pairs "
                                 "(procedural -- this IS the set)", n=len(sets["worst16_paper8"])),
        consensus6=dict(rule="3-way intersection of worst-6, ties broken in D015 "
                             "file order (paper8-ascending), per corrected ## D",
                        reproduces=cons_file == sorted(CONSENSUS6),
                        pair_index_tiebreak_gives=len(cons_idx)),
        naturalgap5=dict(rule="first 5 of 666 pairs by min_over_methods "
                              "(D016 full_pair_trained_accuracy.json)",
                         reproduces=union(mn[:5]) == sorted(NATURALGAP5),
                         sorted_min_first8=v,
                         gap_after_rank5=round(v[5] - v[4], 4)))
    return sets, audit


def comparators():
    """paper8 / GreedyCover / JointGreedy per_model from stored runs (I6)."""
    d9 = json.load(open("./results/D009/runs.json"))["runs"]
    d10 = json.load(open("./results/D010/metrics_by_k.json"))["runs"]
    src = dict(paper8=(d9, "paper8"), GreedyCover=(d9, "cvar_max"),
               JointGreedy=(d10, "joint_energy"))
    out = {}
    for name, (runs, cond) in src.items():
        rs = [r for r in runs if r["k"] == K and r["condition"] == cond
              and not r["error"]]
        assert len(rs) == N_SEEDS, (name, len(rs))   # joint_mmd trap (P1)
        out[name] = dict(source_condition=cond,
                         per_model_by_seed=[r["per_model"] for r in rs],
                         mean_top1_by_seed=[r["mean_top1"] for r in rs])
    return out


def reload_seed0(path, queries, qe, cube_t, hard, n_models):
    hp, _ = hparams_from_shipped(K, n_models)
    net = InferenceModelLLMmap(hp)
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    sd = {kk[len("net."):] if kk.startswith("net.") else kk: vv
          for kk, vv in sd.items()}
    net.load_state_dict(sd)
    tr_t, y_t, c_t, _ = build_traces(queries, "test", qe, cube_t)
    pt, _ = logit_stats(logits_for(net, tr_t, DEV), y_t, c_t, hard, n_models)
    return pt


def main():
    os.makedirs(f"{OUT}/models", exist_ok=True)
    t0 = time.time()
    models = json.load(open("./results/D008/selection.json"))["models"]
    n_models = len(models)
    hard = near_relative_pairs(models)
    sets, audit = model_sets(models)
    print(f"[S2] sets {({s: len(m) for s, m in sets.items()})}; audit "
          f"{({s: a.get('reproduces', a.get('n')) for s, a in audit.items()})}",
          flush=True)
    comp = comparators()

    chains = json.load(open("./results/D018/greedy_chains.json"))["chains"]
    d18 = json.load(open("./results/D018/metrics_by_k.json"))["metrics"]
    hash_ref = json.load(open("./results/D009/runs.json"))["hparams_hash"]

    # ---------------- S0
    avail = {}
    for cname in ("H1-attention", "H1-linear"):
        paths = {r: f"./results/D018/models/{cname}_k{K}_r{r}.ckpt"
                 for r in range(N_SEEDS)}
        avail[cname] = {str(r): os.path.exists(p) for r, p in paths.items()}
    n_have = sum(sum(v.values()) for v in avail.values())
    print(f"[S0] {n_have}/10 D018 checkpoints present: {avail}", flush=True)

    # ---------------- S1
    qe = load_query_embeddings()
    cubes = {p: load_corpus(pool=p) for p in ("build", "val", "test")}
    new, s0_check = {}, {}
    for cname in ("H1-attention", "H1-linear"):
        q = chains[cname][:K]
        tr_b, y_b, _, _ = build_traces(q, "build", qe, cubes["build"])
        tr_v, y_v, _, _ = build_traces(q, "val", qe, cubes["val"])
        tr_t, y_t, c_t, _ = build_traces(q, "test", qe, cubes["test"])
        hp, conf = hparams_from_shipped(K, n_models)
        conf = dict(conf); conf["inference_model"] = hp
        h = hashlib.sha256(json.dumps(
            {kk: vv for kk, vv in hp.items() if kk != "num_queries"},
            sort_keys=True).encode()).hexdigest()
        assert h == hash_ref
        pts = []
        for r in range(N_SEEDS):
            t1 = time.time()
            o, _, _ = run_one(tr_b, y_b, tr_v, y_v, tr_t, y_t, c_t, hp, conf, r,
                              hard, n_models, f"{OUT}/models/{cname}_k{K}_r{r}.ckpt")
            assert not o["error"], o["error"]
            pts.append(o)
            print(f"[S1] {cname} seed {r}: mean_top1 {o['mean_top1']:.6f} "
                  f"({time.time() - t1:.1f}s)", flush=True)
        mt = np.array([p["mean_top1"] for p in pts])
        ref = d18[f"{cname}|attention"][str(K)]
        d_mean = abs(mt.mean() - ref["mean_top1"])
        d_rng = abs((mt.max() - mt.min()) - ref["mean_top1_range"])
        assert d_mean < 1e-6 and d_rng < 1e-6, (cname, d_mean, d_rng)
        for p in pts:   # balanced test set: mean(per_model) must equal mean_top1
            assert abs(np.mean(p["per_model"]) - p["mean_top1"]) < 1e-4

        # the two saved seed-0 checkpoints: bit-identity with retrained seed 0
        path = f"./results/D018/models/{cname}_k{K}_r0.ckpt"
        pt = reload_seed0(path, q, qe, cubes["test"], hard, n_models)
        same = (pt["per_model"] == pts[0]["per_model"]
                and pt["mean_top1"] == pts[0]["mean_top1"])
        s0_check[cname] = dict(path=path, reloaded_mean_top1=pt["mean_top1"],
                               retrained_seed0_mean_top1=pts[0]["mean_top1"],
                               per_model_bit_identical=bool(same))
        assert same, s0_check[cname]
        new[cname] = dict(queries=q, per_model_by_seed=[p["per_model"] for p in pts],
                          mean_top1_by_seed=mt.tolist(),
                          d018_stored_mean_top1=ref["mean_top1"],
                          d018_stored_range=ref["mean_top1_range"],
                          abs_diff_mean=float(d_mean), abs_diff_range=float(d_rng))
        print(f"[S1] {cname}: 5-seed mean {mt.mean():.6f} == D018 "
              f"{ref['mean_top1']:.6f}; range diff {d_rng:.1e}; seed-0 ckpt "
              f"bit-identical {same}", flush=True)

    json.dump(dict(schema="d019-s0-v1", k=K, present=avail, n_present=n_have,
                   note="D018's d018_evaluate.py saved a checkpoint only `if r == 0 "
                        "and k == 8`: seeds 1-4 were never written (not file loss). "
                        "Both files are attention networks; H1-linear_k8_r0 is the "
                        "attention network on the H1-linear chain. No linear-"
                        "classifier fit was ever persisted.",
                   resolution="all 5 seeds x 2 chains retrained (P1/Call 2); saved "
                              "seed-0 checkpoints used as a bit-identity check",
                   seed0_bit_identity=s0_check,
                   retrained_checkpoints=f"{OUT}/models/ (10 files, gitignored)"),
              open(f"{OUT}/checkpoint_availability.json", "w"), indent=1)

    # ---------------- S2 / S3
    allc = {**comp, **new}
    order = ["paper8", "GreedyCover", "JointGreedy", "H1-attention", "H1-linear"]
    table = {}
    for c in order:
        pm = np.array(allc[c]["per_model_by_seed"])        # (5, 37)
        pm_mean = pm.mean(axis=0)
        all37 = float(np.mean(allc[c]["mean_top1_by_seed"]))
        row = dict(all37_mean_top1=round(all37, 4))
        for s, ms in sets.items():
            idx = [models.index(m) for m in ms]
            per_seed = pm[:, idx].mean(axis=1)
            row[s] = dict(mean=round(float(pm_mean[idx].mean()), 4),
                          gap_pp=round(100 * (float(pm_mean[idx].mean()) - all37), 1),
                          seed_min=round(float(per_seed.min()), 4),
                          seed_max=round(float(per_seed.max()), 4))
        table[c] = row
        allc[c]["per_model_mean"] = pm_mean.round(4).tolist()

    print(f"\n{'':14s}{'all-37':>8s}" + "".join(f"{s:>22s}" for s in sets), flush=True)
    for c in order:
        r = table[c]
        print(f"{c:14s}{r['all37_mean_top1']:8.4f}" + "".join(
            f"{r[s]['mean']:>12.3f} ({r[s]['gap_pp']:+5.1f}pp)" for s in sets),
            flush=True)

    json.dump(dict(schema="d019-hard-model-recall-v1", k=K, readout="attention "
                   "network, 5 seeds, for all five conditions (P1/Call 1)",
                   descriptive_only="point estimates; no significance test (D019); "
                                    "seed_min/max are the restricted mean's range "
                                    "across the 5 seeds, for context",
                   models=models, model_sets=sets, model_set_audit=audit,
                   table=table,
                   gap_definition="restricted mean minus the same condition's "
                                  "all-37 mean_top1, in percentage points",
                   conditions={c: dict(source=("D009/runs.json" if c in
                                               ("paper8", "GreedyCover") else
                                               "D010/metrics_by_k.json runs"
                                               if c == "JointGreedy" else
                                               "retrained here (D019/S1)"),
                                       **allc[c]) for c in order},
                   wall_min=round((time.time() - t0) / 60, 2)),
              open(f"{OUT}/hard_model_recall.json", "w"), indent=1)
    print(f"\nwall {(time.time() - t0) / 60:.1f} min; written {OUT}/", flush=True)


if __name__ == "__main__":
    main()
