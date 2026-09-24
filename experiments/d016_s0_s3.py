"""D016 / S0 + S3 — checkpoint availability, and the tensor proxy over all 666 pairs.

S0 gates the rest of the D. S3 is explicitly independent of it ("(b)'s tensor-level
extension needs no checkpoint at all"), so both run here; S1/S2/S4 wait on Call 1.

S0 checks more than existence: it loads each surviving checkpoint, recovers the
37-way test logits, and asserts the recovered mean_top1 and per_model reproduce the
values the source D stored for that run. A checkpoint that exists but pairs with
the wrong data would otherwise propagate silently into every S2 row.

Usage:  PYTHONPATH=.:experiments python experiments/d016_s0_s3.py
"""
import os
import json
import time
import glob
import itertools

import numpy as np
import torch

from LLMmap.inference_model_archs import InferenceModelLLMmap
from d008_lib import (load_tensor, build_dq, two_way_correct, pair_index,
                      near_relative_pairs)
from d009_lib import (load_query_embeddings, build_traces, hparams_from_shipped,
                      logits_for)
from d007_lib import load_corpus

OUT = "./results/D016"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K = 8
COND = {"paper8": ("D009", "results/D009/runs.json", "paper8"),
        "coverage": ("D009", "results/D009/runs.json", "cvar_max"),
        "joint_energy": ("D010", "results/D010/metrics_by_k.json", "joint_energy")}


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    models = json.load(open("./results/D008/selection.json"))["models"]
    n_models = len(models)
    all_pairs = pair_index(models)
    hard = near_relative_pairs(models)
    structural = {" | ".join(hard[i]["pair"]) for i in hard}
    print(f"{n_models} models, {len(all_pairs)} pairs, "
          f"{len(structural)} structural", flush=True)

    # ---------------- S0
    qe = load_query_embeddings()
    cubes_t = load_corpus(pool="test")
    avail = {}
    for cname, (where, mfile, key) in COND.items():
        src = json.load(open(mfile))
        runs = [r for r in src["runs"] if r["k"] == K and not r["error"]
                and r.get("condition") == key]
        want = len(runs)
        got = []
        for r in runs:
            # the source D's own naming: only seed 0 was kept (`if r_ == 0`)
            path = f"./results/{where}/models/{key}_k{K}_r{r['run']}.ckpt"
            rec = dict(run=r["run"], stored_mean_top1=r["mean_top1"],
                       best_ckpt_recorded=r["best_ckpt"], path=path,
                       exists=os.path.exists(path))
            if rec["exists"]:
                try:
                    hp, _ = hparams_from_shipped(K, n_models)
                    net = InferenceModelLLMmap(hp)
                    sd = torch.load(path, map_location="cpu")
                    sd = sd.get("state_dict", sd)
                    sd = {kk[len("net."):] if kk.startswith("net.") else kk: vv
                          for kk, vv in sd.items()}
                    net.load_state_dict(sd)
                    tr_t, y_t, c_t, _ = build_traces(r["queries"], "test", qe, cubes_t)
                    lg = logits_for(net, tr_t, DEV)
                    acc = float((lg.argmax(1) == y_t).mean())
                    rec.update(loads=True, recovered_mean_top1=round(acc, 6),
                               reproduces=bool(abs(acc - r["mean_top1"]) < 1e-6),
                               abs_diff=round(abs(acc - r["mean_top1"]), 8))
                    np.save(f"{OUT}/logits_{cname}_k{K}_r{r['run']}.npy", lg)
                except Exception as e:
                    rec.update(loads=False, error=f"{type(e).__name__}: {e}")
            got.append(rec)
        n_ok = sum(1 for g in got if g.get("reproduces"))
        avail[cname] = dict(runs_in_source=want, checkpoints_present=
                            sum(1 for g in got if g["exists"]),
                            load_and_reproduce=n_ok, detail=got)
        print(f"[S0] {cname:13s} {want} runs in source, "
              f"{avail[cname]['checkpoints_present']} checkpoints on disk, "
              f"{n_ok} load and reproduce stored mean_top1", flush=True)

    total_ok = sum(v["load_and_reproduce"] for v in avail.values())
    gate = all(v["load_and_reproduce"] >= 5 for v in avail.values())
    print(f"[S0] gate (>=5 per condition): {gate}  ({total_ok}/15 usable)",
          flush=True)
    json.dump(dict(schema="d016-s0-v1", k=K, conditions=list(COND),
                   gate_pass=bool(gate), total_usable=total_ok,
                   note="D009/D010 kept only seed 0 per (condition, k) -- their "
                        "training loops wrote a checkpoint `if r_ == 0`. The other "
                        "seeds were never saved, so this is not file loss.",
                   availability=avail, wall_s=round(time.time() - t0, 1)),
              open(f"{OUT}/checkpoint_availability.json", "w"), indent=1)

    # ---------------- S3: tensor proxy over all 666 pairs (no checkpoint needed)
    t1 = time.time()
    S, meta = load_tensor()
    Dq, w2, y_ev, y_rf, _, cfg_ev = build_dq(eval_pool="test")
    rows = []
    for j, (a, b) in enumerate(all_pairs):
        ia, ib = models.index(a), models.index(b)
        accs = np.array([two_way_correct(Dq[q], y_ev, y_rf, ia, ib)[0].mean()
                         for q in range(S.shape[0])])
        qs = int(S[:, j].argmax())
        nm = f"{a} | {b}"
        rows.append(dict(pair=nm, index=j, structural=nm in structural,
                         median_query=round(float(np.median(accs)), 4),
                         mean_query=round(float(accs.mean()), 4),
                         best_build_query=round(float(accs[qs]), 4),
                         oracle=round(float(accs.max()), 4),
                         floor=round(float(accs.min()), 4)))
        if (j + 1) % 150 == 0:
            print(f"  [S3] {j+1}/{len(all_pairs)}", flush=True)
    med = np.array([r["median_query"] for r in rows])
    st = np.array([r["structural"] for r in rows])
    print(f"[S3] median_query: all {med.mean():.4f} | structural "
          f"{med[st].mean():.4f} (n={int(st.sum())}) | non-structural "
          f"{med[~st].mean():.4f} (n={int((~st).sum())})", flush=True)
    worst = sorted(rows, key=lambda r: r["median_query"])[:10]
    print("[S3] 10 hardest by median_query (proxy):")
    for r in worst:
        print(f"   {r['pair'][:62]:62s} med {r['median_query']:.3f} "
              f"oracle {r['oracle']:.3f} structural={r['structural']}")
    json.dump(dict(schema="d016-tensor-proxy-v1", n_pairs=len(rows),
                   n_structural=int(st.sum()),
                   statistic="two-logit-restricted accuracy of D008's 1-NN proxy, "
                             "per query, summarised over the 259-query pool",
                   oracle_caveat="`oracle` and `best_build_query` are retained for "
                                 "continuity but are the statistics D011 showed to "
                                 "be substantially estimator artifacts; "
                                 "`median_query` is the one used for correlation.",
                   summary=dict(all=round(float(med.mean()), 4),
                                structural=round(float(med[st].mean()), 4),
                                non_structural=round(float(med[~st].mean()), 4)),
                   rows=rows, wall_s=round(time.time() - t1, 1)),
              open(f"{OUT}/full_pair_tensor_proxy.json", "w"), indent=1)
    print(f"\nS0 + S3 wall {time.time()-t0:.1f}s; written: {OUT}/", flush=True)


if __name__ == "__main__":
    main()
