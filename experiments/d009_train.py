"""D009 / S0–S4 — verify inputs, then train the closed-set grid.

Grid, as approved (Calls 1 and 2): 4 conditions x k=1..8 x 5 runs = 160 runs.
  cvar_max, mean_greedy_max, paper8 -> 5 TRAINING seeds {0..4}, identical set
                                       across conditions so deltas are paired
  random                            -> D008's own draws 0..4, one seed each,
                                       so its band carries draw AND train
                                       variance (I6 covers selection outputs)

Training uses `LLMmap/trainer.py`'s `train_model` UNMODIFIED. The only
deviation from the shipped `train.py` is which loader is handed to it as the
validation loader: `S_val`, not `S_test` (P1/F2, approved). `S_test` is loaded
once per run, after training, for evaluation only.

Usage:  PYTHONPATH=.:experiments python experiments/d009_train.py
"""
import os
import io
import json
import time
import glob
import shutil
import hashlib
import contextlib
import tempfile

import numpy as np
import torch
import pytorch_lightning as pl

from LLMmap.trainer import train_model
from d009_lib import (load_query_embeddings, build_traces, make_loader,
                      hparams_from_shipped, logits_for, logit_stats,
                      d008_chains, OUT, EMB, QEMB)
from d008_lib import near_relative_pairs
from d007_lib import load_corpus

DEV = "cuda" if torch.cuda.is_available() else "cpu"
KS = list(range(1, 9))
N_RUNS = 5
SMOKE = bool(os.environ.get("D009_SMOKE"))
if SMOKE:
    KS, N_RUNS = [1, 8], 2


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def s0_verify():
    man = json.load(open("./data/corpus_v1/corpus_manifest.json"))
    assert man["status"] == "READY", man["status"]
    models = sorted(s["model"] for s in man["models"] if s["status"] == "VALIDATED")
    assert len(models) == 37
    qe = load_query_embeddings()
    chains, draws, sel = d008_chains()
    assert sel["schema"] == "d008-selection-v1", sel["schema"]
    assert sel["models"] == models, "D008's model order != manifest order"
    for name, ch in chains.items():
        assert len(set(ch[:8])) == 8, name
    return dict(models=models, qe=qe, chains=chains, draws=draws, sel=sel,
                shas=dict(query_embeddings=sha(QEMB),
                          d008_selection=sha(f"{OUT}/../D008/selection.json"),
                          d007_tensor=sel["tensor"]["sha256"]))


def s1_materialise(qe, shas):
    """D009 names `results/D009/query_embeddings.npy` as a deliverable. The
    bytes come from D006/S6 -- this writes them under the required name with
    provenance, rather than recomputing an artifact that already exists and is
    guaranteed consistent with the response embeddings."""
    np.save(f"{OUT}/query_embeddings.npy", qe)
    json.dump(dict(
        n=int(qe.shape[0]), dim=int(qe.shape[1]),
        source="data/corpus_v1/embeddings/_queries.npy",
        source_sha256=shas["query_embeddings"], recomputed=False,
        embedding_model="intfloat/multilingual-e5-large-instruct",
        pooling="mean, NOT normalised", produced_by="D006/S6",
        note="S1 required no compute: D006/S6 embedded all 259 pool queries at "
             "corpus-build time, so query and response embeddings come from the "
             "same run and cannot drift apart."),
        open(f"{OUT}/query_embeddings.index.json", "w"), indent=2)


def run_one(tr_b, y_b, tr_v, y_v, tr_t, y_t, c_t, hp, conf, seed, hard,
            n_models, keep_ckpt=None):
    pl.seed_everything(seed, workers=True)
    loader_tr = make_loader(tr_b, y_b, conf["batch_size"], True)
    loader_va = make_loader(tr_v, y_v, conf["batch_size"], False)
    tmp = tempfile.mkdtemp(prefix="d009_")
    buf = io.StringIO()
    t = time.time()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            _, model = train_model(tmp, siamese=False, loader_train=loader_tr,
                                   loader_test=loader_va, conf=conf)
        err = None
    except Exception as ex:
        err, model = f"{type(ex).__name__}: {ex}", None
    wall = time.time() - t

    best = glob.glob(os.path.join(tmp, "best-*.ckpt"))
    ckpt_name = os.path.basename(best[0]) if best else None
    if keep_ckpt and best:
        os.makedirs(os.path.dirname(keep_ckpt), exist_ok=True)
        shutil.copy(best[0], keep_ckpt)
    if err is None:
        # S4: train/val fit quality from the SAME model, so overfitting is
        # visible per run rather than inferred from the loss curve alone
        tr_acc = float((logits_for(model, tr_b, DEV).argmax(1) == y_b).mean())
        va_acc = float((logits_for(model, tr_v, DEV).argmax(1) == y_v).mean())
        point, counts = logit_stats(logits_for(model, tr_t, DEV), y_t, c_t,
                                    hard, n_models)
    else:
        tr_acc = va_acc = float("nan"); point, counts = {}, {}
    shutil.rmtree(tmp, ignore_errors=True)
    return dict(wall_s=round(wall, 1), error=err, best_ckpt=ckpt_name,
                train_acc=tr_acc, val_acc=va_acc, **point), counts, buf


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(f"{OUT}/models", exist_ok=True)
    v = s0_verify()
    models, qe, chains, draws = v["models"], v["qe"], v["chains"], v["draws"]
    n_models = len(models)
    hard = near_relative_pairs(models)
    s1_materialise(qe, v["shas"])
    print(f"[S0] verified: 37 models, queries {qe.shape}, D008 chains ok",
          flush=True)

    cubes = {p: load_corpus(pool=p) for p in ("build", "val", "test")}
    print(f"[S2] corpus cubes loaded", flush=True)

    runs, counts, hp_hashes = [], {}, set()
    t_all = time.time()
    for k in KS:
        for cond in ("paper8", "mean_greedy_max", "cvar_max", "random"):
            for r in range(N_RUNS):
                if cond == "random":
                    queries, seed = list(draws[str(k)][r]), 0
                else:
                    queries, seed = chains[cond][:k], r
                tr_b, y_b, _, _ = build_traces(queries, "build", qe, cubes["build"])
                tr_v, y_v, _, _ = build_traces(queries, "val", qe, cubes["val"])
                tr_t, y_t, c_t, _ = build_traces(queries, "test", qe, cubes["test"])

                hp, conf = hparams_from_shipped(k, n_models)
                conf = dict(conf); conf["inference_model"] = hp
                # "identical training procedure" enforced, not intended: the
                # hash must differ only in num_queries, which IS the variable
                hp_hashes.add(hashlib.sha256(json.dumps(
                    {kk: vv for kk, vv in hp.items() if kk != "num_queries"},
                    sort_keys=True).encode()).hexdigest())

                keep = (f"{OUT}/models/{cond}_k{k}_r{r}.ckpt" if r == 0 else None)
                res, cnt, buf = run_one(tr_b, y_b, tr_v, y_v, tr_t, y_t, c_t,
                                        hp, conf, seed, hard, n_models, keep)
                res.update(condition=cond, k=k, run=r, seed=seed,
                           queries=queries)
                runs.append(res)
                if res["error"]:
                    print(f"  !! {cond} k={k} r={r}: {res['error']}", flush=True)
                    print(buf.getvalue()[-800:], flush=True)
                else:
                    for key, arr in cnt.items():
                        if isinstance(arr, np.ndarray):
                            counts[f"{cond}|{k}|{r}|{key}"] = arr
            done = [x for x in runs if x["condition"] == cond and x["k"] == k
                    and not x["error"]]
            if done:
                m = np.array([x["mean_top1"] for x in done])
                print(f"[S3] k={k} {cond:16s} test top-1 {m.mean():.4f} "
                      f"[{m.min():.4f},{m.max():.4f}]  "
                      f"train {np.mean([x['train_acc'] for x in done]):.4f} "
                      f"val {np.mean([x['val_acc'] for x in done]):.4f}",
                      flush=True)

    assert len(hp_hashes) == 1, (
        f"hparams differed across runs ({len(hp_hashes)} distinct) -- the "
        f"comparison would not be holding the training procedure constant")

    np.savez_compressed(f"{OUT}/run_counts.npz", **counts)
    json.dump(dict(schema="d009-runs-v1", n_runs=len(runs), ks=KS,
                   n_runs_per_cell=N_RUNS, device=DEV, models=models,
                   hparams_hash=hp_hashes.pop(),
                   hparams_source="confs/default.json inference_model block",
                   shas=v["shas"],
                   validation_split="val (S_val); S_test touched once per run "
                                    "after training, never for stopping (P1/F2)",
                   wall_total_min=round((time.time() - t_all) / 60, 1),
                   runs=runs), open(f"{OUT}/runs.json", "w"), indent=1)
    ok = [r for r in runs if not r["error"]]
    print(f"\n[S3] {len(ok)}/{len(runs)} runs ok in "
          f"{(time.time()-t_all)/60:.1f} min -> {OUT}/runs.json", flush=True)


if __name__ == "__main__":
    main()
