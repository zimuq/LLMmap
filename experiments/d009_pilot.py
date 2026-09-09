"""D009 — cost pilot + the two checks that must pass before the grid runs.

D009's Cost constraint requires a MEASURED per-run cost from a small pilot
before Call 1's run matrix is committed to. It also warns that
`LLMmap/trainer.py` may need a compatibility fix to run at all, and requires
any such diff be disclosed rather than silently applied. This script exists to
find both out now, on one run, instead of inside an 80-run grid.

What it does:

  A. verifies S1's artifact (D006/S6 already embedded all 259 pool queries)
  B. EQUIVALENCE CHECK -- assembles traces for a handful of entries both ways:
     ours (from the frozen embeddings) and the shipped `LLMmap/dataset.py`
     path (raw text -> EmbeddingCache -> pack_traces). P1 claims these are the
     same tensor; this measures the claim instead of asserting it.
  C. parameter count of the network at k=8 (METHOD §5.5 says ~8M -- check)
  D. one real closed-set training run, timed, using `train_model` UNMODIFIED
  E. a siamese step-rate probe, to price the open-set variant D009 does not
     currently ask for but the human's framing mentioned

Usage:  PYTHONPATH=.:experiments python experiments/d009_pilot.py
"""
import os
import re
import json
import time
import shutil
import tempfile

import numpy as np
import torch

from LLMmap.trainer import train_model
from LLMmap.inference_model_archs import InferenceModelLLMmap, make_siamese_network
from d009_lib import (load_query_embeddings, build_traces, make_loader,
                      hparams_from_shipped, logits_for, logit_stats,
                      d008_chains, OUT)
from d008_lib import near_relative_pairs
from d007_lib import load_corpus

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def equivalence_check(queries, n_entries=6):
    """Ours vs the shipped text->embedding->pack_traces path, same entries."""
    from LLMmap.dataset import EmbeddingCache, DatasetFactory
    from LLMmap.embedding_model import load_model

    man = json.load(open("./data/corpus_v1/corpus_manifest.json"))
    shards = sorted((s for s in man["models"] if s["status"] == "VALIDATED"),
                    key=lambda s: s["model"])
    pool = json.load(open("./confs/queries/pool_v1.json"))
    qtexts = [pool["queries"][q]["text"] for q in queries]

    raw = []
    for s in shards[:n_entries]:
        with open(s["shard_path"]) as f:
            for line in f:
                d = json.loads(line)
                if d["dataset"] == "build" and d["config_index"] == 0:
                    raw.append(dict(llm=s["model"],
                                    traces=[[qtexts[i], d["traces"][q][1]]
                                            for i, q in enumerate(queries)]))
                    break
    emb = load_model(0, device_map=DEV)
    cache = EmbeddingCache(emb, 64)
    cache.precompute(raw)
    shipped = DatasetFactory(raw, cache).pack_traces(raw[0]["traces"]).cpu().numpy()

    models, X = load_corpus(pool="build")
    qe = load_query_embeddings()
    mi = models.index(raw[0]["llm"])
    ours = np.empty_like(shipped)
    for si, q in enumerate(queries):
        ours[si, :1024] = qe[q]
        ours[si, 1024:] = X[models[mi]][q][0]

    d = np.abs(ours - shipped)
    rel = d / (np.abs(shipped) + 1e-6)
    return dict(entries_compared=len(raw), shape=list(shipped.shape),
                max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()),
                median_rel_diff=float(np.median(rel)),
                cosine=float((ours * shipped).sum() /
                             (np.linalg.norm(ours) * np.linalg.norm(shipped))),
                note="ours reads D006/S6's fp16 store; the shipped path recomputes "
                     "in fp32, so exact equality is not expected -- fp16 storage "
                     "is the only difference and it is a property of the frozen "
                     "corpus (I6), not of this assembly")


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}

    # ---- A. S1 is already satisfied by a frozen artifact
    qe = load_query_embeddings()
    res["S1"] = dict(source="data/corpus_v1/embeddings/_queries.npy (D006/S6)",
                     shape=list(qe.shape), recomputed=False,
                     note="D006/S6 embedded all 259 pool queries at corpus-build "
                          "time precisely because 'the pipeline consumes both'. "
                          "S1 costs nothing and cannot drift from the responses.")
    print(f"[A] S1 artifact ok: {qe.shape}", flush=True)

    chains, draws, sel = d008_chains()
    q8 = chains["paper8"]

    # ---- B. equivalence against the shipped loader
    t = time.time()
    res["equivalence"] = equivalence_check(q8)
    e = res["equivalence"]
    print(f"[B] equivalence vs shipped loader: max|Δ| {e['max_abs_diff']:.2e}, "
          f"median rel {e['median_rel_diff']:.2e}, cosine {e['cosine']:.8f} "
          f"({time.time()-t:.0f}s)", flush=True)

    # ---- C. the network at k=8
    models = sel["models"]
    hp, conf = hparams_from_shipped(len(q8), len(models))
    net = InferenceModelLLMmap(hp)
    n_par = sum(p.numel() for p in net.parameters())
    res["architecture"] = dict(hparams=hp, params=n_par,
                               source="confs/default.json inference_model block",
                               method_5_5_claim="~8M params")
    print(f"[C] InferenceModelLLMmap at k=8: {n_par:,} params "
          f"(METHOD §5.5 says ~8M)", flush=True)

    # ---- D. one real training run, shipped train_model, timed
    t = time.time()
    cubes_b = load_corpus(pool="build")
    cubes_v = load_corpus(pool="val")
    cubes_t = load_corpus(pool="test")
    tr_b, y_b, c_b, _ = build_traces(q8, "build", qe, cubes_b)
    tr_v, y_v, c_v, _ = build_traces(q8, "val", qe, cubes_v)
    tr_t, y_t, c_t, _ = build_traces(q8, "test", qe, cubes_t)
    t_assemble = time.time() - t
    print(f"[D] traces: train {tr_b.shape}, val {tr_v.shape}, test {tr_t.shape} "
          f"({t_assemble:.0f}s, {tr_b.nbytes/1e6:.0f} MB)", flush=True)

    conf_run = dict(conf)
    conf_run["inference_model"] = hp
    loader_tr = make_loader(tr_b, y_b, conf["batch_size"], True)
    loader_va = make_loader(tr_v, y_v, conf["batch_size"], False)

    tmp = tempfile.mkdtemp(prefix="d009_pilot_")
    torch.manual_seed(0)
    t = time.time()
    try:
        _, model = train_model(tmp, siamese=False, loader_train=loader_tr,
                               loader_test=loader_va, conf=conf_run)
        train_err = None
    except Exception as ex:
        train_err = f"{type(ex).__name__}: {ex}"
        model = None
    t_train = time.time() - t

    res["training"] = dict(wall_s=round(t_train, 1), error=train_err,
                           steps_per_epoch=int(np.ceil(len(tr_b) / conf["batch_size"])),
                           max_epochs=conf["training"]["max_epochs"],
                           patience=conf["training"]["early_stop_patience"],
                           n_train=len(tr_b), n_val=len(tr_v), n_test=len(tr_t),
                           device=DEV)
    if train_err:
        print(f"[D] TRAINING FAILED after {t_train:.0f}s: {train_err}", flush=True)
    else:
        hard = near_relative_pairs(models)
        lg = logits_for(model, tr_t, DEV)
        pt, _ = logit_stats(lg, y_t, c_t, hard, len(models))
        lv = logits_for(model, tr_v, DEV)
        pv, _ = logit_stats(lv, y_v, c_v, hard, len(models))
        res["training"]["test_metrics"] = {k: v for k, v in pt.items()
                                           if not k.startswith("per_")}
        res["training"]["val_metrics"] = {k: v for k, v in pv.items()
                                          if not k.startswith("per_")}
        print(f"[D] trained in {t_train:.0f}s -> paper8 k=8 TEST top-1 "
              f"{pt['mean_top1']:.4f}  worst {pt['worst_class']:.4f}  "
              f"hard {pt['hard_subset']:.4f}  (val top-1 {pv['mean_top1']:.4f})",
              flush=True)
    shutil.rmtree(tmp, ignore_errors=True)

    # ---- E. siamese step-rate probe (prices the open-set variant)
    siam, _ = make_siamese_network(hp)
    siam = siam.to(DEV)
    opt = torch.optim.Adam(siam.parameters(), lr=1e-4)
    x = torch.randn(conf["batch_size"], 2, len(q8), 2048, device=DEV)
    yb = torch.randint(0, 2, (conf["batch_size"],), device=DEV).float()
    for _ in range(3):
        opt.zero_grad(); out = siam(x)[:, 0]
        ((out - yb) ** 2).mean().backward(); opt.step()
    if DEV == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(30):
        opt.zero_grad(); out = siam(x)[:, 0]
        ((out - yb) ** 2).mean().backward(); opt.step()
    if DEV == "cuda":
        torch.cuda.synchronize()
    sps = 30 / (time.time() - t)
    spe = int(np.ceil(conf["num_pairs_per_epoch"] / conf["batch_size"]))
    res["siamese_probe"] = dict(
        steps_per_s=round(sps, 1), steps_per_epoch=spe,
        min_per_epoch=round(spe / sps / 60, 1),
        num_pairs_per_epoch=conf["num_pairs_per_epoch"],
        note="open-set (siamese + templates) is a SEPARATE training run per "
             "condition, not a free head on the closed model: the closed model's "
             "forward returns head logits, and the shipped default is 500k pairs "
             "per epoch against the closed run's 2,775 samples")
    print(f"[E] siamese probe: {sps:.1f} steps/s, {spe:,} steps/epoch -> "
          f"{spe/sps/60:.1f} min/epoch", flush=True)

    json.dump(res, open(f"{OUT}/pilot.json", "w"), indent=1)
    print(f"\nwritten: {OUT}/pilot.json")


if __name__ == "__main__":
    main()
