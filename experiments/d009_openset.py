"""D009 / S8 — the open-set variant, k=8, all four conditions (approved 2026-09-09).

Supplementary deliverable: its result does NOT enter S7's closed-set verdict.

What "open-set" means in the released code, stated precisely because S8 asks for
it. `train.py` without `--is_closed` trains `make_siamese_network` under
`ContrastiveLoss` (a similarity objective, not the closed softmax head), then
`setup_templates.py` calls `templates.template_generation`, which:

    1. runs the shared feature extractor over the TRAIN split,
    2. averages its features per model into one template each,
    3. classifies held-out traces by NEAREST TEMPLATE (L2).

So the metric the paper's open-set path implies, and the one reported here, is
**nearest-template top-1 / top-3 accuracy over the 37 known models**. Those three
steps are executed by the shipped functions (`infer_features`, `build_templates`,
`predict_by_templates`), not reimplemented.

**What this does NOT measure**, stated so the number is not over-read: rejection
of a model absent from training. The open-set path's actual selling point is that
a new LLM can be added by computing one template without retraining — testing
that requires models held out of training entirely, which is a different
experiment and a different corpus split than I2 provides. This measures the
accuracy of the template mechanism on known classes, which is what
`setup_templates.py` itself reports.

Usage:  PYTHONPATH=.:experiments python experiments/d009_openset.py
"""
import os
import io
import json
import time
import glob
import shutil
import contextlib
import tempfile

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from LLMmap.trainer import train_model
from LLMmap.templates import infer_features, build_templates, predict_by_templates
from d009_lib import (load_query_embeddings, build_traces, make_loader,
                      hparams_from_shipped, d008_chains, TracePairDataset, OUT)
from d008_lib import near_relative_pairs
from d007_lib import load_corpus

DEV = "cuda" if torch.cuda.is_available() else "cpu"
K = 8
SMOKE = bool(os.environ.get("D009_SMOKE"))


def topk_from_templates(feats, templates, y, k=3):
    d = np.linalg.norm(feats[:, None, :] - templates[None, :, :], axis=-1)
    order = np.argsort(d, axis=1)
    return {f"top{i}": float(np.mean([(y[n] in order[n, :i])
                                      for n in range(len(y))]))
            for i in (1, k)}, d


def main():
    os.makedirs(OUT, exist_ok=True)
    qe = load_query_embeddings()
    chains, draws, sel = d008_chains()
    models = sel["models"]
    n_models = len(models)
    hard = near_relative_pairs(models)
    cubes = {p: load_corpus(pool=p) for p in ("build", "val", "test")}

    conds = {"paper8": chains["paper8"], "cvar_max": chains["cvar_max"][:K],
             "mean_greedy_max": chains["mean_greedy_max"][:K],
             "random": list(draws[str(K)][0])}
    out = {}
    for cond, queries in conds.items():
        tr_b, y_b, _, _ = build_traces(queries, "build", qe, cubes["build"])
        tr_v, y_v, _, _ = build_traces(queries, "val", qe, cubes["val"])
        tr_t, y_t, c_t, _ = build_traces(queries, "test", qe, cubes["test"])

        hp, conf = hparams_from_shipped(K, n_models)
        conf = dict(conf); conf["inference_model"] = hp
        if SMOKE:
            conf["num_pairs_per_epoch"] = 2000
            conf["num_pairs_per_eval"] = 500
            conf["training"] = dict(conf["training"], max_epochs=1)

        pl.seed_everything(0, workers=True)
        pairs_tr = TracePairDataset(tr_b, y_b, conf["num_pairs_per_epoch"], n_models)
        pairs_va = TracePairDataset(tr_v, y_v, conf["num_pairs_per_eval"], n_models)
        ltr = DataLoader(pairs_tr, batch_size=conf["batch_size"], shuffle=False)
        lva = DataLoader(pairs_va, batch_size=conf["batch_size"], shuffle=False)

        tmp = tempfile.mkdtemp(prefix="d009_open_")
        buf = io.StringIO()
        t = time.time()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                # siamese=True -> train_model returns the shared FEATURE
                # EXTRACTOR, which is exactly what templates.py consumes
                _, feat = train_model(tmp, siamese=True, loader_train=ltr,
                                      loader_test=lva, conf=conf)
            err = None
        except Exception as ex:
            err, feat = f"{type(ex).__name__}: {ex}", None
        wall = time.time() - t
        best = glob.glob(os.path.join(tmp, "best-*.ckpt"))
        if best:
            shutil.copy(best[0], f"{OUT}/models/openset_{cond}_k{K}.ckpt")
        shutil.rmtree(tmp, ignore_errors=True)

        if err:
            print(f"  !! {cond}: {err}\n{buf.getvalue()[-800:]}", flush=True)
            out[cond] = dict(error=err, wall_min=round(wall / 60, 1))
            continue

        # the shipped three steps, via the shipped functions
        yb, fb = infer_features(feat, make_loader(tr_b, y_b, 256, False), DEV)
        templates = build_templates(yb, fb)
        yt, ft = infer_features(feat, make_loader(tr_t, y_t, 256, False), DEV)
        acc, dmat = topk_from_templates(ft, templates, yt)
        pred = predict_by_templates(ft, templates)
        per_model = np.array([(pred[yt == m] == m).mean() for m in range(n_models)])
        hv = []
        for i, info in sorted(hard.items()):
            a, b = info["ab"]
            msk = (yt == a) | (yt == b)
            two = dmat[msk][:, [a, b]]
            p2 = np.where(two.argmin(1) == 0, a, b)
            hv.append(float((p2 == yt[msk]).mean()))
        hv = np.array(hv)

        out[cond] = dict(
            queries=queries, wall_min=round(wall / 60, 1),
            nearest_template_top1=acc["top1"], nearest_template_top3=acc["top3"],
            worst_class=float(per_model.min()),
            worst3_class=float(np.sort(per_model)[:3].mean()),
            hard_subset=float(hv.mean()), hard_subset_min=float(hv.min()),
            n_test=int(len(yt)))
        print(f"[S8] {cond:16s} top-1 {acc['top1']:.4f} top-3 {acc['top3']:.4f} "
              f"worst {per_model.min():.4f} hard {hv.mean():.4f} "
              f"({wall/60:.1f} min)", flush=True)

    json.dump(dict(
        schema="d009-openset-v1", k=K, n_models=n_models,
        chance=round(1 / n_models, 4),
        metric="nearest-template top-1/top-3 over the 37 KNOWN models -- the "
               "metric setup_templates.py itself reports; templates built from "
               "S_build features, evaluated on S_test (I2)",
        not_measured="rejection of a model absent from training. That is the "
                     "open-set path's actual selling point, and testing it needs "
                     "models held out of training entirely -- a different "
                     "experiment than this corpus split supports.",
        supplementary=True,
        note="approved 2026-09-09 as S8 (TACC's F7); does not enter S7's verdict",
        conditions=out), open(f"{OUT}/open_set_metrics.json", "w"), indent=1)
    print(f"\nwritten: {OUT}/open_set_metrics.json")


if __name__ == "__main__":
    main()
