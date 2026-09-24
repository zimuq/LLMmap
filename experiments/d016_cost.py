"""D016 — cost/determinism measurement for Call 1. NOT S1-S5.

D016 frames a missing checkpoint as forcing "a materially different cost". Before
asking design side to choose, measure the thing the choice turns on: what does
retraining one k=8 run actually cost, and does re-running a seed reproduce the
number the source D stored for it? If retraining is deterministic and cheap, the
Call is about bookkeeping, not cost.

One run only. Nothing from S1-S5 is computed here.
"""
import json, time, hashlib, io, contextlib, tempfile, shutil, sys
import numpy as np
import torch, pytorch_lightning as pl
sys.path[:0] = [".", "experiments"]
from LLMmap.trainer import train_model
from d009_lib import (load_query_embeddings, build_traces, make_loader,
                      hparams_from_shipped, logits_for)
from d007_lib import load_corpus

DEV = "cuda" if torch.cuda.is_available() else "cpu"
d9 = json.load(open("results/D009/runs.json"))
run = next(r for r in d9["runs"] if r["k"] == 8 and r["condition"] == "paper8"
           and r["run"] == 1 and not r["error"])          # a seed with NO checkpoint
print(f"target: paper8 k=8 seed {run['run']}, stored mean_top1 {run['mean_top1']:.6f}")

models = json.load(open("results/D008/selection.json"))["models"]
qe = load_query_embeddings()
cb, cv, ct = (load_corpus(pool=p) for p in ("build", "val", "test"))
tr_b, y_b, _, _ = build_traces(run["queries"], "build", qe, cb)
tr_v, y_v, _, _ = build_traces(run["queries"], "val", qe, cv)
tr_t, y_t, _, _ = build_traces(run["queries"], "test", qe, ct)
hp, conf = hparams_from_shipped(8, len(models))
conf = dict(conf); conf["inference_model"] = hp
h = hashlib.sha256(json.dumps({k: v for k, v in hp.items() if k != "num_queries"},
                              sort_keys=True).encode()).hexdigest()
assert h == d9["hparams_hash"], (h, d9["hparams_hash"])

pl.seed_everything(run["run"], workers=True)
tmp = tempfile.mkdtemp(prefix="d016_cost_")
buf = io.StringIO()
t = time.time()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    _, model = train_model(tmp, siamese=False,
                           loader_train=make_loader(tr_b, y_b, conf["batch_size"], True),
                           loader_test=make_loader(tr_v, y_v, conf["batch_size"], False),
                           conf=conf)
wall = time.time() - t
shutil.rmtree(tmp, ignore_errors=True)
lg = logits_for(model, tr_t, DEV)
acc = float((lg.argmax(1) == y_t).mean())
print(f"\nretrain wall           : {wall:.1f} s")
print(f"stored  mean_top1      : {run['mean_top1']:.6f}")
print(f"retrained mean_top1    : {acc:.6f}")
print(f"|diff|                 : {abs(acc-run['mean_top1']):.6f}")
print(f"deterministic          : {abs(acc-run['mean_top1']) < 1e-9}")
print(f"\n12 missing runs x {wall:.1f}s = {12*wall/60:.1f} min")
json.dump(dict(target="paper8 k=8 seed 1 (no checkpoint saved)",
               retrain_wall_s=round(wall, 1), stored=run["mean_top1"],
               retrained=acc, abs_diff=abs(acc - run["mean_top1"]),
               deterministic=bool(abs(acc - run["mean_top1"]) < 1e-9),
               projected_12_runs_min=round(12 * wall / 60, 1),
               note="measurement for Call 1 only; no part of S1-S5 computed"),
          open("results/D016/call1_cost_measurement.json", "w"), indent=1)
