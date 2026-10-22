"""D033 / P1 pilot — cost and determinism of readout (1) (LLMmap open-set siamese encoder). Nothing here is
a D033 result; no S_test split is loaded.

  repro           v1, PAPER8, all 37 models, seed 0: the D009 S8 recipe verbatim (d009_openset.py: shipped
                  confs/default.json hparams, d009_lib.TracePairDataset on S_build / S_val, train_model(siamese=True)).
                  Compares the best checkpoint's state_dict with D009's stored
                  results/D009/models/openset_paper8_k8.ckpt (bit-exact or not), and reports wall / epochs.
  v2 <seed>       v2, PAPER8, a fixed 68-model subset (indices 0-67 of universe_v2; NOT a D033 fold, timing only),
                  same recipe. Reports wall / epochs.
Usage:  PYTHONPATH=.:experiments python experiments/d033_pilot.py repro|v2 <seed>
"""
import io
import os
import sys
import glob
import time
import json
import shutil
import tempfile
import contextlib

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import d031_lib as L31
from d009_lib import build_traces, hparams_from_shipped, TracePairDataset, load_query_embeddings
from LLMmap.trainer import train_model

OUT = "./results/D033/pilot"


def run(view, keep, seed):
    t0 = time.time()
    qe = load_query_embeddings()
    cubes = L31.cubes(view, ("build", "val"))
    tr_b, y_b, _, ms = build_traces(L31.PAPER8, "build", qe, cubes["build"])
    tr_v, y_v, _, _ = build_traces(L31.PAPER8, "val", qe, cubes["val"])
    del cubes
    keep = np.asarray(keep)
    remap = -np.ones(len(ms), int)
    remap[keep] = np.arange(len(keep))
    mb, mv = np.isin(y_b, keep), np.isin(y_v, keep)
    tr_b, y_b, tr_v, y_v = tr_b[mb], remap[y_b[mb]], tr_v[mv], remap[y_v[mv]]
    n = len(keep)
    t_load = time.time() - t0
    hp, conf = hparams_from_shipped(8, n)
    conf = dict(conf)
    conf["inference_model"] = hp
    pl.seed_everything(seed, workers=True)
    ltr = DataLoader(TracePairDataset(tr_b, y_b, conf["num_pairs_per_epoch"], n), batch_size=conf["batch_size"], shuffle=False)
    lva = DataLoader(TracePairDataset(tr_v, y_v, conf["num_pairs_per_eval"], n), batch_size=conf["batch_size"], shuffle=False)
    tmp = tempfile.mkdtemp(prefix="d033_pilot_")
    buf = io.StringIO()
    t = time.time()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        _, feat = train_model(tmp, siamese=True, loader_train=ltr, loader_test=lva, conf=conf)
    wall = time.time() - t
    best = glob.glob(os.path.join(tmp, "best-*.ckpt"))[0]
    ck = torch.load(best, map_location="cpu", weights_only=False)
    return dict(view=view, n_models=n, seed=seed, n_build=int(len(y_b)), n_val=int(len(y_v)),
                load_min=round(t_load / 60, 2), train_min=round(wall / 60, 2), best_ckpt=os.path.basename(best),
                last_epoch=int(ck["epoch"]), node=os.uname().nodename,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None), ck, tmp


def main():
    os.makedirs(OUT, exist_ok=True)
    mode = sys.argv[1]
    if mode == "repro":
        info, ck, tmp = run("v1", range(37), 0)
        ref = torch.load("./results/D009/models/openset_paper8_k8.ckpt", map_location="cpu", weights_only=False)
        a, b = ck["state_dict"], ref["state_dict"]
        assert sorted(a) == sorted(b), "state_dict keys differ from D009 S8"
        diff = {k: float((a[k].double() - b[k].double()).abs().max()) for k in a if a[k].is_floating_point()}
        info.update(d009_best_epoch=int(ref["epoch"]), bit_exact=all(torch.equal(a[k], b[k]) for k in a),
                    max_abs_diff=max(diff.values()))
        tag = "repro_v1"
    else:
        seed = int(sys.argv[2])
        info, ck, tmp = run("v2", range(68), seed)
        tag = f"v2_s{seed}"
    shutil.rmtree(tmp, ignore_errors=True)
    json.dump(info, open(f"{OUT}/{tag}.json", "w"), indent=1)
    print(f"[pilot {tag}] {info}", flush=True)


if __name__ == "__main__":
    main()
