"""D033 / S2(1) — readout (1): LLMmap's open-set siamese encoder, retrained per fold (the D009 S8 path).

  <view> <kind> <shard> <nshard>    run shard `shard` of `nshard` of the view x kind job list.

Job list (chains.json): per fold, arms OURS, PAPER8, FULL, JG minus arms that share an identical ordered chain
(Disclosure 6), x seeds (model-level 0-2; family-level 0 only, Review Call 3 (a)); k = 8 (Call 2 (a)).
Per run: S_build / S_val traces of the fold's TRAINING models only (asserted) -> TracePairDataset (500k / 5k pairs)
-> train_model(siamese=True), confs/default.json hparams; then the shipped infer_features / build_templates on the
S_build traces of ALL enrolled models; S_val and S_test features of all models; Euclidean and cosine distance of
every target to every template. Written to results/D033/feat/<run>.npz (gitignored), checkpoint to
results/D033/models/<run>.ckpt; A1 diagnostics (per-epoch S_val loss / pair accuracy from Lightning's CSV log,
best epoch, epochs run) to results/D033/runs/<run>.json. S_test is written, never printed or summarised.
A stream starts no new run after D033_GUARD_MIN (default 85) minutes; a run whose json exists is skipped.
Usage:  PYTHONPATH=.:experiments python experiments/d033_enc.py v2 model 0 4
"""
import io
import os
import sys
import csv
import glob
import json
import time
import shutil
import tempfile
import contextlib

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist

import d031_lib as L31
import d033_lib as L
from d009_lib import build_traces, hparams_from_shipped, TracePairDataset, load_query_embeddings, make_loader
from LLMmap.trainer import train_model
from LLMmap.templates import infer_features, build_templates

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GUARD = float(os.environ.get("D033_GUARD_MIN", "85"))


def run_name(view, kind, fold, arm, seed):
    return f"{view}_{kind}_f{fold}_{arm}_s{seed}"


def jobs(ch, view, kind):
    seeds = L.SEEDS if kind == "model" else (0,)
    out = []
    for seed in seeds:
        for F in ch["views"][view]["kinds"][kind]["folds"]:
            for arm in L.ARMS:
                if arm in F["shared_training"]:
                    continue
                out.append(dict(view=view, kind=kind, fold=F["fold"], arm=arm, seed=seed, q=F["arms"][arm],
                                train=F["train"], name=run_name(view, kind, F["fold"], arm, seed)))
    return out


def epoch_log(tmp):
    f = glob.glob(os.path.join(tmp, "lightning_logs", "version_*", "metrics.csv"))
    if not f:
        return []
    per = {}
    for r in csv.DictReader(open(f[0])):
        e = int(r["epoch"])
        d = per.setdefault(e, {"epoch": e})
        for k in ("val_loss", "val_acc", "train_acc"):
            if r.get(k):
                d[k] = float(r[k])
    return [per[e] for e in sorted(per)]


def one(j, ctx):
    t0 = time.time()
    ms, n = ctx["models"], len(ctx["models"])
    q = j["q"]
    tr = {s: build_traces(q, s, ctx["qe"], ctx["cubes"][s]) for s in ("build", "val", "test")}
    keep = np.asarray(j["train"])
    remap = -np.ones(n, int)
    remap[keep] = np.arange(len(keep))
    (tb, yb, _, _), (tv, yv, _, _) = tr["build"], tr["val"]
    mb, mv = np.isin(yb, keep), np.isin(yv, keep)
    # held-out exclusion (D033 invariant): the encoder sees training models' S_build / S_val traces only
    assert sorted(set(yb[mb].tolist())) == sorted(keep.tolist()) == sorted(set(yv[mv].tolist()))
    ho = set(range(n)) - set(keep.tolist())
    assert not (set(yb[mb].tolist()) | set(yv[mv].tolist())) & ho
    tbx, ybx, tvx, yvx = tb[mb], remap[yb[mb]], tv[mv], remap[yv[mv]]
    nt = len(keep)
    hp, conf = hparams_from_shipped(len(q), nt)
    conf = dict(conf)
    conf["inference_model"] = hp
    pl.seed_everything(j["seed"], workers=True)
    ltr = DataLoader(TracePairDataset(tbx, ybx, conf["num_pairs_per_epoch"], nt), batch_size=conf["batch_size"],
                     shuffle=False)
    lva = DataLoader(TracePairDataset(tvx, yvx, conf["num_pairs_per_eval"], nt), batch_size=conf["batch_size"],
                     shuffle=False)
    tmp = tempfile.mkdtemp(prefix="d033_enc_")
    buf = io.StringIO()
    t1 = time.time()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        _, feat = train_model(tmp, siamese=True, loader_train=ltr, loader_test=lva, conf=conf)
    wall_train = time.time() - t1
    best = glob.glob(os.path.join(tmp, "best-*.ckpt"))[0]
    ck = torch.load(best, map_location="cpu", weights_only=False)
    ep = epoch_log(tmp)
    shutil.copy(best, f"{L.OUT}/models/{j['name']}.ckpt")
    shutil.rmtree(tmp, ignore_errors=True)

    F = {}
    for s in ("build", "val", "test"):
        t, y, c, _ = tr[s]
        yy, ff = infer_features(feat, make_loader(t, y, 256, False), DEV)
        assert np.array_equal(yy, y)
        F[s] = ff.astype(np.float32)
    T = build_templates(tr["build"][1], F["build"]).astype(np.float64)       # shipped per-model mean (float32)
    assert T.shape == (n, F["build"].shape[1]) and np.isfinite(T).all()
    out = dict(feat_build=F["build"], feat_val=F["val"], feat_test=F["test"], templates=T,
               y_val=tr["val"][1], c_val=tr["val"][2], y_test=tr["test"][1], c_test=tr["test"][2])
    for s in ("val", "test"):
        for metric, nm in (("euclidean", "euc"), ("cosine", "cos")):
            out[f"D_{s}_{nm}"] = cdist(F[s].astype(np.float64), T, metric=metric).astype(np.float32)
    np.savez_compressed(f"{L.OUT}/feat/{j['name']}.npz", **out)

    best_epoch = int(ck["epoch"])
    vl = [e.get("val_loss") for e in ep]
    info = dict(name=j["name"], view=j["view"], kind=j["kind"], fold=j["fold"], arm=j["arm"], seed=j["seed"],
                queries=q, n_train=nt, train_models=[ms[i] for i in keep], best_ckpt=os.path.basename(best),
                best_epoch=best_epoch, epochs_run=len(ep), epoch_log=ep,
                s_val_pair_acc_at_best=next((e.get("val_acc") for e in ep if e["epoch"] == best_epoch), None),
                no_improvement_after_first_epoch=bool(len(vl) > 0 and vl[0] is not None
                                                      and min(v for v in vl if v is not None) >= vl[0]),
                wall_train_min=round(wall_train / 60, 2), wall_min=round((time.time() - t0) / 60, 2),
                hp_conf="confs/default.json", hp_sha256=L31.sha("./confs/default.json"),
                ckpt_sha256=L31.sha(f"{L.OUT}/models/{j['name']}.ckpt"),
                heldout_exclusion_assert=True, node=os.uname().nodename)
    L.jdump(info, f"{L.OUT}/runs/{j['name']}.json")
    return info


def main():
    view, kind, shard, nshard = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    t0 = time.time()
    assert L.committed(f"{L.OUT}/variant_choice.json"), "variant_choice.json must be committed before S2"
    for d in ("models", "feat", "runs"):
        os.makedirs(f"{L.OUT}/{d}", exist_ok=True)
    ch = json.load(open(f"{L.OUT}/chains.json"))
    todo = [j for j in jobs(ch, view, kind)[shard::nshard] if not os.path.exists(f"{L.OUT}/runs/{j['name']}.json")]
    print(f"[S2 enc {view} {kind} {shard}/{nshard}] {len(todo)} runs to do", flush=True)
    if not todo:
        return
    ctx = dict(models=L31.models(view), qe=load_query_embeddings(), cubes=L31.cubes(view, ("build", "val", "test")))
    for i, j in enumerate(todo):
        el = (time.time() - t0) / 60
        if el > GUARD:
            print(f"[S2 enc {view} {kind} {shard}/{nshard}] guard: {el:.0f} min elapsed, {len(todo) - i} runs left "
                  f"for the next job", flush=True)
            break
        info = one(j, ctx)
        print(f"[S2 enc] {j['name']}: best epoch {info['best_epoch']}, {info['epochs_run']} epochs, S_val pair acc "
              f"{info['s_val_pair_acc_at_best']}, train {info['wall_train_min']} min"
              f"{'  NO IMPROVEMENT AFTER EPOCH 1' if info['no_improvement_after_first_epoch'] else ''}", flush=True)
    print(f"[S2 enc {view} {kind} {shard}/{nshard}] stream done ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
