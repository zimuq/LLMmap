"""D016 / S1, S2, S4 — trained two-logit accuracy over all 666 pairs, and its
correlation with the tensor proxy.

S1 recovers the 37-way test logits for all 15 runs (3 conditions x 5 seeds at
k=8). Three come from checkpoints D009/D010 saved; the other 12 are retrained,
approved as Call 1 after the one-run measurement showed retraining is BIT-EXACT
(paper8 k=8 seed 1: stored 0.838919, retrained 0.838919, 9.8 s). A retrained seed
is therefore *the* seed the source D reported, not a new draw. All 15 checkpoints
are saved this time (Call 1b).

S2 restricts those logits to each of the 666 model pairs -- the same operation
`d009_lib.logit_stats` already does for the 65 structural pairs, with the wider
pair list. The per-pair config bootstrap follows D015's convention exactly; the
(666, 25) count array does not exist in any stored artifact and is rebuilt here
(P1/F3, approved).

TIE HANDLING (review amendment, 2026-09-23). The proxy's rank-10 cut sits inside a
7-way tie, so "worst 10" is not a principled cutoff. This reports the worst 12 AND
every pair tied with the 12th, states the tie-break (ascending accuracy, then pair
index -- fixed and reproducible), and never implies the cut is meaningful.

Usage:  PYTHONPATH=.:experiments python experiments/d016_full_pairs.py
"""
import os
import csv
import json
import time
import hashlib
import io
import contextlib
import glob
import shutil
import tempfile

import numpy as np
import torch
import pytorch_lightning as pl

from LLMmap.trainer import train_model
from LLMmap.inference_model_archs import InferenceModelLLMmap
from d008_lib import (pair_index, near_relative_pairs, boot_draws, build_dq,
                      two_way_correct)
from d009_lib import (load_query_embeddings, build_traces, make_loader,
                      hparams_from_shipped, logits_for)
from d007_lib import load_corpus

OUT = "./results/D016"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K, N_SEEDS, N_BOOT, NCFG = 8, 5, 2000, 25
COND = {"paper8": ("D009", "results/D009/runs.json", "paper8"),
        "coverage": ("D009", "results/D009/runs.json", "cvar_max"),
        "joint_energy": ("D010", "results/D010/metrics_by_k.json", "joint_energy")}
SMOKE = bool(os.environ.get("D016_SMOKE"))
if SMOKE:
    N_SEEDS, N_BOOT = 2, 200


def get_logits(cname, where, key, run, qe, cubes, n_models, hash_ref):
    """Checkpoint if one was saved, otherwise retrain (Call 1, approved)."""
    cache = f"{OUT}/logits_{cname}_k{K}_r{run['run']}.npy"
    if os.path.exists(cache):
        return np.load(cache), "cached"
    path = f"./results/{where}/models/{key}_k{K}_r{run['run']}.ckpt"
    tr_t, y_t, _, _ = build_traces(run["queries"], "test", qe, cubes["test"])
    hp, conf = hparams_from_shipped(K, n_models)
    conf = dict(conf); conf["inference_model"] = hp
    h = hashlib.sha256(json.dumps(
        {kk: vv for kk, vv in hp.items() if kk != "num_queries"},
        sort_keys=True).encode()).hexdigest()
    assert h == hash_ref, (h, hash_ref)

    if os.path.exists(path):
        net = InferenceModelLLMmap(hp)
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("state_dict", sd)
        sd = {kk[len("net."):] if kk.startswith("net.") else kk: vv
              for kk, vv in sd.items()}
        net.load_state_dict(sd)
        src = "checkpoint"
    else:
        tr_b, y_b, _, _ = build_traces(run["queries"], "build", qe, cubes["build"])
        tr_v, y_v, _, _ = build_traces(run["queries"], "val", qe, cubes["val"])
        pl.seed_everything(run["run"], workers=True)
        tmp = tempfile.mkdtemp(prefix="d016_")
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            _, net = train_model(tmp, siamese=False,
                                 loader_train=make_loader(tr_b, y_b,
                                                          conf["batch_size"], True),
                                 loader_test=make_loader(tr_v, y_v,
                                                         conf["batch_size"], False),
                                 conf=conf)
        best = glob.glob(os.path.join(tmp, "best-*.ckpt"))
        if best:                                    # Call 1b: save it this time
            os.makedirs(f"{OUT}/models", exist_ok=True)
            shutil.copy(best[0], f"{OUT}/models/{cname}_k{K}_r{run['run']}.ckpt")
        shutil.rmtree(tmp, ignore_errors=True)
        src = "retrained"
    lg = logits_for(net, tr_t, DEV)
    np.save(cache, lg)
    return lg, src


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    models = json.load(open("./results/D008/selection.json"))["models"]
    n_models = len(models)
    pairs = pair_index(models)
    hard = near_relative_pairs(models)
    structural = {" | ".join(hard[i]["pair"]) for i in hard}
    meta = {r["model"]: r for r in csv.DictReader(
        open("./results/D001/model_metadata.csv"))}
    hash_ref = json.load(open("./results/D009/runs.json"))["hparams_hash"]
    qe = load_query_embeddings()
    cubes = {p: load_corpus(pool=p) for p in ("build", "val", "test")}
    _, _, y_ev, _, _, cfg_ev = build_dq(eval_pool="test")
    M = boot_draws(NCFG, N_BOOT)
    print(f"{n_models} models, {len(pairs)} pairs, {len(structural)} structural",
          flush=True)

    # ---------------- S1
    logits, prov = {}, {}
    for cname, (where, mfile, key) in COND.items():
        src = json.load(open(mfile))
        runs = sorted([r for r in src["runs"] if r["k"] == K and not r["error"]
                       and r.get("condition") == key],
                      key=lambda r: r["run"])[:N_SEEDS]
        logits[cname], prov[cname] = [], []
        for r in runs:
            lg, how = get_logits(cname, where, key, r, qe, cubes, n_models, hash_ref)
            acc = float((lg.argmax(1) == y_ev).mean())
            ok = abs(acc - r["mean_top1"]) < 1e-9
            assert ok, (cname, r["run"], acc, r["mean_top1"])
            logits[cname].append(lg)
            prov[cname].append(dict(run=r["run"], source=how,
                                    stored=r["mean_top1"], recovered=acc,
                                    reproduces_exactly=bool(ok)))
        print(f"[S1] {cname:13s} {len(runs)} runs "
              f"({sum(1 for p in prov[cname] if p['source']=='retrained')} retrained)"
              f" — all reproduce stored mean_top1 exactly", flush=True)

    # ---------------- S2
    rows = {}
    for cname in COND:
        cnt = np.zeros((N_SEEDS, len(pairs), NCFG))
        for s, lg in enumerate(logits[cname]):
            for j, (a, b) in enumerate(pairs):
                ia, ib = models.index(a), models.index(b)
                m = (y_ev == ia) | (y_ev == ib)
                two = lg[m][:, [ia, ib]]
                p2 = np.where(two.argmax(1) == 0, ia, ib)
                okv = p2 == y_ev[m]
                cnt[s, j] = np.bincount(cfg_ev[m][okv], minlength=NCFG)
        mean = cnt.sum(axis=2).mean(axis=0) / (2.0 * NCFG)
        tot = M.sum(1)
        draws = np.mean([(cnt[s] @ M.T) / (2.0 * tot) for s in range(N_SEEDS)],
                        axis=0)
        rows[cname] = (mean, draws)
        print(f"[S2] {cname:13s} 666-pair mean {mean.mean():.4f} | "
              f"structural {mean[[i for i,(a,b) in enumerate(pairs) if f'{a} | {b}' in structural]].mean():.4f} | "
              f"min {mean.min():.4f}", flush=True)

    out = []
    for j, (a, b) in enumerate(pairs):
        nm = f"{a} | {b}"
        la, lb = meta[a]["lineage"], meta[b]["lineage"]
        r = dict(pair=nm, index=j, structural=nm in structural,
                 same_lineage=bool(la == lb), lineage_a=la, lineage_b=lb,
                 same_base=bool(meta[a]["base"] and
                                meta[a]["base"] == meta[b]["base"]))
        for cname in COND:
            mean, draws = rows[cname]
            r[cname] = round(float(mean[j]), 4)
            r[cname + "_ci"] = [round(float(np.percentile(draws[j], 2.5)), 4),
                                round(float(np.percentile(draws[j], 97.5)), 4)]
        r["min_over_methods"] = round(min(r[c] for c in COND), 4)
        out.append(r)

    # worst list with the tie made explicit (review amendment)
    srt = sorted(out, key=lambda r: (r["min_over_methods"], r["index"]))
    cut = srt[11]["min_over_methods"]
    worst = [r for r in srt if r["min_over_methods"] <= cut]
    n_ns = sum(1 for r in worst if not r["structural"])
    print(f"\n[S2] worst {len(worst)} pairs (12 + everything tied with the 12th "
          f"at {cut:.3f}): {n_ns} non-structural", flush=True)
    for r in worst:
        print(f"   {r['min_over_methods']:.3f} {'S' if r['structural'] else '-'} "
              f"{'sameLin' if r['same_lineage'] else 'xFamily':8s} {r['pair'][:60]}",
              flush=True)

    json.dump(dict(schema="d016-full-pair-v1", k=K, n_pairs=len(pairs),
                   n_seeds=N_SEEDS, n_boot=N_BOOT, chance=0.5,
                   provenance=prov,
                   tie_break="ascending min_over_methods, then pair index -- fixed "
                             "and reproducible. The worst-N cut is NOT a principled "
                             "threshold: it reports 12 plus every pair tied with "
                             "the 12th (review amendment 2026-09-23).",
                   worst_cut_value=cut, n_worst=len(worst),
                   n_worst_non_structural=n_ns,
                   rows=out), open(f"{OUT}/full_pair_trained_accuracy.json", "w"),
              indent=1)

    # ---------------- S4
    px = {r["pair"]: r for r in json.load(
        open(f"{OUT}/full_pair_tensor_proxy.json"))["rows"]}
    from scipy.stats import spearmanr
    corr = {}
    rng = np.random.default_rng(20260924)
    for cname in COND:
        tr = np.array([r[cname] for r in out])
        pr = np.array([px[r["pair"]]["median_query"] for r in out])
        rho = float(spearmanr(tr, pr).statistic)
        bs = [float(spearmanr(tr[i], pr[i]).statistic)
              for i in (rng.integers(0, len(tr), len(tr)) for _ in range(500))]
        st = np.array([r["structural"] for r in out])
        corr[cname] = dict(
            spearman_all=round(rho, 4),
            ci=[round(float(np.percentile(bs, 2.5)), 4),
                round(float(np.percentile(bs, 97.5)), 4)],
            spearman_structural=round(float(spearmanr(tr[st], pr[st]).statistic), 4),
            spearman_non_structural=round(
                float(spearmanr(tr[~st], pr[~st]).statistic), 4),
            n=len(tr))
        print(f"[S4] {cname:13s} Spearman(trained, proxy) over 666: {rho:+.4f} "
              f"{corr[cname]['ci']} | structural {corr[cname]['spearman_structural']:+.4f}"
              f" | non-structural {corr[cname]['spearman_non_structural']:+.4f}",
              flush=True)
    json.dump(dict(schema="d016-correlation-v1", n_pairs=len(out), n_boot=500,
                   proxy_statistic="median_query (D008 1-NN proxy, averaged over "
                                   "the 259-query pool)",
                   caveats=dict(
                       n65_to_666="RESOLVED -- now all 666 pairs",
                       no_ci="RESOLVED -- bootstrap CI reported",
                       accuracy_not_objective="STILL OPEN -- this correlates two "
                                              "ACCURACIES; neither is the literal "
                                              "CVaR-tail objective value"),
                   correlations=corr),
              open(f"{OUT}/full_pair_correlation.json", "w"), indent=1)

    # cross-D check: the 65 structural rows must match D015's
    d15 = {r["pair"]: r for r in json.load(
        open("./results/D015/per_pair_k8.json"))["rows"]}
    diffs = []
    for r in out:
        if r["pair"] in d15:
            for c, c15 in (("paper8", "paper8"), ("coverage", "coverage"),
                           ("joint_energy", "joint_energy")):
                d = abs(r[c] - d15[r["pair"]][c15])
                if d > 1e-9:
                    diffs.append((r["pair"], c, r[c], d15[r["pair"]][c15]))
    # only meaningful at the full seed count -- D015 averaged 5 seeds
    verdict = ("IDENTICAL" if not diffs else
               f"{len(diffs)} differ (EXPECTED: this run used {N_SEEDS} seeds, "
               f"D015 used 5)" if N_SEEDS != 5 else f"{len(diffs)} MISMATCH")
    print(f"\n[check] 65 structural rows vs D015: {verdict}", flush=True)
    if diffs and N_SEEDS == 5:
        for d in diffs[:5]:
            print("   ", d, flush=True)
    print(f"\nwall {(time.time()-t0)/60:.1f} min; written: {OUT}/", flush=True)


if __name__ == "__main__":
    main()
