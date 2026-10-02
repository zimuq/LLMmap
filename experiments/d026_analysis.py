"""D026 / S2 — two-way probe-pilot analysis (P1 S2, Review amendments).

Readouts (fixed): concat LR C=1.0 (primary) and meanpool LR C=3.0 (row f);
lbfgs, max_iter 20000, tol 1e-6. Features = response embeddings only (I5,
tok200, unnormalised), slots in selection order. Train S_build (75 cfg x 2),
select on S_val log-loss (greedy forward, nested to k=8, exact ties -> lowest
id), evaluate S_test once per (set, k). Bootstrap: boot_draws(25, 2000,
seed=20261001), shared by every arm.

Candidate sets: POOL (0..258), NEW (F1+F2: 259..272), POOL+NEW,
POOL+NEW+F3 (+273..280; secondary (c) only).

Usage:  PYTHONPATH=.:experiments python experiments/d026_analysis.py
"""
import os
import json
import time

import numpy as np
import multiprocessing as mp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from d008_lib import boot_draws
import d026_lib as L

OUT = L.OUT
EXT = "./data/corpus_v1_ext_d026"
SPLITS = ("build", "val", "test")
READOUTS = {"concat": 1.0, "meanpool": 3.0}
MAX_ITER, TOL, K, M_MARGIN = 20000, 1e-6, 8, 0.05
POOL = list(range(259))
NEW = L.SHORT_IDS
F3 = list(range(273, 281))
SETS = {"POOL": POOL, "NEW": NEW, "POOL+NEW": POOL + NEW, "POOL+NEW+F3": POOL + NEW + F3}
D025_CHAINS = {"GLOBAL": [193, 140, 237, 114, 233, 117, 0, 16],
               "FAM:Phi": [9, 140, 65, 145, 18, 56, 44, 131]}
N_JOBS = int(os.environ.get("D026_JOBS", 64))


def load_cube(m, emb_dir, nq, id0=0):
    """{split: (nq, ncfg, 1024) float32} with configs in sorted index order."""
    a = np.load(f"{emb_dir}/{L.slug(m)}.npy", mmap_mode="r")
    rows = json.load(open(f"{emb_dir}/{L.slug(m)}.index.json"))["rows"]
    out = {}
    for s in SPLITS:
        sel = [(i, r["query_index"], r["config"]) for i, r in enumerate(rows) if r["pool"] == s]
        cfgs = sorted({c for _, _, c in sel})
        ci = {c: j for j, c in enumerate(cfgs)}
        cube = np.empty((nq, len(cfgs), a.shape[1]), np.float32)
        for i, q, c in sel:
            cube[q, ci[c]] = a[i]
        out[s] = (cube, cfgs)
    return out


def load_all():
    X = {}
    for m in L.TWINS:
        p = load_cube(m, f"{L.CORPUS_DIR}/embeddings", 259)
        n = load_cube(m, f"{EXT}/embeddings", 22)
        X[m] = {}
        for s in SPLITS:
            assert p[s][1] == n[s][1], "config sets differ between pool and extension"
            X[m][s] = np.concatenate([p[s][0], n[s][0]], axis=0)     # (281, ncfg, 1024)
    return X


def feats(X, S, split, readout):
    """(2*ncfg, d), y, cfg."""
    parts, ys, cs = [], [], []
    for yi, m in enumerate(L.TWINS):
        A = X[m][split][S]                                          # (k, ncfg, 1024)
        F = A.transpose(1, 0, 2).reshape(A.shape[1], -1) if readout == "concat" else A.mean(0)
        parts.append(F)
        ys.append(np.full(A.shape[1], yi))
        cs.append(np.arange(A.shape[1]))
    return np.vstack(parts), np.concatenate(ys), np.concatenate(cs)


def fit(X, S, readout):
    Xtr, ytr, _ = feats(X, S, "build", readout)
    clf = LogisticRegression(C=READOUTS[readout], solver="lbfgs", max_iter=MAX_ITER, tol=TOL)
    clf.fit(Xtr, ytr)
    return clf, int(clf.n_iter_.max())


def val_loss(X, S, readout):
    clf, it = fit(X, S, readout)
    Xv, yv, _ = feats(X, S, "val", readout)
    return log_loss(yv, clf.predict_proba(Xv), labels=[0, 1]), it


def evaluate(X, S, readout):
    clf, it = fit(X, S, readout)
    r = dict(queries=list(map(int, S)), n_iter=it)
    for s in ("val", "test"):
        Xs, ys, cs = feats(X, S, s, readout)
        pred = (clf.decision_function(Xs) > 0).astype(int)
        ok = pred == ys
        r[f"acc_{s}"] = float(ok.mean())
        r[f"cnt_{s}"] = np.bincount(cs[ok], minlength=cs.max() + 1).tolist()
        if s == "val":
            r["logloss_val"] = float(log_loss(ys, clf.predict_proba(Xs), labels=[0, 1]))
    return r


_X = None          # set before the fork-based pool is created (copy-on-write)
_POOL = None


def _vl(args):
    S, readout = args
    return val_loss(_X, S, readout)


def greedy(X, cands, readout):
    S, trace, nonconv = [], [], 0
    for step in range(K):
        rest = [q for q in cands if q not in S]
        res = _POOL.map(_vl, [(S + [q], readout) for q in rest], chunksize=1)
        nonconv += sum(it >= MAX_ITER for _, it in res)
        best = min(zip([l for l, _ in res], rest))
        S.append(int(best[1]))
        trace.append(dict(k=step + 1, q=int(best[1]), val_logloss=float(best[0]),
                          n_tied=int(sum(l == best[0] for l, _ in res))))
    return S, trace, nonconv


def boot(cnt, Mb):
    c = np.asarray(cnt, float)
    return (Mb @ c) / (2 * Mb.sum(1))


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def label(acc_pool4, d, lo, hi):
    if acc_pool4 > 0.95:
        return "CEILING"
    if d >= M_MARGIN and lo > 0:
        return "GO"
    if hi < M_MARGIN:
        return "NO-GO"
    return "INCONCLUSIVE"


def main():
    t0 = time.time()
    global _X, _POOL
    X = load_all()
    _X = X
    _POOL = mp.get_context("fork").Pool(N_JOBS)
    Mb = boot_draws(25, 2000, seed=20261001)
    out = dict(schema="d026-analysis-v1", readouts=READOUTS, boot_seed=20261001, sets={}, wall={})
    for ro in READOUTS:
        out["sets"][ro] = {}
        for name, cands in SETS.items():
            t = time.time()
            S, trace, nc = greedy(X, cands, ro)
            curve = [evaluate(X, S[:k], ro) for k in range(1, K + 1)]
            for r in curve:
                b = boot(r["cnt_test"], Mb)
                r["ci_test"] = ci(b)
                r["optimism"] = r["acc_val"] - r["acc_test"]
            out["sets"][ro][name] = dict(chain=S, trace=trace, nonconverged_fits=nc, curve=curve)
            out["wall"][f"{ro}:{name}"] = round(time.time() - t, 1)
            print(f"[{ro}] {name:12s} chain {S}  acc_test k=4 {curve[3]['acc_test']:.3f} "
                  f"k=8 {curve[7]['acc_test']:.3f}  ({out['wall'][f'{ro}:{name}']} s)", flush=True)

        # ---- contrasts (paired, shared draws)
        sets = out["sets"][ro]

        def delta(a, b, k):
            ca, cb = sets[a]["curve"][k - 1]["cnt_test"], sets[b]["curve"][k - 1]["cnt_test"]
            d = boot(ca, Mb) - boot(cb, Mb)
            return dict(delta=sets[a]["curve"][k - 1]["acc_test"] - sets[b]["curve"][k - 1]["acc_test"],
                        ci=ci(d), identical_sets=sorted(sets[a]["chain"][:k]) == sorted(sets[b]["chain"][:k]))
        pr = delta("POOL+NEW", "POOL", 4)
        pr["acc_pool_k4"] = sets["POOL"]["curve"][3]["acc_test"]
        pr["acc_poolnew_k4"] = sets["POOL+NEW"]["curve"][3]["acc_test"]
        pr["new_in_first4"] = [q for q in sets["POOL+NEW"]["chain"][:4] if q >= 259]
        pr["label"] = label(pr["acc_pool_k4"], pr["delta"], *pr["ci"])
        out.setdefault("primary", {})[ro] = pr
        out.setdefault("new_vs_pool", {})[ro] = {k: delta("NEW", "POOL", k) for k in range(1, K + 1)}
        out.setdefault("f3_vs_poolnew_k4", {})[ro] = delta("POOL+NEW+F3", "POOL+NEW", 4)
        out.setdefault("poolnew_vs_pool", {})[ro] = {k: delta("POOL+NEW", "POOL", k) for k in range(1, K + 1)}
        print(f"[{ro}] PRIMARY {pr}", flush=True)

    # ---- (b) per-probe single-query table (concat == meanpool at k=1 up to C; concat reported)
    per = {}
    for q in NEW + F3:
        r = evaluate(X, [q], "concat")
        per[q] = dict(family=L.FAMILY[q], acc_val=r["acc_val"], acc_test=r["acc_test"],
                      logloss_val=r["logloss_val"], ci_test=ci(boot(r["cnt_test"], Mb)))
    # byte-identical twin answers on the greedy reference configs (128k 22 / 4k 56)
    ref = {}
    for m in L.TWINS:
        ent = L.corpus_entries(m)
        c = L.greedy_reference(m, ent)
        for line in open(f"{EXT}/{L.slug(m)}.jsonl"):
            d = json.loads(line)
            if d["dataset"] == "build" and d["config_index"] == c:
                ref[m] = d
    a, b = (ref[m] for m in L.TWINS)
    assert a["prompt_conf"] == b["prompt_conf"]
    for i, ((_, x), (_, y)) in enumerate(zip(a["traces"], b["traces"])):
        per[259 + i]["byte_identical_ref"] = x == y
    out["per_probe"] = per
    out["byte_identical_ref_new"] = sum(per[q]["byte_identical_ref"] for q in per)

    # ---- (c) F3 window-boundary split by wrapped length, output classes
    from LLMmap.llm import LLM_huggingface
    from LLMmap.prompt_configuration import PromptConf
    qdoc = json.load(open("./confs/queries/pool_d026_ext.json"))
    texts = {r["id"]: r["text"] for r in qdoc["queries"]}
    split_rows = {}
    for yi, m in enumerate(L.TWINS):
        tk = LLM_huggingface(m, tokenizer_only=True,
                             model_load_kargs=dict(revision=L.shard_status(m)["hf_revision"]))
        ent = L.corpus_entries(m)
        for c in sorted(c for (p, c) in ent if p == "test"):
            n = len(tk.tokenizer(PromptConf.from_dict(ent[("test", c)]["prompt_conf"])(texts[273], tk)[0],
                                 add_special_tokens=False)["input_ids"])
            split_rows[(m, c)] = n <= 2047
    wb = {}
    for q in (273, 274):
        clf, _ = fit(X, [q], "concat")
        Xs, ys, cs = feats(X, [q], "test", "concat")
        ok = (clf.decision_function(Xs) > 0).astype(int) == ys
        inside = np.array([split_rows[(L.TWINS[yy], sorted(c for (p, c) in L.corpus_entries(L.TWINS[yy]) if p == "test")[cc])]
                           for yy, cc in zip(ys, cs)])
        wb[q] = dict(acc_inside=float(ok[inside].mean()) if inside.any() else None, n_inside=int(inside.sum()),
                     acc_outside=float(ok[~inside].mean()) if (~inside).any() else None,
                     n_outside=int((~inside).sum()))
    out["f3_window_boundary_split_test"] = wb
    out["f3_output_classes"] = {m: json.load(open(f"{EXT}/{L.slug(m)}.status.json"))["f3_output_classes"]
                                for m in L.TWINS}

    # ---- (d) D025 reference chains, re-read two-way
    out["reference_rows"] = {}
    for ro in READOUTS:
        for name, ch in D025_CHAINS.items():
            for k in (4, 8):
                r = evaluate(X, ch[:k], ro)
                r["ci_test"] = ci(boot(r["cnt_test"], Mb))
                out["reference_rows"][f"{ro}:{name}:k{k}"] = {kk: r[kk] for kk in ("queries", "acc_val", "acc_test", "ci_test")}

    # ---- (e) hand-inspection quotes
    def key(q):
        return (per[q]["logloss_val"], -per[q]["acc_val"], q)
    top_short = sorted(NEW, key=key)[:3]
    top_f3 = sorted(F3, key=key)[:1]
    quotes = {}
    for q in top_short + top_f3:
        quotes[q] = {}
        for m in L.TWINS:
            ent = L.corpus_entries(m)
            refc = L.greedy_reference(m, ent)
            sampled = sorted(c for (p, c), d in ent.items() if p == "build"
                             and d["prompt_conf"]["sampling_hparams"]["do_sample"])[:2]
            want = [refc] + sampled
            rows = {}
            for line in open(f"{EXT}/{L.slug(m)}.jsonl"):
                d = json.loads(line)
                if d["dataset"] == "build" and d["config_index"] in want:
                    rows[d["config_index"]] = d["traces"][q - 259][1][:500]
            quotes[q][m] = [dict(config=c, greedy=(c == refc), response=rows[c]) for c in want]
    out["hand_inspection"] = dict(top_short=top_short, top_f3=top_f3, quotes=quotes)
    out["wall_s"] = round(time.time() - t0, 1)
    json.dump(out, open(f"{OUT}/analysis.json", "w"), indent=1)
    print(f"[S2] wrote {OUT}/analysis.json, {out['wall_s']} s", flush=True)


if __name__ == "__main__":
    main()
