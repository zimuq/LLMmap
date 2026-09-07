"""
D007 — shared loading and per-cell statistics for the separability tensor.

A "cell" is one (query, model-pair): model i's build-split embeddings for that
query as one point cloud (75 points, one per build config) against model j's.
Invariant I3: point cloud vs point cloud, never a collapsed average.

Two statistics per A3 (decided 2026-09-06):
  S_probe   5-fold CV AUC of a linear probe -- PRIMARY, bounded
  S_energy  energy distance -- MANDATORY companion, unbounded

Energy distance is reported scale-free for anything aggregated. D007/P1 §F3:
with the corrected UNNORMALISED embeddings (mean ||v|| ~ 19.6, D006/R10) the raw
statistic scales with vector norm, which varies with response length and
therefore across queries -- and I4's MAX runs across queries. Dividing by the
pooled within-cloud mean distance removes that.
"""
import os
import json
import glob

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cdist

CORPUS = os.environ.get("D006_CORPUS", "./data/corpus_v1")
EMB = os.path.join(CORPUS, "embeddings")
PROBE_C = 1.0          # fixed, never tuned per cell (D004's p>>n leakage reason)
N_FOLDS = 5


def load_corpus(pool="build", token_budget=None):
    """Return (models, X) where X[m] is (n_queries, n_configs, dim) for `pool`.

    `token_budget=None` uses the embeddings of the full 200-token responses.
    A truncated variant lives in a parallel directory (see d007_s1b_reembed).
    """
    emb_dir = EMB if token_budget in (None, 200) else f"{EMB}_tok{token_budget}"
    man = json.load(open(os.path.join(CORPUS, "corpus_manifest.json")))
    assert man["status"] == "READY", (
        f"corpus manifest status is {man['status']}, not READY -- D007's hard "
        f"dependency forbids building the tensor on a partial pair set")
    models = sorted(s["model"] for s in man["models"] if s["status"] == "VALIDATED")
    assert len(models) == 37, f"expected 37 models, got {len(models)}"

    X = {}
    for m in models:
        slug = m.replace("/", "__")
        a = np.load(f"{emb_dir}/{slug}.npy", mmap_mode="r")
        rows = json.load(open(f"{emb_dir}/{slug}.index.json"))["rows"]
        sel = [(i, r["query_index"], r["config"]) for i, r in enumerate(rows)
               if r["pool"] == pool]
        nq = max(s[1] for s in sel) + 1
        nc = len({s[2] for s in sel})
        cube = np.empty((nq, nc, a.shape[1]), dtype=np.float32)
        cfg_ix = {c: k for k, c in enumerate(sorted({s[2] for s in sel}))}
        for i, q, c in sel:
            cube[q, cfg_ix[c]] = a[i]
        X[m] = cube
    return models, X


def probe_auc(A, B, seed=0):
    """5-fold CV AUC of a linear probe. A, B are (n, d) point clouds."""
    Z = np.vstack([A, B])
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    s = []
    for tr, te in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(Z, y):
        clf = LogisticRegression(max_iter=2000, C=PROBE_C).fit(Z[tr], y[tr])
        s.append(roc_auc_score(y[te], clf.decision_function(Z[te])))
    return float(np.mean(s))


def energy(A, B):
    """Energy distance, raw and scale-free.

    Scale-free = raw / (mean within-cloud distance), which cancels the vector-
    norm scaling that makes the raw statistic incomparable across queries under
    unnormalised embeddings (P1 F3).
    """
    xy = cdist(A, B).mean()
    xx = cdist(A, A).mean()
    yy = cdist(B, B).mean()
    raw = float(2 * xy - xx - yy)
    within = (xx + yy) / 2.0
    return raw, float(raw / within) if within > 0 else float("nan")
