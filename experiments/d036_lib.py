"""D036 — distribution vs scaling: CLOUD-raw (undivided joint energy) vs CENTROID-raw (D035) vs CLOUD (D029, sf).

  CLOUD-raw      2 E||X-Y|| - E||X-X'|| - E||Y-Y'|| over the running set's concatenated queries, the 75 S_build
                 points per model; the same float32 XY2 / XX2 accumulators as D010 / D029 (diagonal-inclusive
                 V-statistic means). NEW.
  CLOUD          the same numerator / ((E||X-X'|| + E||Y-Y'||) / 2) = D010's scale-free statistic. Selected here
                 only for code-path checks (D029 full chain, D033 JG^fold chains); its counts are reused verbatim.
  CENTROID-raw   sqrt(sum_q ||mu_vq - mu_v'q||^2) = D035's statistic (D035's distances.npz, sha asserted). Full-model
                 chain and counts reused verbatim; ^fold chains new. I3: ablation arm only (D035 confinement).

Every chain: d024_lib.greedy_cvar (CVaR_0.1 over the pair set, nested k = 1..8, strict `>` over ascending ids, so
ties go to the lowest id). Fold chains use only the fold's training models and the pairs among them.
`energy_stat(..., "sf")` is written to be the same float32 operations as LLMmap/joint_greedy._stat_all_pairs.
"""
import json

import numpy as np

import d031_lib as L31
from d024_lib import greedy_cvar, additive_hooks

OUT = "./results/D036"
VIEW = "v2"
K = 8
GAMMA = 0.1
NPOOL = 259
SCALES = ("raw", "sf")


def energy_stat(acc_xy, acc_xxm, ia, ib, add_xy, add_xxm, scale):
    xy2 = acc_xy + add_xy                      # (n_pairs, nc, nc)
    xxm2 = acc_xxm + add_xxm                   # (n_models, nc, nc)
    xy = np.sqrt(xy2).mean(axis=(1, 2))
    xm = np.sqrt(xxm2).mean(axis=(1, 2))
    xx, yy = xm[ia], xm[ib]
    raw = 2.0 * xy - xx - yy
    if scale == "raw":
        return raw
    assert scale == "sf"
    within = (xx + yy) / 2.0
    return np.where(within > 0, raw / np.maximum(within, 1e-12), np.nan)


def energy_hooks(XY2, XX2, ia, ib, scale):
    n_pairs, nc = XY2.shape[1], XY2.shape[2]

    def init():
        return {"xy": np.zeros((n_pairs, nc, nc), np.float32),
                "xx": np.zeros((XX2.shape[1], nc, nc), np.float32)}

    def score(st, q):
        return energy_stat(st["xy"], st["xx"], ia, ib, XY2[q], XX2[q], scale)

    def update(st, q):
        st["xy"] += XY2[q]
        st["xx"] += XX2[q]
    return init, score, update


def cloud_chain(XY2, XX2, ia, ib, scale):
    return greedy_cvar(NPOOL, *energy_hooks(XY2, XX2, ia, ib, scale), k=K, gamma=GAMMA)


def fold_cloud(XY2, XX2, ia, ib, tr, cols, scale):
    """CLOUD(-raw)^fold: models `tr`, pairs `cols` (d033_lib.train_split). Same restriction as d033_lib.jg."""
    pos = -np.ones(int(max(ia.max(), ib.max())) + 1, int)
    pos[tr] = np.arange(len(tr))
    a, b = pos[ia[cols]], pos[ib[cols]]
    assert (a >= 0).all() and (b >= 0).all()
    xy = XY2[:, cols] if len(cols) < XY2.shape[1] else XY2
    xx = XX2[:, tr] if len(tr) < XX2.shape[1] else XX2
    return cloud_chain(xy, xx, a, b, scale)


def fold_centroid(D2, cols):
    """CENTROID-raw^fold: D035's additive statistic on the fold's training pairs."""
    return greedy_cvar(D2.shape[0], *additive_hooks(D2[:, cols]), k=K, gamma=GAMMA)


def centroid_raw():
    """D035's CENTROID|raw per-query squared distances (259, 3570), sha-checked against D035's S0."""
    s0 = json.load(open("./results/D035/s0.json"))
    npz = "./results/D035/distances.npz"
    assert L31.sha(npz) == s0["distances_npz_sha256"], "D035 distances.npz differs from its S0"
    d = np.load(npz)
    ia, ib = L31.pairs(VIEW)
    assert list(d["models"]) == L31.models(VIEW)
    assert np.array_equal(d["ia"], ia) and np.array_equal(d["ib"], ib)
    return np.asarray(d["CENTROID|raw"])


def d033_folds():
    ch = json.load(open("./results/D033/chains.json"))
    return ch["folds"]["model"], ch["views"][VIEW]["kinds"]["model"]["folds"]
