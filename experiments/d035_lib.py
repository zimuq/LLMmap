"""D035 — fingerprint-definition ablation on corpus v2 (D024 on v2), selection rule held fixed (JointGreedy).

  CLOUD          75 S_build embeddings per (model, query); scale-free joint energy. = D029's v2 JG chain and
                 counts, verbatim; re-selected here ONLY for the code-path check (d024_lib.greedy_cvar +
                 cloud_hooks must reproduce it, per-step objective diff 0).
  CENTROID-raw   per-(model, query) mean of the 75 embeddings; statistic sqrt(sum_q ||mu_vq - mu_v'q||^2).
  CENTROID-norm  the same with each d2_q divided by its median over the 3,570 pairs (D024's primary definition).
  SINGLE         one S_build reference config per model (D024 tier rule, greedy = temperature 0); the same
                 statistic on single points, in CENTROID*'s scaling (chosen on S_val).
  SINGLE-field   D024's literal "lowest temperature field" reading; selection only, never trained.

Every new arm goes through d024_lib.greedy_cvar (CVaR_0.1 over all 3,570 pairs, m = 357, nested k = 1..8, strict
`>` over ascending ids, so ties go to the lowest id).

I3: CENTROID and SINGLE deliberately collapse the point cloud. They exist ONLY as ablation arms of D024 / D035
(human-approved 2026-11-01) and must not be reused as a selection statistic elsewhere.
"""
import json

import numpy as np

import d031_lib as L31
import d029_lib as L29
from d024_lib import single_config, point_d2

OUT = "./results/D035"
CORPUS = "./data/corpus_v2"
VIEW = "v2"
K = 8
GAMMA = 0.1
NCFG_BUILD = 75
SCALINGS = ("raw", "norm")


def models():
    ms = L31.models(VIEW)
    assert ms == json.load(open("./results/D029/chains.json"))["models"], "model order differs from D029"
    return ms


def build_cubes(ms):
    """{model: (259, 75, 1024) fp16 S_build cube}, plus each model's config axis (sorted config ids)."""
    X, axis = {}, {}
    for m in ms:
        cube, cfgs = L29._cube(L29.POOL_EMB, m.replace("/", "__"), "build", L29.NPOOL)
        X[m], axis[m] = cube, cfgs
    return X, axis


def build_prompt_confs(ms, axis):
    """Per model {config_index (int): prompt_conf} for the S_build pool of corpus v2. Asserts that every row of one
    config carries the same prompt_conf, and that the config ids equal the embedding cube's config axis, so cube
    column j is config axis[m][j]."""
    out = {}
    for m in ms:
        pc = {}
        with open(f"{CORPUS}/{m.replace('/', '__')}.jsonl") as fh:
            for line in fh:
                r = json.loads(line)
                if r["dataset"] != "build":
                    continue
                c = int(r["config_index"])
                if c in pc:
                    assert pc[c] == r["prompt_conf"], (m, c, "prompt_conf differs between rows of one config")
                else:
                    pc[c] = r["prompt_conf"]
        assert sorted(pc) == sorted(axis[m]) and len(pc) == NCFG_BUILD, (m, len(pc))
        out[m] = pc
    return out


def single_choice(pcs, ms, reading):
    """D024's tier rule (d024_lib.single_config) per model; None where no tier matches (an S0 stop condition)."""
    return {m: single_config(pcs[m], reading) for m in ms}


def centroid_points(X, ms):
    return np.stack([np.asarray(X[m], np.float64).mean(axis=1) for m in ms])          # (n, 259, 1024)


def single_points(X, ms, axis, cfg):
    """cfg[m] = config_index; the cube column is its position on the model's config axis."""
    return np.stack([np.asarray(X[m][:, axis[m].index(cfg[m]), :], np.float64) for m in ms])


def d2_chunked(P, ia, ib, chunk=32):
    """point_d2 over query chunks (the full (3570, 259, 1024) float64 difference is 7.6 GB)."""
    out = np.empty((P.shape[1], len(ia)))
    for s in range(0, P.shape[1], chunk):
        out[s:s + chunk] = point_d2(P[:, s:s + chunk], ia, ib)
    return out


def normalise(D2):
    """D024's per-query median normalisation. Returns (D2 / median, median); median 0 -> NaN row (reported in S0
    as a stop condition rather than asserted, so S0 still writes its diagnostics)."""
    med = np.median(D2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        N = np.where(med[:, None] > 0, D2 / med[:, None], np.nan)
    return N, med
