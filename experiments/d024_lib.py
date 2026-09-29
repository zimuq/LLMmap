"""D024 — fingerprint-definition ablation: CLOUD vs CENTROID vs SINGLE.

The selection algorithm is held fixed (greedy, CVaR_0.1 over the 666 pairs of a
set-level statistic, nested k=1..8, ties to the lowest query id). Only the
fingerprint inside the statistic changes:

  CLOUD     75 S_build embeddings per (model, query); scale-free joint energy
            (D010). Existing JointGreedy chain; re-selected here ONLY for the
            code-path check.
  CENTROID  per-(model, query) mean of those 75 embeddings.
            d2_q(v,v') = ||mu_vq - mu_v'q||^2, divided by its median over the
            666 pairs for that query; set statistic sqrt(sum_q d2_q / med_q).
            Sensitivity row: the same without the per-query median.
  SINGLE    one reference config per model (fixed tier rule, D024 `## D`), the
            same normalised statistic on single points.

CENTROID and SINGLE deliberately collapse the point cloud (I3) -- they exist ONLY
as ablation arms of D024 (human-approved) and must not be reused as a selection
statistic elsewhere.

All three arms go through ONE greedy wrapper (`greedy_cvar`), so that the
code-path check (wrapper + CLOUD statistic == D010's chain) certifies the
wrapper the new arms use.
"""
import os
import json
import itertools

import numpy as np

from LLMmap.joint_statistic import cvar
from LLMmap.joint_greedy import _stat_all_pairs

CORPUS = "./data/corpus_v1"
EMB = f"{CORPUS}/embeddings"
OUT = "./results/D024"
GAMMA = 0.1
K = 8
HELPFUL = "You are a helpful assistant."


# ------------------------------------------------------------------ configs
def build_prompt_confs(models):
    """Per model, {config_index: prompt_conf} for the build pool. Asserts the
    embedding cube's config axis (load_corpus: sorted config ids) is the same
    0..74 index as the corpus `config_index`."""
    out = {}
    for m in models:
        slug = m.replace("/", "__")
        rows = json.load(open(f"{EMB}/{slug}.index.json"))["rows"]
        cfgs = sorted({r["config"] for r in rows if r["pool"] == "build"})
        assert cfgs == list(range(75)), (m, cfgs[:5], len(cfgs))
        pc = {}
        with open(f"{CORPUS}/{slug}.jsonl") as fh:
            for line in fh:
                r = json.loads(line)
                if r["dataset"] == "build":
                    pc[r["config_index"]] = r["prompt_conf"]
        assert sorted(pc) == list(range(75)), m
        out[m] = pc
    return out


def _tier(pc):
    plain = pc["cot_prompt"] is None and pc["rag_prompt"] is None
    if not plain:
        return None
    if pc["system_prompt"] == "":
        return 1
    if pc["system_prompt"] == HELPFUL:
        return 2
    return 3


def single_config(pcs, reading):
    """D024's fixed SINGLE rule: first non-empty tier (1: no sys/RAG/CoT;
    2: sys == HELPFUL, no RAG/CoT; 3: no RAG/CoT), then lowest temperature,
    then lowest config index.

    `reading` resolves the one ambiguity P1 raises (Call 1):
      "field"     -- lowest `sampling_hparams.temperature` field, literally
      "effective" -- do_sample=False (greedy) counts as temperature 0
    Returns (config_index, tier, prompt_conf) or None if no tier matches.
    """
    for t in (1, 2, 3):
        # tier 3 includes tiers 1-2 by the D's wording ("no RAG, no CoT")
        cand = [c for c, pc in pcs.items()
                if _tier(pc) is not None and (_tier(pc) == t or (t == 3))]
        if cand:
            def temp(c):
                s = pcs[c]["sampling_hparams"]
                if reading == "effective" and not s["do_sample"]:
                    return 0.0
                return float(s["temperature"])
            c = min(cand, key=lambda c: (temp(c), c))
            return c, t, pcs[c]
    return None


# ------------------------------------------------------------------ distances
def pairs_of(models):
    pairs = list(itertools.combinations(models, 2))
    ix = {m: i for i, m in enumerate(models)}
    return pairs, np.array([ix[a] for a, _ in pairs]), np.array([ix[b] for _, b in pairs])


def point_d2(P, ia, ib):
    """P (n_models, nq, dim) one point per (model, query) -> (nq, n_pairs)
    squared distances, float64."""
    P = np.asarray(P, np.float64)
    D = P[ia] - P[ib]                             # (n_pairs, nq, dim)
    return np.einsum("pqd,pqd->qp", D, D)


def centroid_points(X, models):
    return np.stack([np.asarray(X[m], np.float64).mean(axis=1) for m in models])


def single_points(X, models, cfg):
    return np.stack([np.asarray(X[m][:, cfg[m], :], np.float64) for m in models])


def median_normalise(D2):
    med = np.median(D2, axis=1, keepdims=True)
    assert np.all(med > 0), "a query has median pair distance 0"
    return D2 / med, med[:, 0]


# ------------------------------------------------------------------ greedy
def greedy_cvar(n_queries, init, score, update, k=K, gamma=GAMMA):
    """The one greedy all D024 arms use. `score(state, q)` -> per-pair statistic
    (n_pairs,) for the running set plus candidate q; `update(state, q)` adds q.
    Candidates are scanned in ascending id with a strict `>`, so ties go to the
    lowest id. Returns (chain, trace)."""
    state = init()
    sel, trace, remaining = [], [], np.ones(n_queries, bool)
    for _ in range(k):
        best, bq, bs, n_tie = -np.inf, -1, None, 0
        for q in range(n_queries):
            if not remaining[q]:
                continue
            s = score(state, q)
            o = cvar(s, gamma)
            if o > best:
                best, bq, bs, n_tie = o, q, s, 1
            elif o == best:
                n_tie += 1
        sel.append(bq)
        remaining[bq] = False
        update(state, bq)
        trace.append(dict(k=len(sel), query=int(bq), objective_cvar=float(best),
                          mean_stat=float(np.nanmean(bs)), min_stat=float(np.nanmin(bs)),
                          n_exact_ties_at_max=int(n_tie)))
    return sel, trace


def additive_hooks(D):
    """Hooks for sqrt(sum_q D[q]) with D (nq, n_pairs) -- CENTROID / SINGLE."""
    def init():
        return {"acc": np.zeros(D.shape[1])}

    def score(st, q):
        return np.sqrt(st["acc"] + D[q])

    def update(st, q):
        st["acc"] += D[q]
    return init, score, update


def cloud_hooks(XY2, XX2, ia, ib):
    """Hooks for D010's scale-free joint energy (same accumulators, same
    float32 dtype and `_stat_all_pairs` as LLMmap/joint_greedy.py)."""
    n_pairs, nc = XY2.shape[1], XY2.shape[2]

    def init():
        return {"xy": np.zeros((n_pairs, nc, nc), np.float32),
                "xx": np.zeros((XX2.shape[1], nc, nc), np.float32)}

    def score(st, q):
        return _stat_all_pairs(st["xy"], st["xx"], ia, ib, XY2[q], XX2[q], "energy")

    def update(st, q):
        st["xy"] += XY2[q]
        st["xx"] += XX2[q]
    return init, score, update
