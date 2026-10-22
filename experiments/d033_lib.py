"""D033 — shared pieces: families, folds, restricted selection, readout (2) terms, count helpers.
Plan: docs/plans/D033-P1.md (Review D033/P1: Calls 1-3 -> (a); A1 encoder diagnostics).

Views (never compared as levels): v2 = 85 models / 3,570 pairs; v1 = 37 models (D008 order) / 666 pairs.
Fold kinds: "model" (primary, family-stratified, 5 x 17 on v2) and "family" (whole families held out).
"""
import csv
import json
import hashlib
import subprocess
import collections

import numpy as np

import d031_lib as L31

OUT = "./results/D033"
K = 8
NF = 5
KINDS = ("model", "family")
VIEWS = ("v2", "v1")
ARMS = ("OURS", "PAPER8", "FULL", "JG")
OURS_SIM, OURS_LAM = "pearson", 0.7
NC_VARIANTS = ("raw", "sf")
SEEDS = (0, 1, 2)
NCFG = 25


def rng_for(tag):
    return np.random.default_rng(int(hashlib.sha256(tag.encode()).hexdigest(), 16))


def git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def committed(path):
    """True iff `path` is tracked, unmodified, and HEAD is on origin (pushed)."""
    st = subprocess.run(["git", "status", "--porcelain", "--", path], capture_output=True, text=True).stdout.strip()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", path], capture_output=True).returncode == 0
    ahead = subprocess.run(["git", "rev-list", "--count", "origin/main..HEAD"], capture_output=True,
                           text=True).stdout.strip()
    return tracked and st == "" and ahead == "0"


# ------------------------------------------------------------------ families
def metadata():
    rows = {r["model"]: r for r in csv.DictReader(open("./results/D001/model_metadata.csv"))}
    rows.update({r["model"]: r for r in csv.DictReader(open("./results/D027/candidates_metadata.csv"))})
    return rows


def families():
    """model -> family label. Family = connected component of the D027 N' graph (same base or same lineage);
    asserted equal to the `lineage` partition, and labelled by the lineage string."""
    ms = L31.models("v2")
    par = {m: m for m in ms}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for p in json.load(open("./results/D027/n_prime.json"))["pairs"]:
        a, b = p["pair"].split(" | ")
        par[find(a)] = find(b)
    comp = collections.defaultdict(set)
    for m in ms:
        comp[find(m)].add(m)
    rows = metadata()
    lin = collections.defaultdict(set)
    for m in ms:
        assert rows[m]["lineage"], m
        lin[rows[m]["lineage"]].add(m)
    assert sorted(map(sorted, comp.values())) == sorted(map(sorted, lin.values())), \
        "STOP: N' components != lineage groups"
    return {m: rows[m]["lineage"] for m in ms}


def _family_order(fam, rng):
    groups = collections.defaultdict(list)
    for m, f in fam.items():
        groups[f].append(m)
    names = sorted(groups)
    rank = dict(zip(names, rng.permutation(len(names)).tolist()))
    order = sorted(names, key=lambda f: (-len(groups[f]), rank[f]))
    return order, groups


def folds_model(fam):
    rng = rng_for("D033|folds|model")
    order, groups = _family_order(fam, rng)
    seq = []
    for f in order:
        mem = sorted(groups[f])
        seq += [mem[i] for i in rng.permutation(len(mem))]
    return {m: i % NF for i, m in enumerate(seq)}


def folds_family(fam):
    rng = rng_for("D033|folds|family")
    order, groups = _family_order(fam, rng)
    size, out = [0] * NF, {}
    for f in order:
        j = int(np.argmin(size))                  # first minimum = lowest fold index on a tie
        for m in groups[f]:
            out[m] = j
        size[j] += len(groups[f])
    return out


def sibling_enrolled(fam, fold, models):
    """Per model of `models`: True iff some other member of its family (within `models`) is in another fold."""
    return {m: any(fam[o] == fam[m] and fold[o] != fold[m] for o in models if o != m) for m in models}


# ------------------------------------------------------------------ restricted selection
def train_split(view, fold, f):
    ms = L31.models(view)
    tr = [i for i, m in enumerate(ms) if fold[m] != f]
    ho = [i for i, m in enumerate(ms) if fold[m] == f]
    ia, ib = L31.pairs(view)
    keep = np.isin(ia, tr) & np.isin(ib, tr)
    cols = np.where(keep)[0]
    return tr, ho, cols


def ours(sf, cols):
    return [int(q) for q in L31.mmr(sf[:, cols], OURS_LAM, OURS_SIM, K)]


def jg(XY2, XX2, ia, ib, tr, cols):
    from LLMmap.joint_greedy import joint_greedy
    pos = -np.ones(int(max(ia.max(), ib.max())) + 1, int)
    pos[tr] = np.arange(len(tr))
    a, b = pos[ia[cols]], pos[ib[cols]]
    assert (a >= 0).all() and (b >= 0).all()
    xy = XY2[:, cols] if len(cols) < XY2.shape[1] else XY2
    xx = XX2[:, tr] if len(tr) < XX2.shape[1] else XX2
    sel, trace = joint_greedy(xy, xx, a, b, K, L31.GAMMA, "energy", verbose=False)
    return [int(q) for q in sel], [float(t["objective_cvar"]) for t in trace]


# ------------------------------------------------------------------ readout (2): point-to-cloud energy terms
def nc_terms(cube_b, cube_t, models, queries):
    """A[t, m, j] = mean_i ||x_{t,q_j} - Y_{m,q_j,i}|| and W[m, j] = mean_{i,i'} ||Y_i - Y_i'|| (V-statistic,
    diagonal exactly 0), float64 on the fp16/fp32 cubes. Targets t = models x configs (model-major)."""
    n = len(models)
    nct = cube_t[models[0]].shape[1]
    A = np.empty((n * nct, n, len(queries)), np.float64)
    W = np.empty((n, len(queries)), np.float64)
    for j, q in enumerate(queries):
        Y = np.stack([np.asarray(cube_b[m][q], np.float64) for m in models])        # (n, 75, d)
        X = np.concatenate([np.asarray(cube_t[m][q], np.float64) for m in models])  # (n*nct, d)
        y2 = np.einsum("mid,mid->mi", Y, Y)
        x2 = np.einsum("td,td->t", X, X)
        for mi in range(n):
            d2 = x2[:, None] + y2[mi][None, :] - 2.0 * (X @ Y[mi].T)
            A[:, mi, j] = np.sqrt(np.maximum(d2, 0)).mean(1)
            g = y2[mi][:, None] + y2[mi][None, :] - 2.0 * (Y[mi] @ Y[mi].T)
            np.fill_diagonal(g, 0.0)
            W[mi, j] = np.sqrt(np.maximum(g, 0)).mean()
    return A, W


def nc_scores(A, W, cols, variant):
    """Summed per-query energy over the query positions `cols` (indices into A's last axis)."""
    e = 2.0 * A[:, :, cols] - W[None, :, cols]
    if variant == "sf":
        e = e / W[None, :, cols]
    return e.sum(2)


# ------------------------------------------------------------------ counts from a score matrix (lower = closer)
def counts(D, n, ia, ib, ncfg=NCFG):
    """D (n*ncfg, n) distance of each target (model-major) to each enrolled model. Returns D031's count format:
    cnt_model (n, ncfg) top-1 correct (argmin, tie -> lowest index); cnt_pair_all (P, ncfg) both directions
    summed (target a: correct iff D[a] <= D[b] when a < b ... tie -> lower-index member)."""
    y = np.repeat(np.arange(n), ncfg)
    pred = np.argmin(D, 1)
    cnt_model = (pred == y).reshape(n, ncfg).astype(np.float64)
    G = D.reshape(n, ncfg, n)
    d0 = G[ia, :, ia] <= G[ia, :, ib]            # traces of a (lower index); tie -> a
    d1 = G[ib, :, ib] < G[ib, :, ia]             # traces of b; tie -> a (counts as wrong for b)
    return dict(cnt_model=cnt_model, cnt_total=cnt_model.sum(0), cnt_pair_all=(d0.astype(np.uint8) + d1.astype(np.uint8)))


def jdump(obj, path):
    L31.jdump(obj, path)
