"""D038 — S_build-only diagnostic of MMR*'s v2 edge over JointGreedy: query strength (M1), redundancy (M2), the
two-halves transfer test (M3), a build-internal readout (2) anchor (M4). Plan: docs/plans/D038-P1.md.

Two computation paths, each the frozen path of what it must reproduce (P1 Call 1):
  single-query energies  d007_lib.energy's operations (scipy cdist in float64, V-statistic means, raw / within, stored
                         float32) on every model's cloud restricted to a config subset C. With C = all 75 S_build
                         configs this is the frozen tensor S_energy_sf_tok200(_v2). Used by M1, M2, MMR*, TOP-K, GC,
                         Y_max and the pool ceiling.
  set statistics         the float32 XY2 / XX2 accumulators (d010_select.precompute) restricted to the sub-block
                         [..., C, C], with LLMmap/joint_greedy's operations (d036_lib.energy_stat "sf"). Used by JG
                         and Y_joint.
Splits: R = 20, default_rng(20261109 + r).permutation(75); A_r = sorted(first 37), B_r = sorted(last 38); the same
index split for every model (configs are drawn per model, TACC_NOTES Issue 18) and both views. Config set s:
0 = all, 1 + 2r = A_r, 2 + 2r = B_r. Directed split (r, d): d = 0 selects on A_r and measures on B_r, d = 1 the
reverse. Nothing here reads S_val or S_test: cubes are loaded with splits=("build",) only.
"""
import os
import json
import multiprocessing as mp

import numpy as np
from scipy.spatial.distance import cdist

import d031_lib as L31
import d036_lib as L36
from LLMmap.joint_statistic import cvar

OUT = "./results/D038"
VIEWS = ("v2", "v1")
K = 8
GAMMA = 0.1
NPOOL = 259
NCFG = 75
NA = 37
R = 20
SEED0 = 20261109
MMR_STAR = ("pearson", 0.7)
MMR_DESC = (0.3, 0.5)
PAPER8 = list(range(8))
FROZEN_SHA = {"v2": "8a31d7603c27d1e8b455e4b653ba3c3bc5edd4cf50582f9b43443abc73a1f223",     # D029
              "v1": "bd822e3293295af3c2e12d28e9cba153663bcb247f217c52b27a553650e01c5f"}     # D031 / D008


# ------------------------------------------------------------------ splits
def splits():
    out = []
    for r in range(R):
        p = np.random.default_rng(SEED0 + r).permutation(NCFG)
        out.append((np.sort(p[:NA]), np.sort(p[NA:])))
    return out


def config_sets():
    s = [np.arange(NCFG)]
    for a, b in splits():
        s += [a, b]
    return s


def directed():
    """[(r, d, sel_set_index, meas_set_index)] for the 40 directed splits."""
    return [(r, d, 1 + 2 * r + d, 2 + 2 * r - d) for r in range(R) for d in (0, 1)]


# ------------------------------------------------------------------ inputs (S_build only)
def build_cubes(view):
    ms, X = L31.cubes(view, ("build",))["build"]
    assert ms == L31.models(view)
    for m in ms:
        assert X[m].shape == (NPOOL, NCFG, 1024), (view, m, X[m].shape)
    return ms, X


def frozen(view):
    p = L31.VIEW[view]["sf"]
    assert L31.sha(p) == FROZEN_SHA[view], f"STOP: {view} frozen sf tensor hash differs"
    return np.load(p)


# ------------------------------------------------------------------ path 1: single-query energies (builder ops)
G = {}


def _energies_q(q):
    """sf energy of query q for every pair and every config set; the operations of d007_lib.energy on the restricted
    clouds (cdist entries are computed independently, so a sub-block of the full matrix is the restricted cdist)."""
    X, ms, ia, ib, sets = G["X"], G["ms"], G["ia"], G["ib"], G["sets"]
    C = [np.asarray(X[m][q], np.float32) for m in ms]
    xx = np.empty((len(sets), len(ms)))
    for i, c in enumerate(C):
        D = cdist(c, c)
        for s, ix in enumerate(sets):
            xx[s, i] = D[np.ix_(ix, ix)].mean()
    out = np.empty((len(sets), len(ia)), np.float32)
    for p, (a, b) in enumerate(zip(ia, ib)):
        D = cdist(C[a], C[b])
        for s, ix in enumerate(sets):
            xy = D[np.ix_(ix, ix)].mean()
            xa, xb = xx[s, a], xx[s, b]
            raw = float(2 * xy - xa - xb)
            within = (xa + xb) / 2.0
            out[s, p] = float(raw / within) if within > 0 else float("nan")
    return q, out


def energies(view, ms, X, workers):
    """(n_sets, 259, P) float32. Forks: call before any BLAS-threaded work in the process."""
    ia, ib = L31.pairs(view)
    G.update(X=X, ms=ms, ia=ia, ib=ib, sets=config_sets())
    E = np.empty((len(G["sets"]), NPOOL, len(ia)), np.float32)
    with mp.get_context("fork").Pool(workers) as pool:
        for i, (q, e) in enumerate(pool.imap_unordered(_energies_q, range(NPOOL))):
            E[:, q] = e
            if (i + 1) % 50 == 0:
                print(f"  [{view}] energies {i + 1}/{NPOOL}", flush=True)
    G.clear()
    return E


def energies_path(view):
    return f"{OUT}/energies_{view}.npz"


# ------------------------------------------------------------------ path 2: set statistics (accumulator ops)
def accumulators(view, ms, X):
    from d010_select import precompute
    ia, ib = L31.pairs(view)
    XX2, XY2 = precompute(X, ms, [(ms[a], ms[b]) for a, b in zip(ia, ib)])
    assert XY2.shape == (NPOOL, len(ia), NCFG, NCFG) and XX2.shape == (NPOOL, len(ms), NCFG, NCFG)
    return XY2, XX2, ia, ib


def restrict(Z, ix):
    """The [..., C, C] sub-block, one contiguous copy."""
    ix = np.asarray(ix)
    return np.ascontiguousarray(Z[:, :, ix[:, None], ix[None, :]])


def y_joint(XY2h, XX2h, ia, ib, chain):
    """JG's objective for every prefix of `chain`: CVaR_0.1 over pairs of the joint sf statistic, accumulated in chain
    order exactly as joint_greedy does (acc + candidate, then acc += candidate). Returns (objs, per-pair stats)."""
    acc_xy = np.zeros(XY2h.shape[1:], np.float32)
    acc_xx = np.zeros(XX2h.shape[1:], np.float32)
    objs, stats = [], []
    for q in chain:
        s = L36.energy_stat(acc_xy, acc_xx, ia, ib, XY2h[q], XX2h[q], "sf")
        objs.append(float(cvar(s, GAMMA)))
        stats.append(s)
        acc_xy += XY2h[q]
        acc_xx += XX2h[q]
    return objs, np.stack(stats)


def y_max(E, chain):
    """GreedyCover's objective for every prefix: CVaR_0.1 over pairs of MAX coverage (float64, as greedy_cover)."""
    S = np.asarray(E, np.float64)
    cov = np.zeros(S.shape[1])
    out = []
    for q in chain:
        cov = np.maximum(cov, S[q])
        out.append(float(cvar(cov, GAMMA)))
    return out


# ------------------------------------------------------------------ rules (unchanged code; tie counts alongside)
def jg(XY2h, XX2h, ia, ib):
    import d037_lib as L37
    return L37.jg_chain(XY2h, XX2h, ia, ib, GAMMA)


def topk(E):
    sel, rel = L31.topk(E)
    s = np.sort(rel)[::-1]
    return sel, dict(boundary_tie=bool(s[K - 1] == s[K]))


def mmr(E, lam, kind=MMR_STAR[0]):
    """d031_lib.mmr, plus the number of exact maxima at each step (recomputed with the same arithmetic)."""
    sel = L31.mmr(E, lam, kind)
    rel = L31.single_cvar(E)
    relp = (rel - rel.min()) / (rel.max() - rel.min())
    sim = L31.similarity(E, kind)
    ties, rem = [], np.ones(len(rel), bool)
    for j, q in enumerate(sel):
        pen = sim[:, sel[:j]].max(1) if j else np.zeros(len(rel))
        obj = np.where(rem, lam * relp - (1 - lam) * pen, -np.inf)
        assert int(np.argmax(obj)) == q
        ties.append(int((obj == obj.max()).sum()))
        rem[q] = False
    return sel, dict(n_exact_ties=ties)


def gc(E):
    from LLMmap.greedy_cover import greedy_cover
    sel, tr = greedy_cover(E, K, gamma=GAMMA, agg="max", return_trace=True)
    return [int(q) for q in sel], dict(n_ties=[t["n_ties"] for t in tr], objective=[t["objective"] for t in tr])


# ------------------------------------------------------------------ frozen reference chains
def ref_chains(view):
    c31 = json.load(open(f"{L31.OUT}/chains.json"))["views"][view]["arms"]
    out = {"JG": L31.jg_chain(view), "TOP-K[CVaR]": c31["TOP-K"]["sel"][:K],
           "MMR*": c31["MMR|pearson|0.7"]["sel"][:K], "PAPER8": PAPER8,
           **{f"MMR|pearson|{l}": c31[f"MMR|pearson|{l}"]["sel"][:K] for l in MMR_DESC}}
    if view == "v2":
        d29 = json.load(open("./results/D029/chains.json"))["arms"]
        out["GC"] = d29["GC"]["ids"][:K]
        out["JG_trace"] = [t["objective_cvar"] for t in d29["JG"]["trace"][:K]]
    else:
        d8 = json.load(open("./results/D008/selection.json"))
        assert d8["tensor"]["sha256"] == FROZEN_SHA["v1"], "D008's GC chain is on a different v1 tensor"
        c = d8["conditions"]["cvar_max"]
        assert c["gamma"] == GAMMA and c["agg"] == "max"
        out["GC"] = c["queries"][:K]
        out["JG_trace"] = [t["objective_cvar"] for t in
                           json.load(open("./results/D010/selection.json"))["joint_energy"]["trace"][:K]]
    return out


# ------------------------------------------------------------------ M1 / M2
def strength(E):
    """s_q = single-query CVaR_0.1 (d031_lib.single_cvar) and pool rank (1 = strongest; ties to the lower id)."""
    s = L31.single_cvar(E)
    order = np.lexsort((np.arange(len(s)), -s))
    rank = np.empty(len(s), int)
    rank[order] = np.arange(1, len(s) + 1)
    return s, rank


def redundancy(E, chain):
    from d032_s0 import within
    sub = np.asarray(E)[list(chain)]
    Pm, Cm = L31.similarity(sub, "pearson"), L31.similarity(sub, "cos")
    lam = np.linalg.eigvalsh(Pm)
    return dict(within_pearson=within(Pm, range(len(chain))), within_cos=within(Cm, range(len(chain))),
                eff_n=float(lam.sum() ** 2 / (lam ** 2).sum()), min_eig=float(lam.min()),
                step_max_corr=[None] + [float(Pm[j, :j].max()) for j in range(1, len(chain))])


# ------------------------------------------------------------------ M4: readout (2) raw on halves
def readout2(X, ms, ia, ib, ref_ix, probe_ix, chains):
    """D033's frozen point-to-cloud rule (d033_lib.nc_terms / nc_scores 'raw' / counts): reference clouds = ref_ix
    configs, probes = every probe_ix config of every model. Returns {name: [(P_all, P_tail) for k = 1..8]}."""
    import d033_lib as L33
    U = sorted({q for c in chains.values() for q in c})
    pos = {q: j for j, q in enumerate(U)}
    ref = {m: np.asarray(X[m][U])[:, ref_ix] for m in ms}
    prb = {m: np.asarray(X[m][U])[:, probe_ix] for m in ms}
    A, W = L33.nc_terms(ref, prb, ms, list(range(len(U))))
    n, nct = len(ms), len(probe_ix)
    out = {}
    for name, c in chains.items():
        rows = []
        for k in range(1, K + 1):
            cn = L33.counts(L33.nc_scores(A, W, [pos[q] for q in c[:k]], "raw"), n, ia, ib, ncfg=nct)
            rows.append((float(cn["cnt_total"].sum() / (n * nct)),
                         float(cvar(cn["cnt_pair_all"].sum(1) / (2 * nct), GAMMA))))
        out[name] = rows
    return out


def jdump(obj, path):
    L31.jdump(obj, path)


def workers():
    return int(os.environ.get("D038_WORKERS", 64))
