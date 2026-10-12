"""D031 — shared pieces: the two views, Helm et al.'s spectral selection, the generic rules, draws,
training order, and the count format. Plan: docs/plans/D031-P1.md (Review: A1 centred-MMR variants,
A2 draw-arm E_tail = mean of per-draw CVaR, A3 GreedyCover reference row).

Views (never compared as levels):
  v2  corpus v2, 85 models, 3,570 pairs; frozen tensors results/D028/S_energy_{sf,raw}_tok200_v2.npy
  v1  corpus v1, 37 models (D008 order), 666 pairs; frozen tensors results/D007/S_energy_{sf,raw}_tok200.npy
"""
import json
import hashlib
import itertools
import subprocess

import numpy as np

from LLMmap.joint_statistic import cvar

OUT = "./results/D031"
GAMMA = 0.1
NPOOL = 259
K_MAX = 8
N_DRAWS = 20
N_DRAWS_ATT = 10
DRAW_KS = (2, 4, 8)
LAMBDAS = (0.3, 0.5, 0.7)
SIMS = ("cos", "pearson")                      # Review A1: centred similarity added
E_VARIANTS = ("raw", "sf")
BOOT_DRAW_SEED = 20261012                      # Review Call 5 (a)
GMM_KW = dict(covariance_type="full", n_init=10, random_state=0, max_iter=1000, tol=1e-6)
SIGMA_GUARD = 1e-12
TAIL_FLAG_S = 129
CAP_DIAG = 20

VIEW = {
    "v2": dict(sf="./results/D028/S_energy_sf_tok200_v2.npy", raw="./results/D028/S_energy_raw_tok200_v2.npy",
               n=85, P=3570, boot_seed=20261005, n_slot=24),
    "v1": dict(sf="./results/D007/S_energy_sf_tok200.npy", raw="./results/D007/S_energy_raw_tok200.npy",
               n=37, P=666, boot_seed=20260928, n_slot=8),
}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def jdump(obj, path):
    json.dump(obj, open(path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


# ------------------------------------------------------------------ views
def models(view):
    if view == "v2":
        return json.load(open("./results/D027/universe_v2.json"))["models"]
    return json.load(open("./results/D008/selection.json"))["models"]


def pairs(view):
    ms = models(view)
    P = list(itertools.combinations(range(len(ms)), 2))
    return np.array([a for a, _ in P]), np.array([b for _, b in P])


def tensors(view):
    v = VIEW[view]
    sf, raw = np.load(v["sf"]), np.load(v["raw"])
    assert sf.shape == raw.shape == (NPOOL, v["P"]), (sf.shape, raw.shape)
    return sf, raw


def cubes(view, splits=("build", "val", "test")):
    """{split: (models, X)}, X[m] (259, ncfg, 1024); configs sorted (the loaders every D since D009 used)."""
    if view == "v2":
        import d029_lib as L29
        return L29.load_v2(models("v2"), splits=splits, with_ml=False)
    from d007_lib import load_corpus
    out = {}
    for s in splits:
        ms, X = load_corpus(pool=s)
        assert ms == models("v1")
        out[s] = (ms, X)
    return out


def jg_chain(view):
    if view == "v2":
        return json.load(open("./results/D029/chains.json"))["arms"]["JG"]["ids"][:K_MAX]
    return json.load(open("./results/D010/selection.json"))["joint_energy"]["queries"][:K_MAX]


PAPER8 = list(range(8))


# ------------------------------------------------------------------ draws
def draw_rng(arm, view, variant):
    """The D's rule: default_rng(sha256("D031|<arm>|<view>|<variant>")), digest read as an integer."""
    return np.random.default_rng(int(hashlib.sha256(f"D031|{arm}|{view}|{variant}".encode()).hexdigest(), 16))


def draws(arm, view, variant, base, n=N_DRAWS, length=K_MAX):
    rng = draw_rng(arm, view, variant)
    base = np.array(sorted(base), int)
    return [[int(x) for x in rng.permutation(base)[:min(length, len(base))]] for _ in range(n)]


# ------------------------------------------------------------------ Helm et al. 2026
def helm(E):
    """Plain SVD (no centring) -> r_hat = argmax sigma_s/sigma_{s+1} (s = 1..258, sigma guard) ->
    per direction l <= r_hat a 1-D GMM on |U_jl|, K in {1,2} by BIC -> zero-sets, Orthogonal = their
    intersection, Signal = complement; rank score = ||(U Sigma)_{j,1:r_hat}||."""
    from sklearn.mixture import GaussianMixture
    E = np.asarray(E, np.float64)
    U, s, _ = np.linalg.svd(E, full_matrices=False)
    for j in range(U.shape[1]):                            # sign convention only
        if U[np.argmax(np.abs(U[:, j])), j] < 0:
            U[:, j] *= -1
    ratio = s[:-1] / s[1:]
    ok = s[1:] > SIGMA_GUARD * s[0]
    rr = np.where(ok, ratio, -np.inf)
    r_hat = int(np.argmax(rr)) + 1
    order = np.argsort(-rr)
    r_cap = int(np.argmax(rr[:CAP_DIAG])) + 1

    def fit(x):
        x = np.asarray(x, np.float64).reshape(-1, 1)
        g1 = GaussianMixture(1, **GMM_KW).fit(x)
        g2 = GaussianMixture(2, **GMM_KW).fit(x)
        b1, b2 = float(g1.bic(x)), float(g2.bic(x))
        d = dict(bic_K1=b1, bic_K2=b2, K=1 if b1 <= b2 else 2)
        if d["K"] == 2:
            mu = g2.means_[:, 0]
            z = int(np.argmin(np.abs(mu)))
            lab = g2.predict(x)
            d.update(means=mu.tolist(), vars=g2.covariances_[:, 0, 0].tolist(), weights=g2.weights_.tolist(),
                     zero_component=z, rho_hat=float(g2.weights_[z]), zero_set=np.where(lab == z)[0].tolist())
        else:
            d.update(means=g1.means_[:, 0].tolist(), vars=g1.covariances_[:, 0, 0].tolist(), zero_set=[])
        d["n_zero"] = len(d["zero_set"])
        return d

    dirs, signed = [], []
    for l in range(r_hat):
        dirs.append(fit(np.abs(U[:, l])))
        signed.append({k: v for k, v in fit(U[:, l]).items() if k != "zero_set"})   # diagnostic only
    orth = set(range(E.shape[0]))
    for d in dirs:
        orth &= set(d["zero_set"])
    orth = sorted(orth)
    signal = sorted(set(range(E.shape[0])) - set(orth))
    score = np.linalg.norm((U * s)[:, :r_hat], axis=1)
    rank = [int(q) for q in np.lexsort((np.arange(len(score)), -score))]
    return dict(spectrum=s.tolist(), ratios=ratio.tolist(), r_hat=r_hat, runner_up_s=int(order[1]) + 1,
                r_hat_ratio=float(rr[r_hat - 1]), runner_up_ratio=float(rr[order[1]]),
                tail_flag=bool(r_hat > TAIL_FLAG_S), r_hat_cap20=r_cap, guard_excluded=int((~ok).sum()),
                gmm_abs=dirs, gmm_signed_diag=signed, orthogonal=orth, signal=signal,
                n_signal=len(signal), n_orthogonal=len(orth), degenerate_sig_equals_random=len(orth) == 0,
                rank_score=score.tolist(), rank=rank, U_first3=U[:, :3].tolist())


# ------------------------------------------------------------------ generic rules
def single_cvar(Esf):
    return np.array([cvar(Esf[q], GAMMA) for q in range(Esf.shape[0])])


def topk(Esf, k=K_MAX):
    rel = single_cvar(Esf)
    return [int(q) for q in np.lexsort((np.arange(len(rel)), -rel))[:k]], rel


def similarity(Esf, kind):
    R = np.asarray(Esf, np.float64)
    if kind == "pearson":
        R = R - R.mean(1, keepdims=True)
    R = R / np.linalg.norm(R, axis=1, keepdims=True)
    return R @ R.T


def mmr(Esf, lam, kind, k=K_MAX):
    rel = single_cvar(Esf)
    relp = (rel - rel.min()) / (rel.max() - rel.min())
    sim = similarity(Esf, kind)
    sel, rem = [], np.ones(len(rel), bool)
    for _ in range(k):
        pen = sim[:, sel].max(1) if sel else np.zeros(len(rel))
        obj = np.where(rem, lam * relp - (1 - lam) * pen, -np.inf)
        q = int(np.argmax(obj))                     # first maximum = lower id on an exact tie
        sel.append(q)
        rem[q] = False
    return sel


# ------------------------------------------------------------------ training order (D024 Call 2, ext. PAPER8)
def train_order(S, jg, p8=PAPER8):
    S = [int(q) for q in S]
    k = len(S)
    if set(S) == set(jg[:k]):
        return list(jg[:k]), "JG"
    if set(S) == set(p8[:k]):
        return list(p8[:k]), "PAPER8"
    shared = [q for q in jg[:k] if q in S]
    return shared + sorted(set(S) - set(shared)), None


# ------------------------------------------------------------------ jobs (one entry per trained set)
def jobs(ch, view, readout):
    """List of dicts (arm, variant, k, draw, q_train, ident) for one view and readout, from chains.json."""
    V = ch["views"][view]
    out = []
    for name, a in V["arms"].items():
        if name in ("JG", "PAPER8") or (readout == "att" and a["kind"] == "draw" and a["arm"] == "HELM-ORTH"):
            continue
        if a["kind"] == "det":
            for k in range(1, K_MAX + 1):
                t = a["train"][str(k)]
                out.append(dict(name=name, k=k, draw=None, q=t["q"], ident=t["ident"]))
        else:
            nd = N_DRAWS if readout == "lin" else N_DRAWS_ATT
            ks = DRAW_KS if readout == "lin" else (8,)
            for d in range(nd):
                for k in ks:
                    t = a["train"][d].get(str(k))
                    if t is not None:
                        out.append(dict(name=name, k=k, draw=d, q=t["q"], ident=t["ident"]))
    return out


def key(j, seed=None):
    s = f"{j['name']}|{j['k']}"
    if j["draw"] is not None:
        s += f"|d{j['draw']}"
    if seed is not None:
        s += f"|r{seed}"
    return s
