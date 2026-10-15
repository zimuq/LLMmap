"""D032 — attribution factorial: criterion {CVaR_0.1, MEAN} x rule {TOP-K, MMR-PP, MMR-TXT}.
Plan: docs/plans/D032-P1.md (Review D032/P1: approved, Call 2 -> (a), Call 3 as written).

Shared pieces on top of d031_lib (imported, never edited): the two scores, the TXT similarity on the
query-text embeddings E(q), the generalised MMR greedy, chain names, and the D031 <-> D032 name map.
Views as D031 (never compared as levels): v2 85 models / 3,570 pairs; v1 37 models / 666 pairs.
"""
import numpy as np

import d031_lib as L31

OUT = "./results/D032"
CRITS = ("CVaR", "MEAN")
PP_SIMS = L31.SIMS                              # ("cos", "pearson") -- D031 verbatim
TXT_SIMS = ("cos", "ccos")                      # the D: cosine, centred cosine (pool-mean E(q) subtracted)
LAMBDAS = L31.LAMBDAS


def rel(Esf, crit):
    """Single-query score over all pairs: CVaR_0.1 (= D031 TOP-K's score, same function) or the mean."""
    if crit == "CVaR":
        return L31.single_cvar(Esf)
    return np.asarray(Esf, np.float64).mean(1)


def topk(r, k=L31.K_MAX):
    return [int(q) for q in np.lexsort((np.arange(len(r)), -r))[:k]]   # lower id on a tie (D031 rule)


def txt_similarity(E, kind):
    X = np.asarray(E, np.float64)
    if kind == "ccos":
        X = X - X.mean(0, keepdims=True)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X @ X.T


def mmr(r, sim, lam, k=L31.K_MAX):
    """D031 mmr with the relevance and the similarity matrix passed in (same min-max scaling, max redundancy,
    first maximum = lower id on an exact tie)."""
    relp = (r - r.min()) / (r.max() - r.min())
    sel, rem = [], np.ones(len(r), bool)
    for _ in range(k):
        pen = sim[:, sel].max(1) if sel else np.zeros(len(r))
        obj = np.where(rem, lam * relp - (1 - lam) * pen, -np.inf)
        q = int(np.argmax(obj))
        sel.append(q)
        rem[q] = False
    return sel


def chain_names():
    """All 26 chains per view (7 reused from D031 + 19 new), in a fixed order."""
    out = [f"TOP-K[{c}]" for c in CRITS]
    for c in CRITS:
        out += [f"MMR-PP[{c}]|{s}|{l}" for s in PP_SIMS for l in LAMBDAS]
    for c in CRITS:
        out += [f"MMR-TXT[{c}]|{s}|{l}" for s in TXT_SIMS for l in LAMBDAS]
    return out


def d031_name(name):
    """D031 key of a reused chain, else None."""
    if name == "TOP-K[CVaR]":
        return "TOP-K"
    if name.startswith("MMR-PP[CVaR]|"):
        return "MMR|" + name.split("|", 1)[1]
    return None


def variants(rule, crit):
    sims = PP_SIMS if rule == "MMR-PP" else TXT_SIMS
    return [f"{rule}[{crit}]|{s}|{l}" for s in sims for l in LAMBDAS]      # = S_val tie order (D031)


NEW = [n for n in chain_names() if d031_name(n) is None]
CELLS = (("MMR-PP", "MEAN"), ("MMR-TXT", "CVaR"), ("MMR-TXT", "MEAN"))   # S1 picks a star for these
