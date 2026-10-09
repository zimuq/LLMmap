"""D030 — ZeroPrint fingerprints, offline from stored outputs (ZP-512; ZP-200 in P2).

Run with envs/d030-zp/bin/python. Uses the OFFICIAL methods of the pinned repo on the frozen
prepare() state (results/D030/zp_prepare/cache/zeroprint/zeroprint_embeddings.pth):
  _compute_output_embeddings (mpnet, mean over the 20 repeats)
  _estimate_gradients        (ridge Jacobian per base query, alpha = 0.001)
  _aggregate_gradients       (mean over the 2 base queries)
  fingerprint_helper.compute_similarity  ((Pearson + 1) / 2)
`fingerprint_fast` is an independent closed form of the same ridge solution
(X^T X + aI)^-1 X^T Y = X^T (X X^T + aI)^-1 Y, used in P2 for the thousands of
fingerprints; S0.5(b) asserts it matches the official path.
"""
import os
import sys
import json

import numpy as np

ZP_REPO = "/work/11280/zimuq1/vista/ext/ZeroPrint"
PROJ = "/work/11280/zimuq1/vista/LLMmap-project/LLMmap"
PREP = f"{PROJ}/results/D030/zp_prepare"
PROMPTS = f"{PROJ}/confs/baselines/zp_prompts.json"
R = 20


def load_official():
    sys.path.insert(0, ZP_REPO)
    import torch
    from utils.utils import load_config
    from fingerprint.fingerprint_factory import create_fingerprint_method
    config = load_config(f"{ZP_REPO}/config/zeroprint.yaml")
    zp = create_fingerprint_method(config, accelerator=None)
    zp.embedding_data = torch.load(f"{PREP}/cache/zeroprint/zeroprint_embeddings.pth", weights_only=False)
    assert list(zp.embedding_data["all_queries"]) == json.load(open(PROMPTS))["all_queries_order"]
    return zp


def outputs_from_record(rec):
    """Stored unit (one config) -> the expanded output list in official order
    (all_queries order, 20 repeats each, prompt-major)."""
    by = {}
    for s in rec["samples"]:
        by[(s["p"], s["r"])] = s["text"]
    n_p = len({p for p, _ in by})
    assert n_p == 10 and len(by) == 10 * R, "a ZP unit must hold 10 prompts x 20 repeats"
    return [by[(p, r)] for p in range(10) for r in range(R)]


def output_embeddings(zp, outputs):
    return zp._compute_output_embeddings(outputs)          # (10, 768), mean over repeats


def fingerprint_official(zp, out_emb):
    ed = zp.embedding_data
    n0 = len(ed["original_queries"])
    J = zp._estimate_gradients(ed["original_queries"], ed["perturbed_queries_data"],
                               ed["original_embeddings"], ed["perturbed_embeddings"],
                               out_emb[:n0], out_emb[n0:]).cpu()
    return zp._aggregate_gradients(J, zp.config)


def fingerprint_fast(zp, out_emb, alpha=None):
    ed = zp.embedding_data
    alpha = zp.config.get("ridge_alpha", 1.0) if alpha is None else alpha
    origs = list(ed["original_queries"])
    I0 = ed["original_embeddings"].detach().cpu().double().numpy()
    Ik = ed["perturbed_embeddings"].detach().cpu().double().numpy()
    O = out_emb.detach().cpu().double().numpy()
    O0, Ok = O[:len(origs)], O[len(origs):]
    Js = []
    for i, q in enumerate(origs):
        idx = [j for j, d in enumerate(ed["perturbed_queries_data"]) if d["original_query"] == q]
        X = Ik[idx] - I0[i]
        Y = Ok[idx] - O0[i]
        keep = np.linalg.norm(X, axis=1) > 1e-8
        X, Y = X[keep], Y[keep]
        coef = X.T @ np.linalg.solve(X @ X.T + alpha * np.eye(len(X)), Y)     # (D_in, D_out)
        Js.append(coef.T.reshape(-1))
    return np.mean(Js, axis=0)


def similarity(zp, f1, f2):
    import torch
    t = lambda f: f if isinstance(f, torch.Tensor) else torch.tensor(f, dtype=torch.float32)  # noqa: E731
    return zp.fingerprint_helper.compute_similarity(t(f1), t(f2))
