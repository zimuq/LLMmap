"""D031 / S3 — budget section: MET at matched target-side budget, from D030's data (no generation).

  kernels  [GPU]  per-sample kernel sums against every reference (P1 §S3):
                  kx[t, s, p, i, m, lib] = sum over m's 10 reference samples of prompt p of k(target sample i,
                  reference sample); ky[m, lib, p] = off-diagonal reference-reference sum of prompt p.
                  Hamming: integer match counts on MET code points (L = 1000, D030 code path);
                  e5: RBF on the frozen e5 embeddings, D030's frozen sigma, float64.
  check           n = 250 vs D030 (stop on failure): Hamming 18 sum kx - 10 sum ky == D030 score + 10 s_xx
                  exactly; e5 score - D030 score constant per target (<= 1e-9); identical top-1 tie sets.
  counts          draws (n = 8, 25, 50, 100: 20 each; 250: 1) -> D029/D030-format counts per cell and view.

Score with a subsampled target (prompt set P_d, n_s samples per prompt): the prompt-aware MMD without its
target-only term, ref-ref restricted to P_d:
  score_m = 2 sum kx / (|P_d| n_s 10) - sum ky / (|P_d| 90);  Hamming x |P_d| n_s 90 L: 18 sum kx - n_s sum ky.
Draw d: one prompt subset shared by every target unit; sample indices independent per (target unit, prompt).
S_test discipline: counts are written, not summarised (S4 reads).
Usage:  PYTHONPATH=.:experiments python experiments/d031_budget.py kernels|check|counts
"""
import os
import sys
import json
import time

import numpy as np

import d031_lib as L
import d030_s2_derive as DV
from d030_s2_score import codepoints, counts_from, L_CHARS

BS = f"{L.OUT}/budget_scores"
SC30 = "./results/D030/scores"
TGT = list(range(3, 28))                       # test:0..24 in DV.UNITS
REF = [0, 2]                                   # gen_ref, all_ref
LIBS = ("Gen", "All")
SPEC = {8: (8, 1), 25: (25, 1), 50: (25, 2), 100: (25, 4), 250: (25, 10)}


def cmd_kernels():
    import torch
    ms = DV.models()
    assert ms == L.models("v2")
    n = len(ms)
    os.makedirs(BS, exist_ok=True)
    t0 = time.time()
    # ---------------- Hamming
    seq = np.empty((n, 28, 25, 10, L_CHARS), np.int32)
    for i, m in enumerate(ms):
        u = DV.units_of("met", m)
        for j, k in enumerate(DV.UNITS):
            seq[i, j] = codepoints([s["text"] for s in u[k]]).reshape(25, 10, L_CHARS)
    S = torch.tensor(seq, device="cuda")
    del seq
    kx = torch.zeros((n, 25, 25, 10, n, 2), dtype=torch.int32, device="cuda")
    ky = torch.zeros((n, 2, 25), dtype=torch.int64, device="cuda")
    for p in range(25):
        X = S[:, :, p]                                            # (n, 28, 10, L)
        Rr = X[:, REF]                                            # (n, 2, 10, L)
        w = (Rr[:, :, :, None, :] == Rr[:, :, None, :, :]).sum(-1, dtype=torch.int64)
        ky[:, :, p] = w.sum((2, 3)) - 10 * L_CHARS
        R = Rr.reshape(n * 2 * 10, L_CHARS)
        for i in range(n):
            Ti = X[i, TGT].reshape(25 * 10, L_CHARS)
            k = (Ti[:, None, :] == R[None, :, :]).sum(-1, dtype=torch.int32)      # (250, 1700)
            kx[i, :, p] = k.reshape(25, 10, n, 2, 10).sum(-1)
        print(f"[kernels hamming] prompt {p + 1}/25 {time.time() - t0:.0f}s", flush=True)
    np.save(f"{BS}/kx_hamming.npy", kx.cpu().numpy())
    np.save(f"{BS}/ky_hamming.npy", ky.cpu().numpy())
    del S, kx, ky
    torch.cuda.empty_cache()
    # ---------------- e5
    sig = json.load(open(DV.FREEZE))["met_e5_sigma"]["sigma"]
    g = 1.0 / (2 * sig * sig)
    E = np.stack([np.load(f"{DV.DER}/met_e5/{DV.slug(m)}.npy") for m in ms])
    E = torch.tensor(E.reshape(n, 28, 25, 10, 1024), device="cuda", dtype=torch.float64)
    kx = torch.zeros((n, 25, 25, 10, n, 2), dtype=torch.float64, device="cuda")
    ky = torch.zeros((n, 2, 25), dtype=torch.float64, device="cuda")
    for p in range(25):
        X = E[:, :, p]
        Rr = X[:, REF]
        d2 = ((Rr[:, :, :, None, :] - Rr[:, :, None, :, :]) ** 2).sum(-1)
        K = torch.exp(-g * d2)
        ky[:, :, p] = K.sum((2, 3)) - torch.diagonal(K, dim1=2, dim2=3).sum(-1)
        R = Rr.reshape(n * 20, 1024)
        T = X[:, TGT].reshape(n * 25 * 10, 1024)
        d2 = (T * T).sum(1)[:, None] + (R * R).sum(1)[None, :] - 2 * T @ R.T
        K = torch.exp(-g * d2.clamp_min(0))
        kx[:, :, p] = K.reshape(n, 25, 10, n, 2, 10).sum(-1)
    np.save(f"{BS}/kx_e5.npy", kx.cpu().numpy())
    np.save(f"{BS}/ky_e5.npy", ky.cpu().numpy())
    L.jdump(dict(sigma=sig, L=L_CHARS, wall_min=round((time.time() - t0) / 60, 1),
                 shapes=dict(kx=[n, 25, 25, 10, n, 2], ky=[n, 2, 25])), f"{BS}/kernels.json")
    print(f"[kernels] done {(time.time() - t0) / 60:.1f} min", flush=True)


def load(var):
    return np.load(f"{BS}/kx_{var}.npy", mmap_mode="r"), np.load(f"{BS}/ky_{var}.npy")


def score(var, kx, ky, P, I, ns):
    """kx (n, 25, 25, 10, n, 2) restricted to targets; P prompt list; I (n_t, 25, |P|, ns) sample indices or None
    (all 10). Returns (n_t, 25, n_m, 2) scores (int64 Hamming numerator, float64 e5)."""
    sub = np.asarray(kx[:, :, P], np.int64 if var == "hamming" else np.float64)    # (nt, 25, |P|, 10, nm, 2)
    if I is not None:
        sub = np.take_along_axis(sub, I[..., None, None], 3)
    sx = sub.sum((2, 3))                                                            # (nt, 25, nm, 2)
    sy = ky[:, :, P].sum(-1)                                                        # (nm, 2)
    if var == "hamming":
        return 18 * sx - ns * sy[None, None]
    return 2 * sx / (len(P) * ns * 10) - sy[None, None] / (len(P) * 90)


def cmd_check():
    out = {}
    for var in ("hamming", "e5"):
        kx, ky = load(var)
        mine = score(var, kx, ky, list(range(25)), None, 10)                            # (85, 25, 85, 2)
        z = np.load(f"{SC30}/met_{var}.npz")
        ref = z["test"]
        if var == "hamming":
            sxx = z["s_self"][:, TGT]                                                   # (85, 25)
            assert np.array_equal(mine, ref + 10 * sxx[:, :, None, None]), "Hamming n=250 != D030"
            spread = 0.0
        else:
            d = mine - ref
            spread = float(np.max(d.max(2) - d.min(2)))
            assert spread <= 1e-9, spread
        agree = 0
        for lib in range(2):
            a, b = mine[..., lib], ref[..., lib]
            ta = a == a.max(2, keepdims=True)
            tb = b == b.max(2, keepdims=True)
            assert np.array_equal(ta, tb), f"top-1 tie sets differ ({var}, lib {lib})"
            agree += int(ta.shape[0] * ta.shape[1])
        out[var] = dict(exact_integer_equality=var == "hamming", e5_constant_offset_spread=spread,
                        top1_tie_sets_identical_units=agree)
        print(f"[check] {var}: {out[var]}", flush=True)
    L.jdump(out, f"{L.OUT}/budget_check.json")


def draw_spec(view, n, nt):
    """Yields (P, I, ns) per draw for n < 250 (20 draws), one deterministic for 250."""
    npr, ns = SPEC[n]
    if n == 250:
        yield list(range(25)), None, 10
        return
    rng = L.draw_rng("MET-BUDGET", view, f"n{n}")
    for _ in range(L.N_DRAWS):
        P = sorted(rng.choice(25, npr, replace=False).tolist()) if npr < 25 else list(range(25))
        I = rng.random((nt, 25, npr, 10)).argsort(-1)[..., :ns]
        yield P, I, ns


def cmd_counts():
    t0 = time.time()
    full = L.models("v2")
    meta = {}
    for var in ("hamming", "e5"):
        kx, ky = load(var)
        for view in ("v2", "v1"):
            ms = L.models(view)
            ix = [full.index(m) for m in ms]
            kxv = np.asarray(kx[ix][:, :, :, :, ix])                                  # (nt, 25, 25, 10, nm, 2)
            kyv = ky[ix]
            for li, lib in enumerate(LIBS):
                store = {}
                for n in SPEC:
                    cm, cpa, ct, tie, dpa = [], [], [], [], []
                    for P, I, ns in draw_spec(view, n, len(ms)):
                        S = score(var, kxv, kyv, P, I, ns)[..., li]
                        c = counts_from(S)
                        cm.append(c["cnt_model"]); ct.append(c["cnt_total"]); tie.append(c["tie"].mean())
                        dpa.append(np.round(2 * c["cnt_pair_all"]).astype(np.uint8))   # 2x (ties 0.5) -> integer
                    cm, ct, dpa = np.array(cm), np.array(ct), np.array(dpa)
                    store[f"n{n}|cnt_model_draws"] = cm.astype(np.float32)
                    store[f"n{n}|cnt_total_draws"] = ct
                    store[f"n{n}|cnt_model"] = cm.mean(0)
                    store[f"n{n}|cnt_pair_all"] = dpa.mean(0) / 2
                    store[f"n{n}|tie_rate_draws"] = np.array(tie)
                    os.makedirs(BS, exist_ok=True)
                    np.save(f"{BS}/pair2x_{var}_{lib}_{view}_n{n}.npy", dpa)
                    if n == 250:                                                        # = D030's counts
                        z = np.load(f"./results/D030/counts_met_{var}_{lib}_{view}.npz")
                        assert np.allclose(cm[0], z["cnt_model"], atol=1e-12, rtol=0), (var, lib, view)
                        assert np.array_equal(dpa[0] / 2, z["cnt_pair_all"]), (var, lib, view)
                np.savez_compressed(f"{L.OUT}/budget_met_{var}_{lib}_{view}.npz", **store)
                meta[f"{var}|{lib}|{view}"] = dict(n=list(SPEC), draws={str(n): (1 if n == 250 else L.N_DRAWS) for n in SPEC})
                print(f"[counts] {var} {lib} {view} written ({(time.time() - t0) / 60:.1f} min)", flush=True)
    L.jdump(dict(schema="d031-budget-v1", spec={str(k): v for k, v in SPEC.items()}, cells=meta,
                 n250_equals_d030_counts=True, s_test="written, not summarised (S4 reads)",
                 wall_min=round((time.time() - t0) / 60, 1)), f"{L.OUT}/budget_meta.json")


if __name__ == "__main__":
    {"kernels": cmd_kernels, "check": cmd_check, "counts": cmd_counts}[sys.argv[1]]()
