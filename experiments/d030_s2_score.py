"""D030 / P2 S2.2 — score matrices, code-path asserts, D029-format counts, double-BOS reads.
Nothing here prints or writes an S_test summary (P2 §1); the counts are read once, in S3.

Score orientation: higher = more similar (P2 §0). Matrices go to results/D030/scores/ (gitignored):
  test   (85, 25, 85, 2)  S[t, s, m, lib], lib 0 = Gen (gen_ref), 1 = All (all_ref pooled)
  native (85, 85, 2)      S[t, m, lib] for t's native_tgt unit
Subcommands (env):
  met --variant hamming [llmmap-gpu]  exact integer form: with 10 samples per prompt in every unit and
                                      the unit-diagonal kernel K/L, MMD^2_u * 22500 * L =
                                      10 s_xx - 18 s_xy + 10 s_yy (s = off-diagonal match counts summed
                                      over prompts), so the score -(10 s_xx - 18 s_xy + 10 s_yy) is an
                                      exact integer and ties are exact.
  met --variant e5      [llmmap-gpu]  RBF on frozen e5, sigma from s2_freeze.json; same prompt mask
                                      and diagonal exclusion; float64.
  met_assert            [d030-met]    the official `mmd_hamming` / official `_mmd` with a direct RBF
                                      kernel on 20 random (target, reference) pairs per variant,
                                      |delta| <= 1e-9.
  zp --variant 512|200  [d030-zp]     Woodbury fingerprints (float64) from the official per-unit mpnet
                                      means; Pearson by the official formula in float64; asserts vs the
                                      official `_estimate_gradients` + `_aggregate_gradients` and
                                      `compute_similarity` on 5 random units; double-BOS reads.
  os                    [llmmap-gpu]  LLMMAP-OS cosine / Euclidean distances (scipy cdist, as released).
  counts                [llmmap-gpu]  counts_{method}_{variant}_{Gen|All}_{v2|v1}.npz.
  call1                 [llmmap-gpu]  Call 1: D020 JointGreedy|concat|8 linear test logits re-derived on
                                      D020's code path; asserted == D020's stored counts; cnt_pair666.
"""
import os
import sys
import json
import time
import hashlib
import argparse
import itertools

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d030_s2_derive as DV  # noqa: E402

PROJ = DV.PROJ
SC = "./results/D030/scores"
RES = "./results/D030"
L_CHARS = 1000
TGT = [1] + list(range(3, 28))          # native_tgt, test:0..24  (unit indices in DV.UNITS)
REF = [0, 2]                            # gen_ref, all_ref
LIBS = ("Gen", "All")
SEED = 20261010


def sha(p):
    return DV.sha(p)


def freeze():
    return json.load(open(DV.FREEZE))


def save_scores(name, test, native, extra=None):
    os.makedirs(SC, exist_ok=True)
    p = f"{SC}/{name}.npz"
    np.savez(p, test=test, native=native, models=np.array(DV.models()), **(extra or {}))
    return p, sha(p)


# ------------------------------------------------------------------ MET
def codepoints(texts, L=L_CHARS):
    """MET's tokenize_unicode (ord per character, -1 padding), cut / padded to L."""
    out = np.full((len(texts), L), -1, np.int32)
    for i, t in enumerate(texts):
        c = np.frombuffer(t.encode("utf-32-le", "surrogatepass"), np.uint32)[:L]
        out[i, :len(c)] = c
    return out


def met_hamming():
    import torch
    ms = DV.models()
    n = len(ms)
    seq = np.empty((n, 28, 25, 10, L_CHARS), np.int32)
    for i, m in enumerate(ms):
        u = DV.units_of("met", m)
        for j, k in enumerate(DV.UNITS):
            seq[i, j] = codepoints([s["text"] for s in u[k]]).reshape(25, 10, L_CHARS)
    S = torch.tensor(seq, device="cuda")
    del seq
    sxy = torch.zeros((n, 26, n, 2), dtype=torch.int64, device="cuda")
    sself = torch.zeros((n, 28), dtype=torch.int64, device="cuda")
    t0 = time.time()
    for p in range(25):
        X = S[:, :, p]                                            # (n, 28, 10, L)
        within = (X[:, :, :, None, :] == X[:, :, None, :, :]).sum(-1, dtype=torch.int64)  # (n,28,10,10)
        sself += within.sum((2, 3)) - 10 * L_CHARS               # off-diagonal (diagonal = L each)
        R = X[:, REF].reshape(n * 2 * 10, L_CHARS)                # (1700, L)
        T = X[:, TGT]                                             # (n, 26, 10, L)
        for i in range(n):
            Ti = T[i].reshape(26 * 10, L_CHARS)
            k = (Ti[:, None, :] == R[None, :, :]).sum(-1, dtype=torch.int64)       # (260, 1700)
            sxy[i] += k.reshape(26, 10, n, 2, 10).sum((1, 4))
        print(f"[met hamming] prompt {p + 1}/25 {time.time() - t0:.0f}s", flush=True)
    sxx = sself[:, TGT]                                            # (n, 26)
    syy = sself[:, REF]                                            # (n, 2)
    score = -(10 * sxx[:, :, None, None] - 18 * sxy + 10 * syy[None, None, :, :])   # (n,26,n,2)
    score = score.cpu().numpy()
    test = np.ascontiguousarray(score[:, 1:])
    native = np.ascontiguousarray(score[:, 0])
    p, h = save_scores("met_hamming", test, native,
                       dict(scale=np.array(22500 * L_CHARS), s_self=sself.cpu().numpy()))
    print(f"[met hamming] saved {p} sha {h[:12]}", flush=True)


def met_e5():
    import torch
    ms = DV.models()
    n = len(ms)
    sig = freeze()["met_e5_sigma"]["sigma"]
    E = np.stack([np.load(f"{DV.DER}/met_e5/{DV.slug(m)}.npy") for m in ms])     # (n, 7000, 1024)
    E = torch.tensor(E.reshape(n, 28, 25, 10, 1024), device="cuda", dtype=torch.float64)
    g = 1.0 / (2 * sig * sig)
    sxy = torch.zeros((n, 26, n, 2), dtype=torch.float64, device="cuda")
    sself = torch.zeros((n, 28), dtype=torch.float64, device="cuda")
    for p in range(25):
        X = E[:, :, p]                                            # (n, 28, 10, D)
        d2 = ((X[:, :, :, None, :] - X[:, :, None, :, :]) ** 2).sum(-1)
        K = torch.exp(-g * d2)
        sself += K.sum((2, 3)) - torch.diagonal(K, dim1=2, dim2=3).sum(-1)
        R = X[:, REF].reshape(n * 20, 1024)
        T = X[:, TGT].reshape(n * 26 * 10, 1024)
        d2 = (T * T).sum(1)[:, None] + (R * R).sum(1)[None, :] - 2 * T @ R.T
        K = torch.exp(-g * d2.clamp_min(0))
        sxy += K.reshape(n, 26, 10, n, 2, 10).sum((2, 5))
    sxx, syy = sself[:, TGT], sself[:, REF]
    mmd = sxx[:, :, None, None] / 2250 - 2 * sxy / 2500 + syy[None, None] / 2250
    score = (-mmd).cpu().numpy()
    p, h = save_scores("met_e5", np.ascontiguousarray(score[:, 1:]), np.ascontiguousarray(score[:, 0]),
                       dict(sigma=np.array(sig)))
    print(f"[met e5] saved {p} sha {h[:12]} sigma {sig}", flush=True)


def met_assert():
    """d030-met env: official implementations on 20 random (target, reference) pairs per variant."""
    import d030_met as MT
    from model_equality_testing.tests import _mmd
    ms = DV.models()
    rng = np.random.default_rng(SEED)
    sig = freeze()["met_e5_sigma"]["sigma"]
    H = np.load(f"{SC}/met_hamming.npz")
    E5 = np.load(f"{SC}/met_e5.npz")
    scale = float(H["scale"])
    rep = dict(hamming=[], e5=[])
    for _ in range(20):
        t, m = rng.integers(0, 85, 2)
        s = int(rng.integers(-1, 25))                 # -1 = native target
        lib = int(rng.integers(0, 2))
        ut = DV.units_of("met", ms[t])
        ur = DV.units_of("met", ms[m])
        tk = "native_tgt" if s < 0 else f"test:{s}"
        rk = DV.UNITS[REF[lib]]
        X = MT.to_sequences(ut[tk], 25)
        Y = MT.to_sequences(ur[rk], 25)
        off = MT.official(X, Y, 25)
        ours = -(H["native"][t, m, lib] if s < 0 else H["test"][t, s, m, lib]) / scale
        rep["hamming"].append(abs(off - ours))
        # e5: official _mmd with a direct O(n^2) RBF kernel on the frozen embeddings
        Et = np.load(f"{DV.DER}/met_e5/{DV.slug(ms[t])}.npy").astype(np.float64).reshape(28, 250, 1024)
        Er = np.load(f"{DV.DER}/met_e5/{DV.slug(ms[m])}.npy").astype(np.float64).reshape(28, 250, 1024)
        xt = Et[DV.UNITS.index(tk)]
        yr = Er[REF[lib]]
        px = np.array([x["p"] for x in ut[tk]], float)[:, None]
        py = np.array([x["p"] for x in ur[rk]], float)[:, None]

        def rbf(A, B, *_):
            k = lambda a, b: np.exp(-((a[:, None, :] - b[None, :, :]) ** 2).sum(-1) / (2 * sig * sig))  # noqa: E731
            return k(A, A), k(A, B), k(B, B)
        off5 = _mmd(np.hstack([px, xt]), np.hstack([py, yr]), rbf)
        ours5 = -(E5["native"][t, m, lib] if s < 0 else E5["test"][t, s, m, lib])
        rep["e5"].append(abs(off5 - ours5))
    out = {k: dict(n=len(v), max_abs_diff=float(max(v))) for k, v in rep.items()}
    ok = all(v["max_abs_diff"] <= 1e-9 for v in out.values())
    json.dump(dict(**out, ok=ok, tol=1e-9), open(f"{RES}/s2_assert_met.json", "w"), indent=1)
    print(f"[met assert] {out} ok={ok}", flush=True)
    assert ok, "STOP: MET fast form != official"


# ------------------------------------------------------------------ ZP
class Woodbury:
    """Vectorised fingerprint_fast (d030_zp): per base query i, W_i = X_i^T (X_i X_i^T + a I)^-1,
    J_i = (W_i Y_i)^T, fingerprint = mean_i vec(J_i), float64 on GPU."""

    def __init__(self, zp):
        import torch
        ed = zp.embedding_data
        alpha = zp.config.get("ridge_alpha", 1.0)
        assert alpha == 0.001
        I0 = ed["original_embeddings"].detach().cpu().double().numpy()
        Ik = ed["perturbed_embeddings"].detach().cpu().double().numpy()
        self.origs = list(ed["original_queries"])
        self.n0 = len(self.origs)
        self.W, self.idx = [], []
        for i, q in enumerate(self.origs):
            idx = [j for j, d in enumerate(ed["perturbed_queries_data"]) if d["original_query"] == q]
            X = Ik[idx] - I0[i]
            keep = np.linalg.norm(X, axis=1) > 1e-8
            idx = [j for j, k in zip(idx, keep) if k]
            X = X[keep]
            self.W.append(torch.tensor(X.T @ np.linalg.inv(X @ X.T + alpha * np.eye(len(X))), device="cuda"))
            self.idx.append(idx)

    def __call__(self, O):
        """O (B, 10, 768) output embeddings -> (B, 768*768) fingerprints in the official layout."""
        import torch
        O = O.to("cuda", torch.float64)
        F = 0
        for i in range(self.n0):
            Y = O[:, [self.n0 + j for j in self.idx[i]]] - O[:, i:i + 1]          # (B, k, 768)
            coef = torch.einsum("dk,bko->bdo", self.W[i], Y)                     # (B, D_in, D_out)
            F = F + coef.transpose(1, 2).reshape(len(O), -1)
        return F / self.n0


def zp_center(F):
    F = F - F.mean(1, keepdim=True)
    return F / F.norm(dim=1, keepdim=True)


def zp_score(var):
    import torch
    import d030_zp as Z
    zp = Z.load_official()
    wb = Woodbury(zp)
    d = np.load(f"{DV.DER}/zp_mpnet_{var}.npz")
    assert list(d["models"]) == DV.models() and list(d["units"]) == DV.UNITS
    E = torch.tensor(d["E"])                                     # (n, 28, 10, 768) fp32
    n = E.shape[0]
    rng = np.random.default_rng(SEED + int(var))
    # ---- code-path asserts (5 random units; no score read)
    chk = []
    for _ in range(5):
        i, j = int(rng.integers(0, n)), int(rng.integers(0, 28))
        fo = Z.fingerprint_official(zp, E[i, j]).double().cpu()
        ff = wb(E[i, j][None])[0].cpu()
        rel = float((fo - ff).norm() / fo.norm())
        sim = Z.similarity(zp, fo.float(), ff.float())
        chk.append(dict(model=i, unit=j, rel=rel, sim_official_vs_fast=sim))
    # official compute_similarity == our float64 Pearson, on 5 random fingerprint pairs
    pc = []
    for _ in range(5):
        i, j, k, l = (int(x) for x in rng.integers(0, [n, 28, n, 28]))
        fa, fb = wb(E[i, j][None])[0], wb(E[k, l][None])[0]
        off = Z.similarity(zp, fa.float().cpu(), fb.float().cpu())
        ours = float(((zp_center(fa[None]) @ zp_center(fb[None]).T)[0, 0] + 1) / 2)
        pc.append(dict(official=off, ours=ours, abs_diff=abs(off - ours)))
    ok = all(c["rel"] <= 1e-4 and c["sim_official_vs_fast"] >= 1 - 1e-6 for c in chk) and \
        all(c["abs_diff"] <= 1e-5 for c in pc)
    json.dump(dict(variant=var, fingerprint=chk, pearson=pc, ok=ok,
                   tol=dict(rel=1e-4, sim=1 - 1e-6, pearson_abs=1e-5),
                   note="official path is float32 torch; ours float64"),
              open(f"{RES}/s2_assert_zp{var}.json", "w"), indent=1)
    print(f"[zp {var}] asserts ok={ok} max rel {max(c['rel'] for c in chk):.2e} "
          f"max pearson diff {max(c['abs_diff'] for c in pc):.2e}", flush=True)
    assert ok, "STOP: ZP fast path != official"
    # ---- references (85 Gen + 85 All), centred / normalised
    Rf = torch.cat([zp_center(wb(E[:, 0])), zp_center(wb(E[:, 2]))])            # (2n, 768^2)
    out = torch.empty((n, 26, n, 2), dtype=torch.float64)
    for i in range(n):
        T = zp_center(wb(E[i, TGT]))                                             # (26, 768^2)
        s = (T @ Rf.T + 1) / 2                                                   # (26, 2n)
        out[i] = s.reshape(26, 2, n).permute(0, 2, 1).cpu()
    out = out.numpy()
    p, h = save_scores(f"zp_{var}", np.ascontiguousarray(out[:, 1:]), np.ascontiguousarray(out[:, 0]))
    print(f"[zp {var}] saved {p} sha {h[:12]}", flush=True)
    # ---- double-BOS control reads (native rows only; Review stop A)
    ms = DV.models()
    dbl = {}
    Fd = zp_center(wb(torch.tensor(d["dblbos"])))
    Fg = Rf[:n]
    Fn = zp_center(wb(E[:, 1]))
    for k, m in enumerate(d["dblbos_models"]):
        i = ms.index(str(m))
        r_dbl_gen = float(Fd[k] @ Fg[i])
        r_gen_nat = float(Fg[i] @ Fn[i])
        sims = (Fn[i] @ Fg.T + 1) / 2
        sims_rep = sims.clone()
        sims_rep[i] = (Fn[i] @ Fd[k] + 1) / 2
        rank = lambda v: float(1 + (v > v[i]).sum() + 0.5 * ((v == v[i]).sum() - 1))  # noqa: E731
        dbl[str(m)] = dict(pearson_dblbos_vs_genref=r_dbl_gen, pearson_genref_vs_native=r_gen_nat,
                           pearson_dblbos_vs_native=float(Fd[k] @ Fn[i]),
                           rank_native_vs_gen_library=rank(sims.cpu()),
                           rank_native_vs_gen_library_with_dblbos_ref=rank(sims_rep.cpu()))
    path = f"{RES}/dblbos_control.json"
    cur = json.load(open(path)) if os.path.exists(path) else {}
    cur[f"ZP-{var}"] = dbl
    cur["note"] = ("Review D030/P1 stop A: descriptive only, native rows; (i) Pearson(F_dblbos, F_gen_ref) vs "
                   "Pearson(F_gen_ref, F_native_tgt); (ii) rank of the true model for native_tgt against the "
                   "85-model ZP Gen library, with and without that model's reference replaced by gen_ref_dblbos")
    json.dump(cur, open(path, "w"), indent=1)


# ------------------------------------------------------------------ LLMMAP-OS
def os_score():
    from scipy.spatial.distance import cdist
    fz = freeze()["llmmap_os_single"]
    for v in ("v2", "v1"):
        d = np.load(f"{DV.DER}/os_{v}.npz")
        ms = [str(x) for x in d["models"]]
        single = fz[v]["per_model"]
        gen = np.stack([d["build"][i, single[m]["config_index"]] for i, m in enumerate(ms)]).astype(np.float64)
        alll = d["template"]
        T = d["test"].reshape(-1, 384)
        for metric in ("cosine", "euclidean"):
            out = np.empty((len(ms), 25, len(ms), 2))
            for li, R in enumerate((gen, alll)):
                out[:, :, :, li] = -cdist(T, R, metric=metric).reshape(len(ms), 25, len(ms))
            os.makedirs(SC, exist_ok=True)
            p = f"{SC}/os_{metric[:3]}_{v}.npz"
            np.savez(p, test=out, models=np.array(ms))
            print(f"[os] {v} {metric} saved {p} sha {sha(p)[:12]}", flush=True)


# ------------------------------------------------------------------ counts
def view_models():
    uni = json.load(open("./results/D027/universe_v2.json"))
    v1 = sorted(uni["v1"])
    sel = json.load(open("./results/D008/selection.json"))["models"]
    assert sel == v1, "v1 model order differs from D008's"
    return dict(v2=uni["models"], v1=v1)


def counts_from(S, native=None):
    """S (n, 25, n) for one library; returns the D029-format dict plus tie / rank / pair arrays."""
    n = S.shape[0]
    idx = np.arange(n)
    st = S[idx, :, idx]                                           # (n, 25) score of the true model
    mx = S.max(2)
    ties = (S == mx[:, :, None]).sum(2)
    hit = st == mx
    cnt_model = np.where(hit, 1.0 / ties, 0.0)
    rank = 1 + (S > st[:, :, None]).sum(2) + 0.5 * ((S == st[:, :, None]).sum(2) - 1)
    pairs = list(itertools.combinations(range(n), 2))
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    d0 = (S[a, :, a] > S[a, :, b]) + 0.5 * (S[a, :, a] == S[a, :, b])
    d1 = (S[b, :, b] > S[b, :, a]) + 0.5 * (S[b, :, b] == S[b, :, a])
    out = dict(cnt_total=cnt_model.sum(0), cnt_model=cnt_model, tie=(ties > 1).astype(np.int8), rank=rank,
               dir_all=np.stack([d0, d1], 1).astype(np.float64))
    out["cnt_pair_all"] = out["dir_all"].sum(1)
    if native is not None:
        sn = native[idx, idx]
        mxn = native.max(1)
        tn = (native == mxn[:, None]).sum(1)
        out["native_top1"] = np.where(sn == mxn, 1.0 / tn, 0.0)
        out["native_rank"] = 1 + (native > sn[:, None]).sum(1) + 0.5 * ((native == sn[:, None]).sum(1) - 1)
        out["native_tie"] = (tn > 1).astype(np.int8)
    return out


def cmd_counts():
    import d029_lib as L29
    from d008_lib import near_relative_pairs
    vm = view_models()
    full = vm["v2"]
    tg = L29.targets(full)
    pidx = {f"{a} | {b}": j for j, (a, b) in enumerate(itertools.combinations(full, 2))}
    H = [pidx[p] for p in tg["H"]]
    v1 = vm["v1"]
    near = near_relative_pairs(v1)
    struct = sorted(near)                                         # combinations-of-v1 column indices
    files = {("met", "hamming"): "met_hamming", ("met", "e5"): "met_e5",
             ("zp", "512"): "zp_512", ("zp", "200"): "zp_200"}
    for v in ("v2", "v1"):
        files[("os", f"cos_{v}")] = f"os_cos_{v}"
        files[("os", f"euc_{v}")] = f"os_euc_{v}"
    shas = {}
    for (meth, var), f in files.items():
        z = np.load(f"{SC}/{f}.npz")
        shas[f] = sha(f"{SC}/{f}.npz")
        zm = [str(x) for x in z["models"]]
        views = ("v2", "v1") if meth != "os" else (var[-2:],)
        for v in views:
            ms = vm[v]
            ix = [zm.index(m) for m in ms]
            for li, lib in enumerate(LIBS):
                S = z["test"][np.ix_(ix, range(25), ix)][..., li]
                nat = z["native"][np.ix_(ix, ix)][..., li] if "native" in z.files else None
                c = counts_from(S, nat)
                if v == "v2":
                    c["dirH"] = c["dir_all"][H]
                    c["cnt_pair"] = c["dirH"].sum(1)
                else:
                    c["cnt_pair666"] = c["cnt_pair_all"]
                    c["cnt_pair"] = c["cnt_pair_all"][struct]
                    assert len(struct) == 65 and c["cnt_pair666"].shape == (666, 25)
                vv = var if meth != "os" else var[:3]
                np.savez_compressed(f"{RES}/counts_{meth}_{vv}_{lib}_{v}.npz", **c)
    json.dump(dict(score_sha256=shas, written=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   format="D029: cnt_total (25), cnt_model (n x 25, fractional tie credit), cnt_pair (v2: H_all 57, "
                          "v1: D016 65 structural), dirH (v2, 57 x 2 x 25, ties 0.5), cnt_pair666 (v1), dir_all / "
                          "cnt_pair_all (all pairs, combinations order), tie, rank (average), native_* (MET/ZP)"),
              open(f"{RES}/s2_counts_meta.json", "w"), indent=1)
    print(f"[counts] written {len(files)} score files -> counts_*.npz", flush=True)


# ------------------------------------------------------------------ Call 1
def cmd_call1():
    """D020's `fit` verbatim except that the logits are kept (fit returns argmax only)."""
    from sklearn.linear_model import LogisticRegression
    from d009_lib import load_query_embeddings, build_traces, logit_stats
    from d008_lib import near_relative_pairs, pair_index
    from d007_lib import load_corpus
    from d020_paired_linear import chains_and_refs, POOL, MAX_ITER, TOL
    from d023_train import pair666_counts
    models = json.load(open("./results/D008/selection.json"))["models"]
    nm = len(models)
    hard = near_relative_pairs(models)
    chains, _ = chains_and_refs()
    q = chains["JointGreedy"][:8]
    pool, C = POOL["concat"]
    qe = load_query_embeddings()
    cubes = {p: load_corpus(pool=p) for p in ("build", "test")}
    tr, y_tr, _, _ = build_traces(q, "build", qe, cubes["build"])
    te, y_te, c_te, _ = build_traces(q, "test", qe, cubes["test"])
    clf = LogisticRegression(C=C, max_iter=MAX_ITER, tol=TOL, solver="lbfgs")
    clf.fit(pool(tr), y_tr)
    n_it = int(clf.n_iter_.max())
    assert n_it < MAX_ITER
    lg = clf.decision_function(pool(te))
    pt, st = logit_stats(lg, y_te, c_te, hard, nm)
    ref = np.load("./results/D020/linear_counts.npz")
    eq = {k: bool(np.array_equal(st[k], ref[f"JointGreedy|concat|8|{k}"])) for k in ("cnt_total", "cnt_model", "cnt_pair")}
    pairs = pair_index(models)
    pa = np.array([models.index(a) for a, _ in pairs])
    pb = np.array([models.index(b) for _, b in pairs])
    out = dict(chain=q, n_iter=n_it, reproduces_D020=eq, all_equal=all(eq.values()),
               mean_top1=pt["mean_top1"])
    if out["all_equal"]:
        c666 = pair666_counts(lg, y_te, c_te, pa, pb)
        np.savez_compressed(f"{RES}/call1_linear_v1.npz", cnt_pair666=c666, cnt_model=st["cnt_model"],
                            logits=lg.astype(np.float32),
                            cnt_pair=st["cnt_pair"], cnt_total=st["cnt_total"])
        out["path"] = "asserted reproduction (Call 1 primary)"
    else:
        out["path"] = "FALLBACK: 65-pair structural proxy (reproduction failed)"
    json.dump(out, open(f"{RES}/call1_linear_v1.json", "w"), indent=1)
    print(f"[call1] reproduces D020 exactly: {eq} -> {out['path']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["met", "met_assert", "zp", "os", "counts", "call1"])
    ap.add_argument("--variant")
    a = ap.parse_args()
    os.chdir(PROJ)
    if a.cmd == "met":
        {"hamming": met_hamming, "e5": met_e5}[a.variant]()
    elif a.cmd == "zp":
        zp_score(a.variant)
    elif a.cmd == "os":
        os_score()
    else:
        dict(met_assert=met_assert, counts=cmd_counts, call1=cmd_call1)[a.cmd]()


if __name__ == "__main__":
    main()
