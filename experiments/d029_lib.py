"""D029 — shared pieces: frozen v2 inputs, the 85-model pool + multilingual
loader, the target sets / strata, and the logit-derived counts.

Candidate index space (P1 §0):
  0..258     pool query id 0..258        (data/corpus_v2/embeddings)
  259..274   ML probe id 281..296        (data/corpus_v2_ext_ml/embeddings)
Cubes are kept fp16 (d009_lib.build_traces casts per slot, as v1 does).
All numbers produced with these objects are CORPUS V2.
"""
import os
import json
import hashlib
import itertools
import subprocess

import numpy as np

OUT = "./results/D029"
SEED = 20261005
GAMMA = 0.1
NPOOL, NML, ML_FIRST_ID = 259, 16, 281
NCAND = NPOOL + NML
TENSOR = "./results/D028/S_energy_sf_tok200_v2.npy"
TENSOR_META = "./results/D028/tensor_v2.json"
HARD = "./results/D028/hard_sets_v2.json"
HARD_COMMIT = "3f8d2ed"
UNIVERSE = "./results/D027/universe_v2.json"
MANIFEST = "./results/D028/manifest.json"
ML_POOL = "./confs/queries/pool_d028_ml.json"
POOL_EMB = "./data/corpus_v2/embeddings"
ML_EMB = "./data/corpus_v2_ext_ml/embeddings"
ML_QEMB = f"{OUT}/ml_query_emb.npy"
NCFG = {"build": 75, "val": 25, "test": 25}
D027_HP_HASH = "01e27d7f8bccfb2eda4b1a62b8c34aa91c0df902c30bcc0b0ebd9f9cacdd5737"
QWEN_EXCLUDE = {"Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B",
                "Qwen/Qwen2.5-7B-Instruct | Qwen/Qwen2.5-7B-Instruct-1M"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def cand_to_id(c):
    return int(c) if c < NPOOL else ML_FIRST_ID + int(c) - NPOOL


def git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------------ frozen inputs
def assert_inputs():
    """P1 §0. Returns a dict of hashes; any mismatch raises."""
    meta = json.load(open(TENSOR_META))
    uni = json.load(open(UNIVERSE))
    models = uni["models"]
    assert len(models) == 85 and models == sorted(models) and models == meta["models"]
    assert len(uni["candidates"]) == 48 and len(uni["v1"]) == 37
    pairs = [f"{a}|{b}" for a, b in itertools.combinations(models, 2)]
    assert meta["pairs"] == pairs and meta["shape"] == [259, 3570] and meta["reproduction_ok"]
    t_sha = sha(TENSOR)
    assert t_sha == meta["sha256"]["energy_sf"] and t_sha.startswith("8a31d760"), t_sha
    hs = json.load(open(HARD))
    frozen = subprocess.run(["git", "show", f"{HARD_COMMIT}:{HARD[2:]}"], capture_output=True).stdout
    assert frozen == open(HARD, "rb").read(), "hard_sets_v2.json differs from its 3f8d2ed version"
    assert (len(hs["H_N"]), len(hs["H_X"]), len(hs["H_all"])) == (38, 19, 57)
    u_sha = sha(UNIVERSE)
    assert u_sha == hs["sources"]["universe_v2_sha256"], u_sha
    man = json.load(open(MANIFEST))
    assert man["complete_pool"] == 85 and man["complete_ml"] == 85
    assert all(man["models"][m]["ml_status"] == "COMPLETE" for m in models)
    mm = sorted(man["mode_mixed"]["H_all"])
    assert len(mm) == 5 and set(mm) <= set(hs["H_all"])
    ml = json.load(open(ML_POOL))
    assert ml["first_id"] == ML_FIRST_ID and len(ml["queries"]) == NML
    assert [q["id"] for q in ml["queries"]] == list(range(ML_FIRST_ID, ML_FIRST_ID + NML))
    q0 = json.load(open("./confs/queries/pool_v1.json"))
    assert q0["sha256"] == "0a6d098c7b540bf7fc4e111a9d11f58eb4b3ddad80f37ba1f10d6677775ae0e8"
    return dict(tensor_energy_sf_sha256=t_sha, hard_sets_sha256=sha(HARD), hard_sets_commit=HARD_COMMIT,
                universe_sha256=u_sha, manifest_sha256=sha(MANIFEST), ml_pool_sha256=sha(ML_POOL),
                pool_v1_sha256=q0["sha256"])


def universe():
    return json.load(open(UNIVERSE))["models"]


def pair_index(models):
    pairs = list(itertools.combinations(models, 2))
    ix = {m: i for i, m in enumerate(models)}
    ia = np.array([ix[a] for a, _ in pairs])
    ib = np.array([ix[b] for _, b in pairs])
    pidx = {f"{a} | {b}": j for j, (a, b) in enumerate(pairs)}
    return pairs, ia, ib, pidx


# ------------------------------------------------------------------ loader
def _cube(emb_dir, slug, split, nq):
    a = np.load(f"{emb_dir}/{slug}.npy", mmap_mode="r")
    rows = json.load(open(f"{emb_dir}/{slug}.index.json"))["rows"]
    sel = [(i, r["query_index"], r["config"]) for i, r in enumerate(rows) if r["pool"] == split]
    cfgs = sorted({c for _, _, c in sel})
    assert len(cfgs) == NCFG[split], (slug, split, len(cfgs))
    ci = {c: j for j, c in enumerate(cfgs)}
    cube = np.empty((nq, len(cfgs), a.shape[1]), np.float16)
    seen = np.zeros((nq, len(cfgs)), bool)
    for i, q, c in sel:
        cube[q, ci[c]] = a[i]
        seen[q, ci[c]] = True
    assert seen.all(), (slug, split)
    return cube, cfgs


def load_v2(models, splits=("build", "val", "test"), with_ml=True):
    """{split: (models, X)}, X[m] (275 or 259, ncfg, 1024) fp16, configs sorted.
    ML rows are placed at candidate 259+i and must share the pool's config set."""
    out = {}
    for split in splits:
        X = {}
        for m in models:
            s = m.replace("/", "__")
            pc, pcfg = _cube(POOL_EMB, s, split, NPOOL)
            if with_ml:
                mc, mcfg = _cube(ML_EMB, s, split, NML)
                assert mcfg == pcfg, (m, split)
                X[m] = np.concatenate([pc, mc], 0)
            else:
                X[m] = pc
        out[split] = (models, X)
    return out


def query_embeddings_275():
    from d009_lib import load_query_embeddings
    qe = load_query_embeddings()                       # (259, 1024) fp32, asserted
    ml = np.load(ML_QEMB).astype(np.float32)
    assert ml.shape == (NML, 1024)
    return np.concatenate([qe, ml], 0)


# ------------------------------------------------------------------ targets
def targets(models):
    """Pre-registered target sets and strata (Review: n fixed)."""
    hs = json.load(open(HARD))
    uni = json.load(open(UNIVERSE))
    new = set(uni["candidates"])
    man = json.load(open(MANIFEST))
    H = hs["H_all"]
    rows = {r["pair"]: r for r in hs["H_N_rows"] + hs["H_X_rows"]}
    mm = set(man["mode_mixed"]["H_all"])
    T1 = [p for p in hs["H_N"] if "tiny-aya" in p]
    T2 = [p for p in hs["H_N"] if "Ministral-3" in p]
    T3 = [p for p in hs["H_N"] if "Qwen" in p and p not in QWEN_EXCLUDE]
    strata = dict(
        H_all=H, H_N=hs["H_N"], H_X=hs["H_X"],
        stably_hard=[p for p in H if rows[p]["band"] == "stably_hard"],
        no_mode_mixed=[p for p in H if p not in mm],
        fresh_only=[p for p in H if all(m in new for m in p.split(" | "))])
    n = {k: len(v) for k, v in strata.items()}
    assert n == dict(H_all=57, H_N=38, H_X=19, stably_hard=26, no_mode_mixed=52, fresh_only=42), n
    assert (len(T1), len(T2), len(T3)) == (6, 3, 10)
    ix = {m: i for i, m in enumerate(models)}
    blocks = dict(new48=sorted(ix[m] for m in new), v1_37=sorted(ix[m] for m in uni["v1"]))
    model_sets = dict(F_N=sorted(ix[m] for m in hs["F_N"]), M_X=sorted(ix[m] for m in hs["M_X"]),
                      M_N=sorted(ix[m] for m in hs["M_N"]))
    fams = {"|".join(g[0].split("/")[1].split("-")[:2]) + f"#{j}": [ix[m] for m in g]
            for j, g in enumerate(hs["family_groups_N"])}
    return dict(H=H, T1=T1, T2=T2, T3=T3, strata=strata, blocks=blocks, model_sets=model_sets,
                fams=fams, bands={p: rows[p]["band"] for p in H}, mode_mixed=sorted(mm),
                tiny_aya=sorted(ix[m] for m in models if "tiny-aya" in m))


def hard_ab(models, H):
    ix = {m: i for i, m in enumerate(models)}
    pa = np.array([ix[p.split(" | ")[0]] for p in H])
    pb = np.array([ix[p.split(" | ")[1]] for p in H])
    return pa, pb
