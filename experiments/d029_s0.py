"""D029 / S0 — frozen inputs, ML query embeddings, loader checks, v1 code-path
checks, the chains (frozen before any training), and descriptive pointers.
No training, no S_val / S_test accuracy of anything.

  S0.1  assert frozen inputs (d029_lib.assert_inputs)
  S0.2  embed the 16 ML probe TEXTS with D028's I5 function (parity on the 259
        pool texts vs corpus_v1 _queries.npy, stop > 1e-2)            [Call 3]
  S0.L  loader checks: ML/pool prompt_conf per row + ML texts; byte equality
        vs d007_lib.load_corpus on the 34 reused models (3 splits); pair_worker
        on 3 pairs reproduces the frozen v2 tensors (<= 1e-6)
  S0.V  code path on v1: JG wrapper == D025 GLOBAL 16-chain; GC wrapper ==
        D013 gamma=0.1 chain (exact)
  S0.3  v2 chains: JG (259, k16), JG+ML (275, k16), GC (k8), JG8+ML16 (k24)
  S0.4  pointers: I4 coverage (pool queries, v2 tensor) + the joint statistic
        per pair at k=8/16, ranks among 3,570; ML probes selected; tail panel
Writes results/D029/{ml_query_emb.npy,ml_query_emb.json,chains.json,s0.json,
tail_k8.npz}; the png is drawn by d029_plots.py.
Usage:  PYTHONPATH=.:experiments python experiments/d029_s0.py
"""
import os
import gc
import json
import time
import shutil
import tempfile

import numpy as np

import d029_lib as L

T0 = time.time()


def log(msg):
    print(f"[S0 {time.time() - T0:7.0f}s] {msg}", flush=True)


def host_mem_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal"):
            return int(line.split()[1]) / 1e6
    return float("nan")


# ------------------------------------------------------------------ S0.2
def embed_ml_queries():
    import torch
    from transformers import AutoTokenizer, AutoModel
    from d006_s6_embed import mean_pool, I5_MODEL, I5_DIM, MAX_LEN, BATCH
    tok = AutoTokenizer.from_pretrained(I5_MODEL)
    mdl = AutoModel.from_pretrained(I5_MODEL, torch_dtype=torch.float16).cuda().eval()

    def run(texts):                       # == d028_embed.main.run
        out = np.empty((len(texts), I5_DIM), np.float16)
        with torch.no_grad():
            for i in range(0, len(texts), BATCH):
                b = tok(texts[i:i + BATCH], padding=True, truncation=True, max_length=MAX_LEN,
                        return_tensors="pt").to("cuda")
                e = mean_pool(mdl(**b).last_hidden_state, b["attention_mask"])
                out[i:i + len(e)] = e.float().cpu().numpy().astype(np.float16)
        return out

    q0 = json.load(open("./confs/queries/pool_v1.json"))
    stored = np.load("./data/corpus_v1/embeddings/_queries.npy")
    re = run([e["text"] for e in q0["queries"]])
    parity = float(np.abs(re.astype(np.float32) - stored.astype(np.float32)).max())
    log(f"query-embedding parity on 259 pool texts: max |diff| {parity:.3g}")
    assert parity <= 1e-2, f"STOP: I5 query parity {parity}"
    ml = json.load(open(L.ML_POOL))
    E = run([q["text"] for q in ml["queries"]])
    np.save(L.ML_QEMB, E)
    meta = dict(n=L.NML, dim=I5_DIM, dtype="fp16", ids=[q["id"] for q in ml["queries"]],
                embedding_model=I5_MODEL, pooling="mean, NOT normalised", max_length=MAX_LEN,
                function="d028_embed run() == d006_s6_embed mean_pool, fp16 model",
                probe_file=L.ML_POOL, probe_file_sha256=L.sha(L.ML_POOL),
                parity_259_pool_texts_max_abs_diff=parity, sha256=L.sha(L.ML_QEMB))
    json.dump(meta, open(f"{L.OUT}/ml_query_emb.json", "w"), indent=1)
    del mdl
    torch.cuda.empty_cache()
    return meta


# ------------------------------------------------------------------ S0.L
def check_prompt_conf(models):
    ml = json.load(open(L.ML_POOL))
    texts = [q["text"] for q in ml["queries"]]
    n_rows = 0
    for m in models:
        s = m.replace("/", "__")
        pool = {}
        for line in open(f"./data/corpus_v2/{s}.jsonl"):
            d = json.loads(line)
            pool[(d["dataset"], d["config_index"])] = d["prompt_conf"]
        seen = set()
        for line in open(f"./data/corpus_v2_ext_ml/{s}.jsonl"):
            d = json.loads(line)
            key = (d["dataset"], d["config_index"])
            assert d["prompt_conf"] == pool[key], f"STOP: ML prompt_conf != pool for {m} {key}"
            assert [t[0] for t in d["traces"]] == texts, f"STOP: ML texts differ for {m} {key}"
            seen.add(key)
            n_rows += 1
        assert seen == set(pool), m
    return n_rows


def check_v1_bytes(cubes, reuse):
    from d007_lib import load_corpus
    out = {}
    for split in ("build", "val", "test"):
        m1, X1 = load_corpus(pool=split)
        for m in reuse:
            assert np.array_equal(X1[m], cubes[split][1][m][:L.NPOOL].astype(np.float32)), \
                f"STOP: v2 loader != d007_lib for {m} {split}"
        out[split] = len(reuse)
        del X1
        gc.collect()
    return out


def check_pair_worker(cubes, models):
    from joblib import Parallel, delayed
    from d007_build_tensor import pair_worker
    meta = json.load(open(L.TENSOR_META))
    sa, sb = np.array(meta["split_a"]), np.array(meta["split_b"])
    pairs = [("CohereLabs/tiny-aya-earth", "CohereLabs/tiny-aya-water"),
             ("allenai/Llama-3.1-Tulu-3-8B", "allenai/OLMo-2-1124-13B-Instruct"),
             ("tiiuae/Falcon3-10B-Instruct", "tiiuae/Falcon3-7B-Instruct")]
    tmp = tempfile.mkdtemp(prefix="d029_pw_")
    for m in {x for p in pairs for x in p}:
        np.save(f"{tmp}/{m.replace('/', '__')}.npy", cubes["build"][1][m][:L.NPOOL].astype(np.float32))
    res = Parallel(n_jobs=3)(delayed(pair_worker)(tmp, a, b, sa, sb) for a, b in pairs)
    shutil.rmtree(tmp, ignore_errors=True)
    keys = ("probe", "energy_raw", "energy_sf", "probe_A", "probe_B")
    S = {k: np.load(f"./results/D028/S_{k}_tok200_v2.npy", mmap_mode="r") for k in keys}
    pidx = {p: j for j, p in enumerate(meta["pairs"])}
    out = {}
    for (a, b), r in zip(pairs, res):
        j = pidx[f"{a}|{b}"]
        d = {k: float(np.abs(np.asarray(S[k][:, j]) - v).max()) for k, v in zip(keys, r)}
        out[f"{a} | {b}"] = d
        assert max(d.values()) <= 1e-6, f"STOP: pair_worker vs frozen tensor {a}|{b}: {d}"
    return out


# ------------------------------------------------------------------ selection helpers
def jg(XY2, XX2, ia, ib, k):
    from LLMmap.joint_greedy import joint_greedy, peak_k
    sel, trace = joint_greedy(XY2, XX2, ia, ib, k, L.GAMMA, "energy", verbose=True)
    return [int(q) for q in sel], trace, peak_k(trace)


def replay_margins(XY2, XX2, ia, ib, chain):
    """Re-evaluate every remaining candidate at each step: asserts the chain is
    the argmax (first maximum) and records the winner - runner-up margin."""
    from LLMmap.joint_greedy import _stat_all_pairs
    from LLMmap.joint_statistic import cvar
    nq, npair, nc, _ = XY2.shape
    acc_xy = np.zeros((npair, nc, nc), np.float32)
    acc_xx = np.zeros((XX2.shape[1], nc, nc), np.float32)
    out = []
    for step, q_sel in enumerate(chain):
        obj = np.full(nq, -np.inf)
        for q in range(nq):
            if q in chain[:step]:
                continue
            obj[q] = cvar(_stat_all_pairs(acc_xy, acc_xx, ia, ib, XY2[q], XX2[q], "energy"), L.GAMMA)
        assert int(np.argmax(obj)) == q_sel, f"STOP: replay disagrees at step {step}"
        srt = np.sort(obj[np.isfinite(obj)])[::-1]
        runner = int(np.argsort(-obj, kind="stable")[1])
        out.append(dict(k=step + 1, query=int(q_sel), objective=float(srt[0]),
                        runner_up=int(runner), margin=float(srt[0] - srt[1]),
                        n_exact_ties=int(np.sum(obj == srt[0]) - 1)))
        acc_xy += XY2[q_sel]
        acc_xx += XX2[q_sel]
    return out


def joint_stat(XY2, XX2, ia, ib, chain):
    """Scale-free joint energy per pair for the query set `chain`."""
    from LLMmap.joint_greedy import _stat_all_pairs
    nq, npair, nc, _ = XY2.shape
    acc_xy = np.zeros((npair, nc, nc), np.float32)
    acc_xx = np.zeros((XX2.shape[1], nc, nc), np.float32)
    for q in chain[:-1]:
        acc_xy += XY2[q]
        acc_xx += XX2[q]
    return _stat_all_pairs(acc_xy, acc_xx, ia, ib, XY2[chain[-1]], XX2[chain[-1]], "energy")


def rank_low(x):
    """1 = the lowest value among all pairs (least separated)."""
    r = np.empty(len(x), int)
    r[np.argsort(x, kind="stable")] = np.arange(1, len(x) + 1)
    return r


# ------------------------------------------------------------------ main
def main():
    os.makedirs(L.OUT, exist_ok=True)
    mem = host_mem_gb()
    log(f"host MemTotal {mem:.1f} GB; node {os.uname().nodename}")
    assert mem >= 40, "STOP: host memory < 40 GB -- resubmit selection on gg (P1 S0.3)"
    s0 = dict(schema="d029-s0-v1", corpus="v2", host_mem_gb=round(mem, 1))

    s0["inputs"] = L.assert_inputs()
    log("S0.1 frozen inputs asserted")
    models = L.universe()
    uni = json.load(open(L.UNIVERSE))
    reuse = json.load(open("./results/D028/reuse.json"))
    assert len(reuse) == 34 and set(reuse) <= set(uni["v1"])

    s0["ml_query_emb"] = embed_ml_queries()
    log("S0.2 ML query embeddings written")

    cubes = L.load_v2(models)
    log("v2 cubes loaded (pool + ML, 3 splits)")
    s0["loader_checks"] = dict(ml_prompt_conf_rows_checked=check_prompt_conf(models))
    log(f"S0.L prompt_conf/text equality on {s0['loader_checks']['ml_prompt_conf_rows_checked']} ML rows")
    s0["loader_checks"]["v1_byte_equal_models_per_split"] = check_v1_bytes(cubes, reuse)
    log("S0.L byte equality vs d007_lib on 34 reused models x 3 splits")
    s0["loader_checks"]["pair_worker_vs_frozen_tensor"] = check_pair_worker(cubes, models)
    log("S0.L pair_worker reproduces the frozen tensor on 3 pairs")
    for sp in ("val", "test"):                       # selection uses S_build only
        del cubes[sp]
    gc.collect()

    # ---------------- S0.V code path on v1 (exact)
    from d007_lib import load_corpus
    from d010_select import precompute
    from d008_lib import load_tensor
    from LLMmap.greedy_cover import greedy_cover
    m1, X1 = load_corpus(pool="build")
    p1, ia1, ib1, _ = L.pair_index(m1)
    XX2v1, XY2v1 = precompute(X1, m1, p1)
    ch1, _, _ = jg(XY2v1, XX2v1, ia1, ib1, 16)
    ref = json.load(open("./results/D025/selection.json"))["chains"]["GLOBAL"]
    assert ch1 == ref, f"STOP: JG wrapper on v1 {ch1} != D025 GLOBAL {ref}"
    S1, _ = load_tensor()
    gc1 = greedy_cover(S1, 8, gamma=L.GAMMA, agg="max")
    ref_gc = json.load(open("./results/D013/gamma_selection.json"))["chains"]["0.1"]
    assert gc1 == ref_gc, f"STOP: GC wrapper on v1 {gc1} != D013 {ref_gc}"
    s0["v1_code_path"] = dict(jg_k16=ch1, equals_d025_global=True, gc_k8=gc1, equals_d013_gamma01=True)
    log(f"S0.V v1 code path reproduced: JG {ch1[:8]}..., GC {gc1}")
    del X1, XX2v1, XY2v1, S1
    gc.collect()

    # ---------------- S0.3 v2 chains
    pairs, ia, ib, pidx = L.pair_index(models)
    t = time.time()
    XX2, XY2 = precompute(cubes["build"][1], models, pairs)
    assert XY2.shape == (L.NCAND, 3570, 75, 75)
    s0["footprint"] = dict(XY2_GB=round(XY2.nbytes / 1e9, 2), XX2_GB=round(XX2.nbytes / 1e9, 2),
                           precompute_min=round((time.time() - t) / 60, 2))
    log(f"precompute {s0['footprint']}")
    del cubes
    gc.collect()

    t = time.time()
    ch_jg, tr_jg, pk_jg = jg(XY2[:L.NPOOL], XX2[:L.NPOOL], ia, ib, 16)
    w_jg = time.time() - t
    log(f"JG chain {ch_jg} ({w_jg / 60:.1f} min), peak k {pk_jg}")
    t = time.time()
    ch_ml, tr_ml, pk_ml = jg(XY2, XX2, ia, ib, 16)
    w_ml = time.time() - t
    log(f"JG+ML chain {[L.cand_to_id(c) for c in ch_ml]} ({w_ml / 60:.1f} min), peak k {pk_ml}")
    S = np.load(L.TENSOR)
    ch_gc, tr_gc = greedy_cover(S, 8, gamma=L.GAMMA, agg="max", return_trace=True)
    log(f"GC chain {ch_gc}")
    comp = ch_jg[:8] + list(range(L.NPOOL, L.NCAND))
    m_jg = replay_margins(XY2[:L.NPOOL], XX2[:L.NPOOL], ia, ib, ch_jg)
    m_ml = replay_margins(XY2, XX2, ia, ib, ch_ml)
    log("replay margins recorded (chains re-verified as argmax)")

    prefix = next((i for i, (a, b) in enumerate(zip(ch_jg, ch_ml)) if a != b), 16)
    ml_sel = [dict(k=i + 1, cand=c, id=L.cand_to_id(c)) for i, c in enumerate(ch_ml) if c >= L.NPOOL]
    mlq = {q["id"]: q for q in json.load(open(L.ML_POOL))["queries"]}
    for d in ml_sel:
        d.update(lang=mlq[d["id"]]["lang"], task=mlq[d["id"]]["task"])
    q0 = json.load(open("./confs/queries/pool_v1.json"))

    def text(c):
        return mlq[L.cand_to_id(c)]["text"] if c >= L.NPOOL else q0["queries"][c]["text"][:160]

    arms = dict(
        PAPER8=dict(cands=list(range(8)), k_train=list(range(1, 9))),
        JG=dict(cands=ch_jg, k_train=list(range(1, 9)) + [12, 16], trace=tr_jg, peak_k=pk_jg,
                margins=m_jg, wall_min=round(w_jg / 60, 2)),
        **{"JG+ML": dict(cands=ch_ml, k_train=list(range(1, 9)) + [12, 16], trace=tr_ml, peak_k=pk_ml,
                         margins=m_ml, wall_min=round(w_ml / 60, 2))},
        GC=dict(cands=[int(q) for q in ch_gc], k_train=list(range(1, 9)), trace=tr_gc),
        JG8_ML16=dict(cands=comp, k_train=[24], note="P1 companion: JG[:8] + all 16 ML probes in id order"))
    for a in arms.values():
        a["ids"] = [L.cand_to_id(c) for c in a["cands"]]
    ov = {f"{x}~{y}": len(set(arms[x]["cands"][:8]) & set(arms[y]["cands"][:8]))
          for x in ("PAPER8", "JG", "JG+ML", "GC") for y in ("PAPER8", "JG", "JG+ML", "GC") if x < y}
    chains = dict(schema="d029-chains-v1", corpus="v2", frozen_before_training=True, gamma=L.GAMMA,
                  candidate_space="0..258 = pool id; 259..274 = ML id 281..296",
                  tie_rule="first maximum (lowest candidate index); pool before ML", models=models,
                  arms=arms, jgml_prefix_equal_to_jg=prefix, ml_selected=ml_sel, overlaps_k8=ov,
                  query_texts={str(L.cand_to_id(c)): text(c) for a in arms.values() for c in a["cands"]},
                  inputs=s0["inputs"], code_commit=L.git_head())
    json.dump(chains, open(f"{L.OUT}/chains.json", "w"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log("chains.json written")

    # ---------------- S0.4 pointers
    tg = L.targets(models)
    for p in tg["H"]:
        assert p in pidx, p
    ptr = {}
    for name, a in arms.items():
        if name == "JG8_ML16":
            continue
        for k in (8, 16):
            if k > len(a["cands"]):
                continue
            c = a["cands"][:k]
            pool_q = [q for q in c if q < L.NPOOL]
            cov = S[pool_q].max(0) if pool_q else np.zeros(S.shape[1])
            js = joint_stat(XY2, XX2, ia, ib, c)
            rc, rj = rank_low(cov), rank_low(js)
            ptr[f"{name}|{k}"] = dict(
                cvar01_cov=float(np.mean(np.sort(cov)[:357])), mean_cov=float(cov.mean()),
                cvar01_joint=float(np.mean(np.sort(js)[:357])),
                pairs={p: dict(cov=round(float(cov[pidx[p]]), 4), cov_rank=int(rc[pidx[p]]),
                               joint=round(float(js[pidx[p]]), 4), joint_rank=int(rj[pidx[p]]))
                       for p in tg["H"]})
    pool_cov = S.max(0)
    rp = rank_low(pool_cov)
    s0["pointers"] = dict(note="descriptive, S_build only; coverage = I4 max over the chain's POOL queries "
                               "(ML probes have no tensor column); joint = scale-free joint energy incl. ML; "
                               "rank 1 = least separated of 3,570. Not trained accuracy.",
                          per_arm_k=ptr,
                          pool_coverage={p: dict(cov=round(float(pool_cov[pidx[p]]), 4),
                                                 rank=int(rp[pidx[p]])) for p in tg["H"]})
    s0["targets"] = dict(T1=tg["T1"], T2=tg["T2"], T3=tg["T3"],
                         strata_n={k: len(v) for k, v in tg["strata"].items()},
                         strata=tg["strata"], mode_mixed=tg["mode_mixed"])
    s0["ml_selected"] = ml_sel
    s0["jgml_prefix_equal_to_jg"] = prefix

    tail = {"all259": pool_cov}
    for name in ("PAPER8", "JG", "GC", "JG+ML"):
        c = [q for q in arms[name]["cands"][:8] if q < L.NPOOL]
        tail[name] = S[c].max(0)
    np.savez_compressed(f"{L.OUT}/tail_k8.npz", **tail)     # plotted by d029_plots.py (llmmap env)
    s0["wall_min"] = round((time.time() - T0) / 60, 1)
    json.dump(s0, open(f"{L.OUT}/s0.json", "w"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log("S0 done")


if __name__ == "__main__":
    main()
