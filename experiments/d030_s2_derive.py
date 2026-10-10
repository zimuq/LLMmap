"""D030 / P2 S2.1 + S2.0 — derived representations and the pre-scoring freeze. No score here.

Subcommands (each run in the env named):
  zp200   [gen envs]     ZP-200 texts by re-tokenisation (Call 5): ids = tok(text,
                         add_special_tokens=False); if len > 200, decode(ids[:200],
                         skip_special_tokens=True), else the stored text. Tokenizer = the
                         generation tokenizer (d027_llm.LLMv2, tokenizer_only, manifest revision).
                         --env selects the models whose manifest gen_env matches.
  met_e5  [llmmap-gpu]   frozen I5 e5 on every MET completion (d006_s6_embed settings: fp16 model,
                         max_length 512, mean-pool, no normalisation, stored fp16).
  mpnet   [d030-zp]      official ZeroPrint `_compute_output_embeddings` (mpnet, mean over the 20
                         repeats) per unit, ZP-512 and ZP-200, plus the double-BOS sidecar rows.
  os      [llmmap-gpu]   released LLMmap open-set encoder (`load_LLMmap`) on PAPER8 answers,
                         every S_build and S_test config, corpus v2 (85) and corpus v1 (37); the
                         All template by the released `compute_template`.
  freeze  [llmmap-gpu]   S2.0: MET-e5 sigma (Review B1: within-prompt pairs, used; all pairs,
                         diagnostic), LLMMAP-OS SINGLE configs (D024 rule, effective reading),
                         ZP-200 truncation counts, sha256 of every final.json, v1 test-config
                         pairing check -> results/D030/s2_freeze.json.

Unit layout shared with d030_s2_score: per model 28 units, U = gen_ref, native_tgt, all_ref
(all all_ref config groups pooled into one sample set), test:0 .. test:24; samples in (p, r) order.
Usage:  PYTHONPATH=.:experiments <python> experiments/d030_s2_derive.py <cmd> [...]
"""
import os
import sys
import json
import time
import hashlib
import argparse

import numpy as np

PROJ = "/work/11280/zimuq1/vista/LLMmap-project/LLMmap"
MANIFEST = "./confs/baselines/baseline_ext_v1.json"
DATA = "./data/baseline_ext_v1"
DER = f"{DATA}/derived"
DBLBOS_DIR = f"{DATA}/zp_dblbos_control"
DBLBOS_MODELS = ("meta-llama/Meta-Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3",
                 "google/gemma-2-9b-it", "meta-llama/Llama-2-7b-chat-hf")
FREEZE = "./results/D030/s2_freeze.json"
UNITS = ["gen_ref", "native_tgt", "all_ref"] + [f"test:{s}" for s in range(25)]
NP = {"met": 25, "zp": 10}
PER = {"met": 10, "zp": 20}
ZP_CAP = 200
PAPER8 = list(range(8))
E5 = "intfloat/multilingual-e5-large-instruct"


def slug(m):
    return m.replace("/", "__")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def models():
    return sorted(json.load(open(MANIFEST))["models"])


def read_units(path, method):
    """{unit: [samples sorted by (p, r)]}, every unit = n_prompts x per-prompt samples."""
    u = {}
    for line in open(path):
        d = json.loads(line)
        key = f"test:{d['slot']}" if d["row"] == "test_tgt" else d["row"]
        if d["row"] == "test_tgt":
            assert d["config_ref"] == f"test:{d['slot']}", (path, d["config_ref"])
        if key == "all_ref":
            u.setdefault(key, []).extend(d["samples"])
        else:
            assert key not in u, (path, key)
            u[key] = list(d["samples"])
    for k, s in u.items():
        s.sort(key=lambda x: (x["p"], x["r"]))
        keys = [(x["p"], x["r"]) for x in s]
        assert len(set(keys)) == len(keys), (path, k)
        cnt = np.bincount([x["p"] for x in s], minlength=NP[method])
        assert len(cnt) == NP[method] and (cnt == PER[method]).all(), (path, k, cnt.tolist())
    return u


def units_of(method, model, base=None):
    """base: directory holding {slug}.jsonl; default the dataset's own {DATA}/{method}."""
    base = f"{DATA}/{method}" if base is None else base
    u = read_units(f"{base}/{slug(model)}.jsonl", method)
    assert sorted(u) == sorted(UNITS), (model, sorted(set(UNITS) ^ set(u)))
    return u


# ------------------------------------------------------------------ zp200
def cmd_zp200(a):
    import d006_s4_shard as q
    from d027_llm import LLMv2
    man = json.load(open(MANIFEST))["models"]
    todo = [m for m in models() if man[m]["gen_env"] == a.env]
    os.makedirs(f"{DER}/zp200", exist_ok=True)
    os.makedirs(f"{DER}/zp200_dblbos", exist_ok=True)
    for m in todo:
        out = f"{DER}/zp200/{slug(m)}.jsonl"
        if os.path.exists(out + ".done"):
            continue
        llm = LLMv2(m, man[m]["revision"], tokenizer_only=True,
                    trust_remote_code=m in q.TRUST_REMOTE_CODE,
                    tokenizer_kwargs=q.EXTRA_TOKENIZER_KWARGS.get(m),
                    chat_template_fallback=q.CHAT_TEMPLATE_FALLBACK.get(m))
        tok = llm.tokenizer
        jobs = [(f"{DATA}/zp/{slug(m)}.jsonl", out)]
        if m in DBLBOS_MODELS:
            jobs.append((f"{DBLBOS_DIR}/{slug(m)}.jsonl", f"{DER}/zp200_dblbos/{slug(m)}.jsonl"))
        st = {}
        for src, dst in jobs:
            n = n_tr = 0
            by_row = {}
            with open(dst, "w") as fo:
                for line in open(src):
                    d = json.loads(line)
                    texts = [s["text"] for s in d["samples"]]
                    ids = tok(texts, add_special_tokens=False)["input_ids"]
                    smp = []
                    for s, t, i in zip(d["samples"], texts, ids):
                        tr = len(i) > ZP_CAP
                        smp.append(dict(p=s["p"], r=s["r"], text=tok.decode(i[:ZP_CAP], skip_special_tokens=True)
                                        if tr else t, n_retok=len(i), truncated=tr))
                        n += 1
                        n_tr += tr
                        b = by_row.setdefault(d["row"], [0, 0])
                        b[0] += 1
                        b[1] += tr
                    fo.write(json.dumps(dict(model=m, row=d["row"], slot=d["slot"], config_ref=d["config_ref"],
                                             samples=smp)) + "\n")
            st[os.path.basename(os.path.dirname(dst))] = dict(
                n=n, n_truncated=n_tr, frac_truncated=round(n_tr / n, 4),
                by_row={r: dict(n=v[0], n_truncated=v[1], frac=round(v[1] / v[0], 4)) for r, v in by_row.items()})
        json.dump(dict(model=m, env=a.env, revision=man[m]["revision"], tokenizer=llm.load_info["tokenizer_class"],
                       rule="tok(text, add_special_tokens=False); >200 -> decode(ids[:200], skip_special_tokens=True)",
                       stats=st), open(f"{DER}/zp200/{slug(m)}.stats.json", "w"), indent=1)
        open(out + ".done", "w").write("ok\n")
        print(f"[zp200] {m}: {st['zp200']['frac_truncated']:.3f} truncated", flush=True)


# ------------------------------------------------------------------ met_e5
def cmd_met_e5(a):
    import torch
    from transformers import AutoTokenizer, AutoModel
    from d006_s6_embed import mean_pool, MAX_LEN
    os.makedirs(f"{DER}/met_e5", exist_ok=True)
    tok = AutoTokenizer.from_pretrained(E5)
    mdl = AutoModel.from_pretrained(E5, torch_dtype=torch.float16).cuda().eval()
    rev = getattr(mdl.config, "_commit_hash", None)
    for m in models():
        out = f"{DER}/met_e5/{slug(m)}.npy"
        if os.path.exists(out):
            continue
        u = units_of("met", m)
        texts = [s["text"] for k in UNITS for s in u[k]]
        assert len(texts) == 28 * 250
        E = np.empty((len(texts), 1024), np.float16)
        with torch.no_grad():
            for i in range(0, len(texts), 256):
                b = tok(texts[i:i + 256], padding=True, truncation=True, max_length=MAX_LEN,
                        return_tensors="pt").to("cuda")
                E[i:i + 256] = mean_pool(mdl(**b).last_hidden_state, b["attention_mask"]).float().cpu().numpy()
        np.save(out, E)
        print(f"[met_e5] {m}", flush=True)
    json.dump(dict(model=E5, revision=rev, dtype="fp16 model, fp16 store", max_length=MAX_LEN,
                   pooling="mean, not normalised (d006_s6_embed)", layout="28 units x (25 prompts x 10) in UNITS, (p, r) order"),
              open(f"{DER}/met_e5/_meta.json", "w"), indent=1)


# ------------------------------------------------------------------ mpnet
def cmd_mpnet(a):
    import torch
    import d030_zp as Z
    zp = Z.load_official()
    zp._load_embedding_model()
    for var in ("512", "200"):
        out = f"{DER}/zp_mpnet_{var}.npz"
        if os.path.exists(out):
            continue
        base = None if var == "512" else f"{DER}/zp200"
        ms = models()
        E = np.zeros((len(ms), 28, 10, 768), np.float32)
        t0 = time.time()
        for i, m in enumerate(ms):
            u = units_of("zp", m, base)
            for j, k in enumerate(UNITS):
                E[i, j] = Z.output_embeddings(zp, Z.outputs_from_record(dict(samples=u[k]))).float().cpu().numpy()
            if i % 10 == 0:
                print(f"[mpnet {var}] {i + 1}/{len(ms)} {m} {time.time() - t0:.0f}s", flush=True)
        D = np.zeros((len(DBLBOS_MODELS), 10, 768), np.float32)
        dbase = DBLBOS_DIR if var == "512" else f"{DER}/zp200_dblbos"
        for i, m in enumerate(DBLBOS_MODELS):
            recs = [json.loads(x) for x in open(f"{dbase}/{slug(m)}.jsonl")]
            assert len(recs) == 1 and recs[0]["row"] == "gen_ref_dblbos"
            D[i] = Z.output_embeddings(zp, Z.outputs_from_record(recs[0])).float().cpu().numpy()
        np.savez(out, E=E, dblbos=D, models=np.array(ms), dblbos_models=np.array(DBLBOS_MODELS),
                 units=np.array(UNITS))
        print(f"[mpnet {var}] done {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ os
def corpus_rows(corpus, m):
    rows = {"build": {}, "test": {}}
    for line in open(f"./data/{corpus}/{slug(m)}.jsonl"):
        d = json.loads(line)
        if d["dataset"] in rows:
            rows[d["dataset"]][d["config_index"]] = d
    assert sorted(rows["build"]) == list(range(75)) and sorted(rows["test"]) == list(range(25)), (corpus, m)
    return rows


def cmd_os(a):
    import torch
    from LLMmap.inference import load_LLMmap, InferenceModel
    conf, inf = load_LLMmap("./data/pretrained_models/default", device="cuda", verbose=False)
    uni = json.load(open("./results/D027/universe_v2.json"))
    views = {"v2": ("corpus_v2", uni["models"]), "v1": ("corpus_v1", sorted(uni["v1"]))}
    for v, (corpus, ms) in views.items():
        out = f"{DER}/os_{v}.npz"
        if os.path.exists(out):
            continue
        B = np.zeros((len(ms), 75, 384), np.float32)
        T = np.zeros((len(ms), 25, 384), np.float32)
        TPL = np.zeros((len(ms), 384), np.float64)
        for i, m in enumerate(ms):
            rows = corpus_rows(corpus, m)
            for pool, X, n in (("build", B, 75), ("test", T, 25)):
                for c in range(n):
                    tr = rows[pool][c]["traces"]
                    assert [tr[q][0] for q in PAPER8] == list(inf.queries), (m, pool, c)
                    X[i, c] = InferenceModel.__call__(inf, [tr[q][1] for q in PAPER8]).cpu().numpy()[0]
            # released All template: compute_template over the 75 S_build entries (PAPER8 traces)
            entries = [dict(traces=[rows["build"][c]["traces"][q] for q in PAPER8]) for c in range(75)]
            TPL[i] = inf.compute_template(entries)
            assert np.allclose(TPL[i], B[i].astype(np.float64).mean(0), rtol=0, atol=1e-5), m
        np.savez(out, build=B, test=T, template=TPL, models=np.array(ms), corpus=corpus,
                 distance_default=conf.get("distance_fn", "euclidean"))
        print(f"[os] {v}: {len(ms)} models", flush=True)


# ------------------------------------------------------------------ freeze
def cmd_freeze(a):
    import torch
    from d024_lib import single_config
    ms = models()
    uni = json.load(open("./results/D027/universe_v2.json"))
    assert ms == uni["models"]
    fz = dict(schema="d030-s2-freeze-v1", written=time.strftime("%Y-%m-%dT%H:%M:%S"),
              note="S2.0, written before any S_test score is computed (Review D030/P2, B1/B3)")

    # sigma (Review B1): Gen-reference MET completions, 85 x 250, within-prompt pairs
    G = []
    for m in ms:
        E = np.load(f"{DER}/met_e5/{slug(m)}.npy")
        G.append(E[:250].astype(np.float64))         # unit 0 = gen_ref, (p, r) order
    G = np.stack(G)                                  # (85, 250, 1024)
    assert G.shape == (85, 250, 1024)
    g = torch.tensor(G, device="cuda")
    within = []
    for p in range(25):
        X = g[:, p * 10:(p + 1) * 10].reshape(850, 1024)
        D = torch.cdist(X, X)
        iu = torch.triu_indices(850, 850, 1, device="cuda")
        within.append(D[iu[0], iu[1]])
    w = torch.cat(within)
    assert w.numel() == 25 * 850 * 849 // 2
    X = g.reshape(-1, 1024)
    n = X.shape[0]
    allp = []
    for i in range(0, n, 2048):
        D = torch.cdist(X[i:i + 2048], X)
        r = torch.arange(i, min(i + 2048, n), device="cuda")[:, None]
        allp.append(D[torch.arange(n, device="cuda")[None, :] > r])
    allp = torch.cat(allp)
    assert allp.numel() == n * (n - 1) // 2

    def med(x):
        # exact median of an even-or-odd count (numpy convention)
        s = torch.sort(x).values
        k = s.numel()
        return float(s[k // 2]) if k % 2 else float((s[k // 2 - 1] + s[k // 2]) / 2)
    sig = med(w)
    fz["met_e5_sigma"] = dict(
        sigma=sig, rule="median Euclidean distance over within-prompt pairs of the 21,250 Gen-reference "
                        "completions (85 models x 25 prompts x 10), pooled over prompts (Review B1)",
        n_pairs=int(w.numel()), all_pairs_median_diagnostic=med(allp), n_all_pairs=int(allp.numel()),
        embeddings="derived/met_e5 (fp16 store), distances in float64")
    print(f"[freeze] sigma within {sig:.6f}  all-pairs {fz['met_e5_sigma']['all_pairs_median_diagnostic']:.6f}",
          flush=True)
    del g, X, allp, w

    # SINGLE (D024 rule, 'effective' reading = greedy counts as temperature 0; D024 Review Call 1)
    single = {}
    for v, corpus, mv in (("v2", "corpus_v2", ms), ("v1", "corpus_v1", sorted(uni["v1"]))):
        per = {}
        for m in mv:
            pcs = {c: r["prompt_conf"] for c, r in corpus_rows(corpus, m)["build"].items()}
            c, t, pc = single_config(pcs, "effective")
            per[m] = dict(config_index=int(c), tier=int(t), do_sample=pc["sampling_hparams"]["do_sample"],
                          temperature=pc["sampling_hparams"]["temperature"])
        single[v] = dict(corpus=corpus, per_model=per,
                         tier_counts={str(t): sum(1 for x in per.values() if x["tier"] == t) for t in (1, 2, 3)})
    fz["llmmap_os_single"] = single

    # ZP-200 truncation counts
    tr = {}
    for m in ms:
        s = json.load(open(f"{DER}/zp200/{slug(m)}.stats.json"))
        tr[m] = s["stats"]["zp200"]["frac_truncated"]
        if "zp200_dblbos" in s["stats"]:
            tr[m + " [dblbos]"] = s["stats"]["zp200_dblbos"]["frac_truncated"]
    fz["zp200_frac_truncated"] = tr

    # v1 pairing: baseline test units were generated under the same configs as corpus v1's S_test
    pair_ok = {}
    for m in sorted(uni["v1"]):
        rows = corpus_rows("corpus_v1", m)["test"]
        ok = True
        for meth in ("met", "zp"):
            for line in open(f"{DATA}/{meth}/{slug(m)}.jsonl"):
                d = json.loads(line)
                if d["row"] == "test_tgt":
                    ok &= d["prompt_conf"] == rows[d["slot"]]["prompt_conf"]
        pair_ok[m] = bool(ok)
    assert all(pair_ok.values()), [m for m, v in pair_ok.items() if not v]
    fz["v1_test_configs_equal_corpus_v1"] = dict(n=len(pair_ok), all_equal=True)

    fz["final_json_sha256"] = {f"{d}/{f}": sha(f"{DATA}/{d}/{f}") for d in ("met", "zp", "zp_dblbos_control")
                               for f in sorted(os.listdir(f"{DATA}/{d}")) if f.endswith(".final.json")}
    fz["derived_sha256"] = {f: sha(f"{DER}/{f}") for f in ("zp_mpnet_512.npz", "zp_mpnet_200.npz",
                                                           "os_v2.npz", "os_v1.npz")}
    fz["met_e5_meta"] = json.load(open(f"{DER}/met_e5/_meta.json"))
    json.dump(fz, open(FREEZE, "w"), indent=1)
    print(f"[freeze] written {FREEZE}; SINGLE tiers {({v: single[v]['tier_counts'] for v in single})}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["zp200", "met_e5", "mpnet", "os", "freeze"])
    ap.add_argument("--env")
    a = ap.parse_args()
    os.chdir(PROJ)
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
