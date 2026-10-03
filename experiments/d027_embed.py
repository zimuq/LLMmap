"""D027 / S1 embed — I5 embeddings of the screen (and of the (f1) replays), in the
UNCHANGED `llmmap-gpu` env (P1 Call 2: generation-only v2 env).

Same function and settings as d006_s6_embed.py (mean_pool imported; e5 fp16;
max_length 512; batch 64; NOT normalised; stored fp16). Parity first: 50
stored corpus_v1 responses re-embedded must match corpus_v1/embeddings to
<= 1e-2 (max |diff|), else stop.

Writes data/corpus_v2_screen/embeddings/{slug}.npy + .index.json (rows: pool,
config, query_index, empty) for every COMPLETE screen shard, f1_{slug}.npy for
the replays, and results/D027/manifest.json.

Usage:  PYTHONPATH=.:experiments python experiments/d027_embed.py
"""
import os
import json
import glob
import time
import hashlib

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from d006_s6_embed import mean_pool, I5_MODEL, I5_DIM, MAX_LEN, BATCH

SCR = "./data/corpus_v2_screen"
EMB = f"{SCR}/embeddings"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    os.makedirs(EMB, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(I5_MODEL)
    mdl = AutoModel.from_pretrained(I5_MODEL, torch_dtype=torch.float16).cuda().eval()

    def run(texts):
        out = np.empty((len(texts), I5_DIM), np.float16)
        with torch.no_grad():
            for i in range(0, len(texts), BATCH):
                b = tok(texts[i:i + BATCH], padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt").to("cuda")
                e = mean_pool(mdl(**b).last_hidden_state, b["attention_mask"])
                out[i:i + len(e)] = e.float().cpu().numpy().astype(np.float16)
        return out

    # ---- I5 parity on stored corpus_v1 responses
    m0 = "Qwen__Qwen2-7B-Instruct"
    rows = json.load(open(f"./data/corpus_v1/embeddings/{m0}.index.json"))["rows"]
    stored = np.load(f"./data/corpus_v1/embeddings/{m0}.npy", mmap_mode="r")
    ent = {}
    for line in open(f"./data/corpus_v1/{m0}.jsonl"):
        d = json.loads(line)
        ent[(d["dataset"], d["config_index"])] = d
    pick = sorted(np.random.default_rng(20261002).choice(len(rows), 50, replace=False).tolist())
    texts = [ent[(rows[r]["pool"], rows[r]["config"])]["traces"][rows[r]["query_index"]][1] for r in pick]
    parity = float(np.abs(run(texts).astype(np.float32) - stored[pick].astype(np.float32)).max())
    assert parity <= 1e-2, f"I5 path drifted: {parity}"
    print(f"I5 parity max |diff| = {parity}", flush=True)

    man = dict(schema="cdqd-corpus-v2-screen-manifest-v1", i5_parity_max_abs=parity,
               embedding=dict(model=I5_MODEL, pooling="mean, NOT normalised", dtype="fp16",
                              max_length=MAX_LEN, env="llmmap-gpu (unchanged)"), models={})
    for sp in sorted(glob.glob(f"{SCR}/*.status.json")):
        st = json.load(open(sp))
        m, s = st["model"], sp.split("/")[-1][:-len(".status.json")]
        rec = {k: st.get(k) for k in ("status", "gen_env", "config_source", "batch1", "revision",
                                      "n_rows", "n_empty", "empty_rate", "sha256", "gen_per_s",
                                      "pack", "oom_halvings", "wall_s")}
        rec["compat_shims"] = st.get("load", {}).get("compat_shims")
        rec["f1"] = st.get("f1")
        man["models"][m] = rec
        if st.get("status") != "COMPLETE":
            continue
        assert sha(f"{SCR}/{s}.jsonl") == st["sha256"], f"{m}: shard changed after validation"
        npy = f"{EMB}/{s}.npy"
        if not os.path.exists(npy):
            texts, index = [], []
            for line in open(f"{SCR}/{s}.jsonl"):
                d = json.loads(line)
                for qi, (_, a) in enumerate(d["traces"]):
                    texts.append(a)
                    index.append(dict(pool=d["dataset"], config=d["config_index"], query_index=qi,
                                      empty=not a.strip()))
            np.save(npy, run(texts))
            json.dump(dict(model=m, n=len(index), dim=I5_DIM, dtype="fp16", embedding_model=I5_MODEL,
                           pooling="mean, NOT normalised", max_length=MAX_LEN, queries="paper8 = pool ids 0..7",
                           rows=index), open(f"{EMB}/{s}.index.json", "w"))
        rec["embedding_sha256"] = sha(npy)
        f1 = f"{SCR}/f1/{s}.jsonl"
        if os.path.exists(f1):
            fnpy = f"{EMB}/f1_{s}.npy"
            if not os.path.exists(fnpy):
                texts, index = [], []
                for line in open(f1):
                    d = json.loads(line)
                    for qi, (_, a) in enumerate(d["traces"]):
                        texts.append(a)
                        index.append(dict(config=d["config"], query_index=qi))
                np.save(fnpy, run(texts))
                json.dump(dict(model=m, rows=index), open(f"{EMB}/f1_{s}.index.json", "w"))
            rec["f1_embedding_sha256"] = sha(fnpy)
        print(f"  {m}", flush=True)
    n_ok = sum(r["status"] == "COMPLETE" for r in man["models"].values())
    man["n_complete"] = n_ok
    json.dump(man, open("./results/D027/manifest.json", "w"), indent=1)
    print(f"embedded {n_ok} complete shards -> results/D027/manifest.json", flush=True)


if __name__ == "__main__":
    main()
