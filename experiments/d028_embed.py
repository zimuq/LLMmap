"""D028 / S2 — I5 embeddings for corpus v2 (generated pool shards) and the
multilingual extension (all 85), in the UNCHANGED `llmmap-gpu` env with
d006_s6_embed's function and settings (mean pool, fp16 e5, max_length 512,
batch 64, NOT normalised, stored fp16).

Parity: for every reused model, 50 stored corpus_v1 responses are re-embedded
and compared with corpus_v1/embeddings (max |diff| <= 1e-2), else stop.
Embeds only shards whose *.final.json says COMPLETE. Writes
results/D028/manifest.json (every source, shard and embedding sha256).

Usage:  PYTHONPATH=.:experiments python experiments/d028_embed.py
"""
import os
import json
import hashlib

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from d006_s6_embed import mean_pool, I5_MODEL, I5_DIM, MAX_LEN, BATCH

BLOCKS = {"pool": "./data/corpus_v2", "ml": "./data/corpus_v2_ext_ml"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    tok = AutoTokenizer.from_pretrained(I5_MODEL)
    mdl = AutoModel.from_pretrained(I5_MODEL, torch_dtype=torch.float16).cuda().eval()

    def run(texts):
        out = np.empty((len(texts), I5_DIM), np.float16)
        with torch.no_grad():
            for i in range(0, len(texts), BATCH):
                b = tok(texts[i:i + BATCH], padding=True, truncation=True, max_length=MAX_LEN,
                        return_tensors="pt").to("cuda")
                e = mean_pool(mdl(**b).last_hidden_state, b["attention_mask"])
                out[i:i + len(e)] = e.float().cpu().numpy().astype(np.float16)
        return out

    reuse = json.load(open("./results/D028/reuse.json"))
    uni = json.load(open("./results/D027/universe_v2.json"))["models"]
    man = dict(schema="cdqd-corpus-v2-manifest-v1", universe=uni, models={}, parity={})
    rng = np.random.default_rng(20261003)
    for m in reuse:                                   # I5 parity per reused model
        s = m.replace("/", "__")
        rows = json.load(open(f"./data/corpus_v1/embeddings/{s}.index.json"))["rows"]
        stored = np.load(f"./data/corpus_v1/embeddings/{s}.npy", mmap_mode="r")
        ent = {}
        for line in open(f"./data/corpus_v1/{s}.jsonl"):
            d = json.loads(line)
            ent[(d["dataset"], d["config_index"])] = d
        pick = sorted(rng.choice(len(rows), 50, replace=False).tolist())
        texts = [ent[(rows[r]["pool"], rows[r]["config"])]["traces"][rows[r]["query_index"]][1] for r in pick]
        d = float(np.abs(run(texts).astype(np.float32) - stored[pick].astype(np.float32)).max())
        man["parity"][m] = d
        assert d <= 1e-2, f"I5 parity failed for {m}: {d}"
    print(f"I5 parity: max {max(man['parity'].values())} over {len(reuse)} reused models", flush=True)

    for m in uni:
        s = m.replace("/", "__")
        rec = dict(reuse.get(m, {}))
        for blk, base in BLOCKS.items():
            if blk == "pool" and m in reuse:
                rec["pool_embedding"] = f"{base}/embeddings/{s}.npy -> corpus_v1 (symlink)"
                continue
            fin = f"{base}/{s}.final.json"
            if not os.path.exists(fin) or json.load(open(fin))["status"] != "COMPLETE":
                rec[f"{blk}_status"] = "MISSING/INCOMPLETE"
                continue
            f = json.load(open(fin))
            assert sha(f"{base}/{s}.jsonl") == f["sha256"], f"{m} {blk}: shard changed after finalize"
            os.makedirs(f"{base}/embeddings", exist_ok=True)
            npy = f"{base}/embeddings/{s}.npy"
            if not os.path.exists(npy):
                texts, index = [], []
                for line in open(f"{base}/{s}.jsonl"):
                    d = json.loads(line)
                    for qi, (_, a) in enumerate(d["traces"]):
                        texts.append(a)
                        index.append(dict(pool=d["dataset"], config=d["config_index"], query_index=qi,
                                          empty=not a.strip()))
                np.save(npy, run(texts))
                json.dump(dict(model=m, n=len(index), dim=I5_DIM, dtype="fp16", embedding_model=I5_MODEL,
                               pooling="mean, NOT normalised", max_length=MAX_LEN,
                               query_id_offset=0 if blk == "pool" else 281, rows=index),
                          open(f"{base}/embeddings/{s}.index.json", "w"))
            rec[f"{blk}_status"] = "COMPLETE"
            rec[f"{blk}_final"] = f
            rec[f"{blk}_embedding_sha256"] = sha(npy)
        man["models"][m] = rec
        print(f"  {m}", flush=True)
    # Review A1: generation-mode flags (no compute)
    trio = {"tiiuae/Falcon-H1-3B-Instruct", "tiiuae/Falcon-H1-7B-Instruct", "Qwen/Qwen3.5-4B"}
    for m, rec in man["models"].items():
        rec["gen_mode"] = ("batch1-concurrent" if m in trio else
                           "corpus_v1-reuse" if m in reuse else "d006-padded")
        rec["ml_gen_mode"] = "batch1-concurrent" if m in trio else "d006-padded"
    cls = {m: ("batch1" if r["gen_mode"] == "batch1-concurrent" else "padded")
           for m, r in man["models"].items()}
    npr = {p["pair"] for p in json.load(open("./results/D027/n_prime.json"))["pairs"]}
    hall = set(json.load(open("./results/D028/hard_sets_v2.json"))["H_all"])
    man["mode_class_definition"] = ("batch1 = batch1-concurrent; padded = d006-padded or corpus_v1-reuse "
                                    "(both D006's padded batch-64 procedure)")
    man["mode_mixed"] = dict(
        N_prime=sorted(p for p in npr if len({cls[m] for m in p.split(" | ")}) > 1),
        H_all=sorted(p for p in hall if len({cls[m] for m in p.split(" | ")}) > 1))
    man["complete_pool"] = sum(r.get("pool_status") == "COMPLETE" or m in reuse for m, r in man["models"].items())
    man["complete_ml"] = sum(r.get("ml_status") == "COMPLETE" for r in man["models"].values())
    json.dump(man, open("./results/D028/manifest.json", "w"), indent=1)
    print(f"pool complete {man['complete_pool']}/85, ml complete {man['complete_ml']}/85")


if __name__ == "__main__":
    main()
