"""
D006 / S6 — embed every response with the frozen I5 model.

I5 (CLAUDE.md): the embedding model is frozen at
`intfloat/multilingual-e5-large-instruct`, 1024-d. No substitution without a
full re-run. This is NOT the representation D001/D004 used -- those analysed
LLMmap's shipped corpus through LLMmap's own 384-d classifier. Do not conflate
the two artifacts.

I3 is the binding constraint on the storage format: separability must be
computed point-cloud vs point-cloud. So this stores EVERY response's embedding
-- one row per (model, query, config) -- and never a per-model or per-query
average. D001's collapsed-centroid finding (`templates.py:58`, where
`feats[mask].mean(axis=0)` discarded the intra-model spread) is the cautionary
example: the spread that determines overlap cannot be reconstructed once it has
been averaged away.

NORMALISATION: none, matching the paper's own implementation
(`LLMmap/embedding_model.py:15-26`, which mean-pools and returns the raw vector).
S6 originally L2-normalised. Measured cost of that (S6b, 6 models x 600
responses, 15 pairs): probe AUC 0.8056 normalised vs **0.8564 raw**, better on
**15/15 pairs**, max pair delta 0.082; energy distance 0.015 vs 0.423. Vector
magnitude carries model-discriminative information and normalising discarded it.
Storing raw is also the recoverable direction -- normalising afterwards is one
line, un-normalising is impossible -- the same argument that made C7's ceiling
safe. (The saturation audit is unaffected: 0% of pairs at ceiling either way, so
A3's gate holds on both.)

I4 is not computed here, but the format must not foreclose it: coverage of a
query SET is a MAX over its queries, so the index has to let you address rows by
(model, query) without loading everything. Hence one .npy per model shard plus a
row index, rather than one monolithic array.

Storage: 37 x 259 x 125 = 1,197,875 rows x 1024 dims.
  float32 -> 4.9 GB      float16 -> 2.5 GB
Written to $WORK (no purge). fp16 by default: e5 is served in fp16 and the
downstream statistics are nowhere near that precision floor -- but it is
recorded in the manifest either way, because it is exactly the kind of choice
that is invisible later.

Usage:
    PYTHONPATH=. python experiments/d006_s6_embed.py [--dtype fp16|fp32]
"""
import os
import json
import glob
import time
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

CORPUS_DIR = os.environ.get("D006_CORPUS", "./data/corpus_v1")
EMB_DIR = os.path.join(CORPUS_DIR, "embeddings")
I5_MODEL = "intfloat/multilingual-e5-large-instruct"
I5_DIM = 1024
MAX_LEN = 512          # e5's training window; 200-token responses fit easily
BATCH = 64


def mean_pool(h, mask):
    # Matches LLMmap/embedding_model.py:15-18 exactly, including the clamp.
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()
    store_dtype = np.float16 if args.dtype == "fp16" else np.float32

    os.makedirs(EMB_DIR, exist_ok=True)
    manifest_path = os.path.join(CORPUS_DIR, "corpus_manifest.json")
    if not os.path.exists(manifest_path):
        raise SystemExit("No corpus_manifest.json -- run S5 first. S6 embeds only "
                         "shards S5 has VALIDATED.")
    m = json.load(open(manifest_path))
    if m.get("status") != "READY":
        print(f"WARNING: corpus manifest status is {m.get('status')}, not READY. "
              f"Embedding a PARTIAL corpus is fine for inspection, but the result "
              f"MUST NOT be consumed downstream (Call 2).")

    # Embed ONLY shards S5 marked VALIDATED. Globbing *.jsonl would pick up
    # shards a running job is still appending to -- and because this script skips
    # any shard whose .npy already exists, such a truncated embedding would never
    # be corrected. Keying on the manifest makes "complete" the precondition
    # rather than "file present".
    validated = {s["model"].replace("/", "__"): s for s in m["models"]
                 if s["status"] == "VALIDATED"}

    tok = AutoTokenizer.from_pretrained(I5_MODEL)
    mdl = AutoModel.from_pretrained(I5_MODEL, torch_dtype=torch.float16).cuda().eval()

    on_disk = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.jsonl")))
    shards = [p for p in on_disk
              if os.path.basename(p)[:-len(".jsonl")] in validated]
    skipped = len(on_disk) - len(shards)
    print(f"{len(shards)} VALIDATED shard(s) to embed, dtype={args.dtype}"
          + (f" ({skipped} on disk but not validated -- skipped)" if skipped else ""),
          flush=True)

    summary = []
    for jl in shards:
        slug = os.path.basename(jl)[:-len(".jsonl")]
        out_npy = os.path.join(EMB_DIR, f"{slug}.npy")
        out_idx = os.path.join(EMB_DIR, f"{slug}.index.json")
        if os.path.exists(out_npy) and os.path.exists(out_idx):
            print(f"  skip (exists): {slug}", flush=True)
            continue

        texts, index = [], []
        with open(jl) as f:
            for line in f:
                d = json.loads(line)
                for qi, (q, a) in enumerate(d["traces"]):
                    texts.append(a)
                    index.append(dict(pool=d["dataset"], config=d["config_index"],
                                      query_index=qi, empty=not a.strip()))

        t0 = time.time()
        out = np.empty((len(texts), I5_DIM), dtype=store_dtype)
        with torch.no_grad():
            for i in range(0, len(texts), BATCH):
                b = tok(texts[i:i+BATCH], padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt").to("cuda")
                h = mdl(**b).last_hidden_state
                e = mean_pool(h, b["attention_mask"])
                # NO L2 normalisation -- see NORMALISATION note in the docstring.
                out[i:i+len(e)] = e.float().cpu().numpy().astype(store_dtype)

        exp = validated[slug]["n_rows"]
        assert len(texts) == exp, (
            f"{slug}: read {len(texts)} responses but S5 validated {exp}. The "
            f"shard changed under us -- refusing to write a mismatched embedding.")
        np.save(out_npy, out)
        json.dump(dict(model=slug.replace("__", "/"), n=len(index),
                       dim=I5_DIM, dtype=args.dtype,
                       embedding_model=I5_MODEL, pooling="mean, NOT normalised (matches LLMmap/embedding_model.py)",
                       max_length=MAX_LEN,
                       n_empty=sum(1 for r in index if r["empty"]),
                       note="Rows with empty=true are EMPTY model responses, "
                            "kept deliberately (a model going silent on a probe "
                            "is signal). Their embedding is that of the empty "
                            "string; filter explicitly if that is unwanted. "
                            "One row per response. NEVER averaged -- I3 requires "
                            "point-cloud vs point-cloud; the intra-model spread "
                            "cannot be reconstructed after averaging.",
                       rows=index), open(out_idx, "w"))
        dt = time.time() - t0
        summary.append(dict(shard=slug, n=len(texts), wall_s=round(dt, 1),
                            per_s=round(len(texts)/dt, 1),
                            mb=round(out.nbytes/1e6, 1)))
        print(f"  {slug:52s} {len(texts):7,d} rows  {dt/60:5.1f} min  "
              f"{out.nbytes/1e6:7.1f} MB", flush=True)

    # The paper's EmbeddingCache embeds queries as well as responses
    # (dataset.py: handle_one(query); handle_one(resp)), and Phase 4's full
    # pipeline consumes both. 259 vectors, so this is free -- but it is not
    # reconstructible later without re-running the model, so do it now.
    q_npy = os.path.join(EMB_DIR, "_queries.npy")
    if not os.path.exists(q_npy):
        q0 = json.load(open("./confs/queries/pool_v1.json"))
        qt = [e["text"] for e in q0["queries"]]
        qe = np.empty((len(qt), I5_DIM), dtype=store_dtype)
        with torch.no_grad():
            for i in range(0, len(qt), BATCH):
                b = tok(qt[i:i+BATCH], padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt").to("cuda")
                e = mean_pool(mdl(**b).last_hidden_state, b["attention_mask"])
                qe[i:i+len(e)] = e.float().cpu().numpy().astype(store_dtype)
        np.save(q_npy, qe)
        json.dump(dict(n=len(qt), dim=I5_DIM, dtype=args.dtype,
                       embedding_model=I5_MODEL, pooling="mean, NOT normalised",
                       q0_sha256=q0["sha256"]),
                  open(os.path.join(EMB_DIR, "_queries.index.json"), "w"))
        print(f"  embedded {len(qt)} queries -> {q_npy}", flush=True)

    if summary:
        total = sum(s["n"] for s in summary)
        mb = sum(s["mb"] for s in summary)
        print(f"\nembedded {total:,} responses, {mb/1000:.2f} GB")
    json.dump(summary, open(os.path.join(EMB_DIR, "_embed_log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
