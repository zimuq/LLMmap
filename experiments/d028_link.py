"""D028 / S1 (reuse block) — link the 34 reused v1 shards into corpus v2,
unchanged (D028 ## D: "reused means unchanged").

For each reused model: assert the corpus_v1 shard's sha256 equals its D006
status file, then symlink data/corpus_v2/{slug}.jsonl -> ../corpus_v1/{slug}.jsonl
and the I5 embeddings data/corpus_v2/embeddings/{slug}.{npy,index.json} ->
corpus_v1's. Writes results/D028/reuse.json (per model: shard_source
corpus_v1-reuse, gen_env from D027's records, sha256s).

Usage:  python experiments/d028_link.py
"""
import os
import json
import hashlib

REGEN = {"google/gemma-7b-it", "google/gemma-2b-it", "google/gemma-2-9b-it"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    v1 = sorted(s["model"] for s in json.load(open("./data/corpus_v1/corpus_manifest.json"))["models"])
    reuse = [m for m in v1 if m not in REGEN]
    assert len(reuse) == 34
    env = json.load(open("./results/D027/universe_v2.json"))["gen_env"]
    os.makedirs("./data/corpus_v2/embeddings", exist_ok=True)
    out = {}
    for m in reuse:
        s = m.replace("/", "__")
        src = f"./data/corpus_v1/{s}.jsonl"
        st = json.load(open(f"./data/corpus_v1/{s}.status.json"))
        h = sha(src)
        assert h == st["sha256"], f"{m}: corpus_v1 shard changed since D006"
        for a, b in [(f"../corpus_v1/{s}.jsonl", f"./data/corpus_v2/{s}.jsonl"),
                     (f"../../corpus_v1/embeddings/{s}.npy", f"./data/corpus_v2/embeddings/{s}.npy"),
                     (f"../../corpus_v1/embeddings/{s}.index.json", f"./data/corpus_v2/embeddings/{s}.index.json")]:
            if not os.path.lexists(b):
                os.symlink(a, b)
            assert os.path.exists(b), b
        out[m] = dict(shard_source="corpus_v1-reuse", config_source="corpus_v1",
                      gen_env=("D006 " + {"llmmap-gpu-v2": "llmmap-gpu", "llmmap-gpu": "llmmap-gpu",
                                          "llmmap-internlm": "llmmap-internlm"}[env[m]]),
                      template_kwargs=None, sha256=h, n_rows=st["n_rows"],
                      embedding_sha256=sha(f"./data/corpus_v1/embeddings/{s}.npy"))
    json.dump(out, open("./results/D028/reuse.json", "w"), indent=1)
    print(f"linked {len(out)} reused v1 shards (hashes asserted)")


if __name__ == "__main__":
    main()
