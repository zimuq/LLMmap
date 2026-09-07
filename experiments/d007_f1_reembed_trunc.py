"""
D007 / F1 — re-embed the corpus at a 100-token analysis budget.

D006 generated at C7's 200-token CEILING precisely so the analysis budget stays a
separate, revisable knob (A2 verified 100/100 that a 200-token greedy generation
truncated to 100 tokens is exactly the 100-token generation). This produces the
100-token view so `S_probe` can be built at both budgets over the full 666 pairs,
settling which one the tensor freezes.

TRUNCATION IS BY THE GENERATING MODEL'S OWN TOKENIZER, not by characters. "100
tokens" is only equivalent to "generated with max_new_tokens=100" under the
tokenizer that produced them; a character proxy would silently mean different
amounts of text per model, and per-model differences are exactly what the tensor
measures.

That forces a two-pass run: no single environment can load all 37 tokenizers.
`internlm2_5-7b-chat` needs sentencepiece 0.1.99, and under 0.1.99 `EuroLLM-1.7B`
and `Mistral-7B-v0.3` fail with protobuf descriptor errors (D006, verified). So:

    # pass 1 -- 36 models, standard env
    conda activate llmmap-gpu
    PYTHONPATH=.:experiments python experiments/d007_f1_reembed_trunc.py

    # pass 2 -- internlm only, dedicated env
    PYTHONPATH=.:experiments \\
      /work/11280/zimuq1/vista/envs/llmmap-internlm/bin/python \\
      experiments/d007_f1_reembed_trunc.py --only internlm/internlm2_5-7b-chat

Both passes skip shards already written, so re-running is free. The script
refuses to finish silently incomplete: it reports which models it could not
tokenize so the second pass is an explicit step, not a forgotten one.
"""
import os
import json
import glob
import time
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

CORPUS = os.environ.get("D006_CORPUS", "./data/corpus_v1")
SRC = os.path.join(CORPUS, "embeddings")
I5_MODEL = "intfloat/multilingual-e5-large-instruct"
I5_DIM, MAX_LEN, BATCH = 1024, 512, 64
TRUST = {"Deci/DeciLM-7B-instruct", "internlm/internlm2_5-7b-chat"}
SLOW = {"internlm/internlm2_5-7b-chat"}


def mean_pool(h, mask):
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    out_dir = f"{SRC}_tok{args.budget}"
    os.makedirs(out_dir, exist_ok=True)

    man = json.load(open(os.path.join(CORPUS, "corpus_manifest.json")))
    assert man["status"] == "READY", f"corpus is {man['status']}, not READY"
    api = os.environ.get("HUGGINGFACE_API_KEY")

    tok5 = AutoTokenizer.from_pretrained(I5_MODEL)
    e5 = AutoModel.from_pretrained(I5_MODEL, torch_dtype=torch.float16).cuda().eval()

    todo = [s for s in man["models"] if s["status"] == "VALIDATED"]
    if args.only:
        todo = [s for s in todo if s["model"] == args.only]
    skipped = []

    for s in todo:
        m = s["model"]
        slug = m.replace("/", "__")
        if os.path.exists(f"{out_dir}/{slug}.npy"):
            continue
        kw = dict(trust_remote_code=True) if m in TRUST else {}
        tkw = dict(use_fast=False) if m in SLOW else {}
        try:
            gtok = AutoTokenizer.from_pretrained(m, token=api, legacy=False,
                                                 **kw, **tkw)
        except Exception as e:
            skipped.append((m, f"{type(e).__name__}: {str(e)[:80]}"))
            print(f"  SKIP {m}: {type(e).__name__} (needs the other env)", flush=True)
            continue

        texts, index = [], []
        with open(s["shard_path"]) as f:
            for line in f:
                d = json.loads(line)
                for qi, (_, a) in enumerate(d["traces"]):
                    ids = gtok(a, add_special_tokens=False).input_ids[:args.budget]
                    t = gtok.decode(ids, skip_special_tokens=True)
                    texts.append(t)
                    index.append(dict(pool=d["dataset"], config=d["config_index"],
                                      query_index=qi, empty=not t.strip()))
        assert len(texts) == s["n_rows"], f"{m}: {len(texts)} != {s['n_rows']}"

        t0 = time.time()
        arr = np.empty((len(texts), I5_DIM), dtype=np.float16)
        with torch.no_grad():
            for i in range(0, len(texts), BATCH):
                b = tok5(texts[i:i+BATCH], padding=True, truncation=True,
                         max_length=MAX_LEN, return_tensors="pt").to("cuda")
                e = mean_pool(e5(**b).last_hidden_state, b["attention_mask"])
                arr[i:i+len(e)] = e.float().cpu().numpy().astype(np.float16)
        np.save(f"{out_dir}/{slug}.npy", arr)
        json.dump(dict(model=m, n=len(index), dim=I5_DIM, dtype="fp16",
                       embedding_model=I5_MODEL,
                       pooling="mean, NOT normalised",
                       token_budget=args.budget,
                       truncation="by the GENERATING model's own tokenizer",
                       rows=index), open(f"{out_dir}/{slug}.index.json", "w"))
        print(f"  {slug:52s} {len(texts):7,d} rows  {(time.time()-t0)/60:4.1f} min",
              flush=True)

    done = len(glob.glob(f"{out_dir}/*.npy"))
    print(f"\n{done}/37 shards embedded at {args.budget} tokens -> {out_dir}")
    if skipped:
        print(f"{len(skipped)} SKIPPED (run the second pass in the other env):")
        for m, why in skipped:
            print(f"   {m}: {why}")
    if done < 37:
        print("INCOMPLETE -- D007's F1 comparison needs all 37; do not proceed "
              "until the second pass has run.")


if __name__ == "__main__":
    main()
