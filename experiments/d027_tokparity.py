"""D027 / S0 step 6 — tokenisation parity, old env vs new env (P1 Call 5 (f2)).

For every v1 model, render every corpus_v1 prompt (125 configs x 259 queries)
through d027_llm.render (template kwargs incl. the frozen date -- so the old
env reproduces what corpus_v1 rendered on 2026-09-06), then tokenise it as
`LLM_huggingface.generate` does (add_special_tokens=False). Stores two 64-bit
hashes per prompt: of the rendered TEXT and of the token IDS. Comparing the two
env files separates template drift from tokenizer drift. CPU only.

Usage (one run per env; then --compare):
  <env python> experiments/d027_tokparity.py --env-tag old|v2|internlm [--models m1,m2]
  python experiments/d027_tokparity.py --compare
"""
import os
import json
import hashlib
import argparse
import traceback

import numpy as np

from LLMmap.prompt_configuration import PromptConf

OUT = "./results/D027/tokparity"


def h64(b):
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")


def run(env_tag, only):
    import d006_s4_shard as h
    from d027_llm import LLMv2
    os.makedirs(OUT, exist_ok=True)
    queries = [e["text"] for e in json.load(open(h.Q0))["queries"]]
    v1 = sorted(s["model"] for s in json.load(open("./data/corpus_v1/corpus_manifest.json"))["models"])
    for m in (only or v1):
        path = f"{OUT}/{m.replace('/', '__')}.{env_tag}.npz"
        if os.path.exists(path):
            continue
        st = json.load(open(f"./data/corpus_v1/{m.replace('/', '__')}.status.json"))
        try:
            llm = LLMv2(m, st["hf_revision"], tokenizer_only=True,
                        trust_remote_code=m in h.TRUST_REMOTE_CODE,
                        tokenizer_kwargs=h.EXTRA_TOKENIZER_KWARGS.get(m),
                        chat_template_fallback=h.CHAT_TEMPLATE_FALLBACK.get(m))
        except Exception:
            json.dump(dict(model=m, env=env_tag, error=traceback.format_exc()[-2000:]),
                      open(path.replace(".npz", ".error.json"), "w"))
            print(m, "LOAD FAILED", flush=True)
            continue
        keys, th, ih = [], [], []
        for line in open(f"./data/corpus_v1/{m.replace('/', '__')}.jsonl"):
            d = json.loads(line)
            conf = PromptConf.from_dict(d["prompt_conf"])
            ps = [conf(q, llm)[0] for q in queries]
            ids = llm.tokenizer(ps, add_special_tokens=False)["input_ids"]
            for qi, (p, t) in enumerate(zip(ps, ids)):
                keys.append((("build", "val", "test").index(d["dataset"]), d["config_index"], qi))
                th.append(h64(p.encode()))
                ih.append(h64(np.asarray(t, np.int64).tobytes()))
        k = np.array(keys, np.int32)
        o = np.lexsort((k[:, 2], k[:, 1], k[:, 0]))
        np.savez_compressed(path, keys=k[o], text=np.array(th, np.uint64)[o],
                            ids=np.array(ih, np.uint64)[o])
        print(m, len(keys), flush=True)


def compare():
    rows = {}
    for f in sorted(os.listdir(OUT)):
        if not f.endswith(".old.npz") and not f.endswith(".internlm.npz"):
            continue
        slug, tag = f.split(".")[0], f.split(".")[1]
        new = f"{OUT}/{slug}.v2.npz"
        if not os.path.exists(new):
            rows[slug] = dict(old_env=tag, v2="missing (load failed or not run)")
            continue
        a, b = np.load(f"{OUT}/{f}"), np.load(new)
        assert (a["keys"] == b["keys"]).all()
        rows[slug] = dict(old_env=tag, n=int(len(a["keys"])),
                          text_differs=int((a["text"] != b["text"]).sum()),
                          ids_differ=int((a["ids"] != b["ids"]).sum()),
                          ids_differ_same_text=int(((a["ids"] != b["ids"]) & (a["text"] == b["text"])).sum()))
    json.dump(rows, open("./results/D027/tokparity.json", "w"), indent=1)
    bad = {k: v for k, v in rows.items() if v.get("text_differs") or v.get("ids_differ") or "v2" in v}
    print(f"{len(rows)} models; {len(bad)} with any difference:")
    for k, v in bad.items():
        print(" ", k, v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-tag")
    ap.add_argument("--models")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.compare:
        compare()
    else:
        run(a.env_tag, a.models.split(",") if a.models else None)
