"""D028 / S0 steps 3-4 (GPU) — back-translation check, and the faster-mode
parity test for the D027 batch-1 models.

--task backtranslate
    google/gemma-3-12b-it (D027 revision, llmmap-gpu-v2, d027_llm.LLMv2), greedy,
    user turn "Translate into English:\\n\\n<probe>", no system prompt, 200 tokens.
    Output per probe 281-296 -> results/D028/s0_backtranslation.json.

--task parity --model M --mode {bucketed|d006}
    Greedy, the model's first 2 greedy S_build configs (do_sample forced False on
    the first configs if it has fewer than 2), all 259 pool queries:
      reference  = batch 1 (one prompt per generate call)
      candidate  = 'bucketed': within a config, prompts grouped by EXACT token
                   length, <= 64 per call -> zero padding;
                   'd006':     D006's procedure, 259 queries in chunks of 64
                   (left-padded) -- the attention-model baseline.
    Exact-match rate candidate vs reference, degenerate-in-candidate-only count,
    and gen/s of both modes -> results/D028/s0_parity/{slug}.json.

Usage:  PYTHONPATH=.:experiments <v2 python> experiments/d028_s0.py --task ...
"""
import os
import json
import time
import argparse

import torch

from LLMmap.prompt_configuration import PromptConf
import d006_s4_shard as h
from d027_llm import LLMv2

OUT = "./results/D028"
ML_IDS = list(range(281, 297))


def degenerate(r):
    r = (r or "").strip()
    if not r:
        return True
    w = r.split()
    g = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return bool(g) and len(set(g)) / len(g) < 0.5


def revision_and_build(m):
    v1 = {s["model"] for s in json.load(open("./data/corpus_v1/corpus_manifest.json"))["models"]}
    if m in v1:
        st = json.load(open(f"./data/corpus_v1/{m.replace('/', '__')}.status.json"))
        b = {}
        for line in open(f"./data/corpus_v1/{m.replace('/', '__')}.jsonl"):
            d = json.loads(line)
            if d["dataset"] == "build":
                b[d["config_index"]] = d["prompt_conf"]
        return st["hf_revision"], [b[i] for i in range(75)]
    pre = json.load(open("./results/D027/hf_prefacts.json"))[m]
    cf = json.load(open(f"./data/corpus_v2_screen/configs/{m.replace('/', '__')}.json"))
    return pre["sha"], cf["pools"]["build"]


def backtranslate(probes):
    m = "google/gemma-3-12b-it"
    rev, _ = revision_and_build(m)
    llm = LLMv2(m, rev)
    out = {}
    for i in ML_IDS:
        p = llm.make_prompt("", f"Translate into English:\n\n{probes[i]['text']}")
        out[str(i)] = dict(lang=probes[i]["lang"], task=probes[i]["task"], text=probes[i]["text"],
                           back_translation=llm.generate([p], dict(do_sample=False),
                                                         max_new_tokens=h.TOKEN_CEILING)[0].strip())
        print(i, out[str(i)]["back_translation"][:120], flush=True)
    json.dump(dict(model=m, revision=rev, env="llmmap-gpu-v2", decoding="greedy",
                   prompt="Translate into English:\\n\\n<probe>", rows=out),
              open(f"{OUT}/s0_backtranslation.json", "w"), indent=1, ensure_ascii=False)


def gen_batch1(llm, prompts, hp):
    return [llm.generate([p], hp, max_new_tokens=h.TOKEN_CEILING)[0] for p in prompts]


def gen_bucketed(llm, prompts, hp):
    lens = [len(llm.tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    out = [None] * len(prompts)
    groups = {}
    for i, n in enumerate(lens):
        groups.setdefault(n, []).append(i)
    for n, idx in groups.items():
        for s in range(0, len(idx), h.CORPUS_BATCH):
            chunk = idx[s:s + h.CORPUS_BATCH]
            res = llm.generate([prompts[i] for i in chunk], hp, max_new_tokens=h.TOKEN_CEILING)
            for i, r in zip(chunk, res):
                out[i] = r
    return out, dict(n_buckets=len(groups), mean_bucket=round(len(prompts) / len(groups), 2))


def gen_d006(llm, prompts, hp):
    out = []
    for s in range(0, len(prompts), h.CORPUS_BATCH):
        out += llm.generate(prompts[s:s + h.CORPUS_BATCH], hp, max_new_tokens=h.TOKEN_CEILING)
    return out, {}


def parity(m, mode, queries):
    os.makedirs(f"{OUT}/s0_parity", exist_ok=True)
    rev, build = revision_and_build(m)
    llm = LLMv2(m, rev)
    greedy = [i for i, c in enumerate(build) if not c["sampling_hparams"]["do_sample"]][:2]
    forced = 2 - len(greedy)
    cfgs = greedy + [i for i in range(75) if i not in greedy][:forced]
    rec = dict(model=m, mode=mode, configs=cfgs, forced_greedy=forced, per_config=[],
               load=llm.load_info)
    tot = dict(ref_s=0.0, cand_s=0.0, n=0, exact=0, degen_cand_only=0, degen_ref=0)
    for c in cfgs:
        conf = PromptConf.from_dict(dict(build[c], sampling_hparams=dict(do_sample=False)))
        prompts = [conf(q, llm)[0] for q in queries]
        hp = dict(do_sample=False)
        t = time.time(); ref = gen_batch1(llm, prompts, hp); tr = time.time() - t
        t = time.time()
        try:
            cand, info = (gen_bucketed if mode == "bucketed" else gen_d006)(llm, prompts, hp)
            err = None
        except Exception as e:
            cand, info, err = [""] * len(prompts), {}, repr(e)[:300]
        tc = time.time() - t
        ex = sum(a == b for a, b in zip(cand, ref))
        dg = sum(degenerate(a) and not degenerate(b) for a, b in zip(cand, ref))
        rec["per_config"].append(dict(config=c, exact=ex, n=len(ref), degenerate_cand_only=dg,
                                      degenerate_ref=sum(map(degenerate, ref)), error=err,
                                      ref_gen_per_s=round(len(ref) / tr, 3), cand_gen_per_s=round(len(ref) / tc, 3), **info))
        tot["ref_s"] += tr; tot["cand_s"] += tc; tot["n"] += len(ref); tot["exact"] += ex
        tot["degen_cand_only"] += dg
        print(f"  {m} cfg {c}: exact {ex}/{len(ref)} degen {dg} ref {len(ref)/tr:.2f} gen/s "
              f"cand {len(ref)/tc:.2f} gen/s {info} {err or ''}", flush=True)
    rec.update(exact_rate=tot["exact"] / tot["n"], degenerate_cand_only=tot["degen_cand_only"],
               ref_gen_per_s=round(tot["n"] / tot["ref_s"], 3), cand_gen_per_s=round(tot["n"] / tot["cand_s"], 3))
    json.dump(rec, open(f"{OUT}/s0_parity/{m.replace('/', '__')}.{mode}.json", "w"), indent=1)
    del llm
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["backtranslate", "parity"])
    ap.add_argument("--models", default="")
    ap.add_argument("--mode", default="bucketed", choices=["bucketed", "d006"])
    a = ap.parse_args()
    if a.task == "backtranslate":
        probes = {r["id"]: r for r in json.load(open("./confs/queries/pool_d028_ml.json"))["queries"]}
        backtranslate(probes)
    else:
        queries = [e["text"] for e in json.load(open(h.Q0))["queries"]]
        for m in a.models.split(","):
            parity(m, a.mode, queries)


if __name__ == "__main__":
    main()
