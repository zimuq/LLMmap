"""D027 / S0 step 5 — per-model preflight (P1 S0 + Review A2).

For each model (sequential, restartable; one JSON per model in
results/D027/preflight/):
  1. load through d027_llm.LLMv2 at the pinned revision;
  2. chat template present (own; the v1 Together fallback is the only
     exception) -> else DROP (rule 6);
  3. system role (sentinel); date freeze check where the template calls
     strftime_now (the rendered prompt must carry 2026-09-06 in some format);
  4. thinking check: paper8 on build configs 0,1,2 of the model's config
     source, decoded with skip_special_tokens=False, scanned for reasoning
     markers -> any hit = DROP (rule 4); one raw output quoted;
  5. padding sanity (all models; A2 decides only for the hybrids): paper8 on
     one greedy build config, batch 8 vs one prompt at a time; exact-match
     count and degenerate flags;
  6. throughput: one packed greedy call of 64 prompts (paper8 x build 0..7);
  7. peak memory, load time, effective generation_config.

Usage:
  PYTHONPATH=.:experiments python experiments/d027_preflight.py --models <file> [--env-tag v2|old]
"""
import os
import re
import sys
import json
import time
import argparse
import traceback

import torch

from LLMmap.prompt_configuration import PromptConf
import d006_s4_shard as h
from d027_llm import LLMv2, THINK_MARKERS, TEMPLATE_KWARGS_RECORD

OUT = "./results/D027/preflight"
PAPER8 = list(range(8))
DATE_FORMS = ["2026-09-06", "06 Sep 2026", "6 Sep 2026", "September 06, 2026",
              "September 6, 2026", "Sep 06, 2026", "06 September 2026"]


def spec(m):
    """revision, config source rows (build), and v1 quirks."""
    v1 = {s["model"] for s in json.load(open("./data/corpus_v1/corpus_manifest.json"))["models"]}
    if m in v1:
        st = json.load(open(f"./data/corpus_v1/{m.replace('/', '__')}.status.json"))
        build = {}
        for line in open(f"./data/corpus_v1/{m.replace('/', '__')}.jsonl"):
            d = json.loads(line)
            if d["dataset"] == "build":
                build[d["config_index"]] = d["prompt_conf"]
        return dict(v1=True, revision=st["hf_revision"], build=[build[i] for i in range(75)],
                    trust=m in h.TRUST_REMOTE_CODE, tok_kw=h.EXTRA_TOKENIZER_KWARGS.get(m),
                    fallback=h.CHAT_TEMPLATE_FALLBACK.get(m), extra_markers=[])
    pre = json.load(open("./results/D027/hf_prefacts.json"))[m]
    cf = json.load(open(f"./data/corpus_v2_screen/configs/{m.replace('/', '__')}.json"))
    return dict(v1=False, revision=pre["sha"], build=cf["pools"]["build"], trust=False,
                tok_kw=None, fallback=None, extra_markers=pre.get("think_markers", []))


def degenerate(r):
    r = (r or "").strip()
    if not r:
        return True
    w = r.split()
    g = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return bool(g) and len(set(g)) / len(g) < 0.5


def template_text(tok):
    t = getattr(tok, "chat_template", None)
    if isinstance(t, dict):
        return "\n".join(map(str, t.values()))
    return t or ""


def one(m, queries, env_tag):
    sp = spec(m)
    rec = dict(model=m, env_tag=env_tag, v1=sp["v1"], revision=sp["revision"],
               template_kwargs=TEMPLATE_KWARGS_RECORD, drop=None)
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    try:
        llm = LLMv2(m, sp["revision"], trust_remote_code=sp["trust"], tokenizer_kwargs=sp["tok_kw"],
                    chat_template_fallback=sp["fallback"])
    except Exception as e:
        tb = traceback.format_exc()
        rec.update(drop="cannot load" + (" (needs remote code)" if "trust_remote_code" in tb else ""),
                   error=tb[-3000:], load_s=round(time.time() - t0, 1))
        return rec
    rec["load"] = llm.load_info
    rec["load_s"] = round(time.time() - t0, 1)
    if llm.load_info["hf_revision"] != sp["revision"]:
        rec["revision_mismatch"] = llm.load_info["hf_revision"]
    if llm.load_info["chat_template_source"] == "NONE":
        rec["drop"] = "no own chat template (rule 6)"
        return rec
    rec["supports_system_role"] = llm.supports_system_role
    tt = template_text(llm.tokenizer)
    rec["template_uses_strftime_now"] = "strftime_now" in tt
    ex = llm.make_prompt("", "Hello")
    rec["rendered_example"] = ex[-600:]
    if rec["template_uses_strftime_now"]:
        rec["date_frozen_ok"] = any(d in ex for d in DATE_FORMS) or "strftime_now" not in tt
        if not rec["date_frozen_ok"]:
            # some templates only print the date with a system message absent/present
            ex2 = llm.make_prompt("You are a helpful assistant.", "Hello")
            rec["date_frozen_ok"] = any(d in ex2 for d in DATE_FORMS)
            rec["rendered_example_sys"] = ex2[-600:]
    rec["prompt_has_empty_think_block"] = bool(re.search(r"<think>\s*</think>", ex))

    # ---- thinking check: build configs 0..2, paper8, raw decode
    markers = sorted(set(THINK_MARKERS + sp["extra_markers"]))
    hits, raw_quote = [], None
    for ci in range(3):
        conf = PromptConf.from_dict(sp["build"][ci])
        prompts, hp = zip(*[conf(queries[q], llm) for q in PAPER8])
        outs = llm.generate(list(prompts), hp[0], skip_special_tokens=False,
                            max_new_tokens=h.TOKEN_CEILING)
        for q, o in zip(PAPER8, outs):
            if any(k in o for k in markers):
                hits.append(dict(config=ci, query=q, out=o[:300]))
        if raw_quote is None:
            raw_quote = outs[0][:600]
    rec["thinking"] = dict(markers=markers, hits=hits[:5], n_hits=len(hits), raw_quote=raw_quote)
    if hits:
        rec["drop"] = "reasoning trace in output with thinking off (rule 4)"

    # ---- padding sanity: one greedy build config
    gi = next((i for i, c in enumerate(sp["build"]) if not c["sampling_hparams"]["do_sample"]), None)
    pc = dict(sp["build"][gi if gi is not None else 0])
    if gi is None:
        pc = dict(pc, sampling_hparams=dict(pc["sampling_hparams"], do_sample=False))
    conf = PromptConf.from_dict(pc)
    prompts, hp = zip(*[conf(queries[q], llm) for q in PAPER8])
    batched = llm.generate(list(prompts), hp[0], max_new_tokens=h.TOKEN_CEILING)
    single = [llm.generate([p], hp[0], max_new_tokens=h.TOKEN_CEILING)[0] for p in prompts]
    rec["padding_sanity"] = dict(
        config=gi, forced_greedy=gi is None,
        exact=sum(a == b for a, b in zip(batched, single)), n=len(single),
        degenerate_batched_only=sum(degenerate(a) and not degenerate(b) for a, b in zip(batched, single)),
        example=dict(batched=batched[0][:300], single=single[0][:300]))

    # ---- throughput: packed greedy call of 64
    ps = []
    for ci in range(8):
        c = PromptConf.from_dict(dict(sp["build"][ci], sampling_hparams=dict(do_sample=False)))
        ps += [c(queries[q], llm)[0] for q in PAPER8]
    t = time.time()
    llm.generate(ps, dict(do_sample=False), max_new_tokens=h.TOKEN_CEILING)
    dt = time.time() - t
    rec["throughput"] = dict(n=len(ps), wall_s=round(dt, 2), gen_per_s=round(len(ps) / dt, 3))
    rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    rec["wall_s"] = round(time.time() - t0, 1)
    del llm
    torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--env-tag", default="v2")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    queries = [e["text"] for e in json.load(open(h.Q0))["queries"]]
    for m in [l.strip() for l in open(a.models) if l.strip()]:
        path = f"{OUT}/{m.replace('/', '__')}.{a.env_tag}.json"
        if os.path.exists(path):
            continue
        print(f"== {m}", flush=True)
        try:
            rec = one(m, queries, a.env_tag)
        except Exception:
            rec = dict(model=m, env_tag=a.env_tag, drop="preflight crashed",
                       error=traceback.format_exc()[-3000:])
            torch.cuda.empty_cache()
        json.dump(rec, open(path, "w"), indent=1)
        print(f"   drop={rec.get('drop')} load={rec.get('load_s')}s "
              f"think_hits={rec.get('thinking', {}).get('n_hits')} "
              f"pad={rec.get('padding_sanity', {}).get('exact')}/8 "
              f"gen/s={rec.get('throughput', {}).get('gen_per_s')} mem={rec.get('peak_mem_gb')}",
              flush=True)


if __name__ == "__main__":
    main()
