"""D026 / S1 pre-check (tokenizer only, no generation).

The Review (Call 3b) requires the 1k control probes 279/280 to wrap to <= 2,047
tokens in ALL 125 configs of BOTH twins, before S1. If not, the largest L in
{800, 600} that passes everywhere is used. This decides L once for both twins,
then writes the extension's query file `confs/queries/pool_d026_ext.json`.

Usage:  PYTHONPATH=.:experiments python experiments/d026_s1_precheck.py
"""
import os
import json
import hashlib

from LLMmap.llm import LLM_huggingface
from LLMmap.prompt_configuration import PromptConf
import d026_lib as L

QFILE = "./confs/queries/pool_d026_ext.json"
WINDOW = 2047


def main():
    toks = {}
    for m in L.TWINS:
        rev = L.shard_status(m)["hf_revision"]
        toks[m] = LLM_huggingface(m, tokenizer_only=True,
                                  model_load_kargs=dict(revision=rev))
    ents = {m: L.corpus_entries(m) for m in L.TWINS}
    tried = []
    for cl in (1000, 800, 600):
        L.F3_LENGTHS[-1] = cl
        mx = {}
        for m in L.TWINS:
            llm = toks[m]
            probes, _ = L.all_probes(llm.tokenizer)
            lens = [len(llm.tokenizer(PromptConf.from_dict(d["prompt_conf"])(probes[i], llm)[0],
                                      add_special_tokens=False)["input_ids"])
                    for d in ents[m].values() for i in (279, 280)]
            mx[m] = max(lens)
        tried.append(dict(L=cl, max_wrapped=mx, passes=all(v <= WINDOW for v in mx.values())))
        print(tried[-1], flush=True)
        if tried[-1]["passes"]:
            break
    assert tried[-1]["passes"], "no control L in {1000, 800, 600} fits the window"
    control_L = tried[-1]["L"]

    m0 = L.TWINS[0]
    probes, f3meta = L.all_probes(toks[m0].tokenizer)
    probes1, _ = L.all_probes(toks[L.TWINS[1]].tokenizer)
    assert probes == probes1, "probe texts differ between the twins' tokenizers"
    rows = [dict(id=i, query_index=i - L.FIRST_NEW_ID, family=L.FAMILY[i],
                 text=probes[i], **({"f3": f3meta[i]} if i in f3meta else {}))
            for i in sorted(probes)]
    body = dict(schema="corpus_v1_ext_d026-queries-v1", first_id=L.FIRST_NEW_ID,
                n=len(rows), control_L=control_L, filler=L.FILLER_META,
                needle_literal_asterisks=L.NEEDLE_LITERAL_ASTERISKS,
                generation_groups=dict(short=L.SHORT_IDS,
                                       f3=[dict(L=(control_L if b == [279, 280] else a), ids=b)
                                           for a, b in L.F3_GROUPS]),
                queries=rows)
    body["sha256"] = hashlib.sha256(json.dumps(
        [r["text"] for r in rows]).encode()).hexdigest()
    json.dump(body, open(QFILE, "w"), indent=1)
    json.dump(dict(window=WINDOW, tried=tried, control_L=control_L,
                   queries_sha256=body["sha256"]),
              open(f"{L.OUT}/s1_precheck.json", "w"), indent=1)
    print(f"control L = {control_L}; wrote {QFILE} (sha {body['sha256'][:12]})")


if __name__ == "__main__":
    main()
