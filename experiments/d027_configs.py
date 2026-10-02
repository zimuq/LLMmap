"""D027 / S0 step 4 — freeze prompt configs (P1 Call 1, approved; TACC_NOTES
Issue 21 standing rule).

* v1 models: config source = their own corpus_v1 rows (replayed through
  PromptConf.from_dict in S1). Recorded here: sha256 of each model's
  (pool, config_index) -> prompt_conf map, so S1 can assert it replays exactly
  that.
* New models: one full 125-config draw each (75 build / 25 val / 25 test), with
  `random.seed(f"20260906:{model}:{pool}")`, in a process with hash
  randomisation OFF (PYTHONHASHSEED=0, asserted), and a FRESH
  PromptConfFactory per model (`_generate_rag_prompt` shuffles the shared RAG
  documents in place, so a shared factory would make a model's draw depend on
  the models drawn before it). S_test is drawn and frozen, never generated here.
* I2 assert per drawn config: every non-empty component (system prompt, CoT,
  RAG template) and the temperature belong to that pool's split; the empty
  system prompt is allowed in every pool (DECISIONS A7); `do_sample` is shared
  (A5).

Usage:  PYTHONHASHSEED=0 PYTHONPATH=.:experiments python experiments/d027_configs.py
"""
import os
import re
import sys
import json
import random
import hashlib

from LLMmap.prompt_configuration import PromptConfFactory, BUILD, VAL, TEST

CONF_DIR = "./confs/prompt_configurations/"
OUT_DIR = "./data/corpus_v2_screen/configs"
MAN = "./results/D027/configs_manifest.json"
SPLIT = {BUILD: 75, VAL: 25, TEST: 25}
SEED = 20260906


def candidates():
    rows = [l for l in open("./docs/D027.md") if re.match(r"^\| \d+ \| ", l)]
    ms = [r.split("|")[2].strip() for r in rows]
    assert len(ms) == 49, len(ms)
    return ms


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def i2_check(pc, pool, conf):
    sp = pc.train_test_split[pool]
    t = conf["sampling_hparams"]["temperature"]
    assert t in pc.sampling_universe_split[pool]["temperature"], (pool, t)
    sysp, cot, rag_t = conf["raw"]
    if sysp:
        assert pc.params["systems"].index(sysp) in sp["systems"], pool
    if cot:
        assert pc.params["cot_prompts"].index(cot) in sp["cot_prompts"], pool
    if rag_t:
        assert pc.params["rag_prompts"].index(list(rag_t) if isinstance(rag_t, tuple) else rag_t) \
            in sp["rag_prompts"] or pc.params["rag_prompts"].index(tuple(rag_t)) in sp["rag_prompts"], pool


def main():
    assert sys.flags.hash_randomization == 0 and os.environ.get("PYTHONHASHSEED") == "0", \
        "run with PYTHONHASHSEED=0 (TACC_NOTES Issue 21)"
    os.makedirs(OUT_DIR, exist_ok=True)
    man = dict(schema="d027-configs-v1", seed=f"{SEED}:{{model}}:{{pool}}",
               pythonhashseed=0, split_sizes=SPLIT, v1={}, new={})

    v1 = sorted(s["model"] for s in json.load(open("./data/corpus_v1/corpus_manifest.json"))["models"])
    for m in v1:
        pcs = {}
        for line in open(f"./data/corpus_v1/{m.replace('/', '__')}.jsonl"):
            d = json.loads(line)
            pcs[f"{d['dataset']}:{d['config_index']}"] = d["prompt_conf"]
        assert len(pcs) == 125
        man["v1"][m] = dict(config_source="corpus_v1-replay", sha256=sha(pcs))

    for m in candidates():
        pc = PromptConfFactory(CONF_DIR)          # fresh per model (RAG shuffle is in place)
        pools = {}
        for pool, n in SPLIT.items():
            random.seed(f"{SEED}:{m}:{pool}")
            confs = [c.to_dict() for c in pc.sample(n, pool=pool)]
            for c in confs:
                i2_check(pc, pool, c)
            pools[pool] = confs
        body = dict(model=m, config_source="d027-frozen-draw", split_schema=pc.split_schema_version,
                    pools=pools)
        body["sha256"] = sha(pools)
        path = f"{OUT_DIR}/{m.replace('/', '__')}.json"
        json.dump(body, open(path, "w"), indent=1)
        greedy = {p: sum(not c["sampling_hparams"]["do_sample"] for c in v) for p, v in pools.items()}
        nosys = {p: sum(c["system_prompt"] == "" for c in v) for p, v in pools.items()}
        man["new"][m] = dict(config_source="d027-frozen-draw", sha256=body["sha256"], path=path,
                             n_greedy=greedy, n_no_system=nosys,
                             n_hparam_groups=len({json.dumps(c["sampling_hparams"], sort_keys=True)
                                                  for p in (BUILD, VAL) for c in pools[p]}))
        print(m, body["sha256"][:12], greedy, flush=True)

    # reproducibility: a second, independent draw of one model must be identical
    m = candidates()[0]
    pc = PromptConfFactory(CONF_DIR)
    again = {}
    for pool, n in SPLIT.items():
        random.seed(f"{SEED}:{m}:{pool}")
        again[pool] = [c.to_dict() for c in pc.sample(n, pool=pool)]
    assert sha(again) == man["new"][m]["sha256"], "redraw differs"
    man["redraw_check"] = dict(model=m, identical=True)
    json.dump(man, open(MAN, "w"), indent=1)
    print(f"froze {len(man['new'])} new-model draws; {len(man['v1'])} v1 replay hashes")


if __name__ == "__main__":
    main()
