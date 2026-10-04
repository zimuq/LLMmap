"""D028 / S1 — generate corpus v2 pool shards and the multilingual extension.

One invocation = one model, one block (pool | ml), optionally one config range
(part) so batch-1 models fit the 48 h wall. Restartable per config.

* Configs: new models = D027 frozen draws (build/val/test, all 125); the three
  regenerated gemma models = their own corpus_v1 rows (replay); reused v1
  models (ml block only) = their corpus_v1 rows.
* Loading/prompting: d027_llm.LLMv2 at the pinned revision (template kwargs,
  S0/second-Review-approved shims) for v2-env models; DeciLM / internlm run
  this same script in their D006 env (ml block only).
* Mode per model (fixed in S0, results/D028/s0.json):
    d006      make_dataset_entries_for_new_llm(..., batch_size=64): D006 exactly
    bucketed  within a config, prompts grouped by exact token length, <= 64
              per call (zero padding)
    batch1    one prompt per call
* Seed: torch.manual_seed(sha256(model|pool|config_index)) before each config.
Rows: cdqd-corpus-v2 / cdqd-corpus-v2-ext-ml (results/D028/migration_note.md).

Usage:
  <env python> experiments/d028_generate.py --model M --block pool|ml --mode d006|bucketed|batch1
                                            [--env-tag v2|old|internlm] [--part i --nparts n]
  <any python> experiments/d028_generate.py --finalize --model M --block pool|ml
"""
import os
import glob
import json
import time
import hashlib
import argparse

QUERY_FILES = {"pool": "./confs/queries/pool_v1.json", "ml": "./confs/queries/pool_d028_ml.json"}
OUTDIR = {"pool": "./data/corpus_v2", "ml": "./data/corpus_v2_ext_ml"}
GEN_ENV = {"v2": "llmmap-gpu-v2", "old": "llmmap-gpu", "internlm": "llmmap-internlm"}
REGEN = {"google/gemma-7b-it", "google/gemma-2b-it", "google/gemma-2-9b-it"}
POOLS = ("build", "val", "test")


def slug(m):
    return m.replace("/", "__")


def sources(m):
    v1 = {s["model"] for s in json.load(open("./data/corpus_v1/corpus_manifest.json"))["models"]}
    if m in v1:
        st = json.load(open(f"./data/corpus_v1/{slug(m)}.status.json"))
        rows = {}
        for line in open(f"./data/corpus_v1/{slug(m)}.jsonl"):
            d = json.loads(line)
            rows[(d["dataset"], d["config_index"])] = d["prompt_conf"]
        return st["hf_revision"], rows, "corpus_v1-replay" if m in REGEN else "corpus_v1", True
    pre = json.load(open("./results/D027/hf_prefacts.json"))[m]
    cf = json.load(open(f"./data/corpus_v2_screen/configs/{slug(m)}.json"))
    assert cf["sha256"] == json.load(open("./results/D027/configs_manifest.json"))["new"][m]["sha256"]
    rows = {(p, i): c for p in POOLS for i, c in enumerate(cf["pools"][p])}
    return pre["sha"], rows, "d027-frozen-draw", False


def seed_of(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16) % 2**31


def generate(a):
    import torch
    from LLMmap.prompt_configuration import PromptConf
    from LLMmap.dataset_maker import make_dataset_entries_for_new_llm
    import d006_s4_shard as h
    from d027_llm import LLMv2, TEMPLATE_KWARGS_RECORD

    m = a.model
    queries = [e["text"] for e in json.load(open(QUERY_FILES[a.block]))["queries"]]
    rev, rows, csrc, is_v1 = sources(m)
    if a.block == "pool":
        assert not is_v1 or m in REGEN, "reused v1 pool shards are never regenerated"
    keys = sorted(rows, key=lambda k: (POOLS.index(k[0]), k[1]))
    if a.nparts > 1:
        keys = keys[a.part::a.nparts]
    os.makedirs(OUTDIR[a.block], exist_ok=True)
    tag = f".part{a.part}of{a.nparts}" if a.nparts > 1 else ""
    path = f"{OUTDIR[a.block]}/{slug(m)}{tag}.jsonl"
    spath = f"{OUTDIR[a.block]}/{slug(m)}{tag}.status.json"
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if len(d.get("traces", [])) == len(queries):
                done.add((d["dataset"], d["config_index"]))
    status = dict(model=m, block=a.block, mode=a.mode, part=a.part, nparts=a.nparts,
                  gen_env=GEN_ENV[a.env_tag], revision=rev, status="RUNNING",
                  started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    json.dump(status, open(spath, "w"), indent=1)
    q = __import__("d006_s4_shard")
    llm = LLMv2(m, rev, trust_remote_code=m in q.TRUST_REMOTE_CODE,
                tokenizer_kwargs=q.EXTRA_TOKENIZER_KWARGS.get(m),
                chat_template_fallback=q.CHAT_TEMPLATE_FALLBACK.get(m))
    status["load"] = llm.load_info
    t0, n = time.time(), 0
    with open(path, "a") as fh:
        for k in keys:
            if k in done:
                continue
            conf = PromptConf.from_dict(rows[k])
            torch.manual_seed(seed_of(m, *k))
            if a.mode == "d006":
                # OOM handling (S1 rerun, disclosed in R): retry THIS config at 32, then 16, with the
                # same seed; the batch actually used is recorded per config. Everything else stays at 64.
                for bs in (h.CORPUS_BATCH, 32, 16):
                    try:
                        torch.manual_seed(seed_of(m, *k))
                        e = make_dataset_entries_for_new_llm(llm, queries, [conf], pool=k[0],
                                                             batch_size=bs, max_new_tokens=h.TOKEN_CEILING)[0]
                        break
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if bs == 16:
                            raise
                if bs != h.CORPUS_BATCH:
                    status.setdefault("reduced_batch_configs", []).append([k[0], k[1], bs])
                outs = [r for _, r in e["traces"]]
            else:
                prompts, hp = zip(*[conf(qq, llm) for qq in queries])
                hp = hp[0]
                if a.mode == "batch1":
                    outs = [llm.generate([p], hp, max_new_tokens=h.TOKEN_CEILING)[0] for p in prompts]
                else:
                    lens = [len(llm.tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
                    outs, groups = [None] * len(prompts), {}
                    for i, L in enumerate(lens):
                        groups.setdefault(L, []).append(i)
                    for idx in groups.values():
                        for s in range(0, len(idx), h.CORPUS_BATCH):
                            ch = idx[s:s + h.CORPUS_BATCH]
                            for i, r in zip(ch, llm.generate([prompts[i] for i in ch], hp,
                                                             max_new_tokens=h.TOKEN_CEILING)):
                                outs[i] = r
            rec = dict(dataset=k[0], llm=m, traces=[[qq, r] for qq, r in zip(queries, outs)],
                       prompt_conf=rows[k], config_index=k[1], model=m, gen_env=GEN_ENV[a.env_tag],
                       template_kwargs=TEMPLATE_KWARGS_RECORD, config_source=csrc,
                       shard_source=("v2-regenerated" if m in REGEN else "v2-generated")
                       if a.block == "pool" else "v2-ext-ml")
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n += len(queries)
    status.update(status="PART_DONE", generated=n, gen_wall_s=round(time.time() - t0, 1),
                  gen_per_s=round(n / max(1e-9, time.time() - t0), 3),
                  finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    json.dump(status, open(spath, "w"), indent=1)
    print(f"[S1] {m} {a.block}{tag}: {n} generated, {status['gen_per_s']} gen/s", flush=True)


def finalize(a):
    """Merge parts (if any), validate, write the final status."""
    m = a.model
    nq = len(json.load(open(QUERY_FILES[a.block]))["queries"])
    _, rows, _, _ = sources(m)
    parts = sorted(glob.glob(f"{OUTDIR[a.block]}/{slug(m)}.part*of*.jsonl"))
    final = f"{OUTDIR[a.block]}/{slug(m)}.jsonl"
    recs = {}
    for f in (parts or [final]):
        for line in open(f):
            d = json.loads(line)
            if len(d["traces"]) == nq:
                recs[(d["dataset"], d["config_index"])] = line if line.endswith("\n") else line + "\n"
    keys = sorted(recs, key=lambda k: (POOLS.index(k[0]), k[1]))
    if parts:
        with open(final, "w") as fh:
            fh.writelines(recs[k] for k in keys)
    empty = 0
    for line in open(final):
        d = json.loads(line)
        assert d["prompt_conf"] == rows[(d["dataset"], d["config_index"])]
        empty += sum(not r.strip() for _, r in d["traces"])
    ok = set(keys) == set(rows) and len(keys) == 125
    st = dict(model=m, block=a.block, status="COMPLETE" if ok else "INCOMPLETE", n_configs=len(keys),
              n_rows=len(keys) * nq, n_empty=empty, empty_rate=round(empty / max(1, len(keys) * nq), 4),
              parts=[os.path.basename(p) for p in parts],
              part_status=[json.load(open(p.replace(".jsonl", ".status.json"))) for p in parts]
              if parts else [json.load(open(final.replace(".jsonl", ".status.json")))],
              sha256=hashlib.sha256(open(final, "rb").read()).hexdigest())
    json.dump(st, open(final.replace(".jsonl", ".final.json"), "w"), indent=1)
    print(f"[final] {m} {a.block}: {st['status']} {st['n_configs']} cfg, empty {st['empty_rate']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--block", required=True, choices=["pool", "ml"])
    ap.add_argument("--mode", default="d006", choices=["d006", "bucketed", "batch1"])
    ap.add_argument("--env-tag", default="v2")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--finalize", action="store_true")
    a = ap.parse_args()
    finalize(a) if a.finalize else generate(a)
