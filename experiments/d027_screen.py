"""D027 / S1 — the paper8 screen generation (P1 + both Reviews).

Per model (sequential within a job, restartable per model and per config):
  * configs: v1 = its own corpus_v1 rows (replay); new = its frozen draw
    (data/corpus_v2_screen/configs/{slug}.json). Pools build (75) + val (25);
    S_test is never generated.
  * paper8 = pool ids 0..7, prompts through d027_llm.LLMv2 (template kwargs:
    thinking off, frozen strftime_now; S0-approved loader shims).
  * PACKING (Call 6): configs with identical sampling_hparams share calls of
    <= 64 prompts (8 configs x 8 queries), in (pool, config_index) order.
    torch.manual_seed per call from sha256(model|hparams|first config).
    On CUDA OOM the model's chunk is halved and the halving recorded.
  * A2 batch-1 models (Falcon-H1 x2, Qwen3.5-4B; from results/D027/s0.json):
    one prompt per generate call, seeded per prompt.
  * (f1) for v1 models (Call 5): 2 greedy build configs (D024 tier-1
    reference first, then the lowest greedy index), all 259 queries through the
    D006 path at batch 64 (make_dataset_entries_for_new_llm), compared
    byte-for-byte with corpus_v1. Replay outputs saved for the distributional
    1-NN check.
Rows: schema cdqd-corpus-v2-screen = corpus_v1 record + gen_env +
template_kwargs + config_source (results/D027/migration_note.md).

Usage:  <env python> experiments/d027_screen.py --models <list> --env-tag v2|old|internlm
"""
import os
import json
import time
import hashlib
import argparse
import traceback
import collections

import torch

from LLMmap.prompt_configuration import PromptConf
from LLMmap.dataset_maker import make_dataset_entries_for_new_llm
import d006_s4_shard as h
from d027_llm import LLMv2, TEMPLATE_KWARGS_RECORD

OUT = "./data/corpus_v2_screen"
RES = "./results/D027"
PAPER8 = list(range(8))
PACK = 8                      # configs per call -> 64 prompts
GEN_ENV = {"v2": "llmmap-gpu-v2", "old": "llmmap-gpu", "internlm": "llmmap-internlm"}


def slug(m):
    return m.replace("/", "__")


def sources(m):
    """(revision, {(pool, idx): prompt_conf}, config_source, v1-quirks)."""
    v1 = {s["model"] for s in json.load(open("./data/corpus_v1/corpus_manifest.json"))["models"]}
    if m in v1:
        st = json.load(open(f"./data/corpus_v1/{slug(m)}.status.json"))
        rows = {}
        for line in open(f"./data/corpus_v1/{slug(m)}.jsonl"):
            d = json.loads(line)
            if d["dataset"] in ("build", "val"):          # S_test is never generated here
                rows[(d["dataset"], d["config_index"])] = d
        assert len(rows) == 100, (m, len(rows))
        return st["hf_revision"], rows, "corpus_v1-replay", dict(
            trust=m in h.TRUST_REMOTE_CODE, tok_kw=h.EXTRA_TOKENIZER_KWARGS.get(m),
            fallback=h.CHAT_TEMPLATE_FALLBACK.get(m))
    pre = json.load(open(f"{RES}/hf_prefacts.json"))[m]
    cf = json.load(open(f"{OUT}/configs/{slug(m)}.json"))
    man = json.load(open(f"{RES}/configs_manifest.json"))["new"][m]
    assert cf["sha256"] == man["sha256"]
    rows = {(p, i): dict(prompt_conf=c) for p in ("build", "val") for i, c in enumerate(cf["pools"][p])}
    return pre["sha"], rows, "d027-frozen-draw", dict(trust=False, tok_kw=None, fallback=None)


def seed_of(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16) % 2**31


def f1_replay(m, llm, rows, queries, status):
    ref = json.load(open("./results/D024/s0_pilot.json"))["single_tier"]["readings"]["effective"][m]["config_index"]
    greedy = sorted(i for (p, i), d in rows.items()
                    if p == "build" and d["prompt_conf"]["sampling_hparams"]["do_sample"] is False)
    cfgs = ([ref] if ref in greedy else []) + [c for c in greedy if c != ref]
    cfgs = cfgs[:2]
    out, rec = [], []
    for c in cfgs:
        stored = rows[("build", c)]
        e = make_dataset_entries_for_new_llm(llm, queries, [PromptConf.from_dict(stored["prompt_conf"])],
                                             pool="build", batch_size=h.CORPUS_BATCH,
                                             max_new_tokens=h.TOKEN_CEILING)[0]
        same = [a == b for (_, a), (_, b) in zip(e["traces"], stored["traces"])]
        rec.append(dict(config=c, reference=(c == ref), identical=int(sum(same)), n=len(same),
                        identical_paper8=int(sum(same[:8]))))
        out.append(dict(config=c, traces=e["traces"]))
    os.makedirs(f"{OUT}/f1", exist_ok=True)
    with open(f"{OUT}/f1/{slug(m)}.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    status["f1"] = dict(configs=rec, identical_rate=sum(r["identical"] for r in rec) / max(1, sum(r["n"] for r in rec)))


def run_model(m, env_tag, batch1, queries):
    path, spath = f"{OUT}/{slug(m)}.jsonl", f"{OUT}/{slug(m)}.status.json"
    if os.path.exists(spath) and json.load(open(spath)).get("status") == "COMPLETE":
        return
    rev, rows, csrc, q = sources(m)
    status = dict(model=m, status="RUNNING", gen_env=GEN_ENV[env_tag], config_source=csrc,
                  template_kwargs=TEMPLATE_KWARGS_RECORD, batch1=batch1, revision=rev,
                  started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    json.dump(status, open(spath, "w"), indent=1)
    t0 = time.time()
    llm = LLMv2(m, rev, trust_remote_code=q["trust"], tokenizer_kwargs=q["tok_kw"],
                chat_template_fallback=q["fallback"])
    status["load"] = llm.load_info
    status["load_s"] = round(time.time() - t0, 1)
    assert llm.load_info.get("hf_revision") in (rev, None), (llm.load_info.get("hf_revision"), rev)

    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if len(d.get("traces", [])) == 8:
                done.add((d["dataset"], d["config_index"]))
    todo = sorted((k for k in rows if k not in done), key=lambda k: (("build", "val").index(k[0]), k[1]))
    groups = collections.OrderedDict()
    for k in todo:
        groups.setdefault(json.dumps(rows[k]["prompt_conf"]["sampling_hparams"], sort_keys=True), []).append(k)
    pack, halvings, t1, n_gen = PACK, 0, time.time(), 0
    with open(path, "a") as fh:
        for hkey, keys in groups.items():
            hp = json.loads(hkey)
            i = 0
            while i < len(keys):
                chunk = keys[i:i + pack]
                confs = {k: PromptConf.from_dict(rows[k]["prompt_conf"]) for k in chunk}
                prompts = [confs[k](queries[qq], llm)[0] for k in chunk for qq in PAPER8]
                try:
                    if batch1:
                        outs = []
                        for j, p in enumerate(prompts):
                            torch.manual_seed(seed_of(m, *chunk[j // 8], PAPER8[j % 8]))
                            outs.append(llm.generate([p], hp, max_new_tokens=h.TOKEN_CEILING)[0])
                    else:
                        torch.manual_seed(seed_of(m, hkey, *chunk[0]))
                        outs = llm.generate(prompts, hp, max_new_tokens=h.TOKEN_CEILING)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    assert pack > 1, "OOM at one config per call"
                    pack //= 2
                    halvings += 1
                    continue
                for j, k in enumerate(chunk):
                    pc = rows[k]["prompt_conf"]
                    fh.write(json.dumps(dict(
                        dataset=k[0], llm=m, traces=[[queries[qq], outs[j * 8 + n]] for n, qq in enumerate(PAPER8)],
                        prompt_conf=pc, config_index=k[1], model=m, gen_env=GEN_ENV[env_tag],
                        template_kwargs=TEMPLATE_KWARGS_RECORD, config_source=csrc)) + "\n")
                fh.flush()
                n_gen += len(prompts)
                i += len(chunk)
    status.update(gen_wall_s=round(time.time() - t1, 1), n_generated=n_gen,
                  gen_per_s=round(n_gen / max(1e-9, time.time() - t1), 3), pack=pack, oom_halvings=halvings)
    if csrc == "corpus_v1-replay":
        t2 = time.time()
        f1_replay(m, llm, rows, queries, status)
        status["f1"]["wall_s"] = round(time.time() - t2, 1)
    # validate
    seen, empty, total = set(), 0, 0
    for line in open(path):
        d = json.loads(line)
        assert d["prompt_conf"] == rows[(d["dataset"], d["config_index"])]["prompt_conf"]
        seen.add((d["dataset"], d["config_index"]))
        for _, a in d["traces"]:
            total += 1
            empty += not a.strip()
    ok = len(seen) == 100 and total == 800
    status.update(status="COMPLETE" if ok else "INCOMPLETE", n_configs=len(seen), n_rows=total,
                  n_empty=empty, empty_rate=round(empty / max(1, total), 4),
                  sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(),
                  wall_s=round(time.time() - t0, 1), finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    json.dump(status, open(spath, "w"), indent=1)
    del llm
    torch.cuda.empty_cache()
    print(f"   {status['status']} rows={total} empty={empty} gen/s={status['gen_per_s']} "
          f"pack={pack} f1={status.get('f1', {}).get('identical_rate')} wall={status['wall_s']}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--env-tag", default="v2")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    queries = [e["text"] for e in json.load(open(h.Q0))["queries"]]
    b1 = {m for m, v in json.load(open(f"{RES}/s0.json"))["a2_hybrids"].items() if v["batch1"]}
    for m in [l.strip() for l in open(a.models) if l.strip()]:
        print(f"== {m}  batch1={m in b1}", flush=True)
        try:
            run_model(m, a.env_tag, m in b1, queries)
        except Exception:
            err = traceback.format_exc()
            print(err, flush=True)
            sp = f"{OUT}/{slug(m)}.status.json"
            st = json.load(open(sp)) if os.path.exists(sp) else dict(model=m)
            st.update(status="FAILED", error=err[-3000:])
            json.dump(st, open(sp, "w"), indent=1)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
