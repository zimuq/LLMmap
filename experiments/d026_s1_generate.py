"""D026 / S1 — generate the 22 probes for one twin under all 125 of its own
corpus_v1 configs, then embed them (I5). One invocation = one twin.

Generation per config (P1 Call 1 (a), approved; Review Call 3b adds 279/280):
  * F1+F2 (ids 259-272): one `make_dataset_entries_for_new_llm` call at the
    corpus batch size 64 -> all 14 in one batch, exactly the harness.
  * F3: one call per L group [(1000: 279/280), (2000: 273/274), (6000: 275/276),
    (12000: 277/278)], holding that L's two needle probes. Zero padding is
    asserted (equal wrapped token lengths); if lengths ever differ the group
    falls back to batch 1 and the fallback is recorded.
  * torch.manual_seed per (model, pool, config_index); corpus_v1 used none.
Rows are written in corpus_v1's schema; `traces[i]` <-> query id 259 + i.

Guard: after 10 configs, project this twin's wall-clock to 125 configs. Both
twins run in parallel jobs, so 2x this projection is the D's total; stop if
it exceeds 4 GPU-h (status STOPPED_PROJECTION).

Then embed with I5 exactly as d006_s6_embed.py (mean pool, NOT normalised,
max_length 512, fp16), after an embedding-parity check against 50 stored
corpus_v1 pool responses of this twin (max |diff| <= 1e-2).

Usage:  PYTHONPATH=.:experiments python experiments/d026_s1_generate.py --model <hf_name>
"""
import os
import json
import time
import hashlib
import argparse
import collections

import numpy as np
import torch

from LLMmap.prompt_configuration import PromptConf
from LLMmap.dataset_maker import make_dataset_entries_for_new_llm
import d006_s4_shard as h
import d026_lib as L

EXT_DIR = "./data/corpus_v1_ext_d026"
QFILE = "./confs/queries/pool_d026_ext.json"
GUARD_AFTER, GUARD_GPU_H = 10, 4.0


def seed_for(m, pool, c):
    return int(hashlib.sha256(f"{m}|{pool}|{c}".encode()).hexdigest(), 16) % 2**31


def generate(args, qdoc, probes, status, status_path):
    m = args.model
    out_path = f"{EXT_DIR}/{L.slug(m)}.jsonl"
    ent = L.corpus_entries(m)
    order = [(p, c) for p in ("build", "val", "test")
             for c in sorted(c for (pp, c) in ent if pp == p)]
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if len(d.get("traces", [])) == 22:
                done.add((d["dataset"], d["config_index"]))
    llm, prov = L.load_twin(m, L.shard_status(m)["hf_revision"])
    status["provenance"] = prov
    ids = sorted(probes)
    short = [probes[i] for i in L.SHORT_IDS]
    fallbacks, t0, n_new = [], time.time(), 0
    with open(out_path, "a") as fh:
        for k, (pool, c) in enumerate(order):
            if (pool, c) in done:
                continue
            row = ent[(pool, c)]
            conf = PromptConf.from_dict(row["prompt_conf"])
            torch.manual_seed(seed_for(m, pool, c))
            resp = {}
            e = make_dataset_entries_for_new_llm(llm, short, [conf], pool=pool,
                                                 batch_size=h.CORPUS_BATCH,
                                                 max_new_tokens=h.TOKEN_CEILING)[0]
            for i, (_, a) in zip(L.SHORT_IDS, e["traces"]):
                resp[i] = a
            assert e["prompt_conf"] == row["prompt_conf"]
            for Lg, gids in L.F3_GROUPS:
                texts = [probes[i] for i in gids]
                lens = [len(llm.tokenizer(conf(t, llm)[0], add_special_tokens=False)["input_ids"])
                        for t in texts]
                if len(set(lens)) == 1:
                    e = make_dataset_entries_for_new_llm(llm, texts, [conf], pool=pool,
                                                         batch_size=2, max_new_tokens=h.TOKEN_CEILING)[0]
                    outs = [a for _, a in e["traces"]]
                else:
                    fallbacks.append(dict(pool=pool, config=c, L=Lg, lens=lens))
                    outs = [make_dataset_entries_for_new_llm(llm, [t], [conf], pool=pool, batch_size=1,
                                                             max_new_tokens=h.TOKEN_CEILING)[0]["traces"][0][1]
                            for t in texts]
                for i, a in zip(gids, outs):
                    resp[i] = a
            fh.write(json.dumps(dict(dataset=pool, llm=llm.llm_name,
                                     traces=[[probes[i], resp[i]] for i in ids],
                                     prompt_conf=row["prompt_conf"], config_index=c,
                                     model=m)) + "\n")
            fh.flush()
            n_new += 1
            if n_new == GUARD_AFTER:
                per = (time.time() - t0) / n_new
                proj = per * (len(order) - len(done)) / 3600
                status["projection"] = dict(after=n_new, s_per_config=round(per, 2),
                                            this_twin_h=round(proj, 3), both_twins_h=round(2 * proj, 3))
                json.dump(status, open(status_path, "w"), indent=1)
                print(f"[guard] {per:.1f} s/config -> this twin {proj:.2f} h, both {2*proj:.2f} h", flush=True)
                if 2 * proj > GUARD_GPU_H:
                    status["status"] = "STOPPED_PROJECTION"
                    return None
            if n_new % 25 == 0:
                print(f"  {n_new} configs, {time.time()-t0:.0f} s", flush=True)
    status["f3_batch1_fallbacks"] = fallbacks
    status["gen_wall_s"] = round(time.time() - t0, 1)
    del llm
    torch.cuda.empty_cache()
    return out_path


def validate(m, out_path, probes, f3meta, status):
    seen, total, empty = set(), 0, 0
    cls = collections.defaultdict(collections.Counter)
    ent = L.corpus_entries(m)
    for line in open(out_path):
        d = json.loads(line)
        assert d["prompt_conf"] == ent[(d["dataset"], d["config_index"])]["prompt_conf"]
        assert [q for q, _ in d["traces"]] == [probes[i] for i in sorted(probes)]
        seen.add((d["dataset"], d["config_index"]))
        for i, (_, a) in zip(sorted(probes), d["traces"]):
            total += 1
            empty += not a.strip()
            if i in f3meta:
                cls[str(i)][f"{d['dataset']}:{L.f3_output_class(a, f3meta[i]['needle'])}"] += 1
    ok = len(seen) == 125 and total == 125 * 22
    status.update(status="COMPLETE" if ok else "INCOMPLETE", n_configs=len(seen), n_rows=total,
                  n_empty_responses=empty, empty_rate=round(empty / max(total, 1), 4),
                  f3_output_classes={k: dict(v) for k, v in cls.items()},
                  sha256=hashlib.sha256(open(out_path, "rb").read()).hexdigest())
    return ok


def embed(m, out_path, status):
    from transformers import AutoTokenizer, AutoModel
    from d006_s6_embed import mean_pool, I5_MODEL, I5_DIM, MAX_LEN, BATCH
    tok = AutoTokenizer.from_pretrained(I5_MODEL)
    mdl = AutoModel.from_pretrained(I5_MODEL, torch_dtype=torch.float16).cuda().eval()

    def run(texts):
        out = np.empty((len(texts), I5_DIM), np.float16)
        with torch.no_grad():
            for i in range(0, len(texts), BATCH):
                b = tok(texts[i:i + BATCH], padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt").to("cuda")
                e = mean_pool(mdl(**b).last_hidden_state, b["attention_mask"])
                out[i:i + len(e)] = e.float().cpu().numpy().astype(np.float16)
        return out

    # parity: 50 stored pool responses of this twin vs corpus_v1/embeddings
    emb_dir = f"{L.CORPUS_DIR}/embeddings"
    rows = json.load(open(f"{emb_dir}/{L.slug(m)}.index.json"))["rows"]
    stored = np.load(f"{emb_dir}/{L.slug(m)}.npy", mmap_mode="r")
    ent = L.corpus_entries(m)
    rng = np.random.default_rng(20261002)
    pick = sorted(rng.choice(len(rows), 50, replace=False).tolist())
    texts = [ent[(rows[r]["pool"], rows[r]["config"])]["traces"][rows[r]["query_index"]][1] for r in pick]
    diff = float(np.abs(run(texts).astype(np.float32) - stored[pick].astype(np.float32)).max())
    status["embedding_parity_max_abs"] = diff
    assert diff <= 1e-2, f"I5 path drifted: max |diff| {diff}"

    texts, index = [], []
    for line in open(out_path):
        d = json.loads(line)
        for qi, (_, a) in enumerate(d["traces"]):
            texts.append(a)
            index.append(dict(pool=d["dataset"], config=d["config_index"], query_index=qi,
                              query_id=L.FIRST_NEW_ID + qi, empty=not a.strip()))
    t = time.time()
    E = run(texts)
    os.makedirs(f"{EXT_DIR}/embeddings", exist_ok=True)
    npy = f"{EXT_DIR}/embeddings/{L.slug(m)}.npy"
    np.save(npy, E)
    json.dump(dict(model=m, n=len(index), dim=I5_DIM, dtype="fp16", embedding_model=I5_MODEL,
                   pooling="mean, NOT normalised (matches LLMmap/embedding_model.py)",
                   max_length=MAX_LEN, query_id_offset=L.FIRST_NEW_ID, rows=index),
              open(f"{EXT_DIR}/embeddings/{L.slug(m)}.index.json", "w"))
    status["embedding"] = dict(n=len(texts), wall_s=round(time.time() - t, 1),
                               sha256=hashlib.sha256(open(npy, "rb").read()).hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    assert args.model in L.TWINS
    os.makedirs(EXT_DIR, exist_ok=True)
    qdoc = json.load(open(QFILE))
    assert qdoc["control_L"] == L.CONTROL_L, "set D026_CONTROL_L from s1_precheck.json"
    status_path = f"{EXT_DIR}/{L.slug(args.model)}.status.json"
    status = dict(model=args.model, status="RUNNING", schema="corpus_v1 (ext d026 v1)",
                  queries_sha256=qdoc["sha256"], started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    t0 = time.time()
    # the committed query file is the source of truth for the probe texts
    probes = {q["id"]: q["text"] for q in qdoc["queries"]}
    f3meta = {q["id"]: q["f3"] for q in qdoc["queries"] if "f3" in q}
    assert sorted(probes) == list(range(259, 281))
    assert hashlib.sha256(json.dumps([probes[i] for i in sorted(probes)]).encode()).hexdigest() == qdoc["sha256"]
    try:
        out_path = generate(args, qdoc, probes, status, status_path)
        if out_path and validate(args.model, out_path, probes, f3meta, status):
            embed(args.model, out_path, status)
    finally:
        status["wall_s"] = round(time.time() - t0, 1)
        status["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        json.dump(status, open(status_path, "w"), indent=1)
        json.dump(status, open(f"{L.OUT}/s1_{L.slug(args.model)}.json", "w"), indent=1)
    print(f"[S1] {args.model}: {status['status']} wall {status['wall_s']} s", flush=True)
    if status["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
