"""D026 / S0 — harness parity and set-up. No new probe is generated except the
single > 4k observation the D asks for (probe 275 on the 4k model, once).

Per twin:
  (1) PARITY. Regenerate up to 10 greedy S_build configs (the D024 tier-1
      greedy reference first, then the lowest config indices), each through the
      D006 code path exactly as the shard ran it: `PromptConf.from_dict` of the
      stored prompt_conf, all 259 pool queries, batch 64, 200-token ceiling, so
      every pool query sits in the same batch it sat in. Primary rate: queries
      {9, 65, 140, 193} (the D's list), byte-identical vs data/corpus_v1. All-259
      rate reported as a secondary. Also: the 4 queries regenerated ALONE in one
      batch (the batching mode new probes would see), descriptive.
      STOP (Call) if the primary rate < 0.90.
  (2) PROMPT-LENGTH CENSUS over all 125 configs x 259 pool queries, and over the
      6 F3 probes wrapped in every config: counts above the 4k model's sliding
      window (2047) and above 4096 (LongRoPE regime switch of the 128k model,
      decided by the longest sequence in a batch).
  (3) 4k model only: the > 4k behaviour (probe 275 under the greedy reference
      config, as the harness would run it), and a sliding-window liveness
      diagnostic on filler-only text (no probe): last-position logits with the
      shipped sliding_window vs the window disabled.
Plus the filler record and bare token counts of probes 273-278.

Usage:  PYTHONPATH=.:experiments python experiments/d026_s0.py
"""
import os
import sys
import json
import time
import hashlib
import logging
import warnings

import numpy as np
import torch

from LLMmap.prompt_configuration import PromptConf
from LLMmap.dataset_maker import make_dataset_entries_for_new_llm
import d006_s4_shard as h
from d026_lib import (TWINS, OUT, FILLER_META, FILLER_SRC, NEEDLE_LITERAL_ASTERISKS,
                      all_probes, filler_text, shard_status, corpus_entries,
                      greedy_reference, load_twin)

PARITY_Q = [9, 65, 140, 193]
N_PARITY = 10
SW_THRESH, ROPE_THRESH = 2047, 4096


def census(tok, prompts_by_cfg):
    """prompts_by_cfg: {(pool, c): [prompt strings in query order]}."""
    lens, batch_max = [], []
    for key, ps in prompts_by_cfg.items():
        L = [len(x) for x in tok(ps, add_special_tokens=False)["input_ids"]]
        lens.extend(L)
        for i in range(0, len(L), h.CORPUS_BATCH):
            batch_max.append(max(L[i:i + h.CORPUS_BATCH]))
    lens, bm = np.array(lens), np.array(batch_max)
    return dict(n_prompts=int(len(lens)), max=int(lens.max()),
                p50=float(np.median(lens)), p99=float(np.percentile(lens, 99)),
                n_gt_2047=int((lens > SW_THRESH).sum()),
                n_plus200_gt_2047=int((lens + h.TOKEN_CEILING > SW_THRESH).sum()),
                n_gt_4096=int((lens > ROPE_THRESH).sum()),
                n_batches=int(len(bm)),
                n_batches_max_gt_4096=int((bm > ROPE_THRESH).sum()),
                n_batches_max_plus200_gt_4096=int((bm + h.TOKEN_CEILING > ROPE_THRESH).sum()))


class _Cap(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, r):
        self.msgs.append(f"{r.name}:{r.levelname}:{r.getMessage()[:300]}")


def main():
    os.makedirs(OUT, exist_ok=True)
    t_all = time.time()
    q0 = json.load(open(h.Q0))
    queries = [e["text"] for e in q0["queries"]]
    out = dict(schema="d026-s0-v1", q0_sha256=q0["sha256"], parity_queries=PARITY_Q,
               models={}, needle_literal_asterisks=NEEDLE_LITERAL_ASTERISKS)

    tok_hash = {}
    for m in TWINS:
        st = shard_status(m)
        rev = st["hf_revision"]
        ent = corpus_entries(m)
        ref = greedy_reference(m, ent)
        greedy = sorted(c for (p, c), d in ent.items()
                        if p == "build" and d["prompt_conf"]["sampling_hparams"]["do_sample"] is False)
        cfgs = [ref] + [c for c in greedy if c != ref][:N_PARITY - 1]
        print(f"== {m}  rev {rev}  greedy build {len(greedy)}  parity cfgs {cfgs}", flush=True)

        t0 = time.time()
        llm, prov = load_twin(m, rev)
        prov["load_s"] = round(time.time() - t0, 1)
        from huggingface_hub import hf_hub_download
        tj = hf_hub_download(m, "tokenizer.json", revision=rev)
        tok_hash[m] = hashlib.sha256(open(tj, "rb").read()).hexdigest()
        rec = dict(provenance=prov, corpus_shard_sha256=st["sha256"],
                   greedy_build_configs=greedy, parity_configs=cfgs, reference_config=ref)

        # ---- (1) parity
        rows, iso_rows, gen_s, n_gen = [], [], 0.0, 0
        for c in cfgs:
            stored = ent[("build", c)]
            assert [q for q, _ in stored["traces"]] == queries
            conf = PromptConf.from_dict(stored["prompt_conf"])
            t = time.time()
            new = make_dataset_entries_for_new_llm(
                llm, queries, [conf], pool="build", batch_size=h.CORPUS_BATCH,
                max_new_tokens=h.TOKEN_CEILING)[0]
            gen_s += time.time() - t
            n_gen += len(queries)
            assert new["prompt_conf"] == stored["prompt_conf"]
            same = [a == b for (_, a), (_, b) in zip(new["traces"], stored["traces"])]
            iso = make_dataset_entries_for_new_llm(
                llm, [queries[q] for q in PARITY_Q], [conf], pool="build",
                batch_size=h.CORPUS_BATCH, max_new_tokens=h.TOKEN_CEILING)[0]
            iso_same = [a == stored["traces"][q][1] for q, (_, a) in zip(PARITY_Q, iso["traces"])]
            rows.append(dict(config=c, primary=[bool(same[q]) for q in PARITY_Q],
                             all259=int(sum(same)),
                             mismatched=[i for i, s in enumerate(same) if not s][:40]))
            iso_rows.append(dict(config=c, isolated=[bool(x) for x in iso_same]))
            print(f"  cfg {c:3d}: primary {sum(rows[-1]['primary'])}/4  all {sum(same)}/259  "
                  f"isolated {sum(iso_same)}/4", flush=True)
        prim = [x for r in rows for x in r["primary"]]
        rec["parity"] = dict(
            per_config=rows, isolated_batch=iso_rows,
            primary_rate=float(np.mean(prim)), primary_n=len(prim),
            all259_rate=float(sum(r["all259"] for r in rows) / (259 * len(rows))),
            isolated_rate=float(np.mean([x for r in iso_rows for x in r["isolated"]])),
            gen_s=round(gen_s, 1), gen_per_s=round(n_gen / gen_s, 3))

        # ---- (2) census
        probes, f3meta = all_probes(llm.tokenizer)
        pool_prompts = {k: [PromptConf.from_dict(d["prompt_conf"])(q, llm)[0] for q in queries]
                        for k, d in ent.items()}
        rec["census_pool"] = census(llm.tokenizer, pool_prompts)
        f3_ids = list(range(273, 279))
        f3_lens = {}
        for k, d in ent.items():
            conf = PromptConf.from_dict(d["prompt_conf"])
            ps = [conf(probes[i], llm)[0] for i in f3_ids]
            f3_lens[f"{k[0]}:{k[1]}"] = [len(x) for x in llm.tokenizer(ps, add_special_tokens=False)["input_ids"]]
        A = np.array(list(f3_lens.values()))           # (125, 6)
        rec["census_f3_wrapped"] = {str(i): dict(
            min=int(A[:, j].min()), max=int(A[:, j].max()), p50=float(np.median(A[:, j])),
            n_gt_2047=int((A[:, j] > SW_THRESH).sum()), n_gt_4096=int((A[:, j] > ROPE_THRESH).sum()))
            for j, i in enumerate(f3_ids)}
        rec["census_new_short_wrapped_max"] = int(max(
            len(llm.tokenizer(PromptConf.from_dict(d["prompt_conf"])(probes[i], llm)[0],
                              add_special_tokens=False)["input_ids"])
            for d in ent.values() for i in range(259, 273)))
        out.setdefault("probe_ids", {str(i): (t if i < 273 else t[:80] + " ... " + t[-120:])
                                     for i, t in probes.items()})
        out.setdefault("f3_bare", {})[m] = f3meta
        out.setdefault("probe_sha256", {})[m] = hashlib.sha256(
            json.dumps([probes[i] for i in sorted(probes)]).encode()).hexdigest()

        # ---- (3) 4k model: > 4k behaviour + sliding-window liveness
        if "4k" in m:
            cap = _Cap()
            lg = logging.getLogger("transformers")
            lg.addHandler(cap)
            conf = PromptConf.from_dict(ent[("build", ref)]["prompt_conf"])
            p275 = conf(probes[275], llm)[0]
            n_tok = len(llm.tokenizer(p275, add_special_tokens=False)["input_ids"])
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    o = make_dataset_entries_for_new_llm(
                        llm, [probes[275]], [conf], pool="build",
                        batch_size=h.CORPUS_BATCH, max_new_tokens=h.TOKEN_CEILING)[0]
                    resp, err = o["traces"][0][1], None
                except Exception as e:                       # the error IS the observation
                    resp, err = None, repr(e)[:2000]
            n_out = (len(llm.tokenizer(resp, add_special_tokens=False)["input_ids"])
                     if resp is not None else None)
            lg.removeHandler(cap)
            rec["gt4k_behaviour"] = dict(
                probe=275, config=ref, prompt_tokens=n_tok, error=err, response=resp,
                response_tokens=n_out, wall_s=round(time.time() - t, 2),
                peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                warnings=[str(x.message)[:300] for x in w], log=cap.msgs)
            print(f"  >4k: prompt {n_tok} tok, err={err}, out {n_out} tok, "
                  f"{rec['gt4k_behaviour']['wall_s']} s\n  response: {str(resp)[:300]!r}", flush=True)

            # liveness: filler-only, no template, no probe text
            ids = llm.tokenizer(filler_text(), add_special_tokens=False,
                                return_tensors="pt")["input_ids"][:, :3000].to(llm.model.device)
            cfg = llm.model.config
            shipped = cfg.sliding_window
            with torch.no_grad():
                a = llm.model(input_ids=ids).logits[0, -1].float()
                cfg.sliding_window = None
                b = llm.model(input_ids=ids).logits[0, -1].float()
                cfg.sliding_window = shipped
                ids2 = ids[:, :1500]
                a2 = llm.model(input_ids=ids2).logits[0, -1].float()
                cfg.sliding_window = None
                b2 = llm.model(input_ids=ids2).logits[0, -1].float()
                cfg.sliding_window = shipped
            rec["sliding_window_liveness"] = dict(
                shipped=shipped, input="first 3000 / 1500 filler tokens, no template",
                max_abs_logit_diff_3000=float((a - b).abs().max()),
                argmax_same_3000=bool(a.argmax() == b.argmax()),
                max_abs_logit_diff_1500_control=float((a2 - b2).abs().max()))
            print(f"  sliding-window liveness: {rec['sliding_window_liveness']}", flush=True)

        out["models"][m] = rec
        print(f"  parity primary {rec['parity']['primary_rate']:.3f} "
              f"(n={rec['parity']['primary_n']}), all259 {rec['parity']['all259_rate']:.3f}, "
              f"isolated {rec['parity']['isolated_rate']:.3f}; census {rec['census_pool']}", flush=True)
        del llm
        torch.cuda.empty_cache()

    out["filler"] = dict(FILLER_META, path=FILLER_SRC,
                         normalised_chars=len(filler_text()))
    out["tokenizer_json_sha256"] = tok_hash
    rates = [out["models"][m]["parity"]["primary_rate"] for m in TWINS]
    out["parity_stop"] = bool(min(rates) < 0.90)
    out["wall_s"] = round(time.time() - t_all, 1)
    json.dump(out, open(f"{OUT}/s0.json", "w"), indent=1)
    print(f"[S0] wrote {OUT}/s0.json; parity {rates}; STOP={out['parity_stop']}; "
          f"wall {out['wall_s']} s", flush=True)


if __name__ == "__main__":
    main()
