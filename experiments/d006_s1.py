"""
D006 / S1 — batching verification + the C7 confirmatory pilot.

Two phases, one job, because both need the same models resident on the GPU.

PHASE A — verify `dataset_maker.make_dataset_entries_for_new_llm`'s new batched
path (D002 §R4's deferred fix). Three checks, in decreasing order of how badly a
failure would matter:

  A1. GREEDY EXACT MATCH. Batched output must be *character-identical* to
      unbatched under `do_sample=False`. This is the check that actually catches
      a left-padding / attention-mask error: with `padding_side='left'` the
      output slice in `llm.py:87` is correct, and if anything about that changes
      the greedy text diverges immediately. A distributional check would not
      catch it.
  A2. PREFIX EXACTNESS. Greedy at 200 tokens, truncated to the first 100
      generated tokens, must equal greedy at 100 tokens. D006/P1 §F1's amended
      C7 argument — and therefore C7's decided 200-token ceiling — rests on
      this. If it fails, the "generate high, truncate down" reduction is void
      for this stack and C7 becomes a genuine one-way door after all. Reported
      either way; this is the assertion, not an assumption.
  A3. STOCHASTIC DISTRIBUTION + BATCH SWEEP. Under `do_sample=True` exact match
      is impossible (batching changes RNG consumption), so compare length
      distributions batched vs unbatched, and sweep batch size for throughput
      and peak memory.

PHASE B — the C7 pilot. C7 is already decided (200-token ceiling, human,
2026-09-06), so this is confirmatory rather than load-bearing, but it still
answers a live question: does raising the cap actually help separability, or
does mean-pooling dilute the signal at greater length?

  Generate ONCE at 400 tokens, then analyse at 100 / 200 / 400 by truncation —
  the same causal-generation property A2 checks. Report per cap:
    - censoring rate (share of responses ending mid-sentence)
    - separability across the 15 model pairs under the frozen I5 embedding

  Separability is reported under BOTH a bounded statistic (5-fold CV linear
  probe AUC) and an unbounded one (energy distance), with a saturation audit on
  the bounded one. That is D004's lesson applied prospectively: a bounded
  statistic pinned at its ceiling measures the ceiling, not the population.
  NOTE: this is a C7 diagnostic. It is NOT a decision on A3 (which two-sample
  statistic the real tensor uses) — A3 is still open and this must not be
  read as pre-empting it.

Usage:  PYTHONPATH=. python experiments/d006_s1.py
"""
import os
import gc
import json
import time
import random
import itertools

import numpy as np
import torch

from LLMmap.llm import LLM_huggingface
from LLMmap.prompt_configuration import PromptConfFactory, BUILD
from LLMmap.dataset_maker import make_dataset_entries_for_new_llm

OUT_DIR = "./results/D006"
CONF_DIR = "./confs/prompt_configurations/"
POOL = "./confs/queries/pool_v1.json"
I5_MODEL = "intfloat/multilingual-e5-large-instruct"

VERIFY_MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "microsoft/Phi-3-mini-4k-instruct"]
PILOT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-mini-128k-instruct",   # near-relative of the 4k above:
    "HuggingFaceH4/zephyr-7b-beta",         # D001's genuine hard pair, the case
    "microsoft/Phi-3-medium-4k-instruct",   # where dilution would bite hardest
]
CAPS = [100, 200, 400]
N_QUERIES_PILOT, N_CONFIGS_PILOT = 20, 10
N_PAIRS_VERIFY = 200
TERMINAL = '.!?"\')]}`:;*'


def load(name):
    return LLM_huggingface(name, model_load_kargs=dict(
        torch_dtype=torch.bfloat16, device_map="cuda"))


def free(llm):
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def censoring_rate(texts):
    t = [x.strip() for x in texts if x.strip()]
    if not t:
        return None
    return sum(1 for x in t if x[-1] not in TERMINAL) / len(t)


def truncate_to_tokens(text, tok, n):
    """Truncate generated TEXT to its first n generated tokens."""
    ids = tok(text, add_special_tokens=False).input_ids
    return tok.decode(ids[:n], skip_special_tokens=True)


# --------------------------------------------------------------------------
# PHASE A
# --------------------------------------------------------------------------

def phase_a(queries, confs):
    results = {}
    for name in VERIFY_MODELS:
        print(f"\n### PHASE A — {name}", flush=True)
        llm = load(name)
        r = {}

        # ---- A1: greedy batched must be character-identical to unbatched
        greedy = dict(do_sample=False)
        pairs, mismatches = 0, []
        for conf in confs:
            prompts = [conf(q, llm)[0] for q in queries]
            if pairs >= N_PAIRS_VERIFY:
                break
            unb = [llm.generate(p, greedy, max_new_tokens=100)[0] for p in prompts]
            bat = []
            for i in range(0, len(prompts), 8):
                bat.extend(llm.generate(prompts[i:i+8], greedy, max_new_tokens=100))
            for q, u, b in zip(queries, unb, bat):
                pairs += 1
                if u != b:
                    mismatches.append(dict(query=q[:80], unbatched=u[:200], batched=b[:200]))
        r["A1_greedy_exact_match"] = dict(
            pairs_checked=pairs, mismatches=len(mismatches),
            passed=len(mismatches) == 0, examples=mismatches[:3])
        print(f"  A1 greedy batched==unbatched: {pairs-len(mismatches)}/{pairs} "
              f"{'PASS' if not mismatches else 'FAIL'}", flush=True)

        # ---- A2: prefix exactness (C7's 200-token ceiling depends on this)
        tok = llm.tokenizer
        n_pref, bad = 0, []
        for conf in confs[:5]:
            prompts = [conf(q, llm)[0] for q in queries]
            g100 = []
            for i in range(0, len(prompts), 8):
                g100.extend(llm.generate(prompts[i:i+8], greedy, max_new_tokens=100))
            g200 = []
            for i in range(0, len(prompts), 8):
                g200.extend(llm.generate(prompts[i:i+8], greedy, max_new_tokens=200))
            for q, a100, a200 in zip(queries, g100, g200):
                n_pref += 1
                trunc = truncate_to_tokens(a200, tok, 100)
                # a200 truncated to 100 generated tokens must reproduce a100.
                # Compare on token ids to avoid decode-boundary artifacts.
                ids_a = tok(a100, add_special_tokens=False).input_ids
                ids_b = tok(a200, add_special_tokens=False).input_ids[:len(ids_a)]
                if ids_a != ids_b:
                    bad.append(dict(query=q[:80], at100=a100[:200], at200_trunc=trunc[:200]))
        r["A2_prefix_exactness"] = dict(
            pairs_checked=n_pref, mismatches=len(bad),
            passed=len(bad) == 0, examples=bad[:3],
            note="C7's 200-token ceiling and the 'generate high, truncate down' "
                 "reduction depend on this. Failure voids that argument.")
        print(f"  A2 greedy@200[:100] == greedy@100: {n_pref-len(bad)}/{n_pref} "
              f"{'PASS' if not bad else 'FAIL'}", flush=True)

        # ---- A3: stochastic distribution + batch sweep
        sweep = []
        sampl = dict(do_sample=True, temperature=0.7)
        prompts = [conf(q, llm)[0] for conf in confs[:2] for q in queries][:32]
        for B in (1, 8, 16, 32):
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            outs = []
            for i in range(0, len(prompts), B):
                outs.extend(llm.generate(prompts[i:i+B], sampl, max_new_tokens=200))
            dt = time.time() - t0
            sweep.append(dict(batch=B, n=len(prompts), wall_s=round(dt, 1),
                              gen_per_s=round(len(prompts)/dt, 3),
                              peak_gb=round(torch.cuda.max_memory_allocated()/1e9, 1),
                              mean_chars=round(float(np.mean([len(o) for o in outs])), 1)))
            print(f"  A3 batch={B:2d}: {sweep[-1]['gen_per_s']:.3f} gen/s, "
                  f"{sweep[-1]['peak_gb']} GB, mean {sweep[-1]['mean_chars']} chars", flush=True)
        r["A3_batch_sweep"] = sweep
        results[name] = r
        free(llm)
    return results


# --------------------------------------------------------------------------
# PHASE B
# --------------------------------------------------------------------------

def phase_b_generate(queries, confs):
    """One generation pass at 400 tokens; all caps derived by truncation."""
    raw = {}
    for name in PILOT_MODELS:
        print(f"\n### PHASE B generate — {name}", flush=True)
        llm = load(name)
        t0 = time.time()
        entries = make_dataset_entries_for_new_llm(
            llm, queries, confs, pool=BUILD, batch_size=8, max_new_tokens=max(CAPS))
        # store token ids so truncation to a cap is exact, not char-approximate
        recs = []
        for ci, e in enumerate(entries):
            for q, a in e["traces"]:
                ids = llm.tokenizer(a, add_special_tokens=False).input_ids
                recs.append(dict(config=ci, query=q, ids=ids))
        raw[name] = dict(records=recs, wall_s=round(time.time()-t0, 1),
                         n=len(recs))
        print(f"  {len(recs)} generations in {raw[name]['wall_s']}s", flush=True)
        # decode each cap now, while this model's tokenizer is in hand
        for cap in CAPS:
            for r in raw[name]["records"]:
                r[f"text_{cap}"] = llm.tokenizer.decode(
                    r["ids"][:cap], skip_special_tokens=True)
        for r in raw[name]["records"]:
            r.pop("ids")
        free(llm)
    return raw


def embed_all(raw):
    """Embed every (model, cap) response set with the frozen I5 model."""
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(I5_MODEL)
    mdl = AutoModel.from_pretrained(I5_MODEL, torch_dtype=torch.float16).cuda().eval()

    def encode(texts, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i+bs], padding=True, truncation=True,
                    max_length=512, return_tensors="pt").to("cuda")
            with torch.no_grad():
                h = mdl(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1)          # mean pooling, per e5
                e = torch.nn.functional.normalize(e, dim=-1)
            out.append(e.float().cpu().numpy())
        return np.concatenate(out)

    emb = {}
    for cap in CAPS:
        for name, d in raw.items():
            emb[(name, cap)] = encode([r[f"text_{cap}"] for r in d["records"]])
            print(f"  embedded {name} @ {cap}: {emb[(name,cap)].shape}", flush=True)
    del mdl
    torch.cuda.empty_cache()
    return emb


def energy_distance(X, Y):
    """Unbounded two-sample statistic (D004's uncensored companion)."""
    from scipy.spatial.distance import cdist
    xy = cdist(X, Y).mean()
    xx = cdist(X, X).mean()
    yy = cdist(Y, Y).mean()
    return float(2*xy - xx - yy)


def probe_auc(X, Y, seed=0):
    """Bounded statistic: 5-fold CV AUC of a linear probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    Z = np.vstack([X, Y])
    y = np.r_[np.zeros(len(X)), np.ones(len(Y))]
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    scores = []
    for tr, te in skf.split(Z, y):
        clf = LogisticRegression(max_iter=2000).fit(Z[tr], y[tr])
        scores.append(roc_auc_score(y[te], clf.decision_function(Z[te])))
    return float(np.mean(scores))


def phase_b_analyse(raw, emb):
    rows, summary = [], {}
    for cap in CAPS:
        cens = {n: censoring_rate([r[f"text_{cap}"] for r in d["records"]])
                for n, d in raw.items()}
        aucs, eds = [], []
        for a, b in itertools.combinations(PILOT_MODELS, 2):
            A, B = emb[(a, cap)], emb[(b, cap)]
            auc, ed = probe_auc(A, B), energy_distance(A, B)
            aucs.append(auc); eds.append(ed)
            rows.append(dict(cap=cap, model_a=a, model_b=b,
                             probe_auc=round(auc, 4), energy_distance=round(ed, 5)))
        aucs = np.array(aucs)
        summary[cap] = dict(
            censoring_rate_overall=round(float(np.mean(list(cens.values()))), 4),
            censoring_by_model={k: round(v, 4) for k, v in cens.items()},
            probe_auc_mean=round(float(aucs.mean()), 4),
            probe_auc_min=round(float(aucs.min()), 4),
            # D004's saturation audit, applied to the BOUNDED statistic
            probe_auc_frac_at_ceiling=round(float((aucs > 0.999).mean()), 4),
            probe_auc_distinct_values=int(len(np.unique(np.round(aucs, 4)))),
            energy_distance_mean=round(float(np.mean(eds)), 5),
            energy_distance_min=round(float(np.min(eds)), 5))
        print(f"  cap={cap}: censoring={summary[cap]['censoring_rate_overall']:.1%} "
              f"probe_auc_mean={summary[cap]['probe_auc_mean']:.4f} "
              f"(at ceiling {summary[cap]['probe_auc_frac_at_ceiling']:.0%}) "
              f"energy_mean={summary[cap]['energy_distance_mean']:.5f}", flush=True)
    return rows, summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(0)
    np.random.seed(0)

    pool = json.load(open(POOL))
    entries = pool["queries"] if isinstance(pool, dict) else pool  # pool_v1 is a dict
    texts = [e["text"] if isinstance(e, dict) else e for e in entries]
    queries = random.sample(texts, N_QUERIES_PILOT)

    pc = PromptConfFactory(CONF_DIR)
    confs = pc.sample(N_CONFIGS_PILOT, pool=BUILD)

    print(f"pilot: {len(queries)} queries x {len(confs)} configs "
          f"x {len(PILOT_MODELS)} models @ {max(CAPS)} tokens", flush=True)

    out = dict(
        c7_note="C7 was decided (200-token ceiling, human, 2026-09-06). This "
                "pilot is confirmatory, not load-bearing.",
        a3_note="Separability here is a C7 diagnostic ONLY. It is not a "
                "decision on A3 (the tensor's two-sample statistic), which "
                "remains open.",
        n_queries=len(queries), n_configs=len(confs), caps=CAPS)

    out["phase_a"] = phase_a(queries, confs)
    raw = phase_b_generate(queries, confs)
    emb = embed_all(raw)
    rows, summary = phase_b_analyse(raw, emb)
    out["phase_b_summary"] = summary
    out["phase_b_pairs"] = rows
    out["phase_b_wall_s"] = {k: v["wall_s"] for k, v in raw.items()}

    with open(f"{OUT_DIR}/s1_results.json", "w") as f:
        json.dump(out, f, indent=2)

    a1 = all(v["A1_greedy_exact_match"]["passed"] for v in out["phase_a"].values())
    a2 = all(v["A2_prefix_exactness"]["passed"] for v in out["phase_a"].values())
    print(f"\n=== S1 VERDICT ===")
    print(f"A1 greedy batched == unbatched : {'PASS' if a1 else 'FAIL'}")
    print(f"A2 prefix exactness (C7 rests) : {'PASS' if a2 else 'FAIL'}")
    if not a2:
        print("  !! C7's truncation reduction is VOID for this stack. STOP and report.")
    print(f"written: {OUT_DIR}/s1_results.json")


if __name__ == "__main__":
    main()
