"""
D006 / S1c — choose ONE batch size for the whole corpus, on evidence.

Two questions, both raised by S1's A3 sweep:

Q1. DOES BATCH SIZE SHIFT THE RESPONSE-LENGTH DISTRIBUTION?
    A3 on Qwen2.5-0.5B showed mean response length 590/618/606/498 chars at
    B=1/8/16/32 -- an 18% drop at 32. S1b supplies a *mechanism* that would make
    that systematic rather than noise: padded sequences diverge more from the
    unbatched reference (T4: 0.486 vs 0.200), larger batches carry more padding,
    and more numerical perturbation means more flipped near-ties, including
    flips into an early EOS.

    This matters far beyond throughput. If length depends on B, then B is a
    corpus variable. And since larger models need smaller B to fit, B would
    correlate with model size -- i.e. a generation artifact confounded with
    exactly the thing this project fingerprints. That is why the corpus must
    use ONE B everywhere, and why "fall back to a smaller B on OOM" (as
    D006/P1's failure handling proposed) is wrong: it silently makes B a
    per-model variable.

    Test: same prompts, greedy, B in {1,8,16,32,64}. Greedy is deterministic
    per B, so differences across B are the effect itself, not sampling noise.
    Report mean/median generated-TOKEN count (not chars -- chars conflate
    length with vocabulary), the paired shift vs B=1, and how often EOS fires
    early.

Q2. WHAT IS THE LARGEST B THAT FITS THE BIGGEST MODEL?
    Phi-3-medium-128k (14B) is A1's largest. Weights ~28 GB bf16 of 95 GB
    visible, so KV cache is the question. Measured, not estimated.

The corpus B is then: the largest B that fits the biggest model AND does not
shift the length distribution. If those conflict, length wins -- throughput is
worth 4x, a confound in the corpus is worth nothing.

Usage:  PYTHONPATH=. python experiments/d006_s1c_batch_choice.py
"""
import gc
import json
import time
import random

import numpy as np
import torch

from LLMmap.llm import LLM_huggingface
from LLMmap.prompt_configuration import PromptConfFactory, BUILD

OUT = "./results/D006/s1c_batch_choice.json"
Q0 = "./confs/queries/pool_v1.json"
SMALL = "Qwen/Qwen2.5-0.5B-Instruct"
LARGE = "microsoft/Phi-3-medium-4k-instruct"      # 14B, A1's largest class
GREEDY = dict(do_sample=False)
NTOK = 200                                         # C7's ceiling
N_PROMPTS = 96
BATCHES = [1, 8, 16, 32, 64]


def load(name):
    return LLM_huggingface(name, model_load_kargs=dict(
        torch_dtype=torch.bfloat16, device_map="cuda"))


def free(x):
    del x
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run(llm, prompts, B, ntok=NTOK):
    outs = []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i in range(0, len(prompts), B):
        outs.extend(llm.generate(prompts[i:i+B], GREEDY, max_new_tokens=ntok))
    return outs, time.time() - t0, torch.cuda.max_memory_allocated() / 1e9


def main():
    random.seed(0)
    res = {}
    q0 = json.load(open(Q0))["queries"]
    pc = PromptConfFactory("./confs/prompt_configurations/")
    conf = pc.sample(1, pool=BUILD)[0]
    queries = [e["text"] for e in random.sample(q0, N_PROMPTS)]

    # ---------------- Q1: length vs batch size, on the small model
    llm = load(SMALL)
    tok = llm.tokenizer
    prompts = [conf(q, llm)[0] for q in queries]

    ref_tokens = None
    rows = []
    for B in BATCHES:
        outs, wall, peak = run(llm, prompts, B)
        ntoks = np.array([len(tok(o, add_special_tokens=False).input_ids)
                          for o in outs])
        if ref_tokens is None:
            ref_tokens = ntoks
        rows.append(dict(
            batch=B, gen_per_s=round(len(prompts)/wall, 3), peak_gb=round(peak, 2),
            mean_tokens=round(float(ntoks.mean()), 1),
            median_tokens=float(np.median(ntoks)),
            frac_hit_ceiling=round(float((ntoks >= NTOK - 1).mean()), 3),
            frac_early_eos=round(float((ntoks < NTOK - 1).mean()), 3),
            # paired shift against B=1 on the SAME prompts
            mean_paired_delta_vs_b1=round(float((ntoks - ref_tokens).mean()), 2),
            identical_to_b1=int((ntoks == ref_tokens).sum())))
        print(f"  B={B:3d}: {rows[-1]['gen_per_s']:6.3f} gen/s  "
              f"{rows[-1]['peak_gb']:5.2f} GB  mean_tok={rows[-1]['mean_tokens']:6.1f}  "
              f"paired_delta={rows[-1]['mean_paired_delta_vs_b1']:+7.2f}  "
              f"ceiling={rows[-1]['frac_hit_ceiling']:.2f}", flush=True)
    res["Q1_length_vs_batch_small_model"] = dict(model=SMALL, n_prompts=len(prompts),
                                                 max_new_tokens=NTOK, rows=rows)

    # is the shift systematic? paired t-test-ish: compare B=32/64 vs B=1
    deltas = {r["batch"]: r["mean_paired_delta_vs_b1"] for r in rows}
    monotone = all(deltas[BATCHES[i]] >= deltas[BATCHES[i+1]] - 1e-9
                   for i in range(1, len(BATCHES)-1))
    big_shift = any(abs(deltas[b]) > 5 for b in BATCHES[1:])
    res["Q1_verdict"] = dict(
        paired_deltas_vs_b1=deltas,
        monotone_decreasing=bool(monotone),
        any_shift_over_5_tokens=bool(big_shift),
        note="A shift here means B is a corpus variable and MUST be pinned "
             "corpus-wide. No shift means B can be chosen on throughput alone "
             "-- but pinning it is still the safer default.")
    free(llm)

    # ---------------- Q2: memory headroom on the largest model
    print(f"\n### {LARGE} — memory headroom", flush=True)
    llm = load(LARGE)
    prompts_l = [conf(q, llm)[0] for q in queries[:64]]
    mem = []
    for B in [8, 16, 32, 64]:
        try:
            outs, wall, peak = run(llm, prompts_l, B, ntok=NTOK)
            ntoks = np.array([len(llm.tokenizer(o, add_special_tokens=False).input_ids)
                              for o in outs])
            mem.append(dict(batch=B, ok=True, peak_gb=round(peak, 2),
                            gen_per_s=round(len(prompts_l)/wall, 3),
                            mean_tokens=round(float(ntoks.mean()), 1)))
            print(f"  B={B:3d}: OK  {peak:5.2f} GB  "
                  f"{mem[-1]['gen_per_s']:6.3f} gen/s  "
                  f"mean_tok={mem[-1]['mean_tokens']:.1f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            mem.append(dict(batch=B, ok=False, error="OOM"))
            print(f"  B={B:3d}: OOM", flush=True)
            torch.cuda.empty_cache()
            break
    res["Q2_memory_largest_model"] = dict(model=LARGE, visible_gb=95.0, rows=mem)
    free(llm)

    ok = [m["batch"] for m in mem if m.get("ok")]
    res["recommendation"] = dict(
        largest_fitting_batch=max(ok) if ok else None,
        length_shift_detected=bool(big_shift),
        rule="Pin ONE batch size corpus-wide. On OOM in a shard: STOP and "
             "report, do not silently drop B for that model -- a per-model B "
             "is a generation artifact confounded with model size.")
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nlargest B fitting {LARGE}: {max(ok) if ok else 'none'}")
    print(f"length shift with B detected: {big_shift}")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
