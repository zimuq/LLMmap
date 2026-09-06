"""
D006 / S1b — why does batched greedy output differ from unbatched? (A1 FAILED)

S1's A1 check found 116/200 greedy mismatches between batched and unbatched
generation. Two causes are possible and they have OPPOSITE consequences:

  (a) LEFT-PADDING / ATTENTION-MASK BUG. Batched outputs are semantically
      wrong -- the model attends to pad tokens, or the output slice takes the
      wrong boundary. The corpus would be garbage. S4 must not run.

  (b) FLOATING-POINT NON-ASSOCIATIVITY. Batching changes matmul shapes, so
      cuBLAS picks different kernels and reduction orders. Logits differ in the
      last mantissa bits; greedy argmax flips wherever the top-2 margin is
      smaller than that noise, and the sequences diverge from there. The
      computation is mathematically the same function; the corpus is valid,
      just not bit-reproducible. bf16's 8-bit mantissa makes this common.

Distinguishing them is not a judgement call -- these make different, checkable
predictions:

  T1 determinism      same prompt twice, unbatched -> must be identical.
                      If not, something more basic is wrong.
  T2 EQUAL-LENGTH     a batch whose prompts all tokenize to the SAME length has
                      no padding at all. Under (a) mismatches vanish. Under (b)
                      they persist -- shapes still change.
  T3 divergence pos   first differing token index. Under (a) divergence starts
                      at or near token 0 (the prompt was read wrong). Under (b)
                      outputs share a prefix and diverge later, where a near-tie
                      first occurs.
  T4 padding corr     under (a) mismatch rate rises with how much a sequence was
                      padded. Under (b) it is roughly independent of padding.
  T5 logit level      for the FIRST generated token, compare logits directly.
                      Under (b) max|delta| is bf16 noise (~1e-2) and argmax flips
                      ONLY where the top-2 margin is below that noise. Under (a)
                      the logit vectors differ grossly.
  T6 fp32 control     rerun in float32. Under (b) mismatches drop sharply
                      (more mantissa -> fewer near-tie flips). Under (a)
                      precision is irrelevant and they persist.

T5 is the decisive one; the rest corroborate.

Usage:  PYTHONPATH=. python experiments/d006_s1b_diagnose_batching.py
"""
import json
import random

import numpy as np
import torch

from LLMmap.llm import LLM_huggingface
from LLMmap.prompt_configuration import PromptConfFactory, BUILD

OUT = "./results/D006/s1b_batching_diagnosis.json"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
Q0 = "./confs/queries/pool_v1.json"
GREEDY = dict(do_sample=False)
NTOK = 60


def load(dtype):
    return LLM_huggingface(MODEL, model_load_kargs=dict(
        torch_dtype=dtype, device_map="cuda"))


def gen(llm, prompts, B, ntok=NTOK):
    outs = []
    for i in range(0, len(prompts), B):
        outs.extend(llm.generate(prompts[i:i+B], GREEDY, max_new_tokens=ntok))
    return outs


def first_diff_token(llm, a, b):
    ia = llm.tokenizer(a, add_special_tokens=False).input_ids
    ib = llm.tokenizer(b, add_special_tokens=False).input_ids
    for i, (x, y) in enumerate(zip(ia, ib)):
        if x != y:
            return i
    return None if ia == ib else min(len(ia), len(ib))


def main():
    random.seed(0)
    res = {"model": MODEL, "n_tokens": NTOK}

    q0 = json.load(open(Q0))["queries"]
    pc = PromptConfFactory("./confs/prompt_configurations/")
    conf = pc.sample(1, pool=BUILD)[0]

    llm = load(torch.bfloat16)
    tok = llm.tokenizer

    queries = [e["text"] for e in random.sample(q0, 40)]
    prompts = [conf(q, llm)[0] for q in queries]
    lens = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]

    # ---- T1: determinism, unbatched
    a = gen(llm, prompts[:8], 1)
    b = gen(llm, prompts[:8], 1)
    res["T1_determinism_unbatched"] = dict(
        n=8, identical=sum(x == y for x, y in zip(a, b)),
        passed=a == b)
    print(f"T1 determinism (unbatched x2): {res['T1_determinism_unbatched']}", flush=True)

    # ---- T2: EQUAL-LENGTH batch -> zero padding
    from collections import Counter
    common = Counter(lens).most_common(1)[0][0]
    eq_idx = [i for i, L in enumerate(lens) if L == common]
    if len(eq_idx) < 2:
        # build equal-length prompts synthetically by truncating token ids
        target = min(lens)
        eq_prompts = [tok.decode(tok(p, add_special_tokens=False).input_ids[:target],
                                 skip_special_tokens=True) for p in prompts[:8]]
    else:
        eq_prompts = [prompts[i] for i in eq_idx][:8]
    eq_lens = {len(tok(p, add_special_tokens=False).input_ids) for p in eq_prompts}
    u = gen(llm, eq_prompts, 1)
    bb = gen(llm, eq_prompts, len(eq_prompts))
    res["T2_equal_length_batch"] = dict(
        n=len(eq_prompts), distinct_prompt_lengths=sorted(eq_lens),
        zero_padding=len(eq_lens) == 1,
        matches=sum(x == y for x, y in zip(u, bb)),
        mismatches=sum(x != y for x, y in zip(u, bb)))
    print(f"T2 equal-length (no padding): {res['T2_equal_length_batch']}", flush=True)

    # ---- T3/T4: ragged batch, divergence position vs padding amount
    u = gen(llm, prompts, 1)
    bt = gen(llm, prompts, 8)
    rows = []
    for i, (x, y) in enumerate(zip(u, bt)):
        batch_max = max(lens[(i//8)*8:(i//8)*8+8])
        rows.append(dict(idx=i, prompt_len=lens[i], pad=batch_max - lens[i],
                         match=x == y,
                         first_diff=None if x == y else first_diff_token(llm, x, y)))
    mism = [r for r in rows if not r["match"]]
    fd = [r["first_diff"] for r in mism if r["first_diff"] is not None]
    res["T3_divergence_position"] = dict(
        n=len(rows), mismatches=len(mism),
        first_diff_min=int(min(fd)) if fd else None,
        first_diff_median=float(np.median(fd)) if fd else None,
        first_diff_mean=round(float(np.mean(fd)), 2) if fd else None,
        frac_diverging_at_token_0=round(float(np.mean([f == 0 for f in fd])), 3) if fd else None)
    padded = [r for r in rows if r["pad"] > 0]
    unpadded = [r for r in rows if r["pad"] == 0]
    res["T4_padding_correlation"] = dict(
        mismatch_rate_padded=round(float(np.mean([not r["match"] for r in padded])), 3) if padded else None,
        mismatch_rate_unpadded=round(float(np.mean([not r["match"] for r in unpadded])), 3) if unpadded else None,
        n_padded=len(padded), n_unpadded=len(unpadded))
    print(f"T3 divergence: {res['T3_divergence_position']}", flush=True)
    print(f"T4 padding:    {res['T4_padding_correlation']}", flush=True)

    # ---- T5: logit-level comparison on the FIRST generated token (decisive)
    with torch.no_grad():
        sub = prompts[:8]
        single = []
        for p in sub:
            t = tok(p, return_tensors="pt", add_special_tokens=False).to(llm.model.device)
            single.append(llm.model(**t).logits[0, -1].float().cpu())
        tb = tok(sub, padding=True, return_tensors="pt",
                 add_special_tokens=False).to(llm.model.device)
        lb = llm.model(**tb).logits[:, -1].float().cpu()
    deltas, flips, margins = [], 0, []
    for i, s in enumerate(single):
        d = (s - lb[i]).abs().max().item()
        deltas.append(d)
        top2 = torch.topk(s, 2).values
        margin = (top2[0] - top2[1]).item()
        margins.append(margin)
        if s.argmax().item() != lb[i].argmax().item():
            flips += 1
    res["T5_logit_level"] = dict(
        n=len(single),
        max_abs_logit_delta=round(float(max(deltas)), 5),
        median_abs_logit_delta=round(float(np.median(deltas)), 5),
        argmax_flips=flips,
        median_top2_margin=round(float(np.median(margins)), 4),
        interpretation="delta ~1e-2 with flips only where margin < delta => "
                       "bf16 non-associativity. Gross delta => masking bug.")
    print(f"T5 logits: {res['T5_logit_level']}", flush=True)

    del llm
    torch.cuda.empty_cache()

    # ---- T6: fp32 control
    llm32 = load(torch.float32)
    p32 = [conf(q, llm32)[0] for q in queries[:16]]
    u32 = gen(llm32, p32, 1)
    b32 = gen(llm32, p32, 8)
    res["T6_fp32_control"] = dict(
        n=len(p32), matches=sum(x == y for x, y in zip(u32, b32)),
        mismatches=sum(x != y for x, y in zip(u32, b32)),
        bf16_mismatch_rate=round(len(mism)/len(rows), 3))
    print(f"T6 fp32: {res['T6_fp32_control']}", flush=True)

    # ---- verdict
    t2_clean = res["T2_equal_length_batch"]["zero_padding"] and \
        res["T2_equal_length_batch"]["mismatches"] > 0
    logit_small = res["T5_logit_level"]["max_abs_logit_delta"] < 1.0
    late = (res["T3_divergence_position"]["frac_diverging_at_token_0"] or 0) < 0.5
    fp32_better = res["T6_fp32_control"]["mismatches"] < \
        res["T6_fp32_control"]["bf16_mismatch_rate"] * res["T6_fp32_control"]["n"]

    votes = dict(equal_length_still_mismatches=t2_clean, small_logit_delta=logit_small,
                 diverges_late=late, fp32_reduces_mismatch=fp32_better)
    numerics = sum(bool(v) for v in votes.values())
    res["verdict"] = dict(
        votes=votes, votes_for_numerics=numerics, of=len(votes),
        conclusion=("NUMERICAL (fp non-associativity) -- batching is "
                    "mathematically equivalent; corpus valid but not "
                    "bit-reproducible" if numerics >= 3 else
                    "MASKING BUG SUSPECTED -- do NOT run S4"))
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\n=== VERDICT: {res['verdict']['conclusion']}")
    print(f"    ({numerics}/{len(votes)} indicators point to numerics)")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
