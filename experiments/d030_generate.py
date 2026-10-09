"""D030 — generate baseline_ext_v1 rows (S1) and the S0.6 cost pilot.

One invocation = one model (optionally one part of n, for D028's batch-1-concurrent trio).
Loads the model once through d027_llm.LLMv2 at the pinned revision (D028 ML-extension
precedent: same env, same shims, same template kwargs) and generates MET then ZP rows.

Units: one manifest line per (method, row, slot, config_ref):
  gen_ref / native_tgt : native config (no system message, method's native decoding)
  all_ref              : grouped by S_build config (manifest all_draws); MET per (p, r),
                         ZP per (base, r) shared across the base query's 5 variants
  test_tgt             : one S_test slot (MET: all 25; ZP: tzp_order[:T_ZP])
Within a unit the expanded list is prompt-major (p0 r0..rR, p1 ...), as both official
loops order it, chunked at the model's batch size (padded 64, D028's OOM fallback 32/16
for that unit, recorded) or one prompt per call (batch1-concurrent).

`_gen` reproduces LLMmap.llm.LLM_huggingface.generate line for line (left padding,
add_special_tokens=False, pad_token_id = eos, batch_decode skip_special_tokens) and also
returns the generated-token count up to the first EOS and the finish reason.

Pilot (--pilot): seed namespace "pilot", output results/D030/s0_pilot/, never part of
baseline_ext_v1. Its test row is named `test_timing` and is used for timing only (Review
A3): no statistic of any kind is computed on it.

Usage:
  <env python> experiments/d030_generate.py --model M [--methods met,zp] [--tzp T]
                                            [--part i --nparts n] [--pilot [--reduced]]
  <any python> experiments/d030_generate.py --finalize --model M --tzp T
"""
import os
import sys
import json
import time
import hashlib
import argparse

sys.path.insert(0, "./experiments")

MANIFEST = "./confs/baselines/baseline_ext_v1.json"
OUTDIR = "./data/baseline_ext_v1"
PILOT_DIR = os.environ.get("D030_PILOT_DIR", "./results/D030/s0_pilot")   # smoke tests use a separate dir
BATCH = 64


def slug(m):
    return m.replace("/", "__")


def seed_of(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16) % 2**31


def load_prompts(man):
    met = [p["plain"] for p in json.load(open(man["prompts"]["met"]["file"]))["prompts"]]
    zpj = json.load(open(man["prompts"]["zp"]["file"]))
    zp = [p["text"] for p in sorted(zpj["prompts"], key=lambda p: p["prompt_id"])]
    zbase = [p["base"] for p in sorted(zpj["prompts"], key=lambda p: p["prompt_id"])]
    assert zp == zpj["all_queries_order"]
    return dict(met=met, zp=zp), zbase


def units_for(method, mm, prompts, zbase, rows_cfg, tzp, pilot, reduced):
    """Yield (row, slot, config_ref, [(p, r)])."""
    P = len(prompts[method])
    R = 10 if method == "met" else 20
    slots = [(p, r) for p in range(P) for r in range(R)]
    if pilot and reduced:   # batch-1 trio: rates only
        slots = slots[:50] if method == "met" else [(p, r) for p in range(2) for r in range(20)]
    out = []
    for row in ("gen_ref", "native_tgt"):
        if pilot and row == "native_tgt":
            continue
        out.append((row, None, "native", slots))
    draws = mm["all_draws"][method]
    groups = {}
    for p, r in slots:
        c = draws[p][r] if method == "met" else draws[zbase[p]][r]
        groups.setdefault(c, []).append((p, r))
    for c in sorted(groups):
        out.append(("all_ref", None, f"build:{c}", groups[c]))
    if pilot:
        out.append(("test_timing", mm["tzp_order"][0], f"test:{mm['tzp_order'][0]}", slots))
    else:
        tslots = range(25) if method == "met" else mm["tzp_order"][:tzp]
        for s in sorted(tslots):
            out.append(("test_tgt", s, f"test:{s}", slots))
    return out


def _gen(llm, prompts, kw, max_new):
    import torch
    tok = llm.tokenizer
    eos = set()
    for e in (getattr(llm.model.generation_config, "eos_token_id", None), tok.eos_token_id):
        if e is None:
            continue
        eos.update(e if isinstance(e, (list, tuple)) else [e])
    with torch.no_grad():
        enc = tok(prompts, padding=True, return_tensors="pt", add_special_tokens=False,
                  return_token_type_ids=False).to(llm.model.device)
        out = llm.model.generate(**enc, max_new_tokens=max_new, pad_token_id=tok.eos_token_id, **kw)
        gen = [out[i, enc.input_ids[i].shape[0]:] for i in range(len(out))]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
    res = []
    for g, t in zip(gen, texts):
        g = g.tolist()
        k = next((i for i, x in enumerate(g) if x in eos), None)
        res.append((t, len(g) if k is None else k, "length" if k is None else "eos"))
    return res, int(enc.input_ids.shape[1])


def generate(a):
    import torch
    from LLMmap.prompt_configuration import PromptConf
    import d006_s4_shard as q
    from d027_llm import LLMv2, TEMPLATE_KWARGS_RECORD
    from d028_generate import sources

    man = json.load(open(MANIFEST))
    mm = man["models"][a.model]
    prompts, zbase = load_prompts(man)
    rev, rows_cfg, _, _ = sources(a.model)
    assert rev == mm["revision"]
    mode = mm["mode"]
    ns = "pilot" if a.pilot else "D030"
    od = f"{PILOT_DIR}" if a.pilot else OUTDIR
    tag = f".part{a.part}of{a.nparts}" if a.nparts > 1 else ""

    llm = LLMv2(a.model, rev, trust_remote_code=a.model in q.TRUST_REMOTE_CODE,
                tokenizer_kwargs=q.EXTRA_TOKENIZER_KWARGS.get(a.model),
                chat_template_fallback=q.CHAT_TEMPLATE_FALLBACK.get(a.model))
    stat = dict(model=a.model, pilot=a.pilot, mode=mode, part=a.part, nparts=a.nparts,
                gen_env=mm["gen_env"], revision=rev, load=llm.load_info, blocks={},
                started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for method in a.methods.split(","):
        proto = man["protocol"][method]
        os.makedirs(f"{od}/{method}", exist_ok=True)
        path = f"{od}/{method}/{slug(a.model)}{tag}.jsonl"
        done = set()
        if os.path.exists(path):
            for line in open(path):
                try:
                    d = json.loads(line)
                    done.add((d["row"], d["config_ref"]))
                except json.JSONDecodeError:
                    pass
        units = units_for(method, mm, prompts, zbase, rows_cfg, a.tzp, a.pilot, a.reduced)
        if a.nparts > 1:
            units = units[a.part::a.nparts]
        blk = stat["blocks"].setdefault(method, dict(n_gen=0, wall_s=0.0, by_row={}))
        with open(path, "a") as fh:
            for row, slot, cref, slots in units:
                if (row, cref) in done:
                    continue
                if cref == "native":
                    conf, kw = None, dict(proto["native_decoding"])
                    render = lambda t: llm.make_prompt("", t)          # noqa: E731
                    pc = None
                else:
                    pool, idx = cref.split(":")
                    pc = rows_cfg[("build" if pool == "build" else "test", int(idx))]
                    conf = PromptConf.from_dict(pc)
                    kw = dict(conf(prompts[method][0], llm)[1])
                    render = lambda t: conf(t, llm)[0]                  # noqa: E731
                texts = [render(prompts[method][p]) for p, _ in slots]
                seed = seed_of(ns, method, a.model, row, slot, cref)
                t0 = time.time()
                bs = 1 if mode == "batch1-concurrent" else BATCH
                while True:
                    try:
                        torch.manual_seed(seed)
                        res, maxin = [], 0
                        for i in range(0, len(texts), bs):
                            r_, L = _gen(llm, texts[i:i + bs], kw, proto["max_new_tokens"])
                            res += r_
                            maxin = max(maxin, L)
                        break
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if bs <= 16:
                            raise
                        bs //= 2
                dt = time.time() - t0
                rec = dict(schema=man["schema"] + ("-pilot" if a.pilot else ""), method=method,
                           model=a.model, row=row, slot=slot, config_ref=cref, prompt_conf=pc,
                           decoding=dict(kw, max_new_tokens=proto["max_new_tokens"]),
                           samples=[dict(p=p, r=r, text=t, n_tok=n, finish=f)
                                    for (p, r), (t, n, f) in zip(slots, res)],
                           gen_env=mm["gen_env"], revision=rev, mode=mode, batch=bs, seed=seed,
                           max_input_tokens=maxin, wall_s=round(dt, 2),
                           template_kwargs=TEMPLATE_KWARGS_RECORD)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                blk["n_gen"] += len(res)
                blk["wall_s"] += dt
                br = blk["by_row"].setdefault(row, dict(n_gen=0, wall_s=0.0, n_units=0, n_tok=0,
                                                        n_length=0, n_empty=0, batches=[]))
                br["n_gen"] += len(res); br["wall_s"] += dt; br["n_units"] += 1
                br["n_tok"] += sum(n for _, n, _ in res)
                br["n_length"] += sum(f == "length" for _, _, f in res)
                br["n_empty"] += sum(not t.strip() for t, _, _ in res)
                if bs not in br["batches"]:
                    br["batches"].append(bs)
                print(f"[{method}] {row} {cref}: {len(res)} gen in {dt:.1f}s (bs {bs})", flush=True)
    if torch.cuda.is_available():
        stat["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    stat["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    for blk in stat["blocks"].values():
        blk["gen_per_s"] = round(blk["n_gen"] / max(1e-9, blk["wall_s"]), 3)
    os.makedirs(od, exist_ok=True)
    json.dump(stat, open(f"{od}/{slug(a.model)}{tag}.status.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in stat.items() if k != "load"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--methods", default="met,zp")
    ap.add_argument("--tzp", type=int, default=None)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--reduced", action="store_true")
    a = ap.parse_args()
    assert a.pilot or a.tzp in (5, 10, 25), "S1 needs --tzp (fixed at stop A)"
    generate(a)
