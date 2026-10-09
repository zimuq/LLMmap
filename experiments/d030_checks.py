"""D030 / S0.5 — code-path checks. Subcommands (each writes results/D030/s0_checks_<cmd>[_tag].json):

  parity  (generation envs, tokenizer-only; --models list or --env-tag)
      For every model: token ids of OUR rendered native prompt
        tok(llm.make_prompt("", text), add_special_tokens=False)   [d027_llm, TEMPLATE_KWARGS]
      vs MET's official path
        tok(format_as_chat(text), add_special_tokens=True)         [model.py:198-260, verbatim via ast]
      vs ZeroPrint's official path
        tok(apply_chat_template([user], add_generation_prompt=True), add_special_tokens=True,
            truncation=True, max_length=512)                       [instruct_model.py:36-75]
      each also without TEMPLATE_KWARGS, to separate template-kwarg differences from BOS
      handling. Plus (P1 Call 5 / Review) the number of ZP prompts that the official 512-token
      input truncation WOULD cut under this model's ZP config rows (all_ref configs + 25
      S_test slots; rendered length only -- metadata, no generation).
  decoding (v2 env): how transformers treats top_k = 0 (source lines recorded) and the
      native decoding sets as the generation config sees them.
  met     (d030-met env): official mmd_hamming == d030_met.mmd_hamming_fast on toy samples
      and on pilot gen_ref / all_ref samples (Review A3: never test_timing rows).
  zp      (d030-zp env): official get_fingerprint (generation replayed from a pilot gen_ref
      unit) vs d030_zp.fingerprint_official vs fingerprint_fast; self-similarity; determinism.
"""
import os
import sys
import ast
import json
import glob
import argparse
from dataclasses import dataclass, asdict

sys.path.insert(0, "./experiments")
MET_REPO = "/work/11280/zimuq1/vista/ext/model-equality-testing"
MANIFEST = "./confs/baselines/baseline_ext_v1.json"
PILOT = "./results/D030/s0_pilot"
OUT = "./results/D030"


def dump(name, obj):
    json.dump(obj, open(f"{OUT}/s0_checks_{name}.json", "w"), indent=1, ensure_ascii=False)


def _met_format_as_chat():
    """MET's TransformersModel.format_as_chat, extracted verbatim from the pinned file."""
    src = open(f"{MET_REPO}/experiments/sampling/model.py").read()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TransformersModel")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "format_as_chat")

    @dataclass
    class Message:
        role: str
        content: str
    ns = dict(Message=Message, asdict=asdict)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "met_model_format_as_chat", "exec"), ns)
    return ns["format_as_chat"], ast.get_source_segment(src, fn)


def parity(a):
    import d006_s4_shard as q
    from d027_llm import LLMv2, render
    from LLMmap.prompt_configuration import PromptConf
    from d028_generate import sources
    man = json.load(open(MANIFEST))
    met = [p["plain"] for p in json.load(open(man["prompts"]["met"]["file"]))["prompts"]]
    zpj = json.load(open(man["prompts"]["zp"]["file"]))
    zp = zpj["all_queries_order"]
    zbase = [p["base"] for p in sorted(zpj["prompts"], key=lambda p: p["prompt_id"])]
    fmt, _ = _met_format_as_chat()

    class Stub:
        pass
    models = a.models.split(",") if a.models else [m for m, v in man["models"].items()
                                                    if v["gen_env"] == a.gen_env]
    res = {}
    for m in models:
        mm = man["models"][m]
        llm = LLMv2(m, mm["revision"], tokenizer_only=True, trust_remote_code=m in q.TRUST_REMOTE_CODE,
                    tokenizer_kwargs=q.EXTRA_TOKENIZER_KWARGS.get(m),
                    chat_template_fallback=q.CHAT_TEMPLATE_FALLBACK.get(m))
        tok = llm.tokenizer
        st = Stub(); st.tokenizer = tok
        r = dict(bos=tok.bos_token, supports_system=llm.supports_system_role, met={}, zp={})
        for name, texts in (("met", met[:3]), ("zp", zp[:3])):
            eq, eq_nokw, eq_nobos, examples = [], [], [], []
            for t in texts:
                ours = tok(llm.make_prompt("", t), add_special_tokens=False)["input_ids"]
                ours_nokw = tok(render(tok, llm.supports_system_role, "", t, tkw={}),
                                add_special_tokens=False)["input_ids"]
                if name == "met":
                    off = tok(fmt(st, t), add_special_tokens=True)["input_ids"]
                else:
                    s = tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                                add_generation_prompt=True)
                    off = tok([s], truncation=True, max_length=512, padding=True,
                              padding_side="left")["input_ids"][0]
                eq.append(ours == off)
                eq_nokw.append(ours_nokw == off)
                strip = lambda x: x[1:] if tok.bos_token_id is not None and x[:1] == [tok.bos_token_id] else x  # noqa
                eq_nobos.append(strip(strip(off)) == strip(ours_nokw))
                if ours_nokw != off and len(examples) < 1:
                    examples.append(dict(ours_head=tok.convert_ids_to_tokens(ours_nokw[:6]),
                                         official_head=tok.convert_ids_to_tokens(off[:6]),
                                         n_ours=len(ours_nokw), n_official=len(off)))
            r[name] = dict(equal=all(eq), equal_without_template_kwargs=all(eq_nokw),
                           equal_modulo_bos=all(eq_nobos), examples=examples)
        if r["zp"]["examples"]:
            h = r["zp"]["examples"][0]["official_head"]
            r["zp"]["official_double_bos"] = bool(tok.bos_token and h[:2] == [tok.bos_token] * 2)
        # Call 5: rendered ZP prompt lengths under this model's ZP config rows
        _, rows, _, _ = sources(m)
        cfgs = sorted({("build", c) for rr in mm["all_draws"]["zp"] for c in rr}) + \
            [("test", s) for s in range(25)]
        n_over, units_over, maxlen = 0, set(), 0
        for k in cfgs:
            conf = PromptConf.from_dict(rows[k])
            for p, t in enumerate(zp):
                if k[0] == "build" and not any(c == k[1] for c in mm["all_draws"]["zp"][zbase[p]]):
                    continue
                L = len(tok(conf(t, llm)[0], add_special_tokens=False)["input_ids"])
                maxlen = max(maxlen, L)
                if L > 512:
                    n_over += 1
                    units_over.add(f"{k[0]}:{k[1]}")
        r["zp_trunc"] = dict(n_rendered_prompts_over_512=n_over, n_config_units_affected=len(units_over),
                             units=sorted(units_over), max_rendered_tokens=maxlen,
                             test_slots_affected=sorted(int(u.split(":")[1]) for u in units_over
                                                        if u.startswith("test")))
        res[m] = r
        print(m, "met", r["met"]["equal"], r["met"]["equal_without_template_kwargs"],
              "| zp", r["zp"]["equal"], r["zp"]["equal_without_template_kwargs"],
              "| trunc", n_over, flush=True)
    dump(f"parity_{a.tag}", res)


def decoding(a):
    import inspect
    import transformers
    from transformers import GenerationConfig
    from transformers.generation import utils as gu
    src = inspect.getsource(gu)
    lines = [l.strip() for l in src.splitlines() if "top_k" in l and ("TopK" in l or "!= 0" in l or "is not None" in l)]
    man = json.load(open(MANIFEST))
    out = dict(transformers=transformers.__version__, topk_lines=lines[:12])
    for k in ("met", "zp"):
        gc = GenerationConfig(**man["protocol"][k]["native_decoding"],
                              max_new_tokens=man["protocol"][k]["max_new_tokens"])
        gc.validate()
        out[k] = {x: getattr(gc, x) for x in ("do_sample", "temperature", "top_p", "top_k",
                                              "repetition_penalty", "max_new_tokens")}
    dump("decoding", out)
    print(json.dumps(out, indent=1))


def _pilot_units(method, rows=("gen_ref", "all_ref")):
    units = {}
    for f in sorted(glob.glob(f"{PILOT}/{method}/*.jsonl")):
        for line in open(f):
            d = json.loads(line)
            if d["row"] in rows:                       # Review A3: test_timing never read here
                units.setdefault(d["model"], {}).setdefault(d["row"], []).append(d)
    return units


def met(a):
    import numpy as np
    import d030_met as M
    rng = np.random.default_rng(20261009)
    checks, worst = [], 0.0
    for trial in range(20):                         # toy: random code points, shared prompts
        n_p = int(rng.integers(1, 6))
        X = np.concatenate([rng.integers(0, n_p, (40, 1)), rng.integers(-1, 4, (40, 30))], 1)
        Y = np.concatenate([rng.integers(0, n_p, (35, 1)), rng.integers(-1, 4, (35, 30))], 1)
        o, f = M.official(X, Y, n_p), M.mmd_hamming_fast(X, Y)
        worst = max(worst, abs(o - f))
    checks.append(dict(kind="toy", n=20, max_abs_diff=worst))
    units = _pilot_units("met")
    pairs = []
    for m, rr in units.items():
        g = rr.get("gen_ref", [])
        al = [s for u in rr.get("all_ref", []) for s in u["samples"]]
        if g and al:
            pairs.append((f"{m}: gen_ref vs all_ref", g[0]["samples"], al))
    ms = sorted(m for m in units if units[m].get("gen_ref"))
    for x, y in zip(ms, ms[1:]):
        pairs.append((f"{x} vs {y}: gen_ref", units[x]["gen_ref"][0]["samples"], units[y]["gen_ref"][0]["samples"]))
    worst = 0.0
    rows = []
    for name, s1, s2 in pairs:
        X, Y = M.to_sequences(s1, 25), M.to_sequences(s2, 25)
        common = np.intersect1d(np.unique(X[:, 0]), np.unique(Y[:, 0]))
        X, Y = X[np.isin(X[:, 0], common)], Y[np.isin(Y[:, 0], common)]
        o, f = M.official(X, Y, 25), M.mmd_hamming_fast(X, Y)
        worst = max(worst, abs(o - f))
        rows.append(dict(pair=name, n_x=len(X), n_y=len(Y), official=o, fast=f))
    checks.append(dict(kind="pilot (gen_ref / all_ref only)", n=len(rows), max_abs_diff=worst, rows=rows))
    ok = all(c["max_abs_diff"] <= 1e-12 for c in checks)
    dump("met", dict(passed=ok, threshold=1e-12, checks=checks))
    print("MET check", "PASS" if ok else "FAIL", [c["max_abs_diff"] for c in checks])


def zp(a):
    import numpy as np
    import torch
    import d030_zp as Z
    units = _pilot_units("zp", rows=("gen_ref",))
    m = a.model
    rec = units[m]["gen_ref"][0]
    outs = Z.outputs_from_record(rec)
    zpo = Z.load_official()
    allq = list(zpo.embedding_data["all_queries"])

    def replay(model, queries, truncate_length=None):
        assert list(queries) == allq
        return list(outs)
    zpo._generate_model_outputs = replay
    F_off = zpo.get_fingerprint(model=None)                      # official end-to-end path
    F_off = zpo._aggregate_gradients(F_off, zpo.config)
    E = Z.output_embeddings(zpo, outs)
    F_ours = Z.fingerprint_official(zpo, E)
    F_fast = torch.tensor(Z.fingerprint_fast(zpo, E), dtype=torch.float32)
    E2 = Z.output_embeddings(zpo, outs)
    F_ours2 = Z.fingerprint_official(zpo, E2)
    d_ours = float((F_off - F_ours).abs().max())
    rel_fast = float((F_off - F_fast).abs().max() / F_off.abs().max())
    res = dict(model=m, n_outputs=len(outs), fingerprint_dim=int(F_off.numel()),
               max_abs_official_vs_ours=d_ours,
               max_rel_official_vs_fast=rel_fast,
               sim_official_vs_fast=Z.similarity(zpo, F_off, F_fast),
               self_similarity=Z.similarity(zpo, F_off, F_off),
               deterministic_bit_identical=bool(torch.equal(F_ours, F_ours2)),
               fingerprint_norm=float(F_off.norm()))
    res["passed"] = (d_ours <= 1e-6 and rel_fast <= 1e-4 and abs(res["self_similarity"] - 1) < 1e-6
                     and res["deterministic_bit_identical"])
    dump("zp", res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["parity", "decoding", "met", "zp"])
    ap.add_argument("--models", default=None)
    ap.add_argument("--gen-env", default="llmmap-gpu-v2")
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    a = ap.parse_args()
    dict(parity=parity, decoding=decoding, met=met, zp=zp)[a.cmd](a)
