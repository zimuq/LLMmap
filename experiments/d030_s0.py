"""D030 / S0.4 — config draws, T_ZP slot order, LLMMAP-OS overlap, manifest baseline_ext_v1.

Any python (CPU, metadata only). Run AFTER d030_met_prompts.py and d030_zp_prepare.py:
the manifest records the frozen prompt files' sha256 and refuses to be written without them.

* All-reference draws (P1 Call 3, approved), uniform over the model's 75 S_build configs:
    MET : one independent draw per (prompt, sample) slot      -> int[25, 10]
    ZP  : one draw per (base query, repeat), shared by that base query's original and its
          4 perturbations                                     -> int[2, 20]
  rng = numpy default_rng(int(sha256("D030|all|<method>|<model>"), 16) % 2**63)
* T_ZP order (P1 Call 4 + Review A1): type each S_test slot RAG > CoT > sys > none (most
  specific wrapper wins); groups in GLOBAL frequency order sys -> RAG -> CoT -> none, empty
  groups skipped per model; each group sorted by slot index; take round-robin. The full
  25-slot order is written; T_ZP in {5, 10, 25} is its prefix. Reads S_test METADATA only (I2).
* LLMMAP-OS overlap (Review A4): the released open-set encoder's llms_map vs universe_v2.
* Generation env / mode / revision per model: D028's multilingual-extension precedent
  (results/D028/manifest.json `ml_final` env, `ml_gen_mode`; d028_generate.sources revision).

Outputs: confs/baselines/baseline_ext_v1.json, results/D030/s0_freeze.json
"""
import os
import sys
import json
import hashlib
import collections

import numpy as np

sys.path.insert(0, "./experiments")
from d028_generate import sources  # noqa: E402  (imported, not edited)

UNIVERSE = "./results/D027/universe_v2.json"
D028_MANIFEST = "./results/D028/manifest.json"
OS_CONF = "./data/pretrained_models/default/conf.json"
MET_PROMPTS = "./confs/baselines/met_prompts.json"
ZP_PROMPTS = "./confs/baselines/zp_prompts.json"
OUT_MANIFEST = "./confs/baselines/baseline_ext_v1.json"
OUT_S0 = "./results/D030/s0_freeze.json"

TYPE_PRECEDENCE = ("RAG", "CoT", "sys", "none")
RR_ORDER = ("sys", "RAG", "CoT", "none")          # Review A1
N_BUILD, N_TEST = 75, 25
MET_SHAPE = (25, 10)                              # prompts x samples
ZP_SHAPE = (2, 20)                                # base queries x repeats (variants share)
MODE_MAP = {"d006-padded": "padded64", "batch1-concurrent": "batch1-concurrent"}


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def seed_int(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16) % 2**63


def conf_type(pc):
    if pc.get("rag_prompt"):
        return "RAG"
    if pc.get("cot_prompt"):
        return "CoT"
    if pc.get("system_prompt"):
        return "sys"
    return "none"


def tzp_order(types):
    groups = {t: [s for s in range(len(types)) if types[s] == t] for t in RR_ORDER}
    order, k = [], 0
    while len(order) < len(types):
        for t in RR_ORDER:
            if k < len(groups[t]):
                order.append(groups[t][k])
        k += 1
    assert sorted(order) == list(range(len(types)))
    return order


def main():
    for p in (MET_PROMPTS, ZP_PROMPTS):
        assert os.path.exists(p), f"{p} missing: freeze prompts first (S0.3)"
    met_p, zp_p = json.load(open(MET_PROMPTS)), json.load(open(ZP_PROMPTS))
    assert met_p["n_prompts"] == 25 and zp_p["n_prompts"] == 10
    universe = json.load(open(UNIVERSE))
    universe = universe["models"] if isinstance(universe, dict) else universe
    assert len(universe) == 85
    man = json.load(open(D028_MANIFEST))["models"]
    v1 = {s["model"] for s in json.load(open("./data/corpus_v1/corpus_manifest.json"))["models"]}
    assert len(v1) == 37 and v1 <= set(universe)

    models, type_tot, rr_first5 = {}, collections.Counter(), collections.Counter()
    for m in universe:
        rev, rows, csrc, _ = sources(m)
        assert sum(k[0] == "build" for k in rows) == N_BUILD and sum(k[0] == "test" for k in rows) == N_TEST
        types = [conf_type(rows[("test", s)]) for s in range(N_TEST)]
        order = tzp_order(types)
        type_tot.update(types)
        rr_first5.update(types[s] for s in order[:5])
        g = np.random.default_rng(seed_int("D030", "all", "met", m))
        met_all = g.integers(0, N_BUILD, size=MET_SHAPE).tolist()
        g = np.random.default_rng(seed_int("D030", "all", "zp", m))
        zp_all = g.integers(0, N_BUILD, size=ZP_SHAPE).tolist()
        x = man[m]
        models[m] = dict(
            in_v1=m in v1, revision=rev, config_source=csrc,
            gen_env=x["ml_final"]["part_status"][0]["gen_env"],
            mode=MODE_MAP[x["ml_gen_mode"]],
            test_types=types, tzp_order=order,
            tzp_types={T: dict(collections.Counter(types[s] for s in order[:T])) for T in (5, 10, 25)},
            all_draws=dict(met=met_all, zp=zp_all),
            all_distinct_configs=dict(met=len({c for r in met_all for c in r}),
                                      zp=len({c for r in zp_all for c in r})))

    os_conf = json.load(open(OS_CONF))
    os_models = sorted(os_conf["llms_map"])
    overlap = sorted(set(os_models) & set(universe))

    manifest = dict(
        schema="baseline_ext_v1",
        d="D030", p="P1 (approved with amendments A1-A4, 2026-10-09)",
        data_dir="./data/baseline_ext_v1/{met,zp}/{slug}.jsonl",
        corpus_untouched=["corpus_v1", "corpus_v2", "corpus_v2_ext_ml", "all v1/v2 tensors"],
        prompts=dict(met=dict(file=MET_PROMPTS, sha256=sha_file(MET_PROMPTS),
                              sha256_prompts=met_p["sha256_prompts"]),
                     zp=dict(file=ZP_PROMPTS, sha256=sha_file(ZP_PROMPTS),
                             sha256_prompts=zp_p["sha256_prompts"],
                             query_embedding_sha256=zp_p["query_embedding_sha256"])),
        protocol=dict(
            met=dict(repo_commit=met_p["met_commit"], n_prompts=25, samples_per_prompt=10,
                     max_new_tokens=50,
                     native_decoding=dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                          repetition_penalty=1.0),
                     prefill=None, system_message=None),
            zp=dict(repo_commit=zp_p["zp_commit"], n_prompts=10, repeats_per_prompt=20,
                    max_new_tokens=512,
                    native_decoding=dict(do_sample=True, temperature=0.7, top_p=0.9, top_k=50),
                    system_message=None, input_truncation=None)),
        rows=dict(gen_ref="native config", native_tgt="native config, independent seed namespace",
                  all_ref="per-slot S_build config (all_draws)", test_tgt="S_test slot s (MET: all 25; "
                  "ZP: tzp_order[:T_ZP], T_ZP fixed at stop A)"),
        config_decoding="config rows pass the config's own sampling_hparams (do_sample, temperature); "
                        "everything else from the model's generation_config, as corpus v2",
        wrapping="baseline prompt replaces the pool query in PromptConf.__call__ (CoT, RAG, system/"
                 "template) via d027_llm.LLMv2 with TEMPLATE_KWARGS",
        line_fields=["schema", "method", "model", "row", "slot", "config_ref", "prompt_conf", "decoding",
                     "samples[{p, r, text, n_tok, finish}]", "gen_env", "revision", "mode", "batch",
                     "seed", "template_kwargs"],
        seed_rule="torch.manual_seed(int(sha256('D030|<method>|<model>|<row>|<slot>|<config_ref>'),16) % 2**31) "
                  "before each generate group",
        type_rule=dict(precedence=list(TYPE_PRECEDENCE), round_robin=list(RR_ORDER),
                       source="P1 Call 4 + Review A1"),
        all_draw_rule="numpy default_rng(int(sha256('D030|all|<method>|<model>'),16) % 2**63).integers(0,75,size)"
                      "; met size (25,10) per (prompt,sample); zp size (2,20) per (base,repeat), shared "
                      "by the base query's 5 variants",
        models=models)
    os.makedirs(os.path.dirname(OUT_MANIFEST), exist_ok=True)
    json.dump(manifest, open(OUT_MANIFEST, "w"), indent=1)

    s0 = dict(
        manifest=OUT_MANIFEST, manifest_sha256=sha_file(OUT_MANIFEST),
        n_models=len(models), n_v1=sum(v["in_v1"] for v in models.values()),
        test_type_totals=dict(type_tot), tzp5_type_totals=dict(rr_first5),
        models_without_type={t: sorted(m for m, v in models.items() if t not in v["test_types"])
                             for t in RR_ORDER},
        env_counts=dict(collections.Counter(v["gen_env"] for v in models.values())),
        mode_counts=dict(collections.Counter(v["mode"] for v in models.values())),
        llmmap_os=dict(encoder=OS_CONF, is_open=os_conf.get("is_open"), n_train_models=len(os_models),
                       overlap_with_universe=overlap, n_overlap=len(overlap),
                       overlap_in_v1=sorted(set(overlap) & v1),
                       overlap_in_v2_new=sorted(set(overlap) - v1),
                       v1_not_overlapping=sorted(v1 - set(overlap))),
        volume_per_model=dict(met_fixed=750, met_test_per_slot=250, zp_fixed=600, zp_test_per_slot=200))
    json.dump(s0, open(OUT_S0, "w"), indent=1)
    print(json.dumps({k: v for k, v in s0.items() if k != "llmmap_os"}, indent=1))
    print("LLMMAP-OS overlap:", s0["llmmap_os"]["n_overlap"], "of", len(universe),
          "| in v1:", len(s0["llmmap_os"]["overlap_in_v1"]), "| v2-new:", s0["llmmap_os"]["overlap_in_v2_new"])


if __name__ == "__main__":
    main()
