"""D027 / S0 step 9 — assemble the S0 report from preflight/staging/tokparity.

Final status per model: the v2 preflight record, or for a v1 model that could
not run in v2, its D006-env record (Review: v1 fallback, gen_env recorded).
Drops follow the D's rules only; rule 5 is re-applied after drops.
Review A2 for the hybrids (Falcon-H1 x2, Qwen3.5 x2): batch 1 in S1 if any
batched output is degenerate where the unbatched one is not, or if the
exact-match rate is > 20pp below the attention-model baseline (median over
every other surviving model in this preflight).

Writes results/D027/s0.json and results/D027/survivors.txt (new models only).

Usage:  python experiments/d027_s0_report.py
"""
import os
import json
import glob
import subprocess
import statistics

PRE = "./results/D027/preflight"
HYBRIDS = {"tiiuae/Falcon-H1-3B-Instruct", "tiiuae/Falcon-H1-7B-Instruct",
           "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"}
ENV_RANK = ["v2", "old", "internlm"]


def main():
    recs = {}
    for f in glob.glob(f"{PRE}/*.json"):
        r = json.load(open(f))
        recs.setdefault(r["model"], {})[r["env_tag"]] = r
    v1 = sorted(s["model"] for s in json.load(open("./data/corpus_v1/corpus_manifest.json"))["models"])
    cands = [l.split(",")[0] for l in open("./results/D027/candidates_metadata.csv").read().splitlines()[1:]]
    final, missing = {}, []
    for m in v1 + cands:
        if m not in recs:
            missing.append(m)
            continue
        ok = [t for t in ENV_RANK if t in recs[m] and recs[m][t].get("drop") is None]
        if ok:
            r = recs[m][ok[0]]
            final[m] = dict(status="SURVIVOR", gen_env=ok[0], rec=r)
        else:
            t = next(t for t in ENV_RANK if t in recs[m])
            final[m] = dict(status="DROP", reason=recs[m][t]["drop"], gen_env=t, rec=recs[m][t],
                            tried=sorted(recs[m]))
    assert not missing, f"no preflight record for: {missing}"
    v1_drops = [m for m in v1 if final[m]["status"] == "DROP"]

    # rule 5 after drops (candidates only; v1 rows cannot leave the universe)
    surv = [m for m in cands if final[m]["status"] == "SURVIVOR"]
    open("./results/D027/survivors.txt", "w").write("\n".join(surv) + "\n")
    subprocess.run(["python3", "experiments/d027_metadata.py", "--survivors",
                    "./results/D027/survivors.txt"], check=True)
    npr = json.load(open("./results/D027/n_prime.json"))
    for m in npr["candidates_without_neighbour"]:
        final[m].update(status="DROP", reason="no N-neighbour after drops (rule 5)")
    surv = [m for m in surv if m not in npr["candidates_without_neighbour"]]
    open("./results/D027/survivors.txt", "w").write("\n".join(surv) + "\n")
    subprocess.run(["python3", "experiments/d027_metadata.py", "--survivors",
                    "./results/D027/survivors.txt"], check=True)
    npr = json.load(open("./results/D027/n_prime.json"))

    # A2 for hybrids
    pool = [final[m]["rec"]["padding_sanity"]["exact"] / 8 for m in v1 + surv
            if final[m]["status"] == "SURVIVOR" and m not in HYBRIDS]
    base = statistics.median(pool)
    a2 = {}
    for m in HYBRIDS:
        if final.get(m, {}).get("status") != "SURVIVOR":
            continue
        ps = final[m]["rec"]["padding_sanity"]
        rate = ps["exact"] / ps["n"]
        b1 = ps["degenerate_batched_only"] > 0 or rate < base - 0.20
        a2[m] = dict(exact_rate=rate, degenerate_batched_only=ps["degenerate_batched_only"],
                     baseline=base, batch1=b1)

    # S1 projection (node-h): load + 800 gens (+ 518 replay gens for v1 in (f1))
    proj = {}
    for m, f in final.items():
        if f["status"] != "SURVIVOR":
            continue
        r = f["rec"]
        rate = r["throughput"]["gen_per_s"]
        if a2.get(m, {}).get("batch1"):
            rate = r["padding_sanity"]["n"] / r["padding_sanity"]["single_s"]
        n = 800 + (518 if m in v1 else 0)
        proj[m] = round((r["load_s"] + n / rate) / 3600, 3)
    s1 = sum(proj.values())

    think = {m: f["rec"]["thinking"]["raw_quote"] for m, f in final.items()
             if m in cands and "thinking" in f["rec"]}
    out = dict(
        schema="d027-s0-v1",
        env=dict(name="envs/llmmap-gpu-v2 (venv)", freeze="results/D027/env_llmmap-gpu-v2.freeze.txt"),
        staging=json.load(open("./results/D027/staging.json")),
        tokparity=json.load(open("./results/D027/tokparity.json")),
        configs_manifest_sha="see results/D027/configs_manifest.json",
        n_v1=len(v1), v1_drops=v1_drops,
        v1_gen_env={m: final[m]["gen_env"] for m in v1},
        n_candidates=len(cands), survivors=surv,
        drops={m: dict(reason=f["reason"], tried=f.get("tried")) for m, f in final.items()
               if f["status"] == "DROP"},
        compat_shims={m: f["rec"]["load"].get("compat_shims") for m, f in final.items()
                      if f["status"] == "SURVIVOR" and f["rec"]["load"].get("compat_shims")},
        revision_mismatch={m: f["rec"].get("revision_mismatch") for m, f in final.items()
                           if f["rec"].get("revision_mismatch")},
        date_checks={m: f["rec"].get("date_frozen_ok") for m, f in final.items()
                     if f["rec"].get("template_uses_strftime_now")},
        model_class_route={m: f["rec"]["load"].get("model_class_route") for m, f in final.items()
                           if f["status"] == "SURVIVOR" and f["rec"]["load"].get("model_class_route") != "AutoModelForCausalLM"},
        empty_think_block_in_prompt=[m for m, f in final.items() if f["rec"].get("prompt_has_empty_think_block")],
        thinking_raw_quotes=think,
        padding_baseline_median_exact=base, a2_hybrids=a2,
        preflight_table={m: dict(status=f["status"], gen_env=f["gen_env"],
                                 load_s=f["rec"].get("load_s"),
                                 gen_per_s=f["rec"].get("throughput", {}).get("gen_per_s"),
                                 peak_mem_gb=f["rec"].get("peak_mem_gb"),
                                 pad_exact=f["rec"].get("padding_sanity", {}).get("exact"),
                                 system_role=f["rec"].get("supports_system_role"))
                         for m, f in final.items()},
        n_prime=dict(n_universe=npr["n_universe"], n_pairs=npr["n_pairs"],
                     n_pairs_with_candidate=npr["n_pairs_with_candidate"]),
        projection_node_h=dict(s1_generation=round(s1, 2), per_model=proj))
    json.dump(out, open("./results/D027/s0.json", "w"), indent=1)
    print(f"survivors {len(surv)}/{len(cands)}; v1 drops {v1_drops}; |N'| {npr['n_pairs']}; "
          f"S1 projection {s1:.1f} node-h; A2 {a2}")


if __name__ == "__main__":
    main()
