"""
D005 / M3 — invariant I2 as an executable assertion, not a convention.

I2 (CLAUDE.md): "Prompting-config splits (build/val/test) are disjoint at the
individual parameter level — no single system prompt, RAG template, or sampling
setting crosses splits."

I2's failure mode is silent: the code runs and produces plausible output while the
holdout is defeated. That is exactly what happened before D005 --
`PromptConfFactory.sample()` accepted `pool` and never forwarded it. This file
turns the invariant into something that fails loudly.

Run standalone (no pytest required):
    PYTHONPATH=. python experiments/test_i2_disjointness.py
Exit status 0 = I2 holds. Non-zero = a violation, with the offending values named.
"""
import sys
import json
import random

from LLMmap.prompt_configuration import PromptConfFactory, TRAIN, TEST

CONF_DIR = "./confs/prompt_configurations/"
N_SAMPLE = 200
failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
    return cond


def main():
    random.seed(0)
    pc = PromptConfFactory(CONF_DIR)
    split = pc.train_test_split

    # ---- 1. The split FILES themselves: disjoint, and covering every index.
    # This property held all along while the code ignored it -- assert it so a
    # future edit to the split file cannot quietly break it either.
    for coll in ("systems", "cot_prompts", "rag_prompts"):
        tr, te = set(split[TRAIN][coll]), set(split[TEST][coll])
        check(not (tr & te),
              f"[split file] {coll}: train/test share indices {sorted(tr & te)}")
        n = len(pc.params[coll])
        check(tr | te == set(range(n)),
              f"[split file] {coll}: pools do not cover all {n} indices "
              f"(missing {sorted(set(range(n)) - (tr | te))})")

    # ---- 2. Sampling universe: split pools disjoint BY VALUE, not by index.
    # Index-disjointness is not enough -- the original temperature list held 11
    # entries but only 10 distinct temperatures (0.9999999999999999 vs 1), so an
    # index-wise split would have leaked an identical setting across pools.
    su_split = pc.sampling_universe_split
    shared = set(pc.sampling_universe_shared)
    for param in pc.sampling_universe:
        if param in shared:
            continue
        tr = {round(float(v), 6) for v in su_split.get(TRAIN, {}).get(param, [])}
        te = {round(float(v), 6) for v in su_split.get(TEST, {}).get(param, [])}
        check(tr and te, f"[sampling] {param}: missing a per-pool value set")
        check(not (tr & te),
              f"[sampling] {param}: pools share value(s) {sorted(tr & te)}")

    # ---- 3. End-to-end: what `sample()` actually returns must respect the pool.
    # Reverse-map each returned prompt STRING back to its index, because sample()
    # hands back PromptConf objects rather than indices.
    # rag_prompts entries are LISTS, not strings, and PromptConf stores the
    # *materialised* rag_prompt rather than the template -- the template is kept
    # in `raw[2]`. Key everything through json.dumps so all three collections can
    # be reverse-mapped uniformly.
    def key(x):
        return json.dumps(x, sort_keys=True)

    text_to_idx = {coll: {key(t): i for i, t in enumerate(pc.params[coll])}
                   for coll in ("systems", "cot_prompts", "rag_prompts")}
    # Read all three from `raw`, which holds the UNTRANSFORMED
    # (system_prompt, cot_prompt, rag_template) triple. Do not use
    # PromptConf.system_prompt: __init__ does `system_prompt or ""`, so a config
    # with no system prompt (~10%, WITH_SYSTEM_P=0.9) surfaces as "" rather than
    # None and would look like an unknown collection entry.
    getter = {"systems": lambda c: c.raw[0] if len(c.raw) > 0 else None,
              "cot_prompts": lambda c: c.raw[1] if len(c.raw) > 1 else None,
              "rag_prompts": lambda c: c.raw[2] if len(c.raw) > 2 else None}

    drawn = {}
    for pool in (TRAIN, TEST):
        confs = pc.sample(N_SAMPLE, pool=pool)
        seen = {c: set() for c in text_to_idx}
        temps = set()
        for c in confs:
            for coll, get in getter.items():
                val = get(c)
                if not val:      # None, or "" for an unselected prompt
                    continue
                idx = text_to_idx[coll].get(key(val))
                check(idx is not None,
                      f"[end-to-end] {pool}: {coll} value not found in its "
                      f"collection -- cannot verify pool membership")
                if idx is not None:
                    seen[coll].add(idx)
                    check(idx in set(split[pool][coll]),
                          f"[end-to-end] {pool}: {coll} index {idx} is not in "
                          f"the {pool} pool")
            temps.add(round(float(c.sampling_hparams["temperature"]), 6))
        drawn[pool] = (seen, temps)
        allowed = {round(float(v), 6)
                   for v in su_split.get(pool, {}).get("temperature", [])}
        check(temps <= allowed,
              f"[end-to-end] {pool}: temperatures {sorted(temps - allowed)} "
              f"outside the {pool} pool")

    # ---- 4. Cross-pool: nothing actually drawn may appear in both pools.
    for coll in text_to_idx:
        a, b = drawn[TRAIN][0][coll], drawn[TEST][0][coll]
        check(not (a & b),
              f"[cross-pool] {coll}: index(es) {sorted(a & b)} drawn in BOTH pools")
    ta, tb = drawn[TRAIN][1], drawn[TEST][1]
    check(not (ta & tb),
          f"[cross-pool] temperature: value(s) {sorted(ta & tb)} drawn in BOTH pools")

    # ---- report
    if failures:
        print(f"I2 VIOLATED — {len(failures)} failure(s):")
        for f in failures:
            print("  *", f)
        return 1
    print(f"I2 holds. Checked split files, sampling universe, and {N_SAMPLE} "
          f"sampled configs per pool.")
    print(f"  documented carve-out (DECISIONS.md A5): {sorted(shared)} "
          f"shared across pools by design")
    return 0


if __name__ == "__main__":
    sys.exit(main())
