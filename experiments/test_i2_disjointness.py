"""
D005 / M3 + D006 / S3 — invariant I2 as an executable assertion, not a convention.

I2 (CLAUDE.md): "Prompting-config splits (build/val/test) are disjoint at the
individual parameter level — no single system prompt, RAG template, or sampling
setting crosses splits."

I2's failure mode is silent: the code runs and produces plausible output while
the holdout is defeated. That is exactly what happened before D005 --
`PromptConfFactory.sample()` accepted `pool` and never forwarded it. This file
turns the invariant into something that fails loudly.

D006/S3 extended it from two pools to three, with a per-collection policy,
because C4's split is three-way and Call 3 (approved 2026-09-05) deliberately
splits the thin collections only two ways:

    systems      3-way disjoint
    temperature  3-way disjoint, BY VALUE (not by index)
    cot_prompts  build-disjoint only; val and test SHARE  <- documented carve-out
    rag_prompts  build-disjoint only; val and test SHARE  <- documented carve-out
    do_sample    shared across all pools                  <- A5 carve-out

The carve-outs are checked as carve-outs: this file asserts that `build` is
disjoint from the holdout AND that val/test genuinely do share, so that a future
edit which silently makes them differ (or which lets `build` leak) fails here.
An unchecked exception is indistinguishable from a bug.

Run standalone (no pytest required):
    PYTHONPATH=. python experiments/test_i2_disjointness.py
Exit status 0 = I2 holds. Non-zero = a violation, with the offending values named.
"""
import sys
import json
import random

from LLMmap.prompt_configuration import PromptConfFactory, BUILD, VAL, TEST

CONF_DIR = "./confs/prompt_configurations/"
N_SAMPLE = 200

# Call 3's approved policy. "3way" = all pools mutually disjoint.
# "build_only" = build disjoint from the holdout; val and test share it.
POLICY = {"systems": "3way", "cot_prompts": "build_only",
          "rag_prompts": "build_only"}
POOLS = (BUILD, VAL, TEST)

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
    return cond


def main():
    random.seed(0)
    pc = PromptConfFactory(CONF_DIR)
    split = pc.train_test_split

    if set(split) != set(POOLS):
        print(f"I2 TEST ABORTED: expected pools {POOLS}, found {sorted(split)}. "
              f"(schema: {getattr(pc, 'split_schema_version', '?')})")
        return 2

    # ---- 1. The split FILES themselves, per Call 3's policy.
    for coll, policy in POLICY.items():
        sets = {p: set(split[p][coll]) for p in POOLS}
        n = len(pc.params[coll])
        check(set().union(*sets.values()) == set(range(n)),
              f"[split file] {coll}: pools do not cover all {n} indices "
              f"(missing {sorted(set(range(n)) - set().union(*sets.values()))})")
        if policy == "3way":
            for a, b in ((BUILD, VAL), (BUILD, TEST), (VAL, TEST)):
                check(not (sets[a] & sets[b]),
                      f"[split file] {coll}: {a}/{b} share indices "
                      f"{sorted(sets[a] & sets[b])}")
        else:
            check(not (sets[BUILD] & sets[VAL]),
                  f"[split file] {coll}: build leaks into the holdout: "
                  f"{sorted(sets[BUILD] & sets[VAL])}")
            check(not (sets[BUILD] & sets[TEST]),
                  f"[split file] {coll}: build leaks into test: "
                  f"{sorted(sets[BUILD] & sets[TEST])}")
            # the carve-out itself, asserted so it can't silently drift
            check(sets[VAL] == sets[TEST],
                  f"[split file] {coll}: Call 3 specifies val and test SHARE "
                  f"this collection, but they differ "
                  f"(val-only {sorted(sets[VAL]-sets[TEST])}, "
                  f"test-only {sorted(sets[TEST]-sets[VAL])})")

    # ---- 2. Sampling universe: split pools disjoint BY VALUE, not by index.
    # Index-disjointness is not enough -- the original temperature list held 11
    # entries but only 10 distinct temperatures (0.9999999999999999 vs 1), so an
    # index-wise split would have leaked an identical setting across pools.
    su_split = pc.sampling_universe_split
    shared = set(pc.sampling_universe_shared)
    for param in pc.sampling_universe:
        if param in shared:
            continue
        vals = {p: {round(float(v), 6) for v in su_split.get(p, {}).get(param, [])}
                for p in POOLS}
        for p in POOLS:
            check(vals[p], f"[sampling] {param}: pool '{p}' has no value set")
        for a, b in ((BUILD, VAL), (BUILD, TEST), (VAL, TEST)):
            check(not (vals[a] & vals[b]),
                  f"[sampling] {param}: {a}/{b} share value(s) "
                  f"{sorted(vals[a] & vals[b])}")

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
                   for coll in POLICY}
    # Read all three from `raw`, which holds the UNTRANSFORMED
    # (system_prompt, cot_prompt, rag_template) triple. Do not use
    # PromptConf.system_prompt: __init__ does `system_prompt or ""`, so a config
    # with no system prompt (~10%, WITH_SYSTEM_P=0.9) surfaces as "" rather than
    # None and would look like an unknown collection entry.
    getter = {"systems": lambda c: c.raw[0] if len(c.raw) > 0 else None,
              "cot_prompts": lambda c: c.raw[1] if len(c.raw) > 1 else None,
              "rag_prompts": lambda c: c.raw[2] if len(c.raw) > 2 else None}

    drawn = {}
    for pool in POOLS:
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

    # ---- 4. Cross-pool: nothing actually drawn may cross where policy forbids.
    for coll, policy in POLICY.items():
        pairs = ((BUILD, VAL), (BUILD, TEST), (VAL, TEST)) if policy == "3way" \
            else ((BUILD, VAL), (BUILD, TEST))
        for a, b in pairs:
            both = drawn[a][0][coll] & drawn[b][0][coll]
            check(not both,
                  f"[cross-pool] {coll}: index(es) {sorted(both)} drawn in BOTH "
                  f"{a} and {b}")
    for a, b in ((BUILD, VAL), (BUILD, TEST), (VAL, TEST)):
        both = drawn[a][1] & drawn[b][1]
        check(not both,
              f"[cross-pool] temperature: value(s) {sorted(both)} drawn in BOTH "
              f"{a} and {b}")

    # ---- report
    if failures:
        print(f"I2 VIOLATED — {len(failures)} failure(s):")
        for f in failures:
            print("  *", f)
        return 1
    print(f"I2 holds ({getattr(pc, 'split_schema_version', '?')}). Checked split "
          f"files, sampling universe, and {N_SAMPLE} sampled configs per pool "
          f"across {len(POOLS)} pools.")
    print(f"  documented carve-outs: {sorted(shared)} shared across all pools "
          f"(DECISIONS.md A5); "
          f"{[c for c, p in POLICY.items() if p == 'build_only']} "
          f"build-disjoint only, val/test share (D006 Call 3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
