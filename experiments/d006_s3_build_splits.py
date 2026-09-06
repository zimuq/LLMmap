"""
D006 / S3 — construct the three-way build/val/test config split (C4: 75/25/25).

Scheme is Call 3 as approved by design side (2026-09-05), option 3:

  systems (84)      -> 3-way disjoint: 42 / 21 / 21
  temperature (10)  -> 3-way disjoint:  4 /  3 /  3   (interleaved, see below)
  cot_prompts (6)   -> 2-way: build-disjoint (3), val and test SHARE (3)
  rag_prompts (9)   -> 2-way: build-disjoint (4), val and test SHARE (5)
  do_sample (2)     -> shared across all pools (A5 carve-out, unchanged)

The rationale for splitting cot/rag only two ways: the leak I2 exists to prevent
is `build` (where selection happens) seeing what `val`/`test` (where evaluation
happens) will see. A val<->test overlap is an over-fitting concern, not a
validity failure. Forcing three disjoint pools out of 6 and 9 items would leave
val and test with 2-3 wrappers each, recreating exactly the confound A5 already
ruled out for `do_sample`.

WHY A NEW FILE RATHER THAN AN EDIT: `train_test_split.json` is the artifact
D005's M4 forensics reasoned about, and it is what LLMmap's shipped corpus was
(probably) generated under. Overwriting it would destroy that evidence. This
writes `split_v2.json` alongside it; `PromptConfFactory` prefers v2 and falls
back to the legacy file, so D001/D004/D005 remain reproducible.

I7: schema version `cdqd-split-v2`, recorded in the file. Migration note is the
paragraph above plus `## R`.

Temperature is INTERLEAVED over sorted values, not block-partitioned. D005/M2
established why: a contiguous split (build = low temperatures, test = high)
would make "which pool is this config in" partly predictable from temperature
alone -- a smaller version of the confound I2 exists to prevent. Round-robin
over sorted distinct values avoids it. It also dedups by VALUE first, because
the shipped list held 11 entries for 10 distinct temperatures
(0.9999999999999999 vs 1.0) and an index-wise split would have put the same
real temperature in two pools.

Usage:  PYTHONPATH=. python experiments/d006_s3_build_splits.py
"""
import json
import random
import hashlib
import collections

CONF = "./confs/prompt_configurations/"
SEED = 20260906

# per-collection policy: how many pools the collection is split across
THREE_WAY = ("systems",)
TWO_WAY = ("cot_prompts", "rag_prompts")   # build-disjoint; val/test share
POOLS = ("build", "val", "test")


def three_way(indices, rng, weights=(0.5, 0.25, 0.25)):
    idx = list(indices)
    rng.shuffle(idx)
    n = len(idx)
    n_b = round(n * weights[0])
    n_v = round(n * weights[1])
    return sorted(idx[:n_b]), sorted(idx[n_b:n_b+n_v]), sorted(idx[n_b+n_v:])


def two_way(indices, rng, build_frac=0.5):
    idx = list(indices)
    rng.shuffle(idx)
    n_b = round(len(idx) * build_frac)
    return sorted(idx[:n_b]), sorted(idx[n_b:])


def main():
    rng = random.Random(SEED)
    general = json.load(open(CONF + "general.json"))

    params = {}
    for coll, fname, key in (("systems", "systems.json", "system_prompts"),
                             ("cot_prompts", "cot_prompts.json", "cot_prompts"),
                             ("rag_prompts", "rag_prompts.json", "rag_templates")):
        try:
            d = json.load(open(CONF + fname))
            params[coll] = d[key] if isinstance(d, dict) else d
        except FileNotFoundError:
            params[coll] = general[key]

    split = {p: {} for p in POOLS}

    # ---- systems: 3-way disjoint
    for coll in THREE_WAY:
        b, v, t = three_way(range(len(params[coll])), rng)
        split["build"][coll], split["val"][coll], split["test"][coll] = b, v, t

    # ---- cot/rag: build-disjoint, val and test share the holdout half
    for coll in TWO_WAY:
        b, held = two_way(range(len(params[coll])), rng)
        split["build"][coll] = b
        split["val"][coll] = held
        split["test"][coll] = held           # deliberate: Call 3 option 3

    # ---- temperature: dedup by VALUE, then interleave over sorted values
    universe = general["sampling_universe"]
    temps = sorted({round(float(t), 6) for t in universe["temperature"]})
    su_split = {p: {} for p in POOLS}
    for i, t in enumerate(temps):
        su_split[POOLS[i % 3]].setdefault("temperature", []).append(t)

    general["sampling_universe_split"] = su_split
    general["sampling_universe_shared"] = ["do_sample"]      # A5, unchanged
    general["_sampling_split_note"] = (
        "D006/S3 (C4, Call 3): three-way build/val/test. temperature is "
        "deduplicated by VALUE then round-robin interleaved over sorted values "
        "-- a contiguous split would make pool membership partly predictable "
        "from temperature alone (D005/M2). do_sample stays shared across all "
        "pools per the A5 carve-out: with 2 distinct values any disjoint split "
        "puts all-greedy in one pool and all-stochastic in another, a worse "
        "confound than the leakage it removes.")

    payload = dict(
        schema_version="cdqd-split-v2",
        built="D006/S3",
        seed=SEED,
        c4_sizes=dict(build=75, val=25, test=25),
        policy=dict(
            systems="3-way disjoint",
            temperature="3-way disjoint, value-deduped, interleaved",
            cot_prompts="2-way: build-disjoint; val and test SHARE (Call 3 opt 3)",
            rag_prompts="2-way: build-disjoint; val and test SHARE (Call 3 opt 3)",
            do_sample="shared across all pools (A5 carve-out)"),
        migration_note=(
            "Supersedes train_test_split.json (2 pools: train/test), which is "
            "PRESERVED unmodified -- it is the artifact D005/M4's forensics "
            "reasoned about and what LLMmap's shipped corpus was generated "
            "under. PromptConfFactory prefers this file and falls back to the "
            "legacy one, so D001/D004/D005 stay reproducible."),
        **{p: split[p] for p in POOLS})

    with open(CONF + "split_v2.json", "w") as f:
        json.dump(payload, f, indent=1)
    with open(CONF + "general.json", "w") as f:
        json.dump(general, f, indent=1)

    print("split_v2.json written\n")
    for coll in ("systems", "cot_prompts", "rag_prompts"):
        sizes = {p: len(split[p][coll]) for p in POOLS}
        allidx = set().union(*(set(split[p][coll]) for p in POOLS))
        shared_vt = set(split["val"][coll]) == set(split["test"][coll])
        print(f"  {coll:14s} total={len(params[coll]):3d}  "
              f"build={sizes['build']:3d} val={sizes['val']:3d} test={sizes['test']:3d}"
              f"  covers_all={allidx == set(range(len(params[coll])))}"
              f"{'  [val==test by design]' if shared_vt else ''}")
    print(f"  {'temperature':14s} total={len(temps):3d}  " +
          "  ".join(f"{p}={len(su_split[p]['temperature'])}" for p in POOLS))
    for p in POOLS:
        print(f"      {p:5s}: {su_split[p]['temperature']}")

    # ---- the checks that matter, asserted here as well as in the I2 test
    b, v, t = (set(split[p]["systems"]) for p in POOLS)
    assert not (b & v) and not (b & t) and not (v & t), "systems not 3-way disjoint"
    for coll in TWO_WAY:
        assert not (set(split["build"][coll]) & set(split["val"][coll])), \
            f"{coll}: build overlaps the holdout"
    tb, tv, tt = (set(su_split[p]["temperature"]) for p in POOLS)
    assert not (tb & tv) and not (tb & tt) and not (tv & tt), \
        "temperature pools overlap BY VALUE"
    print("\nall S3 assertions pass")


if __name__ == "__main__":
    main()
