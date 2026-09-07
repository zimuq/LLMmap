"""
D006 / S7 — verify the invariants hold on the CORPUS AS GENERATED.

The point of this file is that it checks the artifact, not the intent. D005's
whole lesson was that a split can be correct in its definition file and violated
in what the code actually draws; the analogous risk here is a corpus whose
config splits are right on paper and wrong in the JSONL.

  I1  >=3 near-relative pairs among the models actually present. Recomputed from
      model_metadata.csv over the shards that exist, NOT copied from D001 -- if
      the corpus is PARTIAL, the pair structure of what is present is what
      matters, and it can differ from the pair structure of the full universe.
  I2  Every entry's system / CoT / RAG template must belong to the pool that
      entry is labelled with. This reads the generated data; the split FILE is
      checked separately by test_i2_disjointness.py.
  I3  One embedding row per response, never averaged. Verified by count against
      the shard, and by confirming the rows are not degenerate (a collapsed or
      duplicated point cloud would show as near-zero spread).
  I5  Every embedding produced by the frozen model at the frozen dimension.
  I7  Manifest declares a schema version.

Usage:  PYTHONPATH=. python experiments/d006_s7_verify.py
"""
import os
import csv
import json
import glob
import itertools

import numpy as np

CORPUS = os.environ.get("D006_CORPUS", "./data/corpus_v1")
EMB = os.path.join(CORPUS, "embeddings")
CONF = "./confs/prompt_configurations/"
META = "./results/D001/model_metadata.csv"
I5_MODEL = "intfloat/multilingual-e5-large-instruct"
I5_DIM = 1024

failures, notes = [], []


def check(cond, msg):
    (notes if cond else failures).append(msg)
    return cond


def near_relative_pairs(models):
    """Same base, or adjacent versions within one lineage."""
    rows = {r["model"]: r for r in csv.DictReader(open(META))}
    present = [m for m in models if m in rows]
    pairs = []
    for a, b in itertools.combinations(present, 2):
        ra, rb = rows[a], rows[b]
        same_base = ra.get("base") and ra["base"] == rb.get("base")
        same_lineage = ra.get("lineage") and ra["lineage"] == rb.get("lineage")
        if same_base or same_lineage:
            pairs.append((a, b))
    return pairs


def main():
    manifest = json.load(open(os.path.join(CORPUS, "corpus_manifest.json")))
    split = json.load(open(CONF + "split_v2.json"))
    params = {}
    for coll, fname, key in (("systems", "systems.json", "system_prompts"),
                             ("cot_prompts", "cot_prompts.json", "cot_prompts"),
                             ("rag_prompts", "rag_prompts.json", "rag_templates")):
        d = json.load(open(CONF + fname))
        params[coll] = d[key] if isinstance(d, dict) else d
    idx = {c: {json.dumps(t, sort_keys=True): i for i, t in enumerate(params[c])}
           for c in params}

    validated = [s for s in manifest["models"] if s["status"] == "VALIDATED"]
    models = [s["model"] for s in validated]
    print(f"corpus status: {manifest['status']}  ({len(models)} shards)\n")

    # ---- I7
    check(bool(manifest.get("schema_version")),
          f"I7: manifest schema_version = {manifest.get('schema_version')}")

    # ---- I1, recomputed over what is actually present
    pairs = near_relative_pairs(models)
    check(len(pairs) >= 3,
          f"I1: {len(pairs)} near-relative pairs among the {len(models)} models "
          f"present (requirement >=3)")
    if manifest["status"] != "READY":
        notes.append(f"I1 note: recomputed over the PARTIAL corpus, not copied "
                     f"from D001's 65 -- 8 models are absent, and their pairs "
                     f"with them.")

    # ---- I2 on generated data
    getter = {"systems": 0, "cot_prompts": 1, "rag_prompts": 2}
    viol, checked = [], 0
    for s in validated:
        with open(s["shard_path"]) as f:
            for line in f:
                d = json.loads(line)
                pool, raw = d["dataset"], d["prompt_conf"]["raw"]
                for coll, pos in getter.items():
                    val = raw[pos] if len(raw) > pos else None
                    if not val:
                        continue
                    i = idx[coll].get(json.dumps(val, sort_keys=True))
                    checked += 1
                    if i is None:
                        viol.append(f"{s['model']} {pool}: {coll} value not in collection")
                    elif i not in set(split[pool][coll]):
                        viol.append(f"{s['model']} {pool} cfg{d['config_index']}: "
                                    f"{coll} idx {i} not in the {pool} pool")
    check(not viol,
          f"I2: {checked:,} generated (entry, collection) values checked; "
          f"{len(viol)} outside their labelled pool")
    for v in viol[:5]:
        failures.append(f"    {v}")

    # ---- I3 + I5
    tot_emb, degenerate, uniqueness = 0, [], {}
    for s in validated:
        slug = s["model"].replace("/", "__")
        npy, ix = f"{EMB}/{slug}.npy", f"{EMB}/{slug}.index.json"
        if not (os.path.exists(npy) and os.path.exists(ix)):
            failures.append(f"I3: missing embedding for {s['model']}")
            continue
        a = np.load(npy, mmap_mode="r")
        meta = json.load(open(ix))
        tot_emb += a.shape[0]
        if a.shape != (s["n_rows"], I5_DIM):
            failures.append(f"I3: {s['model']} embedding {a.shape} != "
                            f"({s['n_rows']}, {I5_DIM})")
        if meta.get("embedding_model") != I5_MODEL or meta.get("dim") != I5_DIM:
            failures.append(f"I5: {s['model']} embedded with "
                            f"{meta.get('embedding_model')} dim {meta.get('dim')}")
        # What I3 forbids is a COLLAPSED point cloud -- averaging, or every
        # response mapping to one point. That is a spread test.
        #
        # My first version also failed a shard whose sampled rows were <80%
        # unique. That is not I3's failure mode, and it produced a false
        # positive on Llama3-ChatQA-1.5-8B (355/500 unique, median across the
        # 37 is 497/500). Its repeats are canned refusals -- " Sorry. I cannot
        # find the answer based on the context." x86, " Yes" x29, " No" x22 --
        # because it is RAG/QA-tuned and most of Q_0 is not context-grounded QA.
        # That is a distinctive, highly identifiable behaviour: fingerprint
        # SIGNAL, not corruption. Row uniqueness conflates "the pipeline
        # averaged the data" with "the model repeats itself"; only the first is
        # a defect. Uniqueness is now reported, not gated -- with a floor low
        # enough (<10%) that it only fires on something genuinely collapsed.
        sub = np.asarray(a[:2000], dtype=np.float32)
        spread = float(np.mean(np.std(sub, axis=0)))
        uniq = len(np.unique(sub[:500], axis=0))
        uniqueness[s["model"]] = uniq
        if spread < 1e-4 or uniq < 50:
            degenerate.append(f"{s['model']}: spread={spread:.2e} uniq={uniq}/500")
    lo = sorted(uniqueness.items(), key=lambda kv: kv[1])[:3]
    check(not degenerate,
          f"I3: no collapsed point clouds (per-dim spread on every shard; "
          f"median row-uniqueness {int(np.median(list(uniqueness.values())))}/500, "
          f"lowest {', '.join(f'{m.split(chr(47))[-1]}={u}' for m, u in lo)} "
          f"-- low uniqueness is model repetitiveness, i.e. signal, not a defect)")
    for d in degenerate:
        failures.append(f"    {d}")
    check(tot_emb == sum(s["n_rows"] for s in validated),
          f"I3: {tot_emb:,} embedding rows == {sum(s['n_rows'] for s in validated):,} "
          f"responses (one row per response, never averaged)")

    print("PASS")
    for n in notes:
        print("  +", n)
    if failures:
        print("\nFAIL")
        for f in failures:
            print("  -", f)
    out = dict(corpus_status=manifest["status"], models=len(models),
               near_relative_pairs=len(pairs), i2_values_checked=checked,
               i2_violations=len(viol), embedding_rows=tot_emb,
               row_uniqueness_per_500=uniqueness,
               passed=not failures, notes=notes, failures=failures)
    json.dump(out, open("./results/D006/s7_invariants.json", "w"), indent=1)
    print(f"\nwritten: results/D006/s7_invariants.json")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
