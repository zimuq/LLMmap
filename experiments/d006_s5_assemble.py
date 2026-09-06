"""
D006 / S5 — validate every shard, assemble, and write the corpus manifest.

Call 2 as approved (design side, 2026-09-05):

  FREEZE POINT. The corpus is not "the frozen corpus" under I6 until ALL 37
  shards satisfy every check below. Until then the manifest's top-level
  `status` is PARTIAL, and that field -- not the presence of files -- is what
  downstream code must test. There is deliberately no ambiguous half-done state
  that something could mistake for final.

  ALL-37-OR-NOTHING for anything computing Sep(.,.). Not tidiness: CVaR_gamma
  is a TAIL statistic over the pair set, so dropping one model deletes 36 pairs
  and changes what the tail *is*. A 35/37 corpus is a different experiment, not
  a smaller one. A PARTIAL corpus may be inspected; it may not be consumed.

  MANIFEST pins, per F6, each shard's resolved HF commit hash. from_pretrained
  resolves `main`, which moves; capturing it is free now and impossible later.

Validation per shard ("validated" defined explicitly so PARTIAL is unambiguous):
  - status file says COMPLETE
  - row count == |Q_0| x 125 exactly
  - all 125 config indices present, across the three pools at C4's sizes
  - empty/whitespace-only responses below EMPTY_RATE_MAX (they are DATA, not
    corruption -- see below)
  - file sha256 matches what the shard recorded (catches truncation/corruption)
  - the shard's Q_0 hash and split schema match this run's

Usage:  PYTHONPATH=. python experiments/d006_s5_assemble.py
"""
import os
import csv
import json
import glob
import time
import hashlib
import collections

CORPUS_DIR = os.environ.get("D006_CORPUS", "./data/corpus_v1")
Q0 = "./confs/queries/pool_v1.json"
SPLIT = "./confs/prompt_configurations/split_v2.json"
META = "./results/D001/model_metadata.csv"
MANIFEST = os.path.join(CORPUS_DIR, "corpus_manifest.json")
SCHEMA = "cdqd-corpus-v1"          # I7: first version of this project's corpus
SPLIT_SIZES = {"build": 75, "val": 25, "test": 25}
EMPTY_RATE_MAX = 0.02      # see validate(): empties are data; a HIGH RATE is not


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def a1_universe():
    rows = [r for r in csv.DictReader(open(META))
            if r["proprietary"].strip().lower() not in ("true", "1", "yes")
            and float(r["params_b"]) <= 14]
    assert len(rows) == 37, f"A1 anchor violated: {len(rows)} models, expected 37"
    return [r["model"] for r in rows]


def validate(model, q0_sha, n_queries, split_schema):
    slug = model.replace("/", "__")
    jl = os.path.join(CORPUS_DIR, f"{slug}.jsonl")
    st = os.path.join(CORPUS_DIR, f"{slug}.status.json")
    rec = dict(model=model, shard_path=jl, status="MISSING", problems=[])

    if not os.path.exists(st):
        rec["problems"].append("no status file")
        return rec
    status = json.load(open(st))
    # Includes the per-shard DEVIATION fields, not just performance ones. If a
    # shard ran with trust_remote_code, a substituted chat template, or a
    # different sentencepiece, that has to be a manifest FIELD -- a deviation
    # recorded only in a prose footnote is one that gets lost.
    rec.update({k: status.get(k) for k in
                ("hf_revision", "token_ceiling", "batch", "wall_s", "node_hours",
                 "gen_per_s", "load_s", "finished",
                 "env", "trust_remote_code", "chat_template_source")})

    if status.get("status") != "COMPLETE":
        rec["status"] = status.get("status", "UNKNOWN")
        rec["problems"].append(f"shard status is {rec['status']}")
        return rec
    if not os.path.exists(jl):
        rec["problems"].append("status COMPLETE but no jsonl")
        return rec

    if status.get("q0_sha256") != q0_sha:
        rec["problems"].append("Q_0 hash differs from this run's pool_v1.json")
    if status.get("split_schema") != split_schema:
        rec["problems"].append(
            f"split schema {status.get('split_schema')} != {split_schema}")

    n_rows, n_empty = 0, 0
    per_pool = collections.Counter()
    seen = set()
    with open(jl) as f:
        for ln, line in enumerate(f, 1):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                rec["problems"].append(f"unparseable line {ln} (truncated tail?)")
                break
            key = (d["dataset"], d["config_index"])
            if key in seen:
                rec["problems"].append(f"duplicate config {key}")
            seen.add(key)
            per_pool[d["dataset"]] += 1
            for _, a in d["traces"]:
                n_rows += 1
                if not a or not a.strip():
                    n_empty += 1

    expected = n_queries * sum(SPLIT_SIZES.values())
    if n_rows != expected:
        rec["problems"].append(f"{n_rows} rows, expected {expected}")
    for pool, n in SPLIT_SIZES.items():
        if per_pool[pool] != n:
            rec["problems"].append(f"{pool}: {per_pool[pool]} configs, expected {n}")
    # An empty response is a legitimate model output, not corruption: Q_0
    # contains malformed-alignment and prompt-injection probes, and a model that
    # emits EOS immediately on one of those is telling us something -- it is
    # fingerprint signal, not a defect. (Falcon3-7B: 12 of 32,375 = 0.04%.)
    # My first draft failed a shard on ANY empty response, which is the same
    # over-strict-criterion mistake as S1's character-exact A1 check. What would
    # actually indicate a broken shard is an ANOMALOUS RATE, so that is the test.
    rec["empty_rate"] = round(n_empty / max(n_rows, 1), 5)
    if rec["empty_rate"] > EMPTY_RATE_MAX:
        rec["problems"].append(
            f"{n_empty} empty responses ({rec['empty_rate']:.2%}) exceeds "
            f"{EMPTY_RATE_MAX:.0%} -- anomalous, likely a broken shard rather "
            f"than genuine refusals")

    digest = sha256(jl)
    if status.get("sha256") and status["sha256"] != digest:
        rec["problems"].append("sha256 differs from what the shard recorded "
                               "(file changed or was truncated after writing)")

    rec.update(n_rows=n_rows, n_configs=len(seen), n_empty=n_empty,
               per_pool=dict(per_pool), sha256=digest,
               size_bytes=os.path.getsize(jl))
    rec["status"] = "VALIDATED" if not rec["problems"] else "INVALID"
    return rec


def _modal_env(shards):
    """The environment most shards ran under; anything else is a deviation."""
    envs = collections.Counter(json.dumps(s["env"], sort_keys=True)
                               for s in shards if s.get("env"))
    return json.loads(envs.most_common(1)[0][0]) if envs else None


def main():
    q0doc = json.load(open(Q0))
    q0_sha, n_queries = q0doc["sha256"], q0doc["n"]
    split = json.load(open(SPLIT))
    models = a1_universe()

    shards = [validate(m, q0_sha, n_queries, split["schema_version"])
              for m in models]
    good = [s for s in shards if s["status"] == "VALIDATED"]
    bad = [s for s in shards if s["status"] != "VALIDATED"]

    total_rows = sum(s.get("n_rows", 0) for s in good)
    node_hours = sum(s.get("node_hours") or 0 for s in good)

    manifest = dict(
        schema_version=SCHEMA,
        status="READY" if len(good) == 37 else "PARTIAL",
        freeze_rule=("READY requires all 37 shards VALIDATED. A PARTIAL corpus "
                     "may be inspected but MUST NOT be consumed by anything "
                     "computing Sep(.,.): CVaR_gamma is a tail statistic over "
                     "the pair set, so a missing model changes what the tail "
                     "is, not just its sample size. Downstream code must test "
                     "this field, not the presence of files."),
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        code_commit=os.popen("git rev-parse HEAD").read().strip() or None,
        token_ceiling=next((s.get("token_ceiling") for s in good), None),
        batch=next((s.get("batch") for s in good), None),
        q0=dict(path=Q0, sha256=q0_sha, n=n_queries,
                composition=q0doc["composition"]),
        splits=dict(path=SPLIT, schema=split["schema_version"],
                    sizes=SPLIT_SIZES, policy=split["policy"],
                    sha256=sha256(SPLIT)),
        empty_responses=dict(
            total=sum(s.get("n_empty", 0) for s in good),
            by_model={s["model"]: s.get("n_empty", 0) for s in good
                      if s.get("n_empty")},
            note="Empty responses are retained, not dropped. A model going "
                 "silent on a malformed/injection probe is fingerprint signal. "
                 "Downstream may filter them, but must do so explicitly."),
        # Surfaced at the top level so a reader does not have to scan 37 shard
        # records to discover that some ran differently from the rest.
        deviations=dict(
            trust_remote_code=[s["model"] for s in good
                               if s.get("trust_remote_code")],
            substituted_chat_template=[s["model"] for s in good
                                       if s.get("chat_template_source")
                                       and s["chat_template_source"] != "model's own"],
            non_default_env=[dict(model=s["model"], env=s["env"]) for s in good
                             if s.get("env") and s["env"] != _modal_env(good)],
            modal_env=_modal_env(good),
            # Honest provenance gap, recorded rather than back-filled with an
            # assumption. Per-shard env capture was added partway through S4, so
            # shards generated before it carry no env record. They ran under
            # `llmmap-gpu`, but its package set CHANGED DURING THE RUN -- protobuf
            # and sentencepiece were installed mid-S4 to unblock three models --
            # so the exact set at each of their runtimes is not recoverable.
            # Impact is nil for generated content: both packages affect only
            # tokenizer LOADING, and every one of these shards loaded and
            # produced a full 32,375 rows. Not worth re-running for provenance
            # alone; worth stating so nobody later infers uniformity that was
            # never verified.
            env_not_recorded=[s["model"] for s in good if not s.get("env")]),
        totals=dict(models_validated=len(good), models_expected=37,
                    rows=total_rows,
                    rows_expected=37 * n_queries * sum(SPLIT_SIZES.values()),
                    node_hours=round(node_hours, 2)),
        models=shards,
    )
    os.makedirs(CORPUS_DIR, exist_ok=True)
    json.dump(manifest, open(MANIFEST, "w"), indent=1)

    print(f"corpus status: {manifest['status']}  "
          f"({len(good)}/37 shards validated)")
    print(f"rows: {total_rows:,} / {manifest['totals']['rows_expected']:,}")
    print(f"cost: {node_hours:.1f} node-hours")
    if bad:
        print(f"\n{len(bad)} shard(s) not validated:")
        for s in bad:
            print(f"  {s['model']:52s} {s['status']}")
            for p in s["problems"][:3]:
                print(f"      - {p}")
    print(f"\nmanifest: {MANIFEST}")
    return 0 if manifest["status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
