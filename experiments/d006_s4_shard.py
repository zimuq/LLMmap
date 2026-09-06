"""
D006 / S4 — generate one model's shard of the Phase-1 corpus.

One invocation = one model = one Slurm job. Sharding is mandatory, not an
optimisation: `qgh` caps any single job at 2 days (docs/ENV.md) and the whole
corpus is 221-267 node-hours, so no single job could ever hold it.

Each shard:
  - generates every Q_0 query under ALL 125 configs (75 build + 25 val + 25
    test). D006's rationale, which I agree with: loading a model once and
    collecting everything it will ever be needed for beats re-loading all 37
    later purely to collect the remaining splits. I2 is untouched by this --
    that invariant governs *analysis* (don't select on test), not when raw data
    is collected.
  - is independently restartable: it checks its own output for already-complete
    configs and resumes, so a requeued job doesn't redo finished work.
  - writes its own status file, so assembly (S5) can tell complete from
    truncated without re-reading gigabytes of JSONL.

Token budget is C7's decided 200-token CEILING (human, 2026-09-06). Full text is
stored; any analysis-time budget <=200 is recoverable by truncation, so C7 below
the ceiling is not frozen into this corpus. See D006/P1 §F1.

Usage:
    PYTHONPATH=. python experiments/d006_s4_shard.py --model <hf_name>
"""
import os
import json
import time
import random
import hashlib
import argparse
import traceback

import torch

from LLMmap.llm import LLM_huggingface
from LLMmap.prompt_configuration import PromptConfFactory, BUILD, VAL, TEST
from LLMmap.dataset_maker import make_dataset_entries_for_new_llm

CONF_DIR = "./confs/prompt_configurations/"
Q0 = "./confs/queries/pool_v1.json"
CORPUS_DIR = os.environ.get("D006_CORPUS", "./data/corpus_v1")

TOKEN_CEILING = 200                       # C7, decided 2026-09-06
SPLIT_SIZES = {BUILD: 75, VAL: 25, TEST: 25}   # C4
SEED = 20260906
# D006/S1c: batch size, set from measurement.
#
# S1's A3 sweep suggested mean response length fell 18% at B=32, and I proposed a
# mechanism (more padding -> more bf16 perturbation -> more early-EOS flips) that
# would have made batch size a corpus variable confounded with model size. S1c
# tested it properly -- 96 PAIRED prompts, token counts rather than chars, both
# the smallest and largest models -- and REFUTED it. Paired deltas wander +-4
# tokens around zero with no trend (B=16 is longer than B=1), and mean_tok on the
# 14B is flat at 84.1/81.7/83.2/82.9 across B=8..64. The A3 signal was a
# small-sample artifact (32 prompts, chars).
#
# So batch size does NOT shift the response distribution, and a per-model B would
# be statistically harmless -- equivalent to a different random seed. B is pinned
# corpus-wide anyway because it costs nothing and removes a variable, NOT because
# varying it would confound anything. On OOM a shard still stops and reports
# rather than silently halving B, so that the choice stays visible and
# corpus-wide -- but that is tidiness, not a correctness requirement.
#
# 64 is the largest value MEASURED to fit the largest model (Phi-3-medium 14B:
# 32.75 GB peak of 95 GB visible, 7.873 gen/s, 6.2x over B=8). B=128 was not
# tested and may well be faster; going past the measured range is exactly the
# extrapolation this plan has criticised elsewhere, so it is left on the table.
CORPUS_BATCH = 64         # S1c-measured; recorded in every shard's status file


def shard_paths(model):
    slug = model.replace("/", "__")
    return (os.path.join(CORPUS_DIR, f"{slug}.jsonl"),
            os.path.join(CORPUS_DIR, f"{slug}.status.json"))


def build_configs():
    """Draw the 125 configs. Seeded per-pool so every shard sees the SAME
    configs -- the corpus is a model x query x config grid, and it stops being
    one if each model draws its own configs."""
    pc = PromptConfFactory(CONF_DIR)
    confs = {}
    for pool, n in SPLIT_SIZES.items():
        random.seed(f"{SEED}:{pool}")
        confs[pool] = pc.sample(n, pool=pool)
    return pc, confs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(CORPUS_DIR, exist_ok=True)
    out_path, status_path = shard_paths(args.model)

    q0doc = json.load(open(Q0))
    queries = [e["text"] for e in q0doc["queries"]]
    pc, confs = build_configs()
    n_expected = len(queries) * sum(SPLIT_SIZES.values())

    # ---- resume: which (pool, config_index) pairs already landed?
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue          # truncated tail from a killed job
                if len(d.get("traces", [])) == len(queries):
                    done.add((d["dataset"], d["config_index"]))
    if done:
        print(f"resuming: {len(done)} configs already complete", flush=True)

    status = dict(model=args.model, status="RUNNING", token_ceiling=TOKEN_CEILING,
                  batch=CORPUS_BATCH, n_queries=len(queries), split_sizes=SPLIT_SIZES,
                  n_expected=n_expected, q0_sha256=q0doc["sha256"],
                  split_schema=pc.split_schema_version,
                  started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    json.dump(status, open(status_path, "w"), indent=1)

    if args.dry_run:
        print(f"[dry-run] {args.model}: {n_expected} generations "
              f"({len(queries)} queries x {sum(SPLIT_SIZES.values())} configs) "
              f"@ {TOKEN_CEILING} tok, batch {CORPUS_BATCH}")
        return

    t0 = time.time()
    try:
        llm = LLM_huggingface(args.model, model_load_kargs=dict(
            torch_dtype=torch.bfloat16, device_map="cuda"))
        # F6: pin the exact weights this shard used -- from_pretrained resolves
        # `main`, which can move. Free now, unrecoverable later.
        status["hf_revision"] = getattr(
            getattr(llm.model, "config", None), "_commit_hash", None)
        status["load_s"] = round(time.time() - t0, 1)
        print(f"loaded in {status['load_s']}s  rev={status['hf_revision']}",
              flush=True)

        n_written = 0
        with open(out_path, "a") as fh:
            for pool in (BUILD, VAL, TEST):
                todo = [(i, c) for i, c in enumerate(confs[pool])
                        if (pool, i) not in done]
                print(f"{pool}: {len(todo)} configs to generate", flush=True)
                for i, conf in todo:
                    ent = make_dataset_entries_for_new_llm(
                        llm, queries, [conf], pool=pool,
                        batch_size=CORPUS_BATCH, max_new_tokens=TOKEN_CEILING)[0]
                    ent["config_index"] = i
                    ent["model"] = args.model
                    fh.write(json.dumps(ent) + "\n")
                    fh.flush()          # restartability depends on this
                    n_written += len(ent["traces"])

        wall = time.time() - t0
        # ---- validate before claiming complete
        total = 0
        empty = 0
        seen = set()
        with open(out_path) as f:
            for line in f:
                d = json.loads(line)
                seen.add((d["dataset"], d["config_index"]))
                for _, a in d["traces"]:
                    total += 1
                    if not a.strip():
                        empty += 1
        ok = (total == n_expected and
              len(seen) == sum(SPLIT_SIZES.values()))
        status.update(
            status="COMPLETE" if ok else "INCOMPLETE",
            n_rows=total, n_configs=len(seen), n_empty_responses=empty,
            generated_this_run=n_written, wall_s=round(wall, 1),
            gen_per_s=round(n_written / wall, 3) if wall else None,
            node_hours=round(wall / 3600, 3),
            sha256=hashlib.sha256(open(out_path, "rb").read()).hexdigest(),
            finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
        print(f"{status['status']}: {total}/{n_expected} rows, "
              f"{len(seen)} configs, {empty} empty, {wall/3600:.2f} node-h, "
              f"{status['gen_per_s']} gen/s", flush=True)

    except torch.cuda.OutOfMemoryError:
        # Deliberately NOT caught by retrying at a smaller batch -- see
        # CORPUS_BATCH above. A per-model batch size is a confound, not a
        # workaround. Surface it and let a human decide corpus-wide.
        status.update(status="FAILED_OOM", error=traceback.format_exc(),
                      wall_s=round(time.time() - t0, 1))
        print(f"SHARD OOM at the corpus-wide batch size {CORPUS_BATCH}. "
              f"NOT retrying smaller -- that would make batch size a per-model "
              f"variable correlated with model size. Stop and report.",
              flush=True)
    except Exception:
        status.update(status="FAILED", error=traceback.format_exc(),
                      wall_s=round(time.time() - t0, 1))
        print("SHARD FAILED\n" + status["error"], flush=True)
    finally:
        json.dump(status, open(status_path, "w"), indent=1)

    if status["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
