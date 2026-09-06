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
import sys
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

# --- Per-model load quirks, found by experiments/d006_preflight.py -----------
#
# trust_remote_code executes code published in the model repo. It is scoped to
# an explicit ALLOWLIST rather than enabled globally, so the corpus records
# exactly which two repos' custom code we ran. Both are the mainstream repos
# for their models and both genuinely require it (Deci's model class, InternLM's
# tokenizer+model classes); no other A1 model needs it.
TRUST_REMOTE_CODE = {
    "Deci/DeciLM-7B-instruct",
    "internlm/internlm2_5-7b-chat",
}

# internlm/internlm2_5-7b-chat needs use_fast=False. Its repo ships no
# tokenizer.json, so AutoTokenizer attempts a SentencePiece->fast conversion that
# fails. The underlying cause is NOT transformers: sentencepiece 0.2.x rejects
# internlm2's vocabulary ("piece must not include null character"); 0.1.99
# accepts it. Verified the local tokenizer.model is byte-identical to remote, so
# this is not corruption. A GLOBAL downgrade is unsafe -- under sp 0.1.99,
# EuroLLM-1.7B and Mistral-7B-v0.3 both fail with protobuf descriptor errors --
# so this shard runs in a DEDICATED env (envs/llmmap-internlm, sentencepiece
# 0.1.99) and the deviation is recorded in its status file and the manifest.
# TOKENIZER-only kwargs. These must not reach the model constructor:
# InternLM2ForCausalLM raises on an unexpected `use_fast`. llm.py grew a
# `tokenizer_load_kargs` parameter for exactly this (disclosed in R).
EXTRA_TOKENIZER_KWARGS = {
    "internlm/internlm2_5-7b-chat": dict(use_fast=False),
}

# togethercomputer/Llama-2-7B-32K-Instruct ships NO chat_template (verified in
# its tokenizer_config.json), so apply_chat_template raises and the model cannot
# be prompted at all. Transformers used to fall back to a built-in default
# template; that fallback was removed, which is why LLMmap's original code path
# worked and ours does not.
#
# This is a PROMPTING DECISION that affects what the model outputs and therefore
# its fingerprint, so it is declared here rather than buried: we use the format
# Together documents for this model, "[INST]\n{prompt}\n[/INST]\n\n".
# The template deliberately never mentions "system" -- llm.py's
# _does_template_have_system() greps the template string for that word, and if
# it matched, a system prompt would be handed to a template that silently drops
# it. As written, llm.py prepends the system prompt to the user turn instead,
# which is its normal no-system-role path. Flagged in TACC_NOTES for design side.
CHAT_TEMPLATE_FALLBACK = {
    "togethercomputer/Llama-2-7B-32K-Instruct":
        "{% for m in messages %}{% if m['role'] == 'user' %}"
        "[INST]\n{{ m['content'] }}\n[/INST]\n\n{% endif %}{% endfor %}",
}


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
    ap.add_argument("--limit-configs", type=int, default=None,
                    help="SMOKE TEST ONLY: generate at most N configs per pool. "
                         "Never use for the real corpus -- the shard will "
                         "correctly refuse to report COMPLETE.")
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
        load_kw = dict(torch_dtype=torch.bfloat16, device_map="cuda")
        if args.model in TRUST_REMOTE_CODE:
            load_kw["trust_remote_code"] = True
        tok_kw = EXTRA_TOKENIZER_KWARGS.get(args.model, {})
        import sentencepiece as _sp, transformers as _tf
        status["env"] = dict(sentencepiece=_sp.__version__,
                             transformers=_tf.__version__,
                             python=sys.version.split()[0])
        llm = LLM_huggingface(args.model, model_load_kargs=load_kw,
                              tokenizer_load_kargs=tok_kw or None)
        status["trust_remote_code"] = args.model in TRUST_REMOTE_CODE
        if args.model in CHAT_TEMPLATE_FALLBACK:
            assert getattr(llm.tokenizer, "chat_template", None) is None, \
                (f"{args.model} now ships a chat_template upstream -- remove it "
                 f"from CHAT_TEMPLATE_FALLBACK rather than overriding the "
                 f"model's own template")
            llm.tokenizer.chat_template = CHAT_TEMPLATE_FALLBACK[args.model]
            status["chat_template_source"] = "D006 fallback (model ships none)"
        else:
            status["chat_template_source"] = "model's own"
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
                if args.limit_configs:
                    todo = todo[:args.limit_configs]
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
