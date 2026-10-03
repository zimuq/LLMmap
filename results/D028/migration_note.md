# cdqd-corpus-v2 and cdqd-corpus-v2-ext-ml — schema and migration note (I7)

Written 2026-10-03 in S0, before S1 writes any generation.

## `cdqd-corpus-v2` (pool corpus; `data/corpus_v2/`, gitignored)

- **Universe:** the 85 models of `results/D027/universe_v2.json`. Frozen; no
  model is added or dropped.
- **Record = the `corpus_v1` record + four fields**, for every **generated**
  shard:
  - `gen_env`: `llmmap-gpu-v2`;
  - `template_kwargs`: `{"enable_thinking": false, "thinking": false,
    "strftime_now": "frozen:2026-09-06"}`;
  - `config_source`: `d027-frozen-draw` (new models) | `corpus_v1-replay`
    (the 3 regenerated gemma models);
  - `shard_source`: `v2-generated` | `v2-regenerated`.
- **Reused shards (34 v1 models) are NOT rewritten.** Rewriting would change
  their bytes, and "reused means unchanged".
  - `data/corpus_v2/{slug}.jsonl` is a **symlink** to
    `data/corpus_v1/{slug}.jsonl`. Its embeddings are symlinked the same
    way.
  - The four fields live in `results/D028/manifest.json` per model
    (`shard_source: corpus_v1-reuse`; `gen_env` = the D006 env; no template
    kwargs; `config_source: corpus_v1`).
  - The sha256 is asserted equal to `corpus_v1`'s status file.
- **Rows:** one JSONL line per (pool, config_index), pools build (75) / val
  (25) / test (25); `traces[i]` = pool query id i (0..258), as in v1.
- **Generation procedure** for every generated shard: D006's (per config,
  259 queries in chunks of the corpus batch 64, 200-token ceiling), with the
  per-model mode fixed in S0 (`results/D028/s0.json` → `mode`).
- **Consumers:**
  - join on (model, pool, config_index);
  - `config_index` j is a different config in each model (Issues 18/21);
  - read `shard_source` from the manifest before pooling v1-reused and
    v2-generated rows.

## `cdqd-corpus-v2-ext-ml` (multilingual extension; `data/corpus_v2_ext_ml/`, gitignored)

- Same record as above; `traces[i]` = query id `281 + i` (i = 0..15).
  Texts: `confs/queries/pool_d028_ml.json` (verbatim from D028 `## D`,
  sha256 recorded).
- All 85 models × all 125 of each model's own configs (v1: `corpus_v1`
  rows; new: D027 frozen draws).
- Generated in each model's D027 generation environment (DeciLM:
  `llmmap-gpu`; internlm2.5: `llmmap-internlm`; all others:
  `llmmap-gpu-v2`).
- **Never part of the pool tensor in D028.**

## Tensor

- `S_*_tok200_v2` (schema `cdqd-tensor-v2`): D007's statistics, split seed
  and pair worker, imported unchanged, over 85 models × 259 queries, S_build.
- Frozen with a sha256 in `results/D028/tensor_v2.json`.
