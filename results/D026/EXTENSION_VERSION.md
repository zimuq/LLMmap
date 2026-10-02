# corpus_v1_ext_d026 — version and migration note (I7)

Written 2026-10-01, at P1, before S1 writes any generation.

- **Name / version:** `corpus_v1_ext_d026`, extension version `v1`.
- **Parent:** `data/corpus_v1` (37/37 READY, D006). **Not modified.**
- **Schema: identical to `corpus_v1`.** One JSONL line per (model, pool,
  config_index) with keys `dataset`, `llm`, `traces` (list of
  `[query_text, response]`), `prompt_conf` (`PromptConf.to_dict()`),
  `config_index`, `model`. No field is renamed, added or dropped, so this is
  not a schema change, and no migration of `corpus_v1` is needed.
- **What differs is the query set.** `traces[i]` is new query id `259 + i`
  (i = 0..21; 279/280 = the 1k control added by the P1 Review, 2026-10-02). The id→text map is in `confs/queries/pool_d026_ext.json`
  (written in S1, sha256 recorded in `results/D026/extension_manifest.json`).
  Pool query ids 0..258 keep their meaning, and no new id collides with one.
- **Models:** only the two Phi-3-medium twins (128k, 4k). Every (pool,
  config_index) uses the **same `prompt_conf` dict** as that model's own
  `corpus_v1` row, asserted equal at write time.
- **Embeddings:** `data/corpus_v1_ext_d026/embeddings/{slug}.npy` + `.index.json`,
  I5 model, mean-pooled, **not normalised**, fp16, `max_length` 512, the same
  as `d006_s6_embed.py` (tok200 = full responses).
- **Consumers must join on (model, pool, config_index)**, never on row order,
  and must read query ids from the manifest, never from `traces` position alone.
- **Stage 2 (if any)** would add the other 35 models under a new version
  (`v2`), decided by design-side and the human. This file does not license it.
