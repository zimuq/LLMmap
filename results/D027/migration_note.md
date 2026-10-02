# cdqd-corpus-v2-screen — schema and migration note (I7)

Written 2026-10-02, in S0, before S1 writes any generation.

- **Name / schema id:** `cdqd-corpus-v2-screen` (D027 stage A). Directory
  `data/corpus_v2_screen/` (gitignored). **`data/corpus_v1` is untouched.**
- **Record = the `corpus_v1` record plus three fields.** One JSONL line per
  (model, pool, config_index):
  - `dataset`, `llm`, `traces` (`[query_text, response]`, the 8 `paper8`
    queries = pool ids 0–7 in order), `prompt_conf`, `config_index`,
    `model`: unchanged meaning;
  - **`gen_env`** (new): `llmmap-gpu-v2` (transformers 5.18.0, torch
    2.7.1+cu128) or the D006 env a v1 model fell back to (`llmmap-gpu`,
    `llmmap-internlm`). R flags every N′ pair whose two models differ
    (`mixed_env`, Review A1);
  - **`template_kwargs`** (new): `{"enable_thinking": false, "thinking":
    false, "strftime_now": "frozen:2026-09-06"}`. `date_string` is **not**
    passed, because Llama-3.1-style templates default it to the fixed
    "26 Jul 2024" that `corpus_v1` rendered (S0 correction to P1 Call 3);
  - **`config_source`** (new): `corpus_v1-replay` (the 37 v1 models; the
    `prompt_conf` equals the `corpus_v1` row, asserted) or
    `d027-frozen-draw` (new models; `data/corpus_v2_screen/configs/{slug}.json`,
    sha256 in `results/D027/configs_manifest.json`).
- **Pools generated:** `build` (75) and `val` (25) only. The `test` configs of
  new models are drawn and frozen, **never generated** here (D028 uses them).
- **Embeddings:** `data/corpus_v2_screen/embeddings/{slug}.npy` +
  `.index.json`, made in the **unchanged** `llmmap-gpu` env with
  `d006_s6_embed.mean_pool` (I5, tok200, fp16, unnormalised), the same row
  format as `corpus_v1` (`pool`, `config`, `query_index`, `empty`).
- **Consumers:**
  - join on (model, pool, config_index), never on row order;
  - `config_index` j is a **different config in each model**, as in v1
    (TACC_NOTES Issue 18/21);
  - never mix `corpus_v1` traces for the 37 into the screen's B0′ (D's
    notes).
- **D028** would be a further version (`cdqd-corpus-v2`) with its own note.
