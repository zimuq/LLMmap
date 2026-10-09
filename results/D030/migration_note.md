# Migration note — `baseline_ext_v1` (D030, written in S0 before any generation)

**What it is.** A new extension dataset holding generations for the two external baselines
(MET, ZeroPrint) under their native protocols, for all 85 corpus-v2 models. It is **not** a
corpus-v2 schema change: corpus v1, corpus v2, `corpus_v2_ext_ml` and every v1/v2 tensor are
untouched and are not re-read for generation, except for their `prompt_conf` rows (copied, and
asserted equal at generation time).

**Schema id.** `baseline_ext_v1` (pilot rows: `baseline_ext_v1-pilot`, kept out of the dataset
in `results/D030/s0_pilot/`).

**Manifest.** `confs/baselines/baseline_ext_v1.json` (committed). It fixes, per model: the
generation env, mode and revision (D028 ML-extension precedent), the All-reference config
draws (MET 25 × 10, ZP 2 × 20), the S_test config types and the full 25-slot T_ZP order
(Review A1). Prompts: `confs/baselines/met_prompts.json` (MET draw `25/0.json`, sha
`7cc753df…`), `confs/baselines/zp_prompts.json` (official `prepare()`, seed 1000, sha
`80d9349c…`) and `zp_query_emb.npy`; sha256 of each is in the manifest.

**Data.** `data/baseline_ext_v1/{met,zp}/{slug}.jsonl` on `$WORK` (gitignored). One line per
(method, row, slot, config_ref) unit:

| field | meaning |
|---|---|
| `schema, method, model` | `baseline_ext_v1`, `met` / `zp`, HF id |
| `row` | `gen_ref`, `native_tgt`, `all_ref`, `test_tgt` |
| `slot` | S_test slot index (test rows), else null |
| `config_ref` | `native` / `build:<idx>` / `test:<idx>` |
| `prompt_conf` | the corpus-v2 row's dict (null for native) |
| `decoding` | effective generate kwargs (+ `max_new_tokens`) |
| `samples` | `[{p, r, text, n_tok, finish}]`; `finish` = `eos` / `length`; `n_tok` = generated tokens before the first EOS |
| `gen_env, revision, mode, batch, seed, max_input_tokens, wall_s, template_kwargs` | provenance |

**Consumers** must key units by `(row, config_ref)`, read `samples[].p` as the method's prompt
id (MET 0–24 in `met_prompts.json` order; ZP 0–9 in `all_queries_order`) and `samples[].r` as
the sample / repeat index. ZP's fingerprint needs all 10 prompts × 20 repeats of one unit.
