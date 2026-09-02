# DECISIONS.md — Open Decisions & Decision Log

> **Invariants moved.** What was Part 1 of this file (I1–I7) now lives in
> `CLAUDE.md`, since both agent sessions load that file every session and
> this one is not guaranteed to be read as often. Do not duplicate them
> here — if you need the reasoning behind an invariant, `CLAUDE.md` points
> to the relevant section of `METHOD.md`.

This file holds decisions that are **research-relevant but not yet
settled**, and the record of how each was eventually resolved. Anything
here is fair game for the local design-side agent to resolve on its own
*if* it's a routine call — but the items below are here specifically
*because* they need the human (see `CLAUDE.md`, design-side agent
escalation rules).

Status legend: ⬜ open · ✅ decided · 🅿️ deferred

---

## A. Blocking — gate Phase 1 (corpus construction)

| # | Question | Default if undecided | Status |
|---|---|---|---|
| **A1** | Which models form the universe? Target ~25 models ≤14B for local/TACC execution. **Must satisfy invariant I1.** Candidate near-relative groups: (Llama-3-8B-Instruct + a fine-tune), (Phi-3-mini-4k vs 128k), (Mistral-7B-Instruct v0.1/v0.2/v0.3), (gemma-2-9b-it vs gemma-1.1-7b-it). **D001's ≤14B subset findings feed this directly once available.** | 25 open-weight models ≤14B including the 4 groups above | ⬜ |
| **A2** | Include closed-source models (GPT/Claude)? Cost + compute-node network access implications. | Exclude through Phase 0–4; include only in final validation if budget allows | ⬜ |
| **A3** | Which two-sample statistic for the separability tensor's `Sep(·,·)`? **Changing this later invalidates every number computed so far.** Note: D001's AUC-of-Δ-vs-collapsed-centroid metric is scoped to D001 only and is *not* a decision on this — see D001's Review. | 5-fold CV AUC of a linear probe on the point-cloud embeddings (bounded, interpretable); MMD / energy distance as robustness checks on a subset | ⬜ |
| **A4** | Primary claim: (a) query efficiency at small k, or (b) worst-class accuracy? Determines what the paper's Figure 1 is. | (a) primary — more headroom, harder to dismiss; (b) secondary | ⬜ |

> **A4 note:** whichever claim is primary, the comparison baseline behind it is
> `GreedyCover` at `γ=1.0`, framed as *our reconstruction of mean-based
> selection* — not as "reproducing LLMmap's algorithm." See `METHOD.md §6.5`
> before writing Figure 1's caption.

> **A1 open sub-question (added 2026-09-02, from D002 §R5):** a 70B-class
> model does not fit on a single GH200 (needs ~140GB at bf16 against 95GB
> visible) — this is a memory problem, not a budget one (D002 confirmed
> budget is not binding at any candidate size). Relevant because D001 §R4
> recommended adding Llama-3-70B + Smaug-70B specifically to retain the
> project's own motivating example. Options, none free: **quantize**
> (fits, but perturbs the output distributions this project fingerprints —
> risks confounding the signal under study), **CPU/unified-memory offload**
> (works, much slower, unmeasured), **multi-node sharding** (more
> engineering, unmeasured), or **drop the 70B pair** (I1 is satisfied
> without it — 65 near-relative pairs across 37 ≤14B models — the loss is
> the headline example, not statistical validity). **Awaiting human call.**

## B. Preliminary experiments

| # | Question | Status |
|---|---|---|
| **B1** | Pool-saturation pre-experiment before committing to a full corpus build. | ✅ Became D001 — in progress |
| **B2** | Decompose intra-model noise into config-variation vs sampling-stochasticity? Optional, cheap, not on the critical path. | ⬜ |

## C. Hyperparameters (log when set, don't need to decide early)

| # | Parameter | Proposed | Status |
|---|---|---|---|
| C1 | CVaR tail `γ` | 0.1 default; ablate {1.0, 0.25, 0.1, 0.05} | ⬜ |
| C2 | Query budget `k` | 8 (comparable to the paper); always report the full k=1..8 curve | ⬜ |
| C3 | Initial pool size `\|Q_0\|` | ~250 (raised from an original ~100 baseline, 2026-09-02, design-side) — the paper's 8 + expansions of its 4 query families + published baselines + tokenizer/glitch probes, expanded further once D002 confirmed generation cost is not the binding constraint at 2–3x this scale. See [D003](D003.md)'s amendment + Review addendum. | ✅ |
| C4 | Split sizes | 75 / 25 / 25 build/val/test, disjoint at the parameter level per I2 | ⬜ |
| C5 | Outer-loop params `T, N, n_keep, θ, ε` | T=3–5, N=40, n_keep=3, θ from tensor quantile, ε=0.005 | ⬜ |
| C6 | Generator LLM | `allenai/OLMo-2-1124-13B-Instruct` — set for D003's query-pool generation (2026-09-02, design-side, strict decoupling from the universe over TACC's Qwen3-14B default; see D003 `## Review`). Re-evaluate if Phase 3's targeted-generation step (`METHOD.md §5.4` step D) needs a different tradeoff. | ✅ (for D003; Phase 3 use TBD) |
| C7 | Response truncation | 650 chars, matching the released `confs/default.json`. **D002 §R4 finding (2026-09-02): currently inert** — measured shipped response lengths top out at 667 chars (p99=584, mean=352), and the real cap is `max_new_tokens=100` in `llm.py:9`, never 650 chars. Raising to 200 tokens is affordable (D002 §R3, ~1.8x cost). Decide C7 on information grounds, not cost — still open. | ⬜ |

## D. Deferred

| # | Item | Status |
|---|---|---|
| D1 | Research direction (theory- vs systems-leaning) | 🅿️ resolve after Phase 1 |
| D2 | Target venue | 🅿️ same |
| D3 | Papers 2 & 3 follow-ups | 🅿️ out of scope; keep corpus schema friendly where free |
| D4 | CVaR/robust-submodular literature citations (hardness + approximation results referenced in `METHOD.md` §6.1) — **unverified, must be checked before any writeup** | ⬜ |

---

## Decision log

```
[YYYY-MM-DD] B1 — pool saturation pre-experiment
  Decision: promoted to D001.
  Rationale: near-zero cost, directly gates the CDQD premise (invariant I1).
  Decided by: human + design-side agreement, prior session.

[2026-09-01] D001 §R6 — remeasure the hard-tail premise with a fair instrument
  Decision: promoted to D004. D001's "no exploitable tail" verdict was
  produced by an instrument that violates I3 (compares traces against a
  collapsed centroid) and saturates (88.2% of pairs at an exact ceiling) --
  R6 recommended re-measuring with an I3-compliant, non-saturating statistic
  before treating that verdict as settled, since it bears directly on
  METHOD.md §2's premise and invariant I1.
  Rationale: this is a METHOD.md/invariant-touching call per CLAUDE.md's
  escalation rules, not a routine one -- surfaced to the human rather than
  decided on the design side.
  Decided by: human, 2026-09-01 (approved drafting D004 immediately).

[2026-09-02] D002 R -- closed; 70B memory constraint spun into A1
  Decision: D002 marked CLOSED (GPU_AVAILABLE, high confidence, own
  question fully answered). The unexpected finding that a 70B-class model
  does not fit on one GH200 was NOT treated as reopening D002 -- it is a
  new input to the still-open A1 decision, not a gap in D002's own answer.
  Rationale: D002 asked "is GPU available and what does Phase 1 cost" and
  answered both cleanly; "should we include a 70B pair given it needs
  quantization/offload/sharding" is a methodological A1 question, not an
  infrastructure one.
  Decided by: design-side, routine call (not escalated -- this is a filing
  decision, not the substance of the memory-constraint question itself,
  which IS escalated, see A1 above).

[2026-09-02] D003 P1 -- generator overruled to OLMo-2-13B-Instruct
  Decision: APPROVED WITH AMENDMENTS. TACC recommended Qwen/Qwen3-14B
  (shares vendor lineage with Qwen2/2.5 universe members) with
  allenai/OLMo-2-1124-13B-Instruct offered as a strictly-decoupled
  alternative. Took the alternative.
  Rationale: TACC's own case for Qwen3-14B (anchors + structured prompting
  constrain output more than generator identity) undercuts the reason to
  accept the self-preference risk in the first place -- if generator
  identity matters that little, there's no cost to picking the fully
  decoupled option. C6 exists specifically to guard against this.
  Decided by: design-side, routine call under C6 (Part C -- log when set).

[2026-09-02] D004 P1 -- approved
  Decision: APPROVED. Added one requirement for D004's eventual R: state
  explicitly that f_test (D001's cached embeddings) is the output of
  LLMmap's own contrastively-trained classifier, not the project's frozen
  I5 embedding (multilingual-e5-large-instruct) the real tensor will use --
  so a null result narrows the hard-tail question rather than closing it,
  while a positive result is comparatively strong evidence (the
  representation bias runs toward oversaturation, not away from it).
  Decided by: design-side, routine call (interpretation-limit requirement,
  not a trade-off -- mirrors D001 R5's existing practice).

[2026-09-02] D003 / C3 -- pool target raised from ~100 to ~250
  Decision: raised the Q_0 candidate-pool target from the paper-inherited
  ~100 baseline to ~250. D002 R3/R4 established generation cost is not
  binding at 2-3x this scale (25x250 batched ~= 1.5% of SU balance, ~4
  days, still inside the $SCRATCH purge window) -- so the asymmetric risk
  (a too-narrow pool produces a confound in the mean-vs-CVaR comparison,
  per D003's own stated risk, not merely a null result) favors erring
  generous. P1's already-approved methodological calls (no pre-filter,
  OLMo-2 generator, anchor-recovery approach) are unaffected; only the
  numeric target changes -- no new P/re-review required, per D003's Review
  addendum.
  Decided by: design-side, routine call under Part C ("log when set");
  direction confirmed with the human in chat before being made
  (2026-09-02).
```

## Escalated: two silent I2 violations found in the current generator (2026-09-02)

**Not yet a decision — recording the escalation itself; awaiting the human's
priority call before D005 (drafted, see below) is handed to TACC.**

TACC found, while preparing D004 (full detail: TACC's note appended to
`D001.md` `## Review`, 2026-09-02):

1. `PromptConfFactory.sample()` (`LLMmap/prompt_configuration.py:234-240`)
   accepts a `pool` argument but never forwards it to `sample_one()` — so
   **both the train and test splits currently draw system/CoT/RAG prompts
   from the TRAIN pool.** The split definition itself (`train_test_split.json`)
   is genuinely disjoint; only the one-line forwarding is broken.
2. **Sampling hyperparameters (`temperature`, `do_sample`, …) have no
   train/test split at all** in `confs/prompt_configurations/general.json` —
   the paper's own §7.1 splits `H` into `Htrain`/`Htest`; the released code
   never implemented that split.

Both are **invariant I2** violations (per `CLAUDE.md`'s escalation rule
(a)) that run silently — the code executes and produces numbers with no
error. Whether the shipped `default_dataset.jsonl` (that D001/D004 both
depend on) carries this defect is unverifiable from the artifact itself. If
it does, it inflates separability in both D001 and D004 — a candidate
fourth cause of D001's saturation, alongside the three D001 §R2 already
named.

**Promoted to [D005](D005.md).** Item 1 is a trivial, low-risk one-line
fix; item 2 needs an actual design decision (how to partition
`sampling_universe` into disjoint train/test pools) — TACC proposes a
scheme in D005's P for design-side review, per the code-change rule.

<!-- append new entries below -->
