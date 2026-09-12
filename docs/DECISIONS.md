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
| **A1** | Which models form the universe? | **All 37 open-weight models with `params_b ≤ 14` in the 52-model universe** (`results/D001/model_metadata.csv`) — not a curated subset. Includes all 4 originally-named near-relative groups (Llama-3-8B family, Phi-3-mini-4k/128k, Mistral-7B v0.1/v0.2/v0.3, gemma-2-9b/gemma-1.1-7b) automatically. **Decided 2026-09-05.** | ✅ |
| **A2** | Include closed-source models (GPT/Claude)? Cost + compute-node network access implications. | Exclude through Phase 0–4; include only in final validation if budget allows | ⬜ |
| **A3** | Which two-sample statistic for the separability tensor's `Sep(·,·)`? **Changing this later invalidates every number computed so far.** Note: D001's AUC-of-Δ-vs-collapsed-centroid metric is scoped to D001 only and is *not* a decision on this — see D001's Review. | **Decided 2026-09-06.** 5-fold CV AUC of a linear probe (bounded, interpretable) as primary, matching the original default — but energy distance is now a **mandatory companion statistic on the full tensor** (not an optional subset check), and D004's resolution-audit (fraction of pairs at/near ceiling) is a **formal gate**, not a footnote: if the bounded statistic saturates on the real tensor the way it did on LLmap's own trained representation in D004, energy distance is authoritative for interpretation. Evidence: D006's C7 pilot ran probe AUC on the real, frozen I5 embedding (not D004's proxy) and found 0% of 15 pairs at ceiling — but that's a 15-pair pilot against a 666-pair reality, and D001 was burned once already by skipping this exact audit. See [D007](D007.md). | ✅ |
| **A4** | Primary claim: (a) query efficiency at small k, or (b) worst-class accuracy? Determines what the paper's Figure 1 is. | (a) primary — more headroom, harder to dismiss; (b) secondary | ✅ (a) query efficiency, decided 2026-09-08 by the human; **confirmed under the real trained classifier by D009, 2026-09-09** |

> **A4, real-classifier update (D009, 2026-09-09):** query efficiency
> holds against mean-greedy and random under the actual trained
> pipeline (not just the D004–D008 proxy), CONFIRMED at every `k`.
> Against the paper's own 8 queries specifically, it is a **tie**, not a
> win (+0.007 mean top-1 at `k=8`, inside the seed range) — but the tie
> was reached at a fraction of the paper's own selection cost (Algorithm
> H.1 trains a real classifier 372 times to produce those 8 queries;
> CVaR never touches one during selection). Any writeup's Figure 1
> caption should say *"ties the paper's hand-selected strategy at
> near-zero selection cost,"* not merely *"ties the paper's strategy"* —
> the two claims are equally true and very differently persuasive.

> **A4 resolved 2026-09-08 — (a) query efficiency, per D008's evidence:**
> CVaR-coverage reaches 70% mean accuracy at `k=2` against mean-greedy's
> 4, the paper's own 8 queries' 5, and random's 12 — a clean, large,
> statistically robust win (every CI at every `k` excludes zero on mean
> top-1). Figure 1 is the accuracy-vs-`k` curve; worst-class accuracy is
> reported as a secondary metric, honestly, alongside the gap below.
>
> **Worst-class gap mechanism, verified 2026-09-08 (no new TACC run —
> read directly from `results/D008/selection.json` and
> `confs/queries/pool_v1.json`, both already on disk):** at equal `k`=8
> the paper's own 8 queries beat every algorithmic condition on
> worst-class (0.44 vs. CVaR's 0.28) because 5 of the 8 are direct
> self-identification probes ("What LLM are you exactly?", "What's your
> data cutoff date?", the injection-wrapped "who created you", etc. —
> `confs/queries/default.json` indices 0,2,4,5,6). **This is not a
> selection-algorithm weakness we failed to find a workaround for — it's
> a direct, checkable fact about what the objective rewards.** Our own
> 259-query pool contains all 5 of those exact queries (as literal
> anchors, `pool_v1.json` `anchor_set: "paper8"`) *plus* 40 more
> purpose-generated `generated-banner-grabbing` queries — 45 total
> self-identification-style candidates, freely available. **None of the
> 45 were ever selected** in the top-16 chain of `mean_greedy_max`
> (γ=1.0, our reconstruction of the paper's own claimed objective
> family), `mean_greedy_sum`, or `cvar_max` (γ=0.1, our actual method).
> Only 2 of the paper's 8 (index 1, "build a bomb"; index 3, climate
> true/false — both non-identity behavioral probes) ever get picked, and
> late in the chain. What *does* dominate every chain instead:
> `generated-malformed-alignment` (harmful-request refusal probes — same
> family as the paper's own index 1) and the tokenizer probes.
>
> **Likely mechanism (theory-grounded, not yet independently confirmed
> at the tensor level):** `METHOD.md §4`'s `Sep(q,v,v')` = between-group
> separation (numerator) / within-group spread (denominator). A
> self-ID query plausibly has large *raw* between-model separation (models
> report different names) but poor *within*-model consistency — whether a
> model states its identity is sensitive to system prompt / RAG context /
> sampling config (I2's 25-way split), so the same model answers
> inconsistently across configs. A refusal-behavior probe's yes/no
> response is a more config-stable property. If true, this is a case
> where CDQD's objective is *working as designed* (penalizing
> config-fragile signal) and the mismatch is with the worst-class
> *metric*, which rewards a single lucky config regardless of
> reliability elsewhere — not a bug in `GreedyCover`. Confirming this
> would need TACC to read within-model variance for the 45
> self-ID-style queries off the already-frozen `S_energy_sf_tok200`
> tensor — cheap (post-processing), not yet requested as a D.
>
> **Net effect on the earlier framing:** "our method loses to the
> paper's 8 on worst-case" is accurate as stated, but should not be read
> as "our selection algorithm underperforms theirs" — the paper's own
> claimed algorithm family (mean-greedy) *also* never selects these
> queries when given the chance, which is direct evidence the paper's
> shipped defaults were not purely the output of the greedy search
> Algorithm H.1 describes (whose code and 50-query pool were never
> released — `METHOD.md §1` Fact 1, `PAPER_DEVIATIONS.md` item 4/5).
> They read as hand-augmented with identity probes by the authors, on
> top of whatever the algorithm actually produced. This can't be proven
> as a claim about the paper's own process (no such counterfactual can
> be run without the paper's original code/pool), but it is now a
> directly reproducible fact about *our* pool and *our* objective family.
| **A5** | **New, 2026-09-02, from D005/P1 §F1.** `sampling_universe`'s `do_sample` is a 2-value parameter — I2's literal wording ("no single sampling setting crosses splits") is structurally unsatisfiable for it: any split puts all-greedy decoding in one pool and all-stochastic in the other, which *is* a confound, not a fix. TACC proposes three options (Call B): (1) a documented, explicit carve-out for `do_sample` alone — **TACC recommends this**; (2) reinterpret I2 at the composite-tuple level rather than per-field; (3) implement the paper's `frequency_penalty` dimension so `do_sample` stops being the only lever (bigger change, needs its own D + I7 schema bump). Sets precedent for how "literally unsatisfiable invariant" cases get handled, not just this field. | Option 1 (documented carve-out) — smallest change; I2's actual failure mode is *silent* leakage, and an explicit, disclosed exception isn't that | ✅ Option 1, decided 2026-09-02 by the human |
| **A6** | **New, 2026-09-11, from D010 R9.** Three D's running (D008, D009, D010) have all hit the same wall: 25 `S_test` configs give too little resolution to resolve a `k`-by-`k` delta — only the aggregate/consistency pattern across `k`, never a single clean number. Fixing it means **more test configs, i.e. new generation** — touches I6/I7 (corpus is currently frozen; a size change is a schema-relevant event), real cost, real turnaround. Not urgent — every closed verdict so far (D008/D009/D010) was reached on the aggregate pattern, not blocked by this. | Leave `S_test` frozen at 25 configs until a specific number needs a tighter CI than "consistent across k" (e.g. for a paper figure) — expand only then, sized to that need | ⬜ |

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
>
> **Update 2026-09-02 (D004 §R3):** under D004's fair, non-saturating
> instrument, this exact pair — Llama-3-70B ↔ Smaug-70B — is the **5th
> hardest of 1326 pairs** (percentile 0.30), harder than D001's original
> 0.75 estimate. Doesn't change the tradeoff's shape, but raises what's
> given up under "drop the 70B pair": not a marginal example, a
> near-extreme one.
>
> **Resolved 2026-09-02 — deferred, default applies (≤14B core, no 70B
> pair for now).** Checked the full ranked pair list
> (`results/D004/pairwise_m1.csv`) rather than relying on the single 70B
> data point: the **single hardest pair in the entire 52-model universe is
> `Falcon3-10B ↔ Falcon3-7B` (rank 1, both ≤14B)**, not the 70B pair. Of
> the top 15 hardest pairs, only 2 involve a 70B model; the rest are ≤14B
> or ≤35B (e.g. `Phi-3-mini-128k↔4k` at rank 3, `CohereForAI/aya-23-35B↔8B`
> at rank 11 — 35B fits on one GH200 without the memory problem at all).
> The method's demonstration value is not concentrated in the 70B pair —
> quantize/offload/multi-node engineering work is not warranted right now.
> Revisit if a later draft specifically wants the paper's own headline
> example rather than an equally- or more-extreme substitute.

## B. Preliminary experiments

| # | Question | Status |
|---|---|---|
| **B1** | Pool-saturation pre-experiment before committing to a full corpus build. | ✅ Became D001 — **closed 2026-09-02** |
| **B2** | Decompose intra-model noise into config-variation vs sampling-stochasticity? Optional, cheap, not on the critical path. | ⬜ |

## C. Hyperparameters (log when set, don't need to decide early)

| # | Parameter | Proposed | Status |
|---|---|---|---|
| C1 | CVaR tail `γ` | 0.1 default; ablate {1.0, 0.25, 0.1, 0.05} | ⬜ |
| C2 | Query budget `k` | 8 (comparable to the paper); always report the full k=1..8 curve | ⬜ |
| C3 | Initial pool size `\|Q_0\|` | ~250 (raised from an original ~100 baseline, 2026-09-02, design-side) — the paper's 8 + expansions of its 4 query families + published baselines + tokenizer/glitch probes, expanded further once D002 confirmed generation cost is not the binding constraint at 2–3x this scale. See [D003](D003.md)'s amendment + Review addendum. | ✅ |
| C4 | Split sizes | 75 / 25 / 25 build/val/test, disjoint at the parameter level per I2 (D005's fix; A5 carve-out for `do_sample`). **Decided 2026-09-05 (design-side, routine — matches TODO.md's own plan and the uncontested default; no objection raised).** | ✅ |
| C5 | Outer-loop params `T, N, n_keep, θ, ε` | T=3–5, N=40, n_keep=3, θ from tensor quantile, ε=0.005 | ⬜ |
| C6 | Generator LLM | `allenai/OLMo-2-1124-13B-Instruct` — set for D003's query-pool generation (2026-09-02, design-side, strict decoupling from the universe over TACC's Qwen3-14B default; see D003 `## Review`). Re-evaluate if Phase 3's targeted-generation step (`METHOD.md §5.4` step D) needs a different tradeoff. | ✅ (for D003; Phase 3 use TBD) |
| C7 | Response truncation | **Decided 2026-09-06: 200-token generation ceiling** (up from the released code's inert 650-char / actual 100-token default, `llm.py:9`). Deliberately beyond the paper/released code's own setting — an extension, not a reproduction; justified because generation is causal (200→100 truncation is free and reversible; the reverse isn't) and the 100-token default was measured to censor ~38 points of the corpus (D006/P1 §F1). TACC runs the small pilot (100/200/400 via truncation, ~6 models) as a confirmatory check on this choice, not as what the choice depends on. See `docs/PAPER_DEVIATIONS.md`. | ✅ |

## D. Deferred

| # | Item | Status |
|---|---|---|
| D1 | Research direction (theory- vs systems-leaning) | 🅿️ resolve after Phase 1 |
| D2 | Target venue | 🅿️ same |
| D3 | Papers 2 & 3 follow-ups | 🅿️ out of scope; keep corpus schema friendly where free |
| D4 | CVaR/robust-submodular literature citations (hardness + approximation results referenced in `METHOD.md` §6.1) — **unverified, must be checked before any writeup** | ⬜ |
| D5 | **Writeup framing, 2026-09-10 (human decision — how to present, not what to run).** Report three separate claims, not one "fair comparison": **(1)** selection algorithm — CVaR vs. mean-greedy/random, classifier-free, apples-to-apples, CONFIRMED (D009). **(2)** cost efficiency — ties the paper's 8 at near-zero selection cost vs. their 372 real training runs (D009 + `Appendix H`). **(3)** theoretical formalization, **elevated to co-primary, not a fallback** — first to instantiate the paper's own stated-but-never-implemented Eq.2/Eq.3 via the MI decomposition + point-cloud statistic (`METHOD.md §3–4`), sharpened by `Appendix H` showing Algorithm H.1 doesn't approximate Eq.2/3 either — it optimizes raw accuracy directly. Do not let (2)'s cost-asymmetric comparison bear a "which method is better" claim it structurally can't support. | 🅿️ noted for writeup |

> **D4 scope note, 2026-09-08:** the separability statistic itself
> (energy distance) was a related but separate unverified-citation risk
> — now resolved. Confirmed it is a pre-existing statistic (Székely &
> Rizzo 2004; Rizzo & Székely 2016 review), not the paper's formula (the
> paper never instantiates one, Fact 1) and not this project's
> invention — applying it to the I3 point-cloud structure is CDQD's own
> combination, though, and belongs in `PAPER_DEVIATIONS.md` as a
> deviation/novelty item, not a gap. Full writeup: `PAPER_DEVIATIONS.md`
> item 10. D4 itself still covers only the CVaR/submodularity hardness
> citations, which remain unverified.
>
> **Open, human-escalation item raised by this check (not yet
> actioned):** `METHOD.md §4` currently describes `Sep(·,·)` only
> abstractly (between-group/within-group ratio) and never names energy
> distance or cites it — the concrete formula only lives in `D004.md`'s
> TACC-authored `## P` and this file's A3 note. Naming it explicitly in
> `METHOD.md` would make the method document self-contained, but editing
> `METHOD.md` is a human-escalation item per `CLAUDE.md` rule (d) —
> flagged here, not made.

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

[2026-09-02] D004 R -- TAIL CONFIRMED; D001 closed
  Decision: D004 marked CLOSED (TAIL CONFIRMED, high confidence). D001's
  negative "no exploitable tail" verdict is attributed to its instrument
  (centroid collapse, 2-way framing, resolution ceiling) -- under an
  I3-compliant, non-saturating statistic (energy distance), CVaR_0.1/mean =
  0.616 (CI95 [0.582,0.669]), decisively below 1. METHOD.md section 2 and
  invariant I1 stand as written. D001 marked CLOSED in the same pass -- its
  own R6 open question is what D004 was commissioned to answer.
  Rationale: D004's own "What counts as an answer" table specifies this
  disposition (TAIL CONFIRMED -> D001's negative attributed to instrument,
  METHOD/I1 stand) -- applying it is a routine design-side call once R
  landed, not a new judgment. D004 does not itself decide what to DO about
  A1's 70B question -- that stays open, now with D004's percentile finding
  attached (see A1 update above).
  Decided by: design-side, routine call (disposition was pre-registered in
  D004's own D-file; no new interpretation required).

[2026-09-02] D003 R -- closed; READY, 233-entry pool; R3(b) judgment call endorsed
  Decision: D003 marked CLOSED (READY). Endorsed TACC's judgment call to
  re-run the prompt-injection family after finding 55% of first-pass
  entries were provably dead queries (identical answers across every
  model), even though the family had already met its numeric count target
  -- D003's own stated purpose (avoid a low-diversity-pool confound in the
  eventual mean-vs-CVaR comparison) is better served by a live re-run than
  by a technically-compliant but partly-dead pool, and the defect was
  TACC's own harness bug, not a design flaw, so fixing and re-running was
  the right default over shipping and flagging.
  Decided by: design-side, routine call (D003's own criteria already
  anticipated this class of judgment call; endorsing rather than
  overriding).

[2026-09-02] D005 P1 -- Call A approved, Call B escalated (see A5)
  Decision: approved TACC's proposed interleaved split for `temperature`
  (train/test each span the full range, rather than a contiguous low/high
  split that would confound temperature with split membership). Call B
  (how to handle `do_sample`, a 2-value parameter I2's literal wording
  cannot apply to) escalated to the human as new item A5 -- this is a
  genuine invariant-interpretation question with no obviously-correct
  answer and sets precedent beyond this one field, squarely
  CLAUDE.md escalation category (a).
  Decided by: design-side (Call A, routine); escalated to human (Call B,
  see A5).

[2026-09-02] A5 -- do_sample carve-out, option 1
  Decision: adopt TACC's option 1 -- do_sample stays shared across
  train/test as a documented, explicit exception to I2, rather than a
  literal per-value split (which would confound decoding mode with split
  membership) or a bigger reinterpretation/frequency_penalty change.
  Rationale: smallest change; I2's actual failure mode is silent leakage,
  and a disclosed, deliberate exception for a structurally-unsplittable
  binary parameter isn't that. TACC clear to implement M2/S2 now.
  Decided by: human, 2026-09-02.

[2026-09-02] A1 70B sub-question -- deferred, default applies
  Decision: do not pursue quantize/offload/multi-node for the 70B pair
  right now; ~25x<=14B core stands as the universe (default already on
  file for A1). Supported by data, not just the default: the single
  hardest pair in the whole 52-model universe (D004's fair instrument) is
  Falcon3-10B<->Falcon3-7B, both <=14B; only 2 of the top 15 hardest pairs
  involve a 70B model. The method's demonstration value is not
  concentrated in the paper's own headline pair.
  Decided by: human, 2026-09-02.

[2026-09-04] D005 R -- FIXED; D005 closed
  Decision: D005 marked CLOSED. M1+M2+M3 verified via an actual regression
  test (bug reinjected -> 250 named failures; fix restored -> 0); M4
  (shipped-dataset exposure) reported UNDETERMINABLE with contradictory
  evidence across all three hypotheses tested, correctly not forced into
  a guess either way. No effect on D001/D004's existing numbers -- neither
  called the buggy code path, and any hypothetical leakage would be
  inflationary, which works against D004's confirmed tail, not for it. No
  re-run warranted. Phase-1 corpus construction's remaining blockers are
  now down to A1 alone (C3, D003's pool, and D005's fix are all in place).
  Decided by: design-side, routine call (disposition matches D005's own
  pre-registered "What counts as an answer" table; no new interpretation
  required).

[2026-09-05] A1 -- all 37 <=14B models, not a curated 25
  Decision: universe = every open-weight model with params_b <= 14 in the
  52-model universe (verified count: exactly 37, via
  results/D001/model_metadata.csv). Supersedes the original "~25 curated"
  default.
  Rationale: cost scales linearly with |L| (37 vs 25 is +48% generations,
  still only ~1.8-3.1% of the 6982 SU balance) but wall-clock does not --
  under Vista's 20-concurrent-job cap, sharding by model needs ceil(25/20)
  = 2 waves and ceil(37/20) = 2 waves, identical. "All models satisfying
  the inclusion criterion" is also a cleaner methodology statement than a
  hand-curated subset, and all 4 originally-named near-relative groups
  fall out of the full set automatically.
  Decided by: human, 2026-09-05.

[2026-09-05] C4 -- split sizes 75/25/25, decided
  Decision: adopt the long-standing default (75 build / 25 val / 25 test),
  now enforceable given D005's I2 fix (disjoint at the individual-parameter
  level, with the A5 do_sample carve-out).
  Rationale: matches TODO.md's own plan; no objection raised; needed to
  unblock drafting the Phase-1 D (D006).
  Decided by: design-side, routine call (Part C, "log when set").

[2026-09-05] D006 P1 -- mostly approved (F2-F6, Call 1, Call 2); F1/C7 held for human
  Decision: approved TACC's three-way split scheme (systems/temperature
  3-way, cot/rag 2-way with val/test sharing -- avoids recreating the A5
  confound on a thin value set), the (raw, sampling_hparams) dedup key for
  a newly-found sample() uniqueness bug, the corrected 20h/350-node-hour
  wall-clock guardrail (TACC caught its own earlier utilisation-vs-makespan
  error), model-revision pinning, all 26 proposed tokenizer probes, and the
  reassembly/manifest/all-37-or-nothing design. Held open: F1/C7 (response
  generation ceiling) -- TACC recommends 200 tokens (causal generation
  means truncation is free/reversible below the ceiling, so only the
  ceiling itself is a one-way door) plus a small pilot; design-side agrees
  but this is a Part-C hyperparameter with real, mostly one-directional
  stakes -- reserved for the human per the escalation rules, not decided
  here. TACC clear to run S1+S2 now; S3-S8 wait on F1.
  Decided by: design-side (F2-F6, Call 1, Call 2, routine -- TACC's own
  P1 summary table already triaged these as "design side"); F1 escalated
  to human (see C7 above).

[2026-09-06] C7 -- 200-token generation ceiling, decided; D006 fully unblocked
  Decision: 200 tokens, per TACC's recommendation. Explicitly an extension
  beyond the paper/released code's own setting, not an attempt to
  reproduce it -- accepted as a deliberate, disclosed deviation. TACC runs
  the F1 pilot as confirmation, not as a precondition. D006 S3-S8 now
  fully clear to proceed.
  Decided by: human, 2026-09-06.

[2026-09-06] A3 -- probe AUC primary + mandatory energy-distance companion + gate
  Decision: keep the original default (5-fold CV AUC of a linear probe,
  bounded/interpretable) as the primary Sep(.,.) statistic, but energy
  distance is now a mandatory companion computed on the full tensor (not
  an optional check on a subset), and D004's resolution-audit (fraction
  of pairs at/near ceiling) is a formal gate rather than a footnote.
  Rationale: D006's C7 pilot found probe AUC does not saturate on the
  real, frozen I5 embedding (0% of 15 pairs at ceiling) -- unlike D004's
  proxy instrument (LLmap's own trained representation, which saturated
  95.9%/93.0%) -- so the bounded default is empirically supported here.
  But 15 pairs is not 666, and D001 was burned once already by trusting a
  bounded statistic without this exact audit -- so the gate stays
  mandatory rather than being dropped because the pilot looked clean.
  Unblocks D007 (build the real separability tensor + mandatory hard-tail
  check, TODO.md T1.5/T1.6).
  Decided by: human, 2026-09-06 (design-side recommendation accepted
  as proposed).

[2026-09-06] D006 R -- CLOSED, READY at 37/37
  Decision: D006 marked CLOSED. All 37 models generated, embedded, and
  invariant-verified against the corpus as generated. A late own-bug find
  (L2-normalizing embeddings, which the paper's released code does not
  do) was fixed and the corpus re-embedded; measured +0.051 AUC, 15/15
  pairs improved. This makes D006/R2's earlier "100 tokens separates
  better than 200" finding provisional (measured on the now-corrected-
  away normalization) -- not resolved here, handed to D007 as its F1.
  Decided by: design-side, routine call (disposition matches D006's own
  pre-registered READY criterion; no new interpretation required).

[2026-09-06] D007 P1 -- approved (F1-F4, S2a)
  Decision: approved re-checking the 100-vs-200 analysis-length question
  on corrected embeddings over the full 666 pairs before freezing which
  length the tensor uses (F1, discharges D006/R10's provisional flag);
  approved a split-half winner's-curse correction applied to T1.6's
  falsification criterion, since max_q S_probe[q][p] over 259 noisy
  75-vs-75 estimates is upward-biased in exactly the direction that would
  falsely trigger a TAIL STILL ABSENT stop (F2 -- threshold unchanged,
  only the quantity it's applied to); approved F3 (scale-free energy
  distance for any MAX-aggregated use) and F4 (store per-cell sample
  counts despite current constancy). Approved S2a (a small saturation
  check at the real n=75 scale) to run immediately and standalone.
  Rationale: none of these are value trade-offs -- they are measurement-
  correctness fixes matching methodology already established and human-
  approved (D004's bounded+unbounded+audit-gate pattern; C7's causal
  generate-high/analyze-low asymmetry). No genuine trade-off to escalate.
  Decided by: design-side, routine call.

[2026-09-07] D007 R -- TAIL CONFIRMED on S_energy; D007 closed
  Decision: D007 marked CLOSED. S_probe saturated far more severely than
  anticipated (51.8% of pairs at ceiling even after F2's correction,
  98.2% before it) -- A3's mandatory-gate design correctly deferred to
  S_energy, which resolved cleanly: CVaR_0.1/mean = 0.353, larger tail
  than D004 found on its proxy instrument (0.616). METHOD.md sec 2 and
  I1 stand on the real corpus. Frozen tensor for I6 going forward:
  S_energy_tok200. Also: F2's winner's-curse correction did not prevent
  a false verdict on its own (T1.6 fired on S_probe under both naive and
  corrected numbers) -- A3's audit gate is what actually did; both
  safeguards were worth approving, only one was load-bearing here, and
  this is reported plainly rather than overstating F2's contribution.
  D006/R10's provisional "100 tokens beats 200" finding is refuted (not
  just superseded) on proper remeasurement -- 200 wins on 71.3% of pairs
  under the authoritative statistic. D007.md's own outcome table was
  missing a branch for exactly this result (S_probe censored, S_energy
  resolves) -- fixed in this pass rather than left as a prose-only
  exception. Phase 2 (TODO.md T2.1-T2.4) is now unblocked.
  Decided by: design-side, routine call (disposition follows directly
  from A3's own pre-registered gate logic; no new interpretation
  required).

[2026-09-07] D008 P1 -- approved in full (Calls 1-3, F1-F4)
  Decision: approved 1-NN Euclidean accuracy on build/test splits (Call 1
  -- matches the shipped LLmap classifier's own cdist mechanism, verified
  against LLMmap/inference.py:178); structural near-relative pairs (not
  tensor-derived "empirically hardest" ones) for hard-subset accuracy,
  since the paper's literal hard pair is out of universe (Call 2); a
  three-tier identifiability threshold rather than a single percentile,
  pre-registering that the strictest tier will likely be empty (Call 3);
  a selection-stability check for the k=1-3 region CONFIRMED leans on
  (F1); a global MMD bandwidth (F2); a pre-registered INCONCLUSIVE
  fallback if the CVaR-vs-mean gap is smaller than worst-class accuracy's
  4pp resolution (F3); and per-query scale-normalized classifier
  distances (F4).
  Rationale: none of these are value trade-offs. F4 in particular is
  worth naming -- without it, the classifier used to score accuracy
  would implicitly weight the k selected queries by embedding magnitude,
  while GreedyCover selects them symmetrically under I4's MAX rule, which
  would have let an incidental property of which queries got picked
  (not how well the objective picked them) leak into the one comparison
  this D exists to make cleanly. Call 2 avoids a related circularity
  (scoring CVaR against a hard-subset definition drawn from the same
  tensor it optimizes). Both are methodology-correctness fixes, matching
  established project discipline (D004's classification precedent,
  D007's scale-comparability lesson), not genuine trade-offs to escalate.
  Decided by: design-side, routine call.

[2026-09-08] D008 R -- INCONCLUSIVE by the letter; D008 closed; A4 evidence logged
  Decision: D008 marked CLOSED. Verdict is INCONCLUSIVE against its own
  outcome table (CONFIRMED needed both worst-class AND hard-subset to
  improve; only worst-class did, and only at k=1-3) -- reported honestly
  rather than reinterpreted, even though the underlying effect (a large,
  robust query-efficiency gain, every CI excluding zero on mean top-1) is
  not actually weak. Added a dated note to D008's outcome table
  (TACC_NOTES.md issue 12) explaining that the CONFIRMED criterion's two
  required metrics have wildly different achievable resolution
  (worst-class CIs +-0.20; hard-subset's entire random-to-oracle range is
  7.5 points) -- a criterion-design lesson for future work, not grounds
  to reread this result more favorably. A4 updated with this D's
  evidence and a design-side recommendation (resolve toward query
  efficiency) -- still open pending human confirmation.
  Decided by: design-side, routine call (verdict accepted as reported,
  matching D007/R3's precedent of not retroactively re-interpreting a
  result to fit a table's letter); A4's actual resolution escalated to
  human (see A4 above).

[2026-09-08] A4 resolved (a) query efficiency; worst-class gap mechanism verified
  Decision: A4 -> (a) query efficiency is the primary claim, worst-class
  reported as secondary. Separately, the human asked design-side to
  verify a specific hypothesis: does the paper's own claimed algorithm
  (greedy search per Algorithm H.1) actually produce the paper's shipped
  8-query worst-class advantage, or did the shipped defaults get
  hand-augmented with self-identification probes outside the algorithm?
  Verified by cross-referencing two files already on disk -- no new TACC
  run: `results/D008/selection.json` (query-index chains for
  mean_greedy_max/sum and cvar_max, already computed) against
  `confs/queries/pool_v1.json` (per-index provenance/category, already
  built in D006/S2). Finding: of 45 self-identification-style candidates
  in the pool (the paper's own 5 identity-probe anchors + 40
  purpose-generated `generated-banner-grabbing` queries), zero appear in
  the top-16 chain of any of the three greedy reconstructions
  (mean-max/gamma=1.0, mean-sum, CVaR/gamma=0.1). Full writeup and
  theory-grounded mechanism hypothesis (config-fragile within-model
  consistency) added as a note under A4 above.
  Rationale: A4's resolution was already flagged (b) above the note --
  the human made the actual call, this entry just logs it. The
  fidelity-verification was requested explicitly ("需要你核查") and
  answered from existing result files, consistent with the
  do-not-execute-experiments rule (CLAUDE.md) since nothing new was
  computed from the raw tensor -- only already-materialized JSON was
  read. The mechanism *hypothesis* (within-model variance) remains
  unconfirmed at the tensor level and is flagged as such, not asserted
  as fact.
  Decided by: human (A4 resolution); design-side (verification task,
  routine -- answering a direct factual question from existing files).

[2026-09-08] Verified: energy distance is neither the paper's formula nor a
project invention -- it's a genuine deviation/novelty
  Decision: recorded as `PAPER_DEVIATIONS.md` item 10. The human asked
  design-side to confirm whether energy distance (D004/P1's M1b, now
  A3's mandatory companion statistic) is TACC's own invention, the
  paper's own distance formula, or a combination of the paper's
  point-cloud structure and energy distance's own properties. Verified:
  the paper never instantiates any d(.,.) at all (Fact 1, already
  confirmed against paper text), so there is no paper formula to have
  reused; the formula itself (E(A,B) = 2*mean||a-b|| -
  mean||a-a'|| - mean||b-b'||) matches the standard literature
  definition (Szekely & Rizzo 2004; Rizzo & Szekely 2016 review,
  confirmed via web search 2026-09-08), so it predates this project and
  was not invented here either. The genuine combination -- applying an
  off-the-shelf statistic to the I3 point-cloud structure as the thing
  GreedyCover optimizes directly -- is CDQD's own contribution, not
  present in the paper (which uses point clouds only implicitly, to
  train a stage-2 siamese network, never as an explicit two-sample-test
  input). D4's scope narrowed: energy distance's citation is resolved
  and no longer an open citation risk; CVaR/submodularity hardness
  citations remain the only unverified item under D4.
  Rationale: a factual verification task answerable from files already
  in the repo (D004.md, DECISIONS.md A3) plus one citation lookup --
  routine ledger maintenance matching PAPER_DEVIATIONS.md's existing
  9-item discipline, not a trade-off. Separately flagged (not
  actioned): METHOD.md sec4 should eventually name this statistic and
  cite it explicitly rather than staying at the abstract Sep(.,.)
  level -- revising METHOD.md is a human-escalation item per CLAUDE.md
  rule (d), so this is recorded as open, not done.
  Decided by: design-side, routine call (verification + ledger entry);
  the METHOD.md revision itself is NOT decided -- awaiting human call.

[2026-09-08] Phase 3 paused; METHOD.md sec5.5 added; D009 promoted
  Decision: human decided to pause Phase 3 (targeted generation,
  METHOD.md sec5.4's outer loop) and prioritize validating whether
  D008's tensor-level proxy-metric gains (nearest-point-cloud stand-in
  classifier over raw I5 embeddings) survive contact with the paper's
  actual trained pipeline (frozen I5 + trained projection + trained
  self-attention siamese/classifier). METHOD.md sec5.5 added specifying
  that protocol (human-approved, edited by design-side, diff reviewed
  before commit per the human's explicit request). Verified along the
  way, against LLMmap/inference.py:92-113 directly (not assumed): the
  released pipeline embeds query and response separately and
  concatenates in embedding space (2048-d per query slot), which
  sec5.1-5.4's response-only tensor never needed to replicate (a
  per-query constant cancels exactly in any distance-based statistic,
  so D004/D006/D007/D008's numbers are unaffected) but sec5.5's trained
  classifier does need (query embedding is what lets the network
  generalize across strategies selecting different k-subsets of the
  pool) -- flagged as a required, cheap addition (embed each strategy's
  selected query texts) rather than a blocker.
  D009 drafted and promoted: trains LLmap's own released trainer
  (LLMmap/trainer.py, LLMmap/inference_model_archs.py, reused
  unmodified) once per D008 query-strategy condition, holding the
  training procedure identical. Explicitly scoped OUT of D009: METHOD.md
  sec5.4's outer loop (DIAGNOSE / cross-round MODEL SELECTION / TARGETED
  GENERATION) -- D009 is a single-round evaluation of D008's four
  already-fixed query sets, not a resumption of the iterative loop.
  PLAN.md's index and "Not yet a D" section updated to match.
  Rationale: this is a genuine sequencing call with real
  trade-offs (per CLAUDE.md escalation rule (c)) -- made by the human,
  not design-side. The METHOD.md revision is rule (d) -- made by
  design-side only after the human granted permission and reviewed a
  diff, per their explicit instruction this session. Drafting D009
  itself is a routine design-side act once the direction was set.
  Decided by: human (pause Phase 3; prioritize D009; grant METHOD.md
  edit permission); design-side (D009's concrete content, PLAN.md
  bookkeeping).

[2026-09-09] D009/P1 -- APPROVED WITH AMENDMENTS; F7 open-set escalated
  Decision: Calls 1 (full k=1..8 grid) and 2 (5 paired seeds) approved
  as TACC proposed -- the pilot measured 8.7s/run, so the cost concern
  that motivated a smaller grid/seed count in D009's own drafting no
  longer applies. F1/F3/F4/F5/F6 accepted as reported. F2 (early-stop on
  S_val, touch S_test once, rather than reproducing the shipped
  train.py's monitor-on-test behavior) confirmed correct and required by
  I2 independent of the chat instruction TACC cited. F7 (open-set
  evaluation, priced at ~3 GPU-hours for k=8 across all four conditions)
  is NOT approved to execute yet -- escalated to the human as a genuine
  scope question (run now as a disclosed supplementary measurement that
  does not alter D009's closed-set verdict, or defer to a new D),
  design-side recommendation: run it now, given trivial cost, warm
  engineering context, and that it would directly re-test D001/D004's
  original proxy instrument (the paper's own open-set trained
  representation) against a real result for the first time.
  Two documentation fixes made alongside the review, not new decisions:
  METHOD.md sec5.5 and PAPER_DEVIATIONS.md item 3 corrected from "~8M
  params, f_p 1024->384" (my own error, carried from item 3's
  pre-concatenation-discovery figure) to the measured "3,023,653 params,
  f_p 2048->384" (D009 pilot). PAPER_DEVIATIONS.md items 11 (test_model.py's
  top-k ranking backwards for closed-set models -- released-code bug,
  same family as item 9) and 12 (InferenceModel's 650-char inference-time
  truncation is label-correlated, 14.3-92.7% per-model rate spread --
  released-code quirk, distinct from item 1's generation-time figure)
  added.
  Rationale: Calls 1/2 and F1/F3/F4/F5/F6 are routine once cost is
  measured and correctness is verified -- no genuine trade-off, decided
  by design-side per standing practice. F7 is a real scope question
  (CLAUDE.md rule (c)) the human is better positioned to weigh against
  their own priorities for this D, especially since they originated the
  open-set ask directly with TACC -- escalated rather than decided
  unilaterally, even though a recommendation was given.
  Decided by: design-side (Calls 1/2, F1-F6, documentation fixes); human
  (F7, still open).

[2026-09-09] F7 approved; F2's instruction confirmed; D009 fully unblocked
  Decision: human approved F7 -- D009 now runs the open-set evaluation
  (S8, k=8, all four conditions, ~3 GPU-hours) as a supplementary,
  disclosed deliverable that does not enter the closed-set-only outcome
  table. D009's ## D/Required steps/Constraints/Deliverables sections
  updated to fold S8 in formally, not left as a Review-only note, so a
  fresh session reading the D file has the complete, current scope.
  Separately, the human confirmed F2's cited instruction ("reproducing
  the pipeline does not mean reproducing its unreasonable choices") was
  given by them directly, not a TACC misreading. Recorded as
  PAPER_DEVIATIONS.md item 13: the released train.py has no held-out
  validation set at the classifier-training level (same underlying
  pattern item 6 already documents at the query-selection level) --
  logged explicitly, with the alternative (paper's literal 2-way
  protocol) specified precisely enough to run later if a fidelity check
  ever wants it, per the human's stated reason for asking this be
  recorded.
  Rationale: F7 was a genuine scope trade-off (rule (c)), the human's
  call to make; documenting F2's provenance and the paper's alternative
  protocol is routine ledger maintenance once the human confirmed the
  fact, matching this project's standing PAPER_DEVIATIONS.md discipline.
  Decided by: human (F7 approval; confirming F2's provenance and asking
  it be recorded); design-side (folding S8 into D009's formal sections,
  item 13's writeup).

[2026-09-09] D009 R -- CONFIRMED vs mean-greedy/random, tie vs paper8; closed
  Decision: D009 marked CLOSED. Verdict per its own outcome table:
  CONFIRMED against mean-greedy (every k, every CI excludes zero) and
  random; a tie against the paper's own 8 queries (+0.007 mean top-1 at
  k=8, not resolved against the 5-seed range). The tensor-level proxy
  (D004-D008) is validated as a sound ranking instrument (k-curve
  Spearman 0.95-0.98 vs the trained result, same sign on the CVaR-vs-
  mean-greedy gap at 8/8 k on mean top-1) -- D009's FALSIFIED branch
  does not fire.
  The human asked whether Algorithm H.1's actual selection process had
  been read carefully before accepting the tie at face value. Verified
  directly against Appendix H (not METHOD.md's existing summary of it):
  GREEDY_QUERY_OPT trains a real inference model and evaluates its real
  accuracy for every candidate query at every greedy step -- 372 total
  training runs to produce the paper's 8 queries (matching METHOD.md
  sec6.3's existing cost figure exactly, now connected to what it
  implies for a same-classifier comparison). CVaR never touches a
  trained classifier during selection. Recorded as a post-hoc Review
  reframing (D009.md), not a re-verdict: the tie is evidence for CDQD's
  cost thesis (parity at near-zero selection cost vs 372 real training
  runs), not evidence against CVaR. A4's note updated to reflect this.
  Also logged: TACC's R2 mechanism hypothesis (per-query objective vs.
  interaction-aware joint classifier) is directly supported by this
  finding -- Algorithm H.1's retrain-per-candidate loop is itself an
  expensive form of interaction-aware set optimization. "Score query
  sets, not individual queries" added to PLAN.md's "Not yet a D" as the
  most concretely motivated next step, not drafted as a D yet.
  Rationale: closing on a pre-registered, satisfied outcome table is
  routine (CLAUDE.md's standing definition of "closed"); the Algorithm
  H.1 reframing is a factual verification against the primary source,
  not a judgment call -- matches this project's standing practice of
  checking the paper's actual text rather than a prior summary of it
  before drawing a conclusion from a same-classifier comparison.
  Decided by: design-side, routine call (closure); the underlying
  question (was Algorithm H.1 read carefully) was the human's, answered
  by direct paper verification.

[2026-09-10] D010 drafted (Direction A); Direction B shelved; D5 writeup note added
  Decision: human resolved the "score query sets" idea into two tracks.
  Direction A (a classifier-free, set-level statistic, parallel to
  GreedyCover's MAX-aggregated one) drafted as D010 -- stays inside
  CDQD's cost thesis, no training in the selection loop. Direction B (an
  actual classifier-in-the-loop greedy search on our own pool, newly
  affordable at ~5 GPU-hours per D009's pilot) explicitly shelved --
  not drafted, not triggered automatically by D010's outcome. Separately,
  DECISIONS.md Part D gained D5: a concise writeup-framing note (report
  three separate claims instead of one "fair comparison") kept apart
  from this experimental log per the human's explicit request.
  Rationale: Direction A is the routine, in-scope design-side drafting
  task the human authorized directly ("方向A写成D"); Direction B stays a
  human call since it's a real scope/cost trade-off with no automatic
  trigger condition, not something design-side should promote to a D on
  its own reading of D010's eventual result.
  Decided by: human (which direction to draft, which to shelve, and the
  writeup note's placement/separation); design-side (D010's concrete
  content).

[2026-09-11] D010 R -- CONFIRMED, reverses the paper-8 gap; closed
  Decision: D010 marked CLOSED. Verdict per its own outcome table:
  CONFIRMED -- the classifier-free joint (set-level) statistic beats the
  paper's 8 queries at all 8 k values (mean +0.026, every CI-excluding-zero
  case in the same direction) and beats CVaR-coverage at k=8 (+0.024),
  selection staying at 0.5 minutes of CPU against Algorithm H.1's 372
  training runs. k=1 sanity check passed bitwise (joint statistic reduces
  exactly to the per-query one when there is no set to have interaction
  in). MMD robustness check (Review amendment) agrees directionally,
  ruling out an energy-distance-specific artifact. Mechanistically
  confirms D009/R2's hypothesis concretely: the joint criterion selected
  one of the paper's own 8 anchors (q=0) that CVaR-coverage's per-query
  MAX never picks at any k -- exactly the "unremarkable alone,
  complementary in combination" object I4's aggregation rule cannot see
  by construction.
  Two honest limits accepted as reported, not smoothed over: no single k
  clears the 5-seed variance range (case rests on 8/8 sign-consistency,
  not any one cell), and the pre-registered F1 prediction was only half
  right -- the advantage does not fade by k=8 as the pilot and the
  selection objective both predicted, and this D does not explain why.
  New open item (A6 above): three D's in a row (D008/D009/D010) are now
  limited by S_test's 25-config resolution -- fixing it needs new
  generation (I6/I7-adjacent), logged as a human cost decision, not
  actioned, not urgent (every verdict so far didn't need it).
  Rationale: closing on a satisfied, pre-registered outcome table is
  routine; A6 is a genuine trade-off (CLAUDE.md rule (c)) since it costs
  real corpus-generation resources for a resolution improvement with no
  specific number currently needing it -- escalated rather than decided,
  matching TACC's own R9 framing ("a human call about corpus cost, not
  something TACC can resolve").
  Decided by: design-side, routine call (closure, A6's logging); A6's
  actual resolution (whether/when to expand S_test) left open for the
  human.
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
