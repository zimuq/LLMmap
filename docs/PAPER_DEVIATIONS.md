# PAPER_DEVIATIONS.md — Where CDQD's setup differs from LLmap

> Maintained by design-side. Updated whenever a `DECISIONS.md` item or a
> `D{id}` closes with implications for paper-comparability — not just at
> writeup time. The point of this file is that writeup time is exactly
> when this information is hardest to reconstruct accurately.

## Why this file exists

Every one of these deviations is individually documented somewhere —
`DECISIONS.md`'s per-item log, a specific D-file's `## R`, or
`METHOD.md §6`'s writeup-framing caveats. None of those are indexed by
*"how does our setup differ from LLmap's,"* which is exactly the question
a methods section, a reviewer, or a future session needs answered
directly. This file is that index — a ledger, not a new analysis.

**Two different reference points, often conflated — keep them apart:**
- **"The paper"** — Pasquini, Kornaropoulos, Ateniese, USENIX Sec'25. What
  it *actually reports* is only as reliable as what we've verified against
  it directly.
- **"The released code"** (`pasquini-dario/LLMmap`, LLMmap0.2) — its own
  README states it is **not** a one-to-one conversion of the paper's code
  (`TODO.md` T-1.1). A value read from the code is a fact about the code,
  not automatically a fact about the paper.

**Every row below is tagged accordingly:** `[paper, §X]` (verified
against a specific section), `[code, path]` (verified against the
released artifact, paper's own value not established here), or
`[UNVERIFIED]` (neither confirmed — flagged rather than assumed).

---

## Deviations

### 1. Response generation length — `[paper, §7.1/§6.1.2 — CONFIRMED ABSENT]`

**Ours:** 200-token generation ceiling (decided 2026-09-06, `DECISIONS.md`
C7). **Released code:** `max_new_tokens=100` (`llm.py:9`); the shipped
`confs/default.json`'s 650-char figure was measured inert (D002 §R4 —
real responses top out at 667 chars, the 100-token cap binds first).
**Checked directly against the paper text (2026-09-06, full-text search of
the extracted PDF): no response-length or `max_new_tokens`-equivalent
parameter is specified anywhere.** This upgrades the earlier "unverified"
flag to a confirmed fact about the paper: **it does not report this
setting at all.** "100 tokens" is, and remains, purely a released-code
default — there is no paper value to compare against, only the code's.

**Rationale for 200:** generation is causal (token *t* depends only on
tokens `<t`), so a 200-token response truncated to 100 is byte-identical
to generating at 100 directly — only the *ceiling* is a one-way door once
frozen into the corpus under I6/I7. The 100-token default was measured to
censor ~38 percentage points of responses (D006/P1 §F1). **This is an
extension beyond the code's default; the paper offers no reproduction
target to deviate from in the first place.**

### 2. Model universe — `[paper, §7.1: 42 models; code ships 52]`

**Ours:** 37 open-weight models, `params_b ≤ 14`, no 70B-class pair, no
closed-source (`DECISIONS.md` A1, decided 2026-09-05). **The paper's own
universe (§7.1, Table G.1): 42 LLM versions** — "primarily... popular
open-source models" plus, for closed-source, "the three main models
offered by the two most popular vendors" (OpenAI, Anthropic) = 6
closed-source, **36 open-source**. **Released code ships templates for 52**
(`results/D001/model_metadata.csv`), including the paper's own 6
closed-source rows (`proprietary=True`) but evidently more open-source
models than the paper reported (46 vs. 36) — the shipped artifact is a
**superset** of the paper's own reported universe, not identical to it
(likely models added to the repo post-publication; not investigated
further here). The paper's own motivating pair, Llama-3-70B ↔
Smaug-Llama-3-70B, **is** among its 42 — confirmed genuinely in-paper, not
just in the released code's superset.

**Rationale for ours (37, ≤14B, no 70B, no closed-source):** 70B-class
excluded on a memory constraint (D002 §R5 — does not fit one GH200 at full
precision), not a decision to avoid it in principle; kept excluded after
data (not just cost) showed the single hardest pair in the *entire*
52-model shipped universe is already ≤14B (`Falcon3-10B↔Falcon3-7B`,
D004). Closed-source excluded per A2's long-standing default (cost/API
access), still formally open. **This means our headline demonstration
case is not the paper's own** (Llama-3-70B ↔ Smaug, genuinely in the
paper's 42) — cite `Falcon3-10B↔Falcon3-7B` instead, and say so explicitly
if a reviewer would expect the paper's example.

### 3. Separability representation — same frozen embedding as the paper; we stop one stage earlier

**Corrected 2026-09-06 — the previous version of this entry was wrong to
frame I5 as "a generic embedding, different from LLmap's."** Read directly
from the paper (§6.1, Fig. 3): LLmap's own inference model is a
**two-stage pipeline**, and **stage 1 is exactly our I5.**

- **Stage 1 — embedding `E`, frozen, not tuned in training (paper's own
  words, Fig. 3 caption): `multilingual-e5-large-instruct`, 1024-d.**
  This is invariant I5, verbatim — not a substitute we chose instead of
  the paper's method, but a direct reuse of it.
- **Stage 2 — a *trained* projection (`fp`, 1024→`m=384`) + a small
  self-attention "siamese"/classifier network (~8M params)**, fit
  contrastively (open-set) or supervised (closed-set) **per query
  strategy** — this is what D001/D004's `f_test.npy`/`templates.py`
  actually is (the 384-d figure is the paper's own `m`, confirmed), and
  it's what CDQD avoids retraining per candidate during selection
  (`METHOD.md §6.3`'s "~372 training runs" cost).

**Ours:** Phases 1–3 (query selection, incl. D006's corpus) compute
`Sep(·,·)` directly on **stage-1 `E`-embeddings** — the same frozen model,
stopped one stage earlier than the paper's full trained pipeline.
**Phase 4 (`TODO.md` T4.1, "train inference models on the candidate
strategies") runs the full paper pipeline** — `E` + a freshly trained
projection/siamese network — once per candidate query-selection strategy,
holding the training procedure identical and varying only which queries
were used to build it. **This is the "hold embed/classify constant, vary
only query strategy" comparison — it lives in Phase 4, not throughout the
pipeline**, and Phase 4 has not started (blocked behind Phase 1–3).

**Rationale for stopping at stage 1 during selection:** using the fully
*trained* stage-2 network as the selection-time instrument would be
circular (it's fit on the exact models under test, and a query strategy's
tensor would depend on a network already shaped by that strategy's own
queries) — and would reintroduce the retraining cost CDQD exists to avoid.
D001/D004 used the *paper's own trained* stage-2 output as a proxy
instrument specifically to sanity-check the hard-tail premise before
committing to the real corpus — explicitly flagged there as biased toward
*finding* separation (trained-to-separate representation), which is why a
positive result (D004: TAIL CONFIRMED) was treated as comparatively strong
evidence despite, not because of, that bias.

**A false start worth recording, not just the correction (D006/R10,
2026-09-06):** having the right *model name* for stage 1 is not the same
as having the right *procedure*. D006's first embedding pass L2-normalized
every response vector before storing it — the paper's own stage 1 does
not (verified against `LLMmap/embedding_model.py:15-26` directly, not
assumed from the model name). Measured effect of removing it: **+0.051
probe AUC, 15 of 15 test pairs improved.** Response *magnitude*, not just
direction, carries real model-discriminative signal, and normalizing
discarded it. Fixed; the corpus was re-embedded from stored raw text
(~12 min). **Lesson for future stage-1-adjacent work: verify pooling,
normalization, truncation length, and special-token handling against the
paper's actual code, not just the embedding model's name** — this is
exactly the kind of silent, no-error-message deviation the rest of this
project's invariants (I1–I7) already exist to guard against, just one
layer further upstream than usual.

### 4. Query pool construction — `[paper, Table F.1 for baselines]`

**Ours:** a proxy-LLM-generated candidate pool (`allenai/OLMo-2-1124-13B-Instruct`,
deliberately decoupled from the test universe, C6), 233 entries across the
paper's four query families plus a dedicated prompt-injection-wrapper
pass, **plus new tokenizer-level probes** (digit-chunking, whitespace-
merging, Unicode-byte-fallback, glitch-token — D006 §S2/Call 1, 26
instances) that do not exist in the paper at all (its own §5.1 calls
glitch tokens "unexplored future work"). The paper's original 8 queries
(`confs/queries/default.json`) and its two published baseline sets
(`gpt4o-gen-opt`, `random-opt`, Table F.1) are preserved **verbatim** as
anchors inside our larger pool, not replaced.

**Rationale:** D003's whole justification — a small/narrow pool risks a
confound where mean and CVaR look indistinguishable only because nothing
gave them room to disagree, not because the objective doesn't matter. The
tokenizer probes are a genuinely new contribution area the paper flagged
but didn't build.

### 5. Selection-algorithm baseline — mean-greedy is *our* reconstruction

Already fully documented in `METHOD.md §6.5` — cross-referenced, not
duplicated here. **One-line summary:** our mean baseline is `GreedyCover`
at `γ=1.0`, framed explicitly as *"our reconstruction of mean-based
selection,"* never as *"reproducing LLmap's algorithm"* — Algorithm H.1's
real objective (downstream classifier accuracy) never actually
instantiates a `d(·,·)` distance function to reproduce.

### 6. Train/test protocol — 3-way vs. the paper's 2-way — `[paper/code, §6.2]`

Already fully documented in `METHOD.md §6.2` — cross-referenced, not
duplicated here. **One-line summary:** LLmap's Algorithm H.1 selects on
`T_test` and reports on the same distribution; we use a 3-way
build/val/test split (invariant I2, `DECISIONS.md` C4) and touch `S_test`
once. **Our protocol is stricter, so our absolute numbers may look lower
than the paper's 95.35% even where the method is better** — `METHOD.md`
already requires this be stated in any writeup (TODO.md T4.4's planned
"honest comparability statement" deliverable is this same point, not yet
written as prose).

### 7. Sampling-hyperparameter split — a paper/code deviation *and* our own fix — `[paper, §7.1, confirmed direct 2026-09-06]`

**Confirmed directly against the paper text:** §7.1 defines
`H = [0,1] × [0.65,1]` — **`temperature` × `frequency_penalty`**, two
continuous parameters — and states "we split `H` in two equal sized sets
`Htrain` and `Htest`." **The released code implements neither the split
nor `frequency_penalty` itself** — `sampling_universe` only has
`temperature`/`do_sample`, and `do_sample` (binary, sampling on/off) isn't
in the paper's `H` at all (a released-code parameter with no paper
counterpart, not the reverse). This is a paper-vs-code gap that predates
this project (D005's finding), not something we introduced.

Our fix (D005) restored a real split for `temperature`. `do_sample` — the
code's own addition, not the paper's — can't be disjointly split (only 2
values; would confound decoding mode with split membership). Resolved as
a **documented, explicit exception** (shared across pools — `DECISIONS.md`
A5) rather than forcing a fake split or leaving the gap silent.
`frequency_penalty` remains unimplemented in the released code; matching
the paper's `H` exactly would mean adding it, not just fixing the split
mechanism — noted, not undertaken (would need its own D + I7 bump).

### 8. Cost/training-time framing — already covered, cross-referenced

`METHOD.md §6.3`: CDQD trades LLmap's ~372 classifier-training runs for
~5–15, at the cost of more inference generations (~315K+ vs ~157K).
**"20–70× cheaper" must never be claimed** — inference parallelizes and
is checkpointable, training does not; that's the actual saving.

### 9. System-prompt handling — a released-code bug, not a deliberate deviation — `[code, LLMmap/llm.py, bug]`

`_does_template_have_system` (the released code's check for whether a
model's chat template accepts a system message) tested `"system" in
chat_template` — a substring match on the template's *source text*, not
on what actually happens when a system message is rendered. Measured
against all 37 of D006's models (2026-09-06), this is wrong for **10**,
in both directions:

- **5 gemma models:** the template contains `"system"` only inside a
  `raise_exception('System role not supported')` branch. The substring
  test says yes; rendering actually **raises**, and every gemma shard
  crashed.
- **`nvidia/Llama3-ChatQA-1.5-8B`:** tests true but **silently drops** the
  system content — the more dangerous failure, since the shard completed
  normally with the system prompt missing from ~90% of its configs
  (`WITH_SYSTEM_P=0.9`) rather than erroring.
- **4 models** (`aya-23-8B`, `Llama-3-8B-Gradient-1048k`,
  `Meta-Llama-3-8B-Instruct`, `openchat_3.5`): test false but **do**
  support a system role, so their system prompt was silently prepended
  to the user turn instead of using the role the template actually
  supports.

Fixed with a behavioral probe (render a sentinel system message, check it
survives) rather than a text-based heuristic. **This is a defect in the
released code (LLMmap0.2, "not a one-to-one conversion of the paper's
code," `TODO.md` T-1.1) — not a deliberate deviation, and not something
this project introduced.** Whether the paper's own original experiments
used code with this same defect is unknown; nothing here establishes
that. Consequence: 10 of D006's 37 shards were (re)generated after the
fix (~10 node-hours) — the corpus does not carry this defect, but it
means those 10 models' prompt construction genuinely differs from
whatever the released code alone would have produced.

---

## Known gaps in this ledger (flag rather than guess)

- **The paper PDF itself was checked directly (2026-09-06)** — full-text
  search of an extracted copy — resolving items 1, 2, 3, and 7 above from
  "unverified" to confirmed facts, one of which (item 3) corrected a
  previously-wrong framing rather than just filling in a blank. The PDF
  lives at `docs/LLMmap Fingerprinting for Large Language Models.pdf` in
  the **main checkout**, not this worktree (untracked, same pattern as
  earlier onboarding files) — extract with `pdftotext -layout` before
  grepping; the `Read` tool's PDF path needs `poppler-utils`
  (`pdftoppm`), not present in this environment.
- **Item 2's exact model-list correspondence (which 46 open-source models
  the code adds beyond the paper's 36) is not enumerated** — confirmed
  the *counts* differ (42 paper vs. 52 code, with the same 6 closed-source
  rows) but not which specific open-source models are code-only additions.
  Low priority — doesn't affect our own A1 (already restricted to ≤14B,
  open-weight), only matters if a writeup needs to state precisely which
  paper models we do/don't include.
- `DECISIONS.md` D4 flags that CVaR/robust-submodular hardness citations
  (`METHOD.md §6.1`) are themselves unverified against the literature —
  a related but distinct kind of unverified claim (theory citation, not a
  paper-setup deviation); tracked there, not duplicated here.

<!-- Append new deviations here as they're decided — one entry per row,
     same tag discipline (paper/code/UNVERIFIED), same "ours vs theirs vs
     why" structure. -->
