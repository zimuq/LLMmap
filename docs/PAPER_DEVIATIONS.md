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

### 1. Response generation length — `[code, llm.py:9]`, paper's value UNVERIFIED

**Ours:** 200-token generation ceiling (decided 2026-09-06, `DECISIONS.md`
C7). **Released code:** `max_new_tokens=100` (`llm.py:9`); the shipped
`confs/default.json`'s 650-char figure was measured inert (D002 §R4— real
responses top out at 667 chars, the 100-token cap binds first). **The
paper's own value is not established anywhere in this project.** D006/P1
§F1 and `plans/D006-P1.md:157` both cite "100 tokens" as a fact about the
*shipped corpus*, not as a citation to the paper — this project has never
verified what the published experiments actually used. Treat "100" as a
code default we are deliberately exceeding, not as "the paper's setting."

**Rationale for 200:** generation is causal (token *t* depends only on
tokens `<t`), so a 200-token response truncated to 100 is byte-identical
to generating at 100 directly — only the *ceiling* is a one-way door once
frozen into the corpus under I6/I7. The 100-token default was measured to
censor ~38 percentage points of responses (D006/P1 §F1). **This is an
extension beyond the code's default, explicitly not a reproduction of it**
— human-decided 2026-09-06, aware it is a deviation.

### 2. Model universe — `[UNVERIFIED against the paper; code gives 52]`

**Ours:** 37 open-weight models, `params_b ≤ 14`, no 70B-class pair, no
closed-source (`DECISIONS.md` A1, decided 2026-09-05). **Released code**
ships behavioral templates for 52 models (`results/D001/model_metadata.csv`),
including 70B-class (the paper's own motivating pair, Llama-3-70B ↔
Smaug-Llama-3-70B) and closed-source rows (GPT/Claude, `proprietary=True`).
**Whether this 52-model list is identical to what the published paper
itself evaluated, or a superset/subset the released code happens to ship,
is not verified here.**

**Rationale:** 70B-class excluded on a memory constraint (D002 §R5 — does
not fit one GH200 at full precision), not a decision to avoid it in
principle; kept excluded after data (not just cost) showed the single
hardest pair in the *entire* universe is already ≤14B
(`Falcon3-10B↔Falcon3-7B`, D004). Closed-source excluded per A2's
long-standing default (cost/access), still formally open. **This means
our headline demonstration case is not the paper's own** (Llama-3-70B ↔
Smaug) — cite `Falcon3-10B↔Falcon3-7B` instead, and say so explicitly if
a reviewer would expect the paper's example.

### 3. Separability representation — I5 vs. LLmap's own classifier

**Ours:** frozen `multilingual-e5-large-instruct`, 1024-d, a generic
sentence embedding applied to raw response text (invariant I5). **LLmap's
own instrument:** a 384-d classifier *contrastively trained specifically
to separate these models* (`LLMmap/templates.py`). We used LLmap's own
classifier only as a **proxy instrument** in D001/D004 to test whether an
exploitable hard tail exists at all — explicitly flagged in D004's Review
as **not** the representation the real project uses, with the bias
direction stated (a classifier trained to separate is biased toward
*finding* separation, so D004's positive result is evidence despite that
bias, not because of it).

**Rationale:** using LLmap's own trained classifier as the *actual*
Sep(·,·) instrument would be circular — it's trained on exactly the
models under test. I5 must be frozen and untrained-on-the-universe by
construction (invariant I5's own rationale: comparability across rounds,
not fit to the test set).

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

### 7. Sampling-hyperparameter split — a paper/code deviation *and* our own fix

The paper's own §7.1 (per project notes referencing it) describes
splitting sampling hyperparameters `H` into `Htrain`/`Htest`. **The
released code never implemented this at all** (D005's finding — a
paper-vs-code gap that predates this project, not something we
introduced). Our fix (D005) restored a real split for `temperature`, but
found `do_sample`'s only 2 values can't be disjointly split without
creating a worse confound (all-greedy in one pool, all-stochastic in the
other). Resolved as a **documented, explicit exception** (`do_sample`
shared across pools — `DECISIONS.md` A5) rather than either forcing a
fake split or leaving the gap silent.

### 8. Cost/training-time framing — already covered, cross-referenced

`METHOD.md §6.3`: CDQD trades LLmap's ~372 classifier-training runs for
~5–15, at the cost of more inference generations (~315K+ vs ~157K).
**"20–70× cheaper" must never be claimed** — inference parallelizes and
is checkpointable, training does not; that's the actual saving.

---

## Known gaps in this ledger (flag rather than guess)

- **Item 1's paper-value gap is the most consequential unverified point in
  this file.** If a specific token/length limit for the paper's own
  experiments is ever found (in the paper text, an appendix, or
  supplementary material), record it here immediately with a citation —
  until then, "100 tokens" in any of our docs means *the code's default*,
  not a claim about the paper.
- Item 2's exact paper-reported model count/list is likewise unverified
  against `model_metadata.csv`'s 52 — that list is a fact about the
  released artifact.
- `DECISIONS.md` D4 flags that CVaR/robust-submodular hardness citations
  (`METHOD.md §6.1`) are themselves unverified against the literature —
  a related but distinct kind of unverified claim (theory citation, not a
  paper-setup deviation); tracked there, not duplicated here.

<!-- Append new deviations here as they're decided — one entry per row,
     same tag discipline (paper/code/UNVERIFIED), same "ours vs theirs vs
     why" structure. -->
