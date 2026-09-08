# FINDINGS.md — Accumulated Results

> Project-level picture, read periodically by the human (`CLAUDE.md`). One
> entry per D once it has something worth remembering beyond its own file —
> not a duplicate of `## R`, a distilled takeaway.

---

## D001 — Does exploitable hard-pair structure exist?

**Structural premise: yes, decisively.** Near-relative model pairs (same
base model, fine-tunes, adjacent versions) are measurably harder to
separate than unrelated pairs (Mann-Whitney p = 3.5e-32 over all 1326 pairs
among 52 models; the paper's own motivating case, Llama-3-70B ↔ Smaug-70B,
lands at the 75th hardness percentile). This premise is not in question.

**Exploitable-tail premise: measured as "no," but the measurement is
suspect, not the underlying claim.** 88.2% of pairs registered as perfectly
separable under this instrument, which flattens each model's response
distribution into a single averaged point before comparing — the opposite
of what the project's own point-cloud-vs-point-cloud rule (invariant I3)
requires. Treat "no hard tail" as **unresolved**, not settled, until a
non-saturating, I3-compliant remeasurement is run. Human approved
commissioning that remeasurement on 2026-09-01 — see [D004](D004.md).

**Practical takeaway for A1 (model universe):** a ≤14B universe satisfies
invariant I1 comfortably (65 near-relative pairs across 37 models). The
original recommendation on this page (~25×≤14B core + one 70B pair) is
**superseded — A1 is now decided (2026-09-05): all 37 ≤14B open-weight
models, no 70B pair.** The 70B question surfaced a real memory tradeoff
(D002 §R5 — a 70B model doesn't fit on one GH200 at full precision) that
was deferred rather than engineered around, once checking the full ranked
pair list (D004) showed the single hardest pair in the entire universe is
already ≤14B (`Falcon3-10B↔Falcon3-7B`) — see `DECISIONS.md` A1 for the
full reasoning. Cost/wall-clock analysis for building the corpus over all
37 (not a curated 25) is in [D006](D006.md).

**Closed 2026-09-02: R6's open question (was the "no hard tail" verdict an
instrument artifact?) is answered — see [D004](D004.md). It was.**

---

## D003 — Can a proxy LLM generate the Q_0 candidate query pool?

**Yes.** 233 candidate queries (93% of a design-side-revised ~250 target,
raised from the paper-scale ~100 baseline once D002 showed generation cost
isn't the binding constraint at 2–3x that size). All four query families
plus the prompt-injection variant are represented; all 24 published/paper
anchors recovered and verified via arXiv HTML rendering after a naive PDF
text-extraction split would have silently mis-assigned one entry between
the two baseline sets. Generator (`allenai/OLMo-2-1124-13B-Instruct`) was
deliberately decoupled from the 52-model test universe to avoid
self-preference (`DECISIONS.md` C6).

**Process note worth keeping:** TACC found and fixed two bugs in its own
generation harness mid-run — a deduplication step that was comparing
wrapped-prompt strings (61% shared boilerplate) rather than the inner
question, silently discarding distinct probes as near-duplicates; and a
prompt-wording gap that let 55% of one family's queries degenerate into
neutral trivia with zero inter-model discrepancy (i.e. dead weight for any
selection algorithm). Both disclosed with a preserved before/after diff
rather than fixed silently — the corrected pool is what shipped.

---

## D004 — Does an exploitable hard tail exist under a fair instrument?

**Yes — D001's negative result was a censored-instrument artifact, not a
real absence.** D001 measured 88% of pairs as pinned at a perfect-AUC
ceiling using a statistic that collapsed each model's responses to one
average point before comparing (violates invariant I3). D004 re-measured
the same underlying trace data with an **unbounded** statistic (energy
distance between point clouds, which structurally cannot saturate) and
found a real, large tail: `CVaR₀.₁/mean = 0.616` (bootstrap CI95
[0.582, 0.669]) — decisively below the ≈1 a "no tail" world would produce.
Two bounded statistics run in parallel (a linear-probe AUC and a 52-way
classifier) both saturated almost exactly as pre-registered *before* the
run, which is what makes the attribution to instrument censoring credible
rather than post-hoc.

**`METHOD.md §2` and invariant I1 stand as written.** This was the
open question D001 §R6 flagged and the human approved commissioning a
re-measurement for; it's now closed.

**The headline number for A1:** the project's own motivating pair,
Llama-3-70B ↔ Smaug-Llama-3-70B, is the **5th hardest of 1326 pairs**
(percentile 0.30) under the fair instrument — harder than D001's original
estimate (0.75), not easier. This sharpens (does not resolve) the still-open
A1 question of whether to pay the memory/engineering cost to include a
70B-class pair (`DECISIONS.md` A1, from [D002](D002.md) §R5): the pair in
question is now confirmed to be an unusually strong demonstration case.

**Secondary finding, low urgency:** the `mistral-7b`/`mixtral-8x7b` lineage
annotation likely undercounts true near-relative pairs (same-product-line
pairs land in the top 10 hardest despite being tagged "unrelated") —
means D004's reported effect size is a lower bound, and I1's pair count is
conservative rather than inflated. No action needed unless a future D
depends on exact lineage counts.

---

## D002 — Is GPU compute available on Vista, and what does Phase 1 cost?

**Yes, and it was never really in question — the CPU-only run in D001 was
a wheel-selection mistake, not a platform limitation.** Confirmed by an
on-device tensor op cross-checked against CPU, not by documentation.

**The real finding: SUs are not the constraint, wall-clock is — and it
interacts with storage.** Even the most expensive candidate corpus size
(30 models × 120 queries, unbatched) uses under 8% of the compute
allocation. What actually binds is that unbatched generation takes longer
than `$SCRATCH`'s 10-day no-access purge window; batching the 8 queries
within one prompt config (already possible in the released code, just not
wired up — `dataset_maker.py:36-39` generates one at a time) is a
3.7–6.9x speedup and the difference between finishing inside that window
and not. **Fixing that batching loop is now a hard prerequisite for
Phase-1 corpus construction**, not an optional optimization.

**Unresolved and now the most consequential open A1 question:** a 70B-class
model needs ~140GB of weights against ~95GB visible on one GH200 — doesn't
fit at full precision. Quantizing risks perturbing the very output
distributions this project fingerprints. See `DECISIONS.md` A1.

---

## D005 — Fixing the I2 violations in the prompt-config generator

**Fixed, and proven fixed.** Both defects (the dropped `pool` argument;
`sampling_universe` having no train/test split at all) are corrected, with
a regression test that reintroduces the original bug and confirms it
fails (250 named violations), then confirms the fix passes (0). `do_sample`
(a 2-value parameter I2's literal wording can't apply to without creating
a worse confound) is now a documented, explicit exception rather than a
silent gap (`DECISIONS.md` A5).

**A genuinely new class of bug, worth remembering beyond this D:** fixing
the temperature split turned up a float-aliasing defect nobody had
anticipated — `0.9999999999999999` and `1.0` are different floats
representing the same real-world sampling temperature, so an index-wise
disjoint split would have silently reintroduced value-level leakage while
looking correct. General lesson: any config split defined over a *value*
space needs disjointness checked by value, not by index, whenever the
values are floats.

**Whether the shipped `default_dataset.jsonl` (D001/D004's data source)
carries either defect: UNDETERMINABLE, and genuinely so** — not a shrug.
Four hypotheses about how the dataset was generated were tested against
persona-echo evidence and all four were refuted under the currently
checked-in split file; the likely explanation is that the split file
itself was regenerated after the dataset (seed 42, squashed git history)
rather than being the one actually used. **Doesn't matter for D001/D004
either way:** neither called the buggy code path, and the one hypothesis
positively ruled out is the specific leak this D fixed — so if the shipped
corpus has *some* leakage, it's probably not this one, and any leakage
would inflate separability (work against D004's confirmed tail, not for
it). No re-run warranted.

**Phase-1 corpus construction has one fewer blocker.** With C3 (~250),
D003 (233-entry pool), and D005 (I2 holds) all in place, A1 (model
universe) became the only remaining blocker — **since resolved (37
models, 2026-09-05), and the Phase-1 D is now drafted: [D006](D006.md).**

---

## D006 — The real Phase-1 corpus: built, READY at 37/37

**The project's own corpus (not the shipped LLmap one D001/D004/D005
analyzed) is done.** 1,197,875 responses across 37 models, 259 queries
(D003's 233 + 26 new tokenizer-level probes), 125 build/val/test configs
each, embedded with the frozen I5 model. All of I1/I2/I3/I5/I7 verified
against the corpus *as generated*, not just its definitions.

**A real, measured improvement found late: don't normalize the
embeddings.** TACC checked its embedding procedure against the paper's
actual released code (prompted by a design-side correction to
`PAPER_DEVIATIONS.md` — I5 turned out to be the paper's own frozen
stage-1 embedding, not a substitute for it, so checking the *rest* of the
paper's procedure became worth doing) and found it was L2-normalizing
response vectors, which the paper does not. Removing normalization
improved separability on 15 of 15 test pairs (+0.051 AUC) — response
*magnitude*, not just direction, carries real model-discriminative
information. Fixed; the whole corpus was cheaply re-embedded from stored
raw text.

**One open thread this created, already being handled, not lost:** an
earlier finding (D006's C7 pilot: "100-token responses separate better
than 200-token ones") was measured on the *normalized* embeddings and is
now provisional — it may or may not survive re-measurement on the
corrected ones. [D007](D007.md) re-checks this properly (full 666 pairs,
corrected embeddings) before freezing which analysis length the real
separability tensor uses, rather than carrying the provisional finding
forward unverified.

**A second, unrelated corpus-quality bug also found and fixed during
D006:** the released code's test for "does this model's chat template
support a system role" was a text-substring check on the template
source, not a check on what actually happens when a system message is
rendered — wrong for 10 of the 37 models, in three different ways (5
gemma models crashed instead of receiving the prompt; one model,
`Llama3-ChatQA-1.5-8B`, silently *dropped* the system content in ~90% of
its configs, the more dangerous failure since nothing errored; 4 more
models had their system prompt misplaced into the user turn). Fixed with
a behavioral probe instead of a text heuristic; the 10 affected shards
were regenerated. This is a bug in the released code's rewrite of the
paper (LLMmap0.2), not a paper-vs-code deviation with a known paper value
— recorded in `PAPER_DEVIATIONS.md` (item 9) for that reason.

---

## D007 — The real separability tensor: TAIL CONFIRMED, decisively

**Phase 1 is done. The hard-tail premise now holds on the project's own
corpus, own embedding, and full query pool — not just on D004's proxy
instrument.** `CVaR₀.₁/mean = 0.353` on the unbounded statistic
(`S_energy`) — a *larger* tail than D004 found (0.616), measured on 259
queries and 666 pairs instead of 8 queries and a proxy representation.

**The bounded statistic (probe AUC) turned out to be unusable at this
scale, exactly as A3's design anticipated but more severely than
expected** — 51.8% of pairs pinned at ceiling even after correcting for
selection bias, 98.2% before. This is why A3 made energy distance a
*mandatory* companion with a formal resolution-audit gate rather than an
optional check: applied naively, the bounded statistic alone would have
reported `TAIL STILL ABSENT` — a false falsification that would have
escalated to a `METHOD.md`/I1 revision question for no real reason.

**A secondary methodological lesson, worth remembering beyond this D:** a
statistically well-reasoned safeguard (the winner's-curse correction, F2)
was approved and executed correctly, but turned out not to be what
prevented the wrong verdict — a *different* safeguard (A3's saturation
gate) did that work instead. Both were worth having; only one was
load-bearing here. Worth remembering when weighing which of several
proposed checks to prioritize under time pressure in the future — it
isn't always obvious in advance which one will matter.

**Also resolved: D006's provisional "100 tokens beats 200" finding was
wrong, not just unverified.** Properly remeasured (666 pairs, corrected
embeddings, both statistics), 200 tokens wins clearly. C7's 200-token
ceiling is vindicated a second time.

**The project's own motivating pair remains the throughline.**
`Falcon3-10B ↔ Falcon3-7B` is the 1.5th-hardest of 666 pairs here — under
a completely different representation and query pool than D001/D004
used. Continuity across three independent measurements is about as
strong as this kind of evidence gets.

**Phase 2 (the actual mean-vs-CVaR query-selection comparison) is now
unblocked** — per `TODO.md`'s own framing, computationally cheap
("seconds per run") now that the expensive part (D006+D007) is done.

---

## D008 — Does the objective matter? INCONCLUSIVE by the letter; a real efficiency win underneath

**The literal verdict is INCONCLUSIVE — the underlying finding is not.**
CVaR-coverage selection beats every baseline (mean-greedy, random, the
paper's own 8 queries) on mean top-1 accuracy at every query budget `k`,
with every confidence interval excluding zero. The effect concentrates
exactly where `METHOD.md §8` said the real headroom was: **query
efficiency.** CVaR reaches 70% mean accuracy at **`k`=2**; mean-greedy
needs 4, the paper's own 8 queries need 5, random needs 12. CVaR matches
the paper's full 8-query accuracy at `k`=4 — half as many queries — and
exceeds it at 6.

**Why the verdict is technically INCONCLUSIVE anyway:** D008's CONFIRMED
branch required improvement on *two* metrics (worst-class *and*
hard-subset accuracy) at equal `k`. Worst-class improves sharply but only
at `k`=1–3; hard-subset barely moves for any condition. Investigated
before accepting that as "no effect": the *oracle* ceiling for
hard-subset (best possible score, cherry-picking per pair, on held-out
data) is 0.982 against a random baseline of 0.907 — the entire usable
range is 7.5 points, making "meaningful improvement" on this metric close
to unsatisfiable regardless of whether CVaR actually helps. Recorded as a
criterion-design lesson, not used to reread the result more favorably.

**Against the paper's own 8 queries specifically, the honest picture is
split, not favorable across the board.** At equal `k`=8, CVaR is not
clearly better on mean accuracy (CI touches zero) and is **worse** on
worst-class accuracy (0.28 vs. the paper's 0.44) — the paper's queries
include direct self-identification probes ("what LLM are you exactly")
that are unusually good at 37-way worst-class discrimination specifically,
which a pairwise-separability objective doesn't target. The place CVaR
clearly wins against the paper is query count, not peak accuracy.

**The most consequential finding for what comes next:** of the pairs no
*selected* query separates well, **33 of 37 have a near-perfect query
already sitting in the 259-query pool** — greedy selection simply missed
it. Only 2 pairs (the same `Falcon3` and `Phi-3-medium` pairs D004 and
D007 already flagged) are genuine limits of the pool itself. This means
"the objective doesn't help enough" would most likely be a
**selection-algorithm** gap, not a pool-coverage gap — a distinction
neither of D008's two clean outcomes (CONFIRMED / FALSIFIED) anticipated,
and directly relevant to how Phase 3 (targeted generation) gets scoped.

**Also confirmed:** `γ=1.0` reduces exactly to mean-greedy (numeric
check against an independent implementation); lower `γ` costs nothing on
mean accuracy here (no visible tail-vs-mean trade-off); the conclusion is
robust to swapping the tensor's statistic for a completely different one
(MMD, ρ=0.975 with energy distance); and what generalizes across a
resample of the build configs is the *objective*, not the particular
selected queries (half the `k`=8 chain changes between build-config
halves, but the CVaR-over-mean-greedy advantage reproduces on both).

---

<!-- append new entries below, one per D, once it produces a project-level
     takeaway worth remembering outside its own file -->
