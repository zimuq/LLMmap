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

<!-- append new entries below, one per D, once it produces a project-level
     takeaway worth remembering outside its own file -->
