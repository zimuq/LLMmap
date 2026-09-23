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

**Follow-up, 2026-09-08 (verified from existing files, no new run):**
checked whether the paper's own *claimed* algorithm (greedy search) would
actually select those identity probes, or whether the shipped 8 were
hand-augmented beyond it. Our pool has 45 self-identification-style
candidates available (the paper's 5 identity anchors + 40 generated
`generated-banner-grabbing` queries) — **none are ever selected**, in the
top-16 chain, by any of the three greedy reconstructions computed here,
including `mean_greedy_max` (γ=1.0, the closest reconstruction of the
paper's own claimed objective family). This means the worst-class gap is
not "our selection algorithm missed a good move available to it" — even
the paper's own claimed algorithm family declines the same candidates.
Likely mechanism (theory-grounded per `METHOD.md §4`'s Sep = between-group
/ within-group ratio, not yet confirmed at the tensor level): identity
probes plausibly have config-fragile within-model consistency, which the
objective penalizes by design. Full detail: `DECISIONS.md` A4 note,
`D008.md` addendum.

**The "most consequential finding" from this D was retired by D011,
2026-09-13 — recorded here for the corrected picture, not the original
one.** D008 flagged 37 (later corrected to 32, split-half robust) pairs
where a per-pair single-query oracle scored badly on test. D011 checked
whether either real selected 8-query chain (`GreedyCover`'s own, or the
later `JointGreedy`) actually fails on these pairs — **both resolve all
32** (32/32 each, ≥0.75 test accuracy; the flagged single-query pick
itself scores a median of 0.54, *worse than a random pool query's
0.82*). The flag was measuring an estimator defect (`argmax` over 259
noisy per-query build-time scores, a severe winner's-curse effect) — not
a selection-algorithm gap, not a motivation for Phase 3, not a property
of the model universe. The two pairs this project has tracked as
genuinely hard since D001 (`Falcon3-10B↔7B`, `Phi-3-medium-128k↔4k`) are
not even in the 32. Full detail: D011's `## R`, `D008.md`'s addenda.

**Also confirmed:** `γ=1.0` reduces exactly to mean-greedy (numeric
check against an independent implementation); lower `γ` costs nothing on
mean accuracy here (no visible tail-vs-mean trade-off); the conclusion is
robust to swapping the tensor's statistic for a completely different one
(MMD, ρ=0.975 with energy distance); and what generalizes across a
resample of the build configs is the *objective*, not the particular
selected queries (half the `k`=8 chain changes between build-config
halves, but the CVaR-over-mean-greedy advantage reproduces on both).

---

## D009 — Does the proxy survive a real classifier? CONFIRMED vs. baselines; a tie vs. the paper, at a fraction of its cost

**The headline: everything D004–D008 measured on the tensor-level proxy
holds up once a real classifier is trained.** CVaR-coverage selection,
evaluated by training LLmap's own stage-2 pipeline from scratch (frozen
I5 embedding + trained projection + self-attention siamese/classifier,
`LLMmap/trainer.py`/`inference_model_archs.py`, reused unmodified),
**beats mean-greedy at every `k`=1–8** (every bootstrap CI excludes
zero) **and beats random** — D009's FALSIFIED branch ("the proxy metric
was misleading") does not fire. The proxy's `k`-curve correlates with
the trained result at Spearman 0.95–0.98 across all four conditions, and
the sign of the CVaR-over-mean-greedy gap agrees at 8/8 `k` on mean
top-1. This is the license the project needed to keep selecting on a
cheap tensor instead of training a network per candidate — no longer an
assumption, a measured fact.

**Against the paper's own 8 queries specifically: a tie, not a win — and
that tie is a stronger result than it looks.** Mean top-1 differs by
+0.005 to +0.024 across `k`, none of it resolved against the 5-seed
training-variance range. Read in isolation, this looks like "our method
doesn't beat the paper's queries." **It should not be read in
isolation.** Checked directly against `Appendix H`'s Algorithm H.1
pseudocode (not from memory, not from this project's own prior summary
of it): the paper's 8 queries were produced by training a real inference
classifier and evaluating its real accuracy for **every candidate query
at every one of 8 greedy steps** — 372 total training runs
(`50+49+…+43`, the exact figure `METHOD.md §6.3`'s cost table already
carried, now connected to what it implies here). CVaR never touches a
trained classifier during selection at all. **Tying a strategy that cost
372 real training runs to produce, using a selection process that costs
effectively zero, is evidence for CDQD's core cost thesis — not a sign
that CVaR-coverage failed to find an edge.** Any writeup should say
*"ties the paper's own hand-optimized strategy at near-zero selection
cost,"* not merely *"ties the paper's strategy."*

**A genuinely new mechanism, not just a reframing:** Algorithm H.1's
retrain-per-candidate loop is itself a (very expensive) form of
*interaction-aware* set optimization — each candidate query is scored in
the context of the set already chosen, inside the real classifier. This
is exactly the capability D009's own mechanism hypothesis (R2) says a
per-query tensor statistic (`S_energy[q][pair]`, aggregated by MAX) does
not have. It reframes what a promising next step would need to do: not
generate new queries (Phase 3, still paused, still resting on the same
grounds D008 left it — most "unidentifiable" pairs already have a good
pool query that selection missed), but find a **cheap, set-aware**
scoring method that approximates what Algorithm H.1 gets expensively —
directly testable on the existing frozen tensor and D009's own training
pipeline, no new corpus or GPU budget beyond what already exists.

**Also established:** the absolute accuracy ceiling here (0.8519,
CVaR/`k`=8) is not and should never be quoted against the paper's
95.35% — three-way split vs. the paper's two-way, 75 build configs vs.
150, 37 models vs. 52 (`METHOD.md §6.2`). Training itself is sound and
overfits identically across every condition by construction (train/val
gaps 0.10–0.15, nothing near chance, 160/160 runs converged) — D009's
INCONCLUSIVE branch does not apply. The supplementary open-set
evaluation (S8, not part of the verdict) shows the same ordering as
closed-set, paper's 8 slightly ahead of CVaR by one seed each — treated
as a tie, not read further; it does not test the open-set path's actual
selling point (rejecting a model never seen in training), which needs a
different held-out-model experiment.

## D010 — A cheap, classifier-free refinement reverses the paper-8 gap (mechanism confirmed, magnitude and one mechanism still open)

**Headline: a set-level (multivariate) statistic, still classifier-free,
beats the paper's own 8 queries at every `k`, and beats CVaR-coverage at
`k=8` — for half a minute of CPU.** D009 left a tie against the paper's
8 queries, explained by Algorithm H.1's retrain-per-candidate loop
(372 real training runs) implicitly capturing query-*interaction* that
CVaR-coverage's per-query, MAX-aggregated statistic (I4) cannot see by
construction. D010 tested a cheap, classifier-free way to capture the
same signal: energy distance on *concatenated* query-response
embeddings rather than per-query. Result: mean +0.026 over the paper's
8 across all 8 `k` values (8/8 same sign), +0.024 over CVaR-coverage at
`k=8` — reached without a single classifier retrained during selection.

**The mechanism is not just inferred, it's directly observed.** The
joint criterion's selected set overlaps only 2/8 with CVaR-coverage's
own chain, and it picked **one of the paper's own 8 anchor queries**
(the injection-wrapped "who created you") that CVaR-coverage's per-query
MAX never selects at any `k` across D008, D009, or D010. That is exactly
the object I4's aggregation rule cannot see by design: a query that's
unremarkable alone but complementary once combined with others. This is
concrete confirmation of D009/R2's hypothesis, not just a better number.

**Two things this D does not establish, both honestly reported rather
than glossed over.** (1) No single `k`'s delta clears the 5-seed
training-variance range — the case rests on consistency (8/8 sign
agreement, 5 of 8 CIs excluding zero), not on any one cell, and this is
the same 25-config `S_test` resolution wall D008 and D009 already hit —
now three D's running into the same limitation, escalated as a new open
item (`DECISIONS.md` A6: fixing it needs new test-set generation, a real
corpus-cost decision, not urgent since no verdict so far has needed it).
(2) The pre-registered prediction (small-`k` advantage, converging by
`k=8`) was only half right: the small-`k` half held, but the advantage
does *not* fade by `k=8` — it's among the largest gaps in the table
there. Why a statistic whose own selection objective and pilot both
predicted the benefit should concentrate at `k=2` keeps paying off all
the way to `k=8` is not explained by this D — flagged as the most
interesting open question the project currently has, not urgent, not
promoted to a new D yet.

**Verified, not assumed, throughout:** an exact (not approximate)
additive accumulator for the joint statistic — squared distances
decompose over concatenated queries, only the final square root doesn't
— made the naive-vs-approximate cost trade-off moot (519× faster,
6.51e-06 agreement with direct computation); a `k=1` sanity check
confirmed the joint statistic reduces bitwise-exactly to the per-query
one when there's no interaction to capture; a `k=8` MMD cross-check
(smaller magnitude, 5/8 same queries, same direction) confirmed the
result isn't an artifact of energy distance specifically.

## D011 — The cheapest D in the project retired one of its more-cited findings

**Set out to check whether `JointGreedy` recovers any of D008's flagged
pairs; found there was nothing to recover, and why.** Both `GreedyCover`'s
and `JointGreedy`'s actual 8-query chains resolve **all 32** of D008's
flagged pairs (32/32 each, ≥0.75 test accuracy, D008's own threshold).
D008's per-pair single-query "oracle" — the thing that flagged these
pairs as hard — scores a median of **0.54 on test, worse than a randomly
drawn pool query (0.82)**. Not merely optimistic: actively
anti-informative for this population. Offered as a hypothesis, not
proven: `argmax` over 259 noisy per-query build-time scores
preferentially surfaces queries whose apparent strength is a fluke that
doesn't transfer — the same winner's-curse class of error this project
has now caught three times in its own prior work (D001's censored
statistic, D008's own oracle inflation, and now D008's underlying T2
flag itself).

**A structural, not empirical, dead end for the question D011 actually
wanted to answer.** D011 was designed to distinguish "recovered because
one chosen query happens to be individually strong" from "recovered
because the *set* resolves it jointly, with no single member sufficing"
— the latter being the mechanism D010 exists to test. With every single
query in both chains already succeeding on all 32 pairs, there was no
case left where the second mechanism could even in principle show
itself. Not evidence against set-level interaction — evidence this
specific population was the wrong place to look for it.

**A near-miss worth keeping as a standing lesson, independent of the
substantive result.** The Review-approved fix to run the comparison on
D008's own accuracy scale (rather than an `S_energy` threshold) did not
just sharpen the numbers — it flipped the answer's sign. On the
original (wrong) scale, `JointGreedy` would have looked worse than
`GreedyCover` by 6 pairs (15/32 vs 21/32); on the corrected scale, the
two are tied at 32/32, with `JointGreedy` slightly ahead on mean
accuracy. Three D's in, the same lesson keeps recurring: a reporting
convention, not the underlying data, has repeatedly been what
determined a conclusion's direction.

**What this changes going forward:** D008's "identifiability frontier"
(the T2-derived 32-pair tier) should be retired, not cited as a
selection-algorithm gap or a Phase 3 motivation. `METHOD.md §7` still
lists "the identifiability frontier" as a standalone deliverable — on
current evidence that role belongs to the 2 structurally-hard pairs
tracked since D001 (`Falcon3-10B↔7B`, `Phi-3-medium-128k↔4k`), not the
T2 tier. Flagged for a `METHOD.md` revision, not yet made (human
decision). Phase 3's motivation is unaffected either way — the 2 real
hard pairs remain exactly where D007/D008 left them, untouched by this
D. D010's own real advantage (+0.024 mean top-1 at `k=8`) is confirmed
to come from somewhere other than this population — still unexplained,
same open item as D010/R9's k=8-persistence question.

## D012 — γ<1 is decisively required for JointGreedy — and not for the reason "CVaR helps" implies

**One of the cleanest, most decisive results this project has produced —
decisive as a coarse effect, not literally exceptionless.**
γ=1.0 (mean aggregation) loses to every `γ<1` value tested (0.25, 0.1,
0.05) across almost the entire `k=1..8` × four-metric grid, most gaps'
CIs excluding zero and exceeding the 5-seed training-variance range.
**Correction, 2026-09-22:** the original wording here and in D012's `## R`
headline said "every `k`, every metric, every CI excluding zero" without
qualification; D012/R3's own table already reported the true count —
1 of 24 mean-top-1 comparisons (γ=1.0 vs 0.25 at `k`=1) is INCONCLUSIVE,
not resolved, and that same cell's worst3-class and hard-subset deltas
also have CIs spanning zero. See the dated addendum in D012's `## Review`.
This does not reverse the finding — it is still true at 23 of 24 mean-top-1
cells and the overall pattern is large and consistent — but "every" was
false as literally written. D008, D009, D010, and D011 each hit a
resolution wall somewhere in their own comparisons; D012 hits it at only
one cell instead of pervasively. The pre-registered hypothesis going in
(that `JointGreedy`'s set-level interaction might already implicitly
favor hard pairs under mean-aggregation, making γ *less* necessary than
it was for `GreedyCover`) is not just wrong — γ matters *more* here than
D008's own weaker, proxy-only evidence ever showed it mattering for
`GreedyCover`.

**The mechanism is sharper than "CVaR-weighting helps," and matters for
how this gets written up.** γ=1.0's own selection objective — the thing
`JointGreedy` is directly maximizing at each greedy step — *falls*
monotonically as `k` grows (1.30 → 0.88), while the *trained accuracy*
of the exact same selected queries *rises* (0.54 → 0.80) over the same
range. The objective and the outcome move in opposite directions across
the entire range. This could mean either "the objective just mis-scores
otherwise-fine queries" or "the objective is actively selecting worse
queries" — distinguished by checking the *training*-set accuracy itself
(not just held-out accuracy): γ=1.0's selected queries fit their own
training data worse (0.939 vs. 0.992 for every `γ<1`), meaning they
carry genuinely less usable signal, not merely a differently-distributed
version of the same signal. **CVaR-weighting is not improving a working
selection process here — it is repairing one that is degenerate past
`k`=1** (mean-aggregation's raw statistic gets dimension-dominated by
each additional query's 1024-d contribution, a variant of the same
curse-of-dimensionality risk D010's own pilot first flagged).

**Within `γ<1`, there is no further ordering to find** — 23 of 24
pairwise comparisons among {0.25, 0.1, 0.05} are either statistically
equivalent (a pre-registered TOST equivalence test, not just "failed to
find a difference") or inconsistent in direction across `k`. At `k`=8
all three select the *identical* 8-query set (differing only in slot
order) — which incidentally provided a free, previously-unchecked
confirmation that the trained network is insensitive to query-slot
order (spread 40× smaller than seed noise). `C1` (the CVaR tail
parameter) is now closed for `JointGreedy`: γ<1 required, γ=0.1
recommended as the default on precedent (not because the data
distinguish it from 0.25 or 0.05).

**What's still open, cheaply:** `GreedyCover`'s own γ-sensitivity has
only ever been measured via D008's untrained proxy — training its γ
variants (~5 minutes of compute) is the only way to properly compare
how much *each* algorithm actually needs CVaR-weighting, if that
comparison is ever wanted for a writeup.

## D013 — GreedyCover needs a small γ just as much — which breaks the tidy explanation D012 offered

**Headline: the same γ requirement holds for the other algorithm.**
Trained on the identical protocol, `GreedyCover` at γ=1.0 (plain mean)
loses to γ ≤ 0.25 on mean top-1 at most or all `k`, at the same size as
`JointGreedy` (mean penalty −0.081 vs. −0.079; no detectable difference
between the algorithms). γ=0.1 stays the default; 0.25/0.1/0.05 are not
distinguishable. `C1` is closed for both.

**The interesting part is what it does to D012's explanation.** D012
found `JointGreedy`'s γ=1.0 objective *falls* as queries are added while
its accuracy rises, and concluded CVaR was "repairing an objective that
fails outright." `GreedyCover`'s objective does not fall — I4's MAX
aggregation makes it monotone by construction — and γ=1.0 *still* loses
by the same margin. So that mechanism is true of `JointGreedy` but is
not why γ matters in general; nothing is broken in `GreedyCover` to
repair. **Why a small γ improves selection independent of any objective
pathology is now the project's most interesting unexplained question.**
(A pre-registered prediction that γ would help `GreedyCover` only
narrowly — worst-class, small `k` — was wrong; the structural mechanism
it was built on was right.)

**A γ-shaped caution, held loosely.** γ=0.5 gave `GreedyCover` no benefit
over γ=1.0 — reproducing under training an anomaly D008's proxy had shown
(and that had been dismissed as noise). That is enough to stop treating γ
as a clean continuous dial and to make the rule "γ ≤ 0.25" rather than
"γ < 1." It is *not* yet evidence of a threshold in γ: it is one point,
the proxy and trained anomalies are the same nested chain evaluated twice,
and chains here are composition-unstable. Whether it is a property of γ
or of that chain is untested.

## D014 — γ is a graded, non-linear dial for JointGreedy; for GreedyCover the question is still open

**`JointGreedy`: the γ effect is real and graded, not a chain accident.**
Resampling the build configs 30 times and re-selecting the chains, γ=0.25
beats γ=0.5 by +0.085 (`k`=4) and +0.037 (`k`=8) on the proxy, with CIs far
from zero and a control (γ=0.25 vs 1.0, ≈ +0.10) that reproduces the known
gap. The trained grid shows benefit switching on over a band of γ — no
single cutoff — and flat below ≈0.4 on the full data. A pre-registered
lean toward "γ=0.5's failure is chain luck" was wrong for this algorithm.

**`GreedyCover`: inconclusive, and the failure is informative.** The same
arm failed its positive control at `k`=4 (γ=0.25 vs 1.0: +0.0125,
CI [−0.003, +0.029]). Two readings fit: the instrument is less sensitive
for this algorithm, or its small-`k` γ effect is genuinely smaller once the
selection data is perturbed than on the frozen chains (≈ +0.09 trained).
The same machinery gives `JointGreedy` +0.107, which weighs somewhat toward
the second. Descriptively (post hoc, a lead not a result): at `k`=4 its
resampled medians are flat from γ=1.0 to 0.25 and only rise at γ=0.1; at
`k`=8 they rise steadily with γ=0.5 between its neighbours. If it holds, it
matters for A4 (small-`k` efficiency is the primary claim) and leans toward
`JointGreedy` as the recommended algorithm, and it qualifies D013's "same
γ-sensitivity" — true on frozen full-data chains only.

**Method lessons that paid for themselves.** Chains recur as *families*
across neighbouring γ (γ=0.6 and 0.5 give the same `k`=8 set; γ ∈ {0.45,
0.4, 0.35} likewise), so the trained grid alone showed `GreedyCover` a clean
STEP made of three copies of one chain — declaring Arm B decisive and Arm A
descriptive prevented that. A positive control caught the uninformative arm.
The proxy was confirmed (20 trained runs) to rank chains correctly *within*
a γ, which D009 had never tested. One caution: a failed control is ambiguous
between blindness and a genuinely smaller effect — the rule had assumed the
first.

**Consequence for the method.** `METHOD.md §5.2`'s "γ is a clean continuous
ablation axis" should go: the axis is not linear, its onset moves with how
much data selection sees, and `GreedyCover`'s fine structure is not
established. Replacement wording is proposed and awaits the human.

## D015 — The `hard_subset` gain hides real backward movement on a third of the pairs

**A tiny aggregate number was masking a redistribution, and it was real.**
Breaking `hard_subset`'s single averaged number into its 65 pair-level
components (joint energy vs. coverage, `k`=8) shows the reported +0.41pp
aggregate gain is actually **28 pairs improving, 20 getting worse, 17
unchanged** — gross movement (+0.544/−0.276) three times the net. `coverage`
vs. `paper8` is the same shape (28 up/21 down/16 flat). **A metric that reads
as "improved" can have a third of its structural near-relative pairs doing
worse** — a genuine finding about what an unweighted 65-pair mean can hide,
not a defect in the selection methods themselves.

**Pre-registered before any table existed (P1/F3): "spread vs. concentrated"
is unanswerable at this resolution, and it was right.** The effect being
decomposed is worth only ~13 individual trace-flips across 3,250 decisions —
no single pair can move by less than 2pp, an order of magnitude coarser than
the aggregate. The redistribution question ("does any pair get worse") stayed
answerable and is what this D actually delivers; "most pairs improved a
little" was correctly flagged as an artifact the instrument cannot render,
before it could be mistaken for a result.

**Independent corroboration for the two previously-named hard pairs:**
`Falcon3-10B/7B` and `Phi-3-medium-128k/4k` rank 1st and 2nd hardest under
`paper8`, `coverage`, **and** `joint energy` independently — three different
query sets agree, plus two sources outside this corpus entirely (D001's
AUC statistic on the paper's own shipped artifact; D008's tensor oracle).
Neither is a fine-tune derivation pair — both have an empty `base` field;
they are same-generation siblings released by their org, not parent/child.

**A correction, caught in post-hoc review, worth recording as a process
lesson.** `## R`'s original headline example — "the project's named hardest
pair is 11.6pp worse under coverage than paper8" — attached that resolved
delta to the wrong pair. It belongs to a different, third-ranked pair
(`Phi-3-medium-128k` vs. `Phi-3-mini-4k`, cross-size, not the same-size
sibling pair named since D001); the actually-named-hardest pair's own
coverage-vs-paper8 delta (−0.056) does not resolve. A prose transcription
slip, not a data or join bug — verified against the raw per-pair JSON, all
other headline numbers reproduced exactly on independent recomputation. The
corrected, still-real claim: `coverage` is resolvably worse than `paper8` on
6 of 65 structural pairs at `k`=8, just not on the specific pair this project
has repeatedly pointed to as hardest.

**Consequence for future reporting.** Any writeup quoting `hard_subset`
should quote this redistribution alongside it — the number alone overclaims
uniform improvement. `k`=1 comparisons between `coverage` and `joint energy`
carry no information (they select the identical single query by
construction, D010/R2) — a reporting trap this D and D013/F3 have now both
hit independently; worth a standing caveat if `k`=1 is ever tabulated again.

## Reference — aggregate accuracy across k=1..8, γ=0.1 (design-side consolidation, 2026-09-22)

**Not a new D — a reorganization of already-published per-k means** from
D009 (`paper8`, `cvar_max`=coverage), D010 (`joint_energy`), reproduced by
independent recomputation from `results/D009/runs.json` and
`results/D010/metrics_by_k.json`. Requested by the human to see which
metric actually improves and by how much, across the full budget range.

**mean top-1 is the only metric that is positive at every single k.**
`joint energy` beats `paper8` at all 8 k values, +1.45pp to +4.74pp
(mean ≈ +2.7pp), at the project's headline k=8: **.8452 → .8757, +3.05pp**.
Against `coverage`, the lead is positive at 6 of 8 k (two near-zero dips at
k=4/6, −0.19pp/−0.24pp, both far inside the 5-seed run range — selection
noise, not a reversal): **.8519 → .8757 at k=8, +2.38pp**.

| k | paper8 | coverage(γ=.1) | joint energy(γ=.1) | joint−paper8 | joint−coverage |
|---|---:|---:|---:|---:|---:|
| 1 | .6093 | .6333 | .6333 | +2.40pp | 0 |
| 2 | .7395 | .7585 | .7725 | +3.30pp | +1.40pp |
| 3 | .7602 | .7816 | .8076 | +4.74pp | +2.60pp |
| 4 | .8052 | .8231 | .8212 | +1.60pp | −0.19pp |
| 5 | .8050 | .8102 | .8301 | +2.51pp | +1.99pp |
| 6 | .8294 | .8463 | .8439 | +1.45pp | −0.24pp |
| 7 | .8461 | .8623 | .8672 | +2.11pp | +0.49pp |
| 8 | .8452 | .8519 | .8757 | +3.05pp | +2.38pp |

**worst-class and hard-subset do not support a clean headline number.**
worst-class swings from −5.6pp to +20.0pp against `paper8` depending on k,
with within-condition 5-seed ranges (0.16–0.36) larger than most of the
deltas — sign flips at k=4/5/6 are noise, not a real dip, per this
project's standing worst-class caution (A6, 25 test configs). hard-subset
moves by at most +0.5pp at any k — real but small, and D015 already showed
even that small aggregate move hides real per-pair redistribution
(28 up/20 down at k=8).

**For any future "how much did the method improve" question: quote mean
top-1, at the k of interest, against both `paper8` and `coverage` — not
worst-class (too noisy) and not hard-subset alone (too compressed, and
masks redistribution per D015).**

## Design-side check — does proxy separability track real classifier error? (2026-09-22)

**The human's question:** if low-separability pairs (what the tensor/CVaR
objective targets) do not correspond to high-error pairs (what actually
matters), shrinking γ further just optimizes a misaligned proxy harder.
Checked from already-computed results — **no new tensor computation was
run**; this cross-references two already-published per-pair files:
`results/D008/hard_subset_ceiling.json` (per-pair proxy accuracy over the
candidate pool, computed from the frozen `S_energy` tensor: `median_query` =
median 2-way test accuracy over all 259 queries for that pair; `best_build_query`
= test accuracy of the tensor's own argmax-scored query for that pair;
`oracle` = best-of-259 on test, **optimistic by construction**, D008's own
docstring — treated as the least trustworthy of the three) against
`results/D015/per_pair_k8.json`'s trained per-pair accuracy, both already
computed and reviewed. A Spearman rank correlation across the 65 structural
pairs (not a new statistic from raw tensor/corpus data — an off-the-shelf
correlation between two already-existing derived files).

**Moderate-to-strong positive correlation, strongest exactly where CVaR
actually looks.** `median_query` (proxy) vs. trained `paper8` accuracy:
**ρ = 0.81**; vs. trained `coverage`: 0.76; vs. trained `joint energy`: 0.85.
`best_build_query` (the tensor's own top pick) correlates more weakly
(0.68–0.72) — the pair's *typical* separability across the pool tracks real
difficulty better than the tensor's single favorite query does. **At the
tail — where CVaR/γ actually operates — the agreement is close to exact**:
the proxy's 4 hardest pairs by `median_query` (Phi-3-medium-128k/4k,
Mistral-v0.2/v0.3, Phi-3-mini-128k/4k, Falcon3-10B/7B) are the same 4 pairs
in the trained-`paper8` top-4 hardest, just reordered.

**But mid-ranking agreement is noticeably weaker, in both directions** —
this is the part that bears on "would shrinking γ further help or just
chase proxy noise." Two named examples: `gemma-1.1-2b↔gemma-2b` ranks
12th-hardest by the proxy but only 41st by trained accuracy (proxy
*overstates* its difficulty); `Phi-3-medium-4k↔Phi-3.5-mini-instruct` ranks
8th-hardest when trained but only 30th by the proxy (proxy *understates*
it). Several more pairs disagree by 20+ ranks out of 65 in the middle of
the distribution.

**Reading for C1/γ:** this is reassuring for the *current* default (γ=0.1
already concentrates on a small tail where the proxy-target correlation is
close to exact) but is a real caution against pushing γ substantially
smaller expecting the same alignment to hold — CVaR would start reaching
into the middle-ranked pairs, exactly where this check finds the proxy and
the trained classifier disagree most. **Not a green light or a red light
for "go smaller" — a specific, named risk if it is tried**, with the
concrete failure mode (which pairs would be mis-prioritized) now visible
rather than hypothetical.

**Limits, stated plainly:** n=65 (the structural set only, not all 666
pairs); one specific pair of proxy statistics (`median_query`,
`best_build_query`), not the literal CVaR-tail objective value the
selection algorithms actually optimize; a single k reference point
(k=8's trained numbers) against an aggregate-over-259-queries proxy, not a
matched-budget comparison; no significance test on the correlation itself
(n=65 rank correlation, not bootstrapped). **A rigorous version — the
actual tensor separability statistic vs. trained error, across all 666
pairs, with a proper interval — would require reading the raw `S_energy`
tensor directly and is TACC's work, not design-side's**; this check used
only already-published per-pair result files. Flagged as a candidate D if
the human wants the authoritative version before relying on this for a
γ decision.

## Design-side check — reformulating hard_subset: worst-pair, tail-CVaR, and hard-model-restricted top-1 (2026-09-22)

**Motivation.** The two-logit-restricted `hard_subset` mean (METHOD §8) is
neither the worst pair, a tail average, nor the true 37-way accuracy on the
models that are actually hard — all three flaws were named in
`docs/REVIEW-2026-09-22-hard-pairs.md` §8 and confirmed with concrete
numbers the same day. The human asked whether the metrics that fix each
flaw can be computed from already-run data. **Yes, all three, at k=8, from
`results/D015/per_pair_k8.json` and the already-stored `per_model` arrays
in `results/D009/runs.json`/`results/D010/metrics_by_k.json` — no new
tensor read, no new training, no new selection.**

**(1) Worst pair — min instead of mean, same 2-way-restricted values D015
already has.** Averaging over 65 pairs was hiding a real, sizeable spread:

| | worst pair (min of 65) | tail mean, γ=0.10 (worst 6) | tail mean, γ=0.25 (worst 16) | full mean (65) |
|---|---:|---:|---:|---:|
| paper8 | .712 | .844 | .900 | .964 |
| coverage | **.656** | **.818** | **.895** | .964 |
| joint energy | **.732** | **.853** | **.911** | .968 |

At the worst pair, `coverage` is *worse than paper8* (−5.6pp) while `joint
energy` is best (+2.0pp over paper8) — a **7.6pp spread between methods**,
15–19× the ≈0.4–0.5pp the flat `hard_subset` mean showed. The tail means
(γ=0.10/0.25) tell the same story at smaller magnitude. This ranking —
joint energy > paper8 > coverage — was invisible in the aggregate.

**(2) True 37-way top-1 accuracy, restricted to the models that are
actually hard, not a 2-way-restricted proxy.** Using `per_model` (already
stored per run, never before read this way): restricting to the 10 models
in the 6 hardest structural pairs (by `paper8`, k=8) —

| | hard-models mean (n=10) | all-37 mean_top1 | gap |
|---|---:|---:|---:|
| paper8 | .720 | .845 | **−12.5pp** |
| coverage | .750 | .852 | −10.2pp |
| joint energy | **.803** | .876 | **−7.3pp** |

**The gap between hard models and the population shrinks as the method
improves (−12.5 → −10.2 → −7.3pp), and the improvement on hard models
alone (paper8→joint: +8.3pp) is larger than the overall improvement
(+3.0pp).** The method's real benefit is concentrated on the hard cases —
a defensible, quantified version of "helps with hard pairs" that
`hard_subset` could never show, because it never measured true deployment
(37-way) accuracy at all. Checked for robustness with a looser cutoff (21
models from the 16 hardest pairs, γ=0.25): same direction, smaller
magnitude (gap −5.6 → −5.4 → −5.1pp) — the effect is real and concentrates
most strongly in the genuinely hardest tier, not an artifact of one cutoff
choice.

**Not yet done, and would need a fresh reason to prioritize:** the k=1
version; backfilling this for D012/D013's γ grid and D014's resampled
arms; extending "hard models" to the full 33-of-37 models touching *any*
near-relative pair (too inclusive to be useful — nearly the whole
population, by I1's own design). None of these need new experiments
either; they are the same read-only extraction, just not yet run.

**Consequence for METHOD.md §8 — flagged, not applied.** This suggests
`hard_subset` should be supplemented or replaced by (a) a worst-pair or
tail-CVaR statistic over the 65 pairs, and/or (b) true top-1 restricted to
the hard-pair models, rather than the current two-logit-restricted mean.
This is a METHOD revision (`CLAUDE.md` rule (d)) and needs the human's
explicit sign-off before `METHOD.md §8`'s table changes.

<!-- append new entries below, one per D, once it produces a project-level
     takeaway worth remembering outside its own file -->
