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
invariant I1 comfortably (65 near-relative pairs across 37 models), but the
strongest individual near-relative pairs — including the project's own
motivating case — need a 70B-class partner. Recommendation on record:
~25×≤14B core + one 70B pair (Llama-3-70B + Smaug-70B) — **update
2026-09-02: D002 found this is a memory problem, not a budget problem (the
70B pair doesn't fit on one GH200 at full precision). Still an open A1
call, now with a real tradeoff attached — see [D002](D002.md) and
`DECISIONS.md` A1.**

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

<!-- append new entries below, one per D, once it produces a project-level
     takeaway worth remembering outside its own file -->
