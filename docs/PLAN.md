# PLAN.md — D Index

> The single index. **TACC reads this file to find its next task** — not
> `CLAUDE.md`, not chat history. Design-side keeps this current as D's open,
> run, and close (`CLAUDE.md`).

| id | source | status | gate | deps | result pointer |
|---|---|---|---|---|---|
| [D001](D001.md) | TODO T0.1, T0.2 | **CLOSED** 2026-09-02 — R6 answered by D004: TAIL CONFIRMED | REVIEW (P1 approved w/ amendments, executed) | none | `results/D001/`, `D001.md` `## R` |
| [D002](D002.md) | follow-up from D001/P1 §F5 | **CLOSED** 2026-09-02 — GPU_AVAILABLE, own question fully answered | REVIEW (P1 approved, executed) | none | `results/D002/`, `D002.md` `## R` |
| [D003](D003.md) | ADDENDUM.md item 2 | **CLOSED** 2026-09-02 — READY, 233-entry candidate pool (93% of revised ~250 target) | REVIEW (P1 approved w/ amendments, executed) | none directly; its *consumer* (Phase 1 corpus build, D006) depends on A1 (✅ decided 2026-09-05) and C3 (✅ ~250) | `confs/queries/pool_d003_candidates.json`, `results/D003/`, `D003.md` `## R` |
| [D004](D004.md) | D001 §R6, human-approved 2026-09-01 | **CLOSED** 2026-09-02 — TAIL CONFIRMED (CVaR₀.₁/mean=0.616 on the uncensored statistic); `METHOD.md §2`/I1 stand | REVIEW (P1 approved, executed) | none (pure re-analysis of D001's cached embeddings) | `results/D004/`, `D004.md` `## R` |
| [D005](D005.md) | I2 bug found while preparing D004 (TACC note on D001's `## Review`, 2026-09-02) | **CLOSED** 2026-09-04 — FIXED (M1+M2+M3 verified via regression test); shipped-dataset exposure UNDETERMINABLE (M4), no effect on D001/D004's numbers | REVIEW (P1 approved, executed) | none | `results/D005/`, `D005.md` `## R` |
| [D006](D006.md) | TODO.md Phase 1 (T1.1–T1.4) + D002 §R4 + D003 + D005; A1/C4/C7 decided 2026-09-05/06 | **OPEN — S1–S7 complete, PARTIAL at 29/37.** 8 shards blocked on `GatedRepoError` 403 (license acceptance, human-only action — see `D006.md` §R8); resumes automatically once accepted (submitter is idempotent). S8 (final writeup) pending 37/37. | REVIEW (P1 approved, executing) | none — A1/C3/C4/C7/D005 all satisfied; **blocked on human: accept gated-repo licenses** | `results/D006/`, `docs/plans/D006-P1.md`; `D006.md` `## R` (incremental) |
| [D007](D007.md) | TODO.md Phase 1 (T1.5–T1.6); A3 decided 2026-09-06 | **OPEN** — drafted 2026-09-06, awaiting a P | REVIEW | **hard dependency: D006 must reach READY (37/37), not PARTIAL** — do not start S1 while D006 is still blocked on licenses | `D007.md` `## P`/`## R` pending |

## Not yet a D

- **The actual query-strategy comparison (TODO.md T2.1–T2.4: mean-greedy
  vs. CVaR-coverage, the γ sweep, the identifiability frontier)** —
  blocked on D007 landing (`TAIL CONFIRMED` or better). Per TODO.md's own
  framing this phase is "post-processing on the frozen tensor... seconds
  per run" once D007 closes — the expensive part is D006+D007, not this.

## Notes

- TACC executes only `status: OPEN` D's whose `gate` is satisfied — `AUTO`
  runs immediately, `REVIEW` needs a posted `## Review` with an APPROVED
  verdict. D006's approval is partial by design (see its Review) — TACC
  can act on the approved parts now without waiting for the rest. D007
  has no `## P` yet — TACC should write one, but **must confirm D006 is
  READY first** (D007's own S1) rather than starting on a PARTIAL corpus.
- "Closed" is a design-side call made after `## R` is posted **and** any
  R-driven open questions are resolved for the D's *own* question. D001–D005
  are all closed as of 2026-09-04 — each answered its own question in full,
  even where (D001/D002) the answer spawned a still-open A-series item that
  outlives the D itself.
