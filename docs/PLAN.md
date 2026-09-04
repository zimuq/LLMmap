# PLAN.md — D Index

> The single index. **TACC reads this file to find its next task** — not
> `CLAUDE.md`, not chat history. Design-side keeps this current as D's open,
> run, and close (`CLAUDE.md`).

| id | source | status | gate | deps | result pointer |
|---|---|---|---|---|---|
| [D001](D001.md) | TODO T0.1, T0.2 | **CLOSED** 2026-09-02 — R6 answered by D004: TAIL CONFIRMED | REVIEW (P1 approved w/ amendments, executed) | none | `results/D001/`, `D001.md` `## R` |
| [D002](D002.md) | follow-up from D001/P1 §F5 | **CLOSED** 2026-09-02 — GPU_AVAILABLE, own question fully answered | REVIEW (P1 approved, executed) | none | `results/D002/`, `D002.md` `## R` |
| [D003](D003.md) | ADDENDUM.md item 2 | **CLOSED** 2026-09-02 — READY, 233-entry candidate pool (93% of revised ~250 target) | REVIEW (P1 approved w/ amendments, executed) | none directly; its *consumer* (Phase 1 corpus build) depends on A1 (D001, closed but A1 itself still open) and C3 (now ✅ ~250) | `confs/queries/pool_d003_candidates.json`, `results/D003/`, `D003.md` `## R` |
| [D004](D004.md) | D001 §R6, human-approved 2026-09-01 | **CLOSED** 2026-09-02 — TAIL CONFIRMED (CVaR₀.₁/mean=0.616 on the uncensored statistic); `METHOD.md §2`/I1 stand | REVIEW (P1 approved, executed) | none (pure re-analysis of D001's cached embeddings) | `results/D004/`, `D004.md` `## R` |
| [D005](D005.md) | I2 bug found while preparing D004 (TACC note on D001's `## Review`, 2026-09-02) | **CLOSED** 2026-09-04 — FIXED (M1+M2+M3 verified via regression test); shipped-dataset exposure UNDETERMINABLE (M4), no effect on D001/D004's numbers | REVIEW (P1 approved, executed) | none | `results/D005/`, `D005.md` `## R` |

## Not yet a D

- **Phase-1 corpus construction itself** — needs A1 (still open — D001's
  ≤14B-subset findings are in, but the universe itself isn't chosen; D004's
  result sharpens the 70B question, see A1 note; 70B sub-question resolved
  2026-09-02, deferred). C3 (✅ closed at ~250), D003's finished pool (✅
  delivered, 233 entries), and D005's I2 fix (✅ landed 2026-09-04) are all
  now in place — **the only remaining blocker is A1 itself.** Also inherits
  D002 R4's recommendation to fix `dataset_maker.py`'s batch-1 generation
  loop before this D runs — TACC's 2026-09-04 report reframes this as
  mandatory rather than optional (Vista's `gh` partition caps a single job
  at 2 days; single-node unbatched/under-parallelized generation for the
  ~250×25×125-scale corpus doesn't fit inside that cap at all, so sharding
  across ~20 concurrent jobs is required, not a speed optimization) — needs
  to be folded into this D's scope once drafted, along with an explicit
  design point on how per-model sharded output gets reassembled into one
  I6/I7-compliant frozen corpus.

## Notes

- TACC executes only `status: OPEN` D's whose `gate` is satisfied — `AUTO`
  runs immediately, `REVIEW` needs a posted `## Review` with an APPROVED
  verdict.
- "Closed" is a design-side call made after `## R` is posted **and** any
  R-driven open questions are resolved for the D's *own* question. D001–D005
  are all closed as of 2026-09-04 — each answered its own question in full,
  even where (D001/D002) the answer spawned a still-open A-series item that
  outlives the D itself.
