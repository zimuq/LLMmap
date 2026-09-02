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
| [D005](D005.md) | I2 bug found while preparing D004 (TACC note on D001's `## Review`, 2026-09-02) | OPEN — P1 posted; M1/M3/Call A approved and TACC may proceed on those now; **Call B escalated to human, `DECISIONS.md` A5** — M2/S2 blocked until it returns | REVIEW (P1 partially approved) | none for execution; is a **hard prerequisite** for whichever D builds the real Phase-1 corpus | `docs/plans/D005-P1.md`; `## R` pending |

## Not yet a D

- **Phase-1 corpus construction itself** — needs A1 (still open — D001's
  ≤14B-subset findings are in, but the universe itself isn't chosen; D004's
  result sharpens the 70B question, see A1 note), C3 (✅ closed at ~250),
  D003's finished pool (✅ delivered, 233 entries), and **D005's fix landed
  first** (I2 must actually hold before any real build/val/test split is
  trusted — currently blocked on A5). Also inherits D002 R4's
  recommendation to fix `dataset_maker.py`'s batch-1 generation loop before
  this D runs — not yet its own action item, noted here so it isn't lost
  before that D exists.

## Notes

- TACC executes only `status: OPEN` D's whose `gate` is satisfied — `AUTO`
  runs immediately, `REVIEW` needs a posted `## Review` with an APPROVED
  verdict. D005 has a partial approval (M1/M3/Call A clear now; M2/S2 wait
  on A5).
- "Closed" is a design-side call made after `## R` is posted **and** any
  R-driven open questions are resolved for the D's *own* question. D001,
  D002, D003, D004 are all closed as of 2026-09-02 — each answered its own
  question in full, even where (D001/D002) the answer spawned a still-open
  A-series item that outlives the D itself.
