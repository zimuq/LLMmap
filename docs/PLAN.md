# PLAN.md — D Index

> The single index. **TACC reads this file to find its next task** — not
> `CLAUDE.md`, not chat history. Design-side keeps this current as D's open,
> run, and close (`CLAUDE.md`).

| id | source | status | gate | deps | result pointer |
|---|---|---|---|---|---|
| [D001](D001.md) | TODO T0.1, T0.2 | OPEN — R posted, held open pending D004's answer to R6 (see D001 `## Review`) | REVIEW (P1 approved w/ amendments) | none | `results/D001/`, `D001.md` `## R` |
| [D002](D002.md) | follow-up from D001/P1 §F5 | **CLOSED** 2026-09-02 — GPU_AVAILABLE, own question fully answered | REVIEW (P1 approved, executed) | none | `results/D002/`, `D002.md` `## R` |
| [D003](D003.md) | ADDENDUM.md item 2 | OPEN — P1 approved w/ amendments (generator → OLMo-2-13B-Instruct); C3 target revised ~100→~250 2026-09-02 via Review addendum, no new P needed, awaiting TACC execution | REVIEW (P1 approved) | none directly; its *consumer* (Phase 1 corpus build) depends on A1 (D001) and C3 (now ✅ ~250) | `D003.md` `## P`/`## Review`; `## R` pending |
| [D004](D004.md) | D001 §R6, human-approved 2026-09-01 | OPEN — P1 approved, awaiting TACC execution | REVIEW (P1 approved) | none (pure re-analysis of D001's cached embeddings, confirmed intact on `$WORK`) | `D004.md` `## P`/`## Review`; `## R` pending |
| [D005](D005.md) | I2 bug found while preparing D004 (TACC note on D001's `## Review`, 2026-09-02) | OPEN — drafted, awaiting a P | REVIEW | none for execution; is a **hard prerequisite** for whichever D builds the real Phase-1 corpus | `D005.md` |

## Not yet a D

- **Phase-1 corpus construction itself** — needs A1 (D001, still open),
  C3 sizing (informed by D002, closed — cost is not the binding factor),
  D003's finished pool, and **D005's fix landed first** (I2 must actually
  hold before any real build/val/test split is trusted). Also inherits
  D002 R4's recommendation to fix `dataset_maker.py`'s batch-1 generation
  loop before this D runs — not yet its own action item, noted here so it
  isn't lost before that D exists.

## Notes

- TACC executes only `status: OPEN` D's whose `gate` is satisfied — `AUTO`
  runs immediately, `REVIEW` needs a posted `## Review` with an APPROVED
  verdict. D002, D003, D004 all have one now (D002's executed and closed);
  D005 does not yet have a `## P` to review.
- "Closed" is a design-side call made after `## R` is posted **and** any
  R-driven open questions are resolved for the D's *own* question — see
  D001 (stays open, its own tail-existence question is unresolved pending
  D004) vs. D002 (closed — its own GPU/cost question was fully answered;
  the 70B-memory finding it surfaced became a new A1 consideration, not a
  reason to keep D002 itself open).
