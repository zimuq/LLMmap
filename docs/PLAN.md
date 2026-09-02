# PLAN.md — D Index

> The single index. **TACC reads this file to find its next task** — not
> `CLAUDE.md`, not chat history. Design-side keeps this current as D's open,
> run, and close (`CLAUDE.md`).

| id | source | status | gate | deps | result pointer |
|---|---|---|---|---|---|
| [D001](D001.md) | TODO T0.1, T0.2 | OPEN — R posted, held open pending D004's answer to R6 (see D001 `## Review`) | REVIEW (P1 approved w/ amendments) | none | `results/D001/`, `D001.md` `## R` |
| [D002](D002.md) | follow-up from D001/P1 §F5 | OPEN — P1 approved, awaiting TACC execution | REVIEW (P1 approved) | none | `D002.md` `## P`/`## Review`; `## R` pending |
| [D003](D003.md) | ADDENDUM.md item 2 | OPEN — drafted, awaiting a P | REVIEW | none directly; its *consumer* (Phase 1 corpus build) depends on A1 (D001) and C3 (D002) | `D003.md` |
| [D004](D004.md) | D001 §R6, human-approved 2026-09-01 | OPEN — drafted, awaiting a P | REVIEW | none for execution (reuses D001's cached embeddings if available); Phase 1's full-scale corpus commitment should wait on its answer | `D004.md` |

## Notes

- TACC executes only `status: OPEN` D's whose `gate` is satisfied — `AUTO`
  runs immediately, `REVIEW` needs a posted `## Review` with an APPROVED
  verdict (D001 and D002 both have one now; D003 and D004 do not yet have a
  `## P` to review).
- "Closed" is a design-side call made after `## R` is posted **and** any
  R-driven open questions are resolved — see D001, which has results but
  stays open because of R6.
