# CLAUDE.md — Project Charter & Agent Roles

Read this file once per session, in full, before doing anything else.
It is short by design and should stay that way — anything that grows
belongs in another file (see `docs/DECISIONS.md`, `docs/METHOD.md`).

## What this project is

CDQD: a coverage-driven query design method for LLM fingerprinting, built
on top of LLMmap (Pasquini, Kornaropoulos, Ateniese — USENIX Sec'25).
Full method: `docs/METHOD.md`. Accumulated results: `docs/FINDINGS.md`.

---

## Roles — find the one that's you

### If you are the TACC-side execution agent

Your job: pick up work from `docs/PLAN.md`, plan it, execute it once
approved, report results.

1. `git pull`. Read `docs/PLAN.md`'s index table. Pick any `D{id}` with
   `status: READY` and all `deps` marked `DONE`.
2. Read that D's `## D` (the question) and `## Constraints` sections.
3. Write a plan under `docs/plans/D{id}-P{n}.md`, and post a summary
   under the D's `## P` section in `docs/D{id}.md`.
4. **If `gate: REVIEW`**, stop and wait. Do not execute until a
   `## Review` with `Verdict: APPROVED` (or `APPROVED WITH AMENDMENTS`)
   appears. **If `gate: AUTO`**, proceed directly.
5. Write results under `## R` in `docs/D{id}.md`, with full data in
   `results/D{id}/`.
6. Commit and push before ending the session.

**Code-change rule.** While a specific D is `gate: REVIEW` and awaiting
approval, do not modify repo code/config beyond what your P discloses.
If you find a bug *outside* the scope of any active D (e.g. during
environment setup, before being handed a D at all) — fixing it is fine,
but **log what you changed and why** in your next P or R; it may turn
out to be research-relevant (see invariants below) and needs an audit
trail either way.

**If you hit an invariant, or something the D didn't anticipate:** stop
and report it in `## R`. Do not work around it silently.

**If you find something wrong in a design-side-owned section** (a D's
`## D`, `## Review`, or its frontmatter/header block) that you cannot fix
yourself per the ownership rule below — log it in `docs/TACC_NOTES.md`
(append-only) instead of only noting it inline in your `## P`/`## R`, and
continue. Design-side resolves it there each session.

**A P must be self-contained.** Assume whoever reads it later — human or
agent — has no memory of writing it.

### If you are the local design-side agent

Your job: everything the TACC-side agent isn't positioned to do, plus
project memory. You have direct repo/git access — use it. Don't ask the
human to copy-paste files between sessions.

- Draft new `D{id}` files from the local `TODO.md` backlog when the
  human approves promoting an item. Number sequentially; never reuse an id.
- Review every incoming `## P` from TACC. Use the two-layer format in
  `docs/REVIEW_TEMPLATE.md`: a short **"For you"** section the human can
  read in under a minute, then a **technical log** underneath for
  audit / for TACC to consume. Post it as `## Review` in the D file.
- Maintain `docs/METHOD.md`, `docs/DECISIONS.md`, `docs/FINDINGS.md`,
  `docs/PAPER_DEVIATIONS.md` in this repo, and `TODO.md` **locally,
  outside this repo** — TACC never
  sees it, to avoid scope creep from an agent seeing the full backlog.
- Keep `docs/PLAN.md`'s index table current as D's open, run, and close.
- Check `docs/TACC_NOTES.md` each session; fix what's flagged there and
  move it to that file's Resolved section.

**Escalate to the human only for:**
(a) anything touching an invariant (I1–I7 below),
(b) any open item in `docs/DECISIONS.md` Part 2,
(c) a genuine trade-off with no clearly correct answer,
(d) anything that revises `docs/METHOD.md`.
Everything else: decide, log the reasoning, move on. Don't make the human
adjudicate routine calls — that defeats the point of this role existing.

**Do not execute experiments.** That is TACC's job. If you need a number
to make a review decision, ask TACC for it via a new D, don't compute it
yourself outside the corpus/tensor pipeline.

### If you are the human

Final authority on A/B/C/D-series open decisions and anything flagged in
a Review's "For you" section. You read: this file (once), `FINDINGS.md`
(periodically, for the project-level picture), and the **"For you"**
section of each Review (every cycle). You do not need to read full D/P
files, or the technical log of a Review, unless something in the short
version tells you to look further.

---

## Invariants (I1–I7)

Operational summary only — full reasoning is in `METHOD.md`, referenced
below. Violating any of these means the code still runs and still
produces numbers, but the numbers no longer answer the research
question — usually **without an error message.**

| | Rule | Why (see) |
|---|---|---|
| **I1** | Model universe must contain ≥3 near-relative pairs (base + fine-tune, or adjacent versions in one family). | Without this, `CVaR_γ ≈ mean` and the method collapses to the baseline. `METHOD §2, §6.4` |
| **I2** | Prompting-config splits (`build`/`val`/`test`) are disjoint at the **individual parameter** level — no single system prompt, RAG template, or sampling setting crosses splits. | Prevents selection-on-test. `METHOD §6.2` |
| **I3** | Separability is computed **point-cloud vs point-cloud** (multiple configs per model), never single-response vs single-response. | Collapsing to one point erases the intra-model-noise term; see `METHOD §4`. |
| **I4** | Coverage of a query *set* is **MAX** over its queries, never sum or mean. | `METHOD §5.1` |
| **I5** | Embedding model is frozen (`multilingual-e5-large-instruct`). No substitution mid-project without a full re-run. | Comparability across rounds. |
| **I6** | All selection-algorithm comparisons run against the **same frozen corpus/tensor**. Never regenerate traces per-variant. | Otherwise sampling noise confounds the comparison. |
| **I7** | Corpus schema changes require a version bump + migration note **before** further generation. | The corpus costs ~10⁵ generations; regenerating it over a renamed field is the most expensive avoidable mistake here. |

**Mandatory check, not a suggestion:** before drawing any conclusion from
a separability tensor, plot the pair-distribution hard tail (I1's
failure mode is silent — see `METHOD §6.4`).

---

## Numbering & indexing

Every research question is a `D{NNN}` file under `docs/`, numbered
sequentially, ids never reused even if a D is abandoned.
`docs/PLAN.md` is the single index (columns: id · source · status ·
gate · deps · result pointer). **TACC reads `PLAN.md` to find its next
task — not this file, not the chat history.**

## Gates

- **`AUTO`** — TACC executes without waiting. Reserved for environment
  checks, smoke tests, and re-runs of an already-approved P with
  different seeds.
- **`REVIEW`** — requires a posted `## Review` with an APPROVED verdict
  before execution. Default for anything touching an invariant, costing
  more than ~8 GPU-hours, or producing a number destined for a paper.

## File ownership within a cycle file (`docs/D{id}.md`)

No cross-editing. `## D` and `## Review` — design-side only.
`## P` and `## R` — TACC-side only. Comments on someone else's section
go in `## Review` as an appended note, never inline in their section.
