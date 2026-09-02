# TACC_NOTES.md — issues TACC finds but can't fix itself

> TACC's channel for flagging problems that live in a **design-side-owned**
> section — a D-file's `## D`, `## Review`, or its frontmatter/header block
> — per `CLAUDE.md`'s file-ownership rule. TACC can see these are wrong but
> is not allowed to edit them directly. Append new issues under **Open**
> (numbered, continuing the running count) instead of only noting them ad
> hoc inside a `## P`/`## R`. Design-side reads this each session, fixes
> what it can, and moves the entry to **Resolved** with a one-line summary
> of what changed.
>
> This complements, doesn't replace, flagging something inline inside a
> specific D when the issue is local to that D and self-explanatory there
> (e.g. D003 P1 §F3's C5/C6 citation fix, noted inline and left for design
> side to correct in place). Use this file for things that are easy to
> lose track of across sessions — stale frontmatter, placeholder dates, a
> table that drifted from a later amendment — or that don't belong to any
> single D.

---

## Open

<!-- TACC appends here. One entry per issue: what's wrong, exactly where
     (file + section), and why the ownership rule blocks a direct fix. -->

(none currently open)

---

## Resolved

**Issue 4 (2026-09-02) — D001's "What counts as an answer" table was
stale.** The original decile-enrichment CONFIRMS/FALSIFIES/INCONCLUSIVE
criteria were superseded by the Review's "A1 — Amendment to the D-section
criteria" before P1 ever executed, but the `## D` table itself was never
updated to match — a reader looking only at "What counts as an answer"
would have seen a criterion that was no longer actually in force.
**Fixed:** table replaced with the amended (Mann-Whitney + CVaR/mean
ratio) criteria actually used; original decile-enrichment wording kept as
a note (it's retained as a descriptive secondary statistic, not the
verdict) and the full history stays in the Review section.

**Issue 5 (2026-09-02) — Stale/placeholder frontmatter in D001 and D002.**
D001's `downstream:` field named "D002 (model universe selection)" as the
consumer of its answer; D002 turned out to be about GPU availability/cost,
not model-universe selection (that question is A1, `DECISIONS.md`). Also,
`opened: <date>` was still a literal template placeholder in both D001 and
D002, and D002's `closed:` field said `—` even though its `status:` line
already said CLOSED. **Fixed:** D001's `downstream:` now points to A1;
both `opened:` fields backfilled from `git log --follow` on each file
(D002's is a lower bound — its history is squashed before 2026-09-01, so
the true open date may be slightly earlier and isn't independently
recoverable); D002's `closed:` filled in to match its `status:` line.
