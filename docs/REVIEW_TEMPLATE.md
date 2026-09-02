# Review format

Every `## Review` posted on a `D{id}.md` file follows this shape. The
point is that the human should never need to read past "For you" unless
it tells them to.

```markdown
## Review — D{id} / P{n}

### For you (human)

**Verdict:** APPROVED · APPROVED WITH AMENDMENTS · REJECTED

**What changed** *(only if amendments — one line each, plain language,
no file:line references, no code)*
- ...

**Needs your input:** *(explicit, or literally the word "None")*
- ...

**Follow-ups opened:** *(new D's drafted as a result, one line each)*
- ...

**Next:** *(one line — what happens now)*

---

### Technical log *(for TACC / audit trail — skip unless "For you" told you to)*

*(everything else: code references, statistical justification, cost
figures, what was inspected in the repo and what it showed, judgment
calls and their reasoning. As detailed as it needs to be. This is the
part that gets re-read when writing the paper's Methods section, not
the part the human reads during the loop.)*
```

## Rules

- **"For you" is capped at roughly what fits without scrolling.** If it's
  getting long, something belongs in the technical log instead.
- **No jargon in "For you" that wasn't already used in the D's own
  question.** If a term needs defining, either define it in one clause
  or move that content down.
- **"Needs your input" must be a closed, answerable question**, not "let
  us know if you have thoughts." If there's nothing to ask, say "None" —
  don't pad it.
- The technical log is where invariant reasoning, statistical power
  caveats, and code-inspection findings live. It's allowed to be as long
  as D001's was.
