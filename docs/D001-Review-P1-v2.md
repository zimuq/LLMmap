## Review — D001 / P1

### For you (human)

**Verdict:** APPROVED WITH AMENDMENTS

**What changed**
- The pass/fail test I originally wrote for D001 was too weak to actually
  detect anything (too few known cases to test against) — replaced it with
  a stronger test that uses all the data instead of a small slice of it.
- Added one new number to report: how different our target (worst-case)
  is from the standard target (average-case). This is the single most
  important number in this experiment — it's the direct answer to "is
  there anything here worth optimizing for."
- Added a caveat: this experiment can rank how confusable the Llama-70B /
  Smaug pair is, but under a different scoring method than the paper
  used — so it can't be reported as "reproducing" the paper's 84% number,
  only as a related but distinct measurement.

**Needs your input:** None for D001 itself.

**Follow-ups opened**
- **GPU access appears broken** on the TACC partition used (CPU-only
  torch, no GPU visible to the scheduler). Harmless for this small
  experiment (~40 min either way) but **blocks the next phase**, which
  needs a much larger run. Proposed as a new D — see start prompt for the
  local session.
- Two minor judgment calls (a borderline model-size cutoff, how to count
  parameters for a couple of models) were resolved on the design side.
  No action needed; noted for your awareness.

**Next:** TACC executes the approved plan; results return as D001's `## R`.

---

### Technical log (for TACC / audit trail — skip unless "For you" told you to)

P1 correctly resolved all three items D001 left open, and did so by
inspecting the code rather than assuming: it found that the shipped
"template" for each model is a single averaged vector with all
per-response spread discarded, correctly concluded that centroid
distance therefore cannot serve as the separability measure, and
redesigned the metric around traces instead. That is exactly the
behavior the open items in D001 were meant to test for.

#### A1 — Amendment to the D-section criteria (design-side error, now corrected)

P1 §S4 is right that the decile-enrichment test has almost no power: with
~8–10 documented base↔fine-tune pairs among 52 models, the null
expectation inside a 10% decile is ≈1. That criterion was poorly
specified in D001. Rather than have P work around it, the criterion
itself is amended.

**The `What counts as an answer` table in D001 is replaced by:**

| Outcome | Criterion |
|---|---|
| **CONFIRMS** | *(primary)* Structurally-related pairs have significantly lower separability than unrelated pairs across the **full** distribution (Mann-Whitney over all 1326 pairs, model-level bootstrap CI excluding no-effect), **and** the CVaR/mean ratio (see A2) is meaningfully below 1. |
| **FALSIFIES** | The related-vs-unrelated distributions are statistically indistinguishable, **or** the CVaR/mean ratio ≈ 1 (no exploitable tail). |
| **INCONCLUSIVE** | CIs span no-effect on the primary test. Report; do not force a verdict. |

Decile enrichment is retained as a **descriptive secondary** statistic —
report it, but it no longer carries the verdict. The permutation test and
model-level bootstrap proposed in P1 §S4 are both accepted as specified.

#### A2 — New measurement M6 (add to S3; also compute for the ≤14B subset in S6)

**M6 — Objective divergence ratio.**

```
CVaR_0.1(separability)  /  mean(separability)
```

over all pairs, and separately over the ≤14B subset. This is the direct
numerical answer to "does a CVaR objective differ from a mean objective
on this population" — a ratio near 1 means the two objectives coincide
and the premise fails (the silent-null failure mode invariant I1 exists
to catch). One line of code; report prominently in R alongside the
percentiles.

#### A3 — Interpretation limit to carry into R

The AUC of `Δ = d(trace,T_B) − d(trace,T_A)` measures separability under
the **open-set nearest-template** classifier. The paper's 84% for
Meta-Llama-3-70B-Instruct comes from the **closed-set softmax**
classifier (Figure G.1). Different systems. M4 may report the pair's
percentile rank *under our open-set measure*; it must **not** be reported
as reproducing the paper's 84% confusion. State this explicitly in R.

#### Judgment calls from P1 §S6 — resolved

- **`Phi-3-medium-{4k,128k}` at exactly 14B → INCLUDE.** The boundary is
  soft, and the 4k↔128k pair is a valuable near-relative candidate.
  Excluding it discards ammunition for the very structure D001 tests for.
  Report the 37/9/6 split as primary, 35/11/6 as sensitivity.
- **MoE under total-parameter accounting → AGREED.** Memory residency is
  set by total parameters, not active ones, so total is the correct basis
  for feasibility.

#### Everything else in P1 — accepted as written

S2 (template provenance recomputation): accepted — cheap, resolves
whether shipped templates were built on train only. If not, M1 is
optimistically biased and R must say so.

Test-split-only evaluation (3536 traces, 68/model) is correct and should
be stated as such in R — evaluating with train traces against
train-built templates would be optimistically biased.

The label-order guard in S1 is exactly right. Keep the "stop, do not
patch around it" response if it trips.

#### Design-side follow-ups (not blockers for D001)

1. **GPU availability must be resolved before the corpus-building phase.**
   P1 §F5 reports `torch 2.7.1+cpu` and no GRES advertised on `gh`/`gh-dev`.
   Harmless for D001 (~40 min CPU). The next phase is ~10⁵ generations and
   is not feasible on CPU at that scale. Needs a dedicated D to establish
   GPU access under Slurm and a measured throughput number — not an
   estimate — before that phase's cost can be planned.
2. Two aarch64 fixes were made to the repo (`torch.load()` device
   handling; missing `accelerate` dependency) **before** D001 was handed
   over, i.e. outside any REVIEW-gated plan — confirmed with the human,
   no process violation. Logged here for the audit trail regardless,
   since one of the two touches checkpoint loading and is therefore
   research-relevant, not purely cosmetic.
3. `CLAUDE.md`, `METHOD.md`, and `DECISIONS.md` are now being pushed to
   this repo, resolving P1 §6's open item. Future D's involving the model
   universe (I1) can reference these directly instead of reporting
   descriptively.

#### Note for A1 (model universe) — do not let M1 become A3

The AUC metric adopted here is scoped to D001 only. It is **not** a
decision on the separability statistic for the real tensor (open decision
A3 in `DECISIONS.md`). D001 measures traces against a collapsed centroid;
the tensor measures point cloud against point cloud in embedding space —
different quantities. A3 remains open.
