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
| [D006](D006.md) | TODO.md Phase 1 (T1.1–T1.4) + D002 §R4 + D003 + D005; A1/C4/C7 decided 2026-09-05/06 | **CLOSED** 2026-09-06 — READY at 37/37 (1,197,875 generations); own embedding-normalization bug found and fixed (+0.051 AUC, matches paper's actual procedure) | REVIEW (P1 approved, executed) | none | `results/D006/`, `D006.md` `## R` |
| [D007](D007.md) | TODO.md Phase 1 (T1.5–T1.6); A3 decided 2026-09-06 | **CLOSED** 2026-09-07 — **TAIL CONFIRMED on `S_energy`** (CVaR₀.₁/mean=0.353, uncensored; `S_probe` was 51.8–98.2% ceiling-pinned, A3's gate correctly deferred to `S_energy`). `METHOD.md §2`/I1 stand on the real corpus. Frozen tensor for I6: `S_energy_tok200` | REVIEW (P1 approved, executed) | none | `results/D007/`, `D007.md` `## R` |
| [D008](D008.md) | TODO.md Phase 2 (T2.1–T2.4); D007's frozen tensor | **CLOSED** 2026-09-08 — **INCONCLUSIVE** by the outcome table's letter (worst-class improves at k=1–3 only; hard-subset compressed, 7.5-pt total range). Real, replicated effect underneath: CVaR reaches 70% mean accuracy at k=2 vs mean-greedy's 4 / paper's 5 / random's 12 — a query-*efficiency* gain. New finding: 33 of 37 "unidentifiable" pairs have a near-perfect query already in the pool that selection missed | REVIEW (P1 approved, executed) | none | `results/D008/`, `D008.md` `## R` |
| [D009](D009.md) | `METHOD.md` §5.5 (new, 2026-09-08); D008's frozen selection chains; human decision to pause Phase 3 pending this validation | **CLOSED** 2026-09-09 — **CONFIRMED** vs mean-greedy (every k, every CI excludes zero) and random; **tie** vs the paper's own 8 (+0.007 mean top-1 at k=8, not resolved against seed range) — reframed in a post-hoc Review: Algorithm H.1 (`Appendix H`) retrains a real classifier 372 times to pick those 8, CVaR never touches one during selection, so parity is evidence for CDQD's cost thesis, not against it. Proxy metric (D004–D008) validated as a sound ranking instrument (Spearman 0.95–0.98 vs trained) | REVIEW (P1 approved w/ amendments, executed) | D008 (CLOSED) | `results/D009/`, `D009.md` `## R` |
| [D010](D010.md) | D009 R2/post-hoc Review (2026-09-09); human-scoped 2026-09-10 as "Direction A" | **OPEN** — P1 reviewed and APPROVED WITH AMENDMENTS 2026-09-10 (pilot: statistic is weaker-with-k but not degenerate, AUROC ≥0.985 through k=16; found an exact 519× accumulator, both Calls dissolved by measurement; added a k=8 MMD check + a request to headline wherever the objective peaks). TACC executing S2–S6 | REVIEW (P1 approved w/ amendments) | D008 (CLOSED), D009 (CLOSED) | `results/D010/`, `D010.md` `## R` |

## Not yet a D

- **Targeted generation (TODO.md Phase 3, T3.1+)** — **status unchanged
  by D009: still paused.** D009 confirmed the tensor-level proxy is a
  sound ranking instrument, so the case for Phase 3 rests on the same
  grounds D008 left it on (33/37 "unidentifiable" pairs already have a
  good pool query that selection missed — a selection-algorithm question,
  not a pool-coverage one).
- **"Direction B" — an actual classifier-in-the-loop greedy search
  (Algorithm H.1-style) on our own 259-query pool** — **explicitly
  shelved by human decision, 2026-09-10.** Newly affordable in principle
  (D009's pilot: 8.7s/training run, so ~5 GPU-hours for a full 259-query
  greedy search, not the multi-day cost the paper's own setup implied)
  but deliberately not pursued now — D010 ("Direction A," above) tests
  whether a cheap, classifier-free statistic can capture the same
  interaction signal first. Revisit only if D010 comes back FALSIFIED
  **and** the human decides closing the paper-8 gap is worth the
  classifier-in-the-loop cost — not automatic either way.

## Notes

- TACC executes only `status: OPEN` D's whose `gate` is satisfied — `AUTO`
  runs immediately, `REVIEW` needs a posted `## Review` with an APPROVED
  verdict.
- "Closed" is a design-side call made after `## R` is posted **and** any
  R-driven open questions are resolved for the D's *own* question. D001–D008
  are all closed as of 2026-09-08 — each answered its own question in full,
  even where (D001/D002/D008) the answer spawned a still-open item (A-series,
  or a sequencing question) that outlives the D itself.
