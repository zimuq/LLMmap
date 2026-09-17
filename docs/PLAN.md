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
| [D008](D008.md) | TODO.md Phase 2 (T2.1–T2.4); D007's frozen tensor | **CLOSED** 2026-09-08 — **INCONCLUSIVE** by the outcome table's letter (worst-class improves at k=1–3 only; hard-subset compressed, 7.5-pt total range). Real, replicated effect underneath: CVaR reaches 70% mean accuracy at k=2 vs mean-greedy's 4 / paper's 5 / random's 12 — a query-*efficiency* gain. **"Identifiability frontier" finding retired 2026-09-13 (see addenda + D011):** the flagged 32/37 pairs are resolved 32/32 by any real selected chain — the per-pair oracle that flagged them was an estimator artifact (median 0.54 on test, worse than random), not a selection-algorithm gap | REVIEW (P1 approved, executed) | none | `results/D008/`, `D008.md` `## R` |
| [D009](D009.md) | `METHOD.md` §5.5 (new, 2026-09-08); D008's frozen selection chains; human decision to pause Phase 3 pending this validation | **CLOSED** 2026-09-09 — **CONFIRMED** vs mean-greedy (every k, every CI excludes zero) and random; **tie** vs the paper's own 8 (+0.007 mean top-1 at k=8, not resolved against seed range) — reframed in a post-hoc Review: Algorithm H.1 (`Appendix H`) retrains a real classifier 372 times to pick those 8, CVaR never touches one during selection, so parity is evidence for CDQD's cost thesis, not against it. Proxy metric (D004–D008) validated as a sound ranking instrument (Spearman 0.95–0.98 vs trained) | REVIEW (P1 approved w/ amendments, executed) | D008 (CLOSED) | `results/D009/`, `D009.md` `## R` |
| [D010](D010.md) | D009 R2/post-hoc Review (2026-09-09); human-scoped 2026-09-10 as "Direction A" | **CLOSED** 2026-09-11 — **CONFIRMED**: a classifier-free joint (set-level) statistic reverses the paper-8 gap at every k (mean +0.026, 8/8 positive) and beats CVaR-coverage at k=8 (+0.024), selection staying at 0.5 min CPU vs the paper's 372 training runs. Magnitude unresolved at any single k (25-config test-set resolution, same wall D008/D009 hit); why the advantage persists at k=8 rather than fading as predicted is unexplained | REVIEW (P1 approved w/ amendments, executed) | D008 (CLOSED), D009 (CLOSED) | `results/D010/`, `D010.md` `## R` |
| [D011](D011.md) | D008's identifiability frontier + D010's chain, human-approved 2026-09-13 | **CLOSED** 2026-09-13 — **NOT RECOVERED (0/32), because there was nothing to recover**: both `GreedyCover` and `JointGreedy` already resolve all 32 flagged pairs (32/32, D008's own threshold). D008's per-pair oracle scored a median 0.54 on these pairs — *worse than a random pool query (0.82)* — confirming the "identifiability frontier" tier was an estimator artifact, not real difficulty. A threshold-scale bug caught in review would have shown the opposite (wrong) conclusion had it shipped. D010's real advantage is confirmed to come from elsewhere, still unexplained | REVIEW (P1 approved w/ amendments, executed) | D007 (CLOSED), D008 (CLOSED), D010 (CLOSED) | `results/D011/`, `D011.md` `## R` |
| [D012](D012.md) | `DECISIONS.md` C1 (γ ablation, never run for `JointGreedy`); human-approved 2026-09-16, sharpened from a broader "k and γ sweep" | **CLOSED** 2026-09-17 — **CONFIRMED, decisively**: γ<1 required for `JointGreedy` at every k (1–8) and every metric, every CI excludes zero — the first comparison in this project resolved at all 8 k values. γ=0.1 recommended default; {0.25,0.1,0.05} statistically indistinguishable. Mechanism: γ=1.0's own selection objective falls monotonically (1.30→0.88) while its trained accuracy rises (0.54→0.80) — its selected queries fit worse even on training data (0.939 vs 0.992), i.e. carry less usable signal, not just mis-scored. Bonus: trained network confirmed order-insensitive to query-slot position | REVIEW (P1 approved w/ amendments, executed) | D007 (CLOSED), D008 (CLOSED), D009 (CLOSED), D010 (CLOSED) | `results/D012/`, `D012.md` `## R` |

## Not yet a D

- **Targeted generation (TODO.md Phase 3, T3.1+)** — **paused, and
  explicitly confirmed lowest priority, 2026-09-13 (human decision).**
  D008's "32/37 pairs selection couldn't resolve" was retired by D011 —
  an estimator artifact, not a real gap (see `D008.md`'s addenda). The
  only remaining real motivation is the 2 structurally-hard pairs
  tracked since D001 (`Falcon3-10B↔7B`, `Phi-3-medium-128k↔4k`), which
  D011 confirmed are untouched by any selection-algorithm fix — the
  human has decided that alone does not justify resuming Phase 3 now.
  Revisit only on a new, separate reason, not a re-check of this one.
- **"Direction B" — an actual classifier-in-the-loop greedy search
  (Algorithm H.1-style) on our own 259-query pool** — **still shelved.**
  D010 ("Direction A") came back CONFIRMED, not FALSIFIED — the
  condition that would have reopened this item did not fire. Newly
  affordable in principle either way (D009's pilot: 8.7s/training run,
  ~5 GPU-hours for a full 259-query greedy search), but there is no
  longer even a stated reason to revisit it; stays shelved pending a new
  human reason, not a re-check of this one.
- **New from D010 R9, 2026-09-13, not yet scoped as a D:** (a) why the
  joint statistic's trained-accuracy advantage persists through `k=8`
  when its own selection objective and S1's pilot both predicted it
  should fade after small `k` — unexplained, and **sharpened by D011**:
  D010's advantage is now confirmed to come from *somewhere other than*
  the 32-pair population D011 checked, narrowing but not answering where
  it actually lives; flagged as the most interesting open question in
  the project, not urgent; (b) **closed via [D011](D011.md), 2026-09-13**
  — NOT RECOVERED, because the population itself wasn't real (see
  above); (c) whether CDQD's method going forward is `GreedyCover`
  alone, `JointGreedy` alone, or a two-stage hybrid (cheap MAX-coverage
  first, joint-statistic refinement second) — an architecture/writeup
  question as much as an experimental one, see `DECISIONS.md` D5.
- **D011's `METHOD.md §7` item — resolved 2026-09-14** (`DECISIONS.md`
  D6): §7 now names the 2 structurally-hard pairs instead of the
  retired T2 tier.
- **New from D012 R7, 2026-09-17, optional, not urgent:** `GreedyCover`'s
  own γ-sensitivity has only ever been measured via D008's proxy, never
  trained. If a writeup wants to claim `GreedyCover` and `JointGreedy`
  genuinely differ in how much they need CVaR-weighting (suggested by
  D012/P1 Part 0's tensor-level trade-off finding, not yet proven at the
  trained level), that needs `GreedyCover`'s γ variants trained too —
  ~5 minutes of compute per TACC's estimate. Not drafted as a D.

## Notes

- TACC executes only `status: OPEN` D's whose `gate` is satisfied — `AUTO`
  runs immediately, `REVIEW` needs a posted `## Review` with an APPROVED
  verdict.
- "Closed" is a design-side call made after `## R` is posted **and** any
  R-driven open questions are resolved for the D's *own* question. D001–D008
  are all closed as of 2026-09-08 — each answered its own question in full,
  even where (D001/D002/D008) the answer spawned a still-open item (A-series,
  or a sequencing question) that outlives the D itself.
