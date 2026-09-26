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
| [D012](D012.md) | `DECISIONS.md` C1 (γ ablation, never run for `JointGreedy`); human-approved 2026-09-16, sharpened from a broader "k and γ sweep" | **CLOSED** 2026-09-17 — **CONFIRMED, decisively** (corrected 2026-09-22: 23 of 24 mean-top1 cells, not literally "every" — see `D012.md` `## Review` addendum): γ<1 required for `JointGreedy` across almost the entire k (1–8) × metric grid, most CIs excluding zero — the first comparison in this project resolved at nearly all 8 k values. γ=0.1 recommended default; {0.25,0.1,0.05} statistically indistinguishable. Mechanism: γ=1.0's own selection objective falls monotonically (1.30→0.88) while its trained accuracy rises (0.54→0.80) — its selected queries fit worse even on training data (0.939 vs 0.992), i.e. carry less usable signal, not just mis-scored. Bonus: trained network confirmed order-insensitive to query-slot position | REVIEW (P1 approved w/ amendments, executed) | D007 (CLOSED), D008 (CLOSED), D009 (CLOSED), D010 (CLOSED) | `results/D012/`, `D012.md` `## R` |
| [D013](D013.md) | D012/R7 (optional follow-up, now promoted); D008's original proxy γ-sweep for `GreedyCover`, never trained beyond γ=1.0/0.1; human-approved 2026-09-18 | **CLOSED** 2026-09-18 — **CONFIRMED, D012-shaped, for γ ≤ 0.25**: `GreedyCover` needs a small γ as much as `JointGreedy` (mean penalty for γ=1.0: −0.081 vs −0.079; no detectable difference between the algorithms). Structural hypothesis: mechanism right (objective monotone, I4), effect-size prediction wrong — so D012/R2's "CVaR repairs a broken objective" is `JointGreedy`-specific, not general (D012 carries an addendum). New: γ=0.5 gave no benefit over γ=1.0 on this chain — threshold in γ vs. unlucky chain is untested. C1 closed for both algorithms | REVIEW (P1 approved w/ amendments, executed) | D007 (CLOSED), D008 (CLOSED), D009 (CLOSED), D012 (CLOSED) | `results/D013/`, `D013.md` `## R` |
| [D014](D014.md) | D013 R2 + Review (γ=0.5 anomaly on one chain); D012 addendum (JointGreedy never run between γ=0.25 and 1.0); human-approved 2026-09-19, scope: both algorithms | **CLOSED** 2026-09-19 — γ=0.5 question. `JointGreedy`: **RAMP** — a systematic, graded γ effect, chain luck rejected. `GreedyCover`: **INCONCLUSIVE** — the chain-stability arm failed its pre-registered control (and that failure is itself informative: the small-`k` effect may be smaller under resampling). Chains recur as families across neighbouring γ, so the trained grid alone would have shown a misleading clean STEP for `GreedyCover`; the amendment making Arm B decisive prevented it. Design-side's lean toward chain luck was wrong. `METHOD.md §5.2` wording proposed, awaiting the human | REVIEW (P1 approved w/ amendments, executed) | D007–D013 (all CLOSED) | `results/D014/`, `D014.md` `## R` |
| [D018](D018.md) | Un-shelved "Direction B" (`PLAN.md` "Not yet a D"); human-approved 2026-09-26 (accepts cost, both classifiers) | **OPEN — P1 approved 2026-09-26, no amendments.** S0's cost pilot priced all 5 arms: total for both arms ≈4.8 GPU-hours (attention 2.47h — cheaper than PLAN.md's old ~5h estimate; linear mean-pool-tight 2.30h chosen for the search over concat-tight, the most expensive option at 5.98h due to k-scaling). `S_val` argmax, 1 seed in search + 5 for the final chain, mean-pool-tight for the search (both poolings at final eval) all confirmed. PARITY-OR-BETTER tightened to require TOST equivalence AND |Δ| below the 5-seed run range (TOST alone would be satisfied by noise here). S1–S5 awaiting execution | REVIEW (P1 approved, not yet executed) | D007 (CLOSED), D008 (CLOSED), D009 (CLOSED), D010 (CLOSED), D015 (CLOSED), D017 (CLOSED) | `results/D018/`, `D018.md` `## P` |
| [D017](D017.md) | Human proposal, this session (2026-09-24); human-approved ("先起草D017") | **CLOSED** 2026-09-25 — **CLASSIFIER-AGNOSTIC on the ordering**: `joint energy > paper8` holds with consistent sign at all 8 `k` under the attention network AND both linear poolings, larger under linear (mean Δ +0.018 concat, +0.049 mean-pool, vs the attention network's own numbers). `coverage > paper8` does NOT survive under linear (sign flips) — third confirmation `coverage` alone is the weak link. `hard_subset`'s `joint−paper8` delta was **never resolved on either classifier** (not "lost under linear" — D015 already showed why); reported honestly rather than forced into CLASSIFIER-DEPENDENT. Independent, unrelated finding: a plain linear classifier beats the paper's attention network at every condition/`k` (24/24 cells, +3.9 to +11.0pp) — flagged for the human, not investigated here | REVIEW (P1 approved w/ amendments, executed) | D007 (CLOSED), D008 (CLOSED), D009 (CLOSED), D010 (CLOSED), D015 (CLOSED), D016 (CLOSED) | `results/D017/`, `D017.md` `## R` |
| [D016](D016.md) | FINDINGS.md "reformulating hard_subset" entries (2026-09-22/23); human-approved 2026-09-23 ("是的,起草这个D") | **CLOSED** 2026-09-24 — **NO**: the 65 structural pairs contain the 11 objectively hardest pairs of all 666 (real trained classifier, min across paper8/coverage/joint energy); first non-structural pair is rank 12+ at 0.916 vs. worst structural 0.656. Settles `DECISIONS.md` D7's population question — the structural set is not missing hard pairs; `hard_subset`'s problem is its aggregation (D015), not its membership. Tensor proxy over-stated non-structural difficulty (5 vs. 1 in each side's worst-12, corrected from R's "6" in post-hoc Review); proxy-trained correlation much stronger inside the structural set (+0.75–0.85) than outside it (+0.55–0.70). Full 666-pair trained-accuracy table now available for any tail/worst-pair metric definition, no further TACC work needed | REVIEW (P1 approved w/ amendments, executed) | D007 (CLOSED), D008 (CLOSED), D009 (CLOSED), D010 (CLOSED), D011 (CLOSED), D015 (CLOSED) | `results/D016/`, `D016.md` `## R` |
| [D015](D015.md) | `docs/REVIEW-2026-09-22-hard-pairs.md` §8/§10.A; human-approved 2026-09-22 | **CLOSED** 2026-09-22 — **redistribution CONFIRMED**: the +0.41pp `hard_subset` gain (joint vs coverage, k=8) is 28 pairs up / 20 down / 17 flat, gross movement 3× the net; coverage vs paper8 is the same shape (28/21/16). "Spread vs concentrated" unanswerable at 25-config resolution (P1/F3), as pre-registered. The two named pairs rank 1st/2nd-hardest under all three methods (independent corroboration). **Correction (post-hoc Review):** `## R`'s headline example misattributed a −0.116 resolved delta to the project's named hardest pair (medium-128k/medium-4k); it belongs to a different pair (medium-128k/mini-4k, ranked 3rd); the named hardest pair's own coverage-vs-paper8 delta (−0.056) does not resolve. Redistribution finding and rank-1/2 finding unaffected | REVIEW (P1 approved, executed) | D007 (CLOSED), D009 (CLOSED), D010 (CLOSED), D012 (CLOSED), D013 (CLOSED) | `results/D015/`, `D015.md` `## R` |

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
  (Algorithm H.1-style) on our own 259-query pool** — **un-shelved
  2026-09-26, drafted as [D018](D018.md).** D010 ("Direction A") came
  back CONFIRMED, not FALSIFIED, and the item stayed shelved from
  2026-09-10 pending "a new human reason" — the human gave one
  (2026-09-26: accepts the ~5 GPU-hour cost, wants both the attention
  network and the linear classifier tried), directly motivated by D017's
  finding that a linear classifier is a cheap, viable second readout.
  See D018 for scope, cost-pilot gate, and the three pre-registered
  outcomes.
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
- **D012 R7's follow-up — done as [D013](D013.md), closed 2026-09-18.**
- **From D013/D014, 2026-09-19:** (a) γ=0.5 question — **closed via
  [D014](D014.md)** (`JointGreedy` RAMP; `GreedyCover` INCONCLUSIVE).
  (b) `METHOD.md §5.2` — **revised 2026-09-21** with the human's OK
  (`DECISIONS.md` log).
  (c) Why a small γ helps selection quality *in general* is unexplained —
  the most interesting open scientific question; D014 did not chase it.
  (d) **Optional, cheap, not drafted:** from D014's saved per-resample
  arrays, report descriptive paired contrasts for all γ pairs (notably
  γ=0.1 vs 1.0) at `k`=4 and 8, labelled post hoc — tells whether
  `GreedyCover`'s small-`k` γ effect exists at γ=0.1 under resampling
  (relevant to A4's small-`k` claim). No new selection or training.

## Notes

- TACC executes only `status: OPEN` D's whose `gate` is satisfied — `AUTO`
  runs immediately, `REVIEW` needs a posted `## Review` with an APPROVED
  verdict.
- "Closed" is a design-side call made after `## R` is posted **and** any
  R-driven open questions are resolved for the D's *own* question. D001–D008
  are all closed as of 2026-09-08 — each answered its own question in full,
  even where (D001/D002/D008) the answer spawned a still-open item (A-series,
  or a sequencing question) that outlives the D itself.
