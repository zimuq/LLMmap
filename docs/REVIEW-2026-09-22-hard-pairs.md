# Independent review — D010–D014 and a hard-pair evaluation target

Date: 2026-09-22. Design-side review requested by the human. This is a retrospective evidence audit and a proposal, not approval of a new experiment or a revision of METHOD. Existing D/P/R records remain unchanged.

## For you (human)

**Verdict:** The experiments support retaining JointGreedy and a small gamma as promising choices. They do not establish that the gains are caused by query interaction, or that the method reliably improves a rigorously defined population of hard pairs. Some published-in-repo summaries overstate their own artifacts.

- D012's claim that every sub-1 gamma wins at every k on every metric is factually false. The broader small-gamma result survives.
- D014's trained proxy sanity check covers GreedyCover only, at k=8 and two gamma values; it does not confirm all of JointGreedy's Arm B findings at the trained level.
- The two named difficult pairs have already been evaluated in D009/D010. Their descriptive results improve under joint energy relative to coverage. They are not demonstrated identifiability limits.
- Keep the 65 metadata-defined near-relative pairs. Add a separately frozen, reference-defined empirical hard set if attribution to particular difficult pairs is the intended claim. A fixed tail-risk metric can also express the objective without a binary hard/easy label.

**Needs your input:** None to complete this review. Changing the primary endpoint and METHOD would be a subsequent methodological decision; the definitions below are recommendations, not adopted changes.

**Follow-ups opened:** None. No new D, training, tensor analysis, or experiment was executed.

**Next:** Correct the reporting scope, then specify the hard-pair endpoint before another algorithm/gamma sweep.

---

## Technical log

### 1. Evidence and scope

Read D010–D014 questions/results/reviews, METHOD/FINDINGS/DECISIONS/PLAN, selection/evaluation code, and existing result JSONs. Three auxiliary agents checked D010–D011, D012–D014, and primary literature; the reviewer independently checked the central contradictions and the saved pair values. Reading saved metrics and displaying simple descriptive averages does not constitute a new selection/training experiment. No fresh bootstrap or significance test was run.

The shared frozen corpus, reused baseline outputs, matched training protocol, and paired configuration bootstrap are substantial strengths. D013 tests a competing explanation, and D014 retains an INCONCLUSIVE verdict when its GreedyCover control fails. These are useful checks worth preserving.

The human's present priority, hard-pair performance, changes the emphasis from DECISIONS A4's earlier query-efficiency primary claim. Previous experiments should retain their historical hypotheses; a new endpoint should not retroactively convert exploratory comparisons into preregistered confirmation.

There is also project-level test adaptation: D010 is motivated by the preceding D009 test result, and later questions/interpretations respond to results on the same test configurations. Correct per-run separation and validation early stopping do not make that repeatedly inspected test an untouched confirmation set. Preserve the numerical observations, but qualify project-level discovery claims; future confirmation needs prospectively frozen choices and fresh evaluation configurations under the corpus/split governance rules.

### 2. What D010 actually supports

Existing k=8 trained test results:

| Method | Mean 37-way top-1 | Worst-class | Mean restricted accuracy on 65 near-relative pairs |
|---|---:|---:|---:|
| paper8 | .8452 | .440 | .9637 |
| CVaR coverage | .8519 | .410 | .9644 |
| Joint energy | .8757 | .536 | .9685 |

This is a promising improvement in the trained downstream pipeline. Selection remains classifier-free and cheap conditional on already having the frozen embeddings/distances. Preserve that qualification: corpus construction is not free, and the original paper's inferred 372 candidate training calls are not a measured same-pool wall-clock comparison.

The evidence does not identify the mechanism:

- At k=1 joint energy and coverage are exactly identical, yet both already beat paper8. Therefore eight positive signs against paper8 are not eight observations of interaction gain.
- Against coverage, joint energy has zero k=1 gain and small negative differences at k=4 and k=6. Nested k prefixes are correlated, not independent replications. The illustrative 1/256 sign argument should be omitted.
- Selecting q=0, or changing six of eight queries, does not establish that q=0 carries uniquely useful conditional information. Changed geometry, query scaling, redundancy reduction, nuisance handling, and finite-sample estimation are alternative explanations.
- Set success when each member fails a fixed accuracy threshold would show multi-query utility, but would still not isolate statistical interaction: independent weak evidence can accumulate without cross-query dependence.
- A future mechanism test should distinguish a joint statistic from matched additive/independent-evidence controls. Shuffling configuration alignment can probe dependence, but also changes nuisance correlations; it is not an unambiguous intervention on useful interaction alone.

Recommended wording: **joint scoring improves selected-set performance on the existing benchmark; the contribution of interaction relative to other changes remains unresolved.** Do not say the mechanism was directly observed, as FINDINGS currently does.

Here interaction can only mean joint response evidence/statistical dependence across queries, including dependence induced by shared configurations. The frozen independently generated responses do not test conversational carry-over, online adaptation, or a causal effect of one query on the next response. Also retain the distinction approved in D010: JointGreedy is a separate set-statistic variant; the original coverage algorithm still uses I4's MAX. Adopting the variant as a replacement for the formal main method would require an explicit METHOD/invariant scope decision.

### 3. D011 fixes an inappropriate frontier, but does not certify an intrinsic limit

D011's 32-pair population excludes Phi-3-medium-128k/4k and Falcon3-10B/7B. Thus D011 cannot establish that those two pairs are untouched by selection. METHOD's stronger statement should be corrected in an authorized revision.

The correct criterion-(a) quantifier is: **for each pair, at least one selected query succeeds**. FINDINGS says every query in both chains succeeds on all 32 pairs, which is stronger than the criterion establishes.

There are two further limits to the frontier audit:

1. `experiments/d011_premise.py` computes `both_halves_max = max_q min(acc_A(q), acc_B(q))`. Both halves participate in selecting the successful query. This is a useful descriptive consistency check, not independent holdout validation. `acc_B[argmax acc_A]` separates query selection from evaluation more appropriately, but the T2 population was itself selected using the original test results; that population-level conditioning remains.
2. The .54 build-argmax versus .82 median-query comparison is conditional on this selected T2 population. It is not an unbiased estimate that build argmax is generally anti-informative throughout all 666 pairs. Different energy and accuracy thresholds answering different questions also do not alone prove an estimator bug.

The old T2 frontier is not a suitable fixed hard-pair benchmark. Rejecting that benchmark does not establish that only two pairs are difficult, that those two are intrinsically inseparable, or that query generation is necessary.

D011's set classifier also uses a sum of per-query normalized distance matrices followed by 1NN, rather than D010's joint-energy selector or the trained attention classifier. Its criterion-(b) outcome therefore cannot isolate the latter mechanisms. A smaller reproducibility issue is the self-pair null's `hash(m)` random seed in `d011_recovery.py`: absent a fixed `PYTHONHASHSEED`, Python process randomization changes that null. A stable model-index/digest seed would make it reproducible; no code was changed in this review.

### 4. Saved evidence on the two named pairs

Source: `results/D009/runs.json` and `results/D010/metrics_by_k.json`, k=8, runs 0–4, `per_hard_pair`. Array indices 43 and 64 follow `near_relative_pairs()` and the sorted pair keys used in `logit_stats()`; these are Phi and Falcon respectively.

| Pair | Method | Five saved restricted two-way accuracies | Descriptive mean |
|---|---|---|---:|
| Phi medium 128k / 4k | CVaR coverage | .68, .56, .70, .64, .70 | .656 |
| Phi medium 128k / 4k | paper8 | .64, .80, .74, .66, .72 | .712 |
| Phi medium 128k / 4k | Joint energy | .76, .68, .76, .74, .72 | .732 |
| Falcon3 10B / 7B | CVaR coverage | .82, .82, .82, .78, .76 | .800 |
| Falcon3 10B / 7B | paper8 | .82, .74, .84, .78, .84 | .804 |
| Falcon3 10B / 7B | Joint energy | .94, .76, .82, .76, .88 | .832 |

These are 50 evaluation traces per pair: two models × 25 configurations, with configurations shared across models. The means suggest improvements of 7.6 and 3.2 percentage points relative to coverage; no new uncertainty analysis has been performed. Improvements relative to paper8 are smaller. This is evidence of headroom and variability, not confirmed generalization or evidence that interaction caused the change.

The earlier .800/.840 single-query proxy oracle values are different estimands. They are neither ceilings on an eight-query trained classifier nor valid evidence of an information-theoretic limit.

### 5. Gamma: retain the empirical result, narrow the claims

**Factual correction.** D012 R and its Review claim every sub-1 gamma wins at all eight k values on all metrics with every CI excluding zero. In `results/D012/metrics_by_gamma_k.json`, comparison `1.0_vs_0.25_k1` gives:

| Metric | Stored delta: gamma 1 minus .25 | Stored 95% interval |
|---|---:|---|
| mean_top1 | -.0105688 | [-.0397892, +.0198973] |
| worst_class | 0 | [0, 0] |
| worst3_class | -.0082 | [-.0240, +.0053333] |
| hard_subset | +.0030849 | [-.01428, +.0203754] |

The mean-top1 entry explicitly says INCONCLUSIVE and `exceeds_run_range: false`. This counterexample is enough to refute the universal statement; it does not overturn the substantial overall gamma=.1/.05 versus 1 result.

Other qualifications:

- At k=8, gamma .25/.1/.05 select the same query set, only in different orders. Their similar performance is useful evidence of set stability at that budget, but not three independent successful strategies or a universal flat interval.
- D013 shows small-gamma benefits also exist for coverage. Therefore repairing a joint-statistic-specific defect is not an established general explanation.
- A declining score across increasing k does not imply poor ranking among candidates at a fixed k. Likewise lower training accuracy under one optimization procedure does not prove the representation contains intrinsically less information.
- Low gamma emphasizes low **proxy separability**, not necessarily high classification error. If those rankings misalign, a stronger tail emphasis can target noise or the wrong pairs.

**D014 scope correction.** The 20 trained sanity runs are GreedyCover only, k=8, gamma=.5/.25, proxy-best/proxy-worst resampled chains × five seeds. They support two within-gamma extreme-ordering checks. They do not validate JointGreedy, k=4, typical-chain ranking, the cross-gamma contrast, or the numerical magnitude of all Arm B effects. JointGreedy's Arm B SYSTEMATIC result remains a result for the chosen proxy under the specified resampling protocol.

Arm B varies half-samples of the finite build corpus and uses the same fixed validation population. Its tight intervals concern this perturbation experiment; they do not encompass fresh configuration distributions, new model universes, or new training fits. The failed GreedyCover positive control could reflect the half-sample estimand as well as limited resolution; it does not uniquely identify insufficient power.

### 6. K and statistical interpretation

The experiments evaluate k=1..8 prefixes. They have not established an independently validated rule for selecting k. `peak_k` is the surrogate-objective peak, not the minimum query budget meeting a downstream error target. Retain the full curves and compare methods at equal budgets; if automatic k is desired, choose it on development data using a prespecified risk target or marginal-benefit rule.

Current confidence intervals bootstrap configurations, using shared draws across models/conditions, then average the five fitted networks within each draw. This preserves useful pairing but conditions on those five networks. It does not incorporate resampled training-seed uncertainty. The extra five-seed-range threshold is a heuristic, not a substitute for such an interval. With a fixed 37-model benchmark, do not pretend the 65 overlapping pairs are 65 independent samples. If generalization to new model families is claimed, that requires a separate sampling/design argument, not just configuration bootstrap.

### 7. What hard pair should mean

There is no context-free empirical hardness. It depends on candidate queries, budget, response representation, classifier/training protocol, nuisance distribution, and identification task. A pair hard at k=1 may be easy at k=8.

Distinguish three objects:

| Object | Definition | Use |
|---|---|---|
| Structural near-relative set N | The existing 65 pairs with shared base or lineage, fixed from metadata before tensor inspection | Stable, interpretable benchmark stratum; not a claim that every pair is difficult |
| Fixed empirical hard set H | High error under a declared reference protocol on discovery/development data, frozen before final evaluation | Attribute improvement to the same difficult pairs across methods |
| Active optimization tail T(A,gamma) | Pairs currently in the bottom gamma of a selected set's surrogate scores | Diagnose what the selector emphasizes; not a common evaluation population |

Keep N rather than redefining it in response to the results. Further metadata refinements can be documented prospectively. Do not equate gamma with the proportion of objectively hard model pairs.

**Recommended operational definition.** Fix a reference protocol B0, budget k0, nuisance distribution Pi, and balanced binary decision rule. For pair p={i,j}, define

`e_p(B0,k0) = 0.5 [Pr(predict j | model i) + Pr(predict i | model j)]`.

Here prediction is explicitly restricted to i and j. For minimal disruption, use the same two-logit restriction as the current pipeline. Freeze

`H_ref(k0,tau) = {p : e_p(B0,k0) >= tau}`

using independent discovery/development estimates. For example, tau=.10 means failing to achieve 90% balanced pair accuracy; this is an illustrative target, not an adopted threshold. Set tau from the desired quality target before comparing candidate methods. A fixed paper8 protocol is an interpretable initial B0. A prespecified small reference portfolio can test whether apparent hardness is peculiar to one weak reference, but must not become a post-test best-of-many oracle.

Finite-sample implementation should distinguish clearly-above-threshold, clearly-below-threshold, and uncertain pairs rather than force noisy labels. Membership stability and uncertainty matter with only 25 configurations. If many pairs are screened, simultaneous uncertainty or explicitly exploratory labeling is needed. A fixed relative bottom fraction is possible, but means relatively difficult, not failure to meet an absolute target.

Freeze one reference budget if the same pairs are to be tracked across a k curve. Alternatively report H_ref(k) for each budget and explicitly acknowledge that the populations change. Never define each method's own hard set and compare its mean as if the same pairs were evaluated.

For current D010–D014, any definition devised after inspecting these results is exploratory, even if calculated from existing build data. Future confirmation needs a prospectively frozen protocol and untouched evaluation data.

**Why the theoretical ceiling is different.** For a fixed query set A, the best possible balanced binary error on response distributions P_i^A,P_j^A is `e*_p(A) = (1 - TV(P_i^A,P_j^A))/2`. A budget-limited intrinsic definition would additionally optimize over allowed sets A of size at most k. Poor accuracy of one finite-data 1NN or neural classifier gives no lower bound establishing that every classifier/query set must fail. Thus current evidence cannot certify an identifiability limit.

### 8. The current hard_subset metric is not the proposed main target

`experiments/d009_lib.py:logit_stats` averages two-logit restricted accuracy over all 65 metadata pairs. It is neither the worst pair nor a tail average, and it is not full 37-way accuracy on difficult models. For true model A and scores C>A>B, pair(A,B) counts as correct although deployment predicts C incorrectly.

Recommended evaluation:

1. If the claim is **pair discrimination**, use mean balanced error on the fixed H_ref as the primary attributed outcome, with per-pair results and the full N retained.
2. Also report a prespecified upper-tail risk across all 666 pair errors, such as the mean of the worst beta fraction. Beta is an evaluation choice independent of selection gamma. This is a legitimate common risk functional even though its contributing pairs differ by method. It cannot by itself identify which fixed pairs improved.
3. Retain 37-way mean and worst-class/endpoint accuracy as guardrails. If actual deployment confusion is the target, report the 37-way off-diagonal i↔j confusion, together with endpoint accuracy so errors redirected to a third model cannot masquerade as improvement.
4. State pair weights. Equal pair weights give larger families more influence because family pair counts grow quadratically. Family-macro sensitivity is useful if protection across families is the scientific objective.

A rigorous **risk definition** is necessary; a binary hard/easy label is optional. With unstable hard-set membership, a fixed tail-risk endpoint plus the structural N benchmark may be more defensible initially.

### 9. Literature: borrow principles, not incompatible label semantics

Primary-source search did not identify a universally accepted hard-model-pair definition.

- [LLMmap (USENIX Security 2025)](https://arxiv.org/html/2407.15847) reports closely related versions and confusion with a fine-tuned relative, but does not formalize a universal hard-pair threshold. Its known-version identification task is the closest match.
- [Model Provenance Testing (2025)](https://arxiv.org/html/2502.00706) includes architecturally similar base models and uses external derivation metadata. This supports prospective structural stratification. Its objective is provenance testing, where a parent/child relation is positive, not a pair of identities to separate.
- [Stemma (2026)](https://arxiv.org/html/2607.25880) defines hard negatives using similar architecture/training recipes across distinct provenance groups. The grouping principle is useful; its provenance labels cannot be transplanted into exact-version identification.
- [InterPol (2026)](https://arxiv.org/html/2603.15220) mines the most embedding-similar non-target response for a query and synthesizes harder negatives. These are method/query-dependent training examples, not a frozen model-pair benchmark.

### 10. Minimal-change next steps, in order

**A. Repair measurement before further tuning.** Correct the above reporting claims, then have TACC produce paired per-pair curves from saved counts for paper8, coverage, and joint energy at matched gamma/k. Separate metadata N, a prospectively specified H_ref, and all-pair tail risk. Saved test explorations must be labeled as such; no new generation is needed for that audit.

**B. Keep the current selectors; select gamma and k for the intended endpoint.** Treat .1 as a reasonable incumbent, not a universal optimum. Use development hard-pair/tail risk with overall-performance guardrails; prefer the smallest budget meeting a target. Do not add a dense gamma sweep before establishing whether the selector's low-score pairs coincide with actual high-error pairs.

**C. Only if objective/outcome misalignment persists, add a small correction layer.** Retain classifier-free coverage/joint selection to propose a small candidate portfolio, and use a limited number of development-set classifier evaluations to rerank it for the fixed endpoint. This preserves the frozen corpus and most selection-cost advantages. A fixed build-derived pair weighting or auxiliary tail term is another proposal, but changes METHOD and needs explicit adoption; it is not being implemented here.

Hard-set definitions do not create missing information. If multiple credible matched-budget selections remain poor, investigate separately whether the limitation is query coverage, the frozen representation, the decoder, or nuisance robustness. Failure of the tested methods still does not prove intrinsic impossibility. Existing Phi/Falcon descriptive gains justify evaluating this objective seriously, while the present evidence does not justify promising that smaller gamma or more joint scoring will reliably solve it.
