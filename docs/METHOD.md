# METHOD.md — Coverage-Driven Query Design (CDQD)

> Self-contained summary of the method. Written so that a reader with no access
> to the design conversation can implement and critique it.
>
> Companion files: `TODO.md` (what to verify), `DECISIONS.md` (what not to change).

---

## 1. What we are building on

**LLMmap** (Pasquini, Kornaropoulos, Ateniese — USENIX Security '25) fingerprints
the exact LLM version behind a black-box LLM-integrated application. It sends 8
fixed queries, embeds the (query, response) pairs, and classifies with a small
transformer (~8M params). Reported: 95.35% closed-set accuracy over 42 models.

Two structural facts about it matter here.

**Fact 1 — the two stated objectives are never actually optimized.**
The paper defines two properties a good query should have:

- *Inter-model discrepancy* (Eq. 2): `q* = argmax_q E_{v,v'} [ d(LLM_v(q), LLM_v'(q)) ]`
- *Intra-model consistency* (Eq. 3): `q* = argmin_q E_{s,s'} [ d(s(LLM_v(q)), s'(LLM_v(q))) ]`

Neither appears in the actual selection procedure. Algorithm H.1 is a greedy
search whose objective is **downstream classifier accuracy** —
`f ← train(Q, T_train); A ← eval(f, T_test); q* ← argmax A`. The distance
function `d` is never instantiated. The paper says so itself in §5.1: Eq. 2 and
3 "*could* serve as the basis for an objective function ... we leave this as
future work." The released repo confirms it — `confs/queries/default.json` is a
static hard-coded list of 8 strings, and no greedy-search code or 50-query
candidate pool is published.

**Fact 2 — the query space is a curated discrete set, not a generative space.**
The pool is ~50 hand-written + ChatGPT-4-paraphrased queries. Selection can only
*redistribute* the pool's existing power; it cannot create new power. The paper's
own baselines prove this: with identical greedy and identical training, swapping
the pool changes accuracy — i.e. **pool quality dominates selection quality.**

---

## 2. The gap we are targeting

Mean-based selection is dominated by easy pairs. The paper's own worst case shows
the cost: **Llama-3-70B-Instruct scores 84%**, and the confusion matrix attributes
this to Smaug-Llama-3-70B — *a fine-tune of that same model*. A mean objective
applies no optimization pressure to that pair.

Two observations shape the design:

**(a) LLMmap is noise-limited, not information-limited.**
Single-query accuracy is 72.9% over 42 classes, far above the ~5.4 bits needed.
Information is not the bottleneck; the bottleneck is response variance from
unknown system prompts, RAG wrappers, and temperature. This is consistent with
prompt-injection triggers (which reduce variance without adding signal) buying
+4%. **Implication: chasing 95.3% → 96% fights irreducible noise. The headroom
is at small k (72.9 / 85.9 / 90.5 for k=1,2,3) and in the worst case.**

**(b) Selection alone is capped by the pool.**
If no query in the pool separates Llama-3-70B from Smaug, no selection rule
recovers it. So the method must contain a *generation* component, and the
selection component's real job is to **diagnose where generation should aim.**

---

## 3. Theoretical framing

The two LLMmap objectives are not competing goals requiring a hand-tuned
trade-off. They are the two terms of a single mutual information:

```
    I(V ; Y_q)  =  H(Y_q)  −  H(Y_q | V)
                   ↑           ↑
        inter-model         intra-model
        discrepancy         inconsistency
          (Eq. 2)             (Eq. 3)
```

`V` = model version, `Y_q` = response to query `q`. This is an identity, not a
heuristic. Consequences:

1. The relationship is **subtraction, not a ratio** — no `λ` to tune.
2. Fano's inequality lower-bounds error by `1 − (I(V;Y)+1)/log|L|`, so maximizing
   `I` directly attacks the error floor. LLMmap currently has no such objective.
3. It formalizes the paper's stated-but-unmodeled observation that weak queries
   combine well: `I(V; Y_q2 | Y_q1)` can far exceed `I(V; Y_q2)`.

**Diagnostic table** (which term each paper optimizes):

| | `H(Y)` term | `H(Y|V)` term | Combined? |
|---|---|---|---|
| LLMmap | stated (Eq.2), not optimized | stated (Eq.3), not optimized | no |
| T2I Provenance | not addressed | Corollary 1 addresses it | no |
| LLM Provenance Testing | entropy rejection sampling | not addressed (1 sample per cell) | no |

---

## 4. The core data structure: the separability tensor

Everything happens on one object. Building it is the only expensive step; every
algorithm comparison afterwards is table arithmetic.

```
Layer 0   R[q][v][s]  = raw response text
          |Q| × |L| × |S_build|            ← the expensive artifact
             ↓ frozen embedding E
Layer 1   1024-d vectors
             ↓ group by (q, v) over configs
Layer 2   P(q,v) = { E(o) : o ~ s(LLM_v(q)), s ∈ S_build }
          a POINT CLOUD of |S_build| vectors — not a single point
             ↓ two-sample statistic per pair
Layer 3   S[q][(v,v')] ∈ [0.5, 1.0]        ← what we optimize on
          |Q| × C(n,2)
```

**Why the point cloud is load-bearing.** A two-sample separability statistic has
between-group separation in its numerator (= `H(Y)` = Eq. 2) and within-group
spread in its denominator (= `H(Y|V)` = Eq. 3). **One scalar fuses both stated
objectives with no hand-tuned weight.** Collapse the cloud to a point and the
denominator disappears — leaving discrepancy-only selection, which is exactly the
limitation we identified in Paper 3's rejection sampling.

**Note on sampling.** One draw per `(q, v, s)` triple is sufficient. The 75
configs already mix config variation and sampling stochasticity, which is exactly
what an attacker faces. Repeated draws at fixed config are needed *only* for the
optional `H(Y|V)` decomposition diagnostic (B2), on a small subset.

---

## 5. The algorithm

### 5.1 Set coverage is a MAX

```
cov(pair) = max over selected queries of S[q][pair]
```

One query that separates a pair suffices. Under mean/sum, a query that is weak on
average but *uniquely* covers a hard pair is discarded — the exact behavior we
are fixing.

### 5.2 Objective is CVaR over pairs

```
maximize  CVaR_γ ( { cov(p) : p ∈ pairs } )
        = mean of the worst γ-fraction of pairs
```

`γ = 1.0` recovers the mean objective, i.e. **the baseline is a special case of
our method**, which makes γ a clean continuous ablation axis rather than a
competing alternative.

### 5.3 GreedyCover — where `k` comes from

```
GreedyCover(S, Q_pool, pairs, k, γ):
    Q_sel ← ∅
    cov[p] ← 0 for all p
    repeat k times:
        for each q not yet selected:
            cov_q[p] ← max(cov[p], S[q][p])
            gain[q]  ← CVaR_γ(cov_q) − CVaR_γ(cov)
        q* ← argmax gain
        Q_sel ← Q_sel ∪ {q*};  cov[p] ← max(cov[p], S[q*][p])
    return the nested chain Q^(1) ⊂ ... ⊂ Q^(k),  cov
```

Two properties worth noting:

- **This is the entire answer to "how do we get down to k".** The pool grows
  across rounds; `Q_sel` is always exactly `k`. A larger pool does not lengthen
  the strategy — it makes each round's `k` picks better.
- The greedy chain is **nested**, so it yields the accuracy-vs-k curve for free.
  That curve is our primary claim's figure (see §2a).

### 5.4 CDQD — the outer loop

```
PHASE 0 (paid once)
    R ← collect traces over (Q_0 × L × S_build)
    S ← separability(R)
    Q_pool ← Q_0

REPEAT rounds t = 1..T:

  (A) SELECT
      chain, cov ← GreedyCover(S, Q_pool, pairs, k, γ)

  (B) DIAGNOSE
      hard     ← pairs in the lowest γ-quantile of cov
      frontier ← { p ∈ hard : max over ALL of Q_pool of S[q][p] < θ }
                 ── frontier = an AMMUNITION problem, not a selection problem
                 ── frontier is also a deliverable in its own right

  (C) MODEL SELECTION on held-out configs
      score_t ← CVaR_γ of coverage recomputed on S_val
      keep best-so-far   ← REQUIRED: greedy is NOT monotone under ground-set
                            growth, even though the optimum is
      early-stop if score_t − score_{t−1} < ε

  (D) TARGETED GENERATION  (two-stage, cost-controlled)
      clusters ← merge overlapping hard pairs into confusion clusters
      for each cluster C:
          prompt G with: C's members
                       + queries that score high on C   (few-shot positive)
                       + queries that score low on C    (few-shot negative)
          Q_cand ← G.generate(N)                                     [N ≈ 40]
          STAGE 1: score Q_cand on |C| ≈ 3 models only        ← cheap
          keep top n_keep by S_val                            [n_keep ≈ 3]
      STAGE 2: survivors only pay the full-universe trace cost ← expensive
      Q_pool ← Q_pool ∪ survivors      (monotone growth; never shrinks)
      S      ← S with |survivors| new rows

RETURN best-so-far chain, frontier, Q_pool, S
```

**Two-stage cost arithmetic** (n=42, |S_build|=75, N=50, |C|=3, n_keep=3):

| | generations |
|---|---|
| naive: score all N on the full universe | 50 × 42 × 75 ≈ 157,500 |
| CDQD stage 1 (cluster only) | 50 × 3 × 75 ≈ 11,250 |
| CDQD stage 2 (survivors only) | 3 × 42 × 75 ≈ 9,450 |
| **total** | **≈ 20,700 (−87%)** |

**Feedback to the generator is few-shot semantic, not a numeric gradient.** This
is deliberately cheaper than the GCG-style white-box token optimization the paper
proposes as future work, and it keeps the output in natural language — preserving
the stealth property that matters for the threat model.

**Overfitting guard.** A generated query can score well on its target pair and be
useless or harmful elsewhere. Two defenses: survivors are filtered on `S_val`
(held-out configs), and once promoted they must earn their place through marginal
CVaR gain on the full pair set — greedy will simply not select a query that
covers one pair while displacing coverage elsewhere.

### 5.5 Final validation — closed-set accuracy under the paper's own trained pipeline

Everything in §5.1–5.4 is computed on the separability tensor using a
lightweight stand-in classifier (nearest-point-cloud over the frozen I5
embedding) — never LLmap's actual inference pipeline. This is deliberate
during selection: retraining LLmap's stage-2 network per candidate query set
would be both circular (fit on the exact models under test) and expensive,
exactly what CDQD exists to avoid (§6.3). But it means no number produced by
§5.1–5.4 is the number the paper reports, or the number a reviewer asks for
first — that number requires one further step, run once selection is done.

**Trace construction must match the released code exactly, not just the
embedding model.** `LLMmap/inference.py:92–113` embeds the query text and
the response text **separately** (`emb_queries = E(queries)`,
`emb_outs = E(answers)`), then concatenates them **in embedding space**
per query slot: `trace = [E(query) ; E(response)]`, 2048-d, one per
selected query, stacked into a length-`k` sequence fed to the self-attention
network. **§5.1–5.4's tensor stores response embeddings only (1024-d,
`§4` Layer 1)** — this is not an oversight there: at fixed `q`, `E(query)`
is an identical constant across every model being compared, so it cancels
exactly in any distance-based statistic (`‖(a,c)−(b,c)‖ = ‖a−b‖`) and
carries zero information for a linear probe. It does **not** cancel here —
the self-attention network needs `E(query)` to know *which* of the pool's
queries produced each token, exactly the information that lets it
generalize across strategies that select different `k`-subsets of the
pool. **This D must additionally embed each candidate strategy's selected
query texts** with the same cached I5 model before training — cheap
(≤259 static strings, no new generation, no I7 schema bump) but not
optional.

**Protocol.** For each candidate query strategy under comparison — the
CVaR-coverage chain (`k=1..8`), the mean-greedy `γ=1` baseline chain,
random-`k`, and the paper's own 8 queries — train LLmap's own stage-2
pipeline from scratch (the projection `f_p: 2048→384` — consuming the
full concatenated trace above, not the raw 1024-d embedding — plus the
small self-attention siamese/classifier network, **3,023,653 params
measured**, D009's pilot, 2026-09-09) on that strategy's `k` selected
queries' `[E(query) ; E(response)]`
traces, using the same `S_build` / `S_val` / `S_test` split (I2) as every
other measurement in this project. Hold the training procedure —
architecture, hyperparameters, epochs, optimizer — **identical** across
strategies; only the queries used to build the training corpus vary.
Evaluate closed-set accuracy on `S_test`, touched once (I2).

This is the "hold embed/classify constant, vary only query strategy"
comparison this project has referred to throughout (`PAPER_DEVIATIONS.md`
item 3) — it is what makes the headline comparison one of *query
strategies*, not of *classifiers*, and it is the step that confirms or
falsifies whether the separability-tensor gains from §5.1–5.4 survive
contact with a real trained classifier. **Do not treat a proxy-metric gain
(§5.1–5.4) as the result — it is the hypothesis this step tests.**

**Cost.** One training run per strategy — `§6.3`'s ~5–15 runs, not ~372.
This is where CDQD's training-cost saving becomes an actual measured
number rather than an argument.

**Relationship to §6.2.** Even run this way, absolute accuracy will not be
directly comparable to the paper's reported 95.35% (three-way split vs. the
paper's two-way) — state §6.2's caveat in the same breath as any accuracy
number this protocol produces.

---

## 6. Known caveats — read before writing anything up

### 6.1 The (1−1/e) guarantee does NOT hold under CVaR

- per-pair coverage `max_{q∈Q} S[q][p]` — monotone submodular ✓
- **mean** over pairs — non-negative linear combination of submodular functions,
  still submodular → greedy has (1−1/e) ✓
- **min or CVaR** over pairs — a min of submodular functions is generally **not**
  submodular ✗

This is *robust submodular maximization*: NP-hard and **not approximable to any
constant factor**. Ways to handle it, in order of preference:

1. Present the framing as the contribution: query selection is naturally robust
   submodular maximization, and **LLMmap is implicitly solving its average-case
   relaxation.** This is a stronger claim than "we changed the aggregation."
2. Use γ as a continuous knob and report the whole range — γ=1.0 end has the
   guarantee, γ→0 end does not.
3. Bicriteria approaches (SATURATE-style: spend `k·(1+log …)` elements to reach
   the optimal `k`-element robust value) if a guarantee is needed.

> ⚠️ **D4: the specific citations for CVaR/risk-averse submodular hardness and
> approximation are UNVERIFIED and must be checked against the literature before
> any writeup.** Do not cite from this file.

### 6.2 Our numbers will not be directly comparable to 95.35%

LLMmap's Algorithm H.1 selects on `T_test` and reports on the same distribution.
We use a three-way split (I2 in `DECISIONS.md`) and touch `S_test` once. Our
protocol is stricter, so **our absolute numbers may look lower even where the
method is better.** Every writeup must state this, or reviewers will misread it.
This applies to §5.5's trained-pipeline numbers as much as to §5.1–5.4's
tensor-level ones — a stricter split, not the training step, is the source of
any gap.

### 6.3 The cost claim must be stated carefully

Correct framing: **CDQD decouples query selection from inference-model training,
turning algorithm iteration from hours into seconds.**

| | training runs | generations |
|---|---|---|
| LLMmap Algorithm H.1 | ~372 (50+49+…+43) | ~157K |
| CDQD | ~5–15 | ~315K + increments |

We save GPU *training* time and spend *more* inference. Inference parallelizes
trivially and is checkpointable; the corpus is built once and reused by every
subsequent algorithm comparison. But "20–70× cheaper" is **wrong** and must not
be claimed.

### 6.4 Risk of a degenerate universe

If the model universe contains no genuinely hard pairs, `CVaR_γ` numerically
equals the mean and our method becomes indistinguishable from the baseline —
**with no error message.** See invariant I1 and the mandatory hard-tail check.

### 6.5 On the "expectation baseline" — do not claim to reproduce LLMmap's algorithm

The natural experiment to justify CVaR-coverage is: same candidate query pool,
same downstream pipeline, only the *selection algorithm* differs — compare
against "the expectation-based algorithm the paper describes."

**This cannot be built as a faithful reproduction.** Per Fact 1 (§1), Eq. 2/3
(inter-model discrepancy / intra-model consistency) are never actually
implemented in the paper or the released repo — the distance function `d(·,·)`
is never instantiated, and Algorithm H.1's real objective is downstream
classifier accuracy. Writing a baseline called "the paper's expectation
algorithm" would mean inventing weights and a distance function ourselves —
an unfounded reconstruction, not a reproduction, and fragile to reviewer
pushback ("you're comparing against a strawman you built").

**Resolution — the baseline is not new code.** It is `GreedyCover` (§5.3) with
`γ = 1.0`. Because each tensor cell `S[q][p]` is already a point-cloud-vs-
point-cloud statistic (invariant I3) whose numerator is between-group
separation and denominator is within-group spread, taking the mean over pairs
at `γ = 1.0` is the most faithful available analogue to an "expectation-style"
objective — with no invented weighting on top of what the tensor already
represents.

**Framing requirement for any writeup.** State this explicitly as *"our
faithful reconstruction of mean-based selection,"* never as *"reproducing
LLMmap's algorithm."* The paper does not have an algorithm to reproduce here —
this must not be glossed over in Figure 1's caption or in the text
surrounding it.

---

## 7. Deliverables

1. **The selected strategy** — `k` queries, plus the nested `k=1..8` chain.
2. **The identifiability frontier** — model pairs no query in the pool can
   separate. A genuine negative result that draws the method's boundary honestly.
3. **Reusable artifacts** — the trace corpus and separability tensor. These
   outlive this particular study and support follow-up work.
4. **Closed-set accuracy under the paper's own trained pipeline** (§5.5), per
   candidate strategy — the number that is actually comparable (mod §6.2's
   split caveat) to the paper's reported 95.35%, and the number a reviewer
   asks for before any of 1–3 above.

## 8. Metrics to report (all four, always)

| metric | why |
|---|---|
| mean top-1 accuracy | comparability with the paper; **expect little gain — say so** |
| worst-class accuracy | the actual target of a CVaR objective (paper's 84%) |
| hard-subset accuracy | on the difficult subset defined by the original confusion matrix |
| **queries needed to reach X%** | primary claim; the dimension with real headroom |

Reporting mean accuracy alone while optimizing the worst tail is a metric
mismatch and will (rightly) be caught. **State which classifier produced
these numbers.** Computed cheaply on the tensor-level stand-in classifier
(§5.1–5.4) during selection, these four metrics are proxies; computed via
§5.5's trained pipeline, they are the real result. A writeup must report
both and label which is which — not silently report the cheap one as if it
were the other.
