# Dreams (round 2)

The round-1 "you reinvented X" surfaces are all closed and every new number traces.
So this round is *not* about defense — it is about converting the two soft spots the
critics left open into the paper's two sharpest, most-cited objects:

1. **The XRC framing problem (the only real wound).** The Deployment section says
   embeddinggemma has the "lowest non-degenerate reading cost (XRC50 3.5×)" — but it is
   **6th of 8 finite-XRC models** (granite 1.25×, bge-m3 2.0×, LaBSE 2.48×, SapBERT
   2.63×, qwen3 3.25× all beat it). The correctness critic flags this as a hard
   self-contradiction (N2). The *minimal* fix (drop the superlative) is correct but
   wasteful: it throws away a contribution. The real prize is to find the **cross-lingual
   cost story that still supports deploying embeddinggemma** — i.e. a frontier or a
   conditional cost that the cheap-but-weak models (granite, SapBERT) *lose* on. That is
   the headline dream of this round and it is CPU-only.

2. **RRC is "renamed recall" until the re-ranker-bound becomes the object** (novelty #1).

3. **"Degenerate" is used 4× as a load-bearing qualifier and never defined** (cohesion #4,
   correctness N2 leans on it). A principled cutoff is a tiny new contribution.

4. **The "two line-items of the cross-lingual tax" spine never reached the prose**
   (cohesion #3). One sentence — but there is a *measurable* version that is better than a
   sentence.

I solve each at least once, then push past it.

## Problems on the table (distilled from the 3 critics)

- **P1 (correctness N2, novelty residual, cohesion #4).** XRC50-as-superlative is false
  for embeddinggemma (6th of 8 finite). Need a cross-lingual-cost story that (i) is not
  dominated by cheap weak models and (ii) still names embeddinggemma. *This is the round.*
- **P2 (novelty #1, correctness #2-RRC).** RRC@K is the CDF of first-foreign-twin rank =
  cross-lingual mate-hit@K renamed. The *contribution* must be the re-ranker-budget object
  (a frontier / a knee / a bound), not the quantity. Novelty critic's hand-off route #1
  asks for exactly this: make RRC a curve with dRRC/dK and a knee.
- **P3 (cohesion #4, correctness N2, novelty implicit).** "Degenerate" needs a principled,
  stated cutoff — used to exclude gte/e5 from "non-degenerate" summaries 4×, never defined.
- **P4 (cohesion #3).** The "two line-items of the same bill" spine (reading-cost tax on
  the CLIR benchmark, confusability tax on the alias-graph benchmark) is in story.md but
  not the prose; §6.1 and §6.2 are sequenced, not unified.
- **P5 (correctness N1).** "French and English statistically indistinguishable as targets"
  — no test was run. Needs either a softened word or an actual test.
- **P6 (correctness T-NEW).** XRC is a *population-level* ratio (D_same over 57
  same-language-gold queries, D_cross over 137 cross-gold queries), not a paired
  within-query ratio. A hostile reviewer reads "scan 3.5× as many documents" as paired.
- **P7 (novelty #2, optional).** "XRC monotone-invariant" asserted, not defended (half a
  sentence: a ratio of depths is invariant to any monotone re-scoring).
- **P8 (cohesion #2).** de↔zh asymmetry (+0.23) is a named metric and a stated number with
  no figure and no caption mention — round-1 orphan, still open.
- **P9 (cohesion #1, B2 residual).** §6.2 line 605 "Even the best model reaches RBO 0.39"
  contradicts the "any model" phrasing at abstract/intro/conclusion.

---

## (a) New analyses

### [feasible-now] A1. Robustness-vs-cost frontier — the answer to P1
- **what / how.** Stop reporting XRC as a scalar leaderboard column (where granite "wins").
  Plot every model in a 2-D plane: **x = XRC50 (reading-cost tax, lower better)**, **y =
  CLIR@10 (cross-lingual capability, higher better)**, with marker size = separability AUC.
  Compute the **Pareto frontier**. The story inverts cleanly: granite (XRC50 1.25×) is
  cheap *because it retrieves so little that the few twins it does find are shallow* —
  CLIR@10 0.329; embeddinggemma is the **only model on the upper-left frontier** that pairs
  a low-ish reading cost (3.5×, far under nomic 11.5× / e5 97.75×) with the *top* CLIR@10
  (0.50). The honest claim becomes: "among models worth deploying (CLIR@10 ≥ τ),
  embeddinggemma has the lowest reading cost." That is true and survives N2. The cheap
  models are cheap *conditional on being bad*; the frontier makes that visible in one panel.
  Data already on disk: `xrc_per_model.csv` (XRC50) + `headline_numbers.csv` (CLIR@10, sep).
- **cost.** ~30 lines of matplotlib + a Pareto sweep; one new figure; rewrites the N2
  sentence into a frontier claim. Closes **P1, N2** and upgrades C4. No new eval.
- **novelty payoff.** Converts a *false superlative* into a **true frontier claim** and
  gives the paper its missing "cost-vs-capability" deployment object — the thing an
  industry reviewer actually wants. No prior cross-lingual cost-capability frontier exists.

### [feasible-now] A2. Cost-per-recovered-twin (CPRT) — normalize XRC by what it buys
- **what / how.** XRC alone is gameable: a model that finds *almost no* twins has a tiny
  XRC because the handful it finds are shallow (granite, SapBERT). Define a *yield-adjusted*
  reading cost: **CPRT(m) = D_cross^50(m) / mate-hit@1000(m)** — depth you pay per fraction
  of foreign twins you actually recover. Granite's low D_cross is now divided by its low
  yield; embeddinggemma's slightly higher depth is divided by the best yield (0.94 of twins
  recovered by 1000). This re-orders the table so the deployable model is cheapest *per twin
  it actually returns*, killing the "cheap weak model wins" artifact at the metric level
  rather than via a frontier picture. All inputs on disk (`xrc_per_model.csv` D_cross,
  `rrc_per_model.csv` RRC@1000 = mate-hit@1000).
- **cost.** ~15 lines; one new column in Table 1 or a tiny figure. Closes **P1**.
- **novelty payoff.** A *yield-normalized* cross-lingual reading cost — no precedent. Makes
  the deployment claim a single defensible number instead of a frontier the reviewer must
  read off a scatter.

### [feasible-now] A3. The "degenerate" boundary, measured — answer to P3
- **what / how.** Define degeneracy operationally and *plot the gap that justifies the
  cutoff*: rank all nine models by Recall@10 and show the bimodal split — gte (0.004) and
  e5 (0.178 overall, CLIR@10 0.077) sit far below a clean gap from SapBERT (0.212) /
  LaBSE (0.277). The principled rule (see metric M1 below): a model is **degenerate** if its
  retrieval entropy / coverage falls below a stated floor (e.g. CLIR@10 < 0.10 *or* finds <
  X% of any gold in top-1000). Show the histogram with the cutoff line. This earns the word
  "degenerate" the four times it is used, and the figure *is* the justification.
- **cost.** ~20 lines; one small figure or an inset; one sentence in §5 Setup defining the
  term. Closes **P3, cohesion #4**, and hardens N2 (granite is provably non-degenerate;
  gte/e5 provably are).
- **novelty payoff.** A reusable, stated **collapsed-encoder detector** for CLIR
  leaderboards — small but genuinely new and immediately reusable by other benchmark
  builders. Turns an ad-hoc adjective into a one-line criterion.

### [feasible-now] A4. The re-ranker-budget frontier (dRRC/dK + the knee) — answer to P2
- **what / how.** The novelty critic's route #1. Turn RRC@K from two scalars (K=100, K=1000)
  into a **curve** RRC(K) per model, then compute (i) the **marginal recoverability**
  dRRC/dK and (ii) the **knee** K* past which a deeper re-ranker pool buys almost nothing
  (e.g. the K where dRRC/dK drops below some ε, or the elbow by max-distance-to-chord). The
  deployment read-off: "for embeddinggemma, re-ranking deeper than K* ≈ <value> recovers
  < ε per extra 100 candidates; the residual 5.84% is structurally lost." Pair it with XRC
  on the same axes for a **2-D cross-lingual cost frontier** (reading-depth cost ×
  re-ranker-recoverable fraction) — a single planning object: "how deep must my first stage
  go before re-ranking is worth it, for this model." Inputs: the first-foreign-rank lists
  already underlie `rrc_per_model.csv`; the CDF is recomputable at arbitrary K from them.
- **cost.** ~40 lines; one figure (RRC curves + knees) or fold the knee into cp_fig16. If
  the raw rank lists aren't dumped, recompute RRC at a K-grid from the same source the
  `extra_xrc` script used. Closes **P2, novelty #1**.
- **novelty payoff.** Converts RRC from "renamed recall" into a **per-model re-ranker-budget
  planning tool** — the novelty critic explicitly calls this "the highest-leverage novelty
  upgrade still on the table" with "no precedent I can find." This is the single best
  novelty move available and it is CPU-only.

### [feasible-now] A5. The two-tax decomposition table — answer to P4 (measurable spine)
- **what / how.** Don't just *say* "two line-items"; *show* them side by side. One small
  table, one row per model, two columns: **Reading-cost tax** (XRC50, from the CLIR
  benchmark) and **Confusability tax** (sibling-confusion rate, from the alias-graph
  benchmark). The unifying claim becomes empirical: the two taxes are **only weakly
  correlated across models** (a model can be cheap to read yet easily confused, or
  vice-versa) — which is *why both benchmarks are needed*. If they were redundant, one
  benchmark would do; the low cross-tax correlation is the quantitative justification for
  the two-benchmark design. (Compute the rank correlation between the two columns; report it.)
- **cost.** ~20 lines + 5 cells of prose; uses `xrc_per_model.csv` + alias confusion column.
  Closes **P4, cohesion #3** at the *story* level (better than the one-sentence fix).
- **novelty payoff.** Turns the "two benchmarks, one paper" cohesion seam into a
  *measured* claim: the two taxes are non-redundant, so the two-benchmark asset is
  necessary, not padded. This is the kind of self-justifying design argument reviewers love.

### [feasible-now] A6. Directional-hub significance / "near-tie" test — answer to P5
- **what / how.** Either (a) downgrade the wording to "within 0.01" (the minimal fix), or
  (b) **actually run the test the word implies**: a paired bootstrap / permutation over the
  per-cell column-mean differences (fr-target vs en-target recall across the directed cells)
  and report a CI on the fr−en gap (0.008). With per-cell N as small as 2–7, the honest
  result is almost certainly "CI spans 0," which *legitimizes* the word "indistinguishable"
  with a number behind it. Either closes N1; (b) is stronger.
- **cost.** (a) one word; (b) ~15 lines bootstrap over the directional matrix cells + one
  CI in caption. Closes **P5, N1**.
- **novelty payoff.** Tiny, but (b) converts an over-claim into a *defended* claim and adds
  a CI to a headline near-tie. Pre-empts the "indistinguishable under which test?" reviewer.

### [feasible-now] A7. Paired XRC (within-query reading cost) — answer to P6
- **what / how.** Address T-NEW head-on: the current XRC is population-level (D_same over 57,
  D_cross over 137). Compute a **paired XRC** *restricted to the 57 original queries that
  have both a same-language gold and a foreign twin*: for each such query, depth-to-
  same-gold and depth-to-first-foreign-twin, then the per-query ratio (or the ratio of
  paired medians). Report it alongside the population XRC. If the paired number is close to
  3.5×, that *strengthens* the claim ("the multiplier holds even within the same query");
  if it differs, the paper reports the honest paired version as the headline and the
  population one as the broader picture. Either way the "you scan 3.5× as many documents"
  sentence becomes a *paired* statement and the reviewer objection evaporates.
- **cost.** ~25 lines (needs per-query first-foreign-rank + per-query same-gold depth on the
  57 originals — both already computed inside the `extra_xrc` machinery). Closes **P6,
  T-NEW**. If the per-query depths aren't dumped, this is a small recompute, not a re-eval.
- **novelty payoff.** Makes XRC a *paired within-query* cost like the +0.55 home advantage —
  methodologically cleaner and matches the B3 footnote discipline already in the paper.

### [feasible-now] A8. Confusability-tax × reading-cost-tax joint failure on the SAME model
- **what / how.** A8 is the cross-benchmark version of the joint-failure cut. For each model,
  ask: when it pays a *high* reading-cost tax (high XRC50), does it also pay a *high*
  confusability tax (high sibling-confusion)? Plot the two taxes per model and annotate
  embeddinggemma (low-ish on both) vs e5 (catastrophic XRC, high confusion) vs granite
  (cheap XRC but mid confusion). This is the "two line-items, one bill" picture made
  per-model and tied to the deploy pick. Complements A5's correlation with a labeled scatter.
- **cost.** ~20 lines; one figure; reuses A5's two columns. Closes **P4** with a visual.
- **novelty payoff.** A single panel that shows *why embeddinggemma* across both taxes — the
  deployment pick justified on the unified two-tax plane, not on a contested superlative.

### [feasible-now] A9. "What does the median reader actually pay?" — absolute-depth companion
- **what / how.** XRC is a *ratio*; ratios hide magnitudes (3.5× of depth 2 = depth 7 is
  cheap; 3.5× of depth 50 would not be). Report the **absolute** D_cross^50 next to XRC50 so
  the reader sees embeddinggemma pays "depth 7" in absolute terms — cheaper in *absolute*
  documents-read than several lower-XRC models whose same-language depth is already large.
  This is another route to defusing N2: granite's XRC 1.25× may sit on a *worse* absolute
  D_cross. Check `xrc_per_model.csv` for the absolute D_cross column; if granite's absolute
  D_cross > embeddinggemma's 7, then embeddinggemma genuinely reads *fewer absolute
  documents* to its foreign twin — a TRUE superlative that rescues the deployment sentence.
- **cost.** ~10 lines + a column; possibly recovers a true superlative for free. Closes
  **P1/N2** if the absolute numbers favor embeddinggemma (worth checking first).
- **novelty payoff.** Distinguishes *relative* from *absolute* reading cost — a clean
  methodological point and possibly a no-cost rescue of the deployment claim.

---

## (b) New metric definitions

### [feasible-now] M1. DEG — the degeneracy index (principled "degenerate" cutoff) — P3
- **formula / intuition.** A model is **degenerate** if it fails to populate a usable
  candidate pool at all. Define
  `DEG(m) = 1` iff `CLIR@10(m) < δ_recall` **or** `mate-hit@1000(m) < δ_cover`
  (it retrieves almost no cross-lingual gold *and* almost no foreign twins even at depth
  1000), with δ stated once (e.g. δ_recall = 0.10, δ_cover = 0.10). Equivalently, frame it
  as a **retrieval-entropy / coverage floor**: a degenerate encoder's score distribution is
  near-uniform (it "retrieves almost nothing"), so its robustness metrics are *trivially*
  good (gte "wins" MT-robustness/parity precisely because uniform scores are language-
  invariant — the cohesion critic and the WTA contamination footnote already note this).
  DEG formalizes *why* gte/e5 must be excluded from "non-degenerate" summaries.
- **what it captures that existing metrics don't.** A *single explicit gate* that separates
  "robust because aligned" from "robust because it does nothing" — the failure mode that
  contaminates every cross-lingual robustness composite (WTA wins, parity wins). No existing
  CLIR metric encodes this; people exclude collapsed models by eyeball.
- **cost.** One definition sentence in §5 + a `DEG` flag column. Anchors all 4 "degenerate"
  uses and the WTA caveat. **feasible-now.**

### [feasible-now] M2. CCF — cross-lingual cost frontier (the deployable-cost metric) — P1
- **formula / intuition.** Don't report XRC as a scalar to be minimized; report
  **CCF = the Pareto frontier in (XRC50, CLIR@10) space**, and define a model's
  **frontier status** (on-frontier / dominated) plus its **frontier reading cost** =
  the minimal XRC50 achievable at its CLIR@10 level or higher. embeddinggemma's claim:
  "no model achieves both a lower reading cost *and* a higher CLIR@10" → it is
  Pareto-optimal; granite is dominated on capability. This is the metric form of A1.
- **what it captures that existing metrics don't.** A *capability-conditioned* reading cost.
  Plain XRC rewards models that retrieve nothing; CCF rewards only models that are cheap
  *given they are good*. Directly fixes the degenerate-models-dominate problem at the metric
  level. No cross-lingual cost-capability frontier metric exists in the literature.
- **cost.** ~30 lines (Pareto sweep) + figure. Closes **P1, N2**; upgrades C2/C4.

### [feasible-now] M3. RRK / dRRC — re-ranker knee + marginal recoverability — P2
- **formula / intuition.** `dRRC/dK(m)` = marginal foreign-twins recovered per unit of
  re-ranker depth; the **knee** `K*(m) = argmax` distance-to-chord of the RRC(K) curve (or
  the smallest K with `dRRC/dK < ε`). Plus the **structural loss** `L∞(m) = 1 − RRC@K_max`
  (the fraction unrecoverable by *any* depth). Report (K*, RRC@K*, L∞) per model.
- **what it captures that existing metrics don't.** Recall@K is a single point; (K*, L∞)
  is the *shape* of the recoverability curve — it tells a practitioner the depth past which
  re-ranking is wasted money and the floor below which alignment is the *only* lever. This
  is the re-ranker-bound-as-object the novelty critic asked for: RRC stops being "recall
  renamed" and becomes a budget curve with a knee and an asymptote.
- **cost.** ~30 lines + one figure. Closes **P2, novelty #1**. The strongest novelty metric.

### [feasible-now] M4. CXT — cross-lingual tax (two-line-item composite, *descriptive only*) — P4
- **formula / intuition.** `CXT(m) = (reading-cost tax, confusability tax)` = the *pair*
  (XRC50 on CLIR benchmark, sibling-confusion on alias-graph benchmark), reported as a
  2-vector, never collapsed to a scalar (keeping the CLIR-MRS demotion discipline). The
  contribution is the **claim that the two components are non-redundant** (low rank
  correlation across models), which is the empirical content of "two line-items of the same
  bill."
- **what it captures that existing metrics don't.** It names and *measures* the cross-lingual
  tax as a two-axis object spanning both benchmarks, and proves the two benchmarks are not
  redundant. This is the metric backbone of the unifying spine.
- **cost.** ~15 lines + correlation number + the A5 table. **Explicitly NOT a composite** —
  reported as a pair, sidestepping the CLIR-MRS critique. Closes **P4**.

### [needs-eval] M5. gXRC — guaranteed XRC (conformal reading-depth bound)
- **formula / intuition.** The Conformal-RAG horizon the paper already flags as future work,
  made minimal: calibrate a depth D̂ such that, with coverage 1−α, a foreign twin is found
  by depth D̂(α) — a *distribution-free guarantee* on the reading-cost tax, vs the current
  empirical median. `gXRC_α = D̂_cross(α) / D̂_same(α)`.
- **what it captures.** Turns XRC from a descriptive median into a deployment SLA ("with 90%
  confidence you read at most D̂ documents to reach the foreign twin"). The novelty critic
  notes Conformal-RAG is cited *only* as machinery; a minimal realization would make the
  guarantee a result.
- **cost.** Needs a held-out calibration split + conformal calibration on the rank lists —
  a real (if light) experiment. **needs-eval / backlog** (already `XRC-conformal` in
  needs_eval.md). Do NOT make the paper depend on it; it is upside.

### [feasible-now] M6. LSE — language-self-entropy (degeneracy via score collapse)
- **formula / intuition.** An alternative degeneracy detector with no threshold on recall:
  `LSE(m)` = entropy of the retrieved-document-language distribution, or the variance of the
  model's cosine scores over a query's top-1000. A degenerate encoder returns near-uniform
  scores (low score variance, high language entropy) — it "retrieves almost nothing
  distinctive." Flag DEG when LSE exceeds a floor. Complements M1 with a *score-side* test
  that doesn't depend on gold labels.
- **what it captures that existing metrics don't.** A *label-free* collapsed-encoder
  detector — usable even when no qrels exist. Could let the paper claim degeneracy is
  detectable from the score distribution alone.
- **cost.** Needs the raw score lists (likely on disk for the separability AUC). ~20 lines.
  **feasible-now** if scores are dumped; else light recompute. Closes **P3** from a second
  angle and adds a label-free contribution.

---

## (c) Answers to the feedback

### [paper-framing-only] C-N2-min. Drop the false superlative (the safe fix) — closes: correctness N2
- **what/how.** Replace "has the lowest non-degenerate reading cost (XRC50 3.5×)" with the
  critic's defensible line: "is the best twin-finder (median first-foreign rank 5) and keeps
  a low reading cost (XRC50 3.5×, vs 11.5×–97.75× for nomic/e5)." No new computation.
- **cost.** One sentence. Closes N2 immediately. **Always do this as the floor;** A1/A2/A9
  are the upgrades that make the claim *positive* instead of merely *not-false*.
- **novelty payoff.** None on its own — but it unblocks the paper. Pair with A1 for novelty.

### [feasible-now] C-N2-frontier. The frontier rescue (the real fix) — closes: N2 + upgrades C4
- **what/how.** = A1 + M2. Report the (XRC50, CLIR@10) frontier; state embeddinggemma is
  Pareto-optimal and the cheaper models are cheap *only because* they retrieve less. The
  Deployment sentence becomes: "Among deployable models (CLIR@10 ≥ τ), embeddinggemma has the
  lowest reading cost; the lower-XRC models read shallowly only because they retrieve little
  (granite CLIR@10 0.329 vs 0.50)."
- **cost.** One figure + rewrite. Closes N2 *and* answers the conductor's central question
  ("what IS the right cross-lingual-cost story that still supports deploying it").
- **novelty payoff.** The capability-conditioned cost frontier — a genuinely new deployment
  object and the round's best framing upgrade.

### [feasible-now] C-RRC. Re-ranker-bound as the object — closes: novelty #1
- **what/how.** = A4 + M3. (i) Add the one-clause hedge the novelty critic wrote: "RRC@K is
  the cumulative first-foreign-twin hit rate (mate-hit@K on cross-lingual queries); our
  contribution is reading it as a per-model re-ranker ceiling: 1−RRC@K is provably
  unrecoverable." (ii) Then *exceed* the hedge by adding the RRC curve + knee K* + L∞, so the
  contribution is the *budget object*, not the renamed quantity.
- **cost.** One clause (framing) + one figure (the curve/knee, feasible-now). Closes
  novelty #1 at two levels: the hedge makes the *current* claim honest; the curve makes it
  *novel*.
- **novelty payoff.** RRC stops being "recall renamed" and becomes a re-ranker-budget
  planning tool — the novelty critic's #1 recommended upgrade.

### [feasible-now] C-DEG. Define "degenerate" once — closes: cohesion #4, hardens N2
- **what/how.** = M1 + A3. Add one sentence at first use (§6.1 XRC beat or §5 Setup):
  "We call a model *degenerate* if CLIR@10 < 0.10 and it recovers < 10% of foreign twins by
  depth 1000 (gte-base, e5-large-instruct); we exclude these from 'non-degenerate' summaries
  throughout." Then every later "non-degenerate" and the WTA contamination footnote are
  anchored. Optionally show the bimodal histogram (A3).
- **cost.** One sentence (+ optional inset figure). Closes cohesion #4 and makes N2's
  "granite is non-degenerate" provable.
- **novelty payoff.** A reusable collapsed-encoder criterion — small but new and reusable.

### [paper-framing-only] C-N1. Soften or test the directional near-tie — closes: correctness N1
- **what/how.** Minimal: "statistically indistinguishable" → "nearly tied (fr 0.375 ≈ en
  0.367, within 0.01)." Stretch (A6b): add a bootstrap CI on the fr−en gap and keep the word
  with a number behind it.
- **cost.** One word (min) or ~15 lines (test). Closes N1.
- **novelty payoff.** None for the min fix; small credibility gain for the test.

### [feasible-now] C-T-NEW. State XRC is population-level, or pair it — closes: correctness T-NEW
- **what/how.** Minimal: one clause in the XRC paragraph/caption — "D_same is over the 57
  same-language-gold queries and D_cross over the 137 cross-gold queries (a population-level,
  not paired, ratio)." Stretch (A7): compute the *paired* XRC on the 57 originals and report
  it as the headline, mirroring the B3 home-advantage discipline.
- **cost.** One clause (min) or ~25 lines (paired). Closes T-NEW.
- **novelty payoff.** Methodological cleanliness; the paired version makes XRC a within-query
  cost like the home advantage.

### [paper-framing-only] C-P7. Defend XRC monotone-invariance — closes: novelty #2
- **what/how.** Add the half-sentence: "a ratio of retrieval depths is invariant to any
  monotone re-scaling of similarities, unlike an AUC or a weighted composite — XRC measures
  *how deep you read*, a quantity no score normalization can hide." One line.
- **cost.** One sentence. Closes novelty #2.
- **novelty payoff.** Strengthens XRC's selling point rather than defending an over-claim.

### [paper-framing-only] C-P4. The two-tax spine sentence — closes: cohesion #3
- **what/how.** Minimal (cohesion critic's line): at the head of §6.2, "If §6.1 measured what
  cross-linguality *costs to read*, the alias-graph benchmark measures what it *costs in
  precision* — the second line-item of the same bill: a look-alike compound that out-ranks
  the gold." Stretch: back it with A5/M4 (the measured non-redundancy of the two taxes).
- **cost.** One sentence (min) or + A5 table (measured). Closes cohesion #3.
- **novelty payoff.** None for the sentence; the measured version (A5/M4) makes the
  two-benchmark design *self-justifying*.

### [paper-framing-only] C-P8. Close the de↔zh asymmetry orphan — closes: cohesion #2
- **what/how.** Fold +0.23 into the cp_fig03 caption — "the most asymmetric directed pair is
  de↔zh (+0.23; asymmetry panel not shown)" — or reference cp_fig04 in the anisotropy
  paragraph with a half-sentence.
- **cost.** ~6 words. Closes cohesion #2.
- **novelty payoff.** None — pure orphan cleanup; but a named instrument should not float
  with no payoff.

### [paper-framing-only] C-P9. Fix the §6.2 "best model" RBO regression — closes: cohesion #1
- **what/how.** Rewrite line 605 to match the abstract/intro/conclusion "any model"
  phrasing: "The best cross-lingual RBO any of the nine models reaches is only 0.39
  (Figure ag_rbo) — a ceiling no model beats." Fixes both the framing mismatch and the
  rhetorical inversion (0.39 is a ceiling, not an achievement).
- **cost.** One sentence. Closes cohesion #1.
- **novelty payoff.** None — consistency fix; but it is the one real internal-framing
  contradiction left and is load-bearing for the B2 story.

---

## Wild cards (highest upside, clearly tagged)

### [feasible-now] W1. Deployment-cost simulator: translate XRC×RRC into a dollar/latency number
- **what/how.** The novelty critic's route #2 ("validate XRC against a downstream cost").
  Build a tiny *analytic* cost model — no eval, pure arithmetic over existing depths: for a
  cross-jurisdiction search workload, end-to-end cost ≈ (candidates retrieved) × (re-rank
  cost per candidate) + (LLM tokens at depth D). Show that to hit a target foreign-twin
  recall, embeddinggemma needs to feed the re-ranker a *shallower* pool (because XRC50 3.5×
  and RRC@100 0.74) than e5 (XRC50 97.75×, RRC@100 far lower), so embeddinggemma's
  end-to-end re-rank/LLM budget at fixed recall is the lowest *among deployable models*.
  This makes XRC a **validated headline** (it predicts end-to-end cost) and retires the
  CLIR-MRS-external-validation backlog as moot — the novelty critic says point that eval at
  XRC instead.
- **cost.** ~40 lines of arithmetic + one figure/table; *no model runs* (it's a cost formula
  over depths already on disk, with stated per-candidate cost assumptions). Closes the
  "XRC is just documents-read, is that a real cost?" gap.
- **novelty payoff.** XRC becomes the *validated* cost metric tied to deploy spend — the
  paper's most industry-credible contribution, and it kills the only remaining composite-
  validation debt. **Highest upside of the wildcards; entirely CPU/arithmetic.**

### [feasible-now] W2. "Cheapness is a confound" inversion — the granite trap as a *finding*
- **what/how.** Make the N2 wound a *result*: the very fact that the cheapest XRC models
  (granite 1.25×, SapBERT 2.63×) are the *weak* ones is the paper's sharpest warning to
  practitioners — **a low reading-cost tax can mean your retriever is too weak to find the
  twins at all.** Add one analysis: correlate XRC50 with CLIR@10 across the non-degenerate
  models; if the correlation is *positive* (cheaper reading cost ↔ lower capability among
  the weak tail), that is a quotable trap: "do not minimize XRC blindly — the global minimum
  is a model that retrieves nothing." This flips the whole N2 problem into a contribution.
- **cost.** ~15 lines + one correlation + a sentence. Reuses A1's data. Closes **P1**
  by *owning* it.
- **novelty payoff.** A counter-intuitive deployment lesson ("the cheapest reader may be the
  worst retriever") that no cross-lingual paper has stated — memorable and defensible.

### [needs-eval] W3. Alignment causal probe: does a tiny alignment nudge move XRC/RRC?
- **what/how.** The paper's thesis is "align, don't re-rank." A minimal causal test: take one
  model, apply a cheap post-hoc cross-lingual alignment (e.g. a learned linear map / mean-
  centering per language on a held-out parallel set), and show XRC50 drops and RRC@100 rises
  — i.e. the *lever the paper recommends actually moves the cost metrics*. Even a single
  before/after pair on one model would turn "align, don't re-rank" from a correlational
  inference into a demonstrated intervention.
- **cost.** Requires re-embedding / a fitted alignment map + re-retrieval on one model —
  a real (small) experiment. **needs-eval / backlog.** Do NOT make the paper depend on it;
  flag as the natural next step that XRC/RRC were built to measure.
- **novelty payoff.** Would elevate C3 from "the deficit is at the embedding level
  (correlational)" to "and closing it at the embedding level measurably reduces the
  reading-cost and re-ranker-loss taxes (causal)." Big, but out of scope this round.

### [feasible-now] W4. The "no-home" ablation as the cleanest XRC story
- **what/how.** Spanish has 34 queries with zero Spanish gold — pure cross-lingual. Compute
  XRC and RRC *restricted to the no-home Spanish slice* vs the rest. If the reading-cost tax
  and re-ranker loss are worst exactly where there is no same-language fallback, that is the
  paper's design paying off: the built-in stress test produces the sharpest cost numbers, and
  it sidesteps the population-ratio objection (P6) because the no-home slice has *no*
  same-language gold to ratio against — you report absolute depth there.
- **cost.** ~20 lines; reuses the Spanish split already in the benchmark. Closes **P6** from
  a third angle and showcases the no-home design.
- **novelty payoff.** Ties the headline cost metrics to the benchmark's signature design
  choice (the no-home Spanish stress test) — makes the asset and the metric reinforce each
  other.

---

## Top-3 recommended for this round (editorial pick across channels)

**#1 — A1 + M2 + W2: the cross-lingual cost FRONTIER (capability-conditioned reading cost).**
This is the round. It directly answers the conductor's central question — the right
cross-lingual-cost story that still supports deploying embeddinggemma is a **robustness-vs-cost
frontier**: embeddinggemma is Pareto-optimal in (XRC50, CLIR@10), and the lower-XRC models
(granite 1.25×, SapBERT 2.63×) are cheap *only because* they retrieve too little to find the
twins — which W2 turns into a quotable deployment trap ("the cheapest reader may be the worst
retriever"). [feasible-now], ~1 figure + a rewritten sentence, no eval. Closes the only hard
correctness MISMATCH (N2), upgrades C4, and gives the paper a genuinely new deployment object.

**#2 — A4 + M3 (C-RRC): the re-ranker-budget frontier (RRC curve + knee K* + structural loss
L∞).** The novelty critic explicitly names this "the highest-leverage novelty upgrade still on
the table" with "no precedent I can find." It converts RRC from "renamed recall" into a
per-model re-ranker-budget planning tool: dRRC/dK gives the depth past which re-ranking is
wasted, L∞ gives the floor only alignment can move. Pair it with the one-clause honesty hedge
so the *current* claim is honest and the *curve* is the novelty. [feasible-now], CPU-only on
the existing rank lists, one figure.

**#3 — A5 + M4 + C-P4 + C-DEG (M1): the unifying spine, made measurable, plus the degeneracy
gate.** Deliver the "two line-items of the same bill" spine not as a sentence but as a
*measured* non-redundancy claim (XRC reading-cost tax × sibling confusability tax are weakly
correlated across models → both benchmarks are necessary), and define "degenerate" once
(DEG: CLIR@10 < 0.10 ∧ <10% twins by depth 1000) so the four load-bearing uses and the WTA
contamination footnote are anchored. [feasible-now / paper-framing], a tiny table + two
definitions. Closes cohesion #3 and #4 and turns the two-benchmark structure from a seam into a
self-justifying design argument.

*Floor (do regardless, ~5 one-line edits): C-N2-min, C-N1, C-T-NEW clause, C-P7, C-P8, C-P9 —
the minimal critic fixes that unblock the paper even if none of the Top-3 land. The Top-3 are
the upgrades that make each fix a contribution instead of a patch.*
