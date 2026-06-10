# Dreams (round 3)

The paper is converging: 0 correctness mismatches, novelty defensible, cohesion
near-clean. So this round is **two jobs at once**: (1) close the handful of named
fixes the critics left on the table, and (2) **raise the ceiling** — invent
genuinely novel analyses/metrics that strengthen the four strongest contributions
(the XRC cost axis, the $L_\infty$ alignment-only floor, the content-controlled
parallel benchmark, and the chemistry-confusability findings). Every idea is
tagged `feasible-now` / `needs-eval` / `paper-framing-only`, with cost + the
critic problem it closes + the novelty payoff. The preference, per the conductor,
is **feasible-now CPU-only ideas computable from the rank/score data the
`extra_*` dirs already emit** — those are the ones that can actually run before
submission.

## Problems on the table (distilled from the 3 critics)

**Small remaining fixes (all critics agree these are the closing nits):**
- **P1 (cohesion #1, must-fix).** "**eight** non-degenerate models" at lines 557
  and 657 contradicts the DEG gate (which makes "non-degenerate" = the precise
  7-set). The count 8 is *correct for those two populations* (all-but-`gte`), the
  *word* "non-degenerate" is wrong. One-phrase relabel each.
- **P2 (correctness C-NEW, non-blocking).** Fig.21 caption/prose says
  "**sibling**-confusion rate" but the plotted/quoted quantity is the general
  `confusion_rate` (any look-alike, sibling OR parent, out-ranks gold; values
  0.182/0.068 correct). Delete "sibling-".
- **P3 (novelty #2, near-mandatory cite).** The RRC **knee** $K^*$ is established
  cascade / re-rank-depth prior art and is currently uncited; add one
  cascade/recall-ceiling reference so only the *cross-lingual quantification +
  $L_\infty$ floor* is claimed.

**Smaller polish the critics also flagged (cheap, optional):**
- **P4 (novelty #1).** "presented on a capability-conditioned Pareto frontier" can
  read as claiming the *frontier presentation* as novel — add a half-clause:
  "standard Pareto frame, XRC is the new axis." (optional cite: syftr / RAG
  cost-frontier).
- **P5 (novelty #3).** "alignment-only floor" rests on a correlational chain; once,
  surface the inferential step ("a floor no re-ranker can move; only representation
  alignment can, per §7") rather than baking it into the adjective.
- **P6 (cohesion #2).** §6.1 RRC beat names e5 (a *degenerate* model) as the
  $L_\infty$ range upper end ($0.372$) inside a "non-degenerate" object — relabel
  e5 as the degenerate illustration, or give the non-degenerate max.
- **P7 (cohesion #3 / correctness T-MINOR).** Float order of cp_fig19 (cosmetic);
  universal-blind language list "fr/zh/de" → "fr and zh" (de/es tie at 3).
  Home-advantage hyphenation harmonize. All cosmetic.

**The ceiling-raising mandate (the real work this round).** The critics handed the
dreamer two explicit upside routes:
- **Novelty route 1 (their #1 pick, zero compute):** make the cost frontier
  **cross-lingual-pair-resolved** — one frontier *per directed language pair / per
  query language* — so the deployable object becomes "the capability corner is a
  *different model* for en→de than for es→zh." A frontier whose membership *moves
  with the language direction* is a new deployment artifact no generic Pareto plot
  can be. CPU-only from existing score lists.
- **Novelty route 2 (highest value, needs one small run):** turn the $L_\infty$
  "alignment-only" adjective into a *demonstrated* causal result via the
  PENDING-EVAL alignment probe.

Everything below is organized around (a) new analyses, (b) new metrics, (c) answers
to the feedback — with the ceiling-raisers front-loaded and the closing fixes in (c).

---

## (a) New analyses

### [feasible-now] A1. Per-route cost frontier — the frontier that *moves* with language direction
- **What / how.** Today the cost frontier (Fig.18) is one point per model in global
  $(\mathrm{XRC50}, \mathrm{CLIR@10})$ space. Recompute **both axes restricted to a
  single query-language (or directed query→doc pair)** from the same rank lists the
  `extra_directional_hub` and `extra_xrc_reading_cost` dirs already produce:
  for each query language $\ell$, recompute $\mathrm{XRC50}_\ell$ (median cross/same
  depth ratio over queries asked in $\ell$) and $\mathrm{CLIR@10}_\ell$, then compute
  the Pareto frontier **per $\ell$**. Output: a small-multiples grid (5 panels, one
  per query language) or a single plot with arrows showing how each model's point
  *moves* across routes, plus a table "frontier membership by route." The headline
  object is the **route-conditional capability corner**: e.g. egemma may own en→*
  but bge-m3 or granite may own the homeless es→* route.
- **Cost.** CPU-only, pure re-aggregation of existing first-foreign-rank and
  coverage-depth arrays; no new model runs. ~1 script mirroring
  `extra_cost_frontier.py` with a `groupby(query_language)`. n is thin per cell
  (the Limitations thinness caveat applies — report as *indicative per-route*, like
  the directional matrix), so frame descriptively.
- **Closes.** Novelty critic's #1 over-claim ("standard Pareto plot") *and* their
  explicit route-1 upside; serves the §8 "do not reflexively ensemble / per-language
  routing" recommendation with a concrete artifact.
- **Novelty payoff.** Converts the cost frontier from "standard frame, new axis"
  into a **genuinely new object** — a *per-route deployment map*. Lets the paper
  newly claim: "the single best model is route-dependent; a deployment that fixes
  one model is Pareto-dominated on at least one language direction" — exactly the
  router argument, now grounded.

### [feasible-now] A2. The $L_\infty$ floor decomposed by language direction — *where* is recall structurally lost?
- **What / how.** $L_\infty = 1-\mathrm{RRC@1000}$ is currently one number per model
  (egemma 0.058 … e5 0.372). Recompute $\mathrm{RRC@1000}$ **per directed
  language-pair** from the same first-foreign-rank array, giving an
  $L_\infty$ *matrix* (query-lang × doc-lang) per model. The new claim: the
  alignment-only floor is **not uniform** — it concentrates on specific directed
  edges (almost certainly the hardest en→de / *→zh edges and the homeless es
  routes). This is the structural-floor analog of the directional CLIR matrix.
- **Cost.** CPU-only; reuses the first-foreign-twin rank list already computed for
  RRC. One heatmap per model (or one for the deployed egemma + a pooled one).
- **Closes.** Strengthens the $L_\infty$ novelty (novelty C3/C2) by showing it is a
  *structured* object, not a scalar; pre-empts "is the floor just a few bad
  queries?" — answers *which* routes.
- **Novelty payoff.** New claim: "the unrecoverable floor is a **property of
  specific language directions**, so alignment effort should be spent
  edge-by-edge" — actionable, and no prior cross-lingual recall-ceiling work
  resolves the floor per direction.

### [feasible-now] A3. Re-ranker ROI curve — marginal twins recovered *per unit depth* (the derivative of RRC)
- **What / how.** The knee $K^*$ is read off the RRC curve by eye/an existing
  detector. Make the economic argument explicit: define and plot the **marginal
  recovery** $\Delta\mathrm{RRC}(K) = \mathrm{RRC@}K - \mathrm{RRC@}(K/2)$ (twins
  recovered per doubling of re-rank depth) — the discrete derivative of the budget
  curve. The knee is where this drops below a stated threshold; the *area to the
  right of the knee* is the wasted re-rank budget. This turns "deeper pool buys
  almost nothing" from an adjective into a **plotted, quantified ROI curve** and a
  single number: "past $K^*$, each doubling of depth recovers $<X\%$ more twins."
- **Cost.** CPU-only; a transform of the existing `rrc_knee.csv` curve. Trivial.
- **Closes.** Novelty #2 (knee is incremental/uncited) — by giving the knee a
  *quantified marginal-ROI reading* it stops being "Elastic's blog" and becomes a
  cross-lingual budgeting instrument; pairs naturally with the cascade cite (P3).
- **Novelty payoff.** Lets the paper claim a **deployment ROI rule** ("re-rank to
  depth $K^*$; beyond it, $<X\%$ marginal recall per 2× cost") rather than a knee
  location — a budgeting number a practitioner can act on.

### [feasible-now] A4. The confusability–separability bridge *on the alias-graph*, edge-resolved
- **What / how.** The load-bearing $+0.96$ separability↔CLIR link lives on the
  cross-lingual benchmark; the alias-graph already has AUC 0.55 (confused) vs 0.70
  (not) globally. Cut it **by confuser relation** (sibling vs parent, already a
  column in `extra_confusion_severity`): compute gold-vs-look-alike separability AUC
  *split by sibling vs parent*. Prediction (testable now): siblings drive the
  separability collapse (AUC near 0.5 for sibling confusers, higher for parents),
  mirroring the 18.1% vs 6.2% confusion split. This ties the chemistry-confusability
  finding (C3) to the same separability mechanism as the cross-lingual floor — one
  mechanism, two benchmarks.
- **Cost.** CPU-only; the score arrays and the sibling/parent labels both exist in
  `extra_confusion_severity` / the joint-failure dump.
- **Closes.** Strengthens C3 (chemistry-confusability) and unifies it with the
  separability mechanism; answers "is confusion a *different* failure from
  cross-lingual collapse?" — no, same separability deficit, different axis.
- **Novelty payoff.** New unifying claim: **"both the cross-lingual floor and the
  chemical-confusability tax are the same embedding-level separability deficit,
  measured on two axes"** — strengthens the paper's single-mechanism story and the
  two-tax spine simultaneously.

### [feasible-now] A5. Structure-question penalty as a *cross-lingual* effect (interaction cut)
- **What / how.** Two known findings sit side by side but are never crossed:
  structure questions are the trap (R@10 0.26 vs role 0.60), and cross-lingual
  collapses. Cut **question-type × home-vs-foreign** on the alias-graph: is the
  structure-question penalty *worse cross-lingually* than same-language? i.e.
  compute MoLIR/CLIR (or own/foreign recall) separately for structure vs role
  questions. If structure questions lose *more* of their cross-lingual recall, the
  paper can claim the chemistry trap and the language trap **interact
  multiplicatively** — the worst case is a structure question crossing a language
  boundary, which is the universal-blind core (14/16 are structure + cross-lingual).
- **Cost.** CPU-only; both labels (question type, query/gold language) exist in the
  alias-graph per-query data.
- **Closes.** Deepens C3 and the §7 "structure-style questions are the trap"
  paragraph; gives the universal-blind finding (12%, 14 structure) a *quantified
  interaction* rather than a co-occurrence.
- **Novelty payoff.** New claim: **"the two failure axes are not additive — a
  structure question asked across a language boundary is the compounded worst
  case,"** with a number for the interaction. Memorable, chemistry-specific, and
  no other CLIR paper has the ontology+language crossing to measure it.

### [feasible-now] A6. Confuser-distance decay (minimal version of the "severity law")
- **What / how.** Limitations defers a "graded ChEBI hop-distance decay law"
  because the on-disk relation field is binary. But two levels (sibling=1 hop,
  parent=variable) already give the *first two points* of a decay curve: plot
  confusion-win-rate vs relation type ordered by graph distance (sibling < parent <
  other). The 2.9× sibling/parent ratio *is* the slope of a two-point decay. Frame
  it as "the first two points of a confusability-decay law: win-rate falls 2.9× per
  ontology hop class" — a minimal realization of the deferred law, not the full
  continuous version.
- **Cost.** CPU-only; already-computed `severity_split.csv`. Pure reframing + maybe
  one ordered bar.
- **Closes.** Partially retires the "Severity law" limitation — turns "future work"
  into "first two points measured, continuous law future work."
- **Novelty payoff.** Lets the paper claim a (coarse) **ontology-distance decay of
  confusability** rather than just a binary split — a small but novel quantitative
  law tying retrieval confusion to graph structure.

### [feasible-now] A7. Per-query "double-jeopardy" census (refine the joint-failure 257)
- **What / how.** The joint-failure analysis already classifies 257 confused cases
  (44.4% same-lang sibling). Add the *cross-product census*: of all queries, how
  many suffer **(a) language bias only, (b) confusion only, (c) both, (d) neither**,
  as a 2×2 table per model and pooled. The modal cell ("same-language sibling") is
  the (c) cell; the table makes the compounding *rate* explicit and quantifies how
  often the two traps co-fire vs fire alone.
- **Cost.** CPU-only; the per-query labels (was-confused, was-home-biased) exist in
  the joint-failure dump.
- **Closes.** Sharpens C3's "the two traps compound" claim from a modal statistic
  into a full contingency table; gives a clean "P(both | confused)" number.
- **Novelty payoff.** New claim with a clean number: **"$X\%$ of failures are
  double-jeopardy (language *and* confusion together), more than either trap
  alone"** — the compounding made a measured rate.

### [feasible-now] A8. Score-gap (margin) distributions, not just AUC
- **What / how.** Separability is reported as AUC (a rank statistic). Add the
  **raw cosine margin** distribution: $\mathrm{score(gold)} -
  \max_j \mathrm{score(look\text{-}alike}_j)$, same-language vs cross-language. AUC
  says "under-scored"; the *signed margin* says **by how much** and whether the
  cross-lingual gold is below the look-alike by a *small recoverable* gap or a
  *large structural* one. A bimodal/negative-shifted cross-lingual margin is the
  microscopic picture behind $L_\infty$: the floor twins have large negative margins
  no re-ranker reaches.
- **Cost.** CPU-only; the gold/non-gold score arrays already exist (they feed the
  AUC).
- **Closes.** Hardens the "under-scored, not mis-ordered" claim (C3 mechanism) —
  AUC alone can be attacked as ordinal; the margin gives the magnitude.
- **Novelty payoff.** New claim: **"foreign gold sits a measurable cosine margin
  *below* its confusers — the floor is a score-magnitude deficit, visible as a
  shifted margin distribution,"** the direct microscopic evidence for the
  alignment-not-re-ranking thesis.

---

## (b) New metric definitions

### [feasible-now] M1. RRC-AULC — Area Under the Loss Curve (a scalar that summarizes the whole budget curve)
- **Formula.** $\mathrm{AULC}(m) = \frac{1}{\log K_{\max}}\int_1^{K_{\max}}
  \big(1-\mathrm{RRC@}K(m)\big)\, d(\log K)$ — the average *unrecovered* fraction
  over log-depth, i.e. the area above the RRC curve. Low AULC = recovers most twins
  shallow; high AULC = needs deep pools or never recovers.
- **What it captures that existing metrics don't.** $L_\infty$ is the *asymptotic*
  floor and $K^*$ is the *knee location*; neither captures the **whole shape** —
  how fast and how cheaply a model climbs to its ceiling. AULC is one number that
  *ranks models by re-ranker-friendliness* (cheap-to-recover vs expensive),
  orthogonal to the floor. A model can have a low floor but a slow climb (recoverable
  but only with a deep, expensive pool) — AULC catches that; $L_\infty$ doesn't.
- **Cost.** CPU-only; a trapezoidal sum over the existing `rrc_knee.csv` curve.
  Adds one column to the RRC table.
- **Novelty payoff.** A single deployment scalar — "how re-ranker-friendly is this
  encoder" — that is *not* recall, *not* the floor, *not* the knee. Lets the
  leaderboard carry a re-ranker-economics column.

### [feasible-now] M2. ARI — Alignment Recoverability Index (decompose the gap into floor vs depth)
- **Formula.** For a model, split the cross-lingual shortfall $(1-\mathrm{RRC@}1000$
  treated as 1) into two additive parts at the chosen budget $K$:
  $\underbrace{\mathrm{RRC@}K}_{\text{cheap, re-ranker recovers}} +
  \underbrace{(\mathrm{RRC@}1000 - \mathrm{RRC@}K)}_{\text{expensive, deep pool}} +
  \underbrace{L_\infty}_{\text{alignment-only}} = 1$. Report the triple
  (recoverable-cheaply, recoverable-deeply, alignment-only) as a stacked bar per
  model. ARI $= L_\infty / (1-\mathrm{RRC@}K)$ = the share of the *remaining* gap
  that is alignment-only vs merely deeper-pool.
- **What it captures.** Turns the single floor into a **budget decomposition**: of
  everything a model misses at the practical depth $K$, what fraction is fixable by
  paying for a deeper re-rank pool vs *only* fixable by alignment. This is the exact
  "align vs re-rank" decision made quantitative *per model*.
- **Cost.** CPU-only; three reads off the existing RRC curve (RRC@K, RRC@1000,
  $L_\infty$).
- **Novelty payoff.** The headline thesis ("align, don't re-rank") becomes a
  **measured split**: "for egemma, of the twins missed at depth 100, $Y\%$ are
  recoverable only by alignment, $Z\%$ by a deeper pool" — the most direct possible
  operationalization of the paper's core recommendation.

### [feasible-now] M3. Route-XRC dispersion (XRC-σ) — a single number for "how route-dependent is reading cost"
- **Formula.** $\mathrm{XRC\text{-}}\sigma(m) = \mathrm{std}_\ell\,
  \mathrm{XRC50}_\ell(m)$ over the per-query-language XRC50 values from A1 (or the
  CV, $\sigma/\mu$, for scale-freeness). High dispersion = the model's reading cost
  is wildly uneven across language directions (a deployment risk a global XRC50
  hides); low dispersion = uniformly cheap/expensive.
- **What it captures.** The *global* XRC50 (3.5×) is a median that hides per-route
  variance. A model with XRC50 3.5× could be 2× on en→fr and 30× on es→zh — same
  median, very different deployment risk. XRC-σ surfaces that, complementing A1.
- **Cost.** CPU-only; a std over the per-route XRC50 array that A1 already produces.
- **Novelty payoff.** New deployment dimension: **route-stability of reading cost**.
  Lets the paper rank models not just by *how expensive* but by *how unevenly
  expensive* cross-lingual reading is — a fairness-across-routes notion no existing
  retrieval cost metric has.

### [feasible-now] M4. CCI — Confuser Concentration Index (how few attractors cause the damage)
- **Formula.** Over the confusion events, $\mathrm{CCI} =$ the Gini (or
  top-5-share) of confusion-wins across distinct look-alike compounds. The paper
  already names "universal attractors" (polypeptide, methyl, ethene, hydroxide,
  dioxygen) — CCI quantifies *how concentrated* the confusability tax is on a tiny
  attractor set. High CCI ⇒ a denylist of ~5 compounds removes most confusion;
  low CCI ⇒ diffuse, no easy fix.
- **What it captures.** The qualitative "small set of universal attractors"
  observation becomes a **number with a deployment lever**: if CCI is high, a
  cheap chemistry-aware denylist/boost on ≤5 attractors is the fix; if low, you need
  the encoder. No existing IR metric measures distractor concentration this way.
- **Cost.** CPU-only; the per-confuser win counts already exist
  (`ag_fig5_universal_attractors` source).
- **Novelty payoff.** New claim + lever: **"the confusability tax is concentrated:
  the top-5 attractors cause $W\%$ of all confusion (CCI $=\ldots$), so a 5-compound
  chemistry-aware guard recovers most of it"** — a concrete, cheap, chemistry-aware
  intervention the paper can recommend (and it directly serves the "needs
  chemistry-aware help, not more encoders" §8 line).

### [feasible-now] M5. LPF — Language-Parity Floor (worst-route CLIR, not mean)
- **Formula.** $\mathrm{LPF}(m) = \min_\ell \mathrm{CLIR@}10_\ell(m)$, the recall on
  the *worst* query language — the cross-lingual analog of a worst-group/min-over-
  groups fairness metric. Report it next to mean CLIR@10. The gap
  $\mathrm{CLIR@}10 - \mathrm{LPF}$ is the *route-fairness penalty*.
- **What it captures.** Mean CLIR@10 (0.50) is itself an average over routes; LPF is
  the SLA-relevant worst case — "what recall does your *worst-served* language get?"
  This is the natural sibling of the paper's own thesis ("averages hide collapse")
  applied one level deeper, to the average *across routes*.
- **Cost.** CPU-only; min over the per-route CLIR@10 from A1.
- **Novelty payoff.** Pushes the paper's central rhetorical move (don't average)
  onto the metric itself, and gives a **worst-route SLA number** — lets the paper
  say "egemma's mean CLIR is 0.50 but its worst-route floor is $X$," which is the
  number a deployment SLA actually needs.

### [paper-framing-only] M6. XRC as a coverage-conditioned curve (XRC(C)) — name the censoring honestly as a curve
- **Formula / framing.** Already-computed $D_C^{\text{same}}, D_C^{\text{cross}}$ at
  $C\in\{50, 90, 95\}$; instead of one XRC50 scalar, present **XRC as a function of
  coverage** $\mathrm{XRC}(C) = D_C^{\text{cross}}/D_C^{\text{same}}$, with XRC50 the
  finite anchor and $D_{90}/D_{95}$ shown as the *lower-bound tail* (right-censored,
  as already disclosed). The reading-cost *explodes with coverage* — that growth
  curve is itself the message ("90% coverage is far more than 3.5× the cost").
- **What it captures.** A single XRC50 understates the deep-coverage cost the paper
  already censors away; the curve shows the cost is *worse* than the headline,
  honestly, as a lower bound.
- **Cost.** None beyond what's computed; reframes existing $D_{90}/D_{95}$ from a
  caveat into a (censored) curve.
- **Novelty payoff.** Lets the paper claim the reading-cost penalty is
  *coverage-accelerating* — a stronger, still-honest statement than the median
  multiplier alone, with the censoring shown rather than buried.

---

## (c) Answers to the feedback

### [feasible-now] F1. Relabel the two "eight non-degenerate" sites — closes: cohesion #1 (the round's must-fix)
- **What / how.** Line 557 → "pooled over the eight models with a defined
  cross-lingual recall (all but the degenerate \texttt{gte-base})..."; line 657
  caption → "...for the eight models with a defined RRC curve (all but
  \texttt{gte-base}, whose candidate pool is empty)..." Keeps the count (correct),
  drops the gate-bound term "non-degenerate." ~6 words each. **No number changes.**
- **Cost.** Pure edit. **Novelty payoff.** None (hygiene) — but it protects the
  round's best cohesion move (the DEG gate) from self-contradiction.

### [feasible-now] F2. Delete "sibling-" in Fig.21 — closes: correctness C-NEW
- **What / how.** Fig.21 caption: "(confusion rate, alias-graph benchmark)";
  §6.2 prose already uses the general "confusability tax = a look-alike out-ranking
  the gold," so only the caption word changes. Leave the value 0.182/0.068 and the
  separate (correctly-labeled) sibling-vs-parent severity split untouched.
- **Cost.** One-word edit. **Novelty payoff.** None — removes a hostile-reviewer
  gotcha (the figure's own axis label says `confusion_rate`).

### [feasible-now] F3. Add the cascade / re-rank-depth citation for the knee — closes: novelty #2 (near-mandatory)
- **What / how.** In §2 (the metric/related paragraph that introduces RRC, or the
  Related-Work cross-lingual-ranking paragraph) add one cascade/recall-ceiling
  reference and one sentence: "The knee + diminishing-returns-with-depth shape is
  the established cascade / re-rank-depth result \citep{<cascade>}; our contribution
  is its *cross-lingual* quantification and the structural floor $L_\infty$." Prefer
  a peer-reviewed multi-stage dense-retrieval cascade paper that states the
  first-stage recall ceiling (the Elastic blog is the fallback the critic named).
  Then the knee is "credited and extended," not "rediscovered."
- **Cost.** One bib entry + one sentence. **Novelty payoff.** Converts the only
  *missing-citation* exposure into airtight credit; lets the paper keep the
  $L_\infty$ floor as the sole novel RRC object, cleanly.

### [paper-framing-only] F4. "Standard Pareto frame, XRC is the new axis" half-clause — closes: novelty #1 over-claim
- **What / how.** In the C2 bullet / Fig.18 caption / §6.1, add: "on the standard
  cost-vs-capability Pareto frame, whose cost axis is the new XRC reading-depth
  multiplier." Optionally cite syftr (arXiv:2505.20266) or the RAG cost-frontier
  (arXiv:2511.09545) once in §6.1. Disarms "you drew a Pareto plot."
- **Cost.** One clause (+ optional cite). **Novelty payoff.** Keeps the claim true
  and pre-empts the round's top (minor) over-claim; pairs perfectly with A1, which
  *earns* a stronger frontier claim.

### [paper-framing-only] F5. Surface the inference behind "alignment-only floor" once — closes: novelty #3
- **What / how.** At first use (abstract or §6.1), once write "$L_\infty$ — a floor
  no re-ranker can move; per the separability mechanism (\S\ref{sec:analysis}) only
  representation alignment can" so the correlational step is *visible*, not baked
  into the adjective. Leave the *number* (regression-checked) firm; only the
  *remedy attribution* is hedged. Elsewhere "alignment-only floor" can stay.
- **Cost.** One clause. **Novelty payoff.** Removes the last "strong adjective on a
  correlational chain" exposure without weakening the measured floor.

### [feasible-now] F6. Fix the e5-in-a-non-degenerate-object slip + the cosmetic nits — closes: cohesion #2, #3, correctness T-MINOR
- **What / how.** §6.1 line 649: relabel e5 as the degenerate illustration
  ("...to $0.372$ for the *degenerate* \texttt{e5-large-instruct}, which the gate
  excludes") OR give the non-degenerate max. Universal-blind list (line 939):
  "predominantly French and Chinese" (drop the de/es 3-tie). Home-advantage
  hyphenation → unhyphenated throughout. cp_fig19 float order: optional, do not
  spend a rewrite.
- **Cost.** A few one-line edits. **Novelty payoff.** None — final polish that
  honors the DEG-gate discipline the round established.

### [needs-eval] F7. The alignment-probe causal result — closes: novelty #3 *and* converts $L_\infty$ from inferred to demonstrated
- **What / how.** The Limitations already names it: fit a per-language linear
  alignment map (Procrustes / a small learned rotation) on *one* model's embeddings,
  re-retrieve, recompute XRC50 and RRC@K **before/after**. If $L_\infty$ (or XRC50)
  *drops under alignment* while staying flat under a re-ranker control, "alignment-
  only" becomes *demonstrated*. This is the single highest-value upgrade but it
  needs one (small, CPU/GPU-light) run, so it is backlogged — the paper must stand
  without it (and does).
- **Cost.** One alignment fit + one re-retrieval pass on the existing corpus
  embeddings (no model training). **Novelty payoff.** Turns the paper's headline
  thesis from correlational to **causal** — would be the most memorable result in
  the paper. Tag honestly as the post-submission win.

---

## Wild cards (highest upside, clearly tagged)

### [feasible-now] W1. The "decision-flip" map — does robustness reorder deployment *per route*?
The paper's whole thesis is "robustness reorders the recall ranking." Make it a
*map*: combine A1 (per-route frontier) with the leaderboard to show, **per query
language, which model a recall-only dashboard would pick vs which the
frontier/CLIR picks** — and count the routes where the choice *flips*. If a
recall-only dashboard would deploy model X on es→* but the frontier says bge-m3,
that's a *route where the standard practice ships the wrong model*, quantified.
CPU-only. **Payoff:** the paper's rhetorical core ("the dashboard is wrong")
becomes a counted, per-route fact: "on $R$ of 5 routes, recall-only selection
disagrees with the robustness-aware choice."

### [feasible-now] W2. A "minimum viable robustness panel" — the smallest metric set that preserves the ranking
The paper reports a *family* of metrics. Wild idea: empirically find the **smallest
subset** (greedy/exhaustive over the 81-cell leaderboard) that reproduces the
full-panel model ranking (e.g. by rank correlation ≥ 0.95). If 3 of the ~8 axes
(say CLIR@10, RBO, XRC) recover the full ranking, the paper can *recommend the
minimal panel* a team must actually compute — a concrete, novel "what to report"
artifact, not just "report more." CPU-only over the existing leaderboard.
**Payoff:** turns "report robustness, not mean recall" into a *specific, minimal,
validated dashboard spec* — far more actionable and more citable than "report the
whole family."

### [feasible-now] W3. Confusability-tax *currency conversion* — express both taxes in one unit (extra documents read)
The two-tax spine says the reading-cost tax (XRC, in docs) and the confusability
tax (a rate) are weakly correlated but resists unifying them (correctly, as a
*motivation* not an independence result). Wild bridge: express the **confusability
tax also in "extra documents read"** — the median extra depth you must scan past
the first confuser to reach the gold. Then *both* taxes are in the same unit
(documents), and the weak correlation becomes a statement about two
*commensurable* costs, sharpening the "second line-item of the same bill" metaphor
without claiming independence. CPU-only from the alias-graph rank lists.
**Payoff:** the two-tax metaphor becomes literally additive in a shared unit
("total reading bill = same-language depth + XRC penalty + confuser penalty"),
a genuinely new framing of the chemistry+language cost.

### [needs-eval] W4. Equivalence-audit-lite as a *separability sanity check* (turn a limitation into a robustness panel)
The parallel-gold equivalence audit is deferred (T4/Limitations). A *lite*,
no-human version usable as a confound check: for the parallel human-translated
twins, compute the **embedding cosine between the two language versions of the
same patent** and correlate it with whether that twin was retrieved. If retrieved
twins have higher cross-lingual self-similarity, the failure is alignment (not
gold non-equivalence) — a cheap, automatic partial answer to "are the golds really
equivalent?" Tagged needs-eval only because it touches raw corpus embeddings;
otherwise CPU. **Payoff:** partially retires the equivalence-audit limitation with
data, and *adds independent evidence* for the alignment thesis (twin-similarity
predicts twin-retrieval).

---

## Top-3 recommended for this round (editorial pick across channels)

1. **A1 — Per-route cost frontier (feasible-now, CPU-only).** This is the critics'
   own #1 upside *and* a true ceiling-raiser: it converts the cost frontier from
   "standard Pareto frame, new axis" (the round's top minor over-claim) into a
   **new deployment object** — a frontier whose membership *moves with the language
   direction*. It earns a stronger frontier claim than the F4 disclaimer alone,
   directly grounds the §8 per-language-routing recommendation, and is pure
   re-aggregation of score lists the `extra_*` dirs already emit. Pair it with
   **W1** (decision-flip count) to turn the paper's thesis into a counted per-route
   fact. Highest novelty-per-CPU-cycle on the table.

2. **M2 — ARI / Alignment Recoverability Index (feasible-now, CPU-only).** The
   most direct possible operationalization of the paper's headline thesis: it
   *decomposes* every model's cross-lingual shortfall into recoverable-cheaply /
   recoverable-deeply / **alignment-only** ($L_\infty$). This makes "align, don't
   re-rank" a *measured split per model* rather than a slogan, strengthens the
   $L_\infty$ novelty the novelty critic singled out, and is three reads off the
   existing RRC curve. Plotted as a stacked bar it is also the paper's most
   self-explanatory figure.

3. **The closing-fix bundle: F1 + F2 + F3 (all feasible-now).** Non-negotiable to
   land convergence — relabel the two "eight non-degenerate" sites (cohesion
   must-fix), drop "sibling-" in Fig.21 (correctness C-NEW), and add the one
   cascade/re-rank-depth citation for the knee (the single near-mandatory missing
   cite). Together they retire every blocking-or-near-blocking item the three
   critics raised, at a cost of one bib entry and a handful of one-line edits, and
   they protect the round's best moves (the DEG gate, the RRC budget object) from
   self-contradiction or an uncredited-prior-art attack.

**Honorable mention (best needs-eval):** **F7 (the alignment probe)** — the one
experiment that turns "alignment-only" from correlational to **causal** and would
be the paper's most memorable result. Backlogged; the paper stands without it.
