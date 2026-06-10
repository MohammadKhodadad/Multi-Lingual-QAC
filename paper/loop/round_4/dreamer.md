# Dreams (round 4)

**Operating mode this round is different.** All three critics converged on the
same verdict: the spine is done and novelty-defensible (2nd clean novelty round),
and the *only* live liability is **float overload** — 27 figures + 2 tables = **29
floats** (17 in §6) against an 8-page industry-track budget. So the conductor has
re-pointed my three channels:

- **(a) New analyses** → propose ONLY robustness / sensitivity checks that *harden
  an existing claim without adding a float*, plus the figure-consolidation plan.
- **(b) New metrics** → propose a new measure ONLY if it is a *strictly stronger
  replacement* for one the paper already carries (so the float count does not grow);
  otherwise say **freeze**.
- **(c) Answers to feedback** → the four concrete asks: the ρ_k residual-decomposition
  cite for ARI; the cp_fig22 "nine vs. seven" caption reconcile; the §6.1 ARI
  redundancy trim; and the concrete float-cut/merge plan.

Every idea is tagged `[feasible-now | needs-eval | paper-framing-only]` with cost +
novelty payoff, and biased — per the conductor — toward **hardening + trimming, not
new objects**. Nothing below adds a float; several *remove* one. Nothing makes a
core claim depend on a new run.

---

## Problems on the table (distilled from the 3 critics)

1. **Float overload (cohesion, the dominant concern).** 29 floats / 8-page body, 17
   in §6. Every figure is referenced (0 orphans) but *density*, not reference, is
   the test. Cohesion's ranked cut list: drop the two radars (cp_fig14 + ag_fig10),
   merge mate-retrieval (cp_fig06/07) → 1 two-panel, merge language-collapse
   (cp_fig09/10) → 1 two-panel. 29 → 25, no argument lost. **FREEZE new analysis.**
2. **ARI residual prior-art (novelty, the round's one real residual).** The
   "decompose-and-normalize a reranking remainder" *shape* has a published cousin in
   long-tailed reranking (ρ_k, arXiv:2604.01506). Near-mandatory: cite it, claim
   only the inverse (un-rerankable share) + the alignment floor cross-lingually.
3. **cp_fig22 caption "all nine models" vs. 7 bars plotted (correctness + novelty +
   cohesion all flagged it).** Identity holds for all 9 trivially; figure plots only
   the 7 non-degenerate. One-clause caption reconcile, no value change.
4. **§6.1 ARI read-off re-defines the identity §4 already owns (cohesion seam #1).**
   ~2 sentences of restatement on an over-budget paper. Open §6.1 at the *result*.
5. **Float-order nit (cohesion #3):** cp_fig22 (ARI) floats before cp_fig19 (RRC
   budget) it reads off. Cosmetic; swap the two `\begin{figure}` blocks.
6. **Latent robustness exposure (my read, not a critic ask):** three load-bearing
   scalars — the separability **r=+0.96 (n=7)**, the ARI/L∞ ordering
   (egemma 0.229 < qwen3 0.233, n=7 non-deg), and the two-tax / cheapest-reader
   correlations — all rest on n=5–7 with **no interval reported in-text**. The
   critics judged these *correctly hedged*, but a hostile reviewer's first move on a
   small-n correlation is "show me the CI / a leave-one-out." Hardening these with
   **CPU-only resamples reported as inline numbers (no new float)** is the single
   highest-leverage move available this round and is exactly the channel-(a) the
   conductor opened.

---

## (a) New analyses — robustness/sensitivity ONLY, zero new floats

The unifying design rule for this round: **every hardening result is reported as an
inline parenthetical or a single added table *row/column*, never a new figure.** A
bracketed interval next to an existing point estimate costs 0 floats and converts a
"descriptive on n=7" hedge into a "robust under resampling" claim — pure upside on
an over-budget paper.

### [feasible-now] A1 — Bootstrap CI on the separability correlation r=+0.96 (n=7), reported inline
- **what / how.** The paper's single load-bearing mechanism is
  $r(\text{cross-language AUC},\text{CLIR@10})=+0.96$, already shown "robust" by a
  drop-the-two-collapsers recompute (+0.958, n=7). Harden it one notch further:
  BCa bootstrap over the **7 non-degenerate model points** (resample models with
  replacement, recompute Pearson r, 10k draws), and report the 95% CI as a single
  parenthetical in §7 and the abstract-adjacent sentence. Because the troubleshooter
  already has `extra_correlation_robustness`, this is a ~15-line add to that script
  emitting one number; **no figure** — the result is a bracket: e.g. "+0.96
  (95% CI [·,·], n=7 model-bootstrap)".
- **cost.** CPU-seconds; one new scalar in an existing JSON; ~3 words of new prose.
  Closes: pre-empts the "n=7 correlation, show me an interval" reviewer; hardens C3.
- **novelty payoff.** Lets the paper say the mechanism is **robust under model
  resampling**, not just under one leave-2-out — the strongest small-n defense that
  costs no float. (If the CI is wide but strictly positive, that is *still* a win:
  "positive across all resamples" is the honest, sufficient claim.)
- **caveat tag.** Model-bootstrap on n=7 is itself coarse; report it as
  "model-level bootstrap, n=7, indicative interval" — the *sign-stability* (fraction
  of resamples with r>0) is the load-bearing read, not the interval width. State
  both; lead with sign-stability.

### [feasible-now] A2 — Leave-one-language-out (LOLO) on the two-tax correlation
- **what / how.** The two-tax non-redundancy ρ=−0.59 (n=7, p=.16, n.s.) is already
  honestly quarantined as descriptive. But the *interesting* hardening question for
  a 5-language study is not the model-axis n — it's whether the inverse relationship
  is an artifact of one language's confusion-tax estimate. Recompute the **confusion
  tax (alias-graph) leaving out one language at a time** (5 LOLO recomputes of the
  per-model confusion rate), re-rank, re-correlate with XRC50. Report the **range of
  ρ across the 5 LOLO folds** as one inline clause in §6.2. If ρ stays negative
  across all 5 folds, the "neither benchmark is a proxy for the other" read is
  robust to language composition; if it flips, the paper *honestly* downgrades the
  claim further (it is already n.s., so no downside).
- **cost.** CPU-only; the confusion-rate-by-language is already computed (it drives
  cp_fig21 / ag_fig2); LOLO is a re-aggregation. One inline range, no float.
  Closes: the "is the inverse two-tax an artifact of one language" confound; hardens
  the *descriptive* framing of C2/the two-tax bridge.
- **novelty payoff.** Turns "weakly, inversely correlated (n.s.)" into "inversely
  correlated and the sign is stable to dropping any single language" — a
  composition-robustness claim no current sentence makes, at zero float cost.

### [feasible-now] A3 — τ-sensitivity of the cost-frontier admitted set (the deployment rule's one free parameter)
- **what / how.** The deployment rule hinges on one untuned constant: the admission
  threshold τ=0.40, which yields the admitted set {bge-m3, qwen3-0.6B, embeddinggemma}
  and the "bge-m3 is the cheapest-to-read admitted model" claim. A reviewer will ask:
  *is the recommendation an artifact of τ=0.40?* Sweep τ over a sensible grid (e.g.
  0.30–0.50 in 0.05 steps) and report, **as one added sentence or a 3-row inline
  micro-table**, (i) the τ-range over which the admitted set is stable, (ii) the
  τ-range over which the cheapest-admitted-reader stays bge-m3, and (iii) the τ at
  which embeddinggemma stops being the unique max-CLIR corner (it never does — it's
  the global CLIR max, so that is τ-invariant by construction, which is itself the
  cleanest possible robustness statement). The Pareto *frontier membership* is
  τ-invariant (Pareto-optimality doesn't depend on τ); only the *admitted set* does.
- **cost.** CPU-only; `extra_cost_frontier` already has the (XRC50, CLIR@10) points;
  τ-sweep is a filter loop. One sentence / one 3-row inline table; **no new figure.**
  Closes: "your deployment rule depends on a hand-picked τ"; hardens C4.
- **novelty payoff.** Converts the τ=0.40 footnote-grade caveat into a *stated
  stability interval*: "the recommendation (capability corner = embeddinggemma;
  cheapest admitted reader = bge-m3) holds for all τ∈[·,·]." That is a genuinely
  stronger deployment claim than the current single-τ snapshot, and it pre-empts the
  most obvious "you tuned the threshold" objection.

### [feasible-now] A4 — Sign-stability ("vote") summary for the two non-significant correlations, in place of leaning on p
- **what / how.** Both n.s. correlations (two-tax ρ=−0.59; cheapest-reader ρ=+0.29)
  are currently defended by quoting p (.16, .53). p on n=7 is nearly content-free and
  a reviewer knows it. Replace/augment with a **leave-one-model-out sign-stability
  vote**: recompute each ρ over the 7 leave-one-out subsets (n=6 each) and report
  "ρ negative in k/7 (resp. positive in k/7) leave-one-out folds." This is a more
  honest small-n robustness statement than a p-value and reads as one clause.
- **cost.** CPU-only; trivial re-aggregation of points already on disk. Inline only.
  Closes: the "p on n=7 is meaningless" reviewer reflex; hardens the *honesty* of the
  two-tax bridge and the cheapest-reader caution without changing the (correct) n.s.
  framing.
- **novelty payoff.** Methodologically cleaner small-n reporting; lets the paper keep
  the descriptive framing while showing the *direction* is or isn't fold-stable.
  (If unstable, that strengthens the "do not over-read" message — no downside.)

### [feasible-now] A5 — ARI ordering robustness: is "egemma 0.229 < qwen3 0.233" a real gap or a tie?
- **what / how.** The ARI headline is egemma has the lowest ARI@100 (0.229), qwen3
  next (0.233) — a **0.004 gap**. The critics verified the numbers trace, but never
  asked whether 0.004 survives resampling. Since ARI@K = L∞/(1−RRC@K) is a *pure
  transform of RRC*, and RRC is an empirical hit-rate over the cross-lingual queries
  (n≈137), bootstrap the **per-query first-foreign-twin ranks** for egemma and qwen3,
  recompute ARI@100 per draw, and report **P(egemma ARI@100 < qwen3 ARI@100)** as one
  inline number. If it's high, "lowest of any non-degenerate model" is bootstrap-safe;
  if it's a near-tie, soften to "lowest, statistically tied with qwen3" — which the
  paper can absolutely afford because egemma's recommendation rests on the *frontier
  corner + smallest L∞*, not on the ARI tiebreak.
- **cost.** CPU-only; the per-query ranks underlying RRC already exist in the
  `extra_rrc_budget_frontier` / `extra_ari_decomposition` inputs. One inline
  probability; no float. Closes: "is 0.229 vs 0.233 a distinction without a
  difference?"; hardens C2/C3 ARI.
- **novelty payoff.** Makes the *one new scalar* in the paper bootstrap-defensible at
  the margin where it's closest, which is exactly where a careful reviewer pokes.

### [feasible-now] A6 — XRC50 stability under bootstrap of the depth-population (hardens the headline cost number)
- **what / how.** XRC50 is the headline cost (egemma 3.5×). It's a *population-level*
  ratio of median depths over 57 same / 137 cross queries — the paper already, very
  correctly, flags D90/D95 as right-censored and only reports the median. Add a
  **bootstrap CI on XRC50 itself** for the three frontier members (egemma, bge-m3,
  granite) by resampling the same/cross query depth populations and recomputing the
  ratio of medians. Report as a bracket on the three frontier points only (inline /
  in the existing cost paragraph), not a figure. Confirms the 3.5× / 2.0× / 1.25×
  ordering is stable, not a single-sample artifact.
- **cost.** CPU-only; depth populations already computed for `extra_cost_frontier`.
  Three inline brackets. Closes: "XRC50 is one median; how stable?"; hardens C2/C4.
- **novelty payoff.** Turns the headline reading-cost multiplier from a point
  estimate into an interval-backed one, on the three models the recommendation
  actually rests on — cheap credibility on the paper's signature metric.

### [paper-framing-only] A7 — Reframe the cut radar/merged panels' lost content as one sentence, so the cut costs nothing
- **what / how.** When the radars (cp_fig14/ag_fig10) are cut (see C-d below), the
  half-sentence they carried ("where each top model wins") should be *absorbed*, not
  dropped: append "(`embeddinggemma` leads consistency and separability; the
  per-axis detail is in Tables 1–2 and Figure~\ref{fig:cp_ribbon})" to the leaderboard
  paragraph. This makes the cut argument-neutral — the reviewer never feels a gap.
- **cost.** ~1 sentence; **removes 2 floats** net. Closes: the float-overload concern
  while preserving the "where each wins" beat. novelty payoff: none (pure trim
  hygiene) — but it's what makes the cut *safe*.

---

## (b) New metric definitions — strictly-stronger-replacement test, else FREEZE

The conductor's bar this round: a new metric is allowed **only if it strictly
replaces an existing one** (no float growth). I ran every level-2 idea (CTC, CERC,
LSR, ELI, ARGF) through that filter. **Verdict: FREEZE — none clears the bar this
round.** The metric family (CLIR@k, home, directional, mate, RBO, collapse, sep-AUC,
XRC, RRC, ARI, DEG) is complete, the critics call it converged, and any new scalar
would *add* a definition paragraph and likely a float. Below I record the two that
came *closest* to a strict replacement, so the conductor can see the filter was
honestly applied — but both are tagged **do-not-add-this-round**.

### [needs-eval | do-not-add] B1 — ARGF as a strict replacement for the per-route figure's "existence of corner movement"
- **idea.** An **Alignment-Robust Generalization Floor**: the worst-case ARI@100
  across the 5 routes, $\mathrm{ARGF}(m)=\max_\ell \mathrm{ARI@}100_\ell(m)$, as a
  single per-model number that *summarizes* the per-route frontier without the
  thin-n XRC axis. It would let the paper state route-robustness as one scalar and
  potentially **retire the per-route figure** (a float).
- **why FREEZE.** (i) It rests on the *same* thin per-language cells (es n=0) the
  critics flagged as indicative; a max-over-routes scalar *amplifies* the thinnest
  cell rather than hedging it — exactly the over-claim the four-point contract avoids.
  (ii) cp_fig23 is the round's *new, load-bearing, ×2-referenced* float the critics
  explicitly say **stays**. Replacing it with a scalar that leans on es n=0 trades a
  well-hedged figure for an ill-hedged number. Net: **do not add; freeze.**

### [feasible-now | do-not-add] B2 — A single "robustness gap" scalar to replace the aggregation ribbon (cp_fig17)
- **idea.** Report embeddinggemma's **rank-range [1,4]** (already in the text) plus a
  one-number "capability-minus-robustness rank gap" and *retire the ribbon figure*,
  saving a float.
- **why FREEZE / hand to cut-list instead.** The ribbon is doing real work (it's the
  evidence the recommendation does NOT rest on the composite — a key honesty move,
  and cohesion lists cp_fig17 as *covering* the radar story, so cutting cp_fig17
  would *undermine* the radar cut). The rank-range [1,4] is already stated inline, so
  the *float* is arguably redundant with the prose — but this is a **cut decision**
  (channel c), not a new metric. I route it to the cut-list as a *tertiary*
  candidate, not a metric proposal. **No new metric.**

**Bottom line (b): FREEZE.** The metric family is closed; the right move is to
*harden* the existing scalars (channel a) and *trim* floats (channel c), not to mint
a new measure. I record B1/B2 only to show the replacement filter was applied.

---

## (c) Answers to the feedback — the four concrete asks

### [feasible-now] C-a — Add the ρ_k residual-decomposition citation for ARI
- **closes:** Novelty over-claim #1 (the round's one real residual prior-art); the
  single "missing citation, not a wording choice" surface this round.
- **what / how.** Add the bib entry the novelty critic handed over verbatim:
  ```
  @article{residualrerank2026,
    title  = {Beyond Logit Adjustment: A Residual Decomposition Framework for
              Long-Tailed Reranking},
    author = {Wang and others},
    journal= {arXiv preprint arXiv:2604.01506},
    year   = {2026}
  }
  ```
  Cite it once in the ARI paragraph (§4, ~line 466) with the credit-and-distinguish
  half-clause: *"Normalizing a re-ranking remainder by a recoverable gap has a
  precedent in long-tailed reranking \citep{residualrerank2026}; we invert it (the
  un-rerankable share rather than the reranker's recoverable gain) and tie it to
  representation alignment cross-lingually, with an alignment-only floor that ρ_k has
  no analogue for."* Optionally also drop a one-clause back-reference in §2.
- **cost.** 1 bib entry + ~1 clause. **novelty payoff:** converts ARI's only "you
  reinvented residual decomposition" surface into credited-and-distinguished —
  airtight on novelty, since ARI is the *inverse ratio, different domain, with an
  alignment floor ρ_k lacks*. Near-mandatory; cheap; do it.

### [feasible-now] C-b — Reconcile the cp_fig22 caption ("nine" vs. 7 bars)
- **closes:** Correctness D-NEW + novelty over-claim #2 + cohesion (all three flagged
  the same line).
- **what / how.** The identity *is* true for all 9 (it's arithmetic), but the figure
  plots only the 7 non-degenerate bars. Use the correctness critic's exact minimal
  fix — change line 703 from "The three sum to $1.0$ for all nine models." to:
  > *"The three sum to $1.0$ for every model (the identity closes for all nine; the
  > figure shows the seven non-degenerate)."*
  **Change no value.** The two degenerate models (e5-large-instruct, gte-base) are
  the ones omitted from the panel per `degenerate_models_excluded_from_figure`.
- **cost.** One caption clause. **payoff:** removes the only "a reviewer who counts
  bars will flag it" surface on the exact claim the conductor asked to stress-test.

### [feasible-now] C-c — Trim the §6.1 ARI read-off so it opens at the result, not the re-definition
- **closes:** Cohesion seam #1 (the round's only new seam) — §6.1 (lines 686–689)
  re-states the additive identity §4 (453–458) already owns, on an over-budget paper.
- **what / how.** Use cohesion's exact rewrite. Replace lines 687–690 (the
  re-definition opener) with a single connective clause:
  > *"The ARI decomposition (Figure~\ref{fig:ari}) reports this split per model. For
  > `embeddinggemma` the alignment-only floor is the smallest of any non-degenerate
  > model, and its post-re-rank residual is the lowest: $\mathrm{ARI@}100 = 0.229$
  > (next `qwen3-0.6B` at $0.233$)..."*
  Drops ~1.5 lines, removes the one place ARI reads as restatement rather than
  continuation, and lets §6.1 do its actual job (the numbers). **No number changes.**
- **cost.** ~2 words added, ~2 sentences removed (a net *line saving* on an
  over-budget paper). **payoff:** ARI thread reads as pure continuation; recovers
  vertical space.

### [feasible-now] C-d — The concrete float-cut / merge plan (29 → 25, no argument lost)
- **closes:** The dominant cohesion concern (float overload, 29 floats / 17 in §6).
- **what / how (ranked, exactly the cohesion plan, made concrete for the writer):**

  | # | action | floats | what carries the lost content | net |
  |---|--------|--------|-------------------------------|-----|
  | 1 | **CUT cp_fig14 + ag_fig10 (the two radars)** | −2 | Tables 1–2 (per-axis numbers) + cp_fig17 (aggregation ribbon); absorb the "where each wins" half-sentence via A7 | 29→27 |
  | 2 | **MERGE cp_fig06 + cp_fig07 (mate-retrieval) → one 2-panel** | −1 | identical story (foreign twins buried + how deep); two `subfigure`s under one caption | 27→26 |
  | 3 | **MERGE cp_fig09 + cp_fig10 (language-collapse) → one 2-panel** | −1 | same mechanism (over-representation + which language buries gold); both cited in the single parenthetical at line 958 | 26→25 |

  - **Mechanics for the merges:** wrap each pair in `subfigure` (or a single
    `\includegraphics` of a pre-stitched 2-up PNG if the troubleshooter regenerates
    the plot 2-up) under **one** `\begin{figure}` / one `\caption` / one `\label`,
    then collapse the two `\ref`s to that one label. The two adjacent half-sentences
    that introduce each pair (lines 668/670 for mate; the single parenthetical 958
    for collapse) need no rewrite — just point both at the merged label.
  - **What STAYS, explicitly:** cp_fig22 (ARI, ×3-ref, load-bearing), cp_fig23
    (per-route, ×2-ref, the round's new deployment artifact), cp_fig18 (cost
    frontier), cp_fig19 (RRC budget), cp_fig20 (degeneracy gate), cp_fig17
    (aggregation ribbon — it covers the radars), the teaser, and the two leaderboard
    tables. The cut comes *entirely* from legacy low-information panels.
  - **Stretch (only if still over budget after 25):** cp_fig17 ribbon is the *next*
    candidate (rank-range [1,4] is stated inline), but cutting it weakens the radar
    cut's cover story — so hold it as a *last resort*, not part of the base plan.
- **cost.** Two 2-panel merges + two deletions + ~1 absorbing sentence (A7). **payoff:**
  brings the paper inside the page budget (29→25) with zero argument lost, which is
  the single biggest pre-submission risk the critics name.

### [paper-framing-only] C-e — Float-order swap (cp_fig22 before cp_fig19)
- **closes:** Cohesion seam #3 (cosmetic).
- **what / how.** Swap the two `\begin{figure}` blocks so cp_fig19 (RRC budget, the
  object) precedes cp_fig22 (ARI, the decomposition of it), matching prose order
  (RRC budget → ARI). Or just let LaTeX float them — prose order is already correct.
- **cost.** Block swap. **payoff:** lowest priority; do not let it cost a rewrite.

---

## Wild cards (highest upside, clearly tagged) — all hardening, none add a float

### [feasible-now] W1 — A single "robustness appendix table" that consolidates ALL the channel-(a) resamples
- **idea.** Instead of scattering A1–A6 brackets through the body, collect them into
  **one small appendix table**: each load-bearing scalar (r=+0.96, XRC50 of the 3
  frontier models, ARI@100 egemma vs qwen3, two-tax ρ, cheapest-reader ρ) with its
  point estimate, its resample interval / sign-stability vote, and n. One table, in
  the appendix (which is not in the 8-page body budget), referenced once from §7.
- **why it's a wild card.** It converts "the paper hedges its small-n correlations in
  prose" into "the paper has a *robustness table* showing every load-bearing scalar
  survives resampling" — a structural credibility upgrade that costs **one appendix
  float (outside the body budget)** and lets the body stay clean. This is the single
  highest novelty-of-rigor move available, and it's the natural home for A1–A6.
- **cost.** CPU-only resamples (A1–A6) + one appendix table. **payoff:** a reviewer
  who distrusts n=7 correlations gets a one-stop answer; the body prose stays terse.

### [feasible-now] W2 — Permutation/placebo test that the separability link is not a capability tautology
- **idea.** A skeptic could say "of course cross-language AUC correlates with
  CLIR@10 — they're both 'is the model good.'" Pre-empt with a **partial / placebo
  check**: does cross-language AUC predict CLIR@10 *after* partialling out overall
  Recall@10 (the capability proxy)? Report the partial correlation r(AUC, CLIR | R@10)
  inline. If it stays strongly positive, the separability link is *not* just "good
  models are good" — it's specifically the cross-language separability that drives
  cross-lingual recall, which is a sharper mechanism claim.
- **why it's a wild card.** It directly hardens C3's *causal-adjacent* read ("the
  lever is at the embedding level") against the most sophisticated objection
  (tautology), at zero float cost, with one inline partial-r.
- **cost.** CPU-only (R@10, CLIR@10, AUC all on disk). One inline number. **payoff:**
  upgrades "+0.96 robust" to "+0.96 robust *and not a capability artifact*" — the
  strongest version of the paper's load-bearing mechanism.
- **risk note.** With n=7 a partial correlation has ~4 residual df — report it as
  *directional/descriptive* exactly like the other small-n numbers, leading with the
  sign. Honesty contract: it strengthens but does not "prove."

### [paper-framing-only] W3 — Pre-register the W3 alignment causal probe's ARI target as a one-line "prediction box"
- **idea.** The novelty critic's highest-value *next* result is the alignment causal
  probe with ARI as the before/after target (correctly deferred). Without running it,
  the paper can **state the falsifiable prediction crisply** in Limitations: "an
  alignment intervention should drop egemma's L∞ (0.058) / ARI@100 (0.229) while
  leaving RRC@K-under-re-ranking flat; this is the experiment that would convert the
  correlational 'align, not re-rank' into a causal claim." (The draft already
  gestures at this at lines 1250–1255 — tighten it into one sharp falsifiable
  sentence with the exact numbers.)
- **why it's a wild card.** Costs nothing, adds no float, and makes the paper's
  *next* result feel inevitable and pre-registered — reviewers reward a paper that
  names the exact number its follow-up will move. **Do not run the probe this round**
  (the freeze call is right); just sharpen the sentence.
- **cost.** ~1 sentence. **payoff:** frames the deferred causal result as a precise,
  ARI-targeted prediction rather than vague future work.

### [feasible-now] W4 — "Float-budget audit" line for the writer: reference-count + information-density table (internal, not in paper)
- **idea.** Hand the writer a one-shot internal audit: for each of the 27 figures,
  its `\ref` count and whether its single load-bearing number is *also* stated in
  prose. Any figure that is ×1-referenced AND whose number is already in prose is a
  cut candidate. This *systematizes* the cut decision beyond the three the cohesion
  critic named (e.g. it would surface cp_fig17 as the next candidate, confirming the
  C-d stretch note).
- **why it's a wild card.** It makes the trim *principled and repeatable* rather than
  ad hoc — useful if the page count is still over after the 4-float cut.
- **cost.** A grep over `\ref`/`\label` + a manual prose-number check; internal note,
  not paper content. **payoff:** a defensible, density-ranked cut order if more
  trimming is needed in round 5+.

---

## Top-3 recommended for this round (editorial pick across channels)

1. **C-d + A7 — Execute the 4-float cut/merge (29 → 25).** `[feasible-now]` This is
   the round's #1 ask from all three critics and the single biggest pre-submission
   risk. Cut the two radars (cp_fig14, ag_fig10; covered by Tables 1–2 + cp_fig17),
   merge mate-retrieval (cp_fig06/07) and language-collapse (cp_fig09/10) into two
   2-panel figures, and absorb the radars' lost half-sentence (A7). No argument lost;
   the paper lands inside the page budget. Pair with the three cheap text fixes — the
   **ρ_k cite (C-a)**, the **cp_fig22 caption reconcile (C-b)**, and the **§6.1 ARI
   trim (C-c)** — which together close every residual the critics named at ~1 clause
   each. *This bundle is the whole "ship it" list; do it first.*

2. **W1 ← {A1, A6, A5} — One appendix robustness table consolidating the bootstraps.**
   `[feasible-now]` Harden the three numbers the recommendation actually rests on —
   the separability **r=+0.96** (BCa model-bootstrap CI + sign-stability, A1), the
   headline **XRC50** of the three frontier models (depth-bootstrap CI, A6), and the
   **ARI@100 egemma-vs-qwen3** 0.004 gap (per-query bootstrap of the ordering, A5) —
   and collect them into a *single appendix table* (outside the 8-page body budget),
   referenced once from §7. This converts the paper's prose-level small-n hedging
   into a structural "every load-bearing scalar survives resampling" claim, which is
   the highest novelty-of-rigor move available without touching the frozen spine or
   adding a body float. *The best hardening-without-new-objects move on the table.*

3. **A3 + W2 — τ-sensitivity of the cost frontier, and the separability partial-r.**
   `[feasible-now]` Two inline numbers that pre-empt the two most obvious "you tuned
   it / it's a tautology" objections: (A3) report the τ-interval over which the
   admitted set and "cheapest admitted reader = bge-m3" are stable, noting
   embeddinggemma's max-CLIR-corner status is τ-invariant by construction — turning
   the τ=0.40 snapshot into a stated stability range that hardens C4; and (W2) report
   r(cross-language AUC, CLIR@10 | Recall@10) to show the separability mechanism is
   not just "good models are good," hardening C3's load-bearing link. Both are
   CPU-only, single-scalar, zero-float, and lead with sign/direction at n=7. *Cheap,
   high-leverage hardening of the deployment rule and the mechanism — the two places
   a hostile reviewer pushes hardest.*

**Channel (b) verdict: FREEZE** — the metric family is closed; no level-2 idea
clears the strictly-stronger-replacement bar this round (B1/B2 recorded only to show
the filter was applied). The round's value is entirely in **trimming (top-1)** and
**hardening the existing scalars (top-2, top-3)** — exactly the conductor's bias.
Nothing above adds a body float; several remove one; nothing makes a core claim
depend on a new embedding-model run.
