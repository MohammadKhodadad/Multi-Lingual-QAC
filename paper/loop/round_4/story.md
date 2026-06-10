# Story (round 4)

## Changes since round 3

Round 3 was the *consolidation* round (it turned XRC/RRC into first-class
deployment objects: the cost frontier, the RRC budget curve with the knee $K^{\!*}$
and the $L_\infty$ floor, the DEG gate, and the two-tax spine). The reporter then
verified **two more CPU-only, 0-API objects** that the round-3 prose does **not yet
contain**, and that are the entire substance of round 4:

1. **ARI decomposition (cp_fig22).** Every model's cross-lingual shortfall splits
   *exhaustively* into three additive parts at depth $K$: cheap (RRC@$K$,
   re-ranker-recoverable) + deep (RRC@1000 − RRC@$K$, recoverable only with a deeper
   pool) + the alignment-only floor $L_\infty$. The identity **sums to exactly 1.0
   for all 9 models** (verified: `identity_sum_100 == 1.0` every row), and the scalar
   $\mathrm{ARI@}K = L_\infty/(1-\mathrm{RRC@}K)$ is the share of the *remaining* gap
   that **no re-ranker can recover**. **embeddinggemma has the lowest non-degenerate
   ARI@100 (0.229)** and the **smallest $L_\infty$ floor (0.058)** of any
   non-degenerate model. This is the **quantitative backbone of "align, don't
   re-rank"** — the paper's headline thesis stops being a slogan and becomes a
   *measured per-model split*. Source: `chem_patents/.../extra_ari_decomposition/`
   (`ari_decomposition.csv` + `summary.json`); fig `cp_fig22_ari_decomposition.png`
   (present, byte-identical).

2. **Per-route cost frontier + decision-flip (cp_fig23).** Recomputing the
   (XRC50$_\ell$, CLIR@10$_\ell$) frontier *per query language* shows the capability
   corner is **not stationary**: there are **3 distinct max-CLIR corners across the 5
   routes** (en→qwen3-0.6B, de/es/zh→embeddinggemma, fr→nomic-v2-moe), and a
   recall-only dashboard's pick **flips on 2/5 routes** (en, fr). This is the **honest
   upside of C4**: it *nuances*, does NOT overturn, the single-model recommendation.
   Source: `extra_per_route_frontier/` (`per_route_frontier.csv`,
   `frontier_membership_by_route.csv`, `decision_flip_by_route.csv`, `summary.json`);
   fig `cp_fig23_per_route_frontier.png` (present, byte-identical).

3. **Three closing fixes the critics left on the table (all must-do this round).**
   - **F1 (cohesion #1, the round-3 must-fix).** Lines 557 and 657 still say
     "**eight** non-degenerate models," which contradicts the DEG gate (where
     "non-degenerate" = the precise **7**-set). The count 8 is correct for those two
     populations (all-but-`gte`); only the *word* is wrong. Relabel each to "the eight
     models with a defined cross-lingual recall / RRC curve (all but the degenerate
     `gte-base`)." ~6 words each, no number changes.
   - **F2 (correctness C-NEW).** Fig.21's caption (line 701) still says
     "**sibling**-confusion rate" but the plotted/quoted quantity is the general
     `confusion_rate` (any look-alike out-ranks gold; values 0.182/0.068 correct).
     **Delete "sibling-"** → "(confusion rate, alias-graph benchmark)." Leave the
     value and the separate, correctly-labeled sibling-vs-parent severity split alone.
   - **F3 (novelty #2, near-mandatory cite).** The RRC **knee** $K^{\!*}$ is
     established cascade / re-rank-depth prior art and is currently **uncited**. Add
     one cascade/recall-ceiling reference (a peer-reviewed multi-stage dense-retrieval
     paper that states the first-stage recall ceiling; the Elastic reranker-depth blog
     is the named fallback) plus one sentence in §2 or §4: *"the knee + diminishing-
     returns-with-depth shape is the established cascade result \citep{...}; our
     contribution is its cross-lingual quantification and the $L_\infty$ floor."* This
     converts the **only missing-citation exposure** into airtight credit.

Round-3 *optional* polish that should also land on a final pass (cheap, none
load-bearing): F6's e5 endpoint at line 649–650 is named *inside* the
"non-degenerate" object — relabel it "for the degenerate `e5-large-instruct`, which
the gate excludes"; the universal-blind language list at line 939 ("French, Chinese,
and German") → "predominantly French and Chinese" (de/es tie at 3); harmonize
"home advantage" hyphenation; F4 half-clause crediting the standard Pareto frame and
claiming XRC as the new axis; F5 surface-the-inference-once on "alignment-only floor."

**The convergence call.** The paper is structurally done. Round 4 is the *last
substantive-addition* round: wire cp_fig22/23 in, land F1–F3, then **freeze** the
spine. The "where to add value vs. where to freeze" guidance is a first-class
deliverable this round — see the dedicated section below. The six `needs_eval.md`
items (W3-alignment-causal-probe, W4-formula-injection, CLIRMRS-external-validation,
XRC-conformal-M2, CCI-hop-distance-law, equivalence-audit-spotcheck) are all DONE per
the critic contract; the paper stands without them, and cp_fig22's per-model
$L_\infty$ floor is now the explicit before/after **target** of the forthcoming W3
alignment probe.

---

## Thesis (industrial framing)

> **A chemistry-patent search team must deploy exactly one multilingual embedding
> model, and the number their dashboard shows — average Recall@10 — is the one number
> that hides the failure they will ship.** Average recall is inflated by
> *same-language* hits; the moment a German chemist's query must reach an English or
> Chinese patent (the normal case in a patent family), recall collapses, and no two
> language versions of the same question return the same documents. We make the
> collapse measurable on two content-controlled, patent-grounded benchmarks, quantify
> *what cross-linguality costs* (you read ~3.5× deeper to find a foreign twin; a
> top-100 re-ranker recovers at most ~74% of them, and ~5.8% are structurally
> unrecoverable by any re-ranker), place the deployable models on a
> **cost-vs-capability frontier** (embeddinggemma is the Pareto-optimal capability
> corner; bge-m3 is the cheaper-to-read alternative), and show the durable fix is
> **representation alignment at indexing time, not a monolingual re-ranker at query
> time** — because foreign gold is *under-scored*, not merely mis-ordered, and the
> alignment-only floor $L_\infty \approx 0.058$ is the part of the gap that *no
> re-ranker can move*.

**What round 4 adds to the thesis (without changing it):** the "align, don't
re-rank" recommendation is now *quantitatively decomposed*. For every model, the
cross-lingual shortfall splits exhaustively (sum = 1.0) into a re-ranker-recoverable
part, a deeper-pool part, and the alignment-only floor — and the deployed model
(embeddinggemma) owns the **lowest alignment-only residual (ARI@100 = 0.229) and the
smallest floor (0.058)**. The recommendation's lever is therefore not asserted; it is
the measured remainder after a re-ranker has done everything it can.

**Framing overlay — "the cross-lingual tax has two line-items" (cohesion spine,
MEASURED-but-weak).** Unchanged from round 3: a reading-cost tax (XRC, the depth
multiplier, cross-lingual benchmark) and a confusability tax (the look-alike that
out-ranks the gold, alias-graph benchmark), only weakly — and if anything inversely —
rank-correlated across the seven non-degenerate models ($\rho=-0.59$, $n=7$, $p=0.16$,
n.s.), so neither benchmark is a clean proxy for the other. Connective tissue with the
caveat inline; never load-bearing, never in abstract/intro/conclusion.

Three industrial pillars, each grounded in a file (unchanged from round 3):
1. **The collapse is real, large, and costly.** Best CLIR@10 = **0.50**
   (embeddinggemma); home advantage up to **+0.55**; **XRC50 = 3.5×**; Spanish (34
   queries, 0 Spanish gold) is the built-in no-home stress test.
2. **The collapse is mis-rankable two ways, both deployment bugs.** Cross-lingual RBO
   ceiling **0.39** (alias-graph) / **0.19** (cross-lingual), any model; a confusable
   wrong compound out-ranks every gold on **14–78%** of queries; modal confusion is a
   **same-language sibling (44.4%)** — language bias and chemical confusability
   compound.
3. **The cause is separable representations, so the fix is alignment — and it has a
   measured, decomposable floor.** r(cross-language AUC, CLIR@10) = **+0.96**, robust
   on n=7; knee **K\*=5**, **RRC@100 ≤ 0.74**, **$L_\infty$ = 0.058 unrecoverable by
   any re-ranker** — and (NEW) **ARI@100 = 0.229** isolates exactly the alignment-only
   share of the remaining gap, lowest for the deployed model.

---

## Contributions (numbered, each with a one-line novelty claim)

**C1. Two content-controlled, patent-grounded multilingual chemistry-retrieval
benchmarks built only from human-translated patent text.** *(UNCHANGED — novelty
critic: NOVEL, well-defended; this is the paper's safest novelty, FREEZE.)*
- *Novelty claim:* the first cross-lingual, content-controlled,
  chemistry-ontology-grounded patent-retrieval benchmark whose gold is genuinely
  parallel human-translated patents + ChEBI membership (not `publication_number`
  equivalence, not MT documents) and whose negatives are chemically-confusable
  neighbours. Bounded against CLEF-IP and DAPFAM. *Mandatory cites:* CLEF-IP, DAPFAM,
  CLIRMatrix, ChEBI.

**C2. A cross-lingual robustness-metric family reported co-equally with recall —
anchored by deployment-legible cost objects (XRC, RRC), framed as a
cost-vs-capability frontier with a re-ranker-budget knee, and (NEW round 4)
decomposed by the Alignment-Recoverability Index.**
- *What:* CLIR@k vs MoLIR + home-advantage; directional CLIR matrix +
  hub/asymmetry; mate-retrieval; cross-lingual RBO; language-collapse; separability
  AUC (same vs cross); **XRC** (reading-cost multiplier) on a (XRC50, CLIR@10) Pareto
  frontier; **RRC** as a per-model budget curve with knee $K^{\!*}$ and floor
  $L_\infty$; **(NEW) the ARI decomposition** — every model's shortfall split
  exhaustively into cheap / deep / alignment-only (sum = 1.0), with
  $\mathrm{ARI@}K = L_\infty/(1-\mathrm{RRC@}K)$ the re-ranker-irreducible share; the
  **DEG gate** (CLIR@10<0.10); CLIR-MRS / MRS demoted to table-ordering.
- *Novelty claim:* a retrieval-side, ranking-level robustness suite for CLIR whose
  cost instruments are new deployment objects — XRC as a distribution-free
  reading-depth multiplier on a **standard Pareto frame whose cost axis is new**
  (F4), RRC as a per-model re-ranker-budget curve with a knee (credited to cascade
  prior art, F3) and a **structural floor $L_\infty$** (the genuinely novel object),
  **and the ARI decomposition that operationalizes "align vs re-rank" as an additive
  per-model split that closes to 1.0**. The composite is explicitly NOT a
  contribution; per-axis dominance + the frontier + the decomposition carry the
  result.

**C3. A mechanism finding, confirmed on a content-controlled corpus and made
falsifiable: cross-lingual chemistry-retrieval failure is an embedding-level
separability deficit, so the lever is alignment, not re-ranking — with a *decomposed*
floor.**
- *What:* availability sets the stage but a residual encoder bias remains (slope
  −0.57, n=5, DESCRIPTIVE); modal confusion is a same-language sibling (44.4%);
  structure-style questions are the trap (R@10 0.26, confusion 51%; formula token
  p<0.01); confusion **is** a separability collapse (AUC 0.55 vs 0.70); across models
  r(cross-language AUC, CLIR@10) = **+0.96**, robust on n=7. Bound: knee **K\*=5**,
  **RRC@100 ≤ 0.74**, **$L_\infty$ = 0.058**; **(NEW) ARI@100 = 0.229** for the
  deployed model — the part of the remaining gap only alignment can move.
- *Novelty claim:* we *confirm* the alignment-not-translation finding of
  [2511.19324, 2507.07543] on a content-controlled parallel patent corpus that removes
  the translationese/content confounds those studies could not, and add (i) a
  chemistry-specific same-language-sibling confusability trap and (ii) a
  separability-AUC + RRC-floor + **ARI-decomposition** test that turns "alignment is
  the fix" into a falsifiable per-model split: of the cross-lingual gap a re-ranker
  cannot close, a *measured* fraction is alignment-only. *Mandatory cites:* 2511.19324,
  2507.07543.

**C4. A concrete, audited deployment decision with an operating rule, a cost-frontier
justification, and (NEW round 4) an honest per-route upside frontier.**
- *What:* deploy **embeddinggemma** — the **Pareto-optimal capability corner** of the
  (XRC50, CLIR@10) frontier (unique global max-CLIR; no model is both cheaper-to-read
  AND higher-CLIR), the **smallest alignment-only floor / lowest ARI@100**, and
  per-route it **wins 3/5 routes (de, es, zh) including the two hardest cross-only
  ones**; **bge-m3 is the cheaper-to-read admitted alternative** on the same frontier.
  Report XRC/RRC/CLIR@10/language-parity next to recall; budget the re-ranker by the
  knee (K\*≈5) and respect the $L_\infty$ floor; **do not reflexively ensemble**
  (untuned RRF underperformed; oracle headroom real but not free). **(NEW) The
  capability corner moves across routes — a per-route router is genuine upside
  headroom, not a delivered win** (thin per-route n; reported descriptively).
  Machine-translating the question is safe (paired diff −0.044, p=0.13).
- *Novelty claim:* an industry-track deployment decision grounded in a
  capability-conditioned cost frontier, a re-ranker-budget knee, and an alignment-
  recoverability decomposition (not mean recall or a hand-weighted composite), with a
  negative ensemble result, a **per-route deployment map** that shows the single best
  model is route-dependent, and a QT-vs-DT budget rule re-derived for embedding
  retrieval over patents and quantified as an insignificant null. *Mandatory cites:*
  Oard 1998, Saleh & Pecina 2020, 2605.24297.

**C5. (Supporting) A reproducible QAC generation + audit pipeline.** *(UNCHANGED,
support only; `\todo`→`% TODO`. FREEZE.)*

---

## Section map

Only the sections that change in round 4 carry full beats; sections marked
**FREEZE** are correct as written and must not be re-opened.

### Abstract — **FREEZE the spine; one optional half-sentence**
- *Purpose / beats:* unchanged. Collapse + two benchmarks + metric family + headline
  numbers (CLIR@10 0.50; home +0.55; RBO 0.39/0.19; confusion 14–78%; XRC50 3.5×;
  RRC@100 ≤ 0.74; $L_\infty$ 0.058) + separability cause (+0.96) + alignment-not-
  re-ranking + embeddinggemma the Pareto capability corner (NOT cheapest) + MT-safe.
- *Optional round-4 add:* ARI may be alluded to in **one half-sentence** *only if it
  does not lengthen the abstract* ("...of which a measured fraction is recoverable
  only by alignment, not by re-ranking"). **Do NOT** put the per-route frontier, ARI
  numbers, or any non-significant correlation in the abstract.
- *Honesty note:* no two-tax ρ, no trap ρ, no per-route routing claim in the abstract.

### 1 Introduction — **FREEZE** (contributions list gets the C2/C4 deltas only)
- The contributions list must reflect the C2 ARI addition and the C4 per-route upside
  in **one clause each** (mirroring the contribution text above); the body prose is
  correct and should not be re-opened.

### 2 Related Work — **one cite added (F3), otherwise FREEZE**
- *Round-4 action:* add the **cascade / re-rank-depth citation** for the knee (F3) in
  the paragraph that introduces RRC or in the cross-lingual-ranking paragraph, with
  the one crediting sentence ("knee shape is established cascade prior art; our
  contribution is the cross-lingual quantification + $L_\infty$"). Optionally add the
  F4 Pareto-frame cite (syftr arXiv:2505.20266 or RAG cost-frontier
  arXiv:2511.09545). The six "you reinvented X" boundaries stay CLOSED — do not
  re-open.

### 3 Benchmarks — **FREEZE.**

### 4 Metrics — **add the ARI definition next to RRC; otherwise hold**
- *Purpose:* deliver C2. Round-4 adds **one short paragraph** defining ARI as the
  natural reading of the RRC curve, placed immediately after the RRC budget-curve
  paragraph (currently lines 427–441).
- *New beat (ARI / M2):* state the additive identity and the scalar, grounded:
  *"At any depth $K$, a model's cross-lingual shortfall splits exhaustively into a
  cheaply recoverable part ($\mathrm{RRC@}K$, what a top-$K$ re-ranker can reach), a
  deep-pool part ($\mathrm{RRC@}1000-\mathrm{RRC@}K$), and the alignment-only floor
  $L_\infty$; the three sum to one. We report $\mathrm{ARI@}K =
  L_\infty/(1-\mathrm{RRC@}K)$, the fraction of the *remaining* gap at depth $K$ that
  no re-ranker can recover — the quantitative form of 'align, don't re-rank.'"*
  Cite `cp_fig22`. Source: `extra_ari_decomposition/{ari_decomposition.csv,
  summary.json}`. **Number to use:** identity closes to 1.0 for all 9 models;
  ARI@100 lowest non-degenerate = embeddinggemma 0.229 (next qwen3-0.6B 0.233).
- *Existing beats hold:* CLIR@k/MoLIR/home; directional + hub (cite CLIRMatrix);
  mate-retrieval; RBO (Bailey 2017); language-collapse; separability AUC; XRC (keep
  the population-level clause + monotone-invariance + the "standard Pareto frame, XRC
  is the new axis" F4 half-clause + forward-pointer to the frontier); RRC (keep the
  budget-object framing + the F3 cascade cite); **DEG gate (define ONCE, CLIR@10<0.10,
  exactly {gte, e5}, cite cp_fig20)**; CLIR-MRS demoted.

### 5 Experimental Setup — **one-line update**
- Add `extra_ari_decomposition.py` and `extra_per_route_frontier.py` to the list of
  `experimental_plots/extra_*.py` scripts in the reproducibility paragraph
  (currently lines 509–515). Everything else FREEZE.

### 6 Results — **wire cp_fig22 into §6.1; F1 + F2 relabels; per-route may be §6.1 or §8**
- *§6.1 cross-lingual:*
  - **(3) cost frontier (UPGRADED with F1 + F4).** Keep the XRC paragraph and the
    cp_fig18 frontier exactly. **F1 fix:** line 557 "eight non-degenerate models" →
    "the eight models with a defined cross-lingual recall (all but the degenerate
    `gte-base`)." Add the F4 half-clause crediting the standard Pareto frame.
  - **(5) the re-ranker budget + ARI decomposition (UPGRADED with cp_fig22 + F1).**
    Keep cp_fig06/07 and cp_fig19. **Add a 2–3-sentence ARI read-off** right after the
    RRC-budget paragraph (currently lines 635–651), introducing
    **`cp_fig22_ari_decomposition.png`**: *"The RRC curve decomposes each model's
    cross-lingual shortfall exhaustively into what a re-ranker can recover, what a
    deeper pool can recover, and the alignment-only floor (the three sum to one,
    Fig.~\ref{fig:ari}). For embeddinggemma the floor is the smallest of any
    non-degenerate model and its post-re-rank residual is the lowest:
    $\mathrm{ARI@}100 = 0.229$, so after a cheap top-100 re-rank, less of its
    remaining gap is alignment-bound than for any other deployable model."* **F1
    fix:** line 657 cp_fig19 caption "eight non-degenerate models" → "the eight models
    with a defined RRC curve (all but `gte-base`, whose candidate pool is empty)."
    **F6 (optional):** relabel the e5 endpoint at line 649–650 as the degenerate
    illustration the gate excludes.
  - Source: `extra_ari_decomposition/`. cp_fig22 must be referenced exactly once and
    interpreted (cohesion: no orphan figure).
- *§6.2 alias-graph:* **F2 fix** — Fig.21 caption (line 701) "sibling-confusion rate"
  → "confusion rate, alias-graph benchmark." Value 0.182/0.068 untouched; the
  separate sibling-vs-parent severity split untouched. Everything else FREEZE.
- *§6.3 leaderboards:* FREEZE.

### 7 Analysis — **back-reference ARI in the separability-floor crux; otherwise FREEZE**
- The separability-deficit ⇒ re-ranker-floor beat (currently lines 962–976) is the
  crux that sets up Deployment. Round-4 action: in that paragraph, after "the floor
  $L_\infty=5.84\%$ ... is unrecoverable by any re-ranker," add **one back-reference**:
  *"and the ARI decomposition (\S\ref{ssec:cp}) shows this floor is the *only*
  alignment-bound part — for the deployed model, after a cheap top-100 re-rank, just
  $\mathrm{ARI@}100=0.229$ of the remaining gap requires alignment, which is the
  lowest of any non-degenerate model."* This makes the decomposition the bridge from
  the separability mechanism to the "align, don't re-rank" deployment line. Do NOT
  re-touch the two hedged fragile correlations or the robust +0.96 — correctness
  critic called these the best standing improvements.

### 8 Deployment Recommendation — **add the per-route frontier as honest C4 upside**
This is the **highest-risk-of-overstatement section** and the home of cp_fig23. The
single-model recommendation must stay the spine; the router is headroom.
- *(1) Deploy embeddinggemma — capability corner (UNCHANGED spine).* Keep the
  frontier-choice framing (capability corner, not cheapest; bge-m3 the
  cheaper-to-read alternative; rank-range [1,4] caveat). **Strengthen with ARI:** add
  "...and it carries the smallest alignment-only floor and lowest ARI@100 (0.229) of
  any non-degenerate model" as a one-clause reinforcement, NOT a new claim.
- *(NEW) The capability corner is route-dependent — a per-route router is upside
  headroom.* Introduce **`cp_fig23_per_route_frontier.png`** as a *new paragraph
  bounded by the four-point honesty contract below*:
  1. **embeddinggemma is still the single-model recommendation.** It is the global
     capability corner (max pooled CLIR@10 = 0.5024), has the lowest alignment-only
     floor ($L_\infty=0.058$) and lowest ARI@100 (0.229), and **per-route wins 3/5
     routes (de, es, zh)** — including the two hardest cross-only routes (es has 0
     same-lang gold; zh is the thinnest). The recall-only dashboard picks it on all
     five routes.
  2. **A per-route router is HEADROOM, not a delivered win.** Frame it as consistent
     with the existing oracle/ensemble headroom story: a router *could* add the
     en→qwen3-0.6B and fr→nomic-v2-moe corners. Phrase exactly as: *"a single model
     (embeddinggemma) is the right default; a per-route router that swaps in
     qwen3-0.6B for en and nomic-v2-moe for fr is a plausible upside, but on our
     per-language sample sizes we report it as headroom, not a recommendation."*
  3. **Do NOT overstate routing.** Route corners rest on thin per-language samples
     (de n_same=7, zh n_same=2, es n_same=0; cross-side n=22–34). The **defensible
     spine** is: the ROBUST per-route CLIR@10$_\ell$ axis + frontier membership + the
     *existence* of corner movement (3 distinct corners, decision flips on 2/5 routes,
     en + fr). The **per-route XRC y-axis is explicitly INDICATIVE** and must be
     labeled so; es is XRC-undefined (n_same=0) and must never be imputed.
  4. **This nuances C4 as UPSIDE, never a reversal.** No draft may promote a per-route
     router to a headline result or a deployed recommendation, or drop the thin-n
     caveat.
  - Source: `extra_per_route_frontier/{per_route_frontier.csv,
    frontier_membership_by_route.csv, decision_flip_by_route.csv, summary.json}`.
  - *Placement note:* cp_fig23 belongs in §8 with the ensemble/"do not reflexively
    ensemble" paragraph (it is the same router-headroom argument, now per-route), OR
    as a §6.1 results figure with the decision read-off deferred to §8. Writer's call;
    the figure must be referenced once and interpreted, and the four-point contract
    must hold wherever it lands.
- *(remaining beats UNCHANGED):* report robustness next to recall; do not reflexively
  ensemble (oracle 0.61 / 88% vs 76%; RRF loses; 12% universal-blind core 14/16
  structure); budget the re-ranker by the knee; align-not-re-rank (now reinforced by
  ARI); budget rule (MT the query, human-translate the corpus, Oard 1998 / Saleh &
  Pecina 2020).

### 9 Limitations — **add two thin-n caveats; otherwise hold**
- Keep the round-3 list (scale; XRC50 robust / D90-D95 censored; availability slope
  −0.57 descriptive; the two non-significant correlations; the forthcoming W3 probe).
- **ADD (round 4):**
  1. **Per-route frontier thinness.** The per-route capability corners (cp_fig23) are
     estimated on thin per-language samples (de n_same=7, zh n_same=2, es n_same=0;
     cross-side n=22–34); the per-route XRC axis is **indicative** and es is undefined
     (never imputed). The per-route router is reported as **headroom, not a
     recommendation**.
  2. **W3 probe target made explicit.** Tie the forthcoming alignment causal probe to
     cp_fig22: the per-model $L_\infty$ floor / ARI is the exact before/after quantity
     the probe aims to move (fit a per-language alignment map on one model,
     re-retrieve, recompute XRC50/RRC@100/ARI). UPSIDE-ONLY — the paper does not
     depend on it.

### 10 Conclusion — **FREEZE the spine; one optional clause**
- Restate verbatim: collapse (CLIR@10 0.50, home +0.55, RBO 0.39/0.19, confusion
  14–78%), cost (XRC ~3.5×, knee K\*=5, floor $L_\infty$=0.058), embeddinggemma the
  Pareto capability corner (bge-m3 cheaper-to-read), separability cause (+0.96,
  robust), align-not-re-rank, budget rule. *Optional one clause:* "...the alignment-
  only floor is the only part of the gap a re-ranker cannot move." No per-route
  routing claim, no non-significant correlation, no ARI number in the conclusion.

---

## Where ADDED value/novelty should now go vs. where to FREEZE

The paper has converged (round-3 critics: 0 correctness mismatches, novelty
defensible, cohesion near-clean). This section is a first-class round-4 deliverable.

**FREEZE (do not re-open — re-opening risks regressions the critics already cleared):**
- **C1 benchmarks (§3), Related Work boundaries (§2 except the F3 cite), the DEG gate,
  the N2 frontier framing, the two RBO ceilings + "any model," the MT null, the
  +0.96 robust-vs-fragile correlation treatment, the two-tax MEASURED-but-weak
  framing.** These are the paper's correctness-and-cohesion bedrock; every one was
  individually verified or praised by a round-3 critic. Touching them spends risk for
  no gain.
- **The thesis and the section spine.** Abstract / Intro / Conclusion claims are
  aligned and honest; only the named one-clause additions above are permitted.

**ADD value here (high-yield, low-risk, already grounded):**
- **The ARI decomposition (cp_fig22)** is the single best place to add depth: it is
  verified, closes the thesis quantitatively, and strengthens C2 + C3 + C4 at once.
  Wire it in fully (Metrics def + §6.1 read-off + §7 back-reference + §8 reinforcement).
- **The per-route frontier (cp_fig23)** is the honest upside for C4 — add it as
  bounded headroom under the four-point contract. This is the last *new* analytical
  object; with it, the metric family and the deployment story are complete.

**ADD value ONLY post-submission (needs-eval, frame as forthcoming, never depend on):**
- **The W3 alignment causal probe** is the one experiment that would convert "align,
  don't re-rank" from correlational to **causal** — and cp_fig22 now hands it a precise
  target ($L_\infty$ / ARI before vs after). It is the paper's most memorable *next*
  result; keep it in Limitations as forthcoming.

**STOP generating new analyses after this round.** Further CPU-only objects (the
dreamer's A2–A8, M1/M3–M6, W2–W3) are individually plausible but the marginal
novelty-per-figure is now below the cost of added figure density and new over-claim
surface. The round-3 cohesion critic already flagged §6 figure density as heavy
(eleven figures across §6.1/§6.2); cp_fig22 + cp_fig23 are the last two the paper
should absorb. Round 5+ should be **polish and tightening only** (the F4/F5/F6
clauses, hyphenation, float order, prose compression for page budget), not new
results.

---

## Open narrative risks (for critics to watch)

1. **The per-route frontier must read as UPSIDE, not a reversal of C4 (correctness +
   cohesion — the round's top new risk).** embeddinggemma stays the single-model
   recommendation: global capability corner, lowest $L_\infty$/ARI, wins 3/5 routes
   incl. the hardest. The router is headroom on thin n (de=7, zh=2, es=0 same-lang
   gold). No draft may promote routing to a recommendation or drop the thin-n caveat.
   The per-route XRC axis is INDICATIVE; es XRC is undefined and must never be imputed.
   Source: `extra_per_route_frontier/summary.json`.
2. **ARI is regression-clean (identity sums to 1.0 for all 9) and safe to state firmly
   — but pull numbers from the CSV, not loose prose (correctness).** ARI@100 lowest
   non-degenerate = embeddinggemma 0.229 (next qwen3-0.6B 0.233); $L_\infty$ smallest
   = 0.058. The implementer's inline list was narrated out of recall order; use
   `ari_decomposition.csv` order. Do NOT over-hedge ARI — like the RRC budget object,
   it has no significance caveat.
3. **The "cheapest = embeddinggemma" superlative stays dead (N2, correctness).**
   Capability corner, NOT cheapest; bge-m3 (2.0×) is the min-XRC admitted model at
   τ=0.40.
4. **Two-tax is MEASURED-BUT-WEAK, not independence (correctness + cohesion).**
   ρ=−0.59, n=7, p=0.16 (n.s.); "neither benchmark is a clean proxy"; never
   significant, never in abstract/intro/conclusion. **F2:** Fig.21 caption is
   "confusion rate," not "sibling-confusion rate."
5. **The knee must be CREDITED (novelty F3, near-mandatory).** Add the cascade /
   re-rank-depth cite; claim only the cross-lingual quantification + $L_\infty$ +
   the ARI decomposition. Without this cite a cascade-literate reviewer can assign
   the knee to prior art.
6. **"Non-degenerate" = the precise 7-set everywhere (cohesion F1).** Lines 557 and
   657 must drop the gate-bound word "non-degenerate" for the two 8-model populations
   (count stays 8; relabel to "models with a defined recall / RRC curve"). The DEG
   gate's discipline must not self-contradict.
7. **Two RBO ceilings stay separated and "any model"-framed (correctness + cohesion).**
   0.39 (alias-graph) / 0.19 (cross-lingual), different benchmarks; never
   average/conflate.
8. **MT-of-question is a NULL (correctness).** −0.044, p=0.13; "no significant
   penalty," never "MT helps."
9. **Availability slope is DESCRIPTIVE n=5; +0.96 separability is the robust mechanism
   (correctness).** Do not re-touch the hedged fragile correlations.
10. **Figure-density discipline (cohesion).** cp_fig22 + cp_fig23 are the last two
    figures the paper absorbs; each referenced exactly once and interpreted. No orphan
    panel, no referenced-but-absent figure. After this round, polish only — no new
    results figures.
