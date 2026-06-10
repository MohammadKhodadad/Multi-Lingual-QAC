# Novelty review (round 3)

Reviewer #1 — Novelty Critic. EMNLP Industry Track. Verdicts re-grounded in the
literature (web-searched June 2026) and checked against my round-2 review
(`paper/loop/round_2/critic_novelty.md`). This round's question, set by the
conductor: (a) are the three new consolidation objects — the **cost-vs-capability
frontier**, the **RRC budget frontier** (knee $K^*$, $L_\infty$ alignment-only
floor), and the **degeneracy gate** — *genuinely novel claims* or just standard
plots renamed? (b) did my three round-2 residual nits get fixed? (c) re-verdict
the contributions, rank remaining over-claims, list still-missing citations.

`needs_eval.md` (6 items) is treated as DONE per the critic contract; the paper
must stand without them, and it does.

## Verdict summary (1 paragraph): is the paper's novelty defensible as written?

**Yes — and the round-3 consolidation strengthens it without introducing a new
over-claim, because the paper is scrupulous about *not* claiming the frontier
machinery as novel.** The honest answer to the conductor's headline question is:
**neither the cost frontier nor the RRC budget curve is a novel *method* — both
are textbook instruments (Pareto model-selection plot; cascade recall-ceiling /
re-rank-depth knee), and prior art for each is explicit and easy to find — but the
paper does not claim the method; it claims the *cross-lingual cost axis* (XRC) and
the *cross-lingual mate-twin instantiation with a structural floor* (RRC/$L_\infty$)
plotted on those standard frames.** That framing is correct and defensible: the
cost frontier is "a standard Pareto plot, but the cost axis is XRC (a reading-depth
multiplier nobody else reports)" and the draft never says otherwise — it
explicitly motivates the frontier as "XRC is not a scalar to minimize" rather than
as a new analysis (lines 584–602). The RRC budget frontier is the cleaner of the
two: the *knee + diminishing-returns* reading is established prior art (the Elastic
reranker-depth study; the cascade rule-of-thumb "pick the depth attaining ~90% of
max effectiveness"), so the **knee $K^*$ is INCREMENTAL**, but the **structural
floor $L_\infty$ read as an *alignment-only* bound on the cross-lingual mate-twin
is the genuinely new, regression-checked object** — I found no prior work that
reports a per-model cross-lingual recall-at-depth curve with a named unrecoverable
floor tied to the align-not-rerank thesis. The DEG gate is a reproducibility
convenience (NOT a novelty claim, and correctly not pitched as one). Net contribution
verdicts hold from round 2: C1 NOVEL (well-defended, and the level-2 audit's own
"systematic risk" note confirms the audited parallel corpus is the hardest-to-scoop
asset), C2 NOVEL-via-XRC-axis + the $L_\infty$ object, INCREMENTAL-frame, C3
INCREMENTAL-confirmation + NOVEL-confusability + NOVEL-floor, C4 INCREMENTAL-rule
+ NOVEL-frontier-grounded-decision. **All three of my round-2 residual nits are
fixed in the current draft.** No remaining claim would let a hostile reviewer
reject on novelty grounds; two minor over-claims to tighten, both in the framing
*adjectives* around the two new frontier objects, not in the claims themselves.

---

## Did my round-2 residual nits get fixed? (all three: YES)

| Round-2 nit | Status in round-3 draft | Verdict |
|---|---|---|
| #1 RRC re-ranker-bound pitch (was "renamed recall") | Line 437: *"RRC@K is the cumulative first-foreign-twin hit rate (mate-hit@K restricted to cross-lingual queries); **our contribution is not the quantity but reading it as a per-model re-ranker budget curve**."* The "you renamed recall" attack is pre-empted in the exact words I recommended, and *upgraded* into the budget-frontier object. | **CLOSED + UPGRADED** |
| #2 monotone-invariant needs a one-line reason, not an assertion | Lines 417–418: *"a ratio of retrieval depths is invariant to any monotone re-scaling of similarities, unlike an AUC or a weighted composite."* The justification is now stated, and it matches the authors' own CERC framing in the level-2 PDF (idea 1B: "robust to monotone score transformations"). | **CLOSED** |
| #3 CLIRMatrix attribution on the C2 contribution bullet | Line 133: contribution bullet now reads *"the directional CLIR matrix **(CLIRMatrix-style)**."* The attribution a skimming reviewer sees first is now in the bullet itself, not only in §2/§4. | **CLOSED** |

This is again a clean revision on the novelty axis.

---

## The two focus questions, grounded

### Q1. Is the cost-frontier framing genuinely novel, or just a standard Pareto plot?

**It is a standard Pareto plot — and the paper knows it.** "Pareto-optimal retriever
= no other model with both higher throughput and higher effectiveness" is the
textbook model-selection frame, used routinely to plot embedding models for
deployment (e.g. effectiveness-vs-throughput Pareto selection of MTEB retrievers;
syftr's accuracy-vs-cost-vs-latency Pareto search; the RAG cost-latency-quality
frontier line). So the *frontier method* is **NOT NOVEL**.

What is defensible — and what the paper actually claims — is the **cost axis**. The
frontier's y-axis is **XRC50** (a cross-lingual reading-*depth* multiplier), not
latency / throughput / dollar cost / parameter count, which is what every prior
retrieval Pareto plot uses. I searched specifically for a cross-lingual
reading-depth-ratio cost metric and found none in the published literature; the
nearest neighbour is the authors' *own* level-2 idea **CERC** ("at 95% coverage
EN→EN needs 12 docs, EN→ZH needs 41 — a 3.4× reading-cost multiplier"), i.e. the
conformal version of the same quantity, which the paper correctly relegates to
future-work machinery. So:

- **Frontier-as-method: NOT NOVEL** (standard Pareto). The draft does not claim it.
- **XRC-as-cost-axis: NOVEL** (re-affirmed from round 2; the cleanest positive
  novelty in the paper).
- **The frontier's *deployment payload* — "embeddinggemma is the capability corner,
  not the cheapest reader; bge-m3 is the cheaper-to-read admitted alternative" —
  is a correct, honest, non-superlative reading** and the right way to retire N2.

**Recommendation (minor):** the frontier is defensible *only because* the y-axis
is XRC. The draft is one adjective away from over-claiming in the figure title and
the C2 bullet, where "presented on a capability-conditioned Pareto frontier" can
read as if the *presentation* is the contribution. Add half a clause making
explicit that the frontier is the standard Pareto frame and **XRC is the new
ingredient** ("a standard Pareto frame whose cost axis is the new XRC multiplier").
This costs nothing and disarms the one "you just drew a Pareto plot" reviewer.

### Q2. Is the RRC budget frontier (knee $K^*$, $L_\infty$ floor) now a clearly-novel claim vs prior re-ranker / cascade analysis?

**Partly. Split it into two sub-claims, because they have different verdicts.**

- **Knee $K^*$ + "past the knee a deeper pool buys almost nothing": INCREMENTAL
  (established prior art).** This is exactly the cascade / re-rank-depth tuning
  result. The Elastic semantic-reranker depth study reports "fast increase followed
  by saturation" in 72.6% of cases and recommends "90% of the maximum gain at a much
  smaller depth … re-rank ~3× fewer pairs," with **per-model optimal depths** (their
  Table 2: 236–325 depending on reranker strength). The cascade literature states the
  recall-ceiling-from-first-stage-depth fact as folklore. So the *shape* (sweep $K$,
  find the knee, recall is capped by first-stage depth) is **not new** and the paper
  must not imply it is. The draft is currently safe here — it pitches the knee as a
  *deployment read-off*, not a discovery — but the Related-Work paragraph does **not
  yet cite any cascade / re-rank-depth reference**, which is the one missing citation
  that would let it claim only the cross-lingual quantification (see below).

- **$L_\infty$ as the cross-lingual mate-twin *structural floor* read as an
  *alignment-only* bound: NOVEL (and regression-checked).** The new and defensible
  object is *not* "recall saturates" but "the saturation value $1-\mathrm{RRC}@1000$
  is a per-model, per-language-pair floor on the cross-lingual mate-twin that **no
  re-ranker can move, and only representation alignment can** — and we report it
  per model (0.058 egemma → 0.372 e5)." I found no prior work that (i) computes a
  cross-lingual *mate-twin* recall-at-depth curve per model, (ii) names its
  asymptote as a structural floor, and (iii) ties that floor to the
  align-not-rerank lever as a falsifiable bound. This is the load-bearing novelty
  of the RRC upgrade, and because the regression checks pass it is correctly stated
  firmly (the one new object with no significance caveat). This is a genuine
  conversion of RRC from "renamed recall" (my round-2 weakest-item flag) into a
  deployment-planning object — exactly the dreamer route I handed over, executed.

**Net:** the RRC budget frontier is **NOVEL on the $L_\infty$/alignment-floor
reading, INCREMENTAL on the knee** — and it is *now a clearly-defensible claim*
**provided** the paper cites a cascade/recall-ceiling reference so the knee is
explicitly credited to prior art and only the cross-lingual quantification + the
$L_\infty$ floor are claimed. Without that one cite, a cascade-literate reviewer
can say "the knee is Elastic's blog from 2024." With it, the claim is airtight.

---

## Claim-by-claim (re-verdict; deltas from round 2 only)

### C1 — The two benchmarks
- **NOVEL (well-defended), UNCHANGED.** No new exposure this round. The level-2
  PDF's own honesty ledger (systematic-risk note #3) independently confirms the
  audited parallel corpus is "the asset no one else has" and "structurally harder
  to scoop" than the borrowed-machinery metric ideas — which is precisely why C1,
  not C2's cost objects, is the paper's safest novelty. **No change needed.**

### C2 — The robustness-metric family + the two new frontier objects
- **CLAIM (XRC on a Pareto frontier):** **NOVEL axis on a NOT-NOVEL frame.** See
  Q1. The contribution is XRC; the frontier is standard. Honestly pitched in the
  body; tighten one adjective in the C2 bullet / figure title (over-claims §1).
- **CLAIM (RRC budget curve, knee $K^*$, floor $L_\infty$):** **INCREMENTAL knee +
  NOVEL floor.** See Q2. Needs the cascade cite to be airtight.
- **CLAIM (degeneracy gate, CLIR@10 < 0.10):** **NOT a novelty claim, correctly
  not pitched as one.** It is presented as a reproducibility convenience ("we make
  the exclusion reproducible with a single criterion"), which is exactly right — a
  threshold rule is not a contribution, and the draft does not inflate it. The
  footnote justifying single-criterion over the AND-gate is good hygiene. **No
  exposure.** (One factual robustness point, not novelty: the gate is defended as
  matching the paper's existing exclusions, not as principled in the abstract — fine
  for an industry track.)
- **CLAIM (directional matrix, separability AUC, mate-retrieval, CLIR-MRS):** all
  remain correctly attributed (CLIRMatrix / standard ROC-AUC / borrowed bitext term
  / ordering-convenience). **CLOSED, unchanged.**

### C3 — The mechanism finding
- **INCREMENTAL-confirmation + NOVEL-confusability + NOVEL-floor.** The
  alignment-not-translation core still correctly *confirms* (not discovers)
  [2511.19324, 2507.07543] on a content-controlled corpus. The round-3 addition —
  the RRC $L_\infty$ floor as the falsifiable per-model bound on what re-ranking can
  recover — is the new novel sub-claim and is the right object to make
  "alignment is the fix" testable. **Strengthened, no over-claim.**

### C4 — The deployment recommendation
- **INCREMENTAL-rule + NOVEL-frontier-grounded-decision.** The decision is now
  grounded in the cost frontier + knee rather than a composite, with the N2
  superlative dead ("not the cheapest deployable model … bge-m3 reads shallower").
  The QT-vs-DT null still cites Oard 1998 / Saleh & Pecina 2020 and claims only the
  patent-embedding re-derivation. **Honest, no over-claim.**

### C5 — Pipeline
- **INCREMENTAL (support), unchanged.** Not a novelty exposure.

---

## Highest-risk over-claims (ranked) — all minor this round

1. **The cost frontier's framing adjectives can read as claiming the *frontier
   presentation* as novel (lowest-grade, the round's top exposure).** "Presented on
   a capability-conditioned Pareto frontier" (C2 bullet, abstract, fig title) is
   true but a Pareto-plot-literate reviewer will note the plot itself is standard.
   **Fix:** one clause crediting the frame and claiming the axis — e.g. "on the
   standard cost-vs-capability Pareto frame, with XRC (our new reading-depth
   multiplier) as the cost axis." Keeps the claim true and pre-empts "you drew a
   Pareto plot." (This is the *only* place the paper flirts with method-novelty it
   does not have.)

2. **The RRC knee is stated without crediting cascade / re-rank-depth prior art.**
   The knee + "deeper pool buys almost nothing" is established (Elastic depth study;
   cascade rule-of-thumb). The draft does not over-claim it in words, but the
   *absence of a cite* lets a reviewer assign the knee to prior art and imply the
   paper missed it. **Fix:** add one cascade/recall-ceiling citation in §2 (the
   "two-stage/recall-ceiling cascade reference" the story already flags as optional —
   promote it to actual). Then claim only the cross-lingual quantification + $L_\infty$.
   This converts the knee from "uncredited" to "credited-and-extended."

3. **"alignment-only floor" is a strong phrase resting on a correlational chain.**
   $L_\infty$ is *measured* (RRC@1000 floor), but "only alignment can move it" is an
   *inference* from the separability-deficit mechanism (correlational, $r=+0.96$),
   not a demonstrated intervention (the causal probe is PENDING-EVAL). The draft
   already hedges the upstream mechanism well and Limitations flags the probe as
   forthcoming, so this is low risk — but "alignment-only" is asserted as a property
   of the floor in several places (lines 60, 648–649, 1071). **Fix (optional):** say
   "a floor no re-ranker can move (only representation alignment can, per
   §\ref{sec:analysis})" once, so the inferential step is visible rather than baked
   into the adjective. Do not weaken $L_\infty$ itself — the *number* is
   regression-checked; only the *attribution* of its remedy is correlational.

No remaining over-claim rises to "a hostile reviewer rejects on novelty." The
round-1/round-2 top risks (CLIR-MRS-as-contribution; "first decomposition"; the
"cheapest = embeddinggemma" superlative) are all dead.

---

## Missing citations the paper should add (bib-ready)

The mandatory list from rounds 1–2 is fully integrated (CLEF-IP, CLIRMatrix,
DAPFAM, 2511.19324, 2507.07543, Oard 1998, Saleh & Pecina 2020, Bailey 2017, MMTEB,
2605.31142, 2605.24297, ChEBI, SapBERT, PaECTER, RBO, RRF). Round 3 surfaces **one
near-mandatory and one optional** addition, both to harden the two new objects:

- **(NOW NEAR-MANDATORY — RRC knee) A cascade / re-rank-depth prior-art cite.** The
  knee + diminishing-returns-with-depth result is established; the paper currently
  asserts it with no anchor. Cite a re-rank-depth / recall-ceiling reference and
  claim only the cross-lingual quantification + $L_\infty$. Concretely usable:
  - the Elastic semantic-reranker depth-selection study (saturation + 90%-of-max-at-
    smaller-depth + per-model optimal depth), `https://www.elastic.co/search-labs/blog/elastic-semantic-reranker-part-3`;
    or a peer-reviewed cascade-ranking reference if a blog is undesirable for the
    venue (any multi-stage dense-retrieval paper that states the first-stage recall
    ceiling — the round-2 list already flagged this as the cleanest defuse).
  This is the single citation that turns the RRC budget frontier from "knee
  uncredited" into "knee credited, $L_\infty$ claimed."

- **(OPTIONAL — cost-frontier framing)** A retrieval/RAG Pareto-frontier model-
  selection reference so the *frame* is explicitly credited and only the XRC *axis*
  is claimed. Any of: syftr (Pareto-optimal RAG config search, arXiv:2505.20266);
  the cost-latency-quality RAG frontier (arXiv:2511.09545); or a generic "Pareto-
  optimal retriever = no model with both higher throughput and effectiveness"
  selection reference. One inline cite in the cost-frontier paragraph (§6.1) makes
  the "standard frame, new axis" division unmistakable.

- **(STILL OPTIONAL — mate-retrieval lineage, carried from round 2)** Artetxe &
  Schwenk margin-based bitext mining / LaBSE Tatoeba "mate accuracy"
  (`https://arxiv.org/pdf/1811.01136`; `https://arxiv.org/pdf/2007.01852`) as the
  lineage of the borrowed "mate-retrieval" term. Not blocking.

Neither the cost-frontier nor mate-lineage cite is blocking; the **cascade/re-rank-
depth cite is the one I would now insist on** before submission, because it is the
only place where a missing citation (not a wording choice) lets a reviewer claim
the paper rediscovered known prior art.

## What WOULD make the weakest contribution clearly novel (hand to dreamer)

Round 2's weakest item (RRC-as-renamed-recall) has been fixed — the budget curve +
$L_\infty$ did the job. The new weakest *novelty* item is now the **cost frontier**,
because it is the place most exposed to "standard Pareto plot." Two routes, in
priority order:

1. **Make the frontier *cross-lingual-pair-resolved*, which no Pareto retriever plot
   is.** Right now the frontier is one point per model in global $(\mathrm{XRC50},
   \mathrm{CLIR}@10)$ space — indistinguishable in form from a throughput-vs-recall
   plot. Plot it **per directed language pair** (or per query language), so the
   deployable object becomes "the frontier *moves* across language pairs, and the
   capability corner is not the same model for en→de as for es→zh." A frontier whose
   *membership changes with the language direction* is a genuinely new deployment
   artifact (per-route model selection) that a generic Pareto plot cannot be, and it
   is CPU-only on existing score lists. This converts the cost frontier from "standard
   plot, new axis" into "new object" and directly serves the per-language-routing
   recommendation the paper already gestures at (§8 "do not reflexively ensemble").

2. **Tie the $L_\infty$ floor to a measured alignment intervention (the PENDING-EVAL
   causal probe), making "alignment-only" demonstrated rather than inferred.** The
   draft already names the probe as forthcoming (fit a per-language alignment map,
   re-retrieve, recompute XRC50/RRC@100). If even one model's $L_\infty$ measurably
   *drops* after alignment while staying flat under re-ranking, "alignment-only floor"
   stops being a correlational adjective and becomes the paper's headline causal
   result — and it is exactly the experiment XRC/RRC were built to enable. This is the
   highest-value upgrade overall, but it needs a run; route 1 needs none.

Both keep the "align, don't re-rank" thesis intact and require no new model
training (route 1: zero new compute; route 2: one small alignment fit).
