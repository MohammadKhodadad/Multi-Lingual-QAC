# Novelty review (round 4)

Reviewer #1 — Novelty Critic. EMNLP Industry Track. Verdicts re-grounded in the
literature (web-searched June 2026) and checked against my round-3 review
(`paper/loop/round_3/critic_novelty.md`). This is the **final-stretch** round.
The conductor's three questions:

1. Is **ARI** (ARI@K = L∞/(1−RRC@K), the re-ranker-irreducible share) genuinely
   novel, or a restatement of the recall ceiling / L∞?
2. Is the **per-route corner-movement** framed without over-claiming routing?
3. Final verdict: is every contribution now novelty-defensible for an EMNLP
   industry submission, or is anything still exposed? List last missing cites.

First, the round-3 near-mandatory ask is **CLOSED**: the cascade / recall-ceiling
citation for the RRC knee is present. Line 447 now reads *"The knee and the
diminishing-returns-with-depth shape are the established multi-stage cascade
result that recall is capped by first-stage candidate depth
\citep{nogueira2019multistage,gao2021rethink}; our contribution is its
cross-lingual quantification and the structural floor $L_\infty$."* I verified
both keys resolve in `custom.bib` (lines 357, 368) and that both papers are real
and on-point: Nogueira et al. 2019 (arXiv:1910.14424, *Multi-Stage Document
Ranking with BERT*) explicitly establishes the first-stage recall ceiling and the
quality/latency trade via candidate depth; Gao et al. 2021 (arXiv:2101.08751,
*Rethink Training of BERT Rerankers in Multi-stage Retrieval Pipeline*) is the
cascade re-ranker reference. This was the **only** missing-citation exposure I
insisted on before submission. It is now airtight.

## Verdict summary (1 paragraph): is the paper's novelty defensible as written?

**Yes. Every contribution is novelty-defensible, and the round-4 additions (ARI,
per-route frontier) are both handled with the same discipline that has carried the
paper: the borrowed scaffold is credited, and only the cross-lingual instantiation
is claimed.** On the conductor's headline question: **ARI is not a new
mathematical primitive, and the paper does not claim it is one — it is correctly
pitched as "a natural exhaustive reading of the same shortfall" (line 453), i.e. a
re-expression of the RRC curve, not a new metric.** The underlying facts ARI
re-packages are both established: (i) $1-\text{RRC@}K$ is the bounded-recall /
recall-ceiling result no top-$K$ re-ranker can exceed (now cited), and (ii) the
additive split is an algebraic identity (cheap + deep + floor = 1) that holds by
construction. So ARI's *novelty* is narrow: it is the **cross-lingual,
mate-twin-resolved instantiation** of the recall-ceiling decomposition, read as an
*alignment-vs-re-rank* lever. The closest prior art I could find is **a residual
decomposition for long-tailed reranking (arXiv:2604.01506)** whose ρ_k normalizes
*reranker gain* by the *recoverable* gap — a structurally adjacent but **inverse**
ratio (gain/recoverable vs. irreducible/remaining), in a **different domain**
(image/species/disease classification, not retrieval) and with **no alignment-only
floor**. ARI is therefore defensible as INCREMENTAL-but-genuinely-new-in-this-
setting, *provided the draft never calls it a "new metric/contribution"* — and it
doesn't (C2 bullet calls XRC/RRC the cost objects and lists ARI as the
"decompose[d]" reading; line 466 calls ARI "the quantitative form of 'align, don't
re-rank'," a framing claim, not a method claim). **The per-route frontier is the
round's only place that could over-claim, and it is fenced by an explicit
four-point honesty contract in the draft (lines 1093–1110): embeddinggemma stays
the single-model recommendation, the router is "upside headroom," thin-n is stated
inline and again in Limitations, and the per-route XRC axis is labeled
"indicative."** This is exactly the dreamer route-1 I handed over in round 3,
executed without inflation. No remaining claim lets a hostile reviewer reject on
novelty grounds. One near-mandatory citation surfaces this round (the long-tailed
ρ_k decomposition, to pre-empt "you reinvented residual decomposition"), and one
figure/caption count discrepancy on cp_fig22 should be fixed (cohesion-adjacent,
flagged below).

---

## The three focus questions, grounded

### Q1. Is ARI@K = L∞/(1−RRC@K) genuinely novel, or a restatement of the recall ceiling / L∞?

**It is a re-statement of L∞ *normalized* by the post-re-rank remaining gap — and
the paper says so.** Decompose the claim into its three layers:

- **The additive identity (cheap + deep + floor = 1): NOT NOVEL, and trivially
  true.** At any depth $K$, RRC@K + (RRC@1000 − RRC@K) + (1 − RRC@1000) = 1 by
  construction. The draft states it as an identity ("the three sum to one,"
  line 458), not a discovery. No exposure — it is arithmetic, correctly presented
  as such. (Note: the identity holds for **all nine** models trivially; see the
  figure-count nit below, which is the only thing to fix here.)

- **The recall-ceiling fact ($1-\text{RRC}@K$ un-rerankable): NOT NOVEL, now
  cited.** This is bounded recall / the first-stage recall ceiling — established
  cascade folklore, surfaced repeatedly in my searches ("the first-stage retrieval
  establishes a strict upper bound for retrieval coverage in subsequent stages").
  Credited via nogueira2019multistage / gao2021rethink. No exposure.

- **The scalar ARI@K = L∞/(1−RRC@K) read as the "alignment-only share of the
  *remaining* gap after a top-$K$ re-rank": INCREMENTAL, genuinely new *in this
  setting*, and defensibly framed.** I searched specifically for a named
  irreducible-share / recoverability ratio in retrieval and found **no** prior
  work that (i) computes a cross-lingual *mate-twin* recall-at-depth curve, (ii)
  normalizes its un-rerankable asymptote by the depth-$K$ remaining gap, and (iii)
  ties the result to an align-vs-re-rank lever. The nearest neighbour, the
  long-tailed-reranking residual decomposition (arXiv:2604.01506), defines a
  *different* normalized ratio (reranker **gain** / **recoverable** gap, i.e. a
  reranker-efficiency score that → 1 when the reranker fixes everything) in a
  non-retrieval domain, with no representation-alignment floor. ARI is the
  **inverse complement** (irreducible / remaining) and carries the alignment
  reading that ρ_k has no analogue for. So ARI is not a restatement of ρ_k either.

**Net on Q1:** ARI is **a re-expression of L∞, not a new metric — and the paper
treats it as exactly that.** Its defensible novelty is the cross-lingual,
mate-twin instantiation + the alignment-lever reading, which is the same kind of
"standard frame, new cross-lingual axis" credit the paper already earns for XRC
(Pareto frame) and RRC (cascade knee). The danger is *only* if a draft elevates
ARI to "a new metric/contribution"; the current draft does not. **Keep it that
way — ARI must remain the "reading of the RRC curve," never a fourth cost object.**
One bib-ready citation (ρ_k, below) closes the residual exposure.

### Q2. Is the per-route corner-movement framed without over-claiming routing?

**Yes — this is the most disciplined new-object addition in the paper.** The
per-route frontier (cp_fig23, §8 lines 1086–1126) is the exact dreamer route-1 from
my round-3 handoff ("make the frontier cross-lingual-pair-resolved … a frontier
whose membership changes with the language direction is a genuinely new deployment
artifact"). It is executed with a four-point honesty contract baked into the prose:

1. **The single-model recommendation is the spine, the router is headroom.** "This
   nuances, but does *not* overturn, the single-model recommendation" (line 1093);
   "we report it as headroom, not a recommendation" (line 1103). Correct.
2. **embeddinggemma's primacy is restated with the robust evidence** (global
   capability corner, smallest $L_\infty$, lowest ARI@100, wins 3/5 routes incl.
   the two hardest, recall-only picks it on all five). The corner-movement is the
   *upside*, not a reversal.
3. **Thin-n is stated inline (de n=7, zh n=2, es n=0; cross n=22–34) and the
   per-route XRC axis is labeled INDICATIVE; es XRC is explicitly never imputed**
   (lines 1104–1109), and repeated in Limitations (lines 1211–1219). The figure
   itself (cp_fig23) prints "ROBUST" on the CLIR axis and "INDICATIVE" / "es:
   undefined same-lang denominator" on the XRC axis — figure and prose agree.
4. **The defensible spine is named: the robust per-route CLIR@10$_\ell$ axis +
   frontier membership + the *existence* of corner movement (3 corners, flips on
   2/5 routes).** This is the right load-bearing claim; the XRC y-position of each
   route is correctly demoted to indicative.

This is a model of how to add a deployment-relevant object without over-claiming.
**No over-claim on routing.** The only thing I'd watch (cohesion, not novelty) is
that "per-route router" must never migrate into the abstract / intro / conclusion
as a delivered result — and per the story it does not (the story's open-risk #1
forbids it). Novelty-wise, per-route / per-direction Pareto model-selection for
CLIR is a genuinely new deployment artifact (generic Pareto retriever plots are
one-point-per-model in global space); claiming it as *headroom* rather than a
*method* is the correct, unimpeachable framing.

### Q3. Final-stretch verdict per contribution — is anything still exposed?

| C | Verdict (round 4) | Δ from round 3 | Exposure |
|---|---|---|---|
| **C1** benchmarks | **NOVEL, well-defended** | unchanged (FROZEN) | none — the paper's safest novelty |
| **C2** metric family (XRC/RRC/**ARI**/DEG) | XRC NOVEL-axis; RRC INCREMENTAL-knee + NOVEL-floor; **ARI INCREMENTAL-reading, correctly subordinated** | ARI added, cleanly | **residual:** add ρ_k cite (below) |
| **C3** mechanism (separability ⇒ floor ⇒ ARI) | INCREMENTAL-confirmation + NOVEL-confusability + NOVEL-floor + **ARI as falsifiable per-model split** | strengthened | none new |
| **C4** deployment (frontier + knee + **per-route upside**) | INCREMENTAL-rule + NOVEL-frontier-decision + **per-route as bounded headroom** | per-route added, fenced | none — contract holds |
| **C5** pipeline | INCREMENTAL (support) | unchanged | none |

**Nothing is exposed to a novelty rejection.** The round-1/2/3 top risks
(CLIR-MRS-as-contribution; "first decomposition"; "cheapest = embeddinggemma"; the
uncredited knee) are all dead. The two new objects are credited where borrowed and
claimed only on the cross-lingual instantiation.

---

## Highest-risk over-claims (ranked) — all minor this round

1. **ARI's residual prior-art (the round's one real residual).** The
   "decompose the shortfall into recoverable + irreducible, normalized" *shape* has
   a published cousin in long-tailed reranking (arXiv:2604.01506, ρ_k). A
   reranking-literate reviewer could say "this is a residual decomposition, see
   2604.01506." The defense is easy because ARI is the **inverse ratio in a
   different domain with an alignment floor that ρ_k lacks** — but only if the
   paper *cites* it and claims only the cross-lingual/alignment instantiation.
   **Fix:** one optional-but-recommended cite in the ARI paragraph (§4, ~line 466)
   or §2, with half a clause: *"normalizing a re-ranking remainder by the
   recoverable gap has a precedent in long-tailed reranking
   \citep{residualrerank2026}; we invert it (the un-rerankable share) and tie it to
   representation alignment cross-lingually."* This converts the only residual
   "you reinvented X" surface into credited-and-distinguished. Near-mandatory if
   the venue's reviewer pool overlaps reranking; safe to ship without it only
   because ARI is explicitly *not* claimed as a contribution.

2. **"the three sum to 1.0 for all nine models" vs. cp_fig22 plotting seven
   (cohesion/correctness-adjacent, not novelty, but I caught it while verifying the
   ARI claim).** The prose (lines 458, 689) and the **figure caption (line 703,
   "sum to $1.0$ for all nine models")** say nine; **cp_fig22 itself plots only the
   7 non-degenerate models** and its in-figure title says "7 non-degenerate
   models." The identity *does* hold for all nine trivially, so the prose is true,
   but the **caption claims nine while the figure shows seven** — a reviewer who
   counts bars will flag it. **Fix (hand to writer/correctness):** either change the
   caption to "...for all seven non-degenerate models shown (and trivially for all
   nine)" or note the two degenerate models are omitted from the panel. Pure
   bookkeeping; no number changes. I flag it because it sits on the exact claim the
   conductor asked me to stress-test.

3. **"alignment-only" remains an inference, not an intervention (carried from
   round 3, unchanged risk).** ARI sharpens the claim — it now *quantifies* the
   alignment-only share (0.229 for the deployed model) — which makes the
   correlational basis ($r=+0.96$, separability) carry slightly more weight. The
   draft hedges correctly (Limitations names the causal probe as forthcoming, lines
   1245–1257, and ties the $L_\infty$/ARI floor to the before/after target). Low
   risk, already handled; do **not** weaken the $L_\infty$/ARI numbers (regression-
   checked), only keep the "recoverable by alignment" attribution visibly
   inferential, as the draft does.

No over-claim rises to "a hostile reviewer rejects on novelty."

---

## Missing citations the paper should add (bib-ready)

The mandatory list from rounds 1–3 is fully integrated, **including the round-3
near-mandatory cascade cite** (nogueira2019multistage, gao2021rethink — verified
present and resolving). Round 4 surfaces **one near-mandatory** addition:

- **(NEAR-MANDATORY — ARI residual decomposition) The long-tailed reranking
  residual-decomposition reference**, to credit-and-distinguish the
  "decompose-and-normalize a reranking remainder" shape:
  ```
  @article{residualrerank2026,
    title  = {Beyond Logit Adjustment: A Residual Decomposition Framework for
              Long-Tailed Reranking},
    author = {Wang and others},
    journal= {arXiv preprint arXiv:2604.01506},
    year   = {2026}
  }
  ```
  Cite once in the ARI paragraph (§4) or §2 with the half-clause in over-claim #1.
  This is the single citation that closes ARI's only residual exposure; without it
  ARI is still defensible *because the paper does not claim it as a contribution*,
  but with it the "you reinvented residual decomposition" line is pre-empted.

- **(OPTIONAL — per-route / per-direction model selection lineage)** If the writer
  wants to anchor the per-route frontier's "membership changes with language
  direction" as a known *kind* of analysis, the CLIRMatrix directional decomposition
  (clirmatrix2020, already cited) is the natural anchor — no new key needed; one
  back-reference suffices. Not blocking.

- **(OPTIONAL, carried) Artetxe & Schwenk / LaBSE Tatoeba "mate accuracy"** as the
  lineage of "mate-retrieval" (arXiv:1811.01136 / 2007.01852). Still not blocking.

Only the ρ_k residual-decomposition cite is one I would now add before submission;
it is the sole new "missing citation (not a wording choice)" surface this round,
and it is cheap.

## What WOULD make the weakest contribution clearly novel (hand to dreamer)

Both round-3 routes have been executed: route-1 (per-route frontier) is now cp_fig23,
and the round-3 weakest novelty item (the cost frontier as "standard Pareto plot")
is resolved by the per-route object that a generic Pareto plot cannot be. The new
weakest *novelty* item is **ARI**, because it is a re-expression rather than a new
quantity. The single route that would convert ARI from "a reading of RRC" into a
*causal, headline* result is the one already named in Limitations and now handed a
precise target by cp_fig22:

- **The W3 alignment causal probe, with ARI as the before/after target.** Fit a
  per-language alignment map on one model, re-retrieve, recompute
  XRC50 / RRC@100 / **ARI@100**. If $L_\infty$ / ARI@100 *drops* under alignment
  while staying flat under re-ranking, "the alignment-only share" stops being a
  correlational adjective and becomes the paper's headline **causal** result — and
  ARI is then not just a re-expression of L∞ but the *measured movable quantity* of
  an intervention, which is genuinely novel. This is the highest-value upgrade and
  the paper's most memorable *next* result; it correctly remains UPSIDE-only in
  Limitations and nothing in the paper depends on it. **Do not attempt it this
  round** — the story's freeze call is right; this is a post-submission run.

**Convergence note (novelty axis):** the paper is done on novelty. Freeze the
spine. The only two things I'd land before submission are the ρ_k cite
(near-mandatory, cheap) and the cp_fig22 "nine vs. seven" caption fix
(bookkeeping). Everything else — C1, the DEG gate, the cascade-credited knee, the
XRC-axis / Pareto-frame split, the per-route four-point contract — is
novelty-defensible exactly as written.

---

### Sources (web-searched June 2026)
- Nogueira et al., *Multi-Stage Document Ranking with BERT* — https://arxiv.org/abs/1910.14424
- Gao, Dai, Callan, *Rethink Training of BERT Rerankers in Multi-stage Retrieval Pipeline* — https://dl.acm.org/doi/10.1007/978-3-030-72240-1_26 (arXiv:2101.08751)
- *Beyond Logit Adjustment: A Residual Decomposition Framework for Long-Tailed Reranking* (ρ_k) — https://arxiv.org/abs/2604.01506
- *What Drives Cross-lingual Ranking? Retrieval Approaches with Multilingual LMs* — https://arxiv.org/pdf/2511.19324
- Bounded recall / recall-ceiling in two-stage retrieval (background) — https://www.pinecone.io/learn/series/rag/rerankers/
