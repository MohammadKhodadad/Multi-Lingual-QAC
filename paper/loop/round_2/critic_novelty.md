# Novelty review (round 2)

Reviewer #1 — Novelty Critic. EMNLP Industry Track. Verdicts re-grounded in the
literature (web-searched June 2026) and checked against my own round-1 review
(`paper/loop/round_1/critic_novelty.md`). This round's question: did the round-2
reframes land, and are the two *new headline metrics* (XRC, RRC) genuinely novel?

## Verdict summary (1 paragraph): is the paper's novelty defensible as written?

**Yes — the novelty is now defensible as written, which it was not in round 1.**
All six of my round-1 "you reinvented X" attack surfaces have been closed by
mechanical, accurate citations and explicit boundaries: **CLEF-IP** and **DAPFAM**
now defend C1 (and I verified both — CLEF-IP gold *is* prior-art citation relevance,
not parallel translation; DAPFAM *does* use family-level citation gold, exactly the
`publication_number` equivalence the paper rejects), **CLIRMatrix** is cited in the
same sentence as the directional matrix and the "new instrument" connotation is gone,
**C3 is reframed from "first decomposition" to "we confirm [2511.19324, 2507.07543]
on a content-controlled corpus"** (verified: 2511.19324 does reach the
alignment-not-translation conclusion, on three general benchmarks, not patents — so
the content-control delta is real), the **QT-vs-DT budget rule** now cites Oard 1998 /
Saleh & Pecina 2020, and **cross-lingual RBO** cites Bailey et al. 2017. The single
highest round-1 risk — **CLIR-MRS as a contribution** — is fully neutralized: it is
demoted in the contribution list, the metrics section, the leaderboard, and the
deployment section to "table-ordering convenience," and the paper now *honestly
reports* that the composite ranking is aggregation-sensitive (embeddinggemma's rank
ranges [1,4]) and rests the recommendation on per-axis dominance instead. The two new
headline metrics are the round's real novelty test: **XRC (cross-lingual reading-cost
multiplier) is NOVEL** — I found no prior cross-lingual depth-ratio metric, and the
"documents-read, monotone-invariant" framing fills a clean gap; **RRC (re-ranker
recoverability ceiling) is INCREMENTAL-but-fairly-claimed** — the *concept* that
first-stage recall caps any re-ranker is textbook and widely stated, but the paper
claims only the *cross-lingual per-model quantification* and the falsifiable bound,
which is a fair and useful framing, not an over-claim. Net: C1 NOVEL (well-defended),
C2 NOVEL-via-XRC + INCREMENTAL-suite, C3 INCREMENTAL-confirmation + NOVEL-confusability,
C4 INCREMENTAL-rule + NOVEL-null-on-patents. No remaining claim would let a hostile
reviewer reject on novelty grounds. Three residual nits below, all minor.

---

## Did the round-1 points get addressed?

| Round-1 risk | Status in round-2 draft | Verdict |
|---|---|---|
| R1 #1: CLIR-MRS as a contribution | Demoted everywhere; C2 explicitly says "composite NOT claimed as a contribution"; §6.3 shows rank-range [1,4]; §8 rests on per-axis dominance | **CLOSED** |
| R1 #2: "first decomposition" vs 2511.19324/2507.07543 | Rewritten to "we *confirm*… on a content-controlled parallel corpus that removes the translationese/content confound those studies could not" (§2, C3) | **CLOSED** |
| R1 #3: directional matrix vs CLIRMatrix | "Following the CLIRMatrix-style decomposition" in §4; cited in §2 | **CLOSED** |
| R1 #4: budget rule vs Oard 1998 / Saleh & Pecina | Cited in §2, §4-ish, and §8 ("re-derives the classic query-vs-document translation rule") | **CLOSED** |
| R1 #5: C1 "first" with no CLEF-IP | CLEF-IP + DAPFAM now anchor §2 patent-IR paragraph; "first" narrowed to "content-controlled, chemistry-ontology-grounded" | **CLOSED** |
| R1 #6: separability AUC implied as new | §4 now says "ROC-AUC… is a standard separability diagnostic; our use is the same-vs-cross decomposition and the re-ranker corollary" | **CLOSED** |
| R1 dreamer route #3 (operational separability metric) | Realized as **XRC + RRC**, with Conformal-RAG cited as *future-work machinery* only | **CLOSED + UPGRADED** |
| R1: cross-lingual RBO needs Bailey 2017 | Cited in §2 and §4 | **CLOSED** |
| R1: model-overlap vs 2605.24297 (English-only patent eval) | §2 chemistry paragraph cites it and distinguishes on cross-lingual | **CLOSED** |

Every round-1 actionable was addressed accurately. This is an unusually clean
revision on the novelty axis.

---

## Claim-by-claim (re-verdict)

### C1 — The two benchmarks (load-bearing)
- **CLAIM:** *"The first content-controlled, chemistry-ontology-grounded cross-lingual
  patent-retrieval benchmarks… not `publication_number` equivalence and not
  machine-translated documents."* → **NOVEL (well-defended).**
  Closest prior: CLEF-IP (multilingual en/de/fr patent retrieval, but gold = prior-art
  *citation* relevance judgments — confirmed,
  https://link.springer.com/chapter/10.1007/978-3-030-22948-1_15 ,
  https://ceur-ws.org/Vol-1175/CLEF2009wn-CLEFIP-RodaEt2009.pdf ); DAPFAM (family-level
  *citation*-based gold, the design rejected — confirmed,
  https://arxiv.org/abs/2506.22141 ); CLIRMatrix (Wikipedia parallel,
  https://aclanthology.org/2020.emnlp-main.340/ ).
  The triple intersection (parallel human-translated patents × ChEBI ontology gold ×
  confusable-neighbour negatives) remains unscooped, and the "first" is now narrowed
  to exactly the axes where it is true. **No change needed.**

### C2 — The robustness-metric family
- **CLAIM (suite):** *"A cross-lingual robustness-metric family reported co-equally
  with recall."* → **INCREMENTAL (honestly pitched).** The paper now calls it a
  "purpose-built suite," not "new metrics." Fine.
- **CLAIM (XRC):** *"XRC, the cross-lingual reading-cost multiplier… the factor by
  which a practitioner must read deeper to reach a foreign twin than a same-language
  copy… in documents read and monotone-invariant."* → **NOVEL.**
  I searched specifically for a cross-lingual reading-depth / recall-depth ratio metric
  and found none: prior CLIR work reports Recall@k, nDCG, MAP, MLRS (language-preference),
  and bitext "mate accuracy" — none expresses the cross-vs-same *depth ratio* as a
  deployment cost. The recall-ceiling-of-cascades literature is about *re-rankers*, not a
  cross-lingual depth multiplier. The conformal-IR line (Conformal-RAG; Streamlining
  Conformal IR, https://arxiv.org/pdf/2410.02914 ) gives coverage guarantees but not a
  cross-lingual depth-cost number — and the paper correctly cites Conformal-RAG only as
  the *future-work machinery* for a guaranteed XRC, claiming the empirical XRC on its own.
  This is the cleanest positive novelty in the paper. **Keep; the censoring discipline
  (XRC50 finite headline, D90/D95 as lower bounds) also pre-empts the obvious attack.**
- **CLAIM (RRC):** *"RRC@K… the fraction of cross-lingual queries whose foreign twin
  appears within the top-K candidates; 1−RRC is provably unrecoverable by any top-K
  re-ranker."* → **INCREMENTAL (fairly claimed, not over-claimed).**
  The underlying fact — "a re-ranker cannot recover what first-stage retrieval never
  surfaced; first-stage recall caps the cascade" — is textbook and widely stated in the
  two-stage-retrieval literature (e.g.
  https://www.emergentmind.com/topics/two-stage-retrieval-method ; the recall-ceiling
  phrasing is essentially folklore). The paper does **not** claim to discover this; it
  claims the *cross-lingual, per-model, mate-twin instantiation* ("RRC converts 'align,
  don't re-rank' from a slogan into a per-model upper bound"). That is an accurate and
  useful framing. **Minor risk:** a hostile reviewer could still say "this is just
  Recall@K of the foreign twin renamed." It partly is — RRC@K is literally the cumulative
  CDF of first-foreign-twin rank, i.e. mate-hit@K restricted to cross-lingual queries.
  Recommended one-clause hedge (see over-claims §1) to make the *re-ranker-bound*
  interpretation — not the quantity — the contribution.
- **CLAIM (mate-retrieval):** the draft uses "mate-retrieval / mate-hit@k / mate-MRR."
  → **NOT NOVEL as a name (correctly not claimed).** "Mate retrieval / mate accuracy"
  is established bitext-mining terminology (LASER/LaBSE/Tatoeba). The paper presents it
  plainly as a measurement, not a contribution — good, leave as is.
- **CLAIM (directional matrix, separability AUC, CLIR-MRS):** all three now correctly
  framed as standard-machinery-applied-to-patents (matrix), decomposition-is-the-novelty
  (AUC), and ordering-convenience-only (CLIR-MRS). **All CLOSED.**

### C3 — The mechanism finding
- **CLAIM:** *"availability sets the stage but a residual encoder bias remains… the
  collapse is an embedding-level separability deficit… a monolingual re-ranker cannot
  recover under-scored foreign twins, a claim we bound per-model with RRC."*
  → **INCREMENTAL (the alignment-not-translation core) + NOVEL (chemistry confusability
  + the falsifiable per-model bound).** Verified that 2511.19324 reaches the
  alignment-over-translation conclusion on three *general* benchmarks (its abstract:
  "prioritise semantic multilingual embeddings and targeted learning-based alignment
  over translation-based pipelines"), so the paper's "we confirm, on a content-controlled
  parallel corpus that removes the confound those studies could not" is the honest and
  correct framing. The chemistry same-language-sibling confusability trap and the
  universal-attractor finding remain genuinely novel and under-celebrated.
- **CLAIM (availability slope −0.57 ⇒ residual encoder bias):** **NOVEL (small, clean)**
  and correctly labelled DESCRIPTIVE on n=5 languages. This strengthens C3 honestly.
- **CLAIM (confusion = separability collapse, r=+0.96 robust):** the *robust* mechanism;
  separability AUC machinery is standard, the same-vs-cross decomposition + RRC corollary
  is the novelty. Correctly pitched.

### C4 — The deployment recommendation
- **CLAIM:** *"Deploy embeddinggemma, on per-axis dominance… machine-translating the
  question is safe (−0.044, p=0.13)… spend human-translation budget on the corpus."*
  → **INCREMENTAL (the QT-vs-DT rule) + NOVEL (the null *on patents/embeddings* and the
  per-axis-dominance deployment logic).** Oard 1998 / Saleh & Pecina 2020 now cited; the
  paper claims only the embedding-era patent re-derivation and the quantified null. The
  concurrent English-only patent-embedding eval (2605.24297) is cited and distinguished
  on cross-lingual ordering. **No change needed.**
- **CLAIM (don't reflexively ensemble; RRF loses):** INCREMENTAL but honest; RRF cited;
  routing kept as hypothesis. Fine.

### C5 — Generation + validation pipeline
- **INCREMENTAL (support, correctly positioned and softened).** Now "human validation
  summarized in the system description," `\todo`s moved to LaTeX comments. Acceptable;
  not a novelty exposure.

---

## Highest-risk over-claims (ranked) — all minor this round

1. **RRC framed as a *new metric* rather than a *new use* (lowest-grade exposure).**
   RRC@K is the CDF of first-foreign-twin rank — i.e. cross-lingual mate-hit@K renamed.
   The contribution is the *re-ranker-ceiling interpretation* (1−RRC unrecoverable), not
   the quantity. **Fix:** one clause in §4 RRC definition, e.g. "RRC@K is the cumulative
   first-foreign-twin hit rate (mate-hit@K on cross-lingual queries); our contribution is
   reading it as a *per-model re-ranker ceiling*: 1−RRC@K is provably unrecoverable."
   This pre-empts the only "you renamed recall" attack left.

2. **"XRC… monotone-invariant" — make sure the term is defended, not asserted.** XRC is a
   ratio of depths, so it is invariant to monotone score transforms; that is true and a
   genuine selling point versus a unitless composite. But a picky reviewer will want the
   one-line reason. **Fix (optional):** add a half-sentence in §4 ("a ratio of retrieval
   depths is invariant to any monotone re-scoring of similarities, unlike an AUC or a
   weighted composite"). Strengthens the novelty rather than defends an over-claim.

3. **"the directional CLIR matrix" still reads slightly proprietary in the contribution
   list (C2) even though §2/§4 cite CLIRMatrix.** Low risk because the body attributes it,
   but C2's bullet lists it among "our" family without the cite. **Fix (cosmetic):** in
   C2 say "the directional CLIR matrix (CLIRMatrix-style)" so the contribution list itself
   carries the attribution a skimming reviewer sees first.

No remaining over-claim rises to "a hostile reviewer rejects on novelty." The round-1
top-2 risks (CLIR-MRS-as-contribution; "first decomposition") are both gone.

## Missing citations the paper should add (bib-ready)

The round-1 mandatory list is fully integrated (CLEF-IP, CLIRMatrix, DAPFAM, 2511.19324,
2507.07543, Oard 1998, Saleh & Pecina 2020, Bailey 2017, MMTEB, 2605.31142, 2605.24297).
Only two small, *optional* adds remain, both to harden XRC/RRC novelty:

- **(Optional, RRC framing)** A two-stage-retrieval / recall-ceiling reference so the
  paper can cite the *known* qualitative fact and then claim only the cross-lingual
  quantification — this is the cleanest way to defuse over-claim #1. Any standard
  cascade-retrieval reference works; e.g. a multi-stage dense-retrieval pipeline paper
  that states the first-stage recall ceiling explicitly. Keeps RRC honest as a *new use*.
- **(Optional, mate-retrieval lineage)** Artetxe & Schwenk margin-based bitext mining /
  LaBSE Tatoeba "mate accuracy" (https://arxiv.org/pdf/1811.01136 ;
  https://arxiv.org/pdf/2007.01852 ) as the lineage of the *mate-retrieval* term, so a
  bitext-mining reviewer sees the paper knows "mate retrieval" is borrowed and is using
  it as machinery, not claiming it. One inline cite in §4 mate-retrieval paragraph.

Neither is mandatory; the paper is citation-defensible without them.

## What WOULD make the weakest contribution clearly novel (hand to dreamer)

The weakest *novelty* item is now **RRC** (a renamed CDF with a known ceiling
interpretation) — and the weakest *evidence* item is still the demoted CLIR-MRS (an
unvalidated composite). Two routes, in priority order:

1. **Turn RRC from "renamed recall" into a genuinely new object: a re-ranker-budget
   frontier.** Right now RRC@K is a single number per K. Make it a *curve* and report the
   **marginal recoverability** dRRC/dK and the **knee** — "past top-K* a re-ranker buys
   almost nothing" — and pair it with XRC to give a single 2-D *cross-lingual cost
   frontier* (reading-depth cost on one axis, re-ranker-recoverable fraction on the other)
   per model. A frontier that lets a practitioner read off "how deep must my first stage
   go before re-ranking is worth it, for this language pair" is a deployment object with
   no precedent I can find, and it converts RRC from a recall rename into a planning tool.
   This is the highest-leverage novelty upgrade still on the table and it is CPU-only on
   existing score lists.

2. **Validate XRC (not CLIR-MRS) against a downstream cost.** XRC is "documents read";
   tie it to a *real* cost — re-rank latency or LLM-token budget at depth D — and show
   XRC predicts end-to-end cross-jurisdiction search cost better than mean recall. That
   makes XRC the validated headline and retires the CLIR-MRS-validation backlog item as
   moot (the composite stays demoted permanently). The external-utility eval is already
   in `needs_eval.md` (CLIRMRS-external-validation) — point it at XRC instead.

(Both 1 and 2 keep the paper's "align, don't re-rank" thesis and need no new model runs;
route 1 needs none at all.)
