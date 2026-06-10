# Novelty review (round 1)

Reviewer #1 — Novelty Critic. EMNLP Industry Track. Verdicts are grounded in the
literature (web-searched June 2026) and cross-checked against the authors' own
level-2 novelty audits (`supplementry_material/level2_multilingual_ir_research_directions.md.pdf`).

## Verdict summary (1 paragraph)

The paper's novelty is **defensible, but only after two reframings and four added
citations** — as written it contains one genuinely novel asset, several
incremental-but-fine repackagings, and **two claims a hostile reviewer can shoot
down outright.** The defensible core is C1's *content-controlled parallel asset*:
a multilingual chemistry-patent retrieval benchmark whose gold is genuinely
human-translated parallel text plus ontology membership rather than
`publication_number` equivalence. No prior benchmark combines (parallel
human-translated patents) × (chemistry/ChEBI ontology gold) × (confusable-neighbour
hard negatives), and the level-2 PDF is right that this asset is "structurally
harder to scoop" than the metrics. The weak points are the **metric family (C2)**
and the **mechanism story (C3)**, both of which the paper frames with "first"/"new"
language that the literature does not support: the directional query×document recall
matrix is CLIRMatrix (EMNLP 2020); same-language home advantage, the English
exception, and directional asymmetry are documented in *The Cross-Lingual Cost*
(2507.07543) and *What Drives Cross-lingual Ranking?* (2511.19324); separability/ROC-AUC
of gold-vs-nongold scores is a standard retrieval diagnostic; and the "MT the query,
human-translate the corpus" budget rule (C4) restates the 25-year query-vs-document
translation literature (Oard 1998; Saleh & Pecina ACL 2020). The paper is **safe if
it reframes every contribution as "first *patent-grounded, content-controlled*
instantiation"** and cites the precedents above in the first sentence of each
comparison (the paper already does this well for the calibration/conformal line, so
the template exists). The single biggest exposure is the **composite metric
(CLIR-MRS/MRS)**: an arbitrary hand-weighted aggregate with no validation against an
external criterion is the kind of "invented number" reviewers reject — it should be
demoted from a contribution to a reporting convenience, or validated. Net: novelty is
real but **over-stated in C2/C3/C4 and under-defended in C1**; with the fixes below it
clears the industry-track bar.

---

## Claim-by-claim

### C1 — The two benchmarks (the load-bearing contribution)

- **CLAIM:** *"Two patent-grounded multilingual chemistry-retrieval benchmarks built
  only from human-translated patent text … gold relevance defined by a chemical
  ontology … and by genuinely parallel human-translated patents — not by
  `publication_number` equivalence … and not by machine-translated documents."*
  → **NOVEL (as a combined asset).**
  Closest prior:
  - CLEF-IP 2009–2013 — multilingual (en/fr/de) patent prior-art retrieval, the
    canonical cross-lingual patent benchmark. https://link.springer.com/chapter/10.1007/978-3-030-22948-1_15
    and https://ceur-ws.org/Vol-1175/CLEF2009wn-CLEFIP-RodaEt2009.pdf — **the paper
    does not cite CLEF-IP at all; this is the single most dangerous omission for C1.**
  - DAPFAM (2025) — a *family-level* patent retrieval benchmark that explicitly
    aggregates family members to handle cross-jurisdiction redundancy. https://arxiv.org/pdf/2506.22141
  - CLIRMatrix (EMNLP 2020) — massively multilingual CLIR with Wikipedia parallel
    structure. https://aclanthology.org/2020.emnlp-main.340/
  - ChEBI ontology grounding. https://academic.oup.com/nar/article/36/suppl_1/D344/2506390
  → **Recommended framing:** keep "first," but make it *narrow and true*: "the first
  cross-lingual **chemistry**-patent retrieval benchmark whose gold is **content-controlled**
  (human-translated parallel text + ontology membership) and whose negatives are
  **chemically-confusable neighbours**." Explicitly position against CLEF-IP ("multilingual
  patent retrieval existed, but its gold is prior-art relevance judgments, not parallel
  human translations, and it is not chemistry/ontology-grounded") and against DAPFAM
  ("family-level gold — exactly the `publication_number` equivalence we argue is unsafe").
  This turns two would-be "you reinvented X" attacks into a defended boundary.

- **CLAIM:** *"side-stepping the patent-family non-equivalence problem … members of a
  patent family 'may differ in their claims,' so `publication_number`-equivalence is an
  unsafe gold signal."*
  → **NOVEL framing of a known fact (good).** The fact (WIPO eTISC; PRIME/Fujii &
  Ishikawa) is established; using it as a *benchmark-construction principle* is a clean,
  defensible contribution. DAPFAM is the counter-example that *does* use family gold —
  cite it as the design the paper deliberately rejects. https://arxiv.org/pdf/2506.22141

### C2 — The robustness-metric family

- **CLAIM:** *"A cross-lingual robustness-metric family … CLIR@k and the home-advantage
  gap, mate-retrieval, cross-lingual RBO, language-collapse, and embedding separability
  AUC … reported co-equally with recall."* (as a *family/suite*)
  → **INCREMENTAL.** Each instrument has a precedent; the *assembly* into a co-reported
  patent-retrieval suite is the (modest) novelty. The paper's "orthogonal to MTEB"
  framing is fair, but "purpose-built suite" is the honest level, not "new metrics."

- **CLAIM:** *"directional CLIR matrix … query-language → document-language recall matrix
  and its asymmetry."*
  → **NOT NOVEL (metric); NOVEL only in domain.** Closest prior: **CLIRMatrix** is
  literally a query×document language matrix; directional asymmetry (X→EN ≫ EN→X; Chinese
  weakest) is documented there and in *The Cross-Lingual Cost* (2507.07543) and the CLIR
  survey corpus. https://aclanthology.org/2020.emnlp-main.340/ ,
  https://arxiv.org/abs/2507.07543
  → **Recommended framing:** drop any "new" connotation; present as "we apply the
  CLIRMatrix-style directional analysis to parallel patents" and cite CLIRMatrix in the
  same sentence.

- **CLAIM:** *"CLIR@k vs MoLIR@k and the home advantage … the direct instrument for
  'average recall hides collapse.'"*
  → **INCREMENTAL.** The *phenomenon* (same-language ≫ cross-language; English exception)
  is the headline result of *The Cross-Lingual Cost* (Park & Lee 2025) and *What Drives
  Cross-lingual Ranking?* (2511.19324). The MoLIR/CLIR *split as a named reported pair* is a
  reasonable packaging, but "home advantage" must be presented as a renaming/operationalisation
  of an already-documented same-language bias, not a discovery. https://arxiv.org/abs/2507.07543 ,
  https://arxiv.org/abs/2511.19324

- **CLAIM:** *"Cross-lingual RBO … consistency axis: does the deployed index return the
  same patents regardless of query language."*
  → **NOVEL (application).** RBO (Webber et al. 2010) is standard, and RBO has been used to
  characterise retrieval *consistency across query variations* (Bailey/Moffat et al., SIGIR
  2017). https://people.eng.unimelb.edu.au/ammoffat/abstracts/bmst17sigir.pdf
  → **Recommended framing:** "we treat the five language versions of a query as query
  *variants* and apply RBO consistency analysis (cf. retrieval-consistency-under-query-variation,
  Bailey et al. 2017) **cross-lingually** — a new use of an existing instrument." This is a
  genuinely clean, defensible novelty *if* the query-variation RBO lineage is cited; without
  that citation it looks like an unattributed reinvention. **Add Bailey et al. 2017.**

- **CLAIM:** *"Separability AUC … AUC(score(gold) > score(non-gold)) … foreign gold is
  under-scored, not merely mis-ordered."*
  → **NOT NOVEL (metric); NOVEL (the same-vs-cross split + the re-ranker implication).**
  ROC-AUC over positive/negative similarity scores as a separability/calibration diagnostic
  is standard (e.g. https://arxiv.org/pdf/1910.11005 , https://arxiv.org/html/2510.00137).
  → **Recommended framing:** do not present AUC as a new metric. The *defensible* novelty is
  the **decomposition** (same-language AUC vs cross-language AUC) and the **falsifiable
  consequence** ("under-scored ⇒ a monolingual re-ranker cannot recover it"). Pitch exactly
  that, cite AUC-as-separability as standard machinery.

- **CLAIM:** *"composite … CLIR-MRS = cap × (0.5 + 0.5·rob) … MRS is the min–max-normalised
  mean of five axes."*
  → **NOT NOVEL and HIGH-RISK (this is the weakest contribution).** Hand-weighted composite
  scores with author-chosen coefficients (the ±50% modulation, the min–max five-axis mean) have
  no external validation and are exactly what reviewers attack as arbitrary. The community trend
  is *away* from single composites toward consistency-aware aggregation (MMTEB uses Borda count;
  the multilingual-ranking-robustness study, 2605.31142, shows rankings are sensitive to the
  aggregation scheme itself). https://arxiv.org/html/2502.13595v4 ,
  https://arxiv.org/html/2605.31142v1
  → **Recommended framing:** demote CLIR-MRS/MRS from a *contribution* to a *reporting
  convenience used only to order tables*, and show the per-axis numbers are what carry the
  argument (they already do — `embeddinggemma` leads on CLIR@10, RBO, mate-rank, and separability
  individually, so the composite is not load-bearing). Better: report a rank-aggregation
  (Borda) and a bootstrap CI and state the conclusion is invariant to the weighting. Otherwise a
  reviewer rejects the headline number and the deployment recommendation built on it.

### C3 — The mechanism finding

- **CLAIM:** *"First decomposition of cross-lingual chemistry-retrieval failure into an
  availability confound + a chemistry-specific confusability trap + an embedding-level
  separability deficit … yielding the falsifiable, deployment-relevant claim that a
  monolingual re-ranker cannot recover under-scored foreign twins."*
  → **INCREMENTAL (availability + separability halves are known); NOVEL (the chemistry-confusability
  trap + the explicit re-ranker corollary).**
  Closest prior:
  - *What Drives Cross-lingual Ranking?* (2511.19324) already attributes the cross-lingual gap to
    *weak semantic alignment* (not translation) and concludes systems should *prioritise alignment
    over translation pipelines* — this is **nearly the paper's C3/C4 thesis**, on general benchmarks.
    https://arxiv.org/abs/2511.19324
  - *The Cross-Lingual Cost* (2507.07543): the gap is the retriever's difficulty ranking across
    languages; the fix is enforcing balanced retrieval. https://arxiv.org/abs/2507.07543
  → **Recommended framing:** *do not say "first decomposition."* Say: "we **confirm** the
  alignment-not-translation finding of [2511.19324, 2507.07543] **on a content-controlled parallel
  corpus that removes the translationese/content confounds those studies could not**, and we add a
  chemistry-specific failure mode (sibling-compound confusability) and a separability-AUC test that
  turns 'alignment is the fix' into a falsifiable claim about re-rankers." The content-control is the
  real delta and the paper should lead with it. **Add both citations — currently neither is cited and
  2511.19324 is the closest competitor in the paper.**

- **CLAIM:** *"a chemically-confusable wrong compound out-ranks every gold patent on 14–78% of
  queries … universal attractors (polypeptide/methyl/ethene/hydroxide/dioxygen)."*
  → **NOVEL.** Confusion-rate ("a hard negative beats *all* gold") as a reported retrieval metric,
  instantiated with ontology-derived chemical look-alikes, has no direct precedent I found; nearest
  neighbours are distractor-ranking / hard-negative-mining work that does not report this rate as a
  benchmark axis (https://arxiv.org/html/2510.21440v1). This is one of the paper's most original and
  most defensible findings — **feature it more prominently.**

- **CLAIM:** *"Structure-style questions are the trap (R@10 0.26, confusion 51%) … a
  language-independent formula token measurably helps (p<0.01)."*
  → **NOVEL (small but clean).** Query-phrasing → confusion-susceptibility, with a formula-token
  intervention, is a nice content-specific finding with no obvious precedent.

### C4 — The deployment recommendation

- **CLAIM:** *"machine-translating the question is safe (−0.044, p=0.13) so machine-translate the
  query, but human-translate the corpus."*
  → **NOT NOVEL (the rule); the null-result *evidence on patents* is the only new part.** This is the
  classic **query-translation vs document-translation** question with a 25-year literature the paper
  almost entirely omits:
  - Oard (1998), "Should we translate the documents or the queries…" https://www.researchgate.net/publication/2557827
  - Saleh & Pecina (ACL 2020), "Document Translation vs. Query Translation … Medical Domain" — directly
    on point: query translation suffices/wins in a specialised domain. https://aclanthology.org/2020.acl-main.613/
  → **Recommended framing:** present the budget rule as "consistent with the long QT-vs-DT line
  (Oard 1998; Saleh & Pecina 2020), now re-derived for embedding retrieval over patents, where our
  null result quantifies the query-side MT cost as insignificant." **Add both citations** — otherwise
  a CLIR reviewer will read C4 as unaware of the field's foundational debate. (Note: the paper's claim
  is about the *generated question* being MT'd, while keeping the *corpus* human-translated; that exact
  asymmetric framing is mild novelty, but the underlying rule is old.)

- **CLAIM:** *"do not reflexively ensemble … untuned RRF did not beat the best single model, yet a
  score-aware combiner or per-language routing could win."*
  → **INCREMENTAL but honest.** RRF (Cormack et al. 2009) is cited. The negative result is fine and
  reviewer-pleasing; the routing idea is the "when to translate"/QPP-routing horizon the level-2 PDF
  marks as future work, so it must stay a hypothesis, not a result (the draft does this correctly).

- **CLAIM:** *"Deploy `embeddinggemma`."*
  → **NOVEL (specific finding); but note the overlap risk.** A concurrent English-only study
  *"Benchmarking Patent Embeddings"* (2605.24297) evaluates EmbeddingGemma-300m, BGE-M3, and
  Qwen3-Embedding on patent retrieval. https://arxiv.org/html/2605.24297 → **Cite it** and distinguish:
  "they rank models on monolingual English patent tasks; we rank them on *cross-lingual robustness*,
  where the ordering differs." This pre-empts a "your model comparison is not new" review.

### C5 — Generation + validation pipeline

- **CLAIM:** *"a patent+ontology-grounded multilingual QAC pipeline whose generated queries are
  human-validated and whose auto-grader is calibrated against humans."*
  → **INCREMENTAL (support, correctly positioned).** LLM-generated, human-validated benchmark
  construction is now common practice; as a *supporting* credibility argument this is fine and should
  stay support, not headline. **Risk flag:** the human-eval numbers (8.33/10, 97/100, +4.3pp) are
  `\todo` and not yet under `reports/` — a novelty reviewer will not reject on this, but a "synthetic
  benchmark, why trust it" review is foreseeable; keep C5 visible and, if the numbers cannot be
  traced, soften to qualitative claims.

---

## Highest-risk over-claims (ranked)

1. **CLIR-MRS / MRS as a contribution (C2).** An arbitrary hand-weighted composite presented as a
   deployable score with no external validation, while the field moves to consistency-aware
   aggregation (MMTEB/Borda; 2605.31142). **Fix:** demote to a table-ordering convenience, show
   per-axis dominance carries the result, add a Borda/bootstrap robustness check.
2. **"First decomposition … alignment is the lever" (C3) vs. *What Drives Cross-lingual Ranking?*
   (2511.19324) and *The Cross-Lingual Cost* (2507.07543).** These reach the alignment-not-translation
   conclusion first, on general data. **Fix:** reframe as "confirm on a content-controlled parallel
   corpus + add chemistry-confusability + falsifiable re-ranker test," cite both.
3. **Directional CLIR matrix presented as a new instrument (C2) vs. CLIRMatrix (EMNLP 2020).** **Fix:**
   cite CLIRMatrix in the same sentence; claim only the patent/parallel application.
4. **"MT the query, human-translate the corpus" (C4) vs. the QT-vs-DT literature (Oard 1998; Saleh &
   Pecina 2020).** **Fix:** cite both; claim only the embedding-era patent re-derivation and the
   quantified null.
5. **C1 "first" with no CLEF-IP citation.** The most dangerous *omission*: CLEF-IP is the obvious
   "you reinvented multilingual patent retrieval" attack. **Fix:** cite CLEF-IP and DAPFAM, narrow
   "first" to "first content-controlled chemistry-ontology-grounded" benchmark.
6. **Separability AUC implied as new (C2/C3).** ROC-AUC separability is standard. **Fix:** claim only
   the same-vs-cross decomposition and the re-ranker corollary.

## Missing citations the paper should add (bib-ready)

- **CLEF-IP** — Piroi, Lupu, Hanbury et al., "Multilingual Patent Text Retrieval Evaluation: CLEF–IP"
  (and CLEF-IP 2009–2013 overviews). https://link.springer.com/chapter/10.1007/978-3-030-22948-1_15 ,
  https://ceur-ws.org/Vol-1175/CLEF2009wn-CLEFIP-RodaEt2009.pdf — **(C1, mandatory).**
- **CLIRMatrix** — Sun & Duh, EMNLP 2020, "CLIRMatrix: A Massively Large Collection of Bilingual and
  Multilingual Datasets for CLIR." https://aclanthology.org/2020.emnlp-main.340/ — **(C2 directional
  matrix, mandatory).**
- **DAPFAM** — "DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross-domain patent retrieval,"
  2025, arXiv:2506.22141. https://arxiv.org/pdf/2506.22141 — **(C1, the family-gold design we reject).**
- **What Drives Cross-lingual Ranking?** — arXiv:2511.19324, 2025. https://arxiv.org/abs/2511.19324 —
  **(C3/C4, closest competing thesis; mandatory).**
- **The Cross-Lingual Cost** — Park & Lee, 2025, arXiv:2507.07543 (already in the level-2 PDF).
  https://arxiv.org/abs/2507.07543 — **(C2/C3 home-advantage + directional asymmetry).**
- **Saleh & Pecina** — ACL 2020, "Document Translation vs. Query Translation for CLIR in the Medical
  Domain." https://aclanthology.org/2020.acl-main.613/ — **(C4 budget rule).**
- **Oard (1998)** — "Should we translate the documents or the queries in cross-language information
  retrieval?" — **(C4 budget rule, the canonical statement).**
- **Bailey, Moffat, Scholer, Thomas (SIGIR 2017)** — "Retrieval Consistency in the Presence of Query
  Variations" (RBO for consistency across query variants).
  https://people.eng.unimelb.edu.au/ammoffat/abstracts/bmst17sigir.pdf — **(C2 cross-lingual RBO
  lineage).**
- **MMTEB** — Enevoldsen et al., 2025, arXiv:2502.13595 (Borda aggregation rewards cross-task
  consistency — supports demoting CLIR-MRS). https://arxiv.org/html/2502.13595v4 — **(C2 framing).**
- **Robustness of Multilingual Text Embedding Rankings** — arXiv:2605.31142 (rankings are sensitive to
  the aggregation scheme). https://arxiv.org/html/2605.31142v1 — **(C2, ammunition against composites).**
- **Benchmarking Patent Embeddings** — arXiv:2605.24297 (English-only multi-model patent eval incl.
  EmbeddingGemma/BGE-M3/Qwen3). https://arxiv.org/html/2605.24297 — **(C4 model-comparison overlap).**
- *(Already cited, keep)* BordIRlines, XRAG, Linguistic Nepotism/LangSAE, TRAQ/CONFLARE/Conformal-RAG,
  fairness-OT (Buyl & De Bie; Xian et al.), PRIME/Fujii & Ishikawa, SapBERT, PaECTER, ChEBI, RBO
  (Webber 2010), RRF (Cormack 2009), MIRACL/MMTEB/NeuCLIR/MTEB.

## What WOULD make the weakest contribution clearly novel (hand to dreamer)

The weakest contribution is **C2's composite metric (CLIR-MRS/MRS)** — currently an arbitrary
hand-weighted number. Three routes to make it (or its replacement) clearly novel and unattackable:

1. **Validate the composite against an external criterion.** Pick a ground-truth utility a deployment
   team cares about (e.g. human-judged cross-jurisdiction search satisfaction on a held-out slice, or
   end-to-end downstream RAG answer correctness) and show CLIR-MRS *predicts* it better than mean recall
   does. A composite that demonstrably tracks an external outcome is a contribution; one chosen by fiat
   is not. This is the highest-value addition.
2. **Replace the fixed weights with a principled, validation-free aggregate** — a Borda/rank-aggregation
   over the axes with a bootstrap CI, and *prove the model ordering is invariant* to reasonable weight
   choices. "The winner is stable under any sensible aggregation" is a far stronger claim than "our
   weighting picks a winner."
3. **Turn the separability finding into an operational, falsifiable metric (the level-2 "CERC" idea).**
   The deepest novelty hiding in the data is the **separability-deficit → re-ranker-cannot-help**
   mechanism. Make it operational: at a fixed coverage guarantee (split-conformal), report the
   *cross-lingual reading-cost multiplier* — "to find the foreign twin with 95% probability you must read
   N× more documents than the same-language case." That is a deployment-legible, distribution-free,
   monotone-invariant number with a clear conformal-prediction precedent (Conformal-RAG, SIGIR 2025) to
   cite as machinery but a genuinely new cross-lingual-retrieval-cost question — and it converts the
   weakest contribution (an arbitrary composite) into the strongest (a falsifiable operational guarantee
   that directly supports the "align, don't re-rank" thesis). This is the single best novelty upgrade
   available and it is already half-computed (the separability AUCs exist).
