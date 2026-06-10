# Story (round 1)

## Changes since round 0
First round — no previous reporter/critic/dreamer to fold in. The story is built
from the two source PDFs, the two `key_findings` executive summaries (with real
numbers + exact figure paths), the baseline plots, and the `main.tex` skeleton.
`paper/loop/needs_eval.md` is empty, so nothing is yet marked PENDING-EVAL by the
implementer. Where the *paper as designed* needs a number we do not have on disk
(general-domain transfer; the level-2 metrics CTC/CERC/ELI; an equivalence audit),
I flag the beat **PENDING-EVAL** and the paper stands without it.

---

## Thesis (industrial framing)

> **A chemistry-patent search team must deploy exactly one multilingual embedding
> model, and the number their dashboard shows them — average Recall@10 — is the one
> number that hides the failure they will actually ship.** Average recall is inflated
> by *same-language* hits; the moment a German chemist's query has to reach an
> English or Chinese patent (the normal case in a patent family), recall collapses,
> and no two language versions of the same question even return the same documents.
> We make this collapse measurable, name the model that survives it
> (`embeddinggemma`), and show the fix is **representation alignment at indexing
> time, not a monolingual re-ranker at query time.**

Three industrial pillars, each grounded in a file:

1. **The collapse is real and large.** Best cross-lingual Recall@10 is **0.50**
   (`embeddinggemma`) against a same-language home advantage of up to **+0.55** for
   the most biased model (`chem_patents/.../EXECUTIVE_SUMMARY.md`, headline 1).
   Spanish — **34 queries, 0 Spanish gold documents** — is the benchmark's built-in
   "no-home" stress test that average recall cannot see.
2. **The collapse is mis-rankable two ways, and both are deployment bugs.** Same
   question in five languages returns *different* patents (cross-lingual RBO ceiling
   **0.39** alias / **0.19** chem-patents), and a chemically-confusable *wrong*
   compound out-ranks the right one on **14%–78%** of queries
   (`alias_graph/.../EXECUTIVE_SUMMARY.md`, Q1–Q2).
3. **The cause is separable representations, so the fix is alignment.**
   r(cross-language AUC, CLIR@10) = **+0.96**: foreign golds are *under-scored*, not
   merely mis-ordered (chem-patents headline 7). A re-ranker reads a ranked list it
   cannot repair; the lever is at the embedding level.

The deliverables that carry this thesis: **two patent-grounded benchmarks**
(alias-graph + chem-patents CLIR), a **robustness-metric family** (CLIR-MRS / MRS,
cross-lingual RBO, home-advantage, confusion, separability AUC, mate-retrieval,
language-collapse), and a **single deployment recommendation** with a "report these
metrics, not average recall" operating rule.

---

## Contributions (numbered, each with a one-line novelty claim)

**C1. Two patent-grounded multilingual chemistry-retrieval benchmarks built only
from human-translated patent text.**
- *What:* (a) the **alias-graph** benchmark — 132 questions, 24 ChEBI compounds × 5
  languages, two co-equal relevance lenses (concept-level ~109 gold/query;
  per-publication ~2.4 gold/query), each compound shipping a graph of
  chemically-confusable neighbours (siblings/parents) as hard negatives; (b) the
  **chem-patents CLIR** benchmark — 137 questions in en/de/es/fr/zh (57 human-original
  + 80 machine-translated) over a **23,487-doc** shared `multilingual_GP` haystack,
  with Spanish as a pure query-side (no-home) language.
- *Novelty claim:* **The first cross-lingual chemistry-patent retrieval benchmarks
  in which gold relevance is defined by an ontology alias graph with confusable-
  neighbour hard negatives and by genuinely parallel human-translated patents — not
  by `publication_number` equivalence (which the patent-informatics literature shows
  is unsafe) and not by machine-translated documents (translationese as a confound).**
  Distinct from BordIRlines/XRAG (Wikipedia docs that *differ in content*) and from
  Linguistic Nepotism (MT-imported English docs).

**C2. A cross-lingual robustness-metric family that separates capability from
robustness and turns "average recall hides collapse" into measurable axes.**
- *What:* CLIR@k vs. MoLIR (mono-lingual) and the **home-advantage** gap; directional
  CLIR matrix + direction-asymmetry; **mate-retrieval** (does a query's foreign twin
  ever surface, and at what depth); **cross-lingual RBO** (do five language versions
  agree); **language-collapse** / same-language over-representation; **separability
  AUC** (gold > non-gold, same vs. cross language); and the composite **CLIR-MRS /
  MRS** (capability × robustness, ±50%).
- *Novelty claim:* **A retrieval-side, ranking-level robustness suite for CLIR that
  is reported co-equally with recall and is purpose-built to expose same-language
  bias — orthogonal to the calibration/conformal line (XSCE, Conformal-RAG, fairness-
  OT) which scores *calibration*, and to MTEB which reports a single averaged number.**
  Cross-lingual RBO as a *consistency* metric and CLIR-MRS as a *capability×robustness*
  composite are the named instruments.

**C3. A mechanism finding: cross-lingual failure in chemistry retrieval is a
language-bias / separability problem, and the lever is alignment, not re-ranking.**
- *What:* home-advantage + availability confound (English's apparent lead is mostly
  *where the gold lives*: 42% of an English query's gold is in-English vs 8–10% for
  de/es/zh); bias drives inconsistency (r(same-language share, RBO) = **−0.85** to
  **−0.87**); **structure-style questions are the trap** (Recall@10 0.26, confusion
  51%) while role/formula-token questions are safe; confusion **is** a separability
  collapse (AUC 0.55 confused vs 0.70 otherwise; r(AUC, CLIR@10) = **+0.96**).
- *Novelty claim:* **First decomposition of cross-lingual chemistry-retrieval failure
  into an availability confound + a chemistry-specific confusability trap + an
  embedding-level separability deficit, on a content-controlled parallel corpus —
  yielding the falsifiable, deployment-relevant claim that a monolingual re-ranker
  cannot recover under-scored foreign twins.**

**C4. A concrete, audited deployment recommendation with an operating rule.**
- *What:* deploy **`embeddinggemma`** — winner on both benchmarks (chem-patents
  CLIR-MRS **0.71 [0.67,0.77]**; alias MRS **0.991 [0.86,1.00]**); report **CLIR@10
  + language-parity + MRS next to recall**, never average recall alone; **do not
  reflexively ensemble** (untuned RRF underperformed the best single model; oracle
  headroom is real — CLIR@10 0.61 / alias 88% — but needs a score-aware combiner or
  per-language routing for the homeless es/zh); **machine-translating the *question*
  is safe** (paired human−MT difference −0.044, p=0.13) so spend human-translation
  budget on source patents, not generated Q/A.
- *Novelty claim:* **An industry-track deployment decision grounded in a robustness
  composite rather than mean recall, with a negative result (don't ensemble naively)
  and a budget-allocation rule (MT the query, human-translate the corpus) that
  follow directly from the metrics.**

**C5. (Supporting) A reproducible QAC generation + audit pipeline that the
benchmarks rest on.**
- *What:* per-(document, language, mode) generation of 3 candidates scored on
  Faithfulness (/15) + Quality (/25) and emitted best-first; two modes (technical /
  semantic) × four sampling strategies (`random_any/existing/missing/all`); validated
  against a human annotator (97/100 reviewed, mean **8.33/10**, 0 rejected) with the
  LLM auto-grader **+4.3pp stricter** than humans (system PDF, slides 7–17).
- *Novelty claim:* **A patent+ontology-grounded multilingual QAC pipeline whose
  generated queries are human-validated and whose auto-grader is calibrated against
  humans — the data-quality backbone an industry reviewer demands before trusting a
  new benchmark.** (Support, not headline.)

---

## Section map

### Abstract — purpose / beats / figures+numbers / links
- *Purpose (1 sentence):* state the deployment problem, the two benchmarks + metric
  family, the headline model, and the alignment-not-re-ranking payoff in 150–200 words.
- *Beats:* (1) multilingual chemistry-patent retrieval; average recall hides
  cross-lingual collapse. (2) two patent-grounded benchmarks + a CLIR robustness
  suite. (3) headline numbers: best CLIR@10 0.50, home-advantage up to +0.55, RBO
  ceiling 0.39/0.19, confusion 14–78%. (4) `embeddinggemma` wins both
  (CLIR-MRS 0.71); fix is alignment not re-ranking; MT-of-question is safe.
- *Figures/numbers:* none inline; numbers from both `EXECUTIVE_SUMMARY.md`.
- *Links:* sets up the Introduction's contributions list.

### 1 Introduction — purpose / beats / figures+numbers / links
- *Purpose:* motivate the industrial deployment question and enumerate C1–C5.
- *Beats:* (1) a patent search team picks ONE multilingual model; their dashboard
  shows mean Recall@10. (2) That number is inflated by same-language hits — open with
  Spanish: 34 queries, 0 Spanish gold (chem-patents summary, "defining property").
  (3) Two failures behind the average: inconsistency (RBO 0.39/0.19) and confusion
  (14–78%). (4) We build two benchmarks + a robustness suite to measure it and name
  the survivor. (5) Explicit numbered contributions list = C1–C5. (6) One-line
  spoiler of the deployment rule (report MRS, align don't re-rank).
- *Figures/numbers:* teaser uses `chem_patents/key_findings/figures/fig01_clir_leaderboard.png`
  (overall vs CLIR vs MoLIR recall) as Figure 1; numbers from both summaries.
- *Links:* each contribution forward-references its home section; Related Work next
  defends the novelty claims.

### 2 Related Work — purpose / beats / figures+numbers / links
- *Purpose:* position C1–C4 against the closest prior work and pre-empt "you
  reinvented X" reviews; mine the level-2 PDF's novelty audits.
- *Beats (each = one paragraph, one defended boundary):*
  (1) **Multilingual/CLIR retrieval & benchmarks** — MIRACL/MMTEB/NeuCLIR, and our
  *retrieval-side, ranking-level* robustness suite vs. MTEB's single averaged number
  (level-2 Part 2D contamination context; MTEB maintainers `arXiv:2506.21182`).
  (2) **Cross-lingual RAG / language preference** — BordIRlines (`arXiv:2410.01171`),
  XRAG (Findings EMNLP 2025), Linguistic Nepotism (`arXiv:2509.13930`): they confound
  content with language (different-content Wikipedia, or MT-imported English). Our
  parallel human-translated patents are the content-controlled asset; this is the
  defense for C1 (level-2 Part 1D / Part 0 RAG paragraph).
  (3) **Calibration / conformal / fairness-OT in IR** — XSCE/CalX, TRAQ
  (`arXiv:2307.04642`), CONFLARE (`arXiv:2404.04287`), Conformal-RAG (SIGIR 2025,
  `arXiv:2506.20978`), fairness-OT (Xian et al. ICML 2023 `arXiv:2211.01528`; Buyl &
  De Bie NeurIPS 2022 `arXiv:2202.03814`). Boundary: those score *calibration*; our
  suite scores *ranking robustness* (RBO/CLIR-MRS/confusion). Defends C2.
  (4) **Patent IR & patent-family non-equivalence** — PRIME (Fujii & Ishikawa,
  `arXiv:cs/0206035`), WIPO guidance that family docs "may differ in their claims":
  the reason we do *not* use `publication_number`-equivalence for gold. Defends C1.
  (5) **Chemistry IR / entity models** — SapBERT, PatentTEB/PAECTER (largely
  monolingual): contrast with our cross-lingual, ontology-graph-grounded design.
  (6) **CLIR performance prediction & "when to translate"** — Kishida (2008), Lee et
  al. (2010) as the pre-neural ancestors of our routing/oracle discussion
  (level-2 Part 2C). Defends the C4 routing claim's framing.
- *Figures/numbers:* none.
- *Links:* every boundary maps to the contribution it protects; Benchmarks delivers C1.
- *PENDING-EVAL:* the level-2 metrics (CTC/CERC/ELI/CalX-OT) are **future work**, not
  claimed here — Related Work mentions them only as the research-direction horizon, so
  reviewers see we know the calibration line without us claiming unbuilt results.

### 3 Benchmarks — purpose / beats / figures+numbers / links
- *Purpose:* deliver C1 — the two datasets, their construction, and the design choices
  that make them honest (no MT source docs, no `publication_number` gold).
- *Beats:*
  (1) **Corpus** — Google Patents (10,628 docs / 23,487 rows, en/fr/es/de,
  1999–2025) + EPO bulk (3,773 docs, en/fr/de) + JRC; 14,401 unique docs after dedup,
  0 cross-source duplicates; A61 pharma dominates with the C-class chemistry tail
  (system PDF slides 3–5). The retrieval haystack is the **23,487-row
  `multilingual_GP`** shared corpus.
  (2) **Alias-graph benchmark** — query = multilingual ChEBI alias set (Tin: tin/Sn,
  Zinn, estaño, étain); gold = patents about the concept; hard negatives = up to 10
  confusable-neighbour docs (sibling/parent/tautomer). Two lenses: concept (~109
  gold) vs per-publication (~2.4 gold). 132 Q, 24 compounds × 5 langs (system PDF
  slide 16; alias summary header).
  (3) **Chem-patents CLIR benchmark** — query = a generated question, gold = every
  language version of its source patent; 137 Q (57 human-original same-language-gold +
  80 MT cross-lingual-only); Spanish pure query-side (chem-patents summary "defining
  property").
  (4) **Honesty design** — human-translated source text only (no MT docs); gold by
  ontology/translation, not `publication_number` (defends against the patent-family
  objection from the level-2 Part 0 / Part 2A audit).
- *Figures/numbers:* system PDF "Data sources at a glance" (counts), "Parallel text
  coverage" (GP/EPO/JRC matrices), "IPC distribution". *Note for writer:* these slide
  numbers live in the source PDF, not yet in `reports/`; the benchmark statistics that
  drive results are in the two `EXECUTIVE_SUMMARY.md` headers (137, 23,487, 132, 24).
- *Links:* Benchmarks defines the data; Metrics defines what we measure on it.

### 4 Metrics — purpose / beats / figures+numbers / links
- *Purpose:* deliver C2 — define each robustness axis and *why* it exists, so Results
  can report them co-equally with recall.
- *Beats:* (1) **CLIR@k vs MoLIR + home-advantage** = the "average recall hides
  collapse" instrument. (2) **Directional CLIR + asymmetry** = retrieval is not a
  symmetric similarity. (3) **Mate-retrieval** (mate-hit@k, mate-MRR, first-foreign
  rank) = can the foreign twin be found at all. (4) **Cross-lingual RBO** = do five
  language versions agree (consistency). (5) **Language-collapse / over-representation**
  = the bias mechanism. (6) **Separability AUC** (same vs cross) = under-scoring, not
  mis-ordering. (7) **CLIR-MRS / MRS** = capability {accuracy, CLIR, separability} ×
  (0.5 + 0.5·robustness {consistency, MT-robust, language-parity}); definition verbatim
  from chem-patents summary "Verdict" block.
- *Figures/numbers:* the CLIR-MRS formula (chem-patents summary); the MRS five-axis
  definition (alias summary "Headline numbers" footnote).
- *Links:* Setup says which models/data; Results applies these metrics.

### 5 Experimental Setup — purpose / beats / figures+numbers / links
- *Purpose:* reproducibility — models, data, protocol.
- *Beats:* (1) **9 multilingual embedding models**: embeddinggemma, bge-m3,
  granite-278m, nomic-v2-moe, qwen3-0.6B, LaBSE, SapBERT, e5-large-instruct, gte-base
  (both `EXECUTIVE_SUMMARY.md` leaderboards). (2) Shared haystack (`multilingual_GP`,
  23,487 docs) so both benchmarks retrieve against one corpus; queries/qrels from the
  benchmark datasets. (3) Two relevance lenses (alias) and original/synthetic split
  (chem-patents). (4) Reproduce commands:
  `reports/runs/chem_patents/experimental_codes/run_all.py` and
  `reports/runs/alias_graph/experimental_codes/`.
- *Figures/numbers:* none new; model list from both leaderboards.
- *Links:* hands off to Results.

### 6 Results — purpose / beats / figures+numbers / links
- *Purpose:* deliver the seven chem-patents headlines + the two alias headlines, each
  with its figure and number, then the two leaderboards.
- *Beats (chem-patents):*
  (1) collapse — `fig01_clir_leaderboard.png` (CLIR@10 0.50) + `fig02_home_advantage.png`
  (up to +0.55).
  (2) anisotropy — `fig03_directional_clir_matrix.png`, `fig04_clir_direction_asymmetry.png`
  (en→de 0.12; de↔zh +0.23).
  (3) MT-of-question safe — `fig05_mt_penalty.png` (−0.044, p=0.13).
  (4) twins buried — `fig06_mate_retrieval.png` (mate-hit@10 0.38), `fig07_first_foreign_rank.png`
  (15% never in top-1000; median first-foreign rank 5 for the winner).
- *Beats (alias):*
  (5) inconsistency — `alias_graph/.../fig1_cross_lingual_rbo.png` (RBO ceiling 0.39).
  (6) confusion — `fig2_confusion_both_lenses.png` (14–78%), `fig5_universal_attractors.png`
  (polypeptide/methyl/ethene/hydroxide/dioxygen).
- *Beats (leaderboards):* chem-patents CLIR-MRS table (`fig13_clir_mrs_leaderboard.png`,
  `fig14_robustness_radar.png`) — embeddinggemma 0.71; alias MRS
  (`fig9_robustness_leaderboard.png`, `fig10_robustness_radar.png`) — embeddinggemma 0.991.
- *Figures/numbers:* exact paths above; numbers from both summaries' headline tables.
- *Links:* Analysis explains *why* these happen.

### 7 Analysis — purpose / beats / figures+numbers / links
- *Purpose:* deliver C3 — the mechanism, the trap, and the separability diagnosis.
- *Beats:* (1) **availability confound** — English's lead is where gold lives (42% vs
  8–10%); `chem_patents/.../fig09_language_collapse.png` + `fig10_distractor_language.png`
  (own-language over-fetch up to 49×; same-language noise out-ranks gold on 60%).
  (2) **bias drives inconsistency** — `fig08_consistency_vs_bias.png` (r=−0.85) and
  `alias_graph/.../fig4_bias_drives_inconsistency.png` (r=−0.87).
  (3) **structure-question trap** — `alias_graph/.../fig6_question_type_effect.png`
  (structure R@10 0.26 / confusion 51% vs role 0.60/25%; formula token helps p<0.01).
  (4) **separability** — `chem_patents/.../fig11_separability.png` (r(AUC,CLIR@10)=+0.96)
  and `alias_graph/.../fig8_confusion_is_separability.png` (AUC 0.55 vs 0.70) ⇒ under-
  scoring ⇒ re-ranker cannot fix.
- *Figures/numbers:* exact paths above.
- *Links:* the separability beat sets up Deployment's "alignment not re-ranking".

### 8 Deployment Recommendation — purpose / beats / figures+numbers / links
- *Purpose:* deliver C4 — the single decision and the operating rule.
- *Beats:* (1) **Deploy `embeddinggemma`** (wins both composites; best twin-finder).
  (2) **Report MRS/CLIR@10/language-parity next to recall**, never average alone.
  (3) **Don't reflexively ensemble** — `chem_patents/.../fig12_ensemble_headroom.png`
  (RRF underperforms best single; oracle 0.61) + `alias_graph/.../fig7_ensemble_headroom.png`
  (oracle 88% vs 76%, Chinese largest headroom, ~12% universal-blind core); use a
  score-aware combiner or per-language routing for es/zh.
  (4) **Alignment, not re-ranking** (follows from the separability beat).
  (5) **Budget rule** — MT the query, human-translate the corpus (from C5 + MT-penalty).
- *Figures/numbers:* exact paths above; both leaderboard tables.
- *Links:* Limitations bounds these claims; Conclusion restates.

### 9 Limitations — purpose / beats / figures+numbers / links
- *Purpose:* required ACL section; honest scope.
- *Beats:* (1) **Scale** — 132 + 137 questions, 24 compounds; some language-pair cells
  thin (de column 6–11/cell per system PDF "Scores by language pair"). (2) **Domain
  transfer** — results are chemistry-patent-specific; **PENDING-EVAL**: no general-domain
  MMTEB/MIRACL/NeuCLIR companion number yet (level-2 honesty ledger item 5). (3)
  **Judge dependence** — generated Q/A and auto-grader rely on an LLM judge; human
  validation is 97 items (system PDF) — claim-equivalence is not lay-annotatable
  (level-2 ledger item 4). (4) **5 languages** (en/de/es/fr/zh); not low-resource
  scripts beyond zh. (5) The level-2 metrics (CTC/CERC/ELI) and an equivalence audit
  are **future work**, not claimed.
- *Figures/numbers:* system PDF "Scores by language pair" caption (thin de column).
- *Links:* Conclusion.

### 10 Conclusion — purpose / beats / figures+numbers / links
- *Purpose:* restate thesis + payoff in 4–5 sentences.
- *Beats:* two patent-grounded benchmarks + a CLIR robustness suite reveal that
  average recall hides cross-lingual collapse; `embeddinggemma` is the survivor; the
  fix is alignment at index time, not re-ranking at query time; MT the query, human-
  translate the corpus. One forward pointer to the level-2 research directions.
- *Links:* closes the loop opened in the Introduction.

---

## Open narrative risks (for critics to watch)

1. **System-PDF numbers not yet in `reports/`.** The corpus counts (14,401;
   GP/EPO/JRC matrices; IPC chart) and human-eval numbers (8.33/10; auto-grader +4.3pp)
   live in the *source slides*, not under `reports/`. The hard rule is "every number
   traces to a file." **Risk:** the Benchmarks/C5 beats cite the PDF, not `reports/`.
   The writer must either (a) restrict load-bearing numbers to the two
   `EXECUTIVE_SUMMARY.md` files (137, 23,487, 132, 24, all metric values) and treat
   corpus/human-eval figures as descriptive context with a PDF citation, or (b)
   request the implementer to dump these to `reports/`. Flag for the **fact critic**.
2. **"Synthetic benchmark" credibility.** Queries are LLM-generated. The novelty/
   reviewer risk is "why trust a synthetic benchmark?" — C5 (human validation, 0
   rejected, auto-grader calibrated) is the answer and must not be buried. If the
   **novelty critic** judges C5 too thin, escalate to a small expert re-annotation as
   PENDING-EVAL rather than overclaiming.
3. **General-domain transfer gap.** The level-2 honesty ledger (item 5) is explicit:
   without an MMTEB/MIRACL/NeuCLIR companion, an industry reviewer may read this as
   domain engineering. We frame chemistry-patent specificity as the *contribution*
   (a real deployment niche), but the **methods/novelty critics** should decide
   whether a transfer number must be added to `needs_eval.md`.
4. **Metric-novelty overlap with the calibration/conformal line.** CLIR-MRS/RBO are
   *ranking* metrics, not calibration metrics — but a reviewer who knows XSCE/
   Conformal-RAG/fairness-OT may conflate them. Related Work beat (3) must draw the
   boundary in the *first* sentence of each comparison (the level-2 audits warn the
   densest precedent traffic is in calibration/conformal/fairness machinery). Flag for
   the **novelty critic**.
5. **Two RBO ceilings (0.39 alias vs 0.19 chem-patents).** Different benchmarks, lenses
   and query populations — the writer must not average or conflate them; cite each to
   its own benchmark. Flag for the **fact critic**.
6. **"Don't ensemble" vs "oracle headroom is real" tension.** These coexist
   (untuned RRF loses; a score-aware/router could win) and must be stated together or
   the Deployment section reads as self-contradictory. Flag for the **clarity critic**.
7. **MT-of-question result is a *null* result (p=0.13).** It supports the budget rule
   but must be reported as "no significant penalty," never as "MT improves" — an easy
   over-read. Flag for the **fact critic**.
