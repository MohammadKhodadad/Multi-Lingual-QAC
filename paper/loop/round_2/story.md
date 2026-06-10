# Story (round 2)

## Changes since round 1

This round is a *defense-and-upgrade* pass. Round 1's draft was already cohesive
(the cohesion critic called it "unusually disciplined"); the work now is to (a)
close the one real correctness MISMATCH, (b) survive the novelty critic's two
"shoot-it-down" claims, and (c) fold in the reporter's 8 new CPU-only analyses —
some of which are *adversarial to the round-1 framing* and force honest demotions.
Every change below points at a real file under `reports/runs/.../experimental_plots/extra_*`.

1. **B1 FIX — "English is the easiest target" is dead.** Pooled directed-target
   means are fr 0.375 > en 0.367 > zh 0.350 > de 0.309
   (`chem_patents/.../extra_directional_hub/hub_scores.csv` + `summary.json`).
   The copy-ready replacement sentence is in the summary's
   `writer_replacement_sentence`. The §Results "anisotropy" beat is reframed as a
   **hub-and-spoke graph reading**: hardest directed edge en→de (R@10 0.125),
   most asymmetric pair de↔zh (gap +0.234), and *no clean "easiest target"* — with
   the corpus-composition caveat (en 46% / zh 0.4% of docs) folded into the same
   sentence so it closes B1, T2, and N3 (CLIRMatrix) at once.

2. **NOVELTY REFRAME — content-controlled framing + 4 mandatory citations +
   explicit DAPFAM/CLEF-IP boundaries.** C1's "first" is narrowed to "first
   *content-controlled, chemistry-ontology-grounded* cross-lingual patent
   benchmark," positioned explicitly against **CLEF-IP** (multilingual patent
   retrieval exists, but its gold is prior-art relevance, not parallel human
   translation, and it is not chemistry/ontology-grounded) and **DAPFAM**
   (family-level gold — exactly the `publication_number` equivalence we reject).
   C3 is rewritten from "first decomposition" to "we **confirm** the
   alignment-not-translation finding of **2511.19324** and **2507.07543** on a
   content-controlled parallel corpus that removes the translationese/content
   confound those studies could not," adding the chemistry-confusability trap and
   the falsifiable re-ranker bound as the patent-domain delta. C4's budget rule is
   cited to **Oard 1998 / Saleh & Pecina 2020** (QT-vs-DT). Cross-lingual RBO gets
   **Bailey et al. 2017** (query-variation consistency lineage). Plus the
   model-overlap cite **2605.24297** and the aggregation cites **MMTEB /
   2605.31142**. These are mostly mechanical inline edits; together they remove all
   six "you reinvented X" attack surfaces the novelty critic ranked.

3. **DEMOTE CLIR-MRS — aggregation-invariance FAILS, so per-axis dominance is the
   real claim.** The round-1 dreamer's hope ("egemma is rank-1 under any
   aggregation") is **dead**: embeddinggemma's rank range is **[1,4]** across the
   four schemes (CLIR-MRS=1, winner-take-all=1, Borda=3, equal-weight=4)
   (`chem_patents/.../extra_aggregation_invariance/aggregation_ranks.csv`). So
   M6/A7 are reframed: CLIR-MRS is demoted to an **ordering convenience** for table
   rows, and the load-bearing claim becomes **per-axis capability dominance** —
   embeddinggemma leads CLIR@10, separability, and twin-finding *individually*
   (Table 1), so no composite weighting is load-bearing. The aggregation ribbon
   (`cp_fig17`) is presented honestly **as a range**, with the caveat that
   winner-take-all is contaminated (gte-base "wins" mt-robust/lang-parity by
   retrieving almost nothing). This neutralizes the highest novelty risk (N1)
   without over-claiming invariance.

4. **HEADLINE METRIC UPGRADES — integrate XRC (reading-cost multiplier) and RRC
   (re-ranker recoverability ceiling).** These are the round's two clean positive
   contributions and they replace CLIR-MRS's deployment role with
   distribution-free, deployment-legible numbers:
   - **XRC50 = 3.5×** for embeddinggemma (to find a foreign twin at the median you
     scan 3.5× the documents of a same-language copy: D50_same 2 → D50_cross 7),
     vs granite 1.25 (lowest), nomic 11.5, e5 97.75 (catastrophic), gte degenerate
     (`chem_patents/.../extra_xrc_reading_cost/xrc_per_model.csv` + `summary.json`;
     fig `cp_fig15`). **Censoring discipline:** D90/D95 are right-censored (6–16%
     first-foreign ranks = inf), so the *median* XRC50 is the headline and D90/D95
     are lower bounds only — never quoted as point estimates.
   - **RRC@100 = 0.7445** for embeddinggemma (a top-100 re-ranker can recover at
     most 74% of foreign twins), RRC@1000 = 0.9416, **lost@1000 = 5.84%**
     unrecoverable forever; worst non-degenerate e5 loses 37.2%
     (`extra_xrc_reading_cost/rrc_per_model.csv`; fig `cp_fig16`). This turns
     "align, don't re-rank" from a slogan into a **per-model ceiling**: 1 − RRC is
     provably unrecoverable by any re-ranker.

5. **SOFTEN the 2 fragile correlations; keep auc_cross~clir as the single robust
   mechanism.** The drop-the-collapsers check
   (`chem_patents/.../extra_correlation_robustness/`) shows only
   **auc_cross~clir survives** (+0.961 n8 → +0.958 n7, Spearman 0.976 → 0.964).
   The other two are fragile: **mean_overrep~clir FLIPS SIGN** (−0.600 → +0.419)
   and **home_adv~rbo collapses** (−0.846 → +0.186). So the mechanism story now
   leans entirely on auc_cross~clir; the over-representation and home-adv↔RBO
   correlations are kept as *descriptive observations on n=9 with the two collapsed
   encoders driving the spread*, never as load-bearing mechanisms (closes T5).

6. **STRENGTHEN the encoder-bias claim — availability slope is NEGATIVE.** The
   availability-adjusted regression
   (`alias_graph/.../extra_availability_residual/`) gives slope **−0.572**
   (Pearson −0.87, R²=0.76, n=5), with zh (8% availability) carrying the *largest*
   home advantage (+0.475). So the +0.32 mean home advantage is **residual encoder
   bias, NOT an availability artifact** — the "availability explains it away"
   branch does NOT fire. This is the opposite of the round-1 dreamer's guess and it
   *strengthens* C3. Labeled DESCRIPTIVE (n=5 languages), not inferential. NOTE this
   resolves the round-1 §Analysis "availability confound" framing: availability
   shapes *which language's gold is reachable* (42% vs 8–10%, still true and still
   the over-fetch driver), but it does **not** explain away the same-language bias —
   so the headline sharpens from "mostly availability" to "availability sets the
   stage; a residual encoder bias remains."

7. **COHESION FIXES — split the two-benchmark interleave; harmonize RBO ceilings.**
   The §Analysis paragraph that fused alias numbers (0.63–0.82 vs 0.35–0.47; 42% vs
   8–10%) with chem-patents figures (`cp_fig09`/`cp_fig10`, 49×/60%) under one
   alias-only footnote is split into two benchmark-labelled sentences with two
   footnotes, now led by the **A6 joint cut** ("the modal failure is a same-language
   sibling," 114/257 = 44.4%, `alias_graph/.../extra_joint_failure/`) so the split
   paragraph gains a unifying thesis. The abstract's single RBO ceiling (0.39) is
   harmonized to the body's two ceilings: "(cross-lingual RBO ceiling 0.39 on the
   alias-graph benchmark, 0.19 on the cross-lingual benchmark)". B2 attribution
   fixed ("ceiling across models," not "the best model"). Both `\todo` markers moved
   to LaTeX comments. cp_fig11 caption fixed to describe what the bars show with
   +0.96 as a text statistic. Teaser-vs-leaderboard reorder signposted.

8. **NEW grounded beats added to Analysis/Deployment.** The universal-blind 12%
   orphan is now *earned*: 16/132 = 12%, **14/16 are STRUCTURE questions**
   (`alias_graph/.../extra_joint_failure/universal_blind_profile.csv`), tying the
   structure-trap beat to the oracle-residual. Confusion severity gets a two-level
   split (sibling win-rate 18.1% vs parent 6.2%, 2.9× ratio;
   `alias_graph/.../extra_confusion_severity/`) — scoped honestly as a two-level
   split, with the graded ChEBI hop-distance law marked PENDING-EVAL.

**Items treated as DONE (in `needs_eval.md`, never flagged as missing):**
W4-formula-injection (causal formula rescue), CLIRMRS-external-validation (the
only thing that would *validate* the composite — until then it stays demoted),
XRC-conformal-M2 (guarantee upgrade on top of empirical XRC), CCI-hop-distance-law
(graded confusion law), equivalence-audit-spotcheck (parallel-gold equivalence).

---

## Thesis (industrial framing)

> **A chemistry-patent search team must deploy exactly one multilingual embedding
> model, and the number their dashboard shows — average Recall@10 — is the one
> number that hides the failure they will ship.** Average recall is inflated by
> *same-language* hits; the moment a German chemist's query must reach an English
> or Chinese patent (the normal case in a patent family), recall collapses, and no
> two language versions of the same question return the same documents. We make
> this collapse measurable on two content-controlled patent-grounded benchmarks,
> quantify *what cross-linguality costs* (you scan ~3.5× the documents to find a
> foreign twin; a top-100 re-ranker recovers at most ~74% of them), name the model
> that survives (`embeddinggemma`), and show the fix is **representation alignment
> at indexing time, not a monolingual re-ranker at query time** — because foreign
> gold is *under-scored*, not merely mis-ordered.

Optional framing overlay (recommended, from dreamer W3) — **"the cross-lingual tax
has two line-items"**: cross-lingual retrieval pays a **reading-cost tax** (XRC,
the depth multiplier, measured on the cross-lingual benchmark) and a
**confusability tax** (the look-alike that out-ranks the gold, measured on the
alias-graph benchmark). Each benchmark measures one line-item of the same bill.
This gives the two interleaved benchmarks a single spine and answers the cohesion
seam at the story level. Use it as the connective tissue, not as a load-bearing
claim.

Three industrial pillars, each grounded in a file:

1. **The collapse is real, large, and costly.** Best cross-lingual Recall@10 is
   **0.50** (`embeddinggemma`) against a same-language home advantage up to **+0.55**
   for the most biased model (`chem_patents/.../EXECUTIVE_SUMMARY.md`, headline 1).
   Cross-linguality has a quantified price: **XRC50 = 3.5×** more documents to read
   to find a foreign twin (`extra_xrc_reading_cost/summary.json`). Spanish — **34
   queries, 0 Spanish gold** — is the built-in no-home stress test.
2. **The collapse is mis-rankable two ways, and both are deployment bugs.** Same
   question in five languages returns *different* patents (cross-lingual RBO ceiling
   **0.39** alias-graph / **0.19** cross-lingual), and a chemically-confusable wrong
   compound out-ranks every gold patent on **14%–78%** of queries
   (`alias_graph/.../EXECUTIVE_SUMMARY.md`). The modal confusion is a
   **same-language sibling** (44.4%) — language bias and chemical confusability
   compound (`extra_joint_failure/summary.json`).
3. **The cause is separable representations, so the fix is alignment.**
   r(cross-language AUC, CLIR@10) = **+0.96** and *robust to dropping the two
   collapsed encoders* (+0.958 on n=7, `extra_correlation_robustness/`): foreign
   gold is *under-scored*, not mis-ordered. A re-ranker reads a list it cannot
   repair — and we bound exactly how much it can repair: **RRC@100 ≤ 0.74**,
   **lost@1000 = 5.84%** unrecoverable (`extra_xrc_reading_cost/rrc_per_model.csv`).

---

## Contributions (numbered, each with a one-line novelty claim)

**C1. Two content-controlled, patent-grounded multilingual chemistry-retrieval
benchmarks built only from human-translated patent text.**
- *What:* (a) the **alias-graph** benchmark — 132 questions, 24 ChEBI compounds × 5
  languages, two co-equal relevance lenses (concept ~109 gold/query; per-publication
  ~2.4 gold/query), each compound shipping a graph of chemically-confusable
  neighbours (siblings/parents) as hard negatives; (b) the **cross-lingual (CLIR)**
  benchmark — 137 questions in en/de/es/fr/zh (57 human-original + 80
  MT-cross-lingual) over a **23,487-doc** shared `multilingual_GP` haystack, with
  Spanish as a pure no-home query language.
- *Novelty claim:* **The first cross-lingual, *content-controlled,
  chemistry-ontology-grounded* patent-retrieval benchmark whose gold is genuinely
  parallel human-translated patents + ChEBI ontology membership (not
  `publication_number` equivalence, not machine-translated documents) and whose
  negatives are chemically-confusable neighbours.** Explicitly bounded against
  **CLEF-IP** (multilingual patent retrieval exists, but its gold is prior-art
  relevance judgments, not parallel translations, and it is not
  chemistry/ontology-grounded) and **DAPFAM** (family-level gold = the
  `publication_number` equivalence we deliberately reject); distinct from
  BordIRlines/XRAG (different-content Wikipedia) and Linguistic Nepotism
  (MT-imported English). *Mandatory cites:* CLEF-IP, DAPFAM, CLIRMatrix, ChEBI.

**C2. A cross-lingual robustness-metric family reported co-equally with recall —
now anchored by two deployment-legible cost metrics (XRC, RRC).**
- *What:* CLIR@k vs MoLIR + home-advantage; directional CLIR matrix +
  hub/asymmetry reading; mate-retrieval; cross-lingual RBO; language-collapse /
  over-representation; separability AUC (same vs cross); **XRC** (cross-lingual
  reading-cost multiplier) and **RRC** (re-ranker recoverability ceiling); CLIR-MRS
  / MRS demoted to a *table-ordering convenience*.
- *Novelty claim:* **A retrieval-side, ranking-level robustness suite for CLIR,
  reported co-equally with recall, whose two headline instruments are new
  deployment-cost metrics: XRC expresses the cross-lingual cost as a
  distribution-free reading-depth multiplier (documents-read, not a unitless
  score), and RRC is the literal per-model upper bound on what any top-K re-ranker
  can recover.** Cross-lingual RBO is positioned as a *cross-lingual* application of
  query-variation consistency (Bailey et al. 2017); separability AUC is standard
  machinery whose novelty is the same-vs-cross decomposition + the re-ranker
  corollary; the directional matrix cites CLIRMatrix. *The composite is explicitly
  NOT claimed as a contribution* — per-axis dominance carries the result.

**C3. A mechanism finding, confirmed on a content-controlled corpus and made
falsifiable: cross-lingual chemistry-retrieval failure is an embedding-level
separability deficit, so the lever is alignment, not re-ranking.**
- *What:* availability sets the stage (English's reachable gold 42% vs 8–10% for
  de/es/zh) but a **residual encoder bias remains** (availability-adjusted slope
  −0.57, n=5: home advantage does NOT track availability across languages); the
  modal confusion is a **same-language sibling** (44.4%); structure-style questions
  are the trap (R@10 0.26, confusion 51%) and a language-independent formula token
  helps (p<0.01); confusion **is** a separability collapse (AUC 0.55 vs 0.70) and
  across models r(cross-language AUC, CLIR@10) = **+0.96**, robust on n=7. Bound:
  **RRC@100 ≤ 0.74**, lost@1000 = 5.84%.
- *Novelty claim:* **We *confirm* the alignment-not-translation finding of
  [2511.19324, 2507.07543] on a content-controlled parallel patent corpus that
  removes the translationese/content confounds those studies could not, and add (i)
  a chemistry-specific same-language-sibling confusability trap and (ii) a
  separability-AUC + RRC test that turns "alignment is the fix" into a falsifiable
  per-model re-ranker bound.** *Mandatory cites:* 2511.19324, 2507.07543.

**C4. A concrete, audited deployment decision with an operating rule.**
- *What:* deploy **`embeddinggemma`** — it leads CLIR@10, separability, twin-finding,
  *and the lowest non-degenerate XRC* **individually** (so the recommendation does
  not rest on a composite weighting); report XRC/RRC/CLIR@10/language-parity *next
  to* recall, never average alone; **do not reflexively ensemble** (untuned RRF
  underperformed the best single model; oracle headroom real — CLIR@10 0.61 / alias
  88% — but needs a score-aware combiner or per-language routing for es/zh);
  **machine-translating the question is safe** (paired diff −0.044, p=0.13), so
  spend human-translation budget on the corpus.
- *Novelty claim:* **An industry-track deployment decision grounded in per-axis
  capability dominance and two new cost metrics (XRC/RRC) rather than mean recall or
  a hand-weighted composite, with a negative ensemble result and a QT-vs-DT budget
  rule (Oard 1998; Saleh & Pecina 2020) re-derived for embedding retrieval over
  patents and quantified as an insignificant null.** *Mandatory cites:* Oard 1998,
  Saleh & Pecina 2020, 2605.24297 (English-only patent-embedding overlap, distinguish
  on cross-lingual).

**C5. (Supporting) A reproducible QAC generation + audit pipeline the benchmarks
rest on.**
- *Novelty claim:* **A patent+ontology-grounded multilingual QAC pipeline whose
  generated queries are human-validated and whose auto-grader is calibrated against
  humans.** Kept as *support*, softened to "a reproducible pipeline (human
  validation summarized in the system description)" until the human-eval numbers are
  under `reports/` (they remain in source slides; LaTeX-comment the `\todo`).

---

## Section map

### Abstract — purpose / beats / figures+numbers / links
- *Purpose:* state the deployment problem, the two content-controlled benchmarks +
  metric family, the cost numbers, the headline model, and the
  alignment-not-re-ranking payoff in ~180 words.
- *Beats:* (1) average recall hides cross-lingual collapse. (2) two
  content-controlled patent-grounded benchmarks + a CLIR robustness suite. (3)
  headline numbers: best CLIR@10 0.50; home advantage up to +0.55; **RBO ceiling
  0.39 (alias-graph) / 0.19 (cross-lingual)** — HARMONIZED, both named; confusion
  14–78%; **XRC50 3.5×**; **RRC@100 ≤ 0.74**. (4) cause is a separability deficit
  (r(cross-language AUC, CLIR@10) +0.96 — spell "cross-language AUC" in full to
  match Analysis); fix is alignment, not re-ranking; `embeddinggemma` survives;
  MT-of-question is safe.
- *Numbers:* both `EXECUTIVE_SUMMARY.md` + `extra_xrc_reading_cost/summary.json`.
- *Links:* sets up the Introduction's contributions list.

### 1 Introduction — purpose / beats / figures+numbers / links
- *Purpose:* motivate the deployment question; enumerate C1–C5.
- *Beats:* (1) team picks ONE model; dashboard shows mean Recall@10. (2) inflated by
  same-language hits — open with Spanish (34 queries, 0 Spanish gold). (3) Two
  failures behind the average: **inconsistency** (RBO ceiling 0.39 alias-graph /
  0.19 cross-lingual — say "the best cross-lingual RBO *any model* achieves,"
  per B2, never "the best model") and **confusion** (14–78%). (4) the cost framing
  in one line (XRC ~3.5×; align-not-rerank). (5) explicit numbered contributions
  C1–C5. (6) one-line spoiler of the deployment rule.
- *Figures:* teaser `cp_fig01_clir_leaderboard.png` (overall vs CLIR vs MoLIR).
- *Links:* each contribution forward-references its home section; Related Work next.

### 2 Related Work — purpose / beats / figures+numbers / links
- *Purpose:* position C1–C4 against the closest prior work; pre-empt "you reinvented
  X." Add the round-1 missing citations *inline, in the same sentence* as each claim.
- *Beats (each = one paragraph, one defended boundary):*
  (1) **Multilingual/CLIR benchmarks** — MIRACL/MMTEB/NeuCLIR/MTEB single averaged
  score vs our co-reported robustness suite. Add **CLIRMatrix** here (directional
  matrix precedent) and **MMTEB/2605.31142** (rankings are aggregation-sensitive —
  the reason we test the ribbon and demote the composite).
  (2) **Cross-lingual RAG / language preference** — BordIRlines, XRAG, Linguistic
  Nepotism confound content with language; our parallel human-translated patents are
  the content-controlled asset (defends C1).
  (3) **Calibration / conformal / fairness-OT** — TRAQ, CONFLARE, Conformal-RAG,
  fairness-OT: they score *calibration*; we score *ranking robustness*. (Conformal-RAG
  is also the machinery cite for the future XRC-conformal upgrade — PENDING-EVAL,
  mention only as horizon.) **Move the future-work disclaimer INTO this paragraph**
  (closes G3).
  (4) **Patent IR & patent-family non-equivalence** — PRIME/Fujii & Ishikawa; WIPO
  "may differ in their claims." Add **CLEF-IP** (the canonical multilingual patent
  benchmark we narrow "first" against) and **DAPFAM** (family-gold design we reject).
  This is the single most dangerous round-1 omission — fix here.
  (5) **Chemistry IR / entity models** — SapBERT, PaECTER (monolingual); contrast
  with our cross-lingual ontology-grounded design. Add **2605.24297** (English-only
  EmbeddingGemma/BGE-M3/Qwen3 patent eval) and distinguish on cross-lingual.
  (6) **When to translate** — **Oard 1998; Saleh & Pecina 2020** (QT-vs-DT) as the
  25-year ancestor of our budget rule; cross-lingual RBO cites **Bailey et al. 2017**.
- *Links:* **END with a forward bridge into Benchmarks** ("Having positioned our
  four contributions against CLEF-IP/DAPFAM and the alignment-not-translation line,
  we now build the benchmarks that deliver C1…"), NOT on the future-work disclaimer
  (closes G3).

### 3 Benchmarks — purpose / beats / figures+numbers / links
- *Purpose:* deliver C1 — the two datasets, construction, and the design choices that
  make them honest (no MT source docs, no `publication_number` gold).
- *Beats:* (1) **shared corpus** — `multilingual_GP`, 23,487 docs, en/de/es/fr/zh.
  **Demote the unavailable corpus-construction stats to one clean sentence** ("Detailed
  corpus-construction statistics are deferred to the system description; all
  load-bearing sizes come from the two benchmark datasets") with the `\todo`
  moved to a `% TODO` LaTeX comment (closes G4). (2) **alias-graph benchmark** —
  multilingual ChEBI alias query, ontology gold, confusable-neighbour hard
  negatives, two lenses (concept ~109 / per-publication ~2.4). (3) **cross-lingual
  benchmark** — generated question, gold = every language version; 57 original + 80
  synthetic; Spanish pure query-side. (4) **honesty by design** — human-translated
  source only; ontology/translation gold not `publication_number`; MT used only for
  *questions* (and §Results shows that is a null). (5) **C5 pipeline** — soften to
  "a reproducible pipeline (human validation summarized in the system description)";
  `\todo` → `% TODO` (closes G4 + C5 orphan).
- *Figures/numbers:* benchmark sizes from the two `EXECUTIVE_SUMMARY.md` headers.
- *Links:* Benchmarks defines the data; Metrics defines what we measure.

### 4 Metrics — purpose / beats / figures+numbers / links
- *Purpose:* deliver C2 — define each robustness axis and why it exists. **Add XRC
  and RRC as first-class metric definitions; demote CLIR-MRS in the same section.**
- *Beats:* (1) **CLIR@k vs MoLIR + home-advantage** — the "average recall hides
  collapse" instrument. (2) **Directional CLIR + hub/asymmetry** — retrieval is not
  symmetric (cite CLIRMatrix). (3) **Mate-retrieval** — can the foreign twin be
  found at all. (4) **Cross-lingual RBO** — consistency across language variants
  (cite Bailey et al. 2017). (5) **Language-collapse / over-representation** — the
  bias mechanism. (6) **Separability AUC** (same vs cross) — under-scoring vs
  mis-ordering, the distinction that decides whether a re-ranker can help.
  (7) **XRC — Cross-lingual Reading-cost Multiplier.** Formula: XRC(m) =
  D_C(cross)/D_C(same), the ratio of retrieval depths at which the gold (resp.
  foreign twin) is found for coverage C. **Report XRC50 (median) as the headline;
  state D90/D95 are right-censored at this sample size and are lower bounds, not
  point estimates** (`extra_xrc_reading_cost/summary.json` `headline_coverage`).
  (8) **RRC — Re-ranker Recoverability Ceiling.** Formula: RRC@K(m) = fraction of
  cross-lingual queries whose foreign twin appears within the top-K candidate pool;
  1 − RRC is provably unrecoverable by any re-ranker. (9) **CLIR-MRS / MRS —
  demoted.** State plainly: "We use CLIR-MRS only to order table rows; the per-axis
  numbers carry the argument (Table 1), so no composite weighting is load-bearing,"
  with the ±50% form given for completeness and the bootstrap CI retained.
- *Figures/numbers:* XRC/RRC formulae; the demotion sentence.
- *Links:* Setup says which models/data; Results applies these metrics.

### 5 Experimental Setup — purpose / beats / figures+numbers / links
- *Purpose:* reproducibility — 9 models, shared 23,487-doc haystack, two lenses /
  original-synthetic split, reproduce commands (`run_all.py` + the `extra_*` scripts
  this round). Add a one-line pointer that the new analyses regenerate via the
  `experimental_plots/extra_*.py` scripts.
- *Links:* hands off to Results.

### 6 Results — purpose / beats / figures+numbers / links
- *Purpose:* deliver the cross-lingual headlines, then the alias-graph headlines,
  then the leaderboards. **Keep the two benchmarks in clearly separated subsections
  (G1/cohesion): §6.1 cross-lingual, §6.2 alias-graph, §6.3 leaderboards** — do not
  interleave their numbers within a paragraph.
- *Beats (§6.1 cross-lingual):*
  (1) collapse — `cp_fig01` (CLIR@10 0.50) + `cp_fig02_home_advantage.png` (+0.55).
  Add the **T1 hedge clause** ("much of which availability shapes, §7, though a
  residual encoder bias remains").
  (2) **anisotropy as a hub-and-spoke graph (B1 FIX)** — `cp_fig03`. Use the
  replacement sentence: hardest directed edge en→de (R@10 0.125), most asymmetric
  pair de↔zh (gap +0.234), *no clean "easiest target"* (fr 0.375 ≈ en 0.367), with
  the corpus-composition caveat (en 46% / zh 0.4%) folded in (closes B1, T2, N3).
  Source: `extra_directional_hub/summary.json` `writer_replacement_sentence`.
  (3) **the cost of cross-linguality (NEW)** — `cp_fig15_xrc_reading_cost.png`:
  XRC50 = 3.5× for embeddinggemma (D50_same 2 → D50_cross 7); report it as the
  finite median, D90/D95 as lower bounds. Source: `extra_xrc_reading_cost/`.
  (4) MT-of-question safe — `cp_fig05_mt_penalty.png` (−0.044, p=0.13; null).
  (5) twins buried + the re-ranker ceiling (NEW) — `cp_fig06_mate_retrieval.png`
  (mate-hit@10 0.38), `cp_fig07_first_foreign_rank.png` (15% never in top-1000),
  and `cp_fig16_rrc_reranker_ceiling.png`: RRC@100 0.7445, RRC@1000 0.9416,
  lost@1000 5.84% (egemma). Source: `extra_xrc_reading_cost/rrc_per_model.csv`.
- *Beats (§6.2 alias-graph):*
  (6) inconsistency — `ag_fig1_cross_lingual_rbo.png` (RBO ceiling 0.39).
  (7) confusion — `ag_fig2_confusion_both_lenses.png` (14–78%),
  `ag_fig5_universal_attractors.png`, plus the **two-level severity split (NEW)**:
  sibling win-rate 18.1% vs parent 6.2% (2.9×), egemma 6.1% vs 1.5%
  (`extra_confusion_severity/severity_split.csv`) — scoped as two-level; graded
  hop-distance law is PENDING-EVAL (CCI-hop-distance-law).
- *Beats (§6.3 leaderboards):* both tables, ordered by CLIR-MRS / MRS, with the
  **signpost sentence** that the order changes from Fig 1 (recall → CLIR-MRS
  reshuffles the middle; closes G6) AND the **aggregation-ribbon caveat (NEW)**:
  `cp_fig17_aggregation_ribbon.png` shows embeddinggemma's rank RANGE is [1,4]
  across schemes, so the recommendation rests on per-axis dominance, not on the
  composite (closes N1). Source: `extra_aggregation_invariance/aggregation_ranks.csv`.
  Radars (`cp_fig14`, `ag_fig10`) get the interpretive clause naming the winning axis
  ("leads on consistency and separability, not raw recall"; closes G5).
- *Links:* Analysis explains *why*.

### 7 Analysis — purpose / beats / figures+numbers / links
- *Purpose:* deliver C3 — the mechanism, the trap, the separability diagnosis,
  cleaned of the fragile correlations and the two-benchmark fusion.
- *Beats:*
  (1) **the joint failure mode leads the section (NEW, A6)** — the modal confusion
  is a **same-language sibling** (114/257 = 44.4%; siblings 79.4% of confusions;
  same-language winners 55.6%), `ag_fig12_joint_failure_modes.png`. This is the
  unifying thesis that lets the next two sentences SPLIT cleanly by benchmark
  (closes G1).
  (2) **availability sets the stage, a residual encoder bias remains (REFRAMED).**
  *Alias-graph sentence + alias footnote:* each language retrieves its own-language
  gold (0.63–0.82) far better than foreign (0.35–0.47), and English's reachable gold
  is 42% vs 8–10% for de/es/zh. *But* the availability-adjusted home-advantage slope
  is **−0.57** (n=5, DESCRIPTIVE), with zh (8% availability) carrying the *largest*
  home advantage — so the +0.32 mean home advantage is a **residual encoder bias,
  not an availability artifact** (`extra_availability_residual/`,
  `ag_fig11_availability_residual.png`). *Cross-lingual sentence + chem footnote:*
  the same bias shows as language collapse — over-fetch up to 49× the base rate and
  same-language noise out-ranks the gold on 60% of queries (`cp_fig09`, `cp_fig10`).
  Two sentences, two footnotes (closes G1, T1, T2, strengthens C3).
  (3) **structure-question trap** — `ag_fig6_question_type_effect.png` (structure
  R@10 0.26 / confusion 51% vs role 0.60/25%; formula token p<0.01).
  (4) **bias↔inconsistency, hedged (T5).** Keep r(cross-language AUC, CLIR@10) =
  +0.96 as the load-bearing mechanism AND note it survives dropping the two collapsed
  encoders (+0.958 on n=7, Spearman 0.964; `extra_correlation_robustness/`).
  **Soften** r(home advantage, RBO) = −0.85 and r(over-representation, CLIR) = −0.60
  to "descriptive on n=9, with the two degenerate encoders driving the spread; the
  home-adv↔RBO and over-rep↔CLIR relationships do not survive dropping them" — do NOT
  present them as mechanisms.
  (5) **separability deficit ⇒ re-ranker bound** — `cp_fig11_separability.png`
  (caption FIXED to describe the bars; +0.96 as text statistic, closes G5) and
  `ag_fig8_confusion_is_separability.png` (AUC 0.55 vs 0.70) ⇒ under-scoring ⇒ the
  RRC ceiling (back-reference §6.1: RRC@100 ≤ 0.74, lost@1000 5.84%). This is the
  crux that sets up Deployment's "align, don't re-rank."
- *Links:* the separability + RRC beat sets up Deployment.

### 8 Deployment Recommendation — purpose / beats / figures+numbers / links
- *Purpose:* deliver C4 — the single decision and the operating rule.
- *Beats:*
  (1) **Deploy `embeddinggemma`** — on **per-axis dominance**, not the composite: it
  leads CLIR@10, separability, twin-finding, and the lowest non-degenerate XRC
  *individually*. State the honest caveat that its composite rank ranges [1,4]
  across aggregations (so we lean on per-axis dominance, not invariance).
  (2) **Report XRC/RRC/CLIR@10/language-parity next to recall**, never average alone
  (Spanish no-home is the reason).
  (3) **Don't reflexively ensemble** — `cp_fig12_ensemble_headroom.png` (RRF
  underperforms; oracle 0.61) + `ag_fig7_ensemble_headroom.png` (oracle 88% vs 76%,
  Chinese largest headroom). **Earn the 12% here AND in Analysis:** the universal-blind
  core is 16/132 = 12%, **14/16 are structure questions** about sibling compounds in
  fr/zh/de (`extra_joint_failure/universal_blind_profile.csv`) — needs chemistry-aware
  help, not more encoders (closes the 12% orphan, ties to the structure-trap beat).
  (4) **Align, do not re-rank** — follows from the separability deficit and is now
  bounded by RRC (1 − RRC unrecoverable; embeddinggemma loses 5.84% forever).
  (5) **Budget rule** — MT the query, human-translate the corpus (from the MT-null +
  C5), framed as a re-derivation of Oard 1998 / Saleh & Pecina 2020 for embedding
  retrieval over patents.
- *Links:* Limitations bounds these; Conclusion restates.

### 9 Limitations — purpose / beats / figures+numbers / links
- *Beats:* (1) **Scale** — 132 + 137 questions, 24 compounds; thin directional cells
  read as indicative (the XRC D90/D95 censoring and the n=5 availability regression
  belong here too: report XRC50 as the robust headline, D90/D95 as right-censored
  lower bounds; the availability slope is DESCRIPTIVE on 5 languages, not
  inferential). (2) **Domain transfer** — chemistry-patent-specific; no general-domain
  MMTEB/MIRACL/NeuCLIR companion yet (PENDING-EVAL, in `needs_eval.md`). (3)
  **Composite validation** — CLIR-MRS is a reporting convenience, not externally
  validated; the per-axis dominance is what we stand on (external validation is
  PENDING-EVAL: CLIRMRS-external-validation). (4) **Judge dependence** — generated
  Q/A + auto-grader rely on an LLM judge; claim-level patent equivalence is not
  lay-annotatable (equivalence-audit-spotcheck PENDING-EVAL). (5) **Severity law** —
  confusion severity is reported as a two-level sibling/parent split; the graded ChEBI
  hop-distance decay law is future work (CCI-hop-distance-law PENDING-EVAL). (6) **5
  languages**, one non-Latin script.
- *Links:* Conclusion.

### 10 Conclusion — purpose / beats / figures+numbers / links
- *Beats:* two content-controlled patent-grounded benchmarks + a CLIR robustness
  suite reveal that average recall hides a cross-lingual collapse that *costs* ~3.5×
  the reading budget (XRC) and that a top-100 re-ranker cannot fully recover (RRC);
  `embeddinggemma` is the survivor on per-axis dominance; the cause is an
  embedding-level separability deficit (r(cross-language AUC, CLIR@10) +0.96, robust);
  so the fix is alignment at index time, not re-ranking; and the budget rule is MT the
  query, human-translate the corpus. Restate the two ceilings verbatim (0.39
  alias-graph / 0.19 cross-lingual). Report robustness next to recall.
- *Links:* closes the loop opened in the Introduction.

---

## Open narrative risks (for critics to watch)

1. **XRC censoring honesty (fact critic).** The headline MUST be the finite XRC50
   (median); D90/D95 are right-censored (6–16% first-foreign ranks = inf) and may
   only be stated as lower bounds, never as point estimates. e5's XRC50 97.75 and
   gte's degenerate inf must be labelled, not buried.
   Source: `extra_xrc_reading_cost/summary.json` `headline_coverage`.

2. **Aggregation-ribbon must be a RANGE, not an invariance claim (novelty critic).**
   embeddinggemma's rank range is [1,4]; the paper must say "rankings are
   aggregation-sensitive; our recommendation rests on per-axis dominance," NOT "egemma
   wins under any aggregation." The winner-take-all column is contaminated (gte "wins"
   by retrieving almost nothing) — annotate it. Source: `extra_aggregation_invariance/`.

3. **Two fragile correlations (fact + clarity critic).** Only auc_cross~clir is
   robust to dropping the collapsers. home_adv~rbo and over-rep~clir FLIP/collapse on
   n=7; they must be presented as descriptive observations, not mechanisms, and never
   copied into the abstract/conclusion as load-bearing. Source:
   `extra_correlation_robustness/correlation_robustness.csv`. (Note the internal JSON
   `key_observation` phrasing is misleading — trust the CSV/prose: BOTH are fragile.)

4. **Availability slope is DESCRIPTIVE n=5 (fact critic).** The negative slope
   strengthens the encoder-bias claim but is on 5 language points; label it
   descriptive, not an inferential test. It must not be over-read as "availability is
   irrelevant" — availability still sets which gold is reachable (42% vs 8–10%); it
   just does not *explain away* the same-language bias. Source:
   `extra_availability_residual/summary.json` `interpretation`.

5. **Two RBO ceilings (fact + cohesion critic).** 0.39 (alias-graph) and 0.19
   (cross-lingual) are different benchmarks/lenses/populations — the abstract now
   names both; never average or conflate. The intro/conclusion must say "the best
   cross-lingual RBO *any model* achieves" (ceiling across models), not "the best
   model" (B2).

6. **Don't-ensemble vs oracle-headroom tension (clarity critic).** Untuned RRF loses;
   a score-aware combiner or router could win — state both in the same paragraph.

7. **MT-of-question is a NULL result (fact critic).** Report as "no significant
   penalty" (−0.044, p=0.13), never as "MT helps."

8. **Corpus / human-eval numbers still in source slides, not `reports/` (fact
   critic).** Keep them descriptive with `\todo` moved to LaTeX comments; restrict
   load-bearing numbers to the two `EXECUTIVE_SUMMARY.md` files + the `extra_*` dirs.

9. **Confusion severity is a TWO-LEVEL split only (fact critic).** Report
   sibling-vs-parent (18.1% vs 6.2%); the graded hop-distance law is PENDING-EVAL —
   do not claim a "decay law" the binary on-disk relation field cannot yield.
   Source: `extra_confusion_severity/` + `needs_eval.md` CCI-hop-distance-law.
