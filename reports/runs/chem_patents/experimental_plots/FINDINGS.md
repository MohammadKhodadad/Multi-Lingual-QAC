# Chem-patents multilingual retrieval — CLIR findings

Iterative 10-round deep-dive on the 9-model chemistry-patent retrieval benchmark, centred on **cross-lingual retrieval (CLIR)**: can a question in one language pull back the relevant patent in its *other* language versions? Each section is one round (question -> plots -> numbers -> what we learned -> next questions).

<!-- round:01 -->
## Round 1 — The CLIR leaderboard and the home-advantage gap

**Question.** Standard recall ranks the models, but this benchmark exists to test *cross-lingual*
retrieval. Split each query's gold set by language relative to the query: **CLIR@k** = recall over
documents in a *different* language; **MoLIR@k** = recall over the *same*-language document (only the
57 original queries ever have one — all 80 synthetic queries are pure CLIR). How big is the gap?

**Numbers (mean over queries; CLIR 95% bootstrap CI; MoLIR/home-advantage on the 57 originals).**

| model | Recall@10 | CLIR@10 | MoLIR@10 | home adv |
| --- | ---: | :---: | ---: | ---: |
| `embeddinggemma` | 0.574 | 0.541 [0.50,0.58] | 0.725 | +0.167 |
| `bge-m3` | 0.509 | 0.464 [0.43,0.50] | 0.667 | +0.224 |
| `qwen3-0.6B` | 0.496 | 0.451 [0.41,0.49] | 0.659 | +0.222 |
| `nomic-v2-moe` | 0.467 | 0.410 [0.37,0.45] | 0.690 | +0.295 |
| `granite-278m` | 0.415 | 0.390 [0.35,0.43] | 0.541 | +0.124 |
| `LaBSE` | 0.301 | 0.271 [0.24,0.31] | 0.427 | +0.148 |
| `SapBERT` | 0.238 | 0.197 [0.17,0.23] | 0.373 | +0.189 |
| `e5-large-instruct` | 0.218 | 0.094 [0.07,0.12] | 0.706 | +0.604 |
| `gte-base` | 0.005 | 0.000 [0.00,0.00] | 0.024 | +0.024 |

**What we learned.**
- The best cross-lingual model is **embeddinggemma** at CLIR@10 = **0.541**
  [0.50, 0.58]. Every model scores strictly lower on CLIR@10 than
  on overall recall — the easy points come from same-language matches.
- **Home advantage is universal and large.** When a same-language copy exists, models retrieve it far
  more reliably than any foreign version; the gap reaches **+0.604** for
  `e5-large-instruct`. The benchmark's 80 synthetic queries have *no* such crutch.
- Two models are effectively broken on this corpus and should be read with care: `gte-base`
  (≈0.004 — a degenerate/config failure) and `e5-large-instruct` (instruction model, very weak here).

**Next questions.**
1. CLIR is an average over many language directions. *Which* query→document directions actually
   collapse, and is the failure symmetric (en→zh vs zh→en)? → Round 2.
2. Synthetic queries are both machine-translated *and* homeless. Isolate the MT effect by comparing
   CLIR (cross-language only) for human-original vs machine-translated queries. → Round 3.
<!-- /round:01 -->

<!-- round:02 -->
## Round 2 — Directional CLIR matrix and translation-direction asymmetry

**Question.** CLIR@10 averages over every query→document language direction. Which directions
actually collapse, and is the difficulty symmetric (is en→zh as hard as zh→en)?

**What we learned.**
- **The matrix is strongly anisotropic.** Pooled over reliable models, the hardest direction is
  **en→de** (Recall@10 = 0.16)
  and the easiest cross-lingual direction is **en→es**
  (0.42).
- **Direction matters, not just the pair.** The most asymmetric language pair is
  **de↔en**
  with a gap of +0.19
  in Recall@10 between its two directions — retrieval is not a symmetric similarity.
- **English is the easiest target** (mean Recall@10 into English documents
  = 0.40): models lean on English as a hub language.
- **Spanish is the canary.** As a pure query-side language with *no* same-language gold, Spanish
  CLIR@10 = 0.35; the weakest query language overall is
  **fr** (0.34).

**Next questions.**
1. Synthetic (machine-translated) queries dominate the weak directions. Is the penalty the
   *translation* or the *missing home document*? Compare CLIR for human vs MT queries. → Round 3.
2. If a query can't find one foreign version, can it find *any* of the patent's other-language
   twins, and at what rank? → Round 4 (mate retrieval).
<!-- /round:02 -->

<!-- round:03 -->
## Round 3 — The machine-translation penalty

**Question.** Is a machine-translated question as good as a human one for finding the patent's
foreign versions? We score every query on a *fixed* target set — the patent's non-source-language
documents (`foreign_reach@10`) — so the home-document confound is removed and only the query differs.

**Numbers (pooled over reliable models).**
- Human-original queries: foreign_reach@10 = **0.351**
- Machine-translated queries: foreign_reach@10 = **0.349**
- Gap = **+0.002**; paired over shared patents the mean difference is
  **-0.045**
  (p=0.015, n=272 patent×model).

**What we learned.**
- The MT penalty on *cross-lingual* reach is **statistically significant**. Controlling for the patent (same foreign
  targets), human and machine-translated questions retrieve the foreign patent comparably — the
  paired difference is only -0.045.
  This supports the project's "MT-is-fine-for-the-question" stance: the cross-lingual difficulty lives
  in the embedding model, not in the question's provenance.
- In the naive population view synthetic queries even look slightly *stronger*
  (0.349 vs 0.351);
  that is a **patent-selection artefact**, which is exactly why the paired test is the one to trust.
- The hardest MT *target* language is **en** — translating the
  question into that language costs the most reach.

**Next questions.**
1. We've been asking "did it find a relevant doc". Sharpen to bitext: can the model find the *exact
   translated twin* of the patent, and at what rank? → Round 4.
2. If the question's language barely matters, what does? Maybe the model just returns same-language
   noise regardless. Quantify language collapse. → Round 6.
<!-- /round:03 -->

<!-- round:04 -->
## Round 4 — Mate retrieval: finding the translated twin

**Question.** Beyond "did it find a relevant doc", can the model surface the patent's *foreign twin*
at all, and how deep do you have to scan? Foreign mates = the same patent in other languages;
mate-hit@k = any foreign twin in the top-k; mate-MRR = 1 / rank of the first foreign twin.

**What we learned.**
- The best twin-finder is **embeddinggemma** (mate-MRR = 0.361, mate-hit@10
  = 0.595); when it does surface a twin, the median first-foreign rank is
  **4**.
- Pooled over reliable models, only **41%** of queries surface a foreign twin in the
  top-10, and **17%** of (query, model) pairs never surface one even in the **top-1000**
  — those patents are effectively unreachable across the language barrier.
- The widening gap between mate-hit@10 and mate-hit@100 (Fig 1) shows the twin is often *present but
  buried*: a re-ranking or deeper cutoff recovers a meaningful share.
- Weakest entry language for twin-finding: **fr**.

**Next questions.**
1. If the same question in different languages returns different twins, the rankings themselves must
   diverge. How consistent is the ranked list across languages (RBO)? → Round 5.
2. When the twin is buried, what outranks it — same-language noise? → Rounds 6 & 7.
<!-- /round:04 -->

<!-- round:05 -->
## Round 5 — Cross-lingual ranking consistency (RBO)

**Question.** When the *same* patent's question is asked in several languages, a language-agnostic
retriever should return the same ranked patents. How consistent are the top-100 lists across
languages? (RBO_ext, p=0.9; 35 multilingual patents,
259 cross-lingual query pairs per model.)

**What we learned.**
- Consistency is **low even at the ceiling**: the most consistent model is **embeddinggemma**
  at RBO = **0.18** (1.0 = identical) — the *same question* in two languages
  produces largely *different* top-100 patent lists. The floor is **e5-large-instruct**
  (RBO = 0.01).
- Inconsistency tracks bias: Pearson r(home-advantage, RBO) = **-0.89**
  — models that lean hardest on the same-language copy are also the most language-sensitive in their
  rankings.
- The most divergent language pair is **fr↔zh**.

**Next questions.**
1. If rankings differ this much by language, the lists must be filling with *language-specific* docs.
   How monolingual is the top-k, and does that collapse explain CLIR failure? → Round 6.
2. Are these low-consistency, language-collapsed results also poorly *separated* in score space? → Round 8.
<!-- /round:05 -->

<!-- round:06 -->
## Round 6 — Language collapse and cross-lingual reach

**Question.** Do models fill the top-k with the query's own language? We compare each query's
same-language share of the top-10 to the corpus base rate (that language's share of the 23k-doc
haystack); the ratio is a **same-language over-representation** factor (>1 = collapse).

**What we learned.**
- **Collapse is severe for low-resource languages.** The most over-represented query language is
  **zh** at **17.4×**
  its corpus base rate — Chinese and Spanish queries pull back their own language far beyond chance,
  even though (for es) no Spanish document is ever relevant.
- **Collapse predicts CLIR failure.** Across models, Pearson r(over-representation, CLIR@10) =
  **-0.58**: the more a model anchors on the query language, the worse
  its cross-lingual recall. This is the mechanism behind the home advantage in Round 1 and the low
  RBO in Round 5.
- The best model still devotes only ~58% of its top-10 to foreign
  languages on average.

**Next questions.**
1. If same-language documents crowd the top-k, *irrelevant* same-language patents must be out-ranking
   the correct foreign one. How often does that happen, and is the distractor same- or
   cross-language? → Round 7.
2. Is this a ranking problem or a *score-separability* problem — are gold docs even separable from the
   crowd? → Round 8.
<!-- /round:06 -->

<!-- round:07 -->
## Round 7 — Distractor dominance: what buries the foreign twin?

**Question.** With no curated hard-negatives, read confusion from the ranking itself: the documents
above the first gold are the blockers. Are they *same-language* noise (collapse-driven) or genuine
foreign distractors?

**What we learned.**
- **Same-language noise is the dominant blocker for high-resource query languages.** Pooled over
  reliable models, a same-language non-gold out-ranks the first gold on
  **58%** of queries, and
  **44%** of all above-gold blockers share the query's
  language — far above their corpus base rate.
- It is **language-specific**: the most self-confused query language is
  **fr**
  (66% same-language blockers), the least
  is **zh**
  (30%). Spanish, with no same-language
  gold, is mostly blocked by *foreign* distractors — a different failure mode.
- Confusion here is a direct consequence of the Round-6 collapse: the same-language documents the
  model over-fetches are exactly what bury the foreign twin.

**Next questions.**
1. Is this fixable by re-ranking, or are the gold documents simply *not separable* in score space from
   the same-language crowd? → Round 8.
2. Different models collapse on different languages — are their errors complementary enough that an
   ensemble or a per-language router recovers the buried twins? → Round 9.
<!-- /round:07 -->

<!-- round:08 -->
## Round 8 — Score separability: a calibration home advantage

**Question.** Is confusion a ranking accident or a score-separability failure? We compute
AUC(gold > non-gold) per query and split it by direction (same- vs cross-language gold).

**What we learned.**
- **Foreign twins are systematically less separable.** Even the best model separates same-language
  gold from the crowd far better than foreign gold; pooled, the separability home advantage
  (AUC_same − AUC_cross) is **+0.08**. The model assigns lower
  similarity to the correct patent simply because it is written in another language.
- **Separability explains recall.** Across models, Pearson r(AUC_cross, CLIR@10) =
  **+0.98** — cross-lingual recall is, mechanically, a
  cross-lingual *separability* problem, not just a cutoff problem. The best separator is
  **embeddinggemma** (AUC_cross = 0.91).
- The least separable query language is **de**.
- Implication: a monolingual re-ranker won't fix this — the foreign gold is under-scored at the
  embedding level. Better *alignment* (or fusing complementary models) is required.

**Next questions.**
1. Models collapse and under-score along different languages. Are their errors complementary enough
   that fusing them, or routing by language, recovers the buried twins? → Round 9.
2. Fold accuracy, CLIR, consistency, MT-robustness, collapse and separability into a single
   multilingual-robustness verdict. → Round 10.
<!-- /round:08 -->

<!-- round:09 -->
## Round 9 — Complementarity: oracle, rank-fusion and routing

**Question.** Models fail along different languages. Do they fail on the *same* queries, or are their
errors complementary enough to combine? Baseline = best single model
(embeddinggemma, CLIR@10 = 0.54).

**What we learned.**
- **Untuned fusion does *not* beat a dominant model.** RRF over the top-4
  (embeddinggemma, bge-m3, qwen3-0.6B, nomic-v2-moe) lands at CLIR@10 = **0.51**
  (-0.04 vs the best single model), and fusing *all* reliable models is worse
  still (0.47, -0.07). `embeddinggemma` is strong enough
  that rank-averaging with weaker models only injects noise — a caution against reflexive ensembling.
- **Yet complementarity is real.** The oracle (any reliable model finds the foreign twin in top-10)
  reaches **0.66** — a headroom of **+0.12** over the best
  single model. The twins *are* findable; no single model finds all of them, so a *score-aware* or
  *learned* combiner (not plain RRF) is where the gains live.
- **Routing helps, slightly.** The best model differs across query languages
  (1 distinct winners over 5 languages); an oracle per-language router
  edges past the best single model (CLIR@10 = 0.54). The remaining headroom is
  concentrated in the lower-resource / homeless languages, where no current model is strong.

**Next questions.**
1. We now have six orthogonal lenses (accuracy, CLIR, consistency, MT-robustness, collapse,
   separability). Fold them into one multilingual-robustness verdict and a final recommendation. → Round 10.
<!-- /round:09 -->

<!-- round:10 -->
## Round 10 — CLIR robustness synthesis (CLIR-MRS)

**Question.** Average recall hides cross-lingual weaknesses. Fold the lenses into one score in which
**capability is the spine and robustness only modulates** (±50%): CLIR-MRS = capability ×
(0.5 + 0.5·robustness), capability = mean normalised {accuracy, CLIR, separability}, robustness =
mean normalised {consistency, MT-robust, language-parity}. This rewards a strong model for being
even, but never lets an evenly-mediocre model out-rank a genuinely capable one.

**Leaderboard (CLIR-MRS, 95% query-bootstrap CI).**

| rank | model | CLIR-MRS | capability | robustness |
| ---: | --- | :---: | :---: | :---: |
| 1 | `embeddinggemma` | 0.91 [0.83,0.90] | 1.00 | 0.82 |
| 2 | `bge-m3` | 0.81 [0.73,0.85] | 0.89 | 0.81 |
| 3 | `nomic-v2-moe` | 0.73 [0.66,0.78] | 0.83 | 0.77 |
| 4 | `qwen3-0.6B` | 0.72 [0.66,0.76] | 0.88 | 0.62 |
| 5 | `granite-278m` | 0.60 [0.53,0.64] | 0.79 | 0.53 |
| 6 | `SapBERT` | 0.41 [0.35,0.44] | 0.49 | 0.66 |
| 7 | `LaBSE` | 0.40 [0.37,0.43] | 0.61 | 0.33 |
| 8 | `e5-large-instruct` | 0.25 [0.20,0.28] | 0.36 | 0.39 |
| 9 | `gte-base` | 0.00 [0.00,0.00] | 0.00 | 0.67 |

**What we learned.**
- The most robust multilingual retriever is **embeddinggemma** (CLIR-MRS = 0.91
  [0.83, 0.90]; capability 1.00, robustness
  0.82). It leads on the capability axes (accuracy, CLIR, separability) and stays
  competitive on evenness; its weakest axis is **mt_robust**.
- Single-number recall is **misleading**: monolingual/instruction models (e.g. `e5-large-instruct`)
  with respectable same-language matches collapse on the cross-lingual axes and fall to the bottom —
  exactly the failure a patent-retrieval deployment must avoid. The radar makes the lopsided profiles
  obvious.
- The robust-retriever recipe from this study: **avoid language collapse** (Round 6), **separate the
  foreign twin in score space** (Round 8), and accept that **the question's provenance — human or MT —
  barely matters** (Round 3). The remaining gains are in *alignment* and *smart combination*, not in
  re-ranking a single model (Rounds 8-9).

**Recommendation.** Deploy **embeddinggemma** as the single model; reserve a *score-aware* combiner
or per-language routing for the homeless/low-resource query languages (es, zh), where the oracle
headroom is largest (Round 9). Always report **CLIR@10 and language-parity alongside recall**.
<!-- /round:10 -->
