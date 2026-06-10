# Troubleshooting plan (round 1)

**Bottom line up front.** The three top-3 dreamer picks are all **DO-NOW and CPU-only**.
I verified on disk that the load-bearing raw data exists and is sufficient to compute
XRC (reading-cost multiplier), RRC (re-ranker ceiling), and the aggregation-invariance
ribbon **without any new embedding run**:

- All 9 models × both benchmarks have `parts/<model>/retrieval_results/scored_rankings.parquet`
  with the **full top-1000 ranked list + raw `score` + `rank` + gold `relevance`** columns
  (schema `model, query_id, query_language, chebi_id, rank, corpus_id, corpus_language, score, relevance`).
  `common.py` already exposes these as `score_lists()`, `rank_of()`, `ranked_lists()`,
  and `core_per_query()` (which already computes `first_cross_rank` = first-foreign-rank per query).
- The per-axis aggregation inputs for the ribbon are on disk:
  `chem .../key_findings/headline_numbers.csv` and
  `chem .../experimental_plots/round10_robustness_synthesis/robustness_axes_normalized.csv`
  (6 normalized axes per model + capability/robustness/MRS/rank).
- The confusion winner identity + relation is on disk:
  `alias .../experimental_plots/round02_confusion/confusion_per_query.csv`
  (`winner_name`, `winner_relation`, `score_margin`, `rank_gap`).
- `.venv` has pandas 3.0.1 / scipy 1.17.1 / sklearn 1.8.0 / matplotlib — all CPU analyses runnable.

**One feasibility correction to the dreamer.** A4 / M3 (CCI graded by *ontology hop-distance*)
cannot be a multi-level curve from on-disk data: the hard-negative `relation` column is **binary**
(`sibling` 50,856 / `parent` 6,839) — there is no graded distance field. A *two-level* severity
split (sibling-confusion-rate vs parent-confusion-rate) is trivially DO-NOW (already half-computed
in round02's `win_sibling` / `win_parent`); a true graded hop curve would require building/traversing
the ChEBI graph in `data/alias_graph/alias_graph.json` (12 MB) — a heavier, edge-case-prone CPU job,
demoted to a low-priority stretch, **not** a top pick.

**API budget for this round: 0.** Nothing below calls a paid API. No `--evaluate-mteb`.

---

## DO-NOW (ordered)

Convention for every item: new analysis scripts live next to the suites as
`reports/runs/<suite>/experimental_codes/extra_<name>.py`, import `common as C`, write to a
NEW `experimental_plots/extra_<name>/` dir, and **never** overwrite the existing 10 rounds or
`build_key_findings.py` outputs. Run from repo root inside the venv:
`/home/mehdi/Projects/Multi-Lingual-QAC/.venv/bin/python <script>`.
Do NOT re-run `run_all.py` or `build_key_findings.py` (that would risk perturbing the verified
figures/tables the correctness critic signed off). All new work is additive.

---

### DO-NOW-1 — XRC: Cross-Lingual Reading-Cost Multiplier (top-pick #1; closes N1, N4, W1, M1)

- **Goal.** Replace the arbitrary composite's *deployment* role with a distribution-free,
  monotone-invariant, unitful number: "to find the foreign twin at 95% coverage you must scan
  N× more documents than the same-language copy." Per model + per target language.
- **Files.**
  - New: `reports/runs/chem_patents/experimental_codes/extra_xrc_reading_cost.py`.
  - Reads (via `import common as C`): `C.core_per_query()` (has `first_cross_rank` per
    (model,query) and `first_gold_rank`), `C.ranked_lists()`, `C.split_gold()`, `C.q_lang()`.
  - Same-language denominator: `first_gold_rank(ranked, same)` where `same` =
    `split_gold(gold, qlang)[0]`. **Only the 57 originals have same-language gold** — so the
    same-language D95 is defined on that subpopulation; state this as the denominator's domain
    (mirrors the Fig-1 MoLIR caveat B3). The cross-language numerator is defined on all 137.
- **Method (exact).** For each model `m`:
  1. cross depths = `core_per_query()[m].first_cross_rank` over queries with `n_gold_cross>0`
     (treat `inf` = not found in top-1000; for the 95th percentile use the empirical rank, and if
     the 95th percentile falls in the unfound tail, report it as ">1000" / right-censored — be
     explicit, do not silently drop).
  2. same depths = `first_gold_rank` on same-language gold over the 57 originals.
  3. `D95_cross = np.percentile(cross_depths_finite, 95)` (and a censored flag if >5% are inf);
     `D95_same` likewise. `XRC(m) = D95_cross / D95_same`.
  4. Per-language variant: group cross depths by `query_language` (or by target `doc_lang` of the
     mate) for the per-ℓ table.
  5. Also emit a robust secondary at coverage 90% (`D90`) so the headline does not hinge on the
     thin 95th-percentile tail (n is small).
- **Inputs exist?** YES — `scored_rankings.parquet` ×9 verified present (1.2–1.6 MB each);
  `core_per_query()` self-test in `common.py` confirms `first_cross_rank` is populated.
- **Outputs.** `experimental_plots/extra_xrc_reading_cost/`:
  `xrc_per_model.csv` (model, D95_same, D95_cross, XRC95, XRC90, censored_frac),
  `xrc_per_language.csv`, `summary.json`, and one figure
  `xrc_vs_clir.png` (XRC multiplier on y, CLIR@10 on x, per model — the W1 headline scatter).
- **Runtime.** < 30 s (loads the 9 parquets once; everything else is percentile arithmetic).
- **Verify.** (a) For embeddinggemma, `D95_cross` must be ≥ its median first-foreign rank of 5
  (correctness table) and ≤ 1000; sanity-print `lost_share` (egemma 0.0584) so >95% are found,
  i.e. the 95th percentile is a *finite* rank — good, egemma's XRC is well-defined. (b) gte-base
  (lost_share 0.91) will be censored — the script must label it, not emit a fake number.
  (c) Cross-check the survival-curve numbers already in
  `round04_mate_retrieval/first_foreign_rank_cdf.png` data: P(first foreign ≤ x) crossing 0.95
  gives D95_cross by construction.
- **API cost.** 0.
- **Risk.** Small-n / censoring at the 95th percentile. Mitigation: report D90 alongside,
  report the censored fraction, and frame XRC as "≥" for censored models. Do **not** let any single
  model's XRC become a load-bearing abstract number unless its censored_frac < 0.05.

---

### DO-NOW-2 — RRC: Re-Ranker Recoverability Ceiling (top-pick #3 second half; closes N2, M5, W5)

- **Goal.** Turn "a monolingual re-ranker cannot recover under-scored foreign twins" from a slogan
  into a per-model bound: `RRC(m,K) = fraction of queries whose foreign gold appears within top-K`.
  `1 − RRC` is provably unrecoverable by any re-ranker that only re-orders a top-K pool.
- **Files.** Fold into the SAME script as DO-NOW-1 (`extra_xrc_reading_cost.py`) — both read
  `first_cross_rank`, so do them together to avoid loading the parquets twice. Or a sibling
  `extra_rrc_reranker_ceiling.py`. Reads `C.core_per_query()['first_cross_rank']`.
- **Method.** For K ∈ {100, 1000}: `RRC = mean(first_cross_rank <= K)` per model over queries with
  `n_gold_cross>0`. The "15% never in top-1000" headline = `1 − RRC(K=1000)` pooled (already on
  disk as `round04.pooled_lost_share_top1000 = 0.1542`) — RRC just makes it per-model and at K=100,
  the realistic re-ranker pool.
- **Inputs exist?** YES (same `first_cross_rank`).
- **Outputs.** add `rrc_per_model.csv` (model, RRC@100, RRC@1000, lost@1000) + a figure
  `rrc_ceiling.png` (stacked bar per model: recoverable-by-rerank@100 vs lost-forever) — this is
  W5's one-figure falsification kit.
- **Runtime.** negligible (shares the load with DO-NOW-1).
- **Verify.** Pooled `1 − RRC@1000` must equal 0.1542 (round04 summary) within rounding;
  egemma RRC@1000 = `1 − 0.0584 = 0.9416`. Print both as the consistency check.
- **API cost.** 0.
- **Risk.** None beyond interpretation; keep K explicit (a re-ranker over a top-100 pool ≠ top-1000).

---

### DO-NOW-3 — Aggregation-invariance ribbon (top-pick #2; closes N1, T5, A7, M6)

- **Goal.** Show embeddinggemma is rank-1 under *every* sensible aggregation, so the deployment
  recommendation rests on invariance, not on the CLIR-MRS weights.
- **Files.**
  - New: `reports/runs/chem_patents/experimental_codes/extra_aggregation_invariance.py`.
  - Reads ONLY two CSVs already on disk (no parquet, no embeddings):
    `experimental_plots/round10_robustness_synthesis/robustness_axes_normalized.csv`
    (cols: `accuracy, clir, separability, consistency, mt_robust, lang_parity` already min-max
    normalized) — the canonical 6 axes.
- **Method (four schemes over the 6 normalized axes).**
  1. **Current CLIR-MRS** (reproduce from the same axes: `capability=mean(acc,clir,sep)`,
     `robustness=mean(cons,mt,parity)`, `MRS=cap*(0.5+0.5*rob)`) — sanity must match
     `robustness_scores.csv` rank column exactly.
  2. **Borda count** over per-axis ranks (MMTEB-style): rank each model on each of the 6 axes,
     sum ranks, lower total = better.
  3. **Equal-weight mean** of the 6 normalized axes.
  4. **Per-axis winner-take-all**: count how many of the 6 axes each model wins.
  Emit, per model, the **rank under each scheme** and the **rank range** (min..max across schemes).
- **Inputs exist?** YES — `robustness_axes_normalized.csv` verified
  (egemma row = all 1.0 on capability axes). Also keep `headline_numbers.csv` as a cross-check of raw axes.
- **Outputs.** `experimental_plots/extra_aggregation_invariance/`:
  `aggregation_ranks.csv` (model × 4 schemes + rank_min/rank_max), `summary.json` (one line:
  "embeddinggemma is rank-1 under N/4 schemes; its rank range is [a,b]"), and a figure
  `aggregation_ribbon.png` (per model, a horizontal bar spanning its rank range across schemes).
- **Runtime.** < 5 s (one tiny CSV).
- **Verify.** Scheme-1 ranks MUST reproduce `round10` ranks (egemma=1, bge-m3=2, granite=3, …) —
  if not, the normalization read is wrong. Borda and equal-weight should keep egemma=1 (it leads
  all 3 capability axes); confirm and report whether qwen3/granite swap (the cohesion G6 middle-of-field
  reshuffle) is scheme-dependent.
- **API cost.** 0.
- **Risk.** If a scheme does NOT put egemma rank-1, that is a *finding to report honestly*, not a bug —
  the ribbon's value is the truthful range either way. Flag it for the writer rather than hiding it.

---

### DO-NOW-4 — B1 fix data: directional matrix re-read (hub-and-spoke) (closes B1, T2, A5, N3)

- **Goal.** Kill the false "English is the easiest target" claim with the correct numbers and a
  defensible structural reframing. **B1 is the single MISMATCH the correctness critic flagged** and it
  is *hardcoded* in `build_key_findings.py` L117 and `round02_directional_clir.py`; this item produces
  the replacement numbers the writer needs.
- **Files.**
  - New: `reports/runs/chem_patents/experimental_codes/extra_directional_hub.py`.
  - Reads `experimental_plots/round02_directional_clir/pair_recall.csv` and `asymmetry.csv`
    (both verified present).
- **Method.** From `pair_recall.csv`: (i) pooled per-target-column means over the 8 reliable models
  (recompute the en 0.367 / fr 0.375 / zh 0.350 / de 0.309 the critic published, confirming
  **fr > en**); (ii) hardest directed edge (en→de 0.12); (iii) most asymmetric pair (de↔zh +0.23);
  (iv) node-level "hub score" = mean *incoming* recall per target language, annotated with corpus
  share (en 0.46 / fr 0.37 / es 0.09 / de 0.07 / zh 0.004 from `round06.corpus_base_share`).
- **Inputs exist?** YES (`pair_recall.csv`, `asymmetry.csv` present).
- **Outputs.** `experimental_plots/extra_directional_hub/hub_scores.csv` (target lang, mean-incoming
  recall, corpus share), `summary.json`. (Figure optional — the existing `cp_fig03`/`cp_fig04`
  already cover the matrix; this is primarily a numbers/table deliverable for the writer.)
- **Runtime.** < 5 s.
- **Verify.** Reproduce fr 0.375 > en 0.367 exactly (the critic's recomputation) — this is the gate.
- **API cost.** 0.
- **Note for writer.** Replacement sentence (C-B1 safe fix): drop "English is the easiest target";
  use "the hardest direction is en→de (R@10 0.12) and the most asymmetric pair is de↔zh (gap +0.23)",
  with the corpus-composition caveat (T2). Cite CLIRMatrix (N3) in the same sentence.

---

### DO-NOW-5 — Availability-adjusted home advantage (closes T1, A2, partial N2)

- **Goal.** Decompose the +0.55 home advantage into "availability artifact" vs "residual encoder
  bias" so §Results and §6 stop reading as contradictory.
- **Files.**
  - New: `reports/runs/alias_graph/experimental_codes/extra_availability_residual.py`
    (the per-language availability shares 42% / 8–10% live on the **alias** side, `FINDINGS.md`
    L153–157; home-advantage per language is in the alias `round04_home_advantage.py` outputs).
  - Reads alias `round04_home_advantage` per-language outputs + the availability shares.
- **Method.** Across the language points (en/de/es/fr/zh), regress per-language home advantage on
  per-language in-corpus gold-availability share (OLS, n=5 — explicitly a *descriptive* partial,
  not an inference test at n=5). Report the slope, the R², and the **residual** home advantage
  (mean of residuals, or the intercept-at-mean-availability). State the honest headline that follows:
  if residual ≈ 0 → "home advantage is almost entirely an availability artifact"; if positive →
  "even after controlling for availability, encoders retain a +X same-language bias."
- **Inputs exist?** Per-language home-advantage: need to confirm the exact column in alias
  `round04` outputs (the suite computes same/cross recall by language — `same_recall_by_lang`,
  `cross_recall_by_lang` are in the correctness table). Availability shares: YES, in `FINDINGS.md`.
  **If a clean per-language home-advantage vector is not already dumped**, derive it as
  `same_recall_by_lang − cross_recall_by_lang` from the round04 per-language CSV (both columns
  verified to exist per the correctness number-trace table).
- **Outputs.** `experimental_plots/extra_availability_residual/availability_regression.csv`
  (lang, home_adv, availability_share, fitted, residual), `summary.json` (slope, R², residual_mean),
  `availability_residual.png` (scatter + fit).
- **Runtime.** < 10 s.
- **Verify.** n=5 points; print them. The fit is illustrative — **the writer must label it
  "descriptive (n=5 languages), not an inferential test"** (mirrors the T5 small-n discipline).
- **API cost.** 0.
- **Risk.** n=5 is tiny. Keep it as a *decomposition narrative*, not a p-value. This is the only
  DO-NOW item with a non-trivial "does the exact column exist" dependency — the fallback derivation
  is specified above, so it cannot block.

---

### DO-NOW-6 — n=9 robustness pass: drop-the-collapsers correlations (closes T5, A3)

- **Goal.** Show the load-bearing mechanism correlations survive dropping the two degenerate encoders.
- **Files.**
  - New: `reports/runs/chem_patents/experimental_codes/extra_correlation_robustness.py`.
  - Reads `experimental_plots/round08_separability/per_model.csv` (`auc_cross, clir_at_10`),
    `round05_consistency/per_model.csv` (`rbo, home_adv`), `round06_language_collapse/per_model.csv`
    (`mean_overrep`/`same_lang_overrep`, `clir`). All verified present with the needed columns.
- **Method.** For each load-bearing pair — (auc_cross, clir@10) [r=+0.96], (over-rep, clir@10)
  [r=−0.60], (home_adv, rbo) [r=−0.85] — recompute (i) Pearson on all 9, (ii) Pearson on the 7
  non-collapsed (drop gte-base and e5-large-instruct), (iii) Spearman on both sets. Tabulate.
- **Inputs exist?** YES.
- **Outputs.** `experimental_plots/extra_correlation_robustness/correlation_robustness.csv`
  (pair, pearson_n9, pearson_n7, spearman_n9, spearman_n7), `summary.json`.
- **Runtime.** < 5 s.
- **Verify.** `pearson_n9` for (auc_cross, clir@10) must reproduce +0.96 (round08 summary).
- **API cost.** 0.
- **Note for writer.** Annotate the load-bearing r's: "(n=9 models; r=… on the 7 non-collapsed
  encoders; Spearman ρ=…)". If the relationship weakens on n=7, report it honestly — that is the
  point of the check.

---

### DO-NOW-7 (lower priority) — Joint failure-mode + universal-blind characterization (closes A6, A8, G1 orphans)

- **Goal (A6).** For confused alias queries, cross-tab the winning distractor as
  (same-language non-gold / cross-language sibling / same-language sibling) — the "modal failure is a
  same-language sibling (both traps at once)" joint claim that binds the two benchmarks (helps G1).
- **Goal (A8).** Characterize the ~12% universal-blind core (16/132) so the orphan number is earned
  in Analysis, not first-seen in Deployment.
- **Files.**
  - A6: new `reports/runs/alias_graph/experimental_codes/extra_joint_failure.py`. Reads
    `round02_confusion/confusion_per_query.csv` (`winner_name, winner_relation, language`) + needs
    the winner's *language* via `C.doc_lang`/suffix on the winning corpus-id. NOTE:
    `confusion_per_query.csv` logs `winner_name`+`winner_relation` but **not the winner's corpus-id
    language directly** — recover the winning hard-neg corpus-id by re-running the round02 `best()`
    logic (it has `rank_of` + `hardneg_sets`), or join on `neighbor_map`. This is a light re-derivation,
    still CPU-only.
  - A8: the 16 universal-blind query ids are in alias `round08_model_agreement` output
    (`universal_blind_spots`); join to `q_concept()`, `q_type()`, `q_lang()` to tabulate their
    compound/question-type/language composition.
- **Inputs exist?** YES (confusion_per_query.csv; round08 ids). A6 needs the light winner-language
  re-derivation noted above.
- **Outputs.** `extra_joint_failure/joint_failure_modes.csv` + `universal_blind_profile.csv`,
  `summary.json`.
- **Runtime.** < 20 s.
- **API cost.** 0.
- **Priority.** Do these only after DO-NOW-1..6 land; they are cohesion/depth bonuses, not load-bearers.

---

### DO-NOW-8 (optional stretch) — Two-level confusion severity (sibling vs parent) (partial A4/M3/CCI)

- **Goal.** A clean *two-level* severity reading: confusion rate when the winner is a *sibling*
  (near-twin, catastrophic) vs a *parent* (distant, noise). This is the realizable, on-disk form of
  CCI — NOT the graded hop-distance curve (which on-disk data does not support).
- **Files.** Reuse `round02_confusion/confusion_per_query.csv` columns `win_sibling`, `win_parent`
  (already computed). A tiny aggregation script or even just a table for the writer.
- **Inputs exist?** YES.
- **Outputs.** `extra_confusion_severity/severity_split.csv` (model × {sibling-win-rate,
  parent-win-rate}), confirming "siblings do the damage."
- **Runtime.** < 5 s. **API cost.** 0.
- **Honest scope note.** Do **NOT** claim a "hop-distance decay law" (A4 / W2) from this — the
  relation field is binary. The graded-hop version is BACKLOG below.

---

## BACKLOG-EVAL (exact copy-pasteable lines for needs_eval.md)

These require **new embedding-model runs** (or are inferential experiments needing held-out
re-retrieval / human judgments) and must NOT be run by the implementer. Append verbatim under the
`<!-- implementer appends below this line -->` marker in
`/home/mehdi/Projects/Multi-Lingual-QAC/paper/loop/needs_eval.md` (format:
`- [ ] <id> | <command/what> | <why> | <round>`):

```
- [ ] W4-formula-injection | Re-retrieve the failing structure-style alias queries after injecting the language-independent chemical formula token into each query string, on the existing 9 models (re-encode queries only, corpus embeddings reusable): for each model run the standard retrieval over multilingual_GP / alias corpus with the modified queries, then recompute paired recall/confusion deltas on the SAME queries via reports/runs/alias_graph/experimental_codes round07 logic. | Upgrades the p<0.01 formula-token *observation* into a causal intervention ("adding H2S to the query measurably rescues retrieval"). Needs new query embeddings → eval. | r1
- [ ] CLIRMRS-external-validation | Collect a small held-out external utility signal (human cross-jurisdiction search-satisfaction judgments on a query slice, OR end-to-end RAG answer-correctness on a slice) and compute rank-correlation(CLIR-MRS, utility) vs rank-correlation(mean-recall, utility). | Novelty critic route #1: the only thing that converts CLIR-MRS from a demoted convenience into a *validated* contribution. Needs new human/RAG eval. | r1
- [ ] XRC-conformal-M2 | OPTIONAL: split-conformal version of XRC. The raw per-(query,doc) scores ARE on disk (score_lists()), so a split-conformal D95(cross)/D95(same) with a finite-sample coverage guarantee is technically CPU-computable — but with only 57 same-language-gold queries the calibration/test split is too thin for a credible guarantee. Defer until either benchmark grows OR a larger same-language-gold pool exists. | Conformal coverage guarantee (cite Conformal-RAG SIGIR 2025) on top of the empirical XRC. Empirical M1 ships now (DO-NOW-1); this is the guarantee upgrade. | r1
- [ ] CCI-hop-distance-law | Build the ChEBI taxonomy graph from data/alias_graph/alias_graph.json, compute the true graph hop-distance from each query concept to each winning hard-negative's neighbor_chebi_id, then plot confusion rate vs hop-distance (A4 / W2 "decay law"). NOTE: this is CPU-only but requires a non-trivial graph build + traversal with edge cases; the on-disk hard-negative `relation` field is binary (sibling/parent) so the law cannot come from the existing CSVs. | Donates a domain-specific "confusion decays with ChEBI hop-distance" law if it holds. CPU but graph-construction risk → deferred from DO-NOW. | r1
- [ ] equivalence-audit-spotcheck | Small expert-annotated spot-check that the parallel human-translated golds are claim-level equivalent (a few dozen patent pairs). | Pre-empts the hostile "how do you know your parallel golds are equivalent?" review; current answer is "by construction" (correctness T4). Needs human annotation. | r1
```

**Note:** XRC-conformal and CCI-hop-distance are tagged backlog *defensively* (they are CPU-feasible
in principle but carry small-n / graph-build risk). The empirical XRC (DO-NOW-1), RRC (DO-NOW-2), and
the two-level severity split (DO-NOW-8) ship the load-bearing versions now; the backlog items are pure
upside.

---

## WRITER-ONLY (reframes/citations to pass forward — no computation)

These need **zero compute** and go straight to the next round's writer. Each maps to a critic fix.

- **C-B2 / G2 (RBO ceiling attribution + abstract↔body seam).** Change intro/conclusion "even the
  best model reaches … 0.19" → "the best cross-lingual RBO any model achieves is 0.39 (alias-graph)
  / 0.19 (cross-lingual)". 0.19 is *granite's* ceiling, embeddinggemma is 0.154 — say "ceiling across
  models," not "the best model." Make the abstract carry both numbers explicitly:
  "(0.39 on the alias-graph benchmark, 0.19 on the cross-lingual benchmark)".
- **C-B3 (Fig 1 MoLIR population).** Add to the Fig 1 caption: "MoLIR@10 is defined only on the 57
  original queries (the only ones with a same-language gold); the +0.55 home advantage is measured
  paired within those queries." (Also the denominator caveat for XRC's D95_same, DO-NOW-1.)
- **C-N2 (reframe mechanism as confirm-on-content-controlled).** Rewrite C3 from "first decomposition"
  to "we **confirm** the alignment-not-translation finding of [2511.19324, 2507.07543] on a
  content-controlled parallel corpus that removes the translationese/content confounds those studies
  could not, and add (i) chemistry-specific sibling-compound confusability and (ii) a separability test
  that turns 'alignment is the fix' into a falsifiable re-ranker bound (RRC, DO-NOW-2)." Add both cites.
- **C-N3 (narrow every "first", cite precedents inline).** One pass adding, in the same sentence as
  each claim: CLEF-IP + DAPFAM (C1, narrow to "first content-controlled chemistry-ontology-grounded"),
  CLIRMatrix (directional matrix), Bailey et al. 2017 (cross-lingual RBO lineage), AUC-as-separability
  (standard machinery), Oard 1998 + Saleh & Pecina 2020 (QT-vs-DT budget rule), 2605.24297
  (English-only patent-embedding overlap — distinguish on cross-lingual). ~10 bib entries (all listed
  with URLs in `critic_novelty.md` "Missing citations" section — copy them).
- **C-T2 (corpus-composition caveat).** One sentence in §Results/Limitations: "directional asymmetry
  partly reflects corpus language composition (en 46% / zh 0.4% of documents), not only encoder
  behavior." (Backed by DO-NOW-4 hub_scores.csv.)
- **M6 (demote CLIR-MRS).** Add one sentence: "the winner leads on CLIR@10, RBO, mate-rank, AND
  separability individually (Table 1), so no composite weighting is load-bearing" — paired with the
  DO-NOW-3 invariance result. Cite MMTEB (Borda) + 2605.31142 (rankings are aggregation-sensitive).
- **C-G1 (split the §6 fused paragraph).** Two sentences, two footnotes — one alias (0.63–0.82 /
  0.35–0.47, 42% / 8–10%), one chem-patents (49×, 60%) — exactly as the cohesion critic drafted.
  Optionally lead with DO-NOW-7's joint cut.
- **C-G3 (Related Work → Benchmarks bridge).** Move the future-work disclaimer up into the calibration
  paragraph; end §2 with the forward bridge the cohesion critic drafted.
- **C-G4 (\todo markers).** Replace both in-text red `\todo` blocks with one clean deferral sentence
  each ("Detailed corpus-construction statistics are deferred to the system description; all
  load-bearing sizes come from the two benchmark datasets.") and keep the trace as `% TODO` comment.
- **C-G5 (cp_fig11 caption + radar axis).** Re-caption fig11 so +0.96 is clearly a text statistic the
  bars *motivate* ("per model, cross-language gold (red) is harder to separate than same-language
  (green); the model-level AUC–CLIR@10 correlation is +0.96, text"). Add to each radar one clause
  ("embeddinggemma leads on consistency and separability, not raw recall").
- **C-G6 (teaser vs leaderboard reorder).** One sentence near Table 1: "the order changes from Fig. 1 —
  ranking by CLIR-MRS rather than recall reshuffles the middle of the field, which is the point of the
  paper." (DO-NOW-3 tells you exactly which models swap and whether it is scheme-robust.)
- **C-G-orphans (C5 + 12%).** Soften C5 to "a reproducible pipeline (human validation summarized in the
  system description)" until numbers are under reports/; introduce the 12% in Analysis (DO-NOW-7
  universal-blind profile earns it).
- **W3 (framing overlay — "the cross-lingual tax has two line-items").** Optional unifying device:
  cross-lingual retrieval pays a **reading-cost tax** (XRC, measured on the CLIR benchmark) and a
  **confusability tax** (the look-alike, measured on the alias-graph benchmark) — "two taxes, two
  instruments, one decision." Strongest story-level answer to G1/G2 if cohesion needs more than the
  joint splits. Pure framing.
- **Cohesion nits.** Spell "cross-language AUC" (not bare "AUC") in the abstract's +0.96 claim to match
  Analysis verbatim; harmonize "home-advantage"/"home advantage" spelling; add a half-clause "(two
  related bias proxies)" where the two −0.85/−0.87 correlations use different x-variables.

---

## Round API budget plan (target 0, cap 20)

**Planned API spend this round: 0.** Every DO-NOW item is CPU-only (parquet/CSV re-aggregation +
matplotlib). No `--evaluate-mteb`, no LLM, no code-switch variant E. All five LLM-touching paths
(`--cs-generate-qa` variant E, `--evaluate-mteb`, any new model run) are deliberately avoided.
The two genuinely eval-bound ideas (W4 formula-injection, CLIR-MRS external validation) are routed to
BACKLOG-EVAL, not executed.

---

## Risks & sequencing notes

1. **Protect the verified figures/tables.** The correctness critic verified all 81 leaderboard cells
   and ~95% of prose numbers as EXACT. Therefore: **do NOT re-run `run_all.py` or
   `build_key_findings.py`** this round — they would regenerate (and could perturb) the signed-off
   outputs and would re-emit the B1 "English is the easiest target" string. All DO-NOW work is
   *additive*, writing only to NEW `experimental_plots/extra_*/` dirs. The B1 string fix itself is a
   WRITER-ONLY prose edit in `main.tex` (the writer changes the sentence; the implementer supplies the
   correct numbers via DO-NOW-4). If the team later wants the executive summaries regenerated, fix
   `build_key_findings.py` L117 + `round02_directional_clir.py` L217 first — but that is a *separate,
   flagged* edit, not part of this additive round.

2. **Sequencing (payoff × independence).**
   - First: **DO-NOW-1 + DO-NOW-2** (one script, shared parquet load) — the two top-pick headline
     numbers (XRC, RRC). Highest novelty payoff, lowest risk.
   - Then: **DO-NOW-3** (aggregation ribbon) — neutralizes the single highest-risk over-claim, 5 s.
   - Then: **DO-NOW-4** (B1 replacement numbers) — fixes the one MISMATCH; gates on reproducing fr>en.
   - Then: **DO-NOW-5, DO-NOW-6** (availability residual, correlation robustness) — close T1/T5.
   - Last/optional: **DO-NOW-7, DO-NOW-8** (joint failure, severity split) — cohesion/depth bonuses.

3. **Small-n is the recurring hazard.** XRC's 95th percentile, the n=5 availability regression, and
   the n=9 correlations are all thin. Every one of these DO-NOW items must emit the n, the censored
   fraction, and a secondary (D90, Spearman, residual) so the writer never leans the abstract on a
   fragile tail. Frame XRC for censored models as "≥". This is exactly the discipline the correctness
   critic (T5) and the dreamer (A3) asked for.

4. **The paper stands without any of this.** None of the DO-NOW items is load-bearing for the existing
   claims — they *upgrade* (XRC replaces the composite's deployment role; RRC hardens the mechanism;
   the ribbon de-risks the composite) and *correct* (B1) and *decompose* (T1/T5). No code-switch /
   new-eval result is required for the paper to be complete, per the core principle.

5. **One column-existence dependency** (DO-NOW-5: per-language home advantage). Fallback derivation
   (`same_recall_by_lang − cross_recall_by_lang` from alias round04, both columns verified in the
   correctness number-trace table) is specified, so it cannot block. If even the fallback is messy,
   demote DO-NOW-5 to WRITER-ONLY (the prose hedge "(much of which is a gold-availability artifact,
   §6)" already closes T1 qualitatively).
