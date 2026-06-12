# Final paper-figure suite

Curated figures for the paper *"Multilingual embedding models fail in industrial, domain-specific
QA — a chemistry-patent benchmark."* Everything here is generated from local CSV/parquet artifacts;
**no embedding model is loaded and no network is touched.**

## Regenerate

```bash
cd reports/runs/final_plots/code
../../../../.venv/bin/python run_all.py   # render every candidate -> candidates/ (+ data/)
../../../../.venv/bin/python curate.py    # copy winners -> main/ and appendix/
```

Each figure is saved as PNG (300 dpi) **and** PDF; the exact plotted numbers for every figure are in
`data/<candidate>.csv`. Style, loaders, the chem-patents mode recovery, and normalization helpers live
in `code/fp_common.py`.

## The story → figure map

| Claim | Main figure | What it shows |
|---|---|---|
| **A** technical < semantic | `main/fig_A_technical_vs_semantic` | per-model dumbbell, semantic→technical Recall@10, **both** offices; every model drops on technical |
| **B** transfers across offices | `main/fig_B_source_transfer` | Google-Patents vs EPO Recall@10 per model, Spearman ρ = 0.90 / Pearson r = 0.95 |
| **C** per-language (normalized) | `main/fig_C_per_language_recall` | model × query-language Recall@10 heatmap + language-balanced column (GP) |
| **C** (companion) | `main/fig_C_home_advantage` | each language's recall split into same-language (MoLIR) vs cross-language (CLIR) gold — the honest "home advantage" view |
| **D** distractor latch | `main/fig_D_distractor_latch` | confusion-rate heatmap (a look-alike out-ranks all gold, 14–48%) **+** the universal attractor compounds |
| **E** deployment-cost wrap-up | `main/fig_E_cost_capability` | one scatter: CLIR@10 vs XRC50 (log), colour = ARI@100, Pareto frontier |

### Appendix

| File | Claim | Role |
|---|---|---|
| `figA2_per_query_distribution` | A | per-query Recall@10 ECDF + per-model means — the penalty is a distribution shift, not a few queries |
| `figA3_gap_by_language` | A | semantic−technical gap, model × language — positive in nearly every cell |
| `figA5_mt_robustness` | A | human-original vs MT-translated questions retrieve comparably (methodology defense) |
| `figB3_penalty_transfer` | B | the semantic−technical **gap itself** transfers across offices (7/8 models in the positive quadrant) |
| `figC4_epo_per_language` | C | EPO 3-language mirror of the per-language heatmap |
| `figC5_denominators` | C | per-language query / gold-qrel / haystack counts — why language-balancing is required |
| `figD3_cross_lingual_rbo` | D | cross-lingual RBO ≈ 0.39: same concept in 5 languages returns mostly different documents |
| `figD4_score_collapse` | D | when confused, gold-vs-look-alike separability AUC collapses to ≈0.52 (a coin-flip) |
| `figE4_rrc_budget_curves` | E | RRC@K re-ranker recoverability curves with knee K* and the un-rerankable floor |
| `figE5_ari_decomposition` | E | ARI stacked decomposition: cheap-rerank / deep-pool / alignment-only floor |

## Normalization (applied throughout)

1. **Unequal per-language / per-mode counts.** Never pool raw query rows for a headline number.
   Aggregate per `(model, language[, mode])` first, then take a **language-balanced macro-average**
   (each language weighted equally). 95% CIs are bootstrapped within language
   (`fp_common.lang_balanced_ci`). `figC5_denominators` shows why: GP has 10,827 English vs 400
   Chinese haystack docs, and 0 same-language gold for Spanish.
2. **MT-translated questions are kept** in the main figures. The benchmark allows MT *questions* over
   a human-translated corpus (only source documents must be human-translated), and the 80 synthetic
   queries are what give Spanish/Chinese cross-lingual coverage. `figA5_mt_robustness` confirms human
   and MT questions retrieve comparably for every non-degenerate model (the lone exception is
   `e5-large-instruct`, which is language-siloed and already excluded from cross-lingual claims).
3. **Home-advantage confound.** Per-language recall is decomposed into same-language (MoLIR) vs
   cross-language (CLIR) gold rather than reported as one confounded bar (`fig_C_home_advantage`).
4. **Different corpus sizes (GP 23,787 vs EPO 11,315).** Absolute Recall@10 is not directly
   comparable across offices, so the cross-source figure leads with **rank agreement** (Spearman ρ)
   and annotates the corpus sizes on the axes (`fig_B_source_transfer`).
5. **ARI / XRC / frontier** keep the pipeline's existing min-max / median definitions.

## Model set and exclusions

Eight multilingual embedding models, fixed colour per model across all figures:
`embeddinggemma, bge-m3, qwen3-0.6B, nomic-v2-moe, granite-278m, LaBSE, SapBERT, e5-large-instruct`.

- **`gte-multilingual-base` is excluded everywhere** — its results are a model-loading artifact
  (fails trivial self-retrieval, ~0.005 recall). Already dropped from the eval CSVs.
- **`e5-large-instruct` is degenerate on the cross-lingual instruments only** (CLIR@10 = 0.066 < 0.10
  gate). It is drawn hollow / annotated in cross-lingual figures (E1, E3) and excluded from
  XRC/RRC/ARI, but kept in the technical-vs-semantic and per-language Recall figures.

## Claim E: which of the three designs?

The user asked for all three main-figure designs to compare. All are in `candidates/`:

- **`claimE_E1_scatter` — recommended, promoted to `main/`.** One scatter carries all three
  instruments: x = CLIR@10, y = XRC50 (log), colour = ARI@100, with the Pareto frontier drawn.
  Cleanest, most information-dense, and the degeneracy gate is visible as a vertical rule. Reads well
  at `\linewidth`.
- **`claimE_E2_triptych` — strong alternative.** Three labelled panels (XRC50 bars | RRC@K curves
  with knee K* | ARI stacked bars). Shows every instrument explicitly but is wide (best as a
  full-width / two-column figure) and each panel is small.
- **`claimE_E3_scatter_inset` — weakest with this data.** The frontier scatter plus a top-3 RRC@K
  inset; the inset crowds the upper-right where `nomic-v2-moe` (XRC50 = 14) sits, so it needed extra
  y-headroom. Use only if the inset's RRC preview is wanted in the main text.

Recommendation: **E1 in the main paper**, `figE4`+`figE5` in the appendix for the full RRC and ARI
detail. If a reviewer wants all instruments in the main text, swap E1→E2.

## Other ready-to-promote candidates (not in main/appendix)

- `claimA_A4_metric_robustness` — the A1 dumbbell repeated for MRR@10 and Hit@10 (metric robustness).
- `claimB_B2_slope` — GP→EPO slope chart (rank-stability, few crossings).
- `claimC_C2_bars` — per-language pooled bars with per-model dots (alternative to the C1 heatmap).
- `claimD_D1_confusion_heatmap`, `claimD_D2_attractors` — the two halves of the D hero, if a smaller
  single-panel figure is needed.
- `claimD_D5_structure_trap` — structure-style questions: Recall@10 0.29 / confusion 0.47 vs role
  0.66 / 0.19. Strong, actionable; swap in for D3 or D4 if the question-type angle is preferred.

## Spot-check anchors (current artifacts)

- EPO `embeddinggemma` Recall@10: technical 0.46, semantic 0.70.
- Alias-graph publication-lens confusion: `embeddinggemma` 14%, `e5-large-instruct` 48%.
- Cross-lingual RBO (best model) ≈ 0.39; confused vs non-confused separability AUC 0.52 vs 0.68.
- `embeddinggemma` ARI@100 = 0.31, XRC50 = 2.33× (note: the older paper draft cites 3.5× from a prior
  run; these figures use the **current** `cost_frontier.csv` / `ari_decomposition.csv`).
- Source-transfer Spearman ρ = 0.90 (GP vs EPO model ranking).
