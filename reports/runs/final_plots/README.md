# Final paper-figure suite

Curated figures for the paper *"Multilingual embedding models fail in industrial, domain-specific
QA — a chemistry-patent benchmark."* Everything here is generated from local CSV/parquet artifacts;
**no embedding model is loaded and no network is touched.**

## IMPORTANT — data provenance (read this first)

These figures are built on the **full 524-query** Google-Patents benchmark and the **198-query** EPO
benchmark, rebuilt directly from:
- the eval's top-1000 rankings: `reports/runs/{chem_patents,epo}/parts/<model>/retrieval_results/scored_rankings.parquet`
- the authoritative question sets with modes/types: `data/google_patents/qac/qac_chempatents_best.csv`
  and `data/EPO/qac/qac_epo_best.csv`
- gold reconstructed as every language version of each query's source patent present in the haystack.

This replaces an earlier version of these figures that was silently built on a **stale 135-query
summary** (`question_analysis/question_level_metrics.csv`) and **stale cached qrels** (the old
137-query release, in which Spanish gold = 0). The rebuild is validated: reconstructed pooled
Recall@10 matches the official MTEB `model_comparison.csv` to 4 decimals for all 8 models
(e.g. embeddinggemma 0.5740 vs 0.5739). Spanish now correctly has gold (102 gold qrels / 100 queries).

## Regenerate

```bash
cd reports/runs/final_plots/code
../../../../.venv/bin/python run_all.py   # recompute E metrics on 524 gold + render every candidate
../../../../.venv/bin/python curate.py    # copy winners -> main/ and appendix/
```

`fp_common.py` owns the data layer (524-query rebuild, gold reconstruction, language-balancing).
`claimE_metrics.py` recomputes XRC/RRC/ARI on the 524 gold (the shipped `extra_*.csv` were stale).
Every figure's exact numbers are in `data/<name>.csv`; candidates render to `candidates/` (PNG+PDF).

## Story → figure map

| Claim | Main figure | Headline (524-query) |
|---|---|---|
| **A** technical < semantic | `main/fig_A_technical_vs_semantic` | every model drops on technical, both offices (GP gap 0.23, EPO 0.13) |
| **B** transfers across offices | `main/fig_B_source_transfer` | GP vs EPO Recall@10, Spearman ρ = 0.91 |
| **C** per-language (normalized) | `main/fig_C_per_language_recall` | model × query-language Recall@10 + language-balanced column |
| **C** (companion) | `main/fig_C_home_advantage` | same-language (MoLIR ~0.6) vs cross-language (CLIR ~0.35) gold, all 5 languages |
| **D** distractor latch | `main/fig_D_distractor_latch` | a look-alike out-ranks all gold 14–48%; the universal attractor compounds |
| **E** deployment-cost wrap-up | `main/fig_E_cost_capability` | CLIR@10 vs XRC50 (log), colour = ARI@100, Pareto frontier |

### Appendix

| File | Claim | Role |
|---|---|---|
| `figA2_per_query_distribution` | A | per-query Recall@10 ECDF — the penalty is a distribution shift |
| `figA3_gap_by_language` | A | semantic−technical gap, model × language (positive almost everywhere) |
| `figA6_technical_subtype` | A | **NEW** — within technical, parameter/condition & method are hardest, outcome easiest |
| `figA5_mt_robustness` | A | human-original vs MT-translated questions retrieve comparably (methods defense) |
| `figB3_penalty_transfer` | B | the semantic−technical gap itself transfers across offices |
| `figC4_epo_per_language` | C | EPO 3-language per-language heatmap |
| `figC5_denominators` | C | per-language query / gold-qrel / haystack counts (now with Spanish gold) |
| `figD3_cross_lingual_rbo` | D | cross-lingual RBO ≈ 0.39: same concept, 5 languages → mostly different docs |
| `figD4_score_collapse` | D | confused queries: gold-vs-look-alike separability AUC collapses to ≈0.52 |
| `figE4_rrc_budget_curves` | E | RRC@K re-ranker recoverability curves with knee K* and floor |
| `figE5_ari_decomposition` | E | re-ranker budget decomposition: re-rankable in top-10 / needs top-100 pool / alignment floor beyond top-100 |

## What changed numerically vs the stale (135-query) version

- **Spanish/Chinese now have same-language gold.** C3 home-advantage is complete for all 5 languages
  (was broken: Spanish had no bar, Chinese a phantom 0). C5 Spanish gold qrels: 0 → 102.
- **Technical recall is higher on the full set** (the stale subset over-weighted hard queries): GP
  technical 0.20 → 0.28, so the GP semantic−technical gap is 0.23 (was 0.29). Still clearly present
  in every model and both offices.
- **XRC reading cost is higher than the stale estimate**: embeddinggemma XRC50 2.3× → **5.0×**,
  CLIR@10 0.45 → **0.54**. ARI@100 ≈ 0.32 (stable). e5-large-instruct stays degenerate (CLIR 0.094).
- **Larger n → tighter CIs** throughout.

## Normalization (unchanged principles)

1. **Language-balanced means** (macro over query languages) for any headline number; 95% CIs
   bootstrapped within language. `figC5_denominators` shows why (10,827 English vs 400 Chinese docs).
2. **MT-translated questions kept** in main figures (allowed by the benchmark; give es/zh coverage);
   `figA5` confirms they retrieve like human questions for every non-degenerate model.
3. **Home-advantage confound** decomposed into same-language (MoLIR) vs cross-language (CLIR) gold.
4. **Corpus-size difference** (GP 23,787 vs EPO 11,315): cross-office figure leads with rank
   agreement (Spearman), not raw recall.

## Model set / exclusions

Eight models, fixed colour each: `embeddinggemma, bge-m3, qwen3-0.6B, nomic-v2-moe, granite-278m,
LaBSE, SapBERT, e5-large-instruct`. `gte-multilingual-base` excluded everywhere (loading artifact).
`e5-large-instruct` is degenerate on the cross-lingual instruments only (CLIR@10 = 0.094 < 0.10) —
drawn hollow in E1/E3, excluded from XRC/RRC/ARI, kept in the recall figures.

## Claim E — which design

All three are in `candidates/`. **E1 (`claimE_E1_scatter`) is recommended and promoted to main** —
one scatter carrying all three instruments. `claimE_E2_triptych` is the strong full-width alternative
(every instrument explicit); `claimE_E3_scatter_inset` is the weakest (inset crowding). `figE4`+`figE5`
give the full RRC and ARI detail in the appendix.

## Other ready-to-promote candidates

`claimA_A4_metric_robustness` (MRR/Hit), `claimB_B2_slope`, `claimC_C2_bars`,
`claimD_D1`/`claimD_D2` (the two halves of the D hero), `claimD_D5_structure_trap`.

## Spot-check anchors (current 524-query artifacts)

- Recall@10 reconstructed == MTEB to 4 dp for all 8 models (embeddinggemma 0.5740 vs 0.5739).
- GP technical vs semantic Recall@10: 0.28 vs 0.52; EPO 0.34 vs 0.48.
- Home advantage: MoLIR ≈ 0.59 vs CLIR ≈ 0.35 (all five languages).
- Alias-graph confusion: embeddinggemma 14%, e5 48%; cross-lingual RBO ≈ 0.39.
- embeddinggemma XRC50 = 5.0×, CLIR@10 = 0.54, ARI@100 = 0.32. Source-transfer Spearman ρ = 0.91.
