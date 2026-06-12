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

## Main results table (`tables/main_table.tex`, `data/main_table.csv`)

Generated by `code/table_main.py`. Four metrics per benchmark, two per axis: **nDCG@10** and
**R@10** (IR power), **CLIR@10** (cross-language Recall@10) and **LT%** = CLIR@10/MoLIR@10 (language
transfer; cross-linguality). R@10/CLIR/LT are language-balanced. Both benchmarks reconstruct gold from
the source patent's language versions and validate to MTEB Recall@10 to 4 dp. gte excluded; e5 daggered
(fails the CLIR@10<0.10 gate). Rows ordered by GP nDCG@10.

| Model | GP nDCG | GP R@10 | GP CLIR | GP LT% | EPO nDCG | EPO R@10 | EPO CLIR | EPO LT% | JBC@100 | k\*₈₀ |
|---|---|---|---|---|---|---|---|---|---|---|
| embeddinggemma | **0.50** | **0.57** | **0.54** | 73 | **0.53** | **0.58** | **0.52** | 74 | **0.78** | **147** |
| bge-m3 | 0.45 | 0.51 | 0.47 | **74** | 0.50 | 0.55 | 0.47 | 66 | 0.70 | 304 |
| qwen3-0.6B | 0.43 | 0.49 | 0.45 | 72 | 0.43 | 0.47 | 0.39 | 61 | 0.68 | 425 |
| nomic-v2-moe | 0.42 | 0.46 | 0.41 | 63 | 0.48 | 0.51 | 0.43 | 63 | 0.67 | 367 |
| granite-278m | 0.36 | 0.41 | 0.39 | 72 | 0.35 | 0.39 | 0.36 | 84 | 0.60 | >1000 |
| LaBSE | 0.25 | 0.30 | 0.27 | 59 | 0.24 | 0.27 | 0.25 | **85** | 0.49 | >1000 |
| e5-large-instruct† | 0.21 | 0.21 | 0.09 | 12 | 0.31 | 0.29 | 0.12 | 19 | 0.32 | >1000 |
| SapBERT | 0.20 | 0.24 | 0.20 | 49 | 0.20 | 0.22 | 0.17 | 54 | 0.37 | >1000 |

Cross-corpus rank agreement Spearman ρ(nDCG@10) = 0.95; mean cross-lingual tax (1−LT) ≈ 41% GP / 37% EPO.

**Bilingual coverage (foreign-GOLD, caveat-free).** Language is treated as a retrieval facet (subtopic
recall, Zhai et al. SIGIR 2003): a query is "covered" only when the top-k holds a relevant document in
*both* its own language and another language. Over the GP+EPO queries that have both gold types (n=459):
**JBC@100** (higher better) = fraction of queries whose top-100 contains both a same-language and a
cross-language gold; **k\*₈₀** (lower better) = the smallest depth at which both are present for ≥80% of
queries (>1000 = never reaches 80% within the retrieved top-1000). Both rank the four weak models last —
only the four genuinely cross-lingual models reach 80% bilingual gold coverage (embeddinggemma 147,
bge-m3 304, nomic 367, qwen3 425); granite/LaBSE/SapBERT/e5 never do. Bib entry needed for the caption
cite: `zhai2003subtopic` (Zhai, Cohen & Lafferty, *Beyond Independent Relevance*, SIGIR 2003).

**Foreign-above-home-gold (ranking-interference diagnostic; not in the table).** Fraction of queries
where a foreign item outranks the same-language gold — two readings:

| Model | foreign *doc* > home gold | foreign *gold* (twin) > home gold |
|---|---|---|
| embeddinggemma | 0.42 | 0.13 |
| bge-m3 | 0.46 | 0.23 |
| qwen3-0.6B | 0.42 | 0.15 |
| nomic-v2-moe | 0.38 | 0.12 |
| granite-278m | 0.60 | 0.33 |
| LaBSE | 0.81 | 0.52 |
| SapBERT | 0.71 | 0.33 |
| e5-large-instruct | 0.09 | 0.00 |

Note this is non-monotone with quality: e5-large-instruct scores *lowest* (0.09 / 0.00) because it
siloes — it ranks the same-language gold at the very top, so almost nothing foreign precedes it. So it
reads as "foreign interference above the home answer," a bias/noise diagnostic, not a quality ranking;
the weak-but-language-agnostic LaBSE (0.81 / 0.52) wades through the most foreign content before its
home answer.

**English-pivot rate.** Among non-English queries that have both a home-language gold and an English
version of the patent (N=267): how often a foreign twin outranks the home gold, of which the English
twin specifically, and English's share. The middle column is the English-pivot rate as a percentage of
those queries (English>home count ÷ 267):

| Model | foreign-gold > home | English > home (% of queries) | English share |
|---|---|---|---|
| embeddinggemma | 26 | 5.2% | 54% |
| bge-m3 | 85 | 28.5% | 89% |
| qwen3-0.6B | 26 | 4.9% | 50% |
| nomic-v2-moe | 39 | 10.5% | 72% |
| granite-278m | 97 | 27.0% | 74% |
| LaBSE | 49 | 0.7% | 4% |
| SapBERT | 69 | 15.4% | 59% |
| e5-large-instruct | 0 | 0.0% | — |

bge-m3 and granite route through English most (≈27–29% of non-English queries put the English copy above
the home answer); embeddinggemma and qwen3 respect the home language far more (≈5%). LaBSE (0.7%, balanced
multilingual) and e5-large-instruct (0%, siloes to the home language) essentially never pivot to English.

**Findings — P1 (leaderboard, IR power, cross-corpus consistency).**
embeddinggemma tops both patent offices on every axis (Google Patents / EPO nDCG@10 0.50 / 0.53,
Recall@10 0.57 / 0.58), with bge-m3 a consistent second; the model ordering is near-identical across the
two corpora (Spearman ρ = 0.95 on nDCG@10), so the conclusions are a property of the task, not of one
collection. Yet the absolute numbers are low for an industrial deployment: even the strongest model
leaves roughly 43% of the relevant patents out of the top-10 cross-lingually (CLIR@10 ≤ 0.54), and
performance falls off a cliff below the top two — six of the eight models score under 0.47 CLIR@10, and
three under 0.27. Off-the-shelf multilingual embedders, in other words, are not yet adequate dense
retrievers for domain-specific (chemistry-patent) search.

**Findings — P2 (the cross-lingual tax and the language-transfer collapse).**
Every model pays a steep cross-lingual tax: on average it retains only ~60% (Google Patents) / ~63%
(EPO) of its same-language recall once the answer lives in another language (LT), i.e. roughly 40% of
within-language performance evaporates at the language barrier. LT is also the cleanest discriminator of
multilinguality: e5-large-instruct looks mid-pack on raw IR (nDCG@10 0.21 / 0.31) but is effectively
monolingual — LT 12% / 19%, CLIR@10 below 0.13 — retrieving the same-language copy and ignoring its
foreign twins, which is why it alone trips the CLIR@10 < 0.10 degeneracy gate. Only embeddinggemma and
bge-m3 behave as genuinely cross-lingual retrievers (LT ≈ 73%). The difficulty is concentrated in
distant-script, lower-resource pairs: granite and LaBSE transfer almost losslessly on EPO's three
European languages (LT ≈ 84%) but shed 12–25 points once Spanish and especially Chinese routes enter on
Google Patents (LT 59–72%).

## Spot-check anchors (current 524-query artifacts)

- Recall@10 reconstructed == MTEB to 4 dp for all 8 models (embeddinggemma 0.5740 vs 0.5739).
- GP technical vs semantic Recall@10: 0.28 vs 0.52; EPO 0.34 vs 0.48.
- Home advantage: MoLIR ≈ 0.59 vs CLIR ≈ 0.35 (all five languages).
- Alias-graph confusion: embeddinggemma 14%, e5 48%; cross-lingual RBO ≈ 0.39.
- embeddinggemma XRC50 = 5.0×, CLIR@10 = 0.54, ARI@100 = 0.32. Source-transfer Spearman ρ = 0.91.
