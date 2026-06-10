# Chem-patents multilingual retrieval — key findings (CLIR deep-dive)

*Benchmark:* `MehdiAstaraki/multi-lingual-qac-chem-patents` (`multilingual` variant) — **137**
chemistry-patent questions in **5 languages** (en/de/es/fr/zh; 57 human-original + 80
machine-translated), retrieved against the shared **`multilingual_GP`** haystack (**23,487** docs);
**9** multilingual embedding models. Gold = every language version of a question's source patent.

This study adds **CLIR@k** and a family of cross-lingual metrics the standard MTEB report omits, then
iterates over 10 rounds to explain *why* cross-lingual retrieval fails here and *which model to trust*.

## The benchmark's defining property

Two query populations, and they are not symmetric:
- **original (57)** — question in the source patent's language; has exactly one *same-language* gold.
- **synthetic (80)** — question machine-translated into another language; **no same-language gold at
  all** — pure cross-lingual retrieval.
- **Spanish is a pure query-side language**: 34 Spanish queries, **zero** Spanish gold documents. It
  is the benchmark's built-in "no-home" CLIR stress test.

## Seven headline results

1. **Cross-lingual recall trails same-language recall in every model.** Best CLIR@10 =
   **0.50** (embeddinggemma); the
   home-advantage (MoLIR−CLIR) reaches **+0.55** for the most
   biased model — a same-language copy is far easier than its foreign twin. *(Round 1)*
2. **Retrieval direction is anisotropic.** Hardest direction
   en→de (R@10=0.12);
   English is the easiest *target* language; the most asymmetric pair is
   de↔zh (gap +0.23).
   Retrieval is not a symmetric similarity. *(Round 2)*
3. **Machine translation of the question is *not* the problem.** Controlling for the patent, the paired
   human−MT difference in cross-lingual reach is
   -0.044
   (p=0.13) — statistically insignificant. The project's
   "MT-is-fine-for-the-question" assumption holds. *(Round 3)*
4. **Foreign twins are often buried or lost.** Pooled mate-hit@10 = 0.38;
   **15%** of (query, model) pairs never surface a
   foreign twin even in the **top-1000**. Best twin-finder: embeddinggemma
   (median first-foreign rank 5). *(Round 4)*
5. **Same question, different languages → different rankings.** Cross-lingual RBO ceiling is only
   **0.19**, and r(home-advantage, RBO) =
   **-0.85** — bias drives inconsistency. *(Round 5)*
6. **Language collapse is the mechanism.** Low-resource query languages over-fetch their own language
   up to **49×** the corpus base
   rate; same-language noise out-ranks the gold on
   **60%** of queries; and across models
   r(over-representation, CLIR@10) = -0.60. *(Rounds 6-7)*
7. **It's a separability problem, and complementarity is the lever.** r(cross-language AUC, CLIR@10) =
   **+0.96**: foreign golds are under-scored, not just
   mis-ranked. No single model finds every twin — the oracle reaches CLIR@10 =
   **0.61** (headroom +0.11),
   though plain RRF fusion does **not** beat the dominant model. *(Rounds 8-9)*

## Verdict — CLIR-MRS leaderboard

CLIR-MRS = capability {accuracy, CLIR, separability} × (0.5 + 0.5·robustness {consistency,
MT-robust, language-parity}). Capability is the spine; robustness modulates ±50%.

| rank | model | R@10 | CLIR@10 | home adv | mate-MRR | sep-AUC | CLIR-MRS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `embeddinggemma` | 0.544 | 0.502 | +0.22 | 0.339 | 0.90 | 0.71 |
| 2 | `bge-m3` | 0.476 | 0.437 | +0.23 | 0.317 | 0.87 | 0.70 |
| 3 | `granite-278m` | 0.359 | 0.329 | +0.17 | 0.273 | 0.86 | 0.67 |
| 4 | `nomic-v2-moe` | 0.443 | 0.381 | +0.34 | 0.286 | 0.86 | 0.67 |
| 5 | `qwen3-0.6B` | 0.468 | 0.433 | +0.19 | 0.303 | 0.86 | 0.58 |
| 6 | `LaBSE` | 0.277 | 0.238 | +0.21 | 0.224 | 0.81 | 0.40 |
| 7 | `SapBERT` | 0.212 | 0.179 | +0.18 | 0.151 | 0.79 | 0.40 |
| 8 | `e5-large-instruct` | 0.178 | 0.077 | +0.55 | 0.044 | 0.71 | 0.26 |
| 9 | `gte-base` | 0.004 | 0.000 | +0.02 | 0.000 | 0.49 | 0.00 |

**Most robust multilingual retriever: `embeddinggemma`** (CLIR-MRS = 0.71
[0.67, 0.77]).

## What to do about it

- **Deploy `embeddinggemma`** as the single model; report **CLIR@10 and language-parity next
  to recall** — average recall hides the cross-lingual collapse.
- **Don't reflexively ensemble.** Untuned RRF underperformed the best model here; the oracle headroom
  is real but needs a *score-aware* / learned combiner, or **per-language routing** for the homeless
  / low-resource languages (es, zh).
- **The fix is alignment, not re-ranking.** Foreign twins are under-scored at the embedding level;
  a monolingual re-ranker cannot recover them.
- **Machine-translating the questions is safe** — invest human-translation effort in the source
  patents, not the generated Q/A.

## Figures (key_findings/figures/)

1. `fig01_clir_leaderboard.png` — overall vs CLIR vs MoLIR recall per model.
2. `fig02_home_advantage.png` — same-language home advantage per model.
3. `fig03_directional_clir_matrix.png` — query→document language recall matrix (best model).
4. `fig04_clir_direction_asymmetry.png` — translation-direction asymmetry.
5. `fig05_mt_penalty.png` — paired human vs machine-translated queries.
6. `fig06_mate_retrieval.png` — mate-hit@10/100 and mate-MRR.
7. `fig07_first_foreign_rank.png` — depth to the first foreign twin.
8. `fig08_consistency_vs_bias.png` — RBO consistency vs home-advantage.
9. `fig09_language_collapse.png` — same-language over-representation by query language.
10. `fig10_distractor_language.png` — language of the documents that bury the gold.
11. `fig11_separability.png` — AUC(gold>non-gold), same vs cross language.
12. `fig12_ensemble_headroom.png` — best-single vs fusion vs router vs oracle.
13. `fig13_clir_mrs_leaderboard.png` — CLIR-MRS leaderboard.
14. `fig14_robustness_radar.png` — robustness profiles (radar).

*Reproduce:* `python reports/runs/chem_patents/experimental_codes/run_all.py`
