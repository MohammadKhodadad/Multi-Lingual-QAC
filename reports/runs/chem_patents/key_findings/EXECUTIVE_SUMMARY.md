# Chem-patents multilingual retrieval — key findings (CLIR deep-dive)

*Benchmark:* `MehdiAstaraki/multi-lingual-qac-chem-patents` (`multilingual` variant) — **524**
chemistry-patent questions in **5 languages** (en/de/es/fr/zh; 257 human-original + 267
machine-translated), retrieved against the shared **`multilingual_GP`** haystack (**23,787** docs);
**9** multilingual embedding models. Gold = every language version of a question's source patent.

This study adds **CLIR@k** and a family of cross-lingual metrics the standard MTEB report omits, then
iterates over 10 rounds to explain *why* cross-lingual retrieval fails here and *which model to trust*.

## The benchmark's defining property

Two query populations, and they are not symmetric:
- **original (257)** — question in the source patent's language; has exactly one *same-language* gold.
- **synthetic (267)** — question machine-translated into another language; **no same-language gold at
  all** — pure cross-lingual retrieval.
- **Spanish**: 103 Spanish queries and **138** Spanish gold-doc instances. (The earlier
  137-query release had **zero** Spanish gold — a pure query-side "no-home" CLIR test; the 524-query
  release adds Spanish gold documents, so Spanish is no longer purely query-side.)

## Seven headline results

1. **Cross-lingual recall trails same-language recall in every model.** Best CLIR@10 =
   **0.54** (embeddinggemma); the
   home-advantage (MoLIR−CLIR) reaches **+0.60** for the most
   biased model — a same-language copy is far easier than its foreign twin. *(Round 1)*
2. **Retrieval direction is anisotropic.** Hardest direction
   en→de (R@10=0.16);
   English is the easiest *target* language; the most asymmetric pair is
   de↔en (gap +0.19).
   Retrieval is not a symmetric similarity. *(Round 2)*
3. **Machine translation of the question is *not* the problem.** Controlling for the patent, the paired
   human−MT difference in cross-lingual reach is
   -0.045
   (p=0.02) — statistically insignificant. The project's
   "MT-is-fine-for-the-question" assumption holds. *(Round 3)*
4. **Foreign twins are often buried or lost.** Pooled mate-hit@10 = 0.41;
   **17%** of (query, model) pairs never surface a
   foreign twin even in the **top-1000**. Best twin-finder: embeddinggemma
   (median first-foreign rank 4). *(Round 4)*
5. **Same question, different languages → different rankings.** Cross-lingual RBO ceiling is only
   **0.18**, and r(home-advantage, RBO) =
   **-0.89** — bias drives inconsistency. *(Round 5)*
6. **Language collapse is the mechanism.** Low-resource query languages over-fetch their own language
   up to **17×** the corpus base
   rate; same-language noise out-ranks the gold on
   **58%** of queries; and across models
   r(over-representation, CLIR@10) = -0.58. *(Rounds 6-7)*
7. **It's a separability problem, and complementarity is the lever.** r(cross-language AUC, CLIR@10) =
   **+0.98**: foreign golds are under-scored, not just
   mis-ranked. No single model finds every twin — the oracle reaches CLIR@10 =
   **0.66** (headroom +0.12),
   though plain RRF fusion does **not** beat the dominant model. *(Rounds 8-9)*

## Verdict — CLIR-MRS leaderboard

CLIR-MRS = capability {accuracy, CLIR, separability} × (0.5 + 0.5·robustness {consistency,
MT-robust, language-parity}). Capability is the spine; robustness modulates ±50%.

| rank | model | R@10 | CLIR@10 | home adv | mate-MRR | sep-AUC | CLIR-MRS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `embeddinggemma` | 0.574 | 0.541 | +0.17 | 0.361 | 0.91 | 0.91 |
| 2 | `bge-m3` | 0.509 | 0.464 | +0.22 | 0.335 | 0.88 | 0.81 |
| 3 | `nomic-v2-moe` | 0.467 | 0.409 | +0.29 | 0.292 | 0.88 | 0.73 |
| 4 | `qwen3-0.6B` | 0.496 | 0.451 | +0.22 | 0.313 | 0.89 | 0.71 |
| 5 | `granite-278m` | 0.415 | 0.390 | +0.12 | 0.275 | 0.88 | 0.60 |
| 6 | `SapBERT` | 0.238 | 0.197 | +0.19 | 0.152 | 0.78 | 0.41 |
| 7 | `LaBSE` | 0.301 | 0.271 | +0.15 | 0.227 | 0.82 | 0.40 |
| 8 | `e5-large-instruct` | 0.218 | 0.095 | +0.60 | 0.060 | 0.71 | 0.25 |
| 9 | `gte-base` | 0.005 | 0.000 | +0.02 | 0.000 | 0.48 | 0.00 |

**Most robust multilingual retriever: `embeddinggemma`** (CLIR-MRS = 0.91
[0.83, 0.90]).

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
