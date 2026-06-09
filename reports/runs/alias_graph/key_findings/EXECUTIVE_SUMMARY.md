# Multilingual & alias-graph retrieval: key findings

*Benchmark:* `MehdiAstaraki/multi-lingual-qac-alias-graph` — 132 chemistry-patent questions
(24 ChEBI compounds × 5 languages: en/de/fr/es/zh), 9 multilingual embedding models, ranked against a
large shared patent corpus. Every compound ships with an **alias graph** of chemically-confusable
neighbours (siblings/parents) as hard negatives. Two relevance lenses are reported co-equally:
**concept-level** (any patent about the compound, ~109 gold/query) and **per-publication** (the query's
own source patent + translations, ~2.4 gold/query). Full derivation in
`../experimental_plots/FINDINGS.md`; code in `../experimental_codes/`.

## The two headline questions

**1. "Ask the same compound in five languages — do you get the same ranked patents back?" → No.**
Even the best model reaches a cross-lingual RBO of only **0.39** (1.0 = identical
rankings); two language variants of the *same* question share fewer than 4 of their top-10 documents.
The embedding space is far from language-agnostic, and Chinese is the consistent odd-one-out (Fig 1, 4).

**2. "How often does a confusable wrong compound beat the right one?" → Often, and more in some
languages.** A chemically-similar look-alike out-ranks every gold patent on **14%–78%**
of queries (publication lens) depending on the model; it is worst for German/Chinese and driven almost
entirely by **sibling** compounds, not broader parent classes (Fig 2). The repeat offenders are a small
set of universal attractors — **polypeptide**, **methyl**, **ethene**, **hydroxide**, **dioxygen** (Fig 5).

## What's really going on (mechanism)

- **Home advantage + availability confound (Fig 3, 4).** Every language retrieves its own-language gold
  far better (0.63–0.82) than foreign-language gold (0.35–0.47). Because en/fr source patents dominate,
  42% of an English query's gold is reachable in-English vs only 8–10% for de/es/zh — so English's
  apparent lead is mostly *where the gold lives*, not encoder skill. Retriever language bias is what
  drives the cross-lingual inconsistency (Pearson r = **-0.87** between
  same-language share and RBO).
- **The confuser is chemistry, whether you fall in is language.** The most-threatening look-alike is the
  same compound across all 5 languages (a shared chemical attractor), but *whether* it actually beats
  the gold is language-specific — only ~19% of confused concepts are confused in all their languages.
- **How you ask matters.** `structure`-style questions are the trap (Recall@10 0.26, confusion 51%) vs
  `role` questions (0.60, 25%) — structure descriptions are exactly what siblings share (Fig 6). A
  language-independent **formula token** (H2S, CO2) measurably helps (p < 0.01), even in Chinese.
- **Confusion is a separability failure (Fig 8).** Behind every confusion, the cosine score barely
  separates gold from sibling: AUC(gold>look-alike) = **0.55** for
  confused queries vs **0.70** otherwise.

## What to do about it

- **Ensemble / route by language (Fig 7).** Failures are substantially complementary: the oracle (any of
  9 models) hits **88%** vs **76%**
  for the best single model — and Chinese has the *largest* recoverable headroom. A ~12% universal-blind
  core remains (the methyl/sulfide traps) and needs chemistry-aware help, not just more encoders.
- **Report multilingual robustness, not just average recall.** Folding accuracy + consistency +
  confusion-robustness + language-parity + separability into one **MRS** (Fig 9–10) keeps
  `embeddinggemma` on top but reshuffles the mid-field, rewarding language-balanced encoders that mean
  recall alone would under-rate. Headline metric recommendation: **publication-lens Recall@10 + MRS**;
  never read concept-lens Recall@10 as a quality score (it is capped by ~109 positives).

## Headline numbers (per model)

| model | pub_recall10 | cross_lingual_rbo | confusion_publication | separability_auc | MRS | MRS_ci | accuracy_rank | MRS_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| embeddinggemma | 0.67 | 0.387 | 0.144 | 0.695 | 0.991 | [0.86,1.00] | 1 | 1 |
| qwen3-0.6B | 0.641 | 0.349 | 0.227 | 0.669 | 0.902 | [0.77,0.96] | 2 | 2 |
| bge-m3 | 0.604 | 0.288 | 0.237 | 0.657 | 0.813 | [0.69,0.88] | 3 | 3 |
| LaBSE | 0.511 | 0.314 | 0.295 | 0.655 | 0.779 | [0.63,0.85] | 6 | 4 |
| granite-278m | 0.529 | 0.289 | 0.318 | 0.655 | 0.757 | [0.61,0.84] | 5 | 5 |
| nomic-v2-moe | 0.58 | 0.239 | 0.215 | 0.649 | 0.727 | [0.60,0.83] | 4 | 6 |
| SapBERT | 0.352 | 0.142 | 0.424 | 0.617 | 0.494 | [0.32,0.60] | 7 | 7 |
| e5-large-instruct | 0.237 | 0.021 | 0.481 | 0.61 | 0.239 | [0.17,0.36] | 8 | 8 |
| gte-base | 0.047 | 0.003 | 0.782 | 0.564 | 0.0 | [0.00,0.13] | 9 | 9 |

*pub_recall10 = per-publication Recall@10; cross_lingual_rbo from Round 1; confusion = publication-lens
rate; separability = AUC(gold>look-alike); MRS = min-max-normalised mean of the five axes with 95%
bootstrap CI. Full per-language / per-lens tables and CIs in `headline_numbers.csv` and the round CSVs.*

## Figures
1. `fig1_cross_lingual_rbo.png` — same compound, 5 languages: ranking agreement (Q1).
2. `fig2_confusion_both_lenses.png` — confusion rate, model × language, both lenses (Q2).
3. `fig3_home_advantage.png` — same- vs cross-language gold recall per language.
4. `fig4_bias_drives_inconsistency.png` — language bias ⇒ cross-lingual inconsistency.
5. `fig5_universal_attractors.png` — look-alikes that most often steal the top rank.
6. `fig6_question_type_effect.png` — structure questions are the trap.
7. `fig7_ensemble_headroom.png` — oracle vs best single model, per language.
8. `fig8_confusion_is_separability.png` — confusion = gold/sibling score collapse.
9. `fig9_robustness_leaderboard.png` — Multilingual Robustness Score (95% CI).
10. `fig10_robustness_radar.png` — where each top model wins across the five axes.
