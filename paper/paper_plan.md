# Paper Plan: 6-Page EMNLP Industry Track Draft

This is the writing blueprint for `short_main.tex`. Do not edit the existing
`main.tex`; treat it as the full/archive draft.

## Core Story

Multilingual chemistry-patent search lacks a clean public evaluation set for
cross-lingual retrieval. We built one with an agentic QAC pipeline, validated
its grading against human annotation, and used it to show where retrieval models
fail when the relevant patent is in another language or a chemically close
document competes for the top rank.

Write this as an **Industry Track evaluation-framework paper**, not as a model
leaderboard paper. The practical lesson is that patent-search teams should not
trust average recall alone: they need language-aware qrels, same-vs-cross
retrieval splits, and hard negatives grounded in chemistry.

## Claims To Push

1. **Dataset gap:** To the best of our knowledge, no public benchmark combines
   multilingual chemistry-patent QAC, cross-language qrels, human/LLM validation,
   and chemistry-aware hard negatives.
2. **Practical pipeline:** The data is built by a reproducible LLM-agentic
   workflow: generate candidates, grade faithfulness/quality, keep scored rows,
   and compare the LLM grader with a human annotator.
3. **Evaluation lens:** Metrics must ask where the retrieved document sits:
   same language, another relevant language, unrelated same-language text, or a
   confusable chemical neighbor.
4. **Stress testing:** Alias-graph hard negatives are implemented and should be
   a main result. Code-switch/noise infrastructure exists, but unless we run
   complete results it should remain future/partial.

## Contributions

Use a compact contribution list:

1. A multilingual chemistry-patent QAC benchmark for cross-lingual retrieval.
2. An agentic generation and validation pipeline with human-vs-LLM grading
   analysis.
3. A language-aware retrieval evaluation suite that separates same-language from
   cross-language retrieval and exposes irrelevant same-language distractions.
4. A chemistry-confusability analysis using ChEBI alias-graph hard negatives.

## Section Plan

### Abstract

One paragraph: problem, dataset, generation/validation workflow, language-aware
evaluation, alias-graph stress test, and the lesson that average recall hides
cross-lingual failure.

### Introduction

Open with the real workflow: chemistry patent search across languages. Explain
why ordinary multilingual retrieval benchmarks are insufficient: they do not
control patent-language variants, chemistry concepts, or confusable compounds.
Then introduce the benchmark, QAC pipeline, human/LLM validation, and
contributions.

Key sentence:

> The problem is not only whether a multilingual encoder retrieves a relevant
> patent, but whether it retrieves the right patent when the query language and
> document language differ and chemically similar documents compete for the top
> ranks.

### QAC Generation Pipeline

Describe the system, step by step:

1. Extract/filter chemistry patent text.
2. Build context from title, abstract, and available claim text.
3. Generate multiple QAC candidates per document/language/mode.
4. Grade faithfulness and retrieval quality.
5. Keep best-scored rows with audit metadata.
6. Validate the LLM grading against human annotation.

Implementation anchors: `multilingual_qa.py`, `balanced_multilingual_qa.py`,
and `scripts/analyze_human_vs_llm.py`.

### Benchmark Construction

Explain the released objects: `corpus`, `queries`, `qrels`, and `qac`.
`publication_number` links language variants of the same patent, letting us
separate same-language and cross-language positives. Then introduce the
alias-graph benchmark: ChEBI concepts, multilingual aliases, gold documents, and
taxonomic-neighbor hard negatives.

Implementation anchors: `hf_upload.py`, `alias_graph/builder.py`,
`concept_qa.py`, and `alias_graph/hf_export.py`.

### Evaluation Setup

Evaluate multilingual embedding models as drop-in dense retrievers. Report
standard metrics only as context; the main analysis splits results by query
language, corpus language, same-vs-cross relevant docs, same-language irrelevant
share, language-pair matrix, and alias-graph hard-negative outranking.

Implementation anchors: `mteb/evaluation.py`, `mteb/question_analysis.py`,
`alias_graph/confusion_analysis.py`, and `alias_graph/retrieval_results.py`.

### Results

Keep the results section narrow:

1. QAC quality and human-vs-LLM grader agreement.
2. Cross-language retrieval is harder than same-language retrieval.
3. Same-language distractions can hide cross-language failure.
4. Alias-graph hard negatives show chemically close wrong documents can outrank
   the correct concept/document.

Fill exact numbers only after checking the artifacts.

### Limitations

Chemistry patents may not generalize to all retrieval domains. Patent
publication variants help but do not guarantee full claim-level equivalence.
Human validation is limited in size. LLM graders are useful but not substitutes
for expert patent review. Code-switch/noise analysis is infrastructure until
final results are run.

### Conclusion

Close on the reusable lesson: specialized cross-lingual retrieval benchmarks
can be built with agentic QAC generation plus human/LLM validation, but they
must evaluate language position and hard negatives rather than average recall
alone.

## Figure Inventory Already Available

Figures in `paper/figures/` and referenced by the existing `main.tex`:

### Best Candidates For The 6-Page Body

- `cp_fig01_clir_leaderboard.png`: cross-lingual vs same-language retrieval
  leaderboard; strongest visual for "average recall hides collapse."
- `cp_fig02_home_advantage.png`: same-language home advantage by model.
- `cp_fig03_directional_clir_matrix.png`: query-language to document-language
  matrix; good if we emphasize language-pair position.
- `cp_fig09_10_collapse.png`: language over-representation plus distractor
  language; good for same-language distraction.
- `ag_fig2_confusion_both_lenses.png`: alias-graph confusion rate; strongest
  visual for chemically confusable wrong documents.
- `ag_fig1_cross_lingual_rbo.png`: same compound in five languages returns
  different rankings; useful for consistency.
- `ag_fig12_joint_failure_modes.png`: modal failure is same-language sibling;
  good bridge between language bias and chemical confusability.
- `ag_fig6_question_type_effect.png`: structure questions are harder than role
  questions; useful if we discuss what makes QAC difficult.

### Secondary / Appendix Candidates

- `cp_fig05_mt_penalty.png`: human vs machine-translated query penalty.
- `cp_fig06_07_mate.png`, `cp_fig07_first_foreign_rank.png`,
  `cp_fig06_mate_retrieval.png`: foreign-twin depth / mate retrieval.
- `cp_fig10_distractor_language.png`, `cp_fig09_language_collapse.png`:
  single-panel language-collapse variants.
- `cp_fig11_separability.png`: embedding separability mechanism.
- `cp_fig12_ensemble_headroom.png`: ensemble/router headroom.
- `cp_fig17_aggregation_ribbon.png`: ranking sensitivity to aggregation.
- `cp_fig18_cost_frontier.png`, `cp_fig19_rrc_budget.png`,
  `cp_fig22_ari_decomposition.png`, `cp_fig23_per_route_frontier.png`: strong
  but probably too detailed for this shorter dataset/pipeline paper.
- `ag_fig5_universal_attractors.png`, `ag_fig8_confusion_is_separability.png`,
  `ag_fig11_availability_residual.png`, `ag_fig7_ensemble_headroom.png`:
  useful appendix/supporting alias-graph analysis.
- `cp_fig14_robustness_radar.png`, `cp_fig13_clir_mrs_leaderboard.png`,
  `ag_fig10_robustness_radar.png`, `ag_fig9_robustness_leaderboard.png`:
  avoid as main figures unless we decide the paper should be leaderboard-heavy.

### Missing Figure To Create

- Pipeline overview: corpus extraction → QAC generation → LLM grading → human
  validation → HF/MTEB export → language-aware retrieval analysis. This does not
  appear in the current figure folder and should be Figure 1 if possible.

## Recommended Body Figure/Table Budget

For six pages, use at most 3 figures/tables:

1. **Figure 1:** new pipeline overview.
2. **Table 1:** dataset statistics and human/LLM validation summary.
3. **Figure 2:** either `cp_fig01_clir_leaderboard.png` or
   `cp_fig09_10_collapse.png`.
4. **Figure 3 or Table 2:** `ag_fig2_confusion_both_lenses.png` or a compact
   alias-graph hard-negative table.

Everything else goes to appendix or is omitted.

## Evidence Checklist

- [ ] Exact chem-patents corpus/QAC row counts and languages.
- [ ] Human annotation sample size and file names.
- [ ] `reports/human_eval/summary.json` numbers or rerun
      `scripts/analyze_human_vs_llm.py`.
- [ ] Alias-graph query/concept/document counts.
- [ ] Alias-graph hard-negative outrank/confusion numbers.
- [ ] Decide whether code-switch has final results; otherwise keep as future.
- [ ] Pick final 2--3 body figures and appendix figures.

## Relationship To Existing `main.tex`

Reuse only what serves this cleaner Industry Track story:

- Keep: dataset framing, cross-lingual recall split, language-position metrics,
  alias-graph confusion, human/LLM validation if verified.
- Compress or omit: XRC/RRC/ARI/cost-frontier/per-route details unless the paper
  has extra space.
- Avoid: making the short paper a figure-heavy leaderboard.

The target reading experience:

> We needed this benchmark, built it carefully, validated the QAC pipeline, and
> found retrieval failures that ordinary multilingual evaluation misses.

