# Figures manifest — round 2

Every `\includegraphics` used in `paper/main.tex` (round 2), the basename
referenced via `\graphicspath{{figures/}}`, the `paper/figures/` copy, and the
real source path. Self-lint passed: **23/23 includegraphics targets exist on
disk**; all `\cite` keys resolve in `custom.bib`; all referenced `\label`s
defined; `\begin`/`\end` balanced.

## Cross-lingual (chem-patents) figures — prefix `cp_`

| main.tex basename | paper/figures/ copy | source path |
| --- | --- | --- |
| cp_fig01_clir_leaderboard.png | paper/figures/cp_fig01_clir_leaderboard.png | reports/runs/chem_patents/key_findings/figures/fig01_clir_leaderboard.png |
| cp_fig02_home_advantage.png | paper/figures/cp_fig02_home_advantage.png | reports/runs/chem_patents/key_findings/figures/fig02_home_advantage.png |
| cp_fig03_directional_clir_matrix.png | paper/figures/cp_fig03_directional_clir_matrix.png | reports/runs/chem_patents/key_findings/figures/fig03_directional_clir_matrix.png |
| cp_fig05_mt_penalty.png | paper/figures/cp_fig05_mt_penalty.png | reports/runs/chem_patents/key_findings/figures/fig05_mt_penalty.png |
| cp_fig06_mate_retrieval.png | paper/figures/cp_fig06_mate_retrieval.png | reports/runs/chem_patents/key_findings/figures/fig06_mate_retrieval.png |
| cp_fig07_first_foreign_rank.png | paper/figures/cp_fig07_first_foreign_rank.png | reports/runs/chem_patents/key_findings/figures/fig07_first_foreign_rank.png |
| cp_fig09_language_collapse.png | paper/figures/cp_fig09_language_collapse.png | reports/runs/chem_patents/key_findings/figures/fig09_language_collapse.png |
| cp_fig10_distractor_language.png | paper/figures/cp_fig10_distractor_language.png | reports/runs/chem_patents/key_findings/figures/fig10_distractor_language.png |
| cp_fig11_separability.png | paper/figures/cp_fig11_separability.png | reports/runs/chem_patents/key_findings/figures/fig11_separability.png |
| cp_fig12_ensemble_headroom.png | paper/figures/cp_fig12_ensemble_headroom.png | reports/runs/chem_patents/key_findings/figures/fig12_ensemble_headroom.png |
| cp_fig14_robustness_radar.png | paper/figures/cp_fig14_robustness_radar.png | reports/runs/chem_patents/key_findings/figures/fig14_robustness_radar.png |
| **cp_fig15_xrc_reading_cost.png** (NEW r2) | paper/figures/cp_fig15_xrc_reading_cost.png | reports/runs/chem_patents/experimental_plots/extra_xrc_reading_cost/**xrc_vs_clir.png** |
| **cp_fig16_rrc_reranker_ceiling.png** (NEW r2) | paper/figures/cp_fig16_rrc_reranker_ceiling.png | reports/runs/chem_patents/experimental_plots/extra_xrc_reading_cost/**rrc_ceiling.png** |
| **cp_fig17_aggregation_ribbon.png** (NEW r2) | paper/figures/cp_fig17_aggregation_ribbon.png | reports/runs/chem_patents/experimental_plots/extra_aggregation_invariance/**aggregation_ribbon.png** |

## Alias-graph figures — prefix `ag_`

| main.tex basename | paper/figures/ copy | source path |
| --- | --- | --- |
| ag_fig1_cross_lingual_rbo.png | paper/figures/ag_fig1_cross_lingual_rbo.png | reports/runs/alias_graph/key_findings/figures/fig1_cross_lingual_rbo.png |
| ag_fig2_confusion_both_lenses.png | paper/figures/ag_fig2_confusion_both_lenses.png | reports/runs/alias_graph/key_findings/figures/fig2_confusion_both_lenses.png |
| ag_fig5_universal_attractors.png | paper/figures/ag_fig5_universal_attractors.png | reports/runs/alias_graph/key_findings/figures/fig5_universal_attractors.png |
| ag_fig6_question_type_effect.png | paper/figures/ag_fig6_question_type_effect.png | reports/runs/alias_graph/key_findings/figures/fig6_question_type_effect.png |
| ag_fig7_ensemble_headroom.png | paper/figures/ag_fig7_ensemble_headroom.png | reports/runs/alias_graph/key_findings/figures/fig7_ensemble_headroom.png |
| ag_fig8_confusion_is_separability.png | paper/figures/ag_fig8_confusion_is_separability.png | reports/runs/alias_graph/key_findings/figures/fig8_confusion_is_separability.png |
| ag_fig10_robustness_radar.png | paper/figures/ag_fig10_robustness_radar.png | reports/runs/alias_graph/key_findings/figures/fig10_robustness_radar.png |
| **ag_fig11_availability_residual.png** (NEW r2) | paper/figures/ag_fig11_availability_residual.png | reports/runs/alias_graph/experimental_plots/extra_availability_residual/**availability_residual.png** |
| **ag_fig12_joint_failure_modes.png** (NEW r2) | paper/figures/ag_fig12_joint_failure_modes.png | reports/runs/alias_graph/experimental_plots/extra_joint_failure/**joint_failure_modes.png** |

## Changes from round 1

**Added (5 new figures, all from this round's `extra_*` analyses):**
cp_fig15 (XRC reading cost), cp_fig16 (RRC re-ranker ceiling), cp_fig17
(aggregation ribbon), ag_fig11 (availability residual), ag_fig12 (joint failure
modes).

**Removed from the included set (folded into figureless prose):**
- `cp_fig08_consistency_vs_bias.png` — the home-adv↔RBO correlation it plots is
  now demoted to a *descriptive, non-robust* observation (fragile on n=7), so it
  is stated in one hedged sentence without a figure.
- `ag_fig4_bias_drives_inconsistency.png` — same reason (same-language-share↔RBO
  correlation, fragile). Both copies remain in `paper/figures/` but are no longer
  `\includegraphics`'d.

## Numbers added this round — every value traced to a file under reports/

| value | source file |
| --- | --- |
| XRC50 = 3.5× (egemma, D50 2→7); granite 1.25, nomic 11.5, e5 97.75; gte degenerate | `chem_patents/.../extra_xrc_reading_cost/summary.json` + `xrc_per_model.csv` |
| RRC@100 0.7445, RRC@1000 0.9416, lost@1000 5.84%; e5 loses 37.2% | `chem_patents/.../extra_xrc_reading_cost/rrc_per_model.csv` |
| aggregation rank range [1,4]; rank-1 under 2/4 schemes; gte axes-won contamination | `chem_patents/.../extra_aggregation_invariance/aggregation_ranks.csv` + `summary.json` |
| hub scores fr 0.375 ≈ en 0.367 > zh 0.350 > de 0.309; en→de 0.12; de↔zh +0.23; corpus en 46%/zh 0.4% | `chem_patents/.../extra_directional_hub/summary.json` + `hub_scores.csv` |
| availability slope −0.57 (Pearson −0.87, R²=0.76, n=5); zh +0.47 largest; mean +0.32 | `alias_graph/.../extra_availability_residual/summary.json` + `availability_regression.csv` |
| auc_cross~clir +0.958/Spearman 0.964 on n=7 (robust); over-rep~clir & home-adv~rbo fragile | `chem_patents/.../extra_correlation_robustness/correlation_robustness.csv` |
| joint failure: same-lang sibling 114/257=44.4%; siblings 79.4%; same-lang 55.6% | `alias_graph/.../extra_joint_failure/summary.json` |
| universal-blind 16/132=12%; 14/16 structure; fr5/zh4/de3/es3/en1 | `alias_graph/.../extra_joint_failure/universal_blind_profile.csv` |
| confusion severity: sibling 18.1% vs parent 6.2% (2.9×); egemma 6.1% vs 1.5% | `alias_graph/.../extra_confusion_severity/severity_split.csv` + `summary.json` |
