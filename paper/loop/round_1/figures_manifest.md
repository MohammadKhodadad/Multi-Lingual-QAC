# Figures manifest — round 1

Every `\includegraphics` used in `paper/main.tex`, the basename referenced
(via `\graphicspath{{figures/}}`), the file copied into `paper/figures/`, and
the real source path it maps to. All targets confirmed to exist on disk
(self-lint passed: 20/20 OK).

## Cross-lingual (chem-patents) figures — prefix `cp_`

| main.tex basename | paper/figures/ copy | source path |
| --- | --- | --- |
| cp_fig01_clir_leaderboard.png | paper/figures/cp_fig01_clir_leaderboard.png | reports/runs/chem_patents/key_findings/figures/fig01_clir_leaderboard.png |
| cp_fig02_home_advantage.png | paper/figures/cp_fig02_home_advantage.png | reports/runs/chem_patents/key_findings/figures/fig02_home_advantage.png |
| cp_fig03_directional_clir_matrix.png | paper/figures/cp_fig03_directional_clir_matrix.png | reports/runs/chem_patents/key_findings/figures/fig03_directional_clir_matrix.png |
| cp_fig05_mt_penalty.png | paper/figures/cp_fig05_mt_penalty.png | reports/runs/chem_patents/key_findings/figures/fig05_mt_penalty.png |
| cp_fig06_mate_retrieval.png | paper/figures/cp_fig06_mate_retrieval.png | reports/runs/chem_patents/key_findings/figures/fig06_mate_retrieval.png |
| cp_fig07_first_foreign_rank.png | paper/figures/cp_fig07_first_foreign_rank.png | reports/runs/chem_patents/key_findings/figures/fig07_first_foreign_rank.png |
| cp_fig08_consistency_vs_bias.png | paper/figures/cp_fig08_consistency_vs_bias.png | reports/runs/chem_patents/key_findings/figures/fig08_consistency_vs_bias.png |
| cp_fig09_language_collapse.png | paper/figures/cp_fig09_language_collapse.png | reports/runs/chem_patents/key_findings/figures/fig09_language_collapse.png |
| cp_fig10_distractor_language.png | paper/figures/cp_fig10_distractor_language.png | reports/runs/chem_patents/key_findings/figures/fig10_distractor_language.png |
| cp_fig11_separability.png | paper/figures/cp_fig11_separability.png | reports/runs/chem_patents/key_findings/figures/fig11_separability.png |
| cp_fig12_ensemble_headroom.png | paper/figures/cp_fig12_ensemble_headroom.png | reports/runs/chem_patents/key_findings/figures/fig12_ensemble_headroom.png |
| cp_fig14_robustness_radar.png | paper/figures/cp_fig14_robustness_radar.png | reports/runs/chem_patents/key_findings/figures/fig14_robustness_radar.png |

## Alias-graph figures — prefix `ag_`

| main.tex basename | paper/figures/ copy | source path |
| --- | --- | --- |
| ag_fig1_cross_lingual_rbo.png | paper/figures/ag_fig1_cross_lingual_rbo.png | reports/runs/alias_graph/key_findings/figures/fig1_cross_lingual_rbo.png |
| ag_fig2_confusion_both_lenses.png | paper/figures/ag_fig2_confusion_both_lenses.png | reports/runs/alias_graph/key_findings/figures/fig2_confusion_both_lenses.png |
| ag_fig4_bias_drives_inconsistency.png | paper/figures/ag_fig4_bias_drives_inconsistency.png | reports/runs/alias_graph/key_findings/figures/fig4_bias_drives_inconsistency.png |
| ag_fig5_universal_attractors.png | paper/figures/ag_fig5_universal_attractors.png | reports/runs/alias_graph/key_findings/figures/fig5_universal_attractors.png |
| ag_fig6_question_type_effect.png | paper/figures/ag_fig6_question_type_effect.png | reports/runs/alias_graph/key_findings/figures/fig6_question_type_effect.png |
| ag_fig7_ensemble_headroom.png | paper/figures/ag_fig7_ensemble_headroom.png | reports/runs/alias_graph/key_findings/figures/fig7_ensemble_headroom.png |
| ag_fig8_confusion_is_separability.png | paper/figures/ag_fig8_confusion_is_separability.png | reports/runs/alias_graph/key_findings/figures/fig8_confusion_is_separability.png |
| ag_fig10_robustness_radar.png | paper/figures/ag_fig10_robustness_radar.png | reports/runs/alias_graph/key_findings/figures/fig10_robustness_radar.png |

## Copied but not yet referenced (available for later rounds)

These were copied into `paper/figures/` for convenience but are NOT currently
`\includegraphics`'d (to keep round-1 figure count manageable). They map 1:1 to
their source files with the same prefix convention:
- `cp_fig04_clir_direction_asymmetry.png` (direction asymmetry; described in
  text, figure deferred)
- `cp_fig13_clir_mrs_leaderboard.png` (CLIR-MRS leaderboard bar chart; the
  leaderboard is shown as Table~\ref{tab:cp_board} instead)
- `ag_fig3_home_advantage.png` (alias home advantage; described in Analysis text)
- `ag_fig9_robustness_leaderboard.png` (alias MRS bar chart; shown as
  Table~\ref{tab:ag_board} instead)

## Tables (not figures, numbers traced)

- Table `tab:cp_board` (cross-lingual leaderboard): every cell from
  reports/runs/chem_patents/key_findings/headline_numbers.csv and the
  EXECUTIVE_SUMMARY Verdict table.
- Table `tab:ag_board` (alias-graph leaderboard): every cell from
  reports/runs/alias_graph/key_findings/headline_numbers.csv and the
  EXECUTIVE_SUMMARY headline table.
