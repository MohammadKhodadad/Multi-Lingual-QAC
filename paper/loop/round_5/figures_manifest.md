# Figures manifest (round 5)

Every `\includegraphics` in `paper/main.tex` and the real source path it maps to.
All targets verified present on disk under `paper/figures/` (`\graphicspath{{figures/}}`).

**Round-5 float changes:** 29 floats (27 fig + 2 tab) -> 26 floats (23 fig + 3 tab).
- CUT: cp_fig14_robustness_radar.png, ag_fig10_robustness_radar.png (the two radars; -2 figures).
- MERGE: cp_fig06+cp_fig07 -> cp_fig06_07_mate.png (-1); cp_fig09+cp_fig10 -> cp_fig09_10_collapse.png (-1).
- ADD: Table `tab:robust` (appendix robustness table; +1 table, OUTSIDE the 8-page body budget).
- Net in-body floats = 25 (the appendix table sits outside the body), matching the story target.

| label | basename in main.tex | exists | source / notes |
|---|---|---|---|
| fig:teaser | cp_fig01_clir_leaderboard.png | yes | chem-patents key findings Fig.1 |
| fig:cp_deg | cp_fig20_degeneracy_gap.png | yes | extra_two_tax_degeneracy |
| fig:cp_home | cp_fig02_home_advantage.png | yes | chem-patents key findings Fig.2 |
| fig:cp_dir | cp_fig03_directional_clir_matrix.png | yes | chem-patents key findings Fig.3 |
| fig:cost_frontier | cp_fig18_cost_frontier.png | yes | extra_cost_frontier (caption now carries tau-band, C4) |
| fig:cp_mt | cp_fig05_mt_penalty.png | yes | chem-patents key findings Fig.5 |
| fig:ari | cp_fig22_ari_decomposition.png | yes | extra_ari_decomposition (caption R2 reconciled "all nine"; C2 tie) |
| fig:rrc_budget | cp_fig19_rrc_budget.png | yes | extra_rrc_budget_frontier |
| fig:cp_mate | **cp_fig06_07_mate.png** | yes | **MERGE 1** = cp_fig06_mate_retrieval + cp_fig07_first_foreign_rank (3062x754) |
| fig:two_tax | cp_fig21_two_tax.png | yes | extra_two_tax_degeneracy |
| fig:ag_rbo | ag_fig1_cross_lingual_rbo.png | yes | alias-graph key findings Fig.1 |
| fig:ag_conf | ag_fig2_confusion_both_lenses.png | yes | alias-graph key findings Fig.2 |
| fig:ag_attr | ag_fig5_universal_attractors.png | yes | alias-graph key findings Fig.5 |
| fig:cp_ribbon | cp_fig17_aggregation_ribbon.png | yes | extra_aggregation_invariance (covers cut radars) |
| fig:ag_joint | ag_fig12_joint_failure_modes.png | yes | extra_joint_failure |
| fig:ag_avail | ag_fig11_availability_residual.png | yes | extra_availability_residual |
| fig:cp_collapse | **cp_fig09_10_collapse.png** | yes | **MERGE 2** = cp_fig09_language_collapse + cp_fig10_distractor_language (2705x754) |
| fig:ag_qtype | ag_fig6_question_type_effect.png | yes | alias-graph key findings Fig.6 |
| fig:cp_sep | cp_fig11_separability.png | yes | chem-patents key findings Fig.11 (caption C1: sign-stability + wide CI) |
| fig:ag_sep | ag_fig8_confusion_is_separability.png | yes | alias-graph key findings Fig.8 |
| fig:per_route | cp_fig23_per_route_frontier.png | yes | extra_per_route_frontier |
| fig:cp_ens | cp_fig12_ensemble_headroom.png | yes | chem-patents key findings Fig.12 |
| fig:ag_ens | ag_fig7_ensemble_headroom.png | yes | alias-graph key findings Fig.7 |

## Tables
| label | location | source |
|---|---|---|
| tab:cp_board | §6.3 | chem-patents headline_numbers.csv |
| tab:ag_board | §6.3 | alias-graph headline_numbers.csv |
| **tab:robust** | Appendix (NEW, round 5) | extra_robustness_appendix/{robustness_table.csv, summary.json} |

## Retired source panels (still on disk, no longer referenced)
- cp_fig06_mate_retrieval.png, cp_fig07_first_foreign_rank.png (superseded by cp_fig06_07_mate.png)
- cp_fig09_language_collapse.png, cp_fig10_distractor_language.png (superseded by cp_fig09_10_collapse.png)
- cp_fig14_robustness_radar.png, ag_fig10_robustness_radar.png (cut; content carried by Tables 1-2 + cp_fig17)
