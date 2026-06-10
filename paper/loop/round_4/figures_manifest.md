# Figures manifest (round 4)

Every `\includegraphics` in `paper/main.tex` maps to a real file in
`paper/figures/` (referenced by basename via `\graphicspath{{figures/}}`).
All 27 targets verified present on disk. The two figures NEW to round 4 are
marked **[NEW r4]**; all others are unchanged from round 3.

| label | basename (paper/figures/) | source under reports/runs/ | §ref |
|---|---|---|---|
| fig:teaser | cp_fig01_clir_leaderboard.png | chem_patents key_findings Fig.1 | Intro |
| fig:cp_home | cp_fig02_home_advantage.png | chem_patents key_findings Fig.2 | §6.1 |
| fig:cp_dir | cp_fig03_directional_clir_matrix.png | chem_patents key_findings Fig.3 + extra_directional_hub | §6.1 |
| fig:cost_frontier | cp_fig18_cost_frontier.png | chem_patents experimental_plots/extra_cost_frontier | §6.1 |
| fig:cp_mt | cp_fig05_mt_penalty.png | chem_patents key_findings Fig.5 | §6.1 |
| fig:rrc_budget | cp_fig19_rrc_budget.png | chem_patents experimental_plots/extra_rrc_budget_frontier | §6.1 |
| **fig:ari** | **cp_fig22_ari_decomposition.png** **[NEW r4]** | chem_patents experimental_plots/extra_ari_decomposition (ari_decomposition.csv + summary.json) | §6.1 (def §4, back-ref §7, §8, Limitations) |
| fig:cp_mate | cp_fig06_mate_retrieval.png | chem_patents key_findings Fig.6 | §6.1 |
| fig:cp_rank | cp_fig07_first_foreign_rank.png | chem_patents key_findings Fig.7 | §6.1 |
| fig:two_tax | cp_fig21_two_tax.png | chem_patents experimental_plots/extra_two_tax_degeneracy | §6.2 |
| fig:cp_deg | cp_fig20_degeneracy_gap.png | chem_patents experimental_plots/extra_two_tax_degeneracy | §4 |
| fig:ag_rbo | ag_fig1_cross_lingual_rbo.png | alias_graph key_findings Fig.1 | §6.2 |
| fig:ag_conf | ag_fig2_confusion_both_lenses.png | alias_graph key_findings Fig.2 | §6.2 |
| fig:ag_attr | ag_fig5_universal_attractors.png | alias_graph key_findings Fig.5 | §6.2 |
| fig:cp_ribbon | cp_fig17_aggregation_ribbon.png | chem_patents experimental_plots/extra_aggregation_invariance | §6.3 |
| fig:cp_radar | cp_fig14_robustness_radar.png | chem_patents key_findings Fig.14 | §6.3 |
| fig:ag_radar | ag_fig10_robustness_radar.png | alias_graph key_findings Fig.10 | §6.3 |
| fig:ag_joint | ag_fig12_joint_failure_modes.png | alias_graph experimental_plots/extra_joint_failure | §7 |
| fig:ag_avail | ag_fig11_availability_residual.png | alias_graph experimental_plots/extra_availability_residual | §7 |
| fig:cp_collapse | cp_fig09_language_collapse.png | chem_patents key_findings Fig.9 | §7 |
| fig:cp_distractor | cp_fig10_distractor_language.png | chem_patents key_findings Fig.10 | §7 |
| fig:ag_qtype | ag_fig6_question_type_effect.png | alias_graph key_findings Fig.6 | §7 |
| fig:cp_sep | cp_fig11_separability.png | chem_patents key_findings Fig.11 + extra_correlation_robustness | §7 |
| fig:ag_sep | ag_fig8_confusion_is_separability.png | alias_graph key_findings Fig.8 | §7 |
| **fig:per_route** | **cp_fig23_per_route_frontier.png** **[NEW r4]** | chem_patents experimental_plots/extra_per_route_frontier (4 csv + summary.json) | §8 (Limitations back-ref) |
| fig:cp_ens | cp_fig12_ensemble_headroom.png | chem_patents key_findings Fig.12 | §8 |
| fig:ag_ens | ag_fig7_ensemble_headroom.png | alias_graph key_findings Fig.7 | §8 |

## Round-4 figure discipline
- Both new figures are referenced and interpreted; neither is an orphan.
  - cp_fig22 (fig:ari): defined in §4 (Eq.~\ref{eq:ari}), read-off in §6.1,
    back-referenced at the §7 separability-floor crux, reinforced in §8 deploy
    clause, and named as the W3-probe before/after target in Limitations.
  - cp_fig23 (fig:per_route): introduced in §8 under the four-point honesty
    contract, back-referenced in the Limitations thin-n caveat.
- Figure count is now 27 (was 25). Per story risk #10, these are the **last two**
  figures the paper absorbs; round 5+ is polish only.
- No figure was retired or repointed this round; cp_fig04/08/13/15/16 remain
  present-but-unreferenced superseded panels (unchanged from round 3).
