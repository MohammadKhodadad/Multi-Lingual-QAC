# Figures manifest — round 3

Every `\includegraphics` in `paper/main.tex` (referenced by basename via
`\graphicspath{{figures/}}`) and the real source it maps to. All targets exist on
disk under `paper/figures/`. The four round-3 figures (cp_fig18–21) are
md5-identical to their `extra_*` source PNGs (verified this round).

## Figures used in main.tex (in order of appearance)

| basename (paper/figures/) | LaTeX label | source path | notes |
|---|---|---|---|
| cp_fig01_clir_leaderboard.png | fig:teaser | reports/runs/chem_patents/.../key_findings (Fig.1) | teaser, intro |
| cp_fig02_home_advantage.png | fig:cp_home | chem key_findings Fig.2 | §6.1 |
| cp_fig20_degeneracy_gap.png | fig:cp_deg | reports/runs/chem_patents/experimental_plots/extra_two_tax_degeneracy/degeneracy_gap.png | **NEW** — DEG gate, introduced in §4 Metrics |
| cp_fig03_directional_clir_matrix.png | fig:cp_dir | chem key_findings Fig.3 | §6.1; de↔zh +0.23 folded into caption |
| cp_fig18_cost_frontier.png | fig:cost_frontier | reports/runs/chem_patents/experimental_plots/extra_cost_frontier/cost_frontier.png | **NEW** — cost frontier, §6.1 (load-bearing cost fig) |
| cp_fig05_mt_penalty.png | fig:cp_mt | chem key_findings Fig.5 | §6.1 MT-null |
| cp_fig19_rrc_budget.png | fig:rrc_budget | reports/runs/chem_patents/experimental_plots/extra_rrc_budget_frontier/rrc_budget_frontier.png | **NEW** — RRC budget curve, §6.1 (load-bearing re-ranker fig) |
| cp_fig06_mate_retrieval.png | fig:cp_mate | chem key_findings Fig.6 | §6.1 |
| cp_fig07_first_foreign_rank.png | fig:cp_rank | chem key_findings Fig.7 | §6.1 |
| cp_fig21_two_tax.png | fig:two_tax | reports/runs/chem_patents/experimental_plots/extra_two_tax_degeneracy/two_tax_scatter.png | **NEW** — two-tax bridge, §6.2 opener |
| ag_fig1_cross_lingual_rbo.png | fig:ag_rbo | alias key_findings Fig.1 | §6.2 |
| ag_fig2_confusion_both_lenses.png | fig:ag_conf | alias key_findings Fig.2 | §6.2 |
| ag_fig5_universal_attractors.png | fig:ag_attr | alias key_findings Fig.5 | §6.2 |
| cp_fig17_aggregation_ribbon.png | fig:cp_ribbon | chem extra_aggregation_invariance | §6.3 |
| cp_fig14_robustness_radar.png | fig:cp_radar | chem key_findings Fig.14 | §6.3 |
| ag_fig10_robustness_radar.png | fig:ag_radar | alias key_findings Fig.10 | §6.3 |
| ag_fig12_joint_failure_modes.png | fig:ag_joint | alias extra_joint_failure | §7 |
| ag_fig11_availability_residual.png | fig:ag_avail | alias extra_availability_residual | §7 |
| cp_fig09_language_collapse.png | fig:cp_collapse | chem key_findings Fig.9 | §7 |
| cp_fig10_distractor_language.png | fig:cp_distractor | chem key_findings Fig.10 | §7 |
| ag_fig6_question_type_effect.png | fig:ag_qtype | alias key_findings Fig.6 | §7 |
| cp_fig11_separability.png | fig:cp_sep | chem key_findings Fig.11 | §7 |
| ag_fig8_confusion_is_separability.png | fig:ag_sep | alias key_findings Fig.8 | §7 |
| cp_fig12_ensemble_headroom.png | fig:cp_ens | chem key_findings Fig.12 | §8 |
| ag_fig7_ensemble_headroom.png | fig:ag_ens | alias key_findings Fig.7 | §8 |

## md5 verification of the four round-3 figures (paper copy == source)
- cp_fig18_cost_frontier.png  == extra_cost_frontier/cost_frontier.png         (4ed67244...)
- cp_fig19_rrc_budget.png      == extra_rrc_budget_frontier/rrc_budget_frontier.png (4599f917...)
- cp_fig20_degeneracy_gap.png  == extra_two_tax_degeneracy/degeneracy_gap.png   (6a2cee84...)
- cp_fig21_two_tax.png         == extra_two_tax_degeneracy/two_tax_scatter.png  (b16c967f...)

## Figures retired this round (no longer referenced)
- **cp_fig15_xrc_reading_cost.png** — old XRC50-vs-CLIR scatter; superseded by
  cp_fig18 (cost frontier), which is now the load-bearing cost figure. File still
  on disk; not referenced.
- **cp_fig16_rrc_reranker_ceiling.png** — old per-model RRC bar; superseded by
  cp_fig19 (RRC budget curve with knee + L∞). File still on disk; not referenced.

## Figures present on disk but deliberately unreferenced (clean exclusions)
cp_fig04 (asymmetry — number folded into cp_fig03 caption), cp_fig08, cp_fig13
(old CLIR-MRS leaderboard, replaced by Table 1), ag_fig3, ag_fig4, ag_fig9.
Also unused: extra_rrc_budget_frontier/rrc_xrc_plane.png (the XRC×L∞ planning
plane — available if a future round wants it; not copied to paper/figures/).

## Counts
25 `\includegraphics` in main.tex; 25 unique figure files; all exist; 25 figure
environments balanced; 4 new this round; 2 retired.
