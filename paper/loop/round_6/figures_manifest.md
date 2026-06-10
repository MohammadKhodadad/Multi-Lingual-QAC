# Figures manifest — round 6 (cut-to-6-pages restructure + R6 corrective pass)

All `\includegraphics` use the basename via `\graphicspath{{figures/}}`.
Every source file verified present in `paper/figures/`.

**R6 corrective pass (driven by the round-6 cohesion critic):** `ag_fig2`
(`fig:ag_conf`) was moved out of the body into Appendix B, dropping the body to
**4 figures + 2 tables**; and every body sentence referencing a now-appendix
figure was relabeled `Figure~\ref{...}` → `Appendix Fig.~\ref{...}` (23 sites).

## BODY floats (6 = 4 figures + 2 tables) — count down from 25 (R5) / 7 (R6 first pass)

| label | basename (in figures/) | section | role |
|-------|------------------------|---------|------|
| `fig:teaser`        | `cp_fig01_clir_leaderboard.png`     | §1 Intro    | teaser: collapse vs average (C1) |
| `fig:cost_frontier` | `cp_fig18_cost_frontier.png`        | §6.1        | cost-vs-capability frontier (C4 spine); tau-band raw |
| `fig:rrc_budget`    | `cp_fig19_rrc_budget.png`           | §6.1        | RRC budget curve, knee K*, floor L∞ (C2/C4) |
| `fig:cp_sep`        | `cp_fig11_separability.png`         | §7 Analysis | separability deficit r=+0.96 (C3 mechanism) |
| `tab:cp_board`      | (LaTeX table)                       | §6.3        | cross-lingual leaderboard |
| `tab:ag_board`      | (LaTeX table)                       | §6.3        | alias-graph leaderboard |

Note: `fig:ag_conf` (confusion 14–78%, the C3 chemistry hook) is now in App B;
its number survives in body prose (§6.2) and the `conf` column of `tab:ag_board`.

## APPENDIX floats (20 = 19 figures + 1 table) — relocated, free (do not count toward 6 pages)

### App A — Extended Cross-Lingual Results (`app:cp_extra`)
| label | basename | body §ref |
|-------|----------|-----------|
| `fig:cp_home`     | `cp_fig02_home_advantage.png`        | §6.1 |
| `fig:cp_dir`      | `cp_fig03_directional_clir_matrix.png` | §6.1 |
| `fig:cp_mt`       | `cp_fig05_mt_penalty.png`            | §6.1 |
| `fig:cp_mate`     | `cp_fig06_07_mate.png`               | §6.1 |
| `fig:ari`         | `cp_fig22_ari_decomposition.png`     | §6.1 (+ Limitations) |
| `fig:cp_collapse` | `cp_fig09_10_collapse.png`           | §7 (caption carries 48.7×) |
| `fig:cp_deg`      | `cp_fig20_degeneracy_gap.png`        | §5 Metrics |

### App B — Extended Alias-Graph Results (`app:ag_extra`)
| label | basename | body §ref |
|-------|----------|-----------|
| `fig:ag_conf`  | `ag_fig2_confusion_both_lenses.png`    | §6.2 (moved here in R6 corrective pass) |
| `fig:ag_rbo`   | `ag_fig1_cross_lingual_rbo.png`        | §6.2 |
| `fig:ag_attr`  | `ag_fig5_universal_attractors.png`     | §6.2 |
| `fig:ag_joint` | `ag_fig12_joint_failure_modes.png`     | §7 |
| `fig:ag_avail` | `ag_fig11_availability_residual.png`   | §7 |
| `fig:ag_qtype` | `ag_fig6_question_type_effect.png`     | §7 |
| `fig:ag_sep`   | `ag_fig8_confusion_is_separability.png` | §7 |

### App C — Aggregation, Routing, and Ensemble (`app:agg_route`)
| label | basename | body §ref |
|-------|----------|-----------|
| `fig:cp_ribbon` | `cp_fig17_aggregation_ribbon.png`   | §6.3 |
| `fig:two_tax`   | `cp_fig21_two_tax.png`              | §6.2 |
| `fig:per_route` | `cp_fig23_per_route_frontier.png`   | §8 (+ Limitations) |
| `fig:cp_ens`    | `cp_fig12_ensemble_headroom.png`    | §8 |
| `fig:ag_ens`    | `ag_fig7_ensemble_headroom.png`     | §8 |

### App D — Robustness Ledger and Reproducibility
| label | kind | note |
|-------|------|------|
| `tab:robust` (`app:robust`) | LaTeX table | unchanged from round 5 |
| `app:repro` | label on Reproducibility ¶ | full script list; referenced from §5 Setup |

## Lint (no compiler available; static checks) — re-run after R6 corrective pass
- 23 unique `\includegraphics`; all present in figures/; no duplicate includes.
- 26 float labels (23 fig + 3 tab); each referenced ≥1×; no unreferenced, no dangling refs, no duplicate labels.
  - all 19 appendix figures verified to keep ≥1 body `\ref` (no orphans); `fig:ag_conf` keeps its single body ref at §6.2.
- Braces balanced (899 = 899). All `\begin{X}`/`\end{X}` matched (figure 23/23, table 3/3, tabular 3/3, equation 4/4).
- 36 `\cite` keys; all resolve in custom.bib (custom.bib NOT edited).
- Body floats = **6 (4 fig + 2 tab)** — `\appendix` appears once (line 1015); `ag_fig2`/`fig:ag_conf` now sits after it (lines 1107–1112).
- Appendix-reference labeling: every body `\ref` to a relocated figure now reads `Appendix Fig.~\ref{...}` (24 such sites incl. the moved ag_conf); the 4 body figures (teaser/cost_frontier/rrc_budget/cp_sep) keep plain `Figure~\ref{...}`.
- Figures NOT used anywhere (intentionally dropped in round 5, kept dropped):
  `cp_fig04, cp_fig06, cp_fig07, cp_fig08, cp_fig09, cp_fig10, cp_fig13, cp_fig14,
   cp_fig15, cp_fig16, ag_fig3, ag_fig4, ag_fig9, ag_fig10` (present on disk, unreferenced).
