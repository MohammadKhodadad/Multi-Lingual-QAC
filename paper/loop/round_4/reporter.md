# Reporter handoff (round 4) -> feeds story + writer (round 5)

## TL;DR
Round 4 was a CPU-only honesty-hardening round: it bootstrapped the three load-bearing scalars
(separability r, XRC50 depths, ARI@100 gap), added a tau-sweep of the cost frontier, and stitched two
2-panel figures. ALL FIVE reported numbers were independently verified against the on-disk CSV/JSON and
match exactly. Two results came back WEAKER than the dreamer assumed and must reshape the narrative:
(1) the egemma-vs-qwen3 ARI@100 gap CI straddles 0 (they are TIED, not an egemma win), and (2) the
separability->CLIR link is collinear with general capability (partial r n.s.), so it CANNOT be sold as
a capability-independent mechanism. The separability finding survives via SIGN-STABILITY
(P(r>0)=0.9997), not CI width; the cost rule survives only over a NARROW honest tau-band.

## Verified new results
Each: value -> source path -> figure -> paper section/claim affected. Every value below was opened and
confirmed; the implementer's report is accurate on all counts.

1. A1 separability r (n=7 model-level bootstrap) -- point r = 0.9577, 95% CI [0.730, 0.998]
   (WIDE, small n=7), sign-stability P(r>0) = 0.9997 (32 degenerate draws skipped); r(n=9)=0.8877.
   - src: reports/runs/chem_patents/experimental_plots/extra_robustness_appendix/robustness_table.csv
     (row 2) and .../summary.json key A1_separability_r. VERIFIED EXACT.
   - figure: none (appendix TABLE, no float -- as instructed).
   - affects: the separability claim. Writer headline must be SIGN-STABILITY, not the CI. Present the
     CI honestly as wide at n=7; the load-bearing statement is "the sign of the separability<->floor
     correlation is essentially certain (P(r>0)=0.9997)."

2. A6 XRC50 depth bootstrap (3 frontier members) -- all CIs FINITE (censored-draw frac = 0.0):
   embeddinggemma 3.5 CI [0.909, 12.0]; bge-m3 2.0 CI [0.529, 7.0]; granite-278m 1.25 CI [0.284, 12.25].
   n_cross=137, n_same=57 resampled independently.
   - src: .../robustness_table.csv (rows 3-5) and .../summary.json key A6_XRC50_depth_bootstrap.
     VERIFIED EXACT, gate_matches_ref: true for all three.
   - figure: none.
   - affects: the XRC50 / cost-frontier discussion. Writer may state the CIs are finite (genuine CIs,
     not lower bounds) but WIDE (median-of-discrete-depth bootstrap at this n). Point ordering
     egemma(3.5) > bge-m3(2.0) > granite(1.25) is reproduced; do not over-claim CI tightness.

3. A5 ARI@100 gap (qwen3 - embeddinggemma) -- point gap = 0.004, 95% CI [-0.174, 0.176] (INCLUDES 0),
   order-prob P(ARI_egemma < ARI_qwen3) = 0.5191 (near coin-flip). Underlying values egemma=0.2286,
   qwen3=0.2326 (paired bootstrap, n=137).
   - src: .../robustness_table.csv (row 6) and .../summary.json key A5_ARI100_gap_egemma_vs_qwen3
     (gap_ci_includes_zero: true). VERIFIED EXACT.
   - figure: none.
   - affects: the ARI@100 / re-ranker-irreducible-residual claim. CRITICAL CORRECTION -- see C2 below.

4. W2 partial r(auc_cross, CLIR@10 | Recall@10), n=7 -- partial r = +0.2948, two-sided p = 0.5706
   (n.s.); zero-order r = +0.9577.
   - src: .../robustness_table.csv (row 7) and .../summary.json key
     W2_separability_partial_r_controlling_recall10. VERIFIED EXACT.
   - figure: none (descriptive inline number only -- no float, as instructed).
   - affects: the separability "mechanism" framing. CRITICAL CORRECTION -- see C3 below.

5. tau-sweep of the cost frontier -- admitted-set stable band (admitted == {bge-m3, qwen3, egemma}):
   tau in [0.385, 0.430]; cheapest-admitted == bge-m3: tau in [0.330, 0.435]; egemma max-CLIR@10
   corner tau-invariant = TRUE. Coarse grid confirms the flips: at tau=0.30 granite-278m enters and
   becomes cheapest (recommendation FLIPS to granite); at tau>=0.45 only embeddinggemma is admitted.
   tau=0.40 row reproduces extra_cost_frontier/summary.json exactly (admitted {bge-m3, egemma, qwen3},
   cheapest bge-m3).
   - src: reports/runs/chem_patents/experimental_plots/extra_cost_frontier/tau_sweep.csv and
     .../tau_sweep_summary.json (keys tau_admitted_stable_range, tau_cheapest_bge_range,
     egemma_corner_tau_invariant, HONEST_NARROW_BAND). VERIFIED EXACT, including both flip rows.
   - figure: none new (original cost_frontier.png untouched, confirmed unchanged in this round's diff).
   - affects: the cost-recommendation claim. CRITICAL CORRECTION -- see C4 below.

6. Two stitched 2-panel figures (DO-NOW-4, optional) -- both exist, valid PNGs, render both source
   panels legibly (visually confirmed by opening them):
   - paper/figures/cp_fig06_07_mate.png (3062x754) = cp_fig06_mate_retrieval + cp_fig07_first_foreign_rank.
   - paper/figures/cp_fig09_10_collapse.png (2705x754) = cp_fig09_language_collapse + cp_fig10_distractor_language.
   - OPTIONAL \includegraphics fallback to LaTeX subfigure; per-source panels untouched. Writer may use
     either approach.

## CRITICAL honesty corrections the round-5 writer MUST make (verbatim wording guidance)
These four are the anti-fabrication flags. Each is confirmed against the CSV/JSON above.

C1 -- Separability headline = SIGN-STABILITY, not CI width (CONFIRMED: r=0.9577, CI [0.730,0.998] wide,
P(r>0)=0.9997).
- Writer MUST lead with sign-stability. Suggested phrasing: "Across model-level bootstrap resamples the
  separability-floor correlation is positive in 99.97% of draws (point r=0.96); with only seven
  non-degenerate models the confidence interval is correspondingly wide ([0.73, 1.00]), so we report
  the sign as the robust finding rather than its magnitude."
- Do NOT present [0.730, 0.998] as a tight/precise estimate.

C2 -- egemma and qwen3 are TIED on ARI@100; do NOT claim egemma uniquely lowest (CONFIRMED: gap 0.004,
CI [-0.174, 0.176] includes 0, P=0.519).
- Writer MUST NOT write that embeddinggemma uniquely has the lowest ARI@100 / smallest re-ranker-
  irreducible residual. Required phrasing: "embeddinggemma and qwen3-0.6B are tied for the lowest
  re-ranker-irreducible residual (ARI@100 gap 0.004, 95% CI [-0.174, 0.176], straddling zero)."
- egemma KEEPS the separate distinction of the smallest alignment-only floor L_inf = 0.058 (the
  smallest-floor result from the r3 ARI decomposition cp_fig22). That distinction is still egemma's
  alone -- do not erase it; keep it separate from the ARI@100 ordering, which is a tie.

C3 -- Separability is COLLINEAR with capability; do NOT claim it is independent / "not a tautology"
(CONFIRMED: partial r = +0.29, p = 0.57 n.s., down from zero-order +0.958).
- Writer MUST NOT claim the separability->CLIR mechanism is independent of general capability, nor that
  it is "not a capability artifact," nor "not a tautology." Required softening: "Cross-language AUC and
  overall Recall@10 are strongly collinear across our seven models; once Recall@10 is partialled out
  the separability-CLIR association is no longer significant (partial r=+0.29, p=0.57), so we present
  separability as a descriptive correlate of the cross-lingual floor, not as an effect net of general
  retrieval capability."
- Frame DESCRIPTIVELY only.

C4 -- tau-rule survives only over a NARROW honest band; do NOT say "the rule is robust" (CONFIRMED:
admitted-set stable tau in [0.385,0.430]; cheapest=bge-m3 tau in [0.330,0.435]; egemma corner
tau-invariant=TRUE; recommendation FLIPS to granite at tau<=0.3285).
- Writer MUST state the narrow band explicitly and flag the low-end flip. Required phrasing: "The
  admitted set is stable for tau in [0.39, 0.43] and bge-m3 remains the cheapest admitted reader for
  tau in [0.33, 0.44]; below tau~=0.33 the cheaper-to-read granite-278m enters the admitted set and the
  cheapest-reader recommendation flips to granite, while above tau~=0.45 only embeddinggemma qualifies.
  Only embeddinggemma's status as the unique maximum-CLIR@10 corner is tau-invariant."
- The ONLY unconditional claim is the egemma max-CLIR corner. Do NOT generalize to "the rule is robust."

## Discrepancies / unverifiable claims
None. Every numeric claim in implement_report.md matched the corresponding CSV/JSON cell exactly (A1 r,
CI, P(r>0); A6 three depths + CIs + censored-frac=0; A5 gap, CI, order-prob; W2 partial-r and p; all
five tau-sweep rows and the three summary keys). Both stitched PNGs exist and render correctly. The
implementer's self-reported cosmetic stdout bug (A1 CI mislabeled in a print, since fixed) did NOT
affect the CSV/JSON, which I read directly and which carry the correct A1 CI [0.730, 0.998]. The
implementer's own honesty caveats (tau-band narrow, W2 n.s., A5 tie) are accurate and consistent with
the files -- the report did not overstate anything.

## Changed files this round (git diff --stat summary)
Tracked edits (these are the WRITER's bib/tex, present in the tree but not authored by the implementer
per its report -- left for the writer):
- paper/custom.bib  (+33 lines)
- paper/main.tex    (+175 / -26 lines)

New untracked artifacts authored this round (implementer):
- Scripts: reports/runs/chem_patents/experimental_codes/{extra_robustness_appendix.py, extra_tau_sweep.py, stitch_merged_panels.py}
- Tables/JSON: .../extra_robustness_appendix/{robustness_table.csv, summary.json};
  .../extra_cost_frontier/{tau_sweep.csv, tau_sweep_summary.json}
- Figures: paper/figures/{cp_fig06_07_mate.png, cp_fig09_10_collapse.png}
- paper/loop/round_4/ (loop docs)
- Original extra_cost_frontier/cost_frontier.{csv,png,summary.json} UNTOUCHED (confirmed: not in diff).

## Backlogged (forthcoming) experiments to mention as pending
needs_eval.md was NOT modified this round (nothing new added -- all DO-NOW items were CPU-only and
completed). Critics must treat the following standing items as DONE/forthcoming, not as gaps:
- W4-formula-injection (causal formula-token intervention; needs query re-embed).
- CLIRMRS-external-validation (human/RAG utility signal to validate CLIR-MRS).
- XRC-conformal-M2 (split-conformal XRC; deferred -- calibration split too thin at n_same=57).
- CCI-hop-distance-law (ChEBI hop-distance confusion decay; graph-build risk).
- equivalence-audit-spotcheck (expert claim-equivalence spot-check of parallel golds).
- W3-alignment-causal-probe (LaBSE before/after alignment map; UPSIDE ONLY, do not depend on it; the r3
  ARI decomposition L_inf floor is the natural before/after target).

## Recommended next-round focus (for the story architect)
1. Reframe two pillars honestly. The story must drop any "egemma uniquely lowest ARI@100" and any
   "separability is independent of capability / not a tautology" framing. Recast separability as a
   descriptive correlate whose SIGN is robust, and recast the ARI@100 result as egemma/qwen3 TIED on
   the residual while egemma retains the smallest alignment-only floor (L_inf=0.058). These are the two
   highest-risk overclaims for a hostile reviewer.
2. Position the appendix robustness table as the credibility anchor. It now carries sign-stability,
   finite XRC50 CIs, the ARI-tie CI, and the collinearity partial-r in one place -- lead the
   limitations / robustness section with it and let the honest CIs do the work rather than hiding them.
3. State the tau-band, not tau-robustness. Give the narrow honest band and the low-end granite flip as
   a feature of the cost analysis (sensitivity acknowledged), keeping only the egemma max-CLIR corner
   as the unconditional take-away. Optionally use the two stitched panels to compress figure count.
