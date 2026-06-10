# Correctness review (round 6)

Scope this round: verify the heavy R6 restructure (6-page body, 25→7 body floats,
18 figures relocated to Appendix groups A–C, Related Work 7¶→1¶, prose
compression) plus the R5 numeric cleanup did **not** corrupt any number,
citation, reference, or figure pointer. Bash was read-only throughout.

## Headline: 0 MISMATCH, 0 UNTRACEABLE.
Every load-bearing number still traces to a file under `reports/runs/`. The R5
cleanup landed correctly. No dangling `\ref`, no unresolved `\cite`, no orphaned
moved figure. The two fixed bib entries are real. Nothing introduced by the
restructure is wrong.

## Blocking issues (MISMATCH / UNTRACEABLE) — NONE
No blocking issues. Details below.

## R5-cleanup verification (the four items called out) — ALL CORRECT
| cleanup item | paper now says | source | status |
| --- | --- | --- | --- |
| raw τ stable-band | `[0.385, 0.43]` (L503) | `extra_cost_frontier/tau_sweep_summary.json` → `tau_admitted_stable_range_raw [0.385,0.43]` | **CORRECT** (old [0.39,0.43] gone) |
| raw τ cheapest-bge band | `[0.33, 0.435]` (L504, L529, L839) | same file → `tau_cheapest_bge_range_raw [0.33,0.435]` | **CORRECT** (old [0.33,0.44] gone) |
| over-representation | `48.7×` (abstract L70 implied via XRC, body L747) | `round06_language_collapse/summary.json` → `"overrep": 48.71` | **CORRECT** (old 49× harmonized) |
| L∞ display | `0.058 (5.84\%)` (L553) | `extra_rrc_budget_frontier/summary.json` → `L_inf 0.0584` | **CORRECT** (both forms shown) |

## Number-trace table (headline set + spot-checks)
| claim | paper value | source file | source value | status |
| --- | --- | --- | --- | --- |
| best CLIR@10 | 0.50 (egemma) | chem `headline_numbers.csv` egemma | 0.5024 | MATCH |
| home advantage max | +0.55 (e5) | chem csv e5 `home_adv` | 0.5526 | MATCH |
| alias RBO ceiling | 0.39 | alias csv max `cross_lingual_rbo` | 0.387 (egemma) | MATCH |
| CLIR RBO ceiling | 0.19 | `round05_consistency/summary.json` `ceiling_rbo` | 0.1934 (granite) | MATCH |
| confusion range | 14–78% | alias `EXECUTIVE_SUMMARY` / csv `confusion_publication` | egemma 0.144 … gte 0.782 | MATCH |
| XRC50 egemma | 3.5× | `extra_xrc_reading_cost/xrc_per_model.csv` | 3.5 | MATCH |
| XRC50 bge/granite/nomic/e5 | 2.0 / 1.25 / 11.5 / 97.75 | same csv | 2.0 / 1.25 / 11.5 / 97.75 | MATCH |
| RRC@100 | 0.7445 | `extra_rrc_budget_frontier/summary.json` | 0.7445 | MATCH |
| RRC@1000 | 0.9416 (→"0.942") | same | 0.9416 | MATCH |
| L∞ floor egemma | 0.058 (5.84%) | same `L_inf` | 0.0584 | MATCH |
| L∞ e5 | 0.372 | same `L_inf_by_model` | 0.3723 | MATCH |
| ARI@100 egemma/qwen tie | 0.229 vs 0.233 | `extra_ari_decomposition/summary.json` | 0.2286 / 0.2326 | MATCH |
| ARI@100 gap CI | [−0.174, 0.176] | `extra_robustness_appendix/robustness_table.csv` A5 | −0.174 / 0.1762 | MATCH |
| separability r | +0.96 (n8), +0.958 (n7) | `extra_correlation_robustness/summary.json` | 0.961 / 0.958 | MATCH |
| sep r sign-stability | P(r>0)=0.9997, CI [0.73,1.00] | `robustness_table.csv` A1 | 0.9997 / [0.7301,0.9977] | MATCH |
| partial r \| R@10 | +0.29, p=0.57 | `robustness_table.csv` W2 | 0.2948 / 0.5706 | MATCH |
| cp leaderboard (all cells) | Table 1 | chem `headline_numbers.csv` | every R@10/CLIR/home/sep/MRS cell | MATCH |
| ag leaderboard (all cells) | Table 2 | alias `headline_numbers.csv` | every pubR/RBO/conf/MRS cell + CIs | MATCH |
| en→de hardest edge | 0.12 | `round02_directional_clir/summary.json` | 0.125 | MATCH |
| de↔zh asymmetry gap | +0.23 | `extra_directional_hub/summary.json` | 0.234 | MATCH |
| pooled targets fr/en/zh/de | 0.375/0.367/0.350/0.309 | `extra_directional_hub` `correct_hub_scores` | 0.375/0.367/0.35/0.309 | MATCH |
| corpus comp en/zh | 46% / 0.4% | `extra_directional_hub` caveat | 46% / 0.4% | MATCH (caveat-level) |
| MT penalty | −0.044, p=0.13 | `round03_mt_penalty/summary.json` | −0.0444 / 0.1307 | MATCH |
| pooled mate-hit@10 | 0.38 | `round04_mate_retrieval/summary.json` | 0.375 | MATCH |
| lost-in-top1000 | 15% | same `pooled_lost_share_top1000` | 0.1542 | MATCH |
| egemma median first-foreign | 5 | same `best_median_first_foreign_rank` | 5 | MATCH |
| same-lang noise beats gold | 60% | `round07_distractor_dominance/summary.json` | 0.604 | MATCH |
| oracle CLIR@10 (cp) | 0.61 (+0.11) | `round09_ensemble/summary.json` | 0.6119 / 0.1095 | MATCH |
| oracle vs best (ag) | 88% vs 76% | `round08_model_agreement/summary.json` ALL | 0.8788 / 0.7576 | MATCH |
| sibling vs parent win | 18.1% / 6.2% (2.9×) | `extra_confusion_severity/summary.json` | 0.1813 / 0.0624 / 2.91 | MATCH |
| egemma sibling/parent | 6.1% / 1.5% | `extra_confusion_severity/severity_split.csv` | 0.0606 / 0.0152 | MATCH |
| modal failure same-lang sibling | 44.4%; sib 79.4%; same-lang 55.6% | `extra_joint_failure/summary.json` | 0.4436 / 0.7938 / 0.5564 (n=257) | MATCH |
| availability slope | −0.57 (n=5) | `extra_availability_residual/summary.json` | −0.5719 | MATCH |
| mean home adv (ag) | +0.32 | same `mean_home_adv` | 0.324 | MATCH |
| en avail / zh avail+homeadv | 42% / 8% / +0.47 | `availability_regression.csv` | en 0.42, zh 0.08 / home_adv 0.4746 | MATCH |
| own vs foreign gold recall | 0.63–0.82 / 0.35–0.47 | `round04_home_advantage/summary.json` | same 0.63–0.82, cross 0.348–0.473 | MATCH |
| structure vs role | 0.26/51% vs 0.60/25% | `round07_question_surface/summary.json` | 0.263/0.508 vs 0.595/0.250 | MATCH |
| formula token p | p<0.01 | same `formula_recall.p` | 0.0025 | MATCH |
| universal-blind core | 16/132 (12%), 14 structure | `extra_joint_failure` A8 | 16/132, structure 14 | MATCH |
| confused AUC vs not | 0.55 / 0.70 | `round09_score_separability/summary.json` | 0.549 / 0.698 | MATCH |
| two-tax rho (n=7) | −0.59 | `extra_two_tax_degeneracy/summary.json` | −0.5946 | MATCH |
| cheapest-reader trap rho | +0.29, n=7, p=0.53 | `extra_cost_frontier/summary.json` | 0.2857 / 0.5345 | MATCH |
| degeneracy gate flags | gte + e5 | `extra_two_tax_degeneracy` `recommended_members` | {gte, e5} | MATCH |
| corpus / split counts | 23,487; 137=57+80; 34 es | chem `EXECUTIVE_SUMMARY` header | 23,487; 57/80; 34 | MATCH |
| alias counts | 132 / 24 / ~109 / ~2.4 | alias `EXECUTIVE_SUMMARY` header | 132 / 24 / 109 / 2.4 | MATCH |
| per-route n_same / corners | de 7, zh 2, es 0; 3 corners; flip en,fr (2/5) | `extra_per_route_frontier/summary.json` | de 7, zh 2, es 0; 3 corners; flip en+fr | MATCH |

## Reference / citation / figure-pointer audit (the restructure risk surface)
- **`\cite` resolution:** all **36** distinct cited keys resolve to one of the
  **43** bib entries in `custom.bib`. Zero unresolved citations.
- **Fixed bib entries spot-check (this session's fix):**
  - `whatdrivesclir2025` now carries real authors (Goworek, Macmillan-Scott,
    Özyiğit), full title, and arXiv 2511.19324 — no longer a stub. **CORRECT.**
  - `crosslingualcost2025` now Amiraz et al., title explicitly
    "Retrieval Biases in RAG over **Arabic-English** Corpora", arXiv 2507.07543;
    the Related-Work prose marks it "(Arabic–English)" consistently. **CORRECT.**
- **Dangling `\ref`:** none. Every `\ref{fig:…}`/`\ref{tab:…}` has a matching
  `\label`.
- **Orphaned labels:** the only label-without-ref items are
  `app:cp_extra`, `app:ag_extra`, `app:agg_route` (appendix section anchors),
  `eq:clirmrs` (equation), and `sec:setup` (section). All benign — **no figure
  or table is orphaned**.
- **Body vs appendix float count matches the CUT-NOTE:** body floats are exactly
  `fig:teaser, fig:cost_frontier, fig:rrc_budget, fig:ag_conf, fig:cp_sep` +
  `tab:cp_board, tab:ag_board` = **5 figs + 2 tables = 7**. The 18 relocated
  figures all live under `\appendix` (line 1024+) in groups A
  (`app:cp_extra`), B (`app:ag_extra`), C (`app:agg_route`).
- **Every moved figure keeps ≥1 body ref; none lost its number.** Moved figures
  referenced more than once from the body — `fig:ari` (3×), `fig:cp_mate` (2×),
  `fig:cp_ribbon` (2×), `fig:per_route` (2×) — resolve fine (multiple body refs
  to an appendix float is legal). The brief's "exactly one body \ref" intent —
  *no relocated figure orphaned* — holds: zero moved figures have 0 refs.

## Did prose compression drop/garble any number?
No. I checked every claim whose figure moved to the appendix to confirm the
number survived in body prose:
- **Home advantage +0.55**: figure moved to App. A (`fig:cp_home`); the number is
  preserved in body L455 with the paired-within-query footnote intact. ✓
- **Directional en→de 0.12 / de↔zh +0.23 / fr 0.375 ~ en 0.367**: figure moved
  (`fig:cp_dir`); all four numbers kept in body L466–474. ✓
- **MT penalty −0.044/p=0.13**: figure moved (`fig:cp_mt`); number kept in body
  L535 and again in the budget-rule L916. ✓
- **mate-hit@10 0.38, 15% lost, median rank 5**: figure moved (`fig:cp_mate`);
  all kept in body L543–544. ✓
- **ARI@100 0.229/0.233, L∞ 0.058**: figure moved (`fig:ari`); numbers kept in
  body L566–567 and Analysis L796–798. ✓
- **language collapse 48.7×, 60%**: figure moved (`fig:cp_collapse`); both kept
  in Analysis L747–748. ✓
- **RBO 0.39 (alias)**: figure moved (`fig:ag_rbo`); number kept in body L605. ✓
- **availability −0.57, zh 8%/+0.47, en 42%**: figure moved (`fig:ag_avail`);
  numbers kept in Analysis L736–744. ✓
- **structure 0.26/51% vs role 0.60/25%**: figure moved (`fig:ag_qtype`);
  numbers kept in Analysis L753–755. ✓
- **AUC 0.55/0.70**: figure moved (`fig:ag_sep`); numbers kept in Analysis
  L776. ✓
- **oracle 0.61 (cp) / 88% vs 76% (ag)**: figures moved (`fig:cp_ens`,
  `fig:ag_ens`); numbers kept in Deployment L887–888. ✓
- **two-tax −0.59 / cheapest-trap +0.29**: figure moved (`fig:two_tax`); both
  kept in body L593 + L513 and reiterated in Limitations L943–944. ✓
No claim lost its supporting number when its figure moved to the appendix.

## Design-soundness (unchanged from R5, re-confirmed still honest after compression)
- **MT penalty correctly framed as a null** ("statistically insignificant",
  "we read this as a *null* result … we do not claim it helps") — not "no
  effect." Honest. (L536–539, L916.)
- **Concept-lens recall correctly flagged as mechanically capped** ("capped by
  its ~109 positives and must not be read as recall quality", L259). Honest.
- **Separability r correctly demoted to sign-robust + descriptive**: the body
  reports the sign (P(r>0)=0.9997), the wide n=7 CI, and the partial-r=0.29
  (n.s.) that kills the net-of-capability reading (L799–804, App. table). This
  survived compression intact and remains the most carefully hedged claim.
- **Degeneracy gate** still single-criterion CLIR@10<0.10 flagging exactly
  {gte, e5}, with the stricter AND-gate footnote (L394–397). Honest.
- **Small-n** (n=137/132; de/zh/es per-route cells of 7/2/0) acknowledged in
  Limitations and per-route paragraph; per-route XRC labelled "indicative,"
  es XRC "undefined, never imputed." Honest.

## Overlooked / confounds (status carried from prior rounds — restructure did not regress any)
- **Home-advantage vs gold-availability confound** — addressed (negative slope
  −0.57 regression, L736–744). The minimal-fix sentence is present.
- **English-target asymmetry vs corpus composition** — addressed via the en
  46%/zh 0.4% caveat (L473–476). Present.
- **Oracle/RRF headroom interpretation** — addressed ("not free," untuned RRF
  loses, L885–896). Present.
- **Parallel-gold equivalence (Equivalence-Audit)** — handled honestly as
  future work in Limitations L972–976; corresponds to `needs_eval`
  `equivalence-audit-spotcheck` (DONE-by-contract). No flag.
- **MMTEB/MIRACL/NeuCLIR transfer** — flagged as backlogged (L959–965), matches
  needs_eval. No flag.

## Verified-correct (leave these alone)
Both leaderboard tables (every cell + CIs), all XRC/RRC/ARI/L∞ derived scalars,
the separability-r sign hedging, the τ-bands, 48.7×, the directional-hub
replacement numbers, the per-route corners/flips, every `\cite`, every `\ref`,
and the 5-body-float / 18-appendix-float split are all correct and
well-supported. The restructure preserved correctness completely — the writer
should not "fix" anything in this set.
