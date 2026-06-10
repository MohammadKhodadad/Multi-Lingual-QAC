# Correctness review (round 3)

Reviewer #2 (correctness). Re-audited every numeric claim and figure in the **round-3 revision**
of `paper/main.tex` against the canonical data under `reports/runs/{chem_patents,alias_graph}/`,
with special attention to the **NEW round-3 numbers** sourced from three `experimental_plots/extra_*/`
directories that did not exist when round-2 was written:

- chem: `extra_cost_frontier/`, `extra_rrc_budget_frontier/`, `extra_two_tax_degeneracy/`
- (carried-over extras re-spot-checked: `extra_xrc_reading_cost/`, `extra_correlation_robustness/`,
  `extra_directional_hub/`, `extra_aggregation_invariance/`; alias `extra_availability_residual/`,
  `extra_confusion_severity/`, `extra_joint_failure/`)

I recomputed/cross-checked from on-disk `summary.json` + `*.csv` in each extra dir and from the two
`key_findings/headline_numbers.csv`. I also re-read the round-2 review to confirm the resolution of
its blocking items (N2, N1) and the carried B1/B2/B3.

## Headline verdict

**The two round-2 blocking items are RESOLVED. Every new round-3 headline number traces exactly.**
The cost-frontier, RRC-budget, and degeneracy-gate analyses are all data-faithful:
Pareto set `{embeddinggemma, bge-m3, granite-278m}`; τ=0.40 admitted set
`{bge-m3, qwen3-0.6B, embeddinggemma}` with cheapest-admitted `bge-m3` at XRC50 **2.0**;
DEG gate `CLIR@10<0.10` → exactly `{e5-large-instruct, gte-base}`; RRC `L_inf=0.0584`, `K*=5`,
`RRC@100=0.7445`, `RRC@1000=0.9416`; two-tax `rho=-0.59 (n=7, p=0.16)`; cheapest-reader trap
`rho=+0.29 (n=7, p=0.53)`. The 81 leaderboard cells (45 CP + 36 alias) still match. All 25
referenced figures (incl. the 4 new ones cp_fig18-21) exist.

**N2 (round-2 MISMATCH) RESOLVED.** The false superlative "embeddinggemma has the lowest
non-degenerate reading cost (XRC50 3.5×)" is gone everywhere. The paper now consistently states
embeddinggemma is "the capability corner, **not** the cheapest reader," and that `bge-m3` (2.0×) is
the cheapest *admitted* model at τ=0.40 (abstract L77, intro L116, Results L592-595, Fig.18 caption
L612-613, Deployment L1011-1015). The only remaining "lowest" token (L464) is "lowest SapBERT 0.179,"
a correct reference to the lowest non-degenerate **CLIR@10**, not reading cost. **No false
reading-cost superlative survives anywhere.**

**B2 (RBO "any model" not "best model") RESOLVED and HELD.** Intro L107 "the best cross-lingual RBO
**any** of our nine models achieves," Results L710 "the best cross-lingual RBO **any** of the nine
models reaches," Conclusion L1144 "the best cross-lingual RBO **any** model reaches." No "best model"
RBO framing anywhere. (Note: the conductor referenced "line-605"; in the round-3 file the B2-relevant
RBO lines are 107/710/1144, and line ~600-605 is the cost-frontier prose, which correctly carries the
+0.29 descriptive caveat — see below. The "best model" string at L567/L576 refers to the *directional
matrix figure* and the *XRC median model*, not RBO, and is correct usage.)

**N1 (round-2 "statistically indistinguishable" overclaim) RESOLVED.** Replaced by "nearly tied as
targets (within 0.01 of each other)" (L558). No untested significance claim remains.

**Non-significant correlations correctly quarantined.** The two n.s. correlations
(two-tax `rho=-0.59, p=0.16`; cheapest-reader trap `rho=+0.29, p=0.53`) appear ONLY in the Results
body (L601, L689, L703 + Fig.21 caption) and the Limitations (L1097-1100), each tagged "n.s." /
"descriptive" / "non-significant." **Neither appears in the abstract, the introduction, or the
conclusion.** Confirmed by exhaustive grep.

**ONE NEW non-blocking issue (mislabel, value correct): C-NEW.** Figure 21's caption and prose call
the plotted Y-axis quantity the **"sibling-confusion rate,"** but the source plots the *general*
`confusion_rate` (any look-alike out-ranks the gold), not the sibling-specific `sibling_win_rate`.
The *value* (granite 0.182, egemma 0.068) is correct for `confusion_rate`; only the label is wrong.
A `sibling_win_rate` column exists in the same file with different values (granite 0.1439), so a
hostile reviewer could catch the mislabel. Low severity, easy one-word fix. Details below.

**Counts: MISMATCH = 0. UNTRACEABLE (not already TODO-fenced) = 0. NEW mislabel (non-blocking) = 1
(C-NEW).** N2 and B2 are both **RESOLVED**. The pre-existing `\todo{}`-fenced numbers (corpus dedup
14,401; human-eval 8.33/10 etc.) remain honestly fenced and are NOT counted as blocking.

---

## Blocking issues (MISMATCH / UNTRACEABLE) — none

No MISMATCH and no UNTRACEABLE numbers in the round-3 draft. The round-2 blocker (N2) and overclaim
(N1) are both fixed; B1/B2/B3 remain fixed.

## Non-blocking issue (mislabel) — should fix

### C-NEW (mislabel, value correct). Fig.21 "sibling-confusion rate" should be "confusion rate."
- **Where:** Fig.21 caption (L700-702): "the confusability tax (**sibling-confusion rate**,
  alias-graph benchmark)"; and the §ssec:ag intro prose (L687-694) refers to it as "a confusability
  tax."
- **What the data says:** `extra_two_tax_degeneracy.py` L90/L92 merges
  `extra_confusion_severity/severity_split.csv` column **`confusion_rate`** (renamed
  `confusability_tax`) — NOT `sibling_win_rate`. The figure's own Y-axis label (script L138) is
  "confusability tax = alias-graph **confusion_rate**." `confusion_rate` is the rate at which *any*
  look-alike (sibling OR parent) out-ranks all gold (granite 0.1818, egemma 0.0682, the values the
  paper quotes); `sibling_win_rate` is the *sibling-specific* rate (granite 0.1439, egemma 0.0606),
  a different column with different values.
- **Severity:** Low. The plotted/quoted **values are correct** (0.182, 0.068 = `confusion_rate`); only
  the descriptor "sibling-" is inaccurate and contradicts the figure's own axis label and the source.
- **Minimal fix:** delete "sibling-" — caption → "(confusion rate, alias-graph benchmark)" — and in
  L687-688 keep "confusability tax = a look-alike compound out-ranking the gold" (which is already
  the general definition). Do NOT change the value 0.182. (The *separate* sibling-vs-parent
  severity split — 18.1% sibling vs 6.2% parent, §ssec:ag L728-730 — is a different, correctly-labeled
  analysis from `extra_confusion_severity`; leave it alone.)

---

## Number-trace table (round-3 NEW numbers + conductor checkpoints + spot-checks)

Tolerance ≤ 0.006 on [0,1] metrics; depths/counts exact. "source" = named file under
`reports/runs/.../experimental_plots/extra_*/` unless noted.

| claim (paper) | paper value | source file | source value | status |
|---|---|---|---|---|
| **Pareto frontier set** | {egemma, bge-m3, granite} | `extra_cost_frontier/summary.json` `frontier_members` | [bge-m3, embeddinggemma, granite-278m] | MATCH |
| egemma unique max-CLIR corner | (unique) | `summary.json` `unique_top_clir_model` | embeddinggemma | MATCH |
| τ=0.40 admitted set | {bge-m3, qwen3-0.6B, egemma} | `summary.json` `tau_admitted_set` | [bge-m3, qwen3-0.6B, embeddinggemma] | MATCH |
| **bge-m3 cheapest admitted XRC50** | 2.0 (at τ=0.40) | `summary.json` `tau_admitted_min_xrc_model`/`_value` | bge-m3 / 2.0 | MATCH |
| egemma XRC50 / CLIR@10 (frontier pt) | 3.5× / 0.50 | `cost_frontier.csv` egemma | 3.5 / 0.5024 | MATCH |
| bge-m3 frontier pt | 2.0× / 0.44 | `cost_frontier.csv` bge-m3 | 2.0 / 0.4367 | MATCH |
| granite frontier pt | 1.25× / 0.33 | `cost_frontier.csv` granite | 1.25 / 0.3285 | MATCH |
| dominated finite models | LaBSE/SapBERT/e5/nomic/qwen3 | `summary.json` `dominated_finite_models` | same 5 | MATCH |
| gte off-plane (XRC undefined) | undefined | `cost_frontier.csv` gte `XRC50` blank, censored=True | inf | MATCH |
| **cheapest-reader trap rho** | +0.29, n=7, p=0.53 | `extra_cost_frontier/summary.json` `W2_trap_spearman_rho/p` | 0.2857 / 0.5345, n=7 | MATCH |
| **DEG gate flags exactly {gte, e5}** | {gte-base, e5-large-instruct} | `extra_two_tax_degeneracy/summary.json` `deg_gate.recommended_members` | [e5-large-instruct, gte-base] | MATCH |
| DEG = CLIR@10<0.10 (rule) | CLIR@10<0.10 | `summary.json` `deg_gate.RECOMMENDED` | "DEG = clir_at_10 < 0.10" | MATCH |
| AND-gate flags only gte (e5 RRC@1000=0.63) | RRC@1000=0.63 → gte only | `summary.json` `deg_gate` + `deg_flags.csv` e5 RRC@1000 | 0.6277 ; DEG_strict=[gte-base] | MATCH |
| gte CLIR@10=0.000 (bar not visible) | 0.000 | `deg_flags.csv` gte | 0.0 | MATCH |
| e5 CLIR@10 | 0.077 | `deg_flags.csv` e5 | 0.0766 | MATCH |
| lowest non-deg CLIR@10 = SapBERT | 0.179 | `deg_flags.csv` non-deg min | SapBERT 0.1788 | MATCH |
| **RRC@100 egemma** | 0.7445 | `extra_rrc_budget_frontier/summary.json`, `rrc_knee.csv` | 0.7445 | MATCH |
| **RRC@1000 egemma** | 0.9416 (→0.942) | `rrc_knee.csv` | 0.9416 | MATCH |
| **L_inf egemma (1-RRC@1000)** | 0.0584 / 5.84% | `summary.json` `embeddinggemma.L_inf` | 0.0584 | MATCH |
| **K* egemma (knee)** | 5 | `summary.json`/`rrc_knee.csv` egemma K_star | 5 | MATCH |
| top-100 re-ranker recovers ≤74% | 74% | 1-RRC@100 framing, RRC@100=0.7445 | 0.7445 | MATCH |
| leaves ~25% on the table | 25% | 1-0.7445 | 0.2555 | MATCH |
| L_inf range 0.058–0.372 | 0.058 → 0.372 | `summary.json` `L_inf_by_model` | egemma 0.0584, e5 0.3723 | MATCH |
| knee ≤20 for nearly every non-deg model | ≤20 | `K_star_by_model` | nondeg K* ∈ {2,5,5,10,20,20,20}; e5(deg)=30 | MATCH |
| regression checks pass | "all pass" | `summary.json` `regression_checks_passed` | true (failures []) | MATCH |
| **two-tax rho (non-redundant)** | -0.59, n=7, p=0.16 | `extra_two_tax_degeneracy/summary.json` `two_tax.spearman_rho_n7_nondeg/p` | -0.5946 / 0.1591, n=7 | MATCH |
| two-tax egemma low on both | XRC50 3.5 / conf 0.068 | `two_tax_table.csv` egemma | 3.5 / 0.0682 | MATCH |
| granite reads cheapest 1.25× confuses 0.182 | 1.25 / 0.182 | `two_tax_table.csv` granite | 1.25 / 0.1818 | MATCH (value); label "sibling-" wrong → C-NEW |
| **XRC50 egemma (depth 2→7)** | 3.5× | `extra_xrc_reading_cost/xrc_per_model.csv` | 3.5 (D50_same 2, D50_cross 7) | MATCH |
| XRC50 granite / nomic / e5 | 1.25 / 11.5 / 97.75 | `xrc_per_model.csv` | 1.25 / 11.5 / 97.75 | MATCH |
| **best CLIR@10 / home +0.55** | 0.50 / +0.55 | CP `headline_numbers.csv` egemma / e5 | 0.5024 / 0.5526 | MATCH |
| hardest edge en→de | 0.12 | `extra_directional_hub/summary.json` | 0.125 | MATCH |
| most asymmetric de↔zh gap | +0.23 | `summary.json` `most_asymmetric_pair.gap` | 0.234 | MATCH |
| hub fr/en/zh/de | 0.375/0.367/0.350/0.309 | `hub_scores.csv` | 0.375/0.3673/0.35/0.3091 | MATCH |
| corpus en 46% / zh 0.4% | 46% / 0.4% | `summary.json` corpus caveat | 0.46 / 0.0043 | MATCH |
| **auc_cross~clir robust n=7** | +0.958 (Spearman 0.964) | `extra_correlation_robustness/correlation_robustness.csv` | 0.958 / 0.964 | MATCH |
| auc_cross~clir n=8 pool | +0.96 | `correlation_robustness.csv` pearson_n8 | 0.961 | MATCH |
| over-rep~CLIR flips sign on n=7 | "flips sign" | `correlation_robustness.csv` mean_overrep~clir n7 | +0.419 (was -0.60) | MATCH (honest) |
| home-adv~RBO collapses on n=7 | "collapses" | `correlation_robustness.csv` home_adv~rbo n7 | 0.186 (was -0.846) | MATCH (honest) |
| availability slope (alias) | -0.57, n=5, descriptive | `extra_availability_residual/summary.json` | -0.5719 | MATCH |
| mean home adv (alias) +0.32; zh 8%/+0.47 | +0.32 / 8% / +0.47 | `summary.json` / `availability_regression.csv` | 0.324 / 0.08 / 0.4746 | MATCH |
| own/foreign recall (alias) | 0.63–0.82 / 0.35–0.47 | `availability_regression.csv` | 0.63/0.822 ; 0.348/0.473 | MATCH |
| **joint failure 257 confused** | 257 | `extra_joint_failure/summary.json` `n_confused_cases` | 257 | MATCH |
| modal same-lang sibling 114/257 | 44.4% | `cell_fractions[same-language sibling]` | 0.4436 | MATCH |
| siblings total / same-lang total | 79.4% / 55.6% | `sibling_total_frac` / `same_language_total_frac` | 0.7938 / 0.5564 | MATCH |
| universal-blind 16/132 (12%), 14 structure | 16/132, 14 | `A8_universal_blind` | 16/132, structure 14 | MATCH |
| **confusion severity sibling 18.1% vs parent 6.2%** | 18.1% / 6.2% / 2.9× | `extra_confusion_severity/summary.json` | 0.1813 / 0.0624 / 2.91 | MATCH |
| egemma sibling vs parent | 6.1% / 1.5% | `severity_split.csv` egemma | 0.0606 / 0.0152 | MATCH |
| pooled mate-hit@10 | 0.38 | chem `round04_mate_retrieval/summary.json` | 0.375 | MATCH |
| 15% never in top-1000 (8-model pool) | 15% | `extra_xrc/summary.json` consistency_checks | 0.1542 | MATCH |
| median first-foreign rank (egemma) | 5 | `round04/summary.json` best_median_first_foreign_rank | 5 | MATCH |
| MT diff / p | -0.044 / 0.13 | `round03_mt_penalty/summary.json` | -0.0444 / 0.1307 | MATCH |
| RBO ceiling alias / CP | 0.39 / 0.19 | alias `headline` egemma rbo / CP `headline` max rbo | 0.387 / 0.1934 (granite) | MATCH |
| 49× over-fetch / 60% noise | 49× / 60% | chem round06/07 (carried) | 48.71 / 0.604 | MATCH |
| oracle CLIR@10 / alias 88% vs 76% | 0.61 / 88% vs 76% | chem round09 / alias round08 | 0.6119 / 0.879/0.758 | MATCH |
| structure vs role R@10/conf | 0.26/51% vs 0.60/25% | alias round07 | 0.26/0.51 vs 0.60/0.25 | MATCH |
| AUC confused vs not (alias) | 0.55 / 0.70 | alias round09 | 0.549 / 0.698 | MATCH |
| aggregation rank-range egemma | [1,4] | `extra_aggregation_invariance/summary.json` | [1,4] | MATCH |
| **CP leaderboard (9×5 = 45 cells)** | 45 | CP `headline_numbers.csv` | all 45 | MATCH |
| **alias leaderboard (9×4 = 36 cells)** | 36 | alias `headline_numbers.csv` | all 36 | MATCH |
| benchmark sizes | 132 / 137(57+80) / 24 / corpus 23,487 | exec summaries + headers | all | MATCH |
| Spanish 34 zero-gold no-home | 34 / zero | CP exec summary "defining property" | 34 / 0 | MATCH |
| corpus dedup 14,401; human-eval 8.33/10 | (TODO-fenced) | not under reports/ | — | TODO-fenced (OK) |

---

## Design-soundness findings (per headline claim)

1. **Cost frontier (Fig.18, abstract, intro, Results, Deployment).** Sound and well-bounded. The
   Pareto computation (`min XRC50, max CLIR@10` over finite-XRC models, gte off-plane) is correct;
   the three frontier members and the unique max-CLIR corner trace exactly. Critically the paper
   does NOT call egemma the cheapest — it explicitly says bge-m3 (2.0×) is the cheapest admitted at
   τ=0.40. The cheapest-reader-trap directional read-off is correctly hedged as descriptive
   (`rho=+0.29, n=7, p=0.53, n.s.`) and is NOT used as a statistical claim. **This is the clean fix
   of round-2 N2. Leave the framing alone.** One minor honesty note: τ=0.40 is stated as "untuned"
   (L592), which is the right disclosure for a deployment threshold pulled from thin air.

2. **RRC budget (Fig.19, abstract, Results, Analysis, Deployment).** The cleanest new contribution
   and fully sound. `RRC@K = P[first foreign twin rank ≤ K]`, `1-RRC@K` as "provably unrecoverable
   by any top-K re-ranker" is a legitimate upper bound, and the knee/floor reading is well-scoped.
   All derived numbers (0.7445, 0.9416, 0.0584, K*=5, range 0.058–0.372, e5 0.3723) trace exactly,
   and the source records `regression_checks_passed=true` — so the paper's "we state it firmly" is
   warranted. **Leave alone.**

3. **Degeneracy gate (Fig.20, Metrics §4).** Sound and reproducible. The single criterion
   `CLIR@10<0.10` flags exactly {gte, e5}, matching the paper's exclusions, and the footnote
   correctly explains why the stricter AND-gate flags only gte (e5 RRC@1000=0.6277 ≥ 0.10 clears the
   second criterion). The figure caption values (gte 0.000, e5 0.077, SapBERT floor 0.179) all
   trace. This is exactly the kind of reproducible exclusion rule a hostile reviewer wants.
   **Leave alone.**

4. **Two-tax non-redundancy (Fig.21, §ssec:ag).** The *finding* (the reading-cost tax and the
   confusability tax are weakly/inversely correlated, so both benchmarks are needed) is sound and
   correctly demoted to descriptive: `rho=-0.59, n=7, p=0.16, n.s.`, explicitly "not an independence
   result," kept out of abstract/intro/conclusion. The only defect is the **label** C-NEW: the
   plotted quantity is the general `confusion_rate`, not the "sibling-confusion rate." Fix the word,
   keep the finding and the value.

5. **XRC population caveat (round-2 T-NEW) — now HANDLED.** Round-2 asked for a clause noting XRC is
   a population-level (57 same-gold vs 137 cross-gold queries), not paired, ratio. The Metrics
   section L419-421 now states this explicitly ("D_same is taken over the 57 same-language-gold
   queries and D_cross over the 137 cross-gold queries, so XRC is a population-level, not a paired
   within-query, ratio"). **Resolved.**

6. **Separability mechanism (+0.96, robust).** Still the load-bearing correlation and still honestly
   defended: robust (+0.958, Spearman 0.964, n=7) vs the two fragile bias correlations that "do not
   survive" the drop. Unchanged from round-2 and correct. **Leave alone.**

7. **MT-of-question null (Fig.5).** Still framed correctly ("insignificant," "null," "we do not
   claim it helps," p=0.13). Carried-over; correct.

8. **Concept-lens mechanical cap.** Still flagged ("must not be read as recall quality," L313).
   Correct.

---

## Overlooked / confounds / threats-to-validity (with the minimal fix each)

**T1 — Home-advantage vs gold-availability confound (HANDLED).** §Results L532-534 ties the +0.55 to
"much of which gold availability shapes … though a residual encoder bias remains," and §Analysis
carries the availability-residual analysis (slope -0.57, descriptive). Resolved.

**T2 — Directional asymmetry as corpus composition (HANDLED).** §Results L559 "What asymmetry exists
partly tracks corpus composition (en 46% vs zh 0.4%) … not only encoder behaviour." Resolved.

**T3 — oracle/RRF headroom (HANDLED).** Kept tight in one paragraph (oracle 0.61 / 88% vs 76%, RRF
negative, "not free"). No fix.

**T4 — Parallel-gold equivalence audit (HONESTLY DEFERRED).** Listed in needs_eval and in Limitations
L1116-1120 ("a spot-check that the parallel human-translated golds are claim-level equivalent …
remain future work"). Per the critic contract, DONE/deferred, not a missing experiment.

**T5 — Small-n correlations as mechanisms (HANDLED).** The robust separability link is promoted and
the two fragile bias correlations explicitly demoted to descriptive with the drop-the-collapsers
check. Strongest standing improvement; unchanged. Resolved.

**T6 — Stale baseline run never cited (CLEAN).** Re-checked: the paper cites only the 23,487-corpus
key_findings numbers; the `20260601-235117_137questions` baseline is never pulled in. No leak.

**T7 — Two RBO ceilings stay separated (HANDLED, B2 held).** 0.39 (alias) and 0.19 (CP) remain
attributed to their own benchmarks; "any model" wording is in place in intro/Results/conclusion.

**T-MINOR (non-blocking, carried) — universal-blind language ordering.** §Analysis L939 says the
blind core is "predominantly in French, Chinese, and German." Source `A8_universal_blind.by_language`
= fr 5, zh 4, **de 3, es 3 tied**. fr/zh are unambiguous top two; picking de over es at the 3-tie is
arbitrary but harmless. Optional fix: "predominantly French and Chinese." Not blocking.

**T-NEW-3 (non-blocking) — C-NEW Fig.21 "sibling-confusion rate" mislabel** (see above). One-word
fix; value is correct.

**T-NEW-3b (non-blocking) — τ=0.40 is an arbitrary single threshold.** The admitted-set claim
({bge-m3, qwen3, egemma}) and "bge-m3 is cheapest admitted" depend on one untuned cutoff τ=0.40.
The paper does disclose "untuned" (L592), which is adequate; a hostile reviewer might still ask for
sensitivity. Minimal pre-emption (optional): note that the *Pareto* conclusion (egemma capability
corner, bge-m3 cheaper-to-read) is τ-independent, so the recommendation does not hinge on τ. The
paper already leans on the τ-independent Pareto framing, so this is low risk. Not blocking.

---

## Verified-correct (leave these alone)

- **All round-3 NEW numbers** (Pareto set {egemma,bge-m3,granite}; τ=0.40 admitted {bge-m3,qwen3,
  egemma}, cheapest-admitted bge-m3 2.0; DEG gate {gte,e5}; RRC L_inf 0.0584, K*=5, RRC@100 0.7445,
  RRC@1000 0.9416, range 0.058–0.372; two-tax rho -0.59 n=7 p=0.16; trap rho +0.29 n=7 p=0.53) trace
  **exactly** to their `extra_*/` sources. Do not touch the values.
- **N2 fixed:** no false "lowest reading cost" superlative anywhere; egemma is "the capability corner,
  not the cheapest reader," bge-m3 (2.0×) is cheapest admitted.
- **B2 held:** intro/Results/conclusion all say the best RBO "any model" reaches, never "best model."
- **N1 fixed:** "statistically indistinguishable" → "nearly tied (within 0.01)."
- **Non-significant correlations** kept out of abstract/intro/conclusion; tagged n.s./descriptive in
  body and Limitations. Do NOT promote them.
- **Both leaderboard tables (81 cells)** match `headline_numbers.csv` exactly.
- **All 25 figures** referenced exist under `paper/figures/` (incl. cp_fig18-21).
- **MT-of-question null framing** and **concept-lens mechanical-cap flag** remain correct.
- **TODO-fenced numbers** (corpus dedup 14,401; human-eval) remain honestly fenced; do not un-fence
  until dumped to `reports/`.

---

## Bottom line for the writer

**N2 and B2 are RESOLVED**, and so are N1 and the round-2 T-NEW (XRC population caveat). Every new
round-3 number — cost frontier, RRC budget, degeneracy gate, two-tax — traces exactly to its
`extra_*/` source, the non-significant correlations are quarantined out of abstract/intro/conclusion,
and the 81 leaderboard cells still match. The single new issue is non-blocking: Fig.21's caption/prose
says "**sibling**-confusion rate" but the plotted/quoted quantity is the general `confusion_rate`
(value 0.182/0.068 is correct; just drop "sibling-"). Optionally tweak the T-MINOR universal-blind
language list. Everything else should NOT be "fixed."
