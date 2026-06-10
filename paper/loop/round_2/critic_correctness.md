# Correctness review (round 2)

Reviewer #2 (correctness). Re-audited every numeric claim and figure in the **round-2 revision**
of `paper/main.tex` against the canonical data under `reports/runs/{chem_patents,alias_graph}/`,
with special attention to the **NEW round-2 headline numbers** sourced from the
`experimental_plots/extra_*/` directories that did not exist when round-1 was written:

- chem: `extra_xrc_reading_cost/`, `extra_aggregation_invariance/`,
  `extra_correlation_robustness/`, `extra_directional_hub/`
- alias: `extra_availability_residual/`, `extra_confusion_severity/`, `extra_joint_failure/`

I recomputed/cross-checked from the on-disk `summary.json` + `*.csv` in each extra dir and from
the two `key_findings/headline_numbers.csv`. I also re-read the round-1 review to confirm B1/B2/B3.

## Headline verdict

**B1, B2, B3 are ALL fixed** (details below). **Every new round-2 headline number traces
exactly** to its `extra_*/` source (XRC50=3.5×, RRC@100=0.7445 / lost@1000=0.0584,
aggregation rank-range [1,4], availability slope −0.57, auc_cross~clir +0.958 on n=7,
joint-failure 257/44.4%/79.4%/55.6%, confusion-severity 18.1%/6.2%/2.9×). The fragile
correlations are represented **honestly** — the writer explicitly says the over-rep~CLIR and
home-adv~RBO correlations *do not survive* dropping the degenerate encoders and labels them
"descriptive," while flagging the separability link as the one *robust* correlation. The leaderboard
tables (81 cells) and all carried-over round-1 numbers still match.

**Two NEW issues** surfaced, both about statistical/internal-consistency *framing* of otherwise-real
numbers (neither is a fabricated value):

- **N1 (soundness overclaim — must fix wording):** the intro calls French and English
  *"statistically indistinguishable as targets."* **No significance test exists** in the source —
  `extra_directional_hub.py` computes only the means (fr 0.375, en 0.367) and a `fr > en` boolean.
  The source's own intended phrasing uses a loose "~" ("fr 0.375 ~ en 0.367"), not "statistically
  indistinguishable."
- **N2 (internal contradiction / MISMATCH — must fix):** the Deployment section claims
  `embeddinggemma` "has the **lowest non-degenerate reading cost** (XRC50 3.5×)." It does **not** —
  among the eight finite-XRC models, **five are lower** than 3.5×: granite-278m (1.25×), bge-m3
  (2.0×), LaBSE (2.48×), SapBERT (2.63×), qwen3-0.6B (3.25×). embeddinggemma is 6th, not lowest.
  granite is non-degenerate (CLIR@10 = 0.329, `best_xrc50_model` in the source), and the Results
  section two pages earlier *correctly* says "the cleanest model on this axis is granite-278m
  (1.25×)," so the paper contradicts itself.

**Counts: MISMATCH = 1 (N2), soundness-overclaim = 1 (N1), UNTRACEABLE (not already TODO-fenced) = 0.**
The pre-existing `\todo{}`-fenced numbers (corpus dedup 14,401; human-eval 8.33/10 etc.) remain
honestly fenced and are NOT counted as blocking. All `needs_eval.md` items
(W4-formula-injection, CLIRMRS-external-validation, XRC-conformal, hop-distance-law,
equivalence-audit) are treated as DONE per the contract.

---

## Blocking issues (MISMATCH / overclaim) — must fix

### N2 (MISMATCH, internal contradiction). "embeddinggemma has the lowest non-degenerate reading cost."
- **Where:** §Deployment, "Deploy embeddinggemma, on per-axis dominance," line ~898–899:
  *"…and has the lowest non-degenerate reading cost (XRC50 3.5×)."*
- **What the data says:** `extra_xrc_reading_cost/xrc_per_model.csv`, sorted by XRC50 among
  non-degenerate (finite-XRC) models: granite-278m **1.25**, bge-m3 **2.0**, LaBSE **2.48**,
  SapBERT **2.63**, qwen3-0.6B **3.25**, **embeddinggemma 3.5**, nomic 11.5, e5 97.75. So
  embeddinggemma is the **6th**-lowest of eight, not the lowest — five non-degenerate models read
  *fewer* documents to reach a foreign twin. `summary.json` records `"best_xrc50_model":
  "granite-278m"`, and the Results section line ~530 already states this correctly: *"The cleanest
  model on this axis is granite-278m (1.25×)."* So the Deployment sentence is both wrong and
  self-contradictory.
- **Minimal fix:** drop "lowest non-degenerate reading cost" from the embeddinggemma justification.
  Replace with a defensible per-axis claim, e.g. *"…is the best twin-finder (median first-foreign
  rank 5) and keeps a low reading cost (XRC50 3.5×, vs 11.5×–97.75× for nomic/e5)."* Do not assert
  it is the **lowest** — granite is lower. (embeddinggemma genuinely tops CLIR@10, separability, and
  mate-MRR, so the recommendation survives; only this one superlative is false.)

### N1 (soundness overclaim — must fix wording). "French and English are statistically indistinguishable as targets."
- **Where:** §Results "Retrieval is anisotropic," line ~509: *"French (0.375) and English (0.367)
  are statistically indistinguishable as targets."* (This is the B1 *replacement* text.)
- **What the data says:** `extra_directional_hub.py` computes pooled column means
  (fr 0.375, en 0.367) and a single boolean `fr_gt_en_gate = fr > en`. **There is no t-test,
  bootstrap CI, or permutation test** anywhere in the script or its `summary.json`. The per-cell N
  is tiny (en→de = 7, zh→de = 4, zh→zh = 2), so a formal "indistinguishable" claim would in fact be
  unsupported-but-plausible — yet it was never tested. The source's own
  `writer_replacement_sentence` deliberately says *"no single language is a clean 'easiest target'
  (fr 0.375 ~ en 0.367)"* — the loose "~", not "statistically indistinguishable."
- **Minimal fix:** change "statistically indistinguishable" → "nearly tied" or "within 0.01 of each
  other (fr 0.375 ≈ en 0.367)." This keeps the (correct) B1 point — no language is a clean easiest
  target — without claiming a hypothesis test that was not run. A hostile reviewer will otherwise ask
  "indistinguishable under which test, at what N?"

> Everything below is verified-correct or a non-blocking threat.

---

## Number-trace table (round-2 NEW numbers + spot-checks of carried-over claims)

Tolerance ≤ 0.006 on [0,1] metrics; depths/counts exact. "source" = the named `extra_*/` file
unless noted.

| claim (paper) | paper value | source file | source value | status |
|---|---|---|---|---|
| **XRC50 embeddinggemma** | 3.5× (depth 2→7) | chem `extra_xrc.../summary.json`, `xrc_per_model.csv` | 3.5 (D50_same 2, D50_cross 7) | MATCH |
| XRC50 granite (cleanest) | 1.25× | `xrc_per_model.csv` granite | 1.25 | MATCH |
| XRC50 nomic / e5 | 11.5× / 97.75× | `xrc_per_model.csv` | 11.5 / 97.75 | MATCH |
| gte XRC undefined | undefined | `xrc_per_model.csv` gte XRC50 blank, censored=True | inf/NaN | MATCH |
| "lowest non-degenerate reading cost = egemma" | (implied egemma) | granite XRC50 1.25 < 3.5 | granite lower | **MISMATCH (N2)** |
| **RRC@100 embeddinggemma** | 0.7445 | `rrc_per_model.csv`, `summary.json` | 0.7445 | MATCH |
| RRC@1000 embeddinggemma | 0.9416 | `rrc_per_model.csv` | 0.9416 | MATCH |
| lost@1000 (1−RRC@1000) | 5.84% | `summary.json` lost_at_1000 | 0.0584 | MATCH |
| e5 lost@1000 ("loses 37.2%") | 37.2% | `rrc_per_model.csv` e5 lost_at_1000 | 0.3723 | MATCH |
| "top-100 re-ranker leaves 25% on the table" | 25% | 1−RRC@100 = 1−0.7445 | 0.2555 | MATCH |
| pooled mate-hit@10 | 0.38 | chem round04 (carried) | 0.375 | MATCH |
| 15% never in top-1000 (8-model pool) | 15% | `extra_xrc/summary.json` consistency_checks | 0.1542 | MATCH |
| median first-foreign rank (egemma) | 5 | round04 / `xrc` | 5.0 | MATCH |
| **aggregation rank-range egemma** | [1,4] | `extra_aggregation.../summary.json`, `aggregation_ranks.csv` | [1,4] (clirmrs 1, borda 3, equal 4, wta 1) | MATCH |
| egemma rank-1 under 2/4 schemes | (implied) | `summary.json` rank1_under_n_of_4 | 2 | MATCH |
| wta column "contaminated by gte" | (qualitative) | `summary.json` caveat_winner_take_all | gte wins parity/mt-robust | MATCH |
| **availability slope** | −0.57 | `extra_availability.../summary.json` | −0.5719 | MATCH |
| availability Pearson r / R² | (descriptive) | `summary.json` | r −0.8725, R² 0.7613 | MATCH (not cited as inferential — good) |
| mean home advantage (alias) | +0.32 | `summary.json` mean_home_adv | 0.324 | MATCH |
| Chinese: 8% avail, +0.47 home adv | 8% / +0.47 | `availability_regression.csv` zh | 0.08 / 0.4746 | MATCH |
| own-lang recall range (alias) | 0.63–0.82 | `availability_regression.csv` same_recall | 0.63 / 0.822 | MATCH |
| foreign-lang recall range (alias) | 0.35–0.47 | `availability_regression.csv` cross_recall | 0.348 / 0.473 | MATCH |
| in-English avail / de-es-zh avail | 42% / 8–10% | `availability_regression.csv` | 0.42 / 0.09,0.10,0.08 | MATCH |
| **auc_cross~clir robust (n=7)** | +0.958 (Spearman 0.964) | `extra_correlation.../correlation_robustness.csv` | pearson_n7 0.958, spearman_n7 0.964 | MATCH |
| auc_cross~clir (n=8 pool) | +0.96 | `correlation_robustness.csv` | 0.961 | MATCH |
| home-adv~RBO collapses on n=7 | "collapses" | `correlation_robustness.csv` pearson_n7 | 0.186 (was −0.846) | MATCH (honest) |
| over-rep~CLIR "flips sign" on n=7 | "flips sign" | `correlation_robustness.csv` pearson_n7 | +0.419 (was −0.60) | MATCH (honest) |
| **directional hub: fr 0.375 > en 0.367** | fr 0.375 / en 0.367 | `extra_directional_hub/hub_scores.csv` | 0.375 / 0.3673 | MATCH |
| zh / de hub | 0.350 / 0.309 | `hub_scores.csv` | 0.35 / 0.3091 | MATCH |
| "fr & en statistically indistinguishable" | (stat claim) | no test in `extra_directional_hub.py` | means only, no test | **OVERCLAIM (N1)** |
| hardest direction en→de | 0.12 | `extra_directional_hub/summary.json` | 0.125 | MATCH |
| most asymmetric pair de↔zh | +0.23 | `summary.json` most_asymmetric_pair.gap | 0.234 | MATCH |
| corpus composition en 46% / zh 0.4% | 46% / 0.4% | `summary.json` corpus_composition_caveat / CORPUS_SHARE | 0.461 / 0.0043 | MATCH |
| **joint failure: 257 confused cases** | 257 | `extra_joint_failure/summary.json` | 257 | MATCH |
| same-lang sibling modal (114/257) | 44.4% | `summary.json` cell_fractions | 0.4436 (114) | MATCH |
| siblings total | 79.4% | `summary.json` sibling_total_frac | 0.7938 | MATCH |
| same-language total | 55.6% | `summary.json` same_language_total_frac | 0.5564 | MATCH |
| universal-blind 16/132 (12%), 14 structure | 16/132, 14 | `summary.json` A8_universal_blind | 16/132, structure 14 | MATCH |
| blind "predominantly fr, zh, de" | fr/zh/de | `universal_blind_profile.csv` | fr 5, zh 4, de 3, es 3, en 1 | MATCH (minor: es ties de at 3; see threat T-MINOR) |
| **confusion severity: sibling 18.1% vs parent 6.2%** | 18.1% / 6.2% | `extra_confusion_severity/summary.json` | 0.1813 / 0.0624 | MATCH |
| sibling:parent ratio | 2.9× | `summary.json` pooled_sibling_to_parent_ratio | 2.91 | MATCH |
| egemma sibling vs parent | 6.1% vs 1.5% | `severity_split.csv` embeddinggemma | 0.0606 / 0.0152 | MATCH |
| **CP leaderboard (9 rows × 5 cols)** | 45 cells | chem `headline_numbers.csv` | all 45 | MATCH |
| **alias leaderboard (9 rows × 4 cols)** | 36 cells | alias `headline_numbers.csv` | all 36 | MATCH |
| best CLIR@10 / home +0.55 | 0.50 / +0.55 | chem `headline_numbers.csv` | 0.5024 / 0.5526 | MATCH |
| MT-of-question diff / p | −0.044 / 0.13 | chem round03 (carried) | −0.0444 / 0.1307 | MATCH |
| RBO ceiling alias / CP | 0.39 / 0.19 | alias round01 / chem round05 (carried) | 0.387 / 0.1934 | MATCH |
| 49× over-fetch / 60% noise | 49× / 60% | chem round06/07 (carried) | 48.71 / 0.604 | MATCH |
| oracle CLIR@10 / alias 88% vs 76% | 0.61 / 88% vs 76% | chem round09 / alias round08 (carried) | 0.6119 / 0.879/0.758 | MATCH |
| structure vs role R@10/conf | 0.26/51% vs 0.60/25% | alias round07 (carried) | 0.26/0.51 vs 0.60/0.25 | MATCH |
| AUC confused vs not (alias) | 0.55 / 0.70 | alias round09 (carried) | 0.549 / 0.698 | MATCH |
| benchmark sizes | 132 / 137(57+80) / 24 | headers + code (carried) | 132 / 137 / 24 | MATCH |
| corpus size | 23,487 | chem exec summary (carried) | 23,487 | MATCH |
| corpus dedup 14,401; human-eval 8.33/10 | (TODO-fenced) | not under reports/ | — | TODO-fenced (OK) |

---

## Design-soundness findings (per headline claim)

1. **XRC50 (Fig 15, abstract, intro, deployment).** The metric is sound and well-hedged: the paper
   reports XRC50 (median) as the finite headline and explicitly treats D90/D95 as right-censored
   lower bounds (source `xrc_per_model.csv` confirms many D90/D95 = inf). The ratio-of-depths
   definition matches the source. **One soundness defect: N2** — the *superlative* "lowest
   non-degenerate reading cost" is false (granite 1.25 < egemma 3.5). Fix per N2; the metric itself
   is fine.

2. **RRC (Fig 16, abstract, deployment).** Sound and the cleanest new contribution. `RRC@K =
   P[first foreign twin rank ≤ K]`, and `1 − RRC` as "provably unrecoverable by any top-K
   re-ranker" is a legitimate upper-bound argument. All five derived figures (0.7445, 0.9416,
   0.0584, e5 0.3723, the 25% "left on the table") trace exactly. **Leave alone.**

3. **Aggregation invariance (Fig 17).** This is the right honesty move and is sound: the paper says
   the ranking is aggregation-sensitive (egemma rank-range [1,4], rank-1 only under 2/4 schemes) and
   correctly flags the winner-take-all column as contaminated by gte-base. The recommendation
   explicitly rests on per-axis dominance, not the composite. Source `aggregation_ranks.csv` matches
   row-for-row. **Leave alone** — this directly defuses the "aggregation-sensitivity" reviewer.

4. **Availability residual (Fig 11/ag_fig11).** Correctly labeled **descriptive (n=5 languages),
   not inferential**, in both §Analysis and Limitations. The −0.57 slope, the negative direction
   (least-available zh carries the largest home advantage), and the "residual encoder bias, not
   availability artifact" reading all trace. The writer did **not** upgrade r = −0.87 / R² = 0.76
   to a significance claim — good. **Leave alone.**

5. **Correlation-robustness (the load-bearing +0.96).** This is the central mechanism and it is now
   *honestly* defended: the paper states the separability link is robust to dropping the two
   collapsed encoders (+0.958, Spearman 0.964, n=7) **and** explicitly says the two *bias*
   correlations (over-rep~CLIR, home-adv~RBO) are fragile/descriptive and "do not survive" the same
   drop. This is exactly the right asymmetric treatment — the robust one is promoted, the fragile
   ones demoted, with numbers. **This resolves round-1 threat T5. Leave alone.**

6. **Directional hub (Fig 3, B1 replacement).** The factual replacement (no clean easiest target;
   hardest en→de; most asymmetric de↔zh; corpus-composition caveat) is correct and traces. The
   **only** defect is the word "statistically indistinguishable" (N1) — an untested stat claim on a
   real but untested near-tie. Fix the wording, keep the finding.

7. **Joint failure & confusion severity (Fig 12, prose).** Both new alias analyses trace exactly.
   The severity split is correctly scoped as two-level (sibling/parent), and the paper flags the
   graded hop-distance law as future work (matching `needs_eval` CCI-hop-distance-law). The
   "same-language sibling is the modal failure (44.4%)" is the paper's neatest synthesis and is
   data-faithful. **Leave alone.**

8. **MT-of-question null (Fig 5).** Still framed correctly ("insignificant," "null," "we do not
   claim it helps"). Carried-over from round 1; unchanged; correct.

9. **Concept-lens mechanical cap.** Still flagged correctly ("must not be read as recall quality").
   **Leave alone.**

---

## Overlooked / confounds / threats-to-validity (with the minimal fix each)

**T1 — Home-advantage vs gold-availability confound (NOW HANDLED).** Round 1 asked for a clause
tying the +0.55 home advantage to the availability confound. The round-2 §Results now says the
+0.55 is "much of which gold availability shapes (§Analysis), though a residual encoder bias
remains," and §Analysis carries the full availability-residual analysis. **Resolved.** No fix
needed.

**T2 — Directional asymmetry as corpus composition, not encoder behavior (NOW HANDLED).** Round 1
asked for a sentence noting corpus composition drives the anisotropy. Round-2 §Results line ~511
now says: "What asymmetry exists partly tracks corpus composition (en 46% vs zh 0.4% of documents),
not only encoder behaviour." Traces to `extra_directional_hub` CORPUS_SHARE. **Resolved.**

**T3 — oracle/RRF headroom interpretation (STILL HANDLED).** The "don't ensemble / headroom is
real" tension is kept tight in one paragraph (oracle 0.61 / 88% vs 76%, RRF negative). No fix.

**T4 — Parallel-gold equivalence audit (HONESTLY DEFERRED, now in needs_eval).** The
equivalence-audit-spotcheck is now listed in `needs_eval.md` (added r1) and in Limitations
("a spot-check that the parallel human-translated golds are claim-level equivalent … remain future
work"). Per the critic contract this is **DONE/deferred**, not a missing experiment. No correctness
fix required.

**T5 — Small-n correlations stated as mechanisms (NOW HANDLED — see soundness #5).** The round-2
draft adds the drop-the-collapsers robustness check and demotes the fragile correlations to
"descriptive." This is the strongest improvement over round 1. **Resolved.**

**T6 — Stale baseline run never cited (STILL CLEAN).** Re-checked: the paper still cites only the
23,487-corpus key_findings numbers; the 1,110-doc `20260601-235117_137questions` baseline is never
pulled in. No leak. No fix.

**T7 — Two RBO ceilings stay separated (HANDLED, B2 fixed).** 0.39 (alias) and 0.19 (CP) remain
attributed to their own benchmarks; the B2 "best model → any model" wording fix is in place
(intro line ~107, conclusion line ~1009).

**T-MINOR (non-blocking) — universal-blind language ordering.** §Analysis says the blind core is
"predominantly in French, Chinese, and German." Source `universal_blind_profile.csv` has fr 5,
zh 4, **de 3, es 3 tied**. fr/zh are unambiguously the top two; picking de over es at the 3-tie is
arbitrary but harmless. Optional fix: "predominantly French and Chinese (with German and Spanish
tied behind)," or just "predominantly French and Chinese." Not blocking.

**T-NEW — XRC same-language population (n=57) vs cross (n=137).** XRC50 divides a cross-language
depth (over 137 cross-gold queries) by a same-language depth (over only 57 originals, the only ones
with same-language gold) — `extra_xrc/summary.json` documents both populations. The two depths come
from **different query populations**, so XRC is a *population-level* ratio, not a paired
within-query ratio like the +0.55 home advantage. The paper does not currently say this. A hostile
reviewer could read "you scan 3.5× as many documents" as a paired statement. **Minimal fix:** one
clause in the XRC paragraph or caption — "D_same is over the 57 same-language-gold queries and
D_cross over the 137 cross-gold queries (a population-level, not paired, ratio)." Low severity, but
it pre-empts a methods question and mirrors the B3 footnote already added for the home advantage.

---

## Verified-correct (leave these alone)

- **All seven NEW round-2 headline numbers** (XRC50 3.5×; RRC@100 0.7445 / RRC@1000 0.9416 /
  lost 0.0584; aggregation rank-range [1,4]; availability slope −0.57; auc_cross~clir +0.958 on n=7;
  joint-failure 257/44.4%/79.4%/55.6%; confusion-severity 18.1%/6.2%/2.9×) trace **exactly** to the
  `extra_*/` sources. Do not touch the values.
- **Both leaderboard tables (81 cells)** still match `headline_numbers.csv` exactly.
- **The honest treatment of fragile correlations** (robust separability promoted; bias correlations
  demoted to "descriptive," "do not survive dropping the collapsers") is exactly right — this is the
  single best round-2 improvement. Do **not** soften it.
- **B1 fixed:** "English is the easiest target" is gone; replaced with the correct
  "no single language is a clean easiest target (fr 0.375, en 0.367)" — only the "statistically
  indistinguishable" wording (N1) needs softening.
- **B2 fixed:** intro/conclusion now say "the best cross-lingual RBO any model reaches/achieves,"
  not "the best model."
- **B3 fixed:** the Fig-1 footnote now states MoLIR@10 is defined only on the 57 originals and the
  home advantage is paired within them.
- **All 23 figure files referenced exist** under `paper/figures/`.
- **MT-of-question null framing** and **concept-lens mechanical-cap flag** remain correct.
- **TODO-fenced numbers** (corpus dedup 14,401; human-eval) remain honestly fenced; do not un-fence
  until dumped to `reports/`.

---

## Bottom line for the writer

Fix **N2** (the one real MISMATCH: embeddinggemma is **not** the "lowest non-degenerate reading
cost" — granite-278m at 1.25× is lower and the Results section already says so; drop the
superlative) and soften **N1** ("statistically indistinguishable" → "nearly tied"; no test was
run). Optionally add the one-clause T-NEW caveat that XRC is a population-level (not paired) ratio,
and tweak the T-MINOR language list. **B1/B2/B3 are all resolved, every new round-2 number is
data-faithful, and the fragile correlations are represented honestly.** Everything else should not
be "fixed."
