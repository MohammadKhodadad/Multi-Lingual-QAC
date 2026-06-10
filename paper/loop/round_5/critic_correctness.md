# Correctness review (round 5)

Reviewer #2 (correctness). Re-audited the round-5 revision of `paper/main.tex` against the canonical
data under `reports/runs/{chem_patents,alias_graph}/`, with the conductor's focus on:

1. The **four round-4 honesty corrections (C1–C4)** — are they correctly applied and do they match
   source?
2. The **NEW appendix robustness table** (`tab:robust`, `app:robust`) — do all cells match
   `extra_robustness_appendix/robustness_table.csv` exactly?
3. The **merged/stitched figures** (`cp_fig06_07_mate`, `cp_fig09_10_collapse`) and the **radar
   deletions** (cp_fig14, ag_fig10) — did the figure surgery break any number or reference?
4. The **abstract / intro / conclusion** — do they stay clean of the caveated C1–C4 stats?

Bash was read-only (python json/csv inspection only; no writes, no evals, no API, no git-mutate).

## Headline verdict

**C1, C2, C3 are correctly applied and match source exactly. The appendix robustness table matches
`robustness_table.csv` cell-for-cell. The merged figures and radar deletions broke nothing (all 23
`\includegraphics` targets exist; 0 broken `\ref`; all 38 unique refs resolve). Abstract / intro /
conclusion are clean of every caveated C1–C4 stat (they keep only the protected "+0.96, robust"
surface).**

**One NEW non-blocking nuance (M1, C4 τ-band rounding):** the body and `cp_fig18` caption present the
admitted-stable band as `τ∈[0.39,0.43]`, but the source raw value is `tau_admitted_stable_range_raw =
[0.385, 0.43]` and the conductor's C4 spec lists `[0.385,0.430]`. The lower bound is rounded
0.385→0.39 (and the bge-cheapest upper bound 0.435→0.44, presented as `[0.33,0.44]` vs raw
`[0.33,0.435]`). These are 2-decimal-grid roundings the writer made deliberately
(writer_notes round 5 states "stable tau in [0.39,0.43]"). The 0.385→0.39 rounding **narrows** the
claimed-stable band (conservative, honest direction); the 0.435→0.44 rounding **widens** the
bge-cheapest band by one grid step (mildly anti-conservative). Neither changes any conclusion, but
both deviate from the source raw value and from the C4 spec, so I log it as a NEW low-severity
mismatch for the writer to either (a) restore the raw 3-decimal values, or (b) add "(rounded to the
0.005 τ-grid)" once. **Not blocking.**

**Counts: MISMATCH = 1 (M1, low-severity τ-band rounding; conclusions unaffected). UNTRACEABLE = 0.**

---

## Blocking issues (MISMATCH / UNTRACEABLE) — none

No conclusion-changing MISMATCH and no UNTRACEABLE numbers. M1 below is a sub-grid rounding nuance, not
a blocker.

## NEW non-blocking item

### M1 (C4 τ-band, rounding vs source raw). Body + `cp_fig18` caption present rounded τ-grid endpoints.
- **Where:** main.tex L632–633 (body) and L656–657 (`cp_fig18` caption): "admitted set is stable for
  $\tau \in [0.39, 0.43]$" and "\texttt{bge-m3} remains the cheapest admitted reader for $\tau \in
  [0.33, 0.44]$".
- **Source:** `extra_cost_frontier/tau_sweep_summary.json`: `tau_admitted_stable_range_raw = [0.385,
  0.43]` (paper rounds 0.385→0.39); `tau_cheapest_bge_range_raw = [0.33, 0.435]` (paper rounds
  0.435→0.44). The conductor's C4 spec lists the raw `[0.385,0.430]` and `[0.33,0.435]`.
- **Direction:** 0.385→0.39 narrows the claimed-stable band (conservative). 0.435→0.44 widens the
  bge-cheapest band by one grid step (the only mildly anti-conservative one). Both are within the
  0.005 fine-grid resolution; the writer chose 2dp display intentionally (writer_notes round 5).
- **Severity:** Low. The qualitative C4 claims — narrow band, bge-m3 cheapest in the mid-band,
  granite flip below ~0.33 (granite CLIR@10 = 0.3285 → admitted at τ≤0.3285), only embeddinggemma
  above ~0.45, egemma corner τ-invariant — are all correct against source.
- **Minimal fix (pick one):** restore `[0.385,0.43]` and `[0.33,0.435]`; OR keep the 2dp display and
  append "(rounded to the 0.005 τ-grid; raw $[0.385,0.43]$ / $[0.33,0.435]$)" at first use. Do NOT
  touch the granite-flip or egemma-corner clauses — those are correct.

---

## C1–C4 correction-verification table

Tolerance ≤ 0.006 on [0,1] metrics; depths/counts exact. Source files under
`reports/runs/chem_patents/experimental_plots/`.

### C1 — separability sign-stability (presented as WIDE, not tight)
| element | paper value | source file | source value | status |
|---|---|---|---|---|
| P(r>0) sign-stability | 0.9997 (L1051, L1311, L1328) | `extra_robustness_appendix/summary.json` `A1.sign_stability_P_r_gt_0` | 0.9997 | MATCH |
| point r (headline/table) | 0.96 (L74, L1017, L1296, L1328) | `summary.json` `A1.point_r_n7` 0.9577 → 2dp | 0.9577 | MATCH |
| point r (n=7 narrative) | 0.958 (L1049) | `A1.point_r_n7` 0.9577 → 3dp | 0.9577 | MATCH |
| 95% CI | [0.73,1.00] (L1022, L1051, L1328) | `A1.ci95` | [0.7301, 0.9977] | MATCH |
| framed as WIDE not tight | "wide" L1022/L1051/L1328; "report the sign … not the magnitude" | `summary.json` headline "sign-stability is the load-bearing read, not the CI width" | — | MATCH |
| r(n9) drop-collapsers carried | "+0.958 on n=7"; abstract "robust to dropping the two collapsed encoders" | `A1.r_n9` 0.8877 (n9) vs 0.9577 (n7) | consistent | MATCH |

### C2 — ARI@100 egemma vs qwen3 (TIE), L∞ kept distinct
| element | paper value | source file | source value | status |
|---|---|---|---|---|
| ARI@100 egemma | 0.229 (L706, L721, L1033, L1089, L1110) | `extra_ari_decomposition/summary.json` `ARI_at_100_by_model.embeddinggemma` | 0.2286 | MATCH |
| ARI@100 qwen3-0.6B | 0.233 (same sites) | `ARI_at_100_by_model.qwen3-0.6B` | 0.2326 | MATCH |
| gap | 0.004 (L706, L1090, L1332) | `extra_robustness_appendix/summary.json` `A5.gap_qwen3_minus_egemma` | 0.004 | MATCH |
| gap 95% CI straddles 0 | [-0.174, 0.176] (L707, L1332) | `A5.gap_ci95` | [-0.174, 0.1762]; `gap_ci_includes_zero: true` | MATCH |
| order-prob P | 0.519 (L1332) | `A5.order_prob_P_egemma_lower` | 0.5191 | MATCH |
| framed as TIE | "straddles zero", "tied", "(a tie)" L706/L721/L1034/L1090/L1110/L1339 | `A5.honest_read` "report … as effectively tied, not a strict win" | — | MATCH |
| L∞ egemma 0.058 SMALLEST, kept distinct | 0.058 smallest non-deg; next qwen3 0.073 (L703–704, L1036, L1088, L1109) | `extra_ari_decomposition/summary.json` `L_inf_by_model` egemma 0.0584 (min non-deg), qwen3 0.073 | MATCH |

### C3 — separability partial-r | Recall@10 (descriptive, n.s.)
| element | paper value | source file | source value | status |
|---|---|---|---|---|
| partial r | +0.29 (L1040, L1333) | `extra_robustness_appendix/summary.json` `W2.partial_r` | 0.2948 | MATCH |
| p two-sided | 0.57, n.s. (L1040, L1333) | `W2.p_two_sided` | 0.5706 | MATCH |
| zero-order r | 0.96 (L1333) | `W2.zero_order_r` | 0.9577 | MATCH |
| framed DESCRIPTIVE | "we frame this descriptively rather than causally … no longer significant … descriptive correlate, not an effect net of capability" L1037–1042 | `W2.honest_guidance` "Frame … DESCRIPTIVE … Do NOT claim 'not a capability artifact'" | — | MATCH |

### C4 — τ bands (cost frontier)
| element | paper value | source file | source value | status |
|---|---|---|---|---|
| admitted-stable band | [0.39, 0.43] (L632, L656 cap) | `extra_cost_frontier/tau_sweep_summary.json` `tau_admitted_stable_range_raw` | [0.385, 0.43] | **MISMATCH (M1, rounding)** |
| bge-m3 cheapest band | [0.33, 0.44] (L633, L657 cap, L1085) | `tau_cheapest_bge_range_raw` | [0.33, 0.435] | **MISMATCH (M1, rounding)** |
| τ=0.40 admitted set | {bge-m3, qwen3-0.6B, embeddinggemma} (L628–629) | `verify_tau040_admitted_set` | [bge-m3, embeddinggemma, qwen3-0.6B] | MATCH |
| τ=0.40 cheapest = bge-m3 (2.0×) | bge-m3 2.0× (L630) | `verify_tau040_cheapest_is_bge_m3` true; tau=0.4 cheapest_XRC50 2.0 | true / 2.0 | MATCH |
| granite flip below τ≈0.33 | "below $\tau{\approx}0.33$ … flips to granite" (L634, L657, L1085) | granite CLIR@10 0.3285 (admitted at τ≤0.3285); tau=0.3 cheapest=granite 1.25× | flip ≤0.3285 ≈ 0.33 | MATCH |
| only egemma above τ≈0.45 | "above $\tau{\approx}0.45$ only embeddinggemma" (L636) | tau=0.45/0.5 n_admitted=1, set=embeddinggemma | confirmed | MATCH |
| egemma corner τ-invariant | "Only embeddinggemma's … unique maximum-CLIR@10 corner is $\tau$-invariant" (L637, L658) | `egemma_corner_tau_invariant: true`; `max_clir_corner` = embeddinggemma at all τ | true | MATCH |

---

## Appendix robustness table (`tab:robust`) — cell-by-cell vs `robustness_table.csv`

All six rows verified against `extra_robustness_appendix/robustness_table.csv` (and cross-checked
against `summary.json`). Every cell MATCHES to display-rounding.

| table row | paper point | paper resample/stability | source point | source CI / stability | status |
|---|---|---|---|---|---|
| separability r | 0.96 | P(r>0)=0.9997; CI [0.73,1.00] (n=7) | 0.9577 | [0.7301,0.9977]; P(r>0)=0.9997; n=7 | MATCH |
| XRC50 egemma | 3.5 | CI [0.91,12.0] (finite, wide) | 3.5 | [0.9091,12.0]; censored-draw frac 0.0 | MATCH |
| XRC50 bge-m3 | 2.0 | CI [0.53,7.0] (finite, wide) | 2.0 | [0.5294,7.0]; censored-draw frac 0.0 | MATCH |
| XRC50 granite | 1.25 | CI [0.28,12.25] (finite, wide) | 1.25 | [0.2838,12.25]; censored-draw frac 0.0 | MATCH |
| ARI@100 gap (q−e) | 0.004 | CI [-0.174,0.176]; P=0.519 | 0.004 | [-0.174,0.1762]; order-prob 0.5191 | MATCH |
| partial r \| R@10 | 0.29 | p=0.57 (n.s.); zero-order 0.96 | 0.2948 | two-sided p 0.5706; zero-order 0.9577 | MATCH |

The appendix prose (L1307–1319) is also faithful: "P(r>0)=0.9997 … at n=7 the 95% CI is wide";
"three frontier XRC50 medians have finite but wide CIs (censored-draw fraction = 0)"; "ARI@100 gap CI
straddles zero (a tie)"; "separability–CLIR association no longer significant once Recall@10 is
partialled out." All match `summary.json`. `gates_all_pass: true` supports the "checks pass" tone.

---

## Figure surgery — merged/stitched + radar deletions (no number broken)

**Merged figures (numbers re-traced to source):**
- `cp_fig06_07_mate.png` (L741–747): caption "pooled mate-hit@10 is 0.38" and "15% of (query, model)
  pairs never reach one in the top-1000." Source `chem_patents/key_findings/EXECUTIVE_SUMMARY.md` L36–37:
  "Pooled mate-hit@10 = 0.38; 15% of (query, model) pairs never surface a foreign twin even in the
  top-1000." Best twin-finder egemma median first-foreign rank 5 (EXEC L38–39). **All MATCH.** This
  float replaces the original `fig06_mate_retrieval`/`fig07_first_foreign_rank` pair (EXEC L93–94);
  the stitched PNG exists on disk.
- `cp_fig09_10_collapse.png` (L969–977): caption "up to 49× the corpus base rate" and "same-language
  noise out-ranks the gold on 60% of queries." Source EXEC L43–46: "over-fetch their own language up
  to 49× the corpus base rate; same-language noise out-ranks the gold on 60% of queries." Replaces
  `fig09_language_collapse`/fig10 pair. **All MATCH.** Stitched PNG exists.

**Radar deletions:** cp_fig14 and ag_fig10 (`fig:cp_radar`, `fig:ag_radar`) are removed; only a
CUT-NOTE comment survives (L911–913). Independent checks:
- 0 broken `\ref` to deleted labels (`fig:cp_rank`, `fig:cp_distractor`, `fig:cp_radar`,
  `fig:ag_radar`): grep returns NONE.
- All 38 unique `\ref` targets resolve against 40 `\label` definitions: UNRESOLVED = NONE.
- All 23 `\includegraphics` targets exist under `paper/figures/`: 23/23 OK.
- `app:robust` referenced 5×, all resolve.

No number anywhere depends on the deleted radars; their "where each model wins" beat is carried by
Tables 1–2 + the aggregation ribbon `cp_fig17` (which itself traces — egemma rank ranges [1,4]).

---

## Abstract / Intro / Conclusion cleanliness (caveated C1–C4 stats must be ABSENT)

Programmatic scan of the three protected blocks for the caveated tokens (0.9997, [0.73…, 0.229,
0.233, [-0.174, p=0.57 / 0.5706, "partial", 0.385/0.430, "0.33, 0.44", "tau", per-route): **all three
blocks return NONE (clean).** They retain only the protected surface
`r(cross-language AUC, CLIR@10)=+0.96, robust` (abstract L74, conclusion L1296), where "robust" is
now anchored to the sign-robust meaning defined in §7 body (L1023 "‘Robust’ here means sign-robust
under resampling, not tightly estimated") and the appendix. This matches the round-5 writer_notes
intent and is the correct quarantine. **PASS.**

---

## Design-soundness findings (per the four corrections)

1. **C1 (separability sign-stability).** Sound and now correctly framed. The load-bearing read is the
   sign (P(r>0)=0.9997), the point estimate is reported (0.96 / 0.958 on n=7), and the CI is
   explicitly called WIDE at n=7 ([0.73,1.00]) — exactly the honest framing for a 7-point correlation.
   The redefinition of "robust" as "sign-robust under resampling, not tightly estimated" (L1023) is
   the right hedge, and the abstract/conclusion keep "+0.96, robust" with that meaning anchored in
   body+appendix. Leave alone.

2. **C2 (ARI@100 tie).** Sound. Every ARI@100 site reads as a tie (egemma 0.229 / qwen3 0.233, gap
   0.004, CI [-0.174,0.176] straddles 0), and crucially egemma KEEPS its separate smallest
   alignment-only floor (L∞=0.058, next qwen3 0.073) as a distinct, NON-tied claim — the writer did
   not over-correct by erasing egemma's genuine floor advantage. The ARI@100 tie and the L∞ distinction
   are two different quantities and the paper keeps them textually separate. Leave alone.

3. **C3 (collinearity / partial r).** Sound. §7 now frames separability as a DESCRIPTIVE correlate:
   partial r=+0.29 (p=0.57, n.s.) once Recall@10 is partialled out, with explicit "not an effect net
   of general retrieval capability." This is exactly what `W2.honest_guidance` instructs ("Do NOT
   claim 'not a capability artifact'"), and the paper does not make that forbidden claim anywhere.
   The qualitative mechanism bridge ("the lever is at the embedding level") is retained but sits
   beside the partial-r caveat, not as a causal claim. Leave alone.

4. **C4 (τ-band).** Substantively sound; only M1 (the 0.385→0.39 / 0.435→0.44 grid rounding) is a
   nuance. The narrow-band framing, the bge-m3-cheapest-in-mid-band claim, the granite flip below
   ~0.33 (granite CLIR@10 = 0.3285, so admitted at τ≤0.3285), the "only egemma above ~0.45," and the
   "only egemma's max-CLIR corner is τ-invariant" are all correct against source. The writer never
   says "the rule is robust" — it correctly says the snapshot "survives only over a narrow band." Fix
   M1 (one of the two minimal options) and this is fully clean.

---

## Overlooked / confounds / threats-to-validity (carried; all still HANDLED)

- **T1 home-advantage vs gold-availability (HANDLED, carried).** §Results L566 "much of which gold
  availability shapes … though a residual encoder bias remains"; §Analysis carries the −0.57
  descriptive availability-residual slope. Resolved.
- **T2 directional asymmetry as corpus composition (HANDLED, carried).** §Results L591 "partly tracks
  corpus composition (en 46% vs zh 0.4%) … not only encoder behaviour." Resolved.
- **T3 ARI is a re-presentation, not an independent measurement (HANDLED, carried).** Presented as "a
  natural exhaustive reading of the same shortfall" with RRC as the measured object; identity closes
  to 1.0 for all 9 (`identity_closes_all_models: true`), regression-checked. No over-claim.
- **T4 per-route thinness (HANDLED, carried).** de n_same=7, zh=2, es=0 flagged twice; per-route XRC
  "indicative only"; es XRC undefined and "never imputed." Resolved.
- **T5 parallel-gold equivalence audit (HONESTLY DEFERRED, carried).** Limitations flags claim-level
  equivalence + spot-check as future work; in needs_eval, so DONE/deferred per the critic contract.
- **T6 stale baseline run never cited (CLEAN, carried).** `20260601-235117_137questions` is still not
  pulled into the paper; all numbers come from the 23,487-corpus key_findings + extra_* dirs. No leak.
- **C5 / corpus-dedup / human-eval (FENCED, carried).** corpus dedup 14,401, human-eval 8.33/10,
  97/100, +4.3pp, and MMTEB/MIRACL/NeuCLIR transfer all remain `%`-comment-fenced (§3, §3 pipeline,
  Limitations, Appendix C5) — never in rendered prose. No active `\todo{}` in rendered text.

---

## Verified-correct (leave these alone)

- **C1, C2, C3 corrections** — all values and all framing match source exactly (tables above).
- **Appendix robustness table (`tab:robust`)** — all six rows match `robustness_table.csv` cell-for-cell.
- **Merged figures** — `cp_fig06_07_mate` (mate-hit@10 0.38, 15% lost) and `cp_fig09_10_collapse`
  (49×, 60%) trace to EXECUTIVE_SUMMARY; both stitched PNGs exist.
- **Radar deletions clean** — 0 broken refs, all 38 refs resolve, 23/23 figures on disk, app:robust
  ×5 resolve.
- **Abstract/intro/conclusion** — clean of all caveated C1–C4 tokens; keep only "+0.96, robust"
  (sign-robust meaning anchored in body+appendix).
- **Carried-correct framing** — XRC population caveat, MT-null (−0.044, p=0.13, "insignificant"),
  concept-lens mechanical cap, degeneracy gate {gte-base, e5-large-instruct}, two-tax / trap rho
  non-significant and quarantined, leaderboard cells (Tables 1–2) — all unchanged and correct.

---

## Bottom line for the writer

Three of the four honesty corrections (C1 separability sign-stability, C2 ARI@100 tie with L∞ kept
distinct, C3 partial-r descriptive framing) are applied correctly and match source to the decimal,
and the NEW appendix robustness table matches `robustness_table.csv` cell-for-cell. The merged figures
and radar deletions broke nothing — every figure exists, every ref resolves, and the mate/collapse
numbers (0.38, 15%, 49×, 60%) still trace to the executive summary. The abstract/intro/conclusion are
clean of every caveated stat and keep only the protected "+0.96, robust" surface. The single NEW item
is **M1**: the C4 τ-bands are displayed rounded to the 0.005 grid (`[0.39,0.43]`/`[0.33,0.44]`) rather
than the source raw `[0.385,0.43]`/`[0.33,0.435]`; this narrows the stable band (conservative) and
widens the bge band by one grid step, changes no conclusion, and is fixable with one clause or by
restoring the raw values. **MISMATCH = 1 (low-severity rounding, M1). UNTRACEABLE = 0.**
