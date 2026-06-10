# Story (round 5)

## Changes since round 4

**Round 5 is a polish + honesty-correction + figure-trim round, NOT a new-analysis
round.** The paper has STRUCTURALLY CONVERGED — two consecutive clean rounds
(round-3 and round-4 critics: 0 correctness mismatches, novelty defensible on every
contribution, the only live liability is float overload). The spine, thesis,
contribution list, section order, and every empirical object are FROZEN. **No new
claim is added this round.** Round 5 does exactly three things: (1) make four
existing claims *more honest* per the reporter's verified resamples, (2) make the
paper *shorter* (4-float net cut to land inside the 8-page industry-track body), and
(3) clear the three cheap residuals the round-4 novelty/correctness critics named
(ρ_k cite, fig22 caption reconcile, §6.1 ARI trim).

The round-4 reporter independently re-verified five CPU-only resamples against the
on-disk CSV/JSON (`extra_robustness_appendix/robustness_table.csv` + `summary.json`,
`extra_cost_frontier/tau_sweep.csv` + `tau_sweep_summary.json`). **Two came back
WEAKER than the round-4 prose assumed**, and they reshape the wording of two pillars.
The four mandatory wording corrections (C1–C4 below) are the substance of round 5;
each is grounded in a verified file and is a *softening*, never a new assertion.

### The four mandatory honesty corrections (C1–C4) — verbatim wording guidance

**C1 — Separability headline = SIGN-STABILITY, not CI width.**
- *Verified:* point $r=0.9577$, 95% CI $[0.730, 0.998]$ (WIDE at $n=7$), sign-stability
  $P(r>0)=0.9997$ (`robustness_table.csv` row 2; `summary.json` key `A1_separability_r`).
- *Sites:* §7 line 1025–1028; the cp_fig11 caption line 1045–1046; abstract line 74;
  conclusion line 1278.
- *Required framing:* lead with sign-stability, present the CI honestly as wide.
  Suggested: *"Across model-level bootstrap resamples the separability–floor
  correlation is positive in 99.97\% of draws (point $r=0.96$); with only seven
  non-degenerate models the confidence interval is correspondingly wide
  ($[0.73, 1.00]$), so we report the sign as the robust finding rather than its
  magnitude."* The body keeps saying "$+0.96$, robust"; the appendix table carries the
  CI and $P(r>0)$. **Do NOT present $[0.730,0.998]$ as a tight/precise estimate.** The
  word "robust" in-body now means *sign-robust under resampling*, not *tightly
  estimated* — make that explicit once where the number is introduced.

**C2 — egemma and qwen3 are TIED on ARI@100; egemma keeps its DISTINCT $L_\infty$ win.**
- *Verified:* ARI@100 gap (qwen3 − egemma) $= 0.004$, 95% CI $[-0.174, 0.176]$
  (INCLUDES 0), order-prob $P(\text{ARI}_{\text{egemma}} < \text{ARI}_{\text{qwen3}})
  = 0.5191$ (near coin-flip); egemma $=0.2286$, qwen3 $=0.2326$
  (`robustness_table.csv` row 6; `summary.json` key `A5_ARI100_gap_egemma_vs_qwen3`,
  `gap_ci_includes_zero: true`).
- *Sites:* §6.1 lines 692–694 and 705; §8 line 1078 ("lowest $\mathrm{ARI@}100$
  (0.229) of any non-degenerate model"); §8 line 1097 (same phrase in the per-route
  paragraph).
- *Required:* **MUST NOT** write that embeddinggemma *uniquely* has the lowest ARI@100
  / smallest re-ranker-irreducible residual. Required phrasing: *"embeddinggemma and
  qwen3-0.6B are tied for the lowest re-ranker-irreducible residual (ARI@100 gap
  $0.004$, 95\% CI $[-0.174, 0.176]$, straddling zero)."*
- *KEEP DISTINCT:* embeddinggemma **retains** the separate distinction of the smallest
  *alignment-only floor* $L_\infty = 0.058$ (next non-deg qwen3 $0.073$, from the r3 ARI
  decomposition cp_fig22). That is still egemma's alone — **do not erase it**, and keep
  it textually separate from the ARI@100 ordering, which is now a tie. So §8's
  recommendation reinforcement becomes "smallest alignment-only floor ($L_\infty=0.058$),
  and tied with qwen3 for the lowest ARI@100" — not "lowest ARI@100 of any model."

**C3 — Separability is COLLINEAR with capability; frame it DESCRIPTIVELY (drop any
"independent / not a tautology / not a capability artifact" reading).**
- *Verified:* partial $r(\text{auc\_cross}, \text{CLIR@10}\mid\text{Recall@10}) = +0.2948$,
  two-sided $p = 0.5706$ (n.s.); zero-order $r = +0.9577$ (`robustness_table.csv` row 7;
  `summary.json` key `W2_separability_partial_r_controlling_recall10`).
- *Sites:* §7 line 1037 ("The lever is at the embedding level.") and any nearby clause
  that reads as a *mechanism-independent-of-capability* claim. The mechanism *bridge*
  (separability ⇒ floor ⇒ ARI) stays — only its causal-adjacent strength is softened.
- *Required softening:* *"Cross-language AUC and overall Recall@10 are strongly
  collinear across our seven models; once Recall@10 is partialled out the
  separability–CLIR association is no longer significant (partial $r=+0.29$, $p=0.57$),
  so we present separability as a descriptive correlate of the cross-lingual floor, not
  as an effect net of general retrieval capability."*
- *MUST NOT* claim the separability→CLIR mechanism is independent of general capability,
  "not a capability artifact," or "not a tautology." **Frame DESCRIPTIVELY only.** "The
  lever is at the embedding level" can stay as a *qualitative* reading, but the
  partial-r caveat must sit beside it so it is not read as a net-of-capability claim.

**C4 — The τ-rule survives only over a NARROW honest band; state the band, not
"robust."**
- *Verified:* admitted-set stable $\tau \in [0.385, 0.430]$; cheapest-admitted
  $= $ bge-m3 for $\tau \in [0.330, 0.435]$; egemma's max-CLIR@10 corner
  $\tau$-invariant $=$ TRUE; recommendation FLIPS to granite-278m at $\tau \le 0.3285$
  (granite enters and becomes cheapest); above $\tau \ge 0.45$ only embeddinggemma is
  admitted (`tau_sweep.csv`; `tau_sweep_summary.json` keys `tau_admitted_stable_range`,
  `tau_cheapest_bge_range`, `egemma_corner_tau_invariant`, `HONEST_NARROW_BAND`).
- *Sites:* §6.1 lines 623–633 (the $\tau=0.40$ admitted-set / cheapest-bge-m3
  sentences) and §6.1 line 643–644 (the cp_fig18 caption clause); the same rule when it
  recurs in §8 line 1073–1077.
- *Required:* state the narrow band and flag the low-end granite flip. Suggested: *"The
  admitted set is stable for $\tau \in [0.39, 0.43]$ and bge-m3 remains the cheapest
  admitted reader for $\tau \in [0.33, 0.44]$; below $\tau\approx0.33$ the
  cheaper-to-read granite-278m enters the admitted set and the cheapest-reader
  recommendation flips to granite, while above $\tau\approx0.45$ only embeddinggemma
  qualifies. Only embeddinggemma's status as the unique maximum-CLIR@10 corner is
  $\tau$-invariant."*
- *MUST NOT* say "the rule is robust." **The ONLY unconditional claim is the egemma
  max-CLIR corner** (which is $\tau$-invariant by construction — Pareto-optimality does
  not depend on $\tau$). The cheapest-reader recommendation is conditional on the band.

### The 4-float net cut (29 → 25 floats) — meet the 8-page body budget

All three round-4 critics named float overload (27 figures + 2 tables = **29 floats**,
17 in §6) as the *single dominant pre-submission risk*. The cut is exactly the
cohesion/dreamer plan, and the stitched panels the reporter verified are already on
disk:

| # | action | floats | lost content carried by | net |
|---|--------|--------|--------------------------|-----|
| 1 | **CUT cp_fig14 + ag_fig10 (the two radars)** | −2 | Tables 1–2 (per-axis numbers) + cp_fig17 (aggregation ribbon); absorb "where each wins" in one sentence (see CUT-NOTE) | 29→27 |
| 2 | **REPLACE cp_fig06 + cp_fig07 (mate) with the stitched 2-panel** `paper/figures/cp_fig06_07_mate.png` | −1 | identical story (foreign twins buried + how deep); one `\includegraphics`, one caption, one label | 27→26 |
| 3 | **REPLACE cp_fig09 + cp_fig10 (collapse) with the stitched 2-panel** `paper/figures/cp_fig09_10_collapse.png` | −1 | same mechanism (over-representation + which language buries gold); one float, one caption, one label | 26→25 |

- **Mechanics (merges):** the reporter verified both stitched PNGs render *both* source
  panels legibly: `cp_fig06_07_mate.png` (3062×754) and `cp_fig09_10_collapse.png`
  (2705×754). Preferred path: replace each pair's two `\includegraphics` /
  `\begin{figure}` blocks with **one** `\begin{figure}` containing one
  `\includegraphics{cp_fig06_07_mate.png}` (resp. `cp_fig09_10_collapse.png}`), one
  combined `\caption`, one `\label`, then collapse the two `\ref`s to that single label.
  (LaTeX `subfigure` of the original panels is an acceptable fallback; the per-source
  panels are left untouched on disk either way.) Mate pair lives at §6.1 lines 723–736;
  collapse pair at §6.2/§Analysis lines 968–984 (both `\ref`d in the single parenthetical
  near line 996/958).
- **CUT-NOTE (makes the radar cut argument-neutral):** append one clause to the
  leaderboard paragraph so no beat is lost — *"(embeddinggemma leads consistency and
  separability; the per-axis detail is in Tables 1–2 and
  Figure~\ref{fig:cp_ribbon}.)"* The radars carried only this one half-sentence; the
  tables + ribbon already cover it.
- **What STAYS, explicitly (do not touch):** cp_fig22 (ARI, ×3-ref, load-bearing),
  cp_fig23 (per-route, ×2-ref), cp_fig18 (cost frontier), cp_fig19 (RRC budget),
  cp_fig20 (degeneracy gate), cp_fig17 (aggregation ribbon — it *covers* the radars,
  so it must survive the radar cut), cp_fig11 (separability), the teaser, and the two
  leaderboard tables. **The cut comes entirely from legacy low-information panels.**
- **Stretch (only if still over budget after 25):** cp_fig17 is the *next* candidate
  (rank-range $[1,4]$ is stated inline) — but cutting it undermines the radar cut's
  cover story, so hold it as a last resort, not part of the base plan.

### The three cheap residuals (close them, ~1 clause each)

- **R1 — ρ_k residual-decomposition cite (near-mandatory, novelty).** The
  "decompose-and-normalize a re-ranking remainder" *shape* has a published cousin in
  long-tailed reranking. Add the bib entry verbatim:
  ```
  @article{residualrerank2026,
    title  = {Beyond Logit Adjustment: A Residual Decomposition Framework for
              Long-Tailed Reranking},
    author = {Wang and others},
    journal= {arXiv preprint arXiv:2604.01506},
    year   = {2026}
  }
  ```
  Cite once in the ARI paragraph (§4, near line 466) with the credit-and-distinguish
  half-clause: *"Normalizing a re-ranking remainder by a recoverable gap has a precedent
  in long-tailed reranking \citep{residualrerank2026}; we invert it (the un-rerankable
  share rather than the reranker's recoverable gain) and tie it to representation
  alignment cross-lingually, with an alignment-only floor that $\rho_k$ has no analogue
  for."* This is ARI's only residual "you reinvented X" surface; cheap, near-mandatory.
- **R2 — fig22 caption reconcile (correctness D-NEW + novelty + cohesion).** Line 703
  reads "The three sum to $1.0$ for all nine models." but the figure plots only the 7
  non-degenerate bars. Change to: *"The three sum to $1.0$ for every model (the identity
  closes for all nine; the figure shows the seven non-degenerate)."* **Change no value.**
- **R3 — §6.1 ARI read-off trim (cohesion seam #1).** §6.1 (lines 686–690) re-states the
  additive identity §4 (453–458) already owns. Open §6.1 at the *result*, not the
  re-definition: *"The ARI decomposition (Figure~\ref{fig:ari}) reports this split per
  model. For embeddinggemma the alignment-only floor is the smallest of any
  non-degenerate model..."* Net line saving on an over-budget paper. **No number
  changes.** (Note: the ARI@100 read-off in this same paragraph must also absorb C2 —
  the egemma/qwen3 tie.)

### Optional / lowest-priority polish (only if it costs no rewrite)
- Float-order: cp_fig22 (ARI, line ~697) floats before cp_fig19 (RRC budget, line ~710)
  it reads off; swap the two `\begin{figure}` blocks or let LaTeX float. Cosmetic.
- "home advantage" hyphenation still mixed (2 hyphenated, ~14 not) — harmonize to
  unhyphenated on the final pass; do not spend a content edit.
- Standardize $L_\infty$ to "0.058 (5.84\%)" once, "0.058" thereafter (cohesion minor).

**Everything not named above is FREEZE.** No new figures, no new metrics (channel-(b)
verdict from the dreamer is FREEZE — the metric family is closed), no new correlations,
no re-opening of C1 benchmarks / Related Work boundaries / the DEG gate / the MT null /
the two RBO ceilings / the two-tax framing. The six `needs_eval.md` items
(W3-alignment-causal-probe, W4-formula-injection, CLIRMRS-external-validation,
XRC-conformal-M2, CCI-hop-distance-law, equivalence-audit-spotcheck) are all DONE per the
critic contract; the paper stands without them.

---

## Thesis (industrial framing)

**UNCHANGED from round 4.** No edit to the thesis this round; reproduced so the writer
keeps the spine in view while making the C1–C4 softenings.

> **A chemistry-patent search team must deploy exactly one multilingual embedding
> model, and the number their dashboard shows — average Recall@10 — is the one number
> that hides the failure they will ship.** Average recall is inflated by
> *same-language* hits; the moment a German chemist's query must reach an English or
> Chinese patent (the normal case in a patent family), recall collapses, and no two
> language versions of the same question return the same documents. We make the collapse
> measurable on two content-controlled, patent-grounded benchmarks, quantify *what
> cross-linguality costs* (you read ~3.5× deeper to find a foreign twin; a top-100
> re-ranker recovers at most ~74% of them, and ~5.8% are structurally unrecoverable by
> any re-ranker), place the deployable models on a **cost-vs-capability frontier**
> (embeddinggemma is the Pareto-optimal capability corner; bge-m3 is the cheaper-to-read
> alternative), and show the durable fix is **representation alignment at indexing time,
> not a monolingual re-ranker at query time** — because foreign gold is *under-scored*,
> not merely mis-ordered, and the alignment-only floor $L_\infty \approx 0.058$ is the
> part of the gap that *no re-ranker can move*.

The C1–C4 softenings do **not** change the thesis: embeddinggemma is still the
capability-corner recommendation (its max-CLIR corner is the one $\tau$-invariant
claim, and its $L_\infty=0.058$ floor is still uniquely smallest); "align, don't
re-rank" is still the lever (now stated as a *descriptive* correlate, with the partial-r
caveat); the cost rule still holds (now over a stated band). The pillars are intact;
the prose around two of them is more honest.

---

## Contributions (numbered, each with a one-line novelty claim)

**FREEZE the contribution list as written in the round-4 draft / story.** The C1–C4
softenings live in the *body* (§6/§7/§8) and the appendix, not in the intro
contributions list — the contributions list is at the protected surface (intro), so it
must NOT gain ARI@100 numbers, the partial-r, or the τ-band. The one permitted touch is
that C2's bullet keeps ARI as "the re-ranker-irreducible share read off the RRC curve"
(framing, not a uniqueness claim about egemma). Reproduced for reference:

- **C1.** Two content-controlled, patent-grounded multilingual chemistry-retrieval
  benchmarks built only from human-translated patent text. *(NOVEL, well-defended,
  FREEZE.)*
- **C2.** A cross-lingual robustness-metric family reported co-equally with recall —
  XRC / RRC cost objects on a cost-vs-capability frontier, the RRC knee, the structural
  floor $L_\infty$, and the ARI decomposition (the re-ranker-irreducible share, credited
  to ρ_k and distinguished by the cross-lingual + alignment-floor instantiation — R1).
- **C3.** A mechanism finding: cross-lingual chemistry-retrieval failure is an
  embedding-level separability *correlate* (descriptive, collinear with capability — C3),
  so the lever is alignment, not re-ranking, with a measured floor.
- **C4.** A concrete, audited deployment decision (embeddinggemma = capability corner;
  bge-m3 = cheaper-to-read alternative over a stated τ-band — C4) with the per-route
  upside as bounded headroom.
- **C5.** (Supporting) A reproducible QAC generation + audit pipeline. *(FREEZE.)*

---

## Section map

Only the sections that change in round 5 carry beats; all others are **FREEZE**. Every
change below is a softening, a trim, or a cite — none adds a claim.

### Abstract — **FREEZE; one tiny C1 honesty tweak only**
- The separability sentence (line 74) currently says "$+0.96$, robust to dropping..."
  Keep it, but make "robust" mean sign-robust: it is fine as-is *if* the body and
  appendix carry the sign-stability/CI; the abstract may keep "$+0.96$, robust" without
  the CI (do not bloat the abstract). **No ARI@100 number, no τ-band, no partial-r, no
  per-route claim in the abstract** (protected surface).

### 1 Introduction — **FREEZE**
- Contributions list unchanged (see above). Protected surface: no ARI@100 tie, no
  partial-r, no τ-band.

### 2 Related Work — **one cite added (R1), otherwise FREEZE**
- Add the ρ_k residual-decomposition cite (R1) — primary home is the §4 ARI paragraph;
  optionally a one-clause back-reference here. All other Related Work boundaries CLOSED.

### 3 Benchmarks — **FREEZE.**

### 4 Metrics — **add the ρ_k cite (R1) to the ARI paragraph; otherwise FREEZE**
- The ARI definition paragraph (≈ lines 452–467) gets the one credit-and-distinguish
  half-clause + `\citep{residualrerank2026}` (R1). No change to the identity, the
  scalar, or any number.

### 5 Experimental Setup — **FREEZE** (extra_* scripts already listed)
- The round-4 reporter notes `extra_robustness_appendix.py`, `extra_tau_sweep.py`, and
  `stitch_merged_panels.py` were authored this round; the writer MAY add them to the
  reproducibility script list, but this is optional housekeeping, not required.

### 6 Results — **C2 + C4 softenings; R2 + R3 trims; the two merges; the radar cut**
- *§6.1 cost frontier (C4 + radar-cut cover sentence):*
  - Apply **C4** to lines 623–633 and the cp_fig18 caption clause (643–644): replace the
    single-τ snapshot with the stated band + the granite low-end flip + the
    τ-invariant egemma corner (wording above). The "$\tau=0.40$" snapshot stays as the
    reference point inside the band; the band and the flip are the new honest framing.
  - cp_fig18 (cost frontier) STAYS.
- *§6.1 RRC budget + ARI read-off (R3 trim + C2 tie + R2 caption):*
  - **R3:** open the §6.1 ARI paragraph (686–690) at the result, not the re-definition.
  - **C2:** the ARI@100 read-off (692–694, 705) must say egemma and qwen3 are *tied* for
    the lowest re-ranker-irreducible residual (gap 0.004, CI straddles 0), while egemma
    keeps the *smallest $L_\infty$ floor* (0.058) as a separate, still-true distinction.
  - **R2:** fig22 caption (line 703) "all nine models" → "every model (the identity
    closes for all nine; the figure shows the seven non-degenerate)."
  - **MERGE 1:** replace the cp_fig06 + cp_fig07 mate pair (lines 723–736) with the
    single stitched float `cp_fig06_07_mate.png`; collapse the two `\ref`s.
  - cp_fig19 (RRC budget) STAYS; optional float-order swap with cp_fig22.
- *§6.2 alias-graph:* **FREEZE** (fig21 caption already fixed in round 4). The collapse
  pair merge (MERGE 2) sits here/§Analysis — replace cp_fig09 + cp_fig10 (lines 968–984)
  with the stitched `cp_fig09_10_collapse.png`; collapse the two `\ref`s.
- *§6.3 leaderboards:* **FREEZE**, plus the CUT-NOTE clause (the radar "where each wins"
  sentence) appended to the leaderboard paragraph so the radar cut loses no beat.
- *Radar cut:* delete the cp_fig14 (line 902) and ag_fig10 (line 910) `\begin{figure}`
  blocks and their `\ref`s; the CUT-NOTE carries their content.

### 7 Analysis — **C1 + C3 softenings; otherwise FREEZE**
- *Separability crux (lines 1024–1038):*
  - **C1:** where "$+0.96$ ... robust" is introduced (1025–1028), add the sign-stability
    reading + the wide-CI honesty (lead with $P(r>0)=0.9997$; CI $[0.73,1.00]$ wide at
    $n=7$); the cp_fig11 caption (1045–1046) may carry the CI.
  - **C3:** add the collinearity/partial-r caveat (partial $r=+0.29$, $p=0.57$ n.s.;
    "descriptive correlate, not net of general capability") beside "The lever is at the
    embedding level" (1037). Do not delete the mechanism bridge — soften its strength.
- The two hedged fragile correlations and the rest of §7 stay **FROZEN**.

### 8 Deployment Recommendation — **C2 + C4 softenings; otherwise FREEZE**
- *(1) Deploy embeddinggemma — capability corner (spine UNCHANGED):* lines 1070–1084.
  - **C2:** line 1078 "lowest $\mathrm{ARI@}100$ (0.229) of any non-degenerate model" →
    "smallest alignment-only floor ($L_\infty=0.058$), and tied with qwen3-0.6B for the
    lowest ARI@100 (gap 0.004, CI straddles 0)." Keep the $L_\infty$ distinction;
    demote ARI@100 to a tie.
  - **C4:** the τ-rule restatement at lines 1073–1077 picks up the band wording (or
    simply back-references the §6.1 band). The "unique maximum-CLIR corner" (1071) is
    the τ-invariant claim — keep it firm; that is the one unconditional take-away.
- *(per-route paragraph, lines 1086–1110):* **C2** at line 1097 — same fix as line 1078
  (tie on ARI@100, keep the $L_\infty$ distinction). The four-point honesty contract on
  the router stays verbatim; **FREEZE** the rest of the paragraph.

### 9 Limitations — **point at the new appendix robustness table; otherwise FREEZE**
- Add **one sentence** leading the robustness/limitations discussion to the new appendix
  table (W1 — see below): the load-bearing small-$n$ scalars (separability $r$ with
  sign-stability + CI, the three frontier XRC50 CIs, the ARI@100 egemma-vs-qwen3 tie CI,
  the partial-r) are consolidated there. This *positions the honest CIs as a credibility
  anchor* rather than hiding them. The forthcoming-W3-probe paragraph (with $L_\infty$ /
  ARI as the before/after target) stays. No new claim.

### 10 Conclusion — **FREEZE; C1 wording only**
- Conclusion's "$+0.96$, robust" (line 1278) keeps the body's sign-robust meaning; no
  ARI@100 number, no τ-band, no partial-r in the conclusion (protected surface). Spine
  unchanged.

### Appendix — **add the robustness table (W1), outside the 8-page body budget**
- One appendix table consolidating the round-4 verified resamples (the natural home for
  C1/C2/C3 and the XRC50 CIs), each row = scalar / point estimate / resample interval or
  sign-stability vote / $n$:
  - separability $r=0.96$ (95% CI $[0.73, 1.00]$, $P(r>0)=0.9997$, $n=7$);
  - XRC50 of the three frontier members — egemma $3.5$ $[0.909, 12.0]$, bge-m3 $2.0$
    $[0.529, 7.0]$, granite-278m $1.25$ $[0.284, 12.25]$ (finite but WIDE,
    median-of-discrete-depth bootstrap; censored-draw frac $=0.0$);
  - ARI@100 gap (qwen3 − egemma) $=0.004$, 95% CI $[-0.174, 0.176]$ (straddles 0),
    $P(\text{egemma}<\text{qwen3})=0.519$;
  - partial $r(\text{auc\_cross},\text{CLIR@10}\mid\text{Recall@10})=+0.29$, $p=0.57$ (n.s.).
- Source: `extra_robustness_appendix/{robustness_table.csv, summary.json}`. The table is
  outside the body page budget; referenced once from §9. It converts the prose-level
  small-$n$ hedging into a structural "every load-bearing scalar's interval is reported"
  credibility move — **this is the round's one new float, and it is in the appendix.**

---

## What the writer must prioritize (in order)

1. **The four honesty corrections C1–C4** (highest priority — these are the
   anti-fabrication flags; the two pillars currently *overclaim*). Get the exact wording
   from the C1–C4 blocks above. Two are softenings of currently-too-strong sentences
   (C2 egemma-uniqueness, C3 separability-independence); two add an honest band/CI
   (C1 wide CI + sign-stability, C4 narrow τ-band + granite flip).
2. **The 4-float net cut** (cut 2 radars, swap in the 2 stitched panels + CUT-NOTE) — the
   single biggest pre-submission risk; lands the paper at 25 floats inside the 8-page body.
3. **The three cheap residuals** (R1 ρ_k cite, R2 fig22 caption, R3 §6.1 trim) — ~1
   clause each, close every named critic residual.
4. **The appendix robustness table (W1)** — the credibility anchor that makes the C1–C4
   honest CIs do the work; outside the body budget.
5. **Optional polish** (float-order swap, hyphenation, $L_\infty$ notation) — only if it
   costs no rewrite.

---

## Open narrative risks (for critics to watch)

1. **C2 must not regress to "egemma uniquely lowest ARI@100."** Every site (1078, 1097,
   692–694, 705) must read as a *tie* with qwen3 on ARI@100, while egemma *keeps* the
   distinct smallest-$L_\infty$ (0.058) win. Erasing the $L_\infty$ distinction is as
   wrong as keeping the uniqueness claim — both must be exactly right.
2. **C3 must not migrate into a "tautology-busted / capability-independent" claim.** The
   partial-r is n.s.; separability is a *descriptive* correlate. The mechanism bridge
   stays, its causal-adjacent strength is softened.
3. **C4: only the egemma max-CLIR corner is unconditional.** The cheapest-reader (bge-m3)
   and admitted-set claims are conditional on the stated band; the granite low-end flip
   must be stated, not hidden.
4. **C1: "robust" now means sign-robust.** The wide CI $[0.73,1.00]$ must be presented
   honestly (not as a tight estimate); lead with $P(r>0)=0.9997$.
5. **The cut must lose no argument.** The CUT-NOTE clause must land in the leaderboard
   paragraph; the two merged floats must each be referenced once with one caption; no
   orphan, no dangling `\ref` to a deleted radar.
6. **Protected surfaces stay clean** (abstract / intro / conclusion): no ARI@100 number,
   no τ-band, no partial-r, no per-route routing claim. Held since round 2 — do not
   regress.
7. **No new claims.** Round 5 adds zero new assertions. Anything that reads as a *new*
   result (not a softening, trim, cite, or appendix consolidation of verified resamples)
   is out of scope and a regression risk.
