# Cohesion review (round 5)

Reviewer #3 (Cohesion). Scope: flow, framing, internal consistency. I did not
re-verify numbers against `reports/` (Reviewer #2) nor judge novelty
(Reviewer #1). I read the current `paper/main.tex` (round-5 revision, 1361
lines), the two stitched 2-panel figures it now references
(`cp_fig06_07_mate.png` 3062×754, `cp_fig09_10_collapse.png` 2705×754 — both
present and both panels legible on disk), `paper/loop/round_5/story.md`,
`writer_notes.md`, `figures_manifest.md`, and my own
`paper/loop/round_4/critic_cohesion.md` to confirm the round-4 must-fix (float
overload) landed and that the round-5 honesty softenings (C1–C4) and the figure
cut did not open new seams. I confirmed float counts, label/ref counts,
deleted-radar dangling-ref absence, and protected-surface honesty on disk as
cohesion questions only.

## Overall: does it read as one story? (1 paragraph)

Yes — and this is the cleanest the paper has read in five rounds. The round-4
liability (float overload) is **resolved**: the body now carries **23 figures +
2 in-body tables = 25 in-body floats**, with the round's one new float
(`tab:robust`) parked in the appendix outside the 8-page budget — exactly the
cohesion/dreamer plan's 29→25 target. The two radars are gone with **zero
dangling references** (`fig:cp_radar`, `fig:ag_radar`, `fig:cp_rank`,
`fig:cp_distractor` all return 0 hits; `cp_fig14`/`ag_fig10` survive only inside
an explanatory `% CUT-NOTE` comment), and the two stitched panels each carry one
`\includegraphics`, one combined caption, one label, with their `\ref`s
collapsed (`fig:cp_mate` ×2, `fig:cp_collapse` ×1 — both legitimately
referenced). The four honesty softenings read as a **confident story, not a
wishy-washy retreat**: each one keeps the load-bearing claim firm and demotes
only the over-reach. C1 redefines "robust" once, in-body, as *sign-robust under
resampling* ($P(r>0)=0.9997$) and presents the wide $n{=}7$ CI honestly while
the abstract/conclusion keep "$+0.96$, robust" anchored to that meaning — the
mechanism still reads as the section's load-bearing finding (line 1016, "this is
the section's load-bearing mechanism"). C2 holds a genuinely tricky double
framing perfectly at all five sites (the ARI@100 tie with qwen3 *and*
embeddinggemma's still-distinct smallest $L_\infty=0.058$ floor are kept
textually separate everywhere — 705–706, 719–721, 1033–1036, 1088–1089,
1109–1110), so the recommendation is reinforced, not weakened. C3 keeps "the
lever is at the embedding level" (line 1036) and adds the partial-r caveat
beside it as a *descriptive* frame — the mechanism bridge survives, only its
causal-adjacent strength is softened, with no "tautology-busted/independent"
reading anywhere. C4 states the narrow τ-band and the granite low-end flip
firmly and reserves "$\tau$-invariant" for the one unconditional claim
(embeddinggemma's max-CLIR corner). The protected surfaces held: abstract, intro
contributions, and conclusion are **clean** of ARI@100, 0.229, τ/τ-band,
partial-r, and any per-route/router claim (verified by scan). The appendix
robustness table is referenced **five times** from the body (§6.1 707, §7 1023 &
1042 & cp_fig11 caption 1052, §9 1212), positioning the honest CIs as a
credibility anchor rather than burying them. The cost-object spine and the ARI
define→quantify→bridge→deploy→future-target arc that were the round-3/4
structural wins are intact and undisturbed. The paper is now a tight, honest,
on-budget industry-track submission.

## Did the round-4 must-fix close + did the trim open new seams?

1. **Float overload (round-4 joint #2, the dominant pre-submission risk) —
   CLOSED.** 29 floats → 26 floats → **25 in-body** (23 fig + 2 tab in body; the
   appendix `tab:robust` is the 26th, outside the body budget). `begin{figure}`
   = 23, `begin{table}` = 3, `includegraphics` = 23 — all balanced. The cut came
   entirely from legacy low-information panels (two radars cut, two pairs
   merged); every load-bearing float (cp_fig22 ×3, cp_fig23 ×2, cp_fig18,
   cp_fig19, cp_fig11, cp_fig17, teaser) stayed. Resolved exactly as planned.

2. **§6.1 ARI re-definition redundancy (round-4 joint #1) — CLOSED via R3.** The
   §6.1 ARI paragraph (701–709) now opens at the *result* ("The ARI
   decomposition (Figure~\ref{fig:ari}) reports this split per model. For
   `embeddinggemma` the alignment-only floor is the smallest…"), not at a
   re-statement of the §4 identity. The only redundancy in the ARI thread is
   gone; the thread now reads as continuation, not restatement.

3. **No new seams from the trim.** The CUT-NOTE clause lands naturally inside the
   leaderboard paragraph (lines 832–834: "embeddinggemma leads consistency and
   separability, not raw recall alone; the per-axis detail is in
   Tables~1–2 and Figure~\ref{fig:cp_ribbon}"), so the radars' lone beat ("where
   each model wins") is carried with no orphan. Both stitched-panel captions are
   coherent and self-contained (Left/Right structure matches the rendered panels:
   mate-hit + MRR | first-foreign depth for cp_fig06_07; over-representation |
   burying-language for cp_fig09_10).

## Broken glue joints (ranked, each: location -> problem -> concrete fix)

**1. MINOR (cosmetic, inherited; round-4 joint #3 still open) — cp_fig22 (ARI)
floats *before* cp_fig19 (RRC budget) it reads off.** cp_fig22 is included at
line 713, cp_fig19 at line 728; the prose beat order is correct (RRC budget
686–698 → ARI read-off 701–709), so the source-order figure blocks are reversed
relative to the prose.
- *Problem.* Pure float-placement nit; LaTeX will likely reorder both to the same
  page region anyway. The writer explicitly deferred this (writer_notes "cohesion
  #3 … DEFERRED — cosmetic"), consistent with story.md marking it
  optional/lowest-priority. Not a logical break.
- *Fix.* Swap the two `\begin{figure}` blocks (move cp_fig19 at 728 above
  cp_fig22 at 713) on the final typesetting pass, or let LaTeX float. Do not
  spend a content edit. Lowest priority; carried since round 3.

**2. TRIVIAL (caption rounding) — `cp_fig09_10_collapse` caption says "up to
$49\times$" while the rendered left panel labels the Chinese bar $48.7\times$.**
Lines 973–974 ("up to $49\times$ the corpus base rate") vs the panel's
"$48.7\times$"; the Analysis prose (line 956) also says "$49\times$."
- *Problem.* The body rounds $48.7$ to $49$; the figure shows the unrounded
  value. This is the same number, and rounding $48.7\to49$ is defensible, but a
  sharp reader comparing caption to panel sees a $0.3$ mismatch. Cohesion-trivial,
  not a correctness flag (that is Reviewer #2's call on whether $49$ is the
  on-disk headline).
- *Fix.* Either write "up to $\sim$$49\times$" / "$48.7\times$" in caption + line
  956 to match the panel, or leave it — defensible as a round. If touched at all,
  harmonize the two prose sites (956, 973) and the panel to one rendering. Do not
  spend a real edit unless doing the final polish pass.

**3. TRIVIAL (number-flavour, optional) — $L_\infty$ still appears in two
notations, "$0.058$" and "$5.84\%$".** "$0.058$" at 692/703/719/734/1036/1088/
1289 and "$5.84\%$" at 1030/1191. Each is locally clear and the two are the same
number; flagged in round 4 as a tightness-only nit.
- *Fix (optional, carried).* Standardize to "$0.058$ ($5.84\%$)" at first body
  use (line 692) and "$0.058$" thereafter, per story.md's optional-polish item.
  No content change, lowest priority.

There are **no major or moderate glue joints this round.** The four honesty
softenings, the cut, the merges, and the appendix table are all clean.

## Unmet promises / orphan results

- **All five contributions still deliver and are referenced.** C1 → §3 +
  abstract/intro; C2 → §4 (incl. the ARI def + the R1 ρ_k credit-and-distinguish
  clause, 465–469) + §6 + Deployment; C3 → §7 (separability + RRC floor + ARI
  bridge, now with the C3 descriptive-correlate caveat, 1037–1042); C4 → §8
  (frontier + τ-band + per-route upside); C5 → §3 + Appendix. The contributions
  list itself was correctly **not** touched by C1–C4 (protected intro surface).
- **Zero orphan figures.** All 23 included figures are referenced at least once
  (verified by label-vs-ref count): teaser ×4, ari ×3, cost_frontier/per_route/
  rrc_budget/cp_mate/cp_ribbon ×2, all others ×1. The two merged floats
  (`fig:cp_mate`, `fig:cp_collapse`) are each referenced and interpreted in text.
- **The appendix robustness table is referenced from the body — promise kept.**
  `tab:robust`/`app:robust` is `\ref`d from §6.1 (707, the ARI gap CI), §7 (1023
  separability sign-stability, 1042 partial-r, and the cp_fig11 caption 1052),
  and §9 (1212, the consolidating sentence). It is not a dropped-in appendix; it
  is woven into every site that carries a softened scalar. This is the round's
  cleanest new joint.
- **The C1/C2/C3/C4 softenings created no orphan claim.** Each softening sits
  beside the firm claim it qualifies; none introduces a number that is then never
  used. The ρ_k cite (R1) is cited exactly once with a distinguishing clause; the
  bib entry exists (`custom.bib:390`).
- **The retired/merged source panels stay retired.** cp_fig06/07, cp_fig09/10,
  cp_fig14, ag_fig10 remain on disk but unreferenced (superseded/cut) — no
  regression, no dangling include.

## Terminology & notation inconsistencies

- **ARI@100 tie is framed identically at all five sites** (705–706, 719–721,
  1033–1035, 1089, 1109–1110): always "$0.229$ vs.\ $0.233$, gap $0.004$, CI
  straddles zero, tied." The C2 risk (regressing to "egemma uniquely lowest
  ARI@100") did **not** materialize anywhere.
- **$L_\infty=0.058$ kept as a *separate, distinct* embeddinggemma win at all
  sites** (703 "smallest of any non-degenerate model, next qwen3 0.073," 719,
  1035–1036, 1088, 1108–1109). The C2 over-correction risk (erasing the
  $L_\infty$ distinction) also did **not** materialize. Both halves are exactly
  right.
- **"robust" is now disambiguated.** The body redefines it once
  (1023, "Robust here means sign-robust under resampling, not tightly
  estimated"); every other body use of "robust" for the separability link
  (1009, 1019, 1050) is consistent with that meaning; abstract (74) and
  conclusion (1296) keep "$+0.96$, robust" un-expanded, correctly relying on the
  body anchor. The remaining "robust" uses (1119, 1133, 1217, 1236) refer to
  *other* robust signals (per-route CLIR@10 axis, XRC50 median) and are not in
  conflict. Clean.
- **τ / τ-band naming is uniform** (628 τ=0.40 reference point; 632–638 the band
  [0.39,0.43] / [0.33,0.44] / flip <0.33 / only-egemma >0.45 / "τ-invariant"
  corner; cp_fig18 caption 656–658 same; §8 1084–1087 same). No drift.
- **Both RBO ceilings (0.39 / 0.19), both n.s. correlations (two-tax −0.59;
  cheapest-reader trap +0.29) carry n.s. + n=7** and stay out of
  abstract/intro/conclusion. Held.
- **"home advantage" hyphenation almost fully harmonized** — now **1 hyphenated**
  ("home-advantage", down from 2) vs 15 unhyphenated. One straggler remains; the
  long-running cosmetic carryover is nearly closed.

## Abstract/Conclusion alignment issues

- **Abstract is clean of all softened-stat leaks** (verified scan, lines 47–80):
  no ARI@100, no 0.229, no τ/τ-band, no partial-r, no per-route/router. The
  separability sentence (74) keeps "$+0.96$, robust to dropping the two collapsed
  encoders" — appropriate at the protected surface, with the sign-robust meaning
  and wide CI carried in the body+appendix. The $L_\infty$ floor framing (72,
  "a measured share recoverable only by alignment") is the floor, not an ARI
  number. No bloat from the round's softenings.
- **Conclusion is clean** (verified, 1281–1300): no 0.229, no τ-band, no
  partial-r, no per-route. It keeps "$+0.96$, robust" (1296) consistent with the
  body's sign-robust meaning, and restates the spine (collapse, cost, the
  $L_\infty=0.058$ floor "the only part of the gap a re-ranker cannot move,"
  embeddinggemma the Pareto corner, bge-m3 cheaper-to-read, align-not-re-rank,
  budget rule). Abstract ↔ conclusion still make the same claims with the same
  emphasis.
- **No emphasis whiplash from the softenings.** The two pillars that were
  softened (separability mechanism, τ-rule) still read as firm load-bearing
  findings in the abstract/conclusion; only the body carries the honest hedge.
  This is the correct split and the round nailed it.

## What's already cohesive (leave alone)

- **The float cut is exactly right — do not cut further.** 25 in-body floats
  lands the budget; cp_fig17 (the radar cover story) correctly survives, so the
  CUT-NOTE clause is honest. The "stretch" cut of cp_fig17 named in story.md is
  **not** needed and would undermine the radar cut — leave it.
- **The C2 double framing (ARI@100 tie + distinct $L_\infty$ win) is the round's
  hardest cohesion task and it is executed flawlessly at all five sites.** Do not
  touch any of them.
- **The appendix robustness table → body reference chain is a model credibility
  move.** Five body sites point to one consolidated table of honest intervals.
  Leave the chain intact.
- **The ARI define→quantify→bridge→deploy→future-target arc** (§4 → §6.1 → §7 →
  §8 → §9) is undisturbed by the softenings; the C2 tie threaded through it
  cleanly. Round-3/4 structural win preserved.
- **The per-route §8 paragraph still honors the four-point honesty contract**
  (single-model default, router = headroom, XRC axis indicative, es never
  imputed, thin-n in both §8 and Limitations), and the C2 tie folded into it
  (1109–1110) without breaking the contract.
- **The cost-object spine, the DEG gate anchoring every "non-degenerate," the two
  RBO ceilings, the MT null, descriptive availability** — all the honesty bedrock
  from rounds 2–4 is intact; nothing regressed.

---

### Bottom line for the conductor

**Overload resolved.** 29 → **25 in-body floats** (23 fig + 2 tab; the appendix
`tab:robust` is the 26th float, outside the 8-page body) — the round-4 dominant
risk is closed, the two radars cut with zero dangling refs, the two stitched
2-panel figures (`cp_fig06_07_mate`, `cp_fig09_10_collapse`) each carry one
coherent caption matching their rendered panels, and the CUT-NOTE clause carries
the radars' only beat. **No new seams from the trim.** The four honesty
softenings (C1 sign-robust + wide CI; C2 ARI@100 tie with the $L_\infty=0.058$
win kept distinct; C3 descriptive-correlate partial-r caveat; C4 narrow τ-band +
granite flip) all read as a **confident story** — each demotes only the
over-reach and keeps the load-bearing claim firm; none reads wishy-washy. The
appendix robustness table is referenced from the body **five times** as a
credibility anchor. Protected surfaces (abstract/intro/conclusion) are clean of
ARI@100/τ-band/partial-r/per-route. **The only residuals are three trivial,
optional polish nits**: (1) the inherited cosmetic float-order of cp_fig22 before
cp_fig19 (deferred, let LaTeX float); (2) caption "$49\times$" vs panel
"$48.7\times$" rounding; (3) the $L_\infty$ "$0.058$"/"$5.84\%$" dual notation.
None is a logical break. The paper is on-budget, internally consistent, and
tells one tight story end-to-end. **Recommendation: ship after a one-line
typesetting pass for the float-order nit; no content edit required.**
