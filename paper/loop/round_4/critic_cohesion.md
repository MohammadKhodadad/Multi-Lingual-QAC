# Cohesion review (round 4)

Reviewer #3 (Cohesion). Scope: flow, framing, internal consistency. I did not
re-verify numbers against `reports/` (Reviewer #2) nor judge novelty
(Reviewer #1). I read the current `paper/main.tex` (round-4 revision, 1304
lines), the two new figures it references (cp_fig22 ARI decomposition, cp_fig23
per-route frontier — both present on disk, 116 KB / 186 KB, rendered),
`paper/loop/round_4/story.md`, and my own `paper/loop/round_3/critic_cohesion.md`
to confirm the two round-3 must-fixes landed and to judge whether the round's
two additions (ARI threaded across §4/§6.1/§7/§8, per-route in §8) read as one
thread or as bolt-ons. I confirmed figure presence, reference counts, and
population labels on disk as cohesion questions only.

## Overall: does it read as one story? (1 paragraph)

Yes, and the two round-3 seams are both closed. The ARI decomposition is woven in
as a genuine *continuation* of the RRC thread, not a redundant restatement: §4
defines it as "the RRC curve admits a natural exhaustive reading of the same
shortfall" (line 453) and gives the identity + scalar; §6.1 reports the numbers
(embeddinggemma floor smallest, ARI@100 = 0.229 lowest, qwen3 next at 0.233,
lines 686–695); §7 uses it as the bridge from the separability mechanism to
"align, don't re-rank" (lines 1034–1037); §8 reinforces the recommendation in one
clause (lines 1078, 1097); and §9 hands it to the forthcoming W3 probe as the
before/after target (lines 1252–1254). That is a define → quantify →
mechanism-bridge → deploy-reinforce → future-target arc — exactly one thread, and
it strengthens C2/C3/C4 simultaneously without spawning a parallel claim. The
per-route §8 paragraph (1086–1110) holds the four-point honesty contract
verbatim: it opens "This nuances, but does *not* overturn, the single-model
recommendation," keeps embeddinggemma as the default (global corner, lowest
floor/ARI, wins 3/5 routes incl. the two hardest), frames the router as "headroom,
not a recommendation," labels the per-route XRC axis "indicative," and never
imputes the es XRC — no whiplash with the single-model spine, and the thin-n
caveat is mirrored in Limitations (1211–1219). The honesty discipline held
perfectly at the three protected surfaces: abstract, intro-prose, and conclusion
carry **no** ARI number, **no** per-route routing claim, and **no** non-significant
correlation (the intro's only per-route mention is inside the C4 *contribution
bullet*, lines 159–160, explicitly "genuine upside headroom rather than a
delivered win" — correct). Both must-fixes are confirmed: "eight non-degenerate"
is gone (relabeled to "the eight models with a defined cross-lingual recall / RRC
curve," lines 585, 714) and the fig21 caption now reads "confusion rate,
alias-graph benchmark" with no "sibling-" (line 759); the F3 cascade cite is in
place (line 447). The **one real cohesion liability this round is not a logical
break but a budget one**: the paper now carries **27 figures + 2 tables = 29
floats**, 17 of them in §6 alone, which is over the industry-track page budget —
and the §6.1 ARI read-off re-explains an identity §4 already owns. The spine is
done; the recommendation below is to **freeze the analysis and cut ~3–4 floats**,
not to add or reopen anything.

## Did the 2 round-3 seams close? (one line each)

1. **"eight non-degenerate models" (lines 557, 657 in round 3) — CLOSED.**
   Both sites now read "the eight models with a defined cross-lingual recall (all
   but the degenerate `gte-base`)" (line 585) and "the eight models with a defined
   RRC curve (all but `gte-base`, whose candidate pool is empty)" (line 714). The
   gate-bound word "non-degenerate" is now used **only** for the precise 7-set
   (lines 489, 631, 746, 759), so the DEG gate no longer self-contradicts. Clean.

2. **fig21 "sibling-confusion rate" caption — CLOSED.** The cp_fig21 caption now
   reads "the confusability tax (confusion rate, alias-graph benchmark)" (lines
   757–759); no "sibling-" anywhere in a caption. The separate, correctly-labeled
   sibling-vs-parent severity split in §6.2 prose (lines 786–788) is untouched and
   still uses "sibling" correctly. Clean.

Round-3 *secondary* carryover: **home-advantage hyphenation still mixed** (2
hyphenated at lines 141/954, ~14 unhyphenated). Cosmetic, unchanged across three
rounds now; harmonize to unhyphenated on the final polish pass — do not spend a
content edit on it.

## Broken glue joints (ranked, each: location -> problem -> concrete fix)

**1. MINOR (the round's only new seam) — the §6.1 ARI read-off (lines 686–689)
re-explains the additive identity that §4 (lines 453–458) already defined, in
near-identical words.**
- *Problem.* §4 already states "a model's cross-lingual shortfall splits into a
  cheaply recoverable part (RRC@K ...), a deep-pool part ..., and the
  alignment-only floor ...; the three sum to one" (453–458). The §6.1 paragraph
  then opens by re-stating the *same* identity — "The RRC curve decomposes each
  model's cross-lingual shortfall exhaustively into what a re-ranker can recover,
  what a deeper pool can recover, and the alignment-only floor; the three sum to
  one for every model" (687–689) — before it gets to its actual job, which is the
  *numbers* (embeddinggemma floor smallest, ARI@100 = 0.229). On a paper that is
  over its float budget, spending two sentences re-defining a definition that lives
  three pages earlier is the one place the ARI thread reads as restatement rather
  than continuation. This is mild and local; the thread itself is sound.
- *Fix.* Open the §6.1 paragraph at the result, not the re-definition. Replace
  lines 687–690 with one connective clause: *"The ARI decomposition
  (Figure~\ref{fig:ari}) reports this split per model. For `embeddinggemma` the
  alignment-only floor is the smallest of any non-degenerate model ..."* This drops
  ~1.5 lines, removes the only redundancy in the thread, and lets the read-off do
  what §4 set it up to do. ~2 words of new text, ~2 sentences removed.

**2. MINOR (figure-count / page budget — the dominant cohesion concern this round,
but a CUT recommendation, not a logical break).** §6 carries **15 figures + 2
tables = 17 floats**; the whole paper carries **27 figures + 2 tables = 29 floats**
for an 8-page industry-track body.
- *Problem.* This is no longer "heavy" (my round-3 word for 11 §6 figures) — it is
  over budget. Every figure is referenced (0 orphans, verified), but reference is
  not the test for an 8-page paper; *density* is. At ~29 floats the figures
  physically crowd out the prose, and the industry track will read this as a paper
  that could not bear to cut. The round-4 story.md agrees ("STOP generating new
  analyses ... cp_fig22 + cp_fig23 are the last two the paper should absorb"). The
  spine is now complete, so this is the round to FREEZE the analysis and CUT.
- *Fix (ranked cut candidates, ~3–4 floats recoverable, no argument lost):*
  - **(a) Merge or cut the two radar figures (cp_fig14 + ag_fig10).** They share a
    single half-sentence (line 821, "show *where* each top model wins") and restate
    the per-axis story the two leaderboard tables (Tab. 1/2) and the aggregation
    ribbon (cp_fig17) already carry. The radars are the lowest-information floats in
    the paper. **Cut both** (move to appendix if wanted); the leaderboards +
    cp_fig17 fully cover "where each model wins." Saves 2 floats.
  - **(b) Merge the mate-retrieval pair (cp_fig06 + cp_fig07) into one two-panel
    figure.** They are introduced in two adjacent half-sentences (lines 668, 670)
    and tell one story (foreign twins are buried + how deep). One two-panel figure
    reads identically. Saves 1 float.
  - **(c) Merge the language-collapse pair (cp_fig09 + cp_fig10) into one figure.**
    Both cited in a single parenthetical (line 958, "(Figures ...09, ...10)"); they
    are the same mechanism (over-representation + which language buries the gold).
    Two panels, one float. Saves 1 float.
  - Doing (a) alone takes the paper from 29 → 27 floats; (a)+(b)+(c) → 25.
    cp_fig22 and cp_fig23 (the round's new, load-bearing additions) **stay** — they
    are the highest-reference new floats (×3 and ×2). The cut comes entirely from
    legacy low-information panels.

**3. MINOR (float order, inherited) — cp_fig22 (ARI) floats *before* cp_fig19 (RRC
budget) it conceptually reads off.** cp_fig22 is included at line 697 and cp_fig19
at line 710, but cp_fig19 (the RRC budget curve) is the object cp_fig22
decomposes, and its prose beat (RRC budget, lines 666–684) *precedes* the ARI
read-off (686–695). So in source order the ARI figure floats ahead of the RRC
figure it depends on.
- *Fix.* Swap the two `\begin{figure}` blocks so cp_fig19 precedes cp_fig22, or
  let LaTeX float them — the prose order (RRC budget → ARI) is correct, so this is
  cosmetic. Lowest priority; do not let it cost a rewrite. Same class of nit I
  flagged for cp_fig16/cp_fig19 in earlier rounds.

## Unmet promises / orphan results

- **All five contributions still deliver and are referenced.** C1 → §3 +
  abstract/intro; C2 → §4 (now including the ARI definition, lines 452–467) + §6 +
  Deployment; C3 → §7 (separability + RRC floor + ARI bridge, 1034–1037); C4 → §8
  (frontier + per-route upside, 1086–1110); C5 → §3 + Appendix. The C2 ARI clause
  (line 141) and the C4 per-route clause (lines 159–160) were added to the intro
  contributions list as story.md specified, each in one clause, body prose
  unre-opened.
- **Zero orphan figures.** All 27 included figures are referenced at least once
  (verified by label-vs-ref count): teaser ×4, ari ×3, cost_frontier/per_route/
  rrc_budget ×2, all others ×1. cp_fig22 is referenced at §4 (467), §6.1 (690), §9
  (1253); cp_fig23 at §8 (1093) and §9 (1212). Both new figures are interpreted
  in text, not dropped in. (Note: story.md risk #10 asked each to be referenced
  "exactly once"; cp_fig22's three references are legitimate *threaded* references
  — definition, read-off, future-target — not duplication, and reading them as one
  thread is the round's success, so I read "exactly once" as satisfied in spirit.)
- **No result is dropped in without interpretation.** The ARI def, the §6.1
  read-off, and the per-route paragraph each state their number, cite their figure,
  and draw the deployment read-off.
- **The retired panels stay retired.** cp_fig04, cp_fig08, cp_fig13, cp_fig15,
  cp_fig16 remain absent/unreferenced as superseded — no regression.

## Terminology & notation inconsistencies

- **"non-degenerate" = the precise 7-set everywhere** (round-3 must-fix resolved):
  used only at lines 489, 631, 746, 759, all gate-correct; the two former "eight"
  slips are relabeled (585, 714). No contradiction remains.
- **ARI naming is clean and uniform.** "ARI," "alignment-recoverability index,"
  "ARI@100," "alignment-only floor," "$L_\infty$" are used identically across §4,
  §6.1, §7, §8, §9. The number 0.229 appears identically at 692, 705, 1036, 1078,
  1097 (and never in abstract/intro/conclusion). $L_\infty=0.058$ vs the equivalent
  $5.84\%$ phrasing both appear (e.g. 678 "0.058," 1033/1178 "5.84\%") — these are
  the same number in two notations and are each locally clear, but for tightness
  consider standardizing to "0.058 (5.84%)" once and "0.058" thereafter. Minor.
- **Per-route naming is clean.** "per-route," "route-dependent," "capability
  corner," "(XRC50$_\ell$, CLIR@10$_\ell$)," "indicative" used uniformly in §8 +
  Limitations; "router … headroom, not a recommendation" phrasing matches between
  §8 (1100–1103) and Limitations (1216–1217).
- **Both RBO ceilings (0.39 / 0.19) still named + "any model"-framed** (abstract,
  intro, §6.2, conclusion) — no regression.
- **Both non-significant correlations carry n.s. + n=7** (two-tax 746–747/1205,
  trap 631/1206) and stay out of abstract/intro/conclusion. Held.
- **"home advantage" hyphenation still mixed** (cosmetic, carryover).

## Abstract/Conclusion alignment issues

- **Abstract is clean of all three protected leaks.** No ARI@100 number, no
  per-route routing claim, no non-significant ρ. The only ARI-adjacent phrasing is
  the L∞ floor framing on line 72 ("a measured share recoverable only by alignment,
  not re-ranking") — that is the floor, not an ARI number, and matches the
  story.md-permitted optional half-sentence. Good restraint; the abstract did not
  bloat with the round's additions.
- **Conclusion is clean** (verified): no 0.229, no per-route, no ρ. It restates
  the spine — collapse (CLIR@10 0.50, home +0.55, RBO 0.39/0.19, confusion 14–78%),
  cost (XRC ~3.5×, knee K\*=5, floor L∞=0.058 "the only part of the gap a
  re-ranker cannot move"), embeddinggemma the Pareto capability corner (bge-m3
  cheaper-to-read), separability cause (+0.96, robust), align-not-re-rank, budget
  rule. The optional "only part a re-ranker cannot move" clause landed (line 1271)
  without an ARI number. Abstract and conclusion make the same claims with the same
  emphasis.
- **Emphasis split unchanged and deliberate** (abstract last sentence = decision;
  conclusion last sentence = the two recommended habits). Leave as-is, as in
  rounds 2–3.

## What's already cohesive (leave alone)

- **The ARI thread is the round's structural win — leave the arc intact.** Define
  (§4) → quantify (§6.1) → mechanism-bridge (§7) → deploy-reinforce (§8) →
  future-target (§9). It makes "align, don't re-rank" a measured per-model split at
  five touchpoints with one number (0.229) and one floor (0.058), and it
  strengthens C2/C3/C4 at once. The only edit it needs is trimming the §6.1
  re-definition (joint #1); the arc itself must not be disturbed.
- **The per-route §8 paragraph honors the four-point honesty contract verbatim.**
  Single-model default kept, router = headroom, XRC axis indicative, es never
  imputed, thin-n caveat in both §8 and Limitations. No whiplash with the
  capability-corner spine. This was the round's top over-claim risk and it reads as
  bounded upside. Do not touch the framing.
- **The cost-object spine is still one continuous thread** (Intro cost clause →
  Metrics XRC/RRC/ARI/DEG → §6.1 frontier/budget/ARI → two-tax bridge → §7
  separability-⇒-floor-⇒-ARI → §8 budget-the-knee + align-not-re-rank + per-route
  upside → Conclusion). The ARI addition extended it cleanly rather than branching.
- **The DEG gate still anchors every "non-degenerate" usage** (round-3 win
  preserved; the two former slips now fixed).
- **The N2 superlative stays dead, the two RBO ceilings stay separated, the MT null
  stays a null, availability stays descriptive (n=5), +0.96 stays the robust
  mechanism** — all the round-3 honesty bedrock is intact; nothing regressed.

---

### Bottom line for the conductor

**Seams closed:** both round-3 must-fixes (the "eight"/"seven" relabel and the
fig21 "sibling-" deletion) are confirmed fixed; F3 cascade cite present. **New
seams:** only one real one — the §6.1 ARI paragraph re-defines an identity §4
already owns (joint #1, ~2-sentence trim). ARI is integrated as **one thread**
with RRC (not a restatement); the per-route §8 paragraph is **coherent with the
single-model recommendation** (no whiplash, contract held). **The paper is
overloaded — on figure count, not on logic.** At **27 figures + 2 tables = 29
floats (17 in §6)** it is over the industry-track page budget. This is the round
to **FREEZE the analysis** (cp_fig22/23 are the last additions — agreed) and
**CUT ~3–4 legacy low-information floats**: drop the two radars (cp_fig14 +
ag_fig10, fully covered by the tables + cp_fig17), and merge the mate-retrieval
pair (cp_fig06/07) and the language-collapse pair (cp_fig09/10) into one
two-panel figure each. That takes the paper from 29 → 25 floats with no argument
lost. Round 5+ should be polish only: the §6.1 ARI trim, the float-order swap,
hyphenation, and prose compression for the page budget — no new results.
