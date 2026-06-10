# Cohesion review (round 3)

Reviewer #3 (Cohesion). Scope: flow, framing, internal consistency. I did not
re-verify numbers against `reports/` (Reviewer #2) nor judge novelty
(Reviewer #1). I read the current `paper/main.tex` (round-3 revision, 1181 lines),
the four new figures it references (cp_fig18 cost frontier, cp_fig19 RRC budget,
cp_fig20 degeneracy gap, cp_fig21 two-tax — all rendered and inspected),
`paper/loop/round_3/story.md`, and my own `paper/loop/round_2/critic_cohesion.md`
to check whether the five round-2 glue-joint fixes landed and whether this round's
heavy additions (four figures + a DEG-gate paragraph + the two-tax spine) opened
new seams. I confirmed figure presence, reference counts, and population labels on
disk as cohesion questions only.

## Overall: does it read as one story? (1 paragraph)

Yes — and it is the most cohesive the paper has been. Four of the five round-2
seams are fully closed, and the three new "cost objects" (cost frontier, RRC
budget curve, DEG gate) plus the two-tax spine are woven in as one continuous
thread rather than bolted on: the cost story now runs Intro cost clause
(lines 112–116) → adjacent Metrics definitions with the DEG gate locked between
them (§4) → three §6.1 beats (frontier, RRC budget, separability back-reference)
→ Deployment's "budget the re-ranker by the knee" + "align, don't re-rank" →
Conclusion (knee $K^{\!*}{=}5$, floor $L_\infty=0.058$). The DEG gate is the
quiet hero of the round: defining "degenerate" once (§4, CLIR@10<0.10, exactly
{gte, e5}) retires the four undefined "non-degenerate" usages I flagged in
round 2 and re-anchors the winner-take-all contamination footnote — a genuine
cohesion upgrade, not a patch. The honesty discipline held perfectly: both
non-significant correlations (two-tax $\rho{=}{-}0.59$, trap $\rho{=}{+}0.29$)
appear only in §6.1, §6.2, and Limitations, and are absent from
abstract/intro/conclusion, each carrying its `n.s.` caveat inline. The §6.1/§6.2
figure density is heavy (eleven figures across the two subsections) but did not
overload the prose — every figure is interpreted, and the two retired figures
(cp_fig15 XRC bar, cp_fig16 RRC bar) were cleanly dropped (0 references) in favor
of cp_fig18/cp_fig19, so no orphan-figure bloat was introduced. The one real new
seam is a side effect of the round's best move: by making "non-degenerate" a
*precise* 7-model term (the DEG gate), the paper now contradicts itself in the
two older places that still say "**eight** non-degenerate models" (lines 557,
657). That is a one-word relabel in each spot. Everything else is local polish.

## Did the 5 round-2 seams close? (one line each)

1. **B2 line-605 RBO "best model" regression — CLOSED.** The §6.2 sentence that
   introduces the RBO figure now reads "The best cross-lingual RBO \emph{any} of
   the nine models reaches is only $0.39$ ... a ceiling no model beats"
   (lines 710–711), matching abstract (64), intro (107), and conclusion (1144)
   verbatim in both phrasing ("any model") and rhetorical direction (ceiling, not
   achievement). The two remaining "best model" strings (lines 567, 576) are
   unrelated correct uses ("best model" = which model's matrix is shown; "the best
   model \texttt{embeddinggemma} reaches a same-language gold at depth 2"). Clean.

2. **de↔zh asymmetry orphan — CLOSED.** The $+0.23$ is now folded into the
   cp_fig03 caption ("The most asymmetric directed pair is
   de$\leftrightarrow$zh (gap $+0.23$; asymmetry panel not shown)," lines 569–570)
   *and* stated in the §6.1 prose (line 555). cp_fig04 remains correctly
   unreferenced (0 references) as a superseded panel — no longer an orphan, because
   the named instrument now has a caption home. Exactly the round-2 fix executed.

3. **Two-line-items spine reaching prose — CLOSED, and upgraded.** §6.2 now opens
   with the bridging sentence ("If \S\ref{ssec:cp} measured what cross-linguality
   costs to \emph{read} ... the second line-item of the same bill," lines 684–687)
   and backs it with the measured-but-weak two-tax non-redundancy
   (cp_fig21, $\rho{=}{-}0.59$, $n{=}7$, $p{=}0.16$, n.s., inline) framed strictly
   as "neither benchmark is a clean proxy for the other ... It is not an
   independence result" (line 694). This converts the round-2 "sequenced but not
   unified" seam into an explicit spine. The honesty framing matches story.md risk
   #2 word-for-word — descriptive/motivating, never significant, never
   "independent."

4. **"degenerate" defined once — CLOSED, and this is the round's strongest
   cohesion move.** §4 now has a dedicated "The degeneracy gate" paragraph
   (lines 443–456) defining DEG as a single criterion (CLIR@10<0.10), stating it
   flags exactly {gte, e5}, citing cp_fig20, with a footnote justifying the
   single-criterion over the AND-gate (matches story.md risk #5). Every later
   "non-degenerate" qualifier and the winner-take-all contamination footnote
   (lines 773–777) are now anchored to it. The four-times-undefined qualifier I
   flagged in round 2 is gone.

5. **RRC@100-vs-@1000 framing — CLOSED.** The §6.1 RRC beat now states the
   practical-vs-absolute pairing in one breath ("a realistic top-100 re-ranker
   recovers at most $74\%$ ... leaving $\sim$25\% on the table ... and the floor
   $L_\infty=\mathbf{0.058}$ ... unrecoverable by \emph{any} re-ranker," lines
   645–649), exactly the framing Deployment used, pulled one step earlier as the
   round-2 fix requested. The depth-vs-ceiling story now reads as one fact.

Round-2 *secondary* carryover: **home-advantage hyphenation still mixed**
(2 hyphenated at lines 132/957, 14 unhyphenated). Cosmetic, unchanged from round 2;
harmonize to unhyphenated on a final pass.

## Broken glue joints (ranked, each: location -> problem -> concrete fix)

**1. NEW SEAM (the only real one) — "eight non-degenerate models" (lines 557,
657) contradicts the DEG gate, which defines "non-degenerate" as the 7-model
set.**
- *Problem.* The round's best move — locking the DEG gate (§4, lines 443–456) so
  that "degenerate" means exactly {gte, e5} and "non-degenerate" means the
  remaining **seven** models — made "non-degenerate" a *precise cardinality term*.
  The paper now uses "**seven** non-degenerate models" correctly at four sites
  (lines 463, 600, 688, 701) but "**eight** non-degenerate models" at two:
  - line 557, the directional-hub pooling ("pooled over the eight non-degenerate
    models, French ($0.375$) and English ($0.367$) ...");
  - line 657, the cp_fig19 RRC-budget caption ("for the eight non-degenerate
    models, with the knee $K^{\!*}$ ...").
  A reader who internalizes the §4 gate (= 7) and then reads "eight non-degenerate"
  has a direct contradiction at the two most quantitative beats of §6.1. The
  underlying populations are *defensible but different*: line 557 pools every model
  with a defined recall (all but `gte`, which retrieves ~nothing → 8); line 657
  plots every RRC curve except `gte`'s undefined/flat one (→ 8). Both legitimately
  exclude only `gte`, but the term "non-degenerate" is now reserved for the
  gate's 7-set, so the label is wrong even though the count is right for those
  sets.
- *Fix.* Relabel the two "eight" sites so they do not use the gate-bound term.
  Line 557: "pooled over the eight models with defined cross-lingual recall (all
  but the degenerate \texttt{gte-base}), French ($0.375$) and English ($0.367$)
  ..." Line 657 caption: "Re-ranker budget curves ... for the eight models with a
  defined RRC curve (all but \texttt{gte-base}, whose pool is empty) ...". This
  keeps the (correct) count, removes the contradiction with the §4 definition, and
  reads naturally. ~6 words each; no figure or number changes. This is the single
  must-fix of the round.

**2. MINOR — the e5 $L_\infty$ range in the §6.1 RRC beat (line 649) names a
degenerate model as the upper end of a "non-degenerate" object.**
- *Problem.* The RRC budget beat says "This floor ranges from $0.058$
  (\texttt{embeddinggemma}) to $0.372$ (\texttt{e5-large-instruct})" (lines
  649–650), but `e5-large-instruct` is one of the two models the DEG gate just
  excluded (§4). Citing it as the range endpoint inside the headline budget-curve
  paragraph slightly muddies the "we exclude {gte, e5} from non-degenerate
  summaries" discipline established four paragraphs earlier — a reader may wonder
  whether e5 is in or out for RRC. The cp_fig19 caption (lines 656–661) correctly
  scopes the curve to non-degenerate models, so the prose and caption are subtly
  out of step on whether e5 participates.
- *Fix.* Either (a) drop the e5 endpoint and give the non-degenerate range
  ("ranges from $0.058$ (\texttt{embeddinggemma}) to $0.226$ across the seven
  non-degenerate models" — pick the real non-deg max), or (b) keep e5 but label it
  as the degenerate illustration: "... to $0.372$ for the degenerate
  \texttt{e5-large-instruct}, which the gate excludes." Option (b) is the smaller
  edit and preserves the dramatic contrast. Low priority but it tightens the gate
  discipline the round worked to establish.

**3. MINOR — float/reference order in §6.1: cp_fig20 (DEG gate) floats into §4 but
cp_fig21 (two-tax) opens §6.2; the two "new analysis" figures bracket the section
asymmetrically.**
- *Problem.* This is a pure float-placement nit, not a logical break. cp_fig20 is
  referenced at line 449 (§4 Metrics) and its `\begin{figure}` sits at line 458,
  so the DEG figure lands early — correct, it is defined there. But the §6.1
  figure sequence in source order is cp_fig20 (§4) → cp_fig02 → cp_fig03 → cp_fig18
  → cp_fig05 → cp_fig19 → cp_fig06 → cp_fig07, i.e. the new load-bearing figures
  (18, 19) are interleaved with the older ones in *reference* order, which is
  correct, but cp_fig19 (RRC budget) is referenced at line 641 and floats at 655,
  *before* cp_fig06/cp_fig07 (the mate-retrieval figures it conceptually builds on,
  referenced at 636–639). Same micro-issue I flagged for cp_fig16 in round 2,
  inherited by cp_fig19.
- *Fix.* Acceptable as-is (LaTeX floats; the prose introduces mate-retrieval depth
  at 636–639 *before* the RRC consequence at 641, so reference order is actually
  fine). If a clean float order is wanted, move the cp_fig19 `\begin{figure}`
  block after cp_fig07's. Lowest priority; do not let it cost a rewrite.

## Unmet promises / orphan results

- **All five contributions deliver and are referenced.** C1 → §3 + abstract/intro;
  C2 → §4 (now including the frontier/budget/DEG framing) + §6 + Deployment;
  C3 → §7 (separability + RRC floor); C4 → §8 (frontier-grounded decision);
  C5 → §3 + Appendix (softened, `% TODO` comments). No contribution is promised and
  dropped.
- **The four new figures are each referenced exactly once and interpreted in
  text.** cp_fig18 (frontier, line 587), cp_fig19 (RRC budget, line 641), cp_fig20
  (DEG gate, line 449), cp_fig21 (two-tax, line 689) — each has a self-contained
  caption with its headline number and a prose interpretation. No new orphan.
- **Two figures cleanly retired.** cp_fig15 (old XRC bar) and cp_fig16 (old RRC
  bar) are now referenced 0 times, replaced by cp_fig18/cp_fig19 — exactly the
  figure-retirement bookkeeping story.md risk #10 asked for. cp_fig04 (asymmetry),
  cp_fig08, cp_fig13 remain present-but-unreferenced as superseded panels; cp_fig04
  is no longer an orphan because its number now lives in the cp_fig03 caption.
- **No result is dropped in without interpretation.** The new DEG-gate paragraph,
  the cost-frontier beat, the RRC-budget beat, and the two-tax spine each state
  their number, cite their figure, and draw the deployment read-off.

## Terminology & notation inconsistencies

- **"seven" vs "eight" non-degenerate — the one real inconsistency** (joint #1
  above): "seven" at lines 463/600/688/701 (gate-correct), "eight" at lines
  557/657 (contradicts the gate). New this round, caused by the DEG gate making the
  term precise.
- **"non-degenerate" is now defined and consistently anchored** (round-2 joint #4
  resolved): the §4 gate paragraph (lines 443–456) is the single definition, and
  the qualifier is used as a gate reference everywhere except the two "eight" slips.
- **XRC / RRC / DEG-gate naming is clean.** "XRC," "XRC50," "RRC," "RRC@K,"
  "$L_\infty$," "$K^{\!*}$," "degeneracy gate," "Pareto-optimal capability corner,"
  "cost-vs-capability frontier" are used uniformly across Metrics, Results,
  Deployment, Conclusion, and Limitations. The N2 superlative is dead — no "lowest
  reading cost = embeddinggemma" anywhere; the frontier framing ("capability
  corner, not cheapest; bge-m3 the cheaper-to-read alternative") is identical at
  abstract (76–78), C4 (lines 152–153), §6.1 (591–602), Deployment (1004–1019), and
  Conclusion (1151–1153).
- **Both RBO ceilings ($0.39$ / $0.19$) named and "any model"-framed at all four
  sites** (abstract 64, intro 107, §6.2 710, conclusion 1144). Round-2 B2 fully
  resolved.
- **Both non-significant correlations carry `n.s.` and `n=7` consistently** (two-tax
  lines 689/703/1098, trap 601/1099) and never leak into
  abstract/intro/conclusion. Honesty discipline held.
- **"home advantage" hyphenation still mixed** (2 hyphenated lines 132/957, 14
  unhyphenated). Cosmetic carryover from round 2; harmonize on final pass.
- **$\tau{=}0.40$ is consistently flagged "stated (untuned)"** at §6.1 (line 593)
  and named in the cp_fig18 caption — matches story.md risk #4.

## Abstract/Conclusion alignment issues

- **Strong alignment, now including the frontier.** Abstract and Conclusion make
  the same claims with the same emphasis: collapse (CLIR@10 $0.50$, home $+0.55$),
  two RBO ceilings ($0.39$/$0.19$, both named both ends), confusion ($14$–$78\%$),
  the cost objects (XRC $\sim3.5\times$; RRC@100 $\le 0.74$; $L_\infty=0.058$; knee
  $K^{\!*}{=}5$ — the conclusion now states the knee explicitly), separability cause
  ($+0.96$, robust), alignment-not-re-ranking, the frontier verdict
  (embeddinggemma the capability corner, bge-m3 cheaper-to-read), and the budget
  rule. The conclusion introduces no claim the abstract lacks.
- **Honesty note respected at both ends.** Neither non-significant correlation
  appears in the abstract or conclusion (verified by grep: $\rho$ strings occur
  only at 601, 689, 703, 1098–1099). CLIR-MRS/MRS correctly absent from both as a
  claim.
- **One emphasis nit (unchanged, harmless):** abstract's last sentence leads with
  embeddinggemma (model-pick), conclusion's last sentence leads with the two
  recommended habits (report robustness / treat as representation problem). This is
  the deliberate abstract=decision / conclusion=method-habit framing I flagged in
  round 2; leave it, do not "harmonize."

## What's already cohesive (leave alone)

- **The DEG gate as connective tissue.** Defining "degenerate" once and re-using it
  to anchor the XRC "undefined" note (line 582), the RRC scope, the two-tax n=7,
  and the winner-take-all contamination footnote (lines 773–777) is the cleanest
  structural win of the round. Do not disturb — just fix the two "eight" labels so
  they honor it.
- **The cost-object thread is one continuous spine.** Intro cost clause →
  adjacent Metrics definitions (XRC, RRC, DEG between them) → three §6.1 beats →
  two-tax bridge into §6.2 → Analysis separability-⇒-RRC-floor crux → Deployment's
  "budget the knee" + "align don't re-rank" → Conclusion. Five+ touchpoints, same
  instruments, same emphasis. This was the round's biggest cohesion risk and it
  reads as one thread.
- **The two-tax spine is correctly load-bearing-light.** It is framed as the
  *motivation* for two benchmarks, with the negative sign, the sub-0.6 $|\rho|$, and
  the n.s. caveat inline, and explicitly "not an independence result." It carries
  the §6.1→§6.2 transition without resting any conclusion on it. Exactly the
  scoping story.md demanded.
- **The N2 superlative stays dead.** The frontier framing is internally consistent
  everywhere; the "capability corner, not cheapest" phrasing never slips back to a
  cheapest-reader superlative. Keep.
- **The reframed Analysis spine still holds** (joint-failure ¶1 → availability-but-
  residual ¶2 → structure-trap → bias↔inconsistency hedged → separability-⇒-RRC-
  floor), landing on Deployment's "align, don't re-rank." The negative availability
  slope is consistently "descriptive (n=5)." Untouched and intact.
- **§6.1/§6.2 figure density did not overload the prose.** Eleven figures across
  the two subsections is heavy for 8 pages, but each is interpreted and the two
  retired bars kept the net count flat. Length/balance still appropriate; the new
  content displaced no payoff section.
