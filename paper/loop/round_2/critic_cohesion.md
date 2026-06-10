# Cohesion review (round 2)

Reviewer #3 (Cohesion). Scope: flow, framing, internal consistency. I did not
re-verify numbers against `reports/` (Reviewer #2) nor judge novelty
(Reviewer #1). I read the current `paper/main.tex` (round-2 revision), the 23
referenced figures (all present on disk), `paper/loop/round_2/story.md`, and my
own `paper/loop/round_1/critic_cohesion.md` to check whether the six round-1
glue-joint fixes landed and whether this round's heavy edits (XRC/RRC, CLIR-MRS
demotion, novelty reframes) opened new seams. I confirmed benchmark attribution
of numbers/figures only as a cohesion question.

## Overall: does it read as one story? (1 paragraph)

Yes — and it is now tighter than round 1. All six round-1 broken glue joints are
fixed (verified individually below), and the two heaviest new threads — the
XRC/RRC cost metrics and the CLIR-MRS demotion — are genuinely woven in, not
bolted on. The cost story is the clearest win: XRC and RRC are introduced as a
pair in the Intro ("both are costly," lines 111–114), defined adjacently in
Metrics (§4, the last two definitions before the demoted composite), measured in
a dedicated Results beat (§6.1 "The cost of cross-linguality" + "Foreign twins
are buried, and a re-ranker cannot reach them all"), and cashed out twice in
Deployment ("Align, do not re-rank" and the headroom paragraph) — the same two
instruments, same emphasis, four touchpoints. The CLIR-MRS demotion is executed
honestly and consistently: every place the composite appears it is explicitly
labelled "table-ordering convenience" and immediately followed by the per-axis-
dominance claim, so the round's most adversarial edit (aggregation-invariance
FAILS) reads as a strength rather than a retraction. The spine ("average recall
hides a costly cross-lingual collapse; we make it measurable, name the survivor,
trace it to a separability deficit, so align don't re-rank") is intact across
abstract / intro / results / analysis / deployment / conclusion. The remaining
seams are small and local: one residual "best model" RBO phrasing in §6.2 that
contradicts the "any model" phrasing everywhere else (B2), the de↔zh asymmetry
number that still has no figure (round-1 orphan, not closed), and a couple of
new-this-round micro-frictions from the XRC/RRC additions (the "two line-items"
spine from story.md never made it into the prose, so the two benchmarks are
sequenced cleanly but not explicitly *unified*; and XRC50's degenerate/
catastrophic outliers are introduced before the reader is told which models are
"degenerate"). None require re-architecting; all are ≤2-sentence fixes.

## Did the 6 round-1 fixes land? (one line each)

1. **Analysis two-benchmark interleave (joint #1) — LANDED.** The fused paragraph
   is split into two benchmark-labelled sentences ("On the *alias-graph*
   benchmark…" line 776; "On the *cross-lingual* benchmark…" line 788) with two
   separate footnotes, and is now led by the joint-failure thesis
   ("modal failure is a same-language sibling," §Analysis ¶1) that gives the split
   a unifying spine. Clean.
2. **Abstract single-RBO-ceiling (joint #2) — LANDED.** Abstract line 64 now reads
   "cross-lingual RBO ceiling $0.39$ on the alias-graph benchmark, $0.19$ on the
   cross-lingual benchmark," matching intro (105–107) and conclusion (1009)
   verbatim. The abstract↔body↔conclusion RBO mismatch is gone.
3. **Related Work → Benchmarks transition (joint #3) — LANDED.** §2 now ends on a
   forward bridge (lines 263–267: "Having positioned our four contributions
   against CLEF-IP/DAPFAM… we now build the benchmarks that deliver C1") and the
   future-work/conformal disclaimer was moved up into the calibration paragraph
   (lines 222–227). The hard edge into §3 is gone.
4. **`\todo{}` markers in body prose (joint #4) — LANDED.** Zero red `\todo{}` in
   body text; the four untraced-number flags are now `% TODO` LaTeX comments
   (lines 288, 339, 977, 1040), invisible to the reader. The corpus and C5
   paragraphs demote the unavailable numbers to one clean "deferred to the system
   description" sentence each (lines 285–287, 335–338).
5. **cp_fig11 caption over-claim + radar under-interpretation (joint #5) —
   LANDED.** cp_fig11 caption now describes the bars ("cross-language gold (red)
   is systematically harder to separate… than same-language gold (green)") and
   presents $+0.96$ as a *text statistic the figure motivates* (lines 870–876).
   The radars get the interpretive clause ("leads on consistency and separability,
   not on raw recall alone," line 658).
6. **Teaser-vs-leaderboard ordering signpost (joint #6) — LANDED.** §6.3 now
   carries the signpost sentence (lines 653–656: "Note the order changes from
   Figure~\ref{fig:teaser}: ranking by CLIR-MRS rather than raw recall reshuffles
   the middle of the field (granite-278m and qwen3-0.6B swap), which is the point
   of the paper"). The inconsistency is now evidence for the thesis.

Round-1 *secondary* items: the "two related bias proxies" half-clause landed
(line 845); the universal-blind 12% orphan is fixed — it is now *introduced in
Analysis* (lines 829–832, tied to the structure-trap beat) and *re-used* in
Deployment (line 919), so it is earned before it is leveraged. The home-advantage
hyphenation is still mixed (2 hyphenated at lines 130/849, ~13 unhyphenated) —
cosmetic, unchanged.

## Broken glue joints (ranked, each: location -> problem -> concrete fix)

**1. §6.2 line 605 says "Even the best model reaches a cross-lingual RBO of only
$0.39$" — the one place the B2 distinction regressed.**
- *Problem.* story.md risk #5 (B2) is explicit: the RBO ceiling must be phrased
  as "the best cross-lingual RBO *any model* achieves" (a ceiling *across* models)
  and never as "the best model" (which reads as one model's score). The intro
  (line 106, "*any* of our nine models achieves"), abstract (line 64), and
  conclusion (line 1009, "any model reaches") all use the correct "any model"
  phrasing. But the Results §6.2 sentence that actually *introduces* the figure
  reverts to "Even the best model reaches… $0.39$." A reader cross-checking the
  intro's "any model" claim against the Results beat sees two different framings
  of the same number at the two most load-bearing sites. It is also subtly
  self-undercutting: "even the best model only reaches 0.39" frames 0.39 as a
  *best-case achievement*, whereas the paper's argument is that 0.39 is a *ceiling
  nobody beats* — opposite rhetorical direction.
- *Fix.* Rewrite line 605 to "Even the best-performing model tops out at a
  cross-lingual RBO of $0.39$ — the ceiling *no* model in our suite beats
  (Figure~\ref{fig:ag_rbo})," or more simply "The best cross-lingual RBO any of
  the nine models reaches is only $0.39$ (Figure~\ref{fig:ag_rbo})," matching the
  intro phrasing verbatim. One-sentence edit, closes the only abstract/intro↔body
  framing mismatch left.

**2. The de↔zh asymmetry number ($+0.23$) still has no figure — round-1 orphan
not closed (lines 506–507).**
- *Problem.* §Metrics names "direction-asymmetry" as an instrument (lines
  360–366) and §6.1 states "the most asymmetric pair is de$\leftrightarrow$zh
  (gap $+0.23$)" (line 507), but cp_fig04 (`cp_fig04_clir_direction_asymmetry.png`,
  present on disk) is not referenced anywhere, and the cp_fig03 caption (lines
  519–522) does not fold the asymmetry in. So the asymmetry is a named metric and a
  stated number with no visible payoff — exactly the round-1 orphan I flagged, and
  it was not addressed this round. Same value $+0.23$ also collides visually with
  the bge-m3 "home" cell ($+0.23$) in Table~1 (line 692); harmless but a reader
  scanning for the asymmetry value could mis-anchor.
- *Fix.* Cheapest: fold the asymmetry into the cp_fig03 caption — "…the most
  asymmetric directed pair is de$\leftrightarrow$zh ($+0.23$; asymmetry panel not
  shown)." Or reference cp_fig04 with a half-sentence in the anisotropy paragraph.
  Either closes the orphan for ~6 words; do not leave a named instrument with no
  panel and no caption mention.

**3. NEW SEAM — the "cross-lingual tax has two line-items" spine (story.md thesis
overlay) never reaches the prose, so §6.1 and §6.2 are *sequenced* but not
*unified*.**
- *Problem.* story.md (lines 140–147) proposes the connective tissue that makes
  the two benchmarks one story: cross-lingual retrieval pays a *reading-cost tax*
  (XRC, measured on the cross-lingual benchmark) and a *confusability tax* (the
  look-alike, measured on the alias-graph benchmark) — "each benchmark measures
  one line-item of the same bill." The draft delivers both halves but never states
  the unifying frame. §6.1 ends on the re-ranker ceiling and §6.2 opens cold with
  "Asking the same compound in five languages…" (line 604) — a clean topic switch,
  but the reader is left to infer *why these two benchmarks belong in one paper*.
  The Intro's "two distinct failures… both are deployment bugs" (line 103) is the
  closest gesture, but it is upstream of the benchmarks and does not name the
  tax/line-item framing. This is the round-1 "two-benchmark structure is the one
  persistent strain" seam, now *narrower* (the numbers no longer blend) but still
  *open* at the story level.
- *Fix.* Add one bridging sentence at the head of §6.2 (or end of §6.1):
  "If §6.1 measured what cross-linguality *costs to read*, the alias-graph
  benchmark measures what it *costs in precision* — the second line-item of the
  same bill: a look-alike compound that out-ranks the gold." This is connective
  tissue, not a load-bearing claim (story.md explicitly scopes it that way), and it
  converts the cleanest remaining seam into a one-sentence spine. Lowest-effort
  highest-cohesion edit in the paper.

**4. NEW SEAM — XRC's outlier models ("degenerate," "catastrophic") are named in
§6.1 (lines 530–535) before the reader is told what makes a model degenerate.**
- *Problem.* The XRC beat says granite is "cleanest" ($1.25\times$), nomic
  "balloons" ($11.5\times$), e5 "catastrophic" ($97.75\times$), and gte "is
  degenerate (it retrieves almost nothing, so XRC is undefined)." The parenthetical
  for gte is good, but "the worst non-degenerate model" language recurs in the RRC
  beat (line 572), the leaderboard caveat (line 667–669, "gte-base 'wins'… because
  it retrieves almost nothing"), and Deployment (line 899, "lowest non-degenerate
  XRC50") — i.e. "degenerate"/"non-degenerate" becomes a load-bearing qualifier
  used four times, but it is never *defined* as a term. The reader assembles the
  definition piecemeal (gte retrieves ~nothing; e5 nearly so). For a number-dense
  industry reader this is a small but repeated friction.
- *Fix.* One clause at first use (the XRC beat, line 534): "…\texttt{gte-base} is
  *degenerate* — it retrieves almost nothing (R@10 $0.004$), so we exclude it and
  \texttt{e5-large-instruct} from 'non-degenerate' summaries throughout." Then
  every later "non-degenerate" is anchored. Alternatively define it once in §5
  Experimental Setup. Either removes the four-times-undefined qualifier.

**5. NEW SEAM (minor) — RRC is defined in Metrics §4 as a "ceiling" but the §6.1
beat reports two RRC numbers (RRC@100, RRC@1000) whose relationship to "the
ceiling" needs one word.**
- *Problem.* §4 defines RRC@K and says "$1-\mathrm{RRC@}K$ is provably
  unrecoverable by any top-$K$ re-ranker" (lines 419–421) — clean. But §6.1 (lines
  568–572) reports both RRC@100 $=0.7445$ and RRC@1000 $=0.9416$ and the prose
  glosses them as "a top-100 re-ranker can recover at most $74\%$… and $5.84\%$ are
  unrecoverable forever ($1-\mathrm{RRC@1000}$)." The two K's do double duty (one is
  a realistic re-ranker pool, one is the "forever" bound) but the text does not
  signpost *why two K's*. A reader can reconstruct it, but the "$74\%$ at top-100"
  vs "$5.84\%$ forever" pairing reads as two facts rather than one (depth-vs-ceiling)
  story until Deployment (lines 946–948) finally makes the "$25\%$ on the table at
  top-100 vs $5.84\%$ forever" contrast explicit.
- *Fix.* Pull that Deployment framing one step earlier: in §6.1, add "—so a
  realistic top-100 re-ranker leaves $\sim25\%$ on the table, and $5.84\%$ is lost
  to *any* re-ranker, however deep." This makes K=100 vs K=1000 read as the
  practical-vs-absolute bound in one breath, matching Deployment. Optional polish,
  not a break.

## Unmet promises / orphan results

- **C5 promise/payoff is now consistent (improved from round 1).** The Intro
  contribution (C5, lines 148–150) is softened to "with human validation summarized
  in the system description," the Benchmarks payoff (lines 330–338) and Appendix
  (lines 1035–1039) match that softened phrasing, and the human-eval numbers are
  `% TODO` comments. C5 is no longer a promise the reader watches fail in real time;
  it is honestly scoped as supporting evidence. Resolved.
- **Directional asymmetry remains the one true orphan** (joint #2 above): a named
  metric (§4) and a stated number (§6.1) with no figure and no caption mention.
- **No new orphans from the XRC/RRC additions.** Every new figure (cp_fig15 XRC,
  cp_fig16 RRC, cp_fig17 ribbon, ag_fig11 availability, ag_fig12 joint-failure) is
  referenced and interpreted in text, with a headline number in the caption. The
  two-level confusion-severity split (lines 622–629) is grounded and its scope
  ("graded ChEBI hop-distance law is future work") is honestly flagged in-paragraph
  and in Limitations. No result is dropped in without interpretation.
- **cp_fig04, cp_fig08, cp_fig13, ag_fig3, ag_fig4, ag_fig9** are present on disk
  but unreferenced — correctly excluded as superseded panels (cp_fig13 is the old
  CLIR-MRS leaderboard now replaced by Table~1; ag_fig3/4 fold into the reframed
  Analysis), *except* cp_fig04 (asymmetry), which should be either used or its
  number folded into a caption (joint #2). The rest are clean exclusions, not
  orphans.

## Terminology & notation inconsistencies

- **"any model" vs "best model" for the RBO ceiling — the one real inconsistency**
  (joint #1 above): correct at lines 64/106/1009, wrong at line 605.
- **"degenerate" / "non-degenerate" is used as a load-bearing qualifier 4× but
  never defined** (joint #4 above). New this round (it rides in with XRC/RRC).
- **XRC / RRC naming is clean and consistent.** "XRC," "XRC50," "cross-lingual
  reading-cost multiplier," "RRC," "RRC@K," "re-ranker recoverability ceiling" are
  used uniformly across Metrics, Results, Deployment, Conclusion, and Limitations.
  Eq. labels (eq:xrc, eq:rrc) are referenced at point of use (lines 529, 568). Good.
- **CLIR-MRS / MRS demotion language is uniform.** Every appearance is tagged
  "table-ordering convenience" / "ordering convenience" / "reporting convenience"
  (abstract implicitly, lines 134–135, 424, 438–439, 981–983) and never claimed as a
  contribution. The two composite names are kept distinct (CLIR-MRS for
  cross-lingual, MRS for alias-graph) — the role file's named worry is still handled
  correctly. Keep.
- **"cross-language AUC" is now spelled in full in the abstract** (line 72),
  matching Analysis (lines 393, 859) and Conclusion (line 1017) — round-1 joint
  resolved; the $r(\text{cross-language AUC},\text{CLIR@10})=+0.96$ string is now
  textually identical at all four sites.
- **"home advantage" hyphenation still mixed** (line 130 "home-advantage," line
  849 "home-advantage," ~13 others unhyphenated). Cosmetic; harmonize to
  unhyphenated "home advantage" (the dominant form) on a final pass.
- **Language codes** (en/de/es/fr/zh) uniform; figures referenced in ascending
  order within each subsection except the deliberate cp_fig16 (RRC) appearing in
  text before cp_fig06/cp_fig07 (mate-retrieval) in §6.1 — the RRC sentence (line
  563–572) cites Fig.~16 then Figs.~6/7 are placed after. This is a minor float-order
  vs reference-order mismatch: the RRC figure is referenced before the mate figures
  it conceptually depends on. Acceptable (LaTeX will float them), but if a clean
  reference order matters, move the cp_fig16 `\begin{figure}` after cp_fig07, or
  reorder the prose so mate-retrieval (the depth) precedes the re-ranker ceiling
  (the consequence). Low priority.

## Abstract/Conclusion alignment issues

- **Strong alignment, now including the cost metrics.** Abstract and Conclusion
  make the same claims with the same emphasis: collapse (CLIR@10 $0.50$, home
  $+0.55$), two RBO ceilings ($0.39$/$0.19$, both named in both), confusion
  ($14$–$78\%$), the cost pair (XRC $\sim3.5\times$ in abstract line 67–70 and
  conclusion line 1012; RRC "top-100 re-ranker cannot fully recover" in both),
  separability cause ($+0.96$, robust), alignment-not-re-ranking, and the budget
  rule. The conclusion introduces no claim the abstract lacks. XRC/RRC are now
  first-class in both, so the round's headline upgrade is reflected at both ends.
- **CLIR-MRS is correctly absent from both abstract and conclusion as a claim** —
  consistent with its demotion. Good: the demotion did not leave a dangling
  composite reference at the framing endpoints.
- **One emphasis nit:** the abstract's last sentence (lines 74–77) bundles
  embeddinggemma + MT-null + budget rule, while the conclusion (lines 1018–1021)
  bundles alignment + budget rule + "report robustness next to recall." Both are
  faithful, but the abstract leads its final clause with embeddinggemma and the
  conclusion leads with alignment — a tiny emphasis swap. Not a problem; flag only
  so a later edit does not "fix" one to match the other and lose the deliberate
  abstract=model-pick / conclusion=method-habit framing.

## What's already cohesive (leave alone)

- **The XRC/RRC thread is integrated, not bolted on.** Introduced as a pair in the
  Intro cost clause, defined adjacently in Metrics, measured in two §6.1 beats,
  and cashed out in Deployment's "Align, do not re-rank" and headroom paragraphs —
  four touchpoints, same two instruments. This was the round's biggest cohesion
  risk and it reads as one thread. Do not disturb the structure.
- **The CLIR-MRS demotion is executed cleanly and turned into a strength.** The
  aggregation-ribbon "[1,4] range" caveat (lines 661–672) + the per-axis-dominance
  recommendation (Deployment lines 895–902) make the adversarial finding (invariance
  FAILS) reinforce the thesis instead of undercutting it. The winner-take-all
  contamination footnote (lines 667–669) is honest and well-placed.
- **The reframed Analysis spine holds.** "The modal failure is a same-language
  sibling" (¶1) → "availability sets the stage; a residual encoder bias remains"
  (¶2, now split cleanly by benchmark) → "structure-style questions are the trap"
  → "bias↔inconsistency, but the robust signal is one" → "separability deficit, so
  re-ranking cannot fix it" is a clean causal chain that lands on Deployment's
  "align, don't re-rank." The negative availability slope is consistently labelled
  "descriptive (n=5)" (lines 786, 800, 969). Keep.
- **The novelty reframes are cohesive with the body.** C1's "content-controlled,
  chemistry-ontology-grounded" narrowing, the CLEF-IP/DAPFAM boundaries (§2 ¶5,
  lines 229–241), and C3's "we *confirm*" reframe (§2 ¶3 + contributions C3) are
  stated consistently in Related Work, the contributions list, and the section
  bodies. The "we do not claim to discover" hedge (line 209) matches the
  contribution phrasing exactly. No drift between the defensive framing and the
  delivery.
- **The four-claim spine, the deployment thread, and section transitions** are all
  still intact and were not damaged by the heavy edits. Length/balance is still
  appropriate for 8 pages; the new XRC/RRC content displaced no payoff section.
- **The "don't ensemble vs oracle headroom" tension** is still handled in one
  paragraph (Deployment lines 910–921, "the two facts coexist"). Leave it.
