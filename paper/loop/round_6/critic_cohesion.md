# Cohesion review (round 6)

## Overall: does it read as one story? (1 paragraph)

Yes — the heavy cut did **not** break the story, and the mechanical restructure
was executed cleanly. The body now holds exactly the 7 planned floats (5 figures
at lines 172/517/572/621/806 + the 2 leaderboard tables at 658/683), all 18
relocated figures sit in grouped Appendix A–C with the robustness table in App D,
and **every relocated figure keeps at least one body `\ref`** (verified
label-by-label: no orphan floats, no figure whose only `\ref` was trimmed away).
All ten load-bearing headline numbers (CLIR@10 0.50, home +0.55, RBO 0.39/0.19,
confusion 14–78%, XRC50 3.5×, RRC@100 0.74, L∞ 0.058, r +0.96, MT −0.044) still
appear in body prose, so no beat lost its number when its figure left. The three
carried cleanups all landed (tau-bands `[0.385,0.43]`/`[0.33,0.435]`; collapse
`48.7×` in prose+caption; L∞ `0.058 (5.84%)` introduced once at L553 with all
other sites bare `0.058`). The Related Work 7¶→1¶ collapse is the cleanest win:
one paragraph, **all 28 citations preserved**, every positioning intact, and the
`(Arabic–English)` qualifier on `crosslingualcost2025` correctly present (L203).
The single real concern is **length**: the body is right at the 6-page boundary
with essentially no margin, so I would pre-emptively action the first further cut
rather than gamble on the compile.

## Length verdict: does the 6-page body fit?

**Verdict: borderline — plausibly ~6.0–6.3 pages; treat as "fits only if the
compile is kind," and stage one further cut now.**

Estimate basis (cannot compile):
- Body = lines 1–1023. Non-blank, non-comment prose ≈ **880 lines**, of which the
  `Limitations` block (L925–996, ~50 lines) does **not** count → ~**830 counted
  prose lines**. At ACL two-column density (~190 prose lines/full page) that is
  **~4.3–4.4 pages of text**.
- 7 body floats, all `width=\linewidth` (single-column) images + the two
  `\small` tables. Each figure ≈ 0.22–0.27 page; the two tables ≈ 0.18–0.22 page
  each. Floats ≈ **~1.7–1.9 pages**.
- Title/author/abstract block ≈ 0.35 page.
- **Total ≈ 6.0–6.3 pages.** That is over budget in the unlucky case and exactly
  at budget in the lucky case. EMNLP review limit is hard at 6.

Because the margin is zero-to-negative, **stage the writer's named first-further
cut now** rather than discover the overflow at submission:

**Cut 1 (the writer's flagged first cut — do this): move `ag_fig2`
(`fig:ag_conf`, L621–628) to Appendix B.** Its 14–78% confusion result already
lives in prose (L609–612) and `tab:ag_board` carries the per-model `conf` column,
so nothing is orphaned. This drops the body to **4 figures + 2 tables** and buys
~0.25 page — enough to clear 6pp comfortably. The only narrative cost: the
chemistry-confusability "hook" loses its in-body picture, but the prose sentence
+ the leaderboard column carry it. When you move it, add it to App B's intro
sentence (L1113 already says "Figures referenced from §6.2 / §7" — fine) and keep
the single body `\ref` at L611.

**If still over after Cut 1 (unlikely), Cut 2 — prose, not floats:** trim the
Deployment section's two longest paragraphs, which still carry round-5 verbosity:
- The per-route paragraph (L852–877) is ~26 lines and **duplicates** the
  Limitations per-route paragraph (L949–957) almost clause-for-clause (same
  `n_same=7/2/0`, same "indicative XRC axis," same "es undefined, never imputed").
  Cut the thin-sample mechanics here to one clause and let Limitations own them:
  delete L873–877 ("The route corners rest on thin per-language samples … not a
  reversal of the deploy-one-model decision") down to a single sentence. Saves
  ~0.1 page with zero information loss (it is verbatim-redundant with Limitations).
- The "Deploy embeddinggemma" paragraph (L827–850) restates the tau-band
  (L838–841) that the Results cost-frontier paragraph already stated twice
  (L502–503 prose + L526–528 caption). Compress L838–841 to "…over the stated
  τ-band (§\ref{ssec:cp})" and drop the inline `[0.33,0.435]`/`τ≈0.33` repetition.

Do **not** cut below the 4 spine figures (teaser, cost_frontier, rrc_budget,
cp_sep) + 2 tables — those carry C1–C4.

## Broken glue joints (ranked, each: location -> problem -> concrete fix)

1. **L991 (Limitations) references `Figure~\ref{fig:ari}`, which now lives in
   Appendix A (L1086).** -> This resolves fine (LaTeX cross-refs across the
   appendix boundary), and `fig:ari` is also `\ref`'d at L387 (Metrics) and L562
   (Results), so it is *not* orphaned. The only nit: a Limitations paragraph
   pointing the reader at an appendix figure for a "precise before/after target"
   reads slightly heavy for a forward-looking probe. -> *Optional*: change
   "The ARI decomposition (Figure~\ref{fig:ari}) hands this probe…" to "The ARI
   decomposition (Appendix~\ref{app:cp_extra}) hands this probe…" so the reader
   knows it is an appendix object, matching how the rest of the body now cites
   relocated floats. Low priority; not a breakage.

2. **`fig:cp_deg` is `\ref`'d only from the Metrics degeneracy-gate paragraph
   (L393), and that ref reads as in-line evidence ("flags exactly gte-base and
   e5-large-instruct (Figure~\ref{fig:cp_deg})").** -> The figure is now in
   App A (L1100). The sentence still parses, but a reader at L393 hits a forward
   pointer to an appendix figure with no "see Appendix" cue. -> *Minor fix*:
   make appendix-bound refs explicit where the body leans on them as evidence,
   e.g. "(Appendix Fig.~\ref{fig:cp_deg})". This is a readability polish, not a
   broken ref — apply consistently or not at all (see terminology note).

3. **Appendix section labels `app:cp_extra`, `app:ag_extra`, `app:agg_route` are
   defined (L1031/1112/1167) but never `\ref`'d.** -> Not an error and produces
   no warning, but they are dead labels. -> Either `\ref` them once from the body
   (e.g. "extended cross-lingual results in Appendix~\ref{app:cp_extra}") to give
   the reader a single grouped pointer, or delete the three unused `\label`s.
   Adding the grouped pointer is the better fix — it makes the body→appendix
   handoff explicit (see "see Appendix" readability below).

4. **Mixed citation style for appendix figures: the body sometimes names the
   figure bare ("Figure~\ref{fig:cp_home}", L458) and sometimes flags the
   appendix ("Appendix~\ref{app:robust}", L568/785).** -> After moving 18 figures
   out of the body, ~15 body sentences now say "Figure~X" for a figure the reader
   cannot find on the current page. -> **Recommended fix (single cheap pass):**
   for the relocated figures, prefix the ref with "Appendix" where the figure is
   the evidentiary anchor of the sentence (e.g. L458 →
   "(Appendix Fig.~\ref{fig:cp_home})", L465, L536, L728, L744, L756, L777). This
   is the one flow seam the restructure genuinely introduced: body prose that
   reads "(Figure~X)" mid-argument when X is pages away in the appendix. It
   resolves correctly but mildly disrupts the reader. Not blocking, but it is the
   highest-value polish for "do appendix refs read cleanly."

## Unmet promises / orphan results

None. Checked each contribution C1–C5 against delivery and each body float
against a `\ref`:
- **C1** (two content-controlled benchmarks) — delivered §3, `tab:cp_board`/
  `tab:ag_board` in body. The `publication_number`-non-equivalence defense
  survives in the Related Work single paragraph (L210–214, DAPFAM clause) and §3.
- **C2** (robustness-metric family) — all 11 metric definitions kept in §4
  (verified 11 `\paragraph` blocks). XRC/RRC/ARI equations (Eq.1–3) and the
  degeneracy gate all present; CLIR-MRS (Eq.4) kept with the "table-ordering only"
  framing intact (L399, L408–411).
- **C3** (separability mechanism) — `fig:cp_sep` kept in body (L806), r=+0.96 in
  prose, sign-robustness + partial-r caveat intact.
- **C4** (deployment recommendation) — cost_frontier + rrc_budget kept in body;
  every deployment `\paragraph` beat present.
- **C5** (generation pipeline) — kept as §3 paragraph + App D note, correctly
  scoped as "supporting evidence."
- Every Results/Analysis subsection traces to a contribution; no orphan analysis.
- **Ref balance:** all 23 figure labels + 3 table labels each `\ref`'d ≥1× from
  the body (the round-6 open-risk #1 is cleared). No "??" risk.

## Terminology & notation inconsistencies

- **L∞ dual notation:** correct — `0.058 (5.84%)` introduced once (L553), bare
  `0.058` everywhere else; no stray `5.84%` standing in for the floor. ✓
- **Collapse factor:** `48.7×` in both prose (L746) and caption (L1093); no
  surviving `49×`. ✓
- **tau-bands:** `[0.385,0.43]` (L502) and `[0.33,0.435]` (L503/527/839)
  consistent; no stale `[0.39,0.43]`/`[0.33,0.44]`. ✓
- **CLIR-MRS/MRS, home advantage, mate-retrieval, language codes** — consistent
  throughout. ✓
- **One residual notation nit:** the *only* terminology issue introduced by the
  restructure is the Figure-vs-Appendix-Figure inconsistency in glue joint #4
  above. Pick one convention for relocated-figure refs and apply it uniformly.

## Abstract/Conclusion alignment issues

Aligned, no drift. The abstract (L47–80), the contribution list (L123–165), and
the conclusion (L1001–1020) make the same claims with the same emphasis and the
same numbers: collapse (0.50/+0.55), inconsistency (RBO 0.39/0.19), confusion
(14–78%), cost (XRC50 3.5×, RRC@100 0.74, L∞ 0.058), separability (r +0.96), the
embeddinggemma-capability-corner / bge-m3-cheaper-reader split, and the
budget rule (MT the query, human-translate the corpus). The "report robustness
next to recall; align don't re-rank" thesis closes all three. The industrial
"which model to deploy + measure honestly" thread is visible end-to-end
(Intro L86–117 → Metrics framing L303 → Results → Deployment §8 → Conclusion).
No academic-drift. ✓

## What's already cohesive (leave alone)

- The Related Work 7→1 collapse: one tight paragraph, 28 citations preserved,
  every positioning (CLEF-IP/DAPFAM family-gold rejection, alignment-not-
  translation confirm-not-discover, English-only `patentembeddings2026` ordering,
  the 25-year query-vs-doc translation question) survives as a clause. Do not
  touch it further — it is at the floor of compressibility without losing a cite.
- The Metrics section: 11 definitions kept, prose compressed to def + one-line
  why-it-matters as intended. Coherent and reads as a single contribution (C2).
- The two leaderboard tables and their captions (self-contained, source-traced).
- The Limitations section (freeze — does not count toward 6pp; correctly absorbs
  the small-n caveats and the per-route thinness).
- The float-relocation mechanics: each figure kept its caption + `\label`, only
  the `\begin{figure}` block moved, single body `\ref` preserved. Textbook.
