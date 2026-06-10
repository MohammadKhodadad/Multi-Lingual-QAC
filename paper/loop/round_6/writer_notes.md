# Writer notes — round 6 (cut-to-6-pages restructure)

**Scope:** pure length compliance for the EMNLP Industry Track 6-page limit +
the carried R5 text cleanup. NO new claims, NO new analyses, NO new figures,
`custom.bib` NOT touched. Started fresh from the untouched round-5 `main.tex`
(23 body figures + 3 tables).

---

## R6 CORRECTIVE PASS (appended) — driven by the round-6 cohesion critic

The cohesion critic verified the restructure read as one story but flagged the
body as right at the 6-page boundary with zero margin, and named two cheap fixes.
Applied EXACTLY these two, nothing else (no new claims, no number changes, no
`custom.bib` edits):

1. **Moved `ag_fig2` (`fig:ag_conf`, the alias-graph confusion figure) from the
   body into Appendix B.** The body keeps its 14–78% confusion number in prose
   (§6.2) and in `tab:ag_board`'s `conf` column; the body sentence that cited it
   now points to the appendix figure: "...worst for German and Chinese
   (Appendix Fig.~\ref{fig:ag_conf})." This drops the body to **4 figures + 2
   tables = 6 body floats**. The figure block (caption + `\label`, unchanged) was
   placed at the top of App B (`app:ag_extra`), so `fig:ag_conf` now sits after
   `\appendix` (lines 1107–1112).

2. **Appendix-reference labeling pass.** Every body sentence that referenced a
   figure now living in the appendix had its `Figure~\ref{...}` prefixed with
   "Appendix" → `Appendix Fig.~\ref{...}`, so the reader knows the float is not
   nearby. **23 such relabels applied** (the cohesion critic estimated ~15; there
   were more body→appendix references than estimated). Sites span 18 distinct
   appendix figures (cp_mate, cp_ribbon, per_route ×2 each; ari ×3). The 4
   body-resident figures (`fig:teaser`, `fig:cost_frontier`, `fig:rrc_budget`,
   `fig:cp_sep`) and the 2 body tables keep plain `Figure~`/`Table~` and were NOT
   relabeled. Counting the moved ag_conf ref, the body now has 24 `Appendix Fig.`
   sites total.

**Cohesion critic glue-joints addressed:** joint #2 (cp_deg in-line evidence ref
now `Appendix Fig.`), joint #4 (the mixed-citation seam — the whole point of the
labeling pass, now uniform), and the optional Limitations `fig:ari`/`fig:per_route`
polish (both relabeled). Joint #1/#3 (dead appendix section `\label`s) left as-is:
they produce no warning and the instruction scoped this pass to exactly the two
fixes above.

**Self-lint after corrective pass (no compiler):** braces 899=899; figure 23/23,
table 3/3, tabular 3/3, equation 4/4, itemize/abstract/document all matched; all
23 `\includegraphics` targets present on disk; 26 labels each `\ref`'d ≥1× (all
19 appendix figures retain ≥1 body ref — verified label-by-label, no orphans); no
duplicate labels; no dangling refs; 36 `\cite` keys all resolve in custom.bib
(untouched). `\appendix` appears exactly once; `ag_fig2`/`fig:ag_conf` is after it.

**Body float count now: 4 figures + 2 tables = 6 floats.**

**6-page fit estimate now:** The cohesion critic's pre-correction estimate was
~6.0–6.3 pages (over in the unlucky case). Dropping one full-width single-column
figure reclaims ~0.22–0.27 page of two-column float area, putting the body at
**~5.8–6.05 pages — fits 6 pages in the expected case, with a thin (~0.2 page)
margin in the unlucky case.** The labeling pass is text-neutral (adds the word
"Appendix" to 23 inline refs ≈ <0.02 page, negligible). If a compile still shows
a sliver over, the critic's flagged Cut 2 (prose-only: trim the per-route
paragraph that duplicates Limitations, and compress the repeated tau-band in the
Deploy paragraph) buys another ~0.1–0.15 page with zero information loss — but
that was explicitly OUT of scope for this corrective pass and was not applied.

---

## (Below: original round-6 first-pass notes — unchanged)

## Sections touched
- **Related Work:** 7 `\paragraph` blocks → **1 tight paragraph**. Every one of
  the 28 citation keys preserved. Kept positioning vs CLEF-IP/DAPFAM (family-gold
  rejected; `publication_number` unsafe), the alignment-not-translation line
  (confirm-not-discover), the C1 content-control defense, ChEBI grounding, and the
  QT-vs-DT lineage. Conformal/calibration hedge compressed to one clause.
- **Metrics:** each `\paragraph` compressed to definition + one-line why. Kept all
  three equations (XRC Eq.1, RRC Eq.2, ARI Eq.3) and CLIR-MRS Eq.4. Degeneracy-gate
  figure (`cp_fig20`) moved to App A; the two-criterion footnote trimmed and kept.
- **Setup:** Reproducibility ¶ cut from a ~10-script inline list to one sentence;
  full script list moved to the appendix Reproducibility note (`app:repro`),
  referenced from Setup.
- **Results / Analysis / Deployment:** prose beats and every load-bearing number
  retained, wording tightened; 18 figure blocks physically relocated to the
  appendix (each keeps its single body `\ref`). Per-route ¶ left intact (it
  already carried the thin-sample caveat).
- **Limitations / Conclusion:** frozen (Limitations does not count toward 6 pages;
  Conclusion left as the single dense paragraph).
- **Appendix:** restructured into four grouped sections —
  A (cross-lingual extras, 7 figs), B (alias-graph extras, 6 figs),
  C (aggregation/routing/ensemble, 5 figs), D (robustness ledger `tab:robust` +
  Reproducibility + C5 validation note, all carried unchanged).

## Float budget (first pass — SUPERSEDED by the R6 corrective pass above)
- **Body: 7 floats** = 5 figures (`cp_fig01` teaser, `cp_fig18` cost frontier,
  `cp_fig19` RRC budget, `ag_fig2` confusion, `cp_fig11` separability) + 2 tables
  (`tab:cp_board`, `tab:ag_board`).
  → After the corrective pass: **6 floats** (`ag_fig2` moved to App B) =
  4 figures + 2 tables.
- **Appendix: 19 floats** (first pass) → **20 floats** after the corrective pass
  = 19 relocated figures + `tab:robust`.

## Carried R5 cleanup applied
- **W-M1** raw tau-bands restored: `[0.385,0.43]` (admitted-stable) and
  `[0.33,0.435]` (cheapest-bge) at all sites (cost-frontier prose, `cp_fig18`
  caption, Deployment). No stale `[0.39,0.43]`/`[0.33,0.44]` remain.
- **W-N2** `49×` → `48.7×` (Analysis prose + the relocated `cp_fig09_10_collapse`
  caption).
- **W-N3** L∞ dual notation introduced once as `0.058 (5.84\%)` at the RRC
  paragraph; the two bare-percentage sites (Analysis, Deployment) converted to
  `0.058`. Existing `$0.058$` sites unchanged.
- **PROSE NOTE** `crosslingualcost2025` cited with `(Arabic--English)` qualifier
  as prior art on the same-language head start — not implied to cover our 5 langs.
- **W-Guard** embeddinggemma's L∞=0.058 floor kept textually distinct from the
  tied ARI@100 (gap 0.004, CI straddles 0); the tie wording does not touch the floor.
- **W-N1** (float order) moot after relocation — no action.

## Self-lint (no LaTeX compiler installed)
- 23 unique `\includegraphics`, all present on disk, no duplicate includes.
- 26 float labels each referenced ≥1×; no unreferenced labels, no dangling refs,
  no duplicate labels.
- Braces balanced; all environments matched.
- 36 `\cite` keys all resolve in custom.bib.

## Length estimate (cannot compile) — see the corrective-pass estimate above for the current figure
- Body prose ~8065 words (down ~1007, ≈11%, from round-5 ~9072), concentrated in
  Related Work and Metrics. (Unchanged by the corrective pass — labeling adds only
  the word "Appendix" to 23 inline refs.)
- Body floats cut 25 → 7 (first pass) → **6** (corrective pass: `ag_fig2` moved),
  reclaiming the dominant length win plus a final ~0.22–0.27 page.
- **First-pass estimate was ~6.5–7.5 pages; the cohesion critic re-estimated
  ~6.0–6.3.** After the corrective pass (the critic's named first-further cut),
  the current estimate is **~5.8–6.05 pages — expected to fit 6 pages**, thin
  margin in the unlucky case. The flagged prose-only Cut 2 (per-route/Deploy
  trim) remains available if needed but was out of scope here. Do NOT cut below
  the **four spine figures** (teaser, cost frontier, RRC budget, separability) +
  2 tables.

## Open `\todo{trace:...}` items (unchanged this round — comments only, not in body)
- Corpus dedup count / GP-EPO-JRC coverage matrix (Benchmarks comment).
- C5 human-eval numbers (Benchmarks comment + appendix C5 note).
- MMTEB/MIRACL/NeuCLIR transfer measurement (Limitations comment).
None of these block submission; all are invisible LaTeX comments.
