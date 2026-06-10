# Writer notes — round 2

One-voice revision pass over `paper/main.tex`. Snapshot at
`round_2/draft.tex`; figures in `round_2/figures_manifest.md`. Self-lint passed
(23/23 includegraphics exist; all `\cite` keys resolve in `custom.bib`; all
referenced `\label`s defined; `\begin`/`\end` balanced; **zero in-text
`\todo`** — both former `\todo`s moved to `% TODO` LaTeX comments). Every new
number traces to a verified `extra_*` file (table in figures_manifest).

## Sections touched
Abstract, Introduction (+contributions C1–C5), Related Work (rewritten/expanded),
Benchmarks, Metrics (+XRC/RRC defs), Experimental Setup, Results §6.1/§6.2/§6.3,
Analysis (restructured), Deployment, Limitations, Conclusion, Appendix,
`custom.bib` (+10 entries).

## Figures
**Added (5):** cp_fig15 (XRC), cp_fig16 (RRC), cp_fig17 (aggregation ribbon),
ag_fig11 (availability residual), ag_fig12 (joint failure modes).
**Dropped from the included set:** cp_fig08 and ag_fig4 — the home-adv↔RBO /
share↔RBO correlations they plot are now demoted to non-robust descriptive
observations (fragile on n=7), so they live in one hedged figureless sentence.

## Bib entries added (10)
clefip2013, clirmatrix2020, dapfam2025, whatdrivesclir2025 (2511.19324),
crosslingualcost2025 (2507.07543), salehpecina2020, oard1998,
bailey2017consistency, patentembeddings2026 (2605.24297), rankingrobustness2026
(2605.31142).

## Required edits this round — status
1. **B1 FIX (done).** "English is the easiest target" is gone. Replaced with the
   hub-and-spoke reading using `writer_replacement_sentence`: hardest edge
   en→de (0.12), most asymmetric de↔zh (+0.23), *no clean easiest target*
   (fr 0.375 ≈ en 0.367 > zh 0.350 > de 0.309), corpus-composition caveat
   (en 46% / zh 0.4%) folded in. Closes B1, T2, N3-directional in one paragraph.
2. **Novelty reframe (done).** C1 narrowed to "first *content-controlled,
   chemistry-ontology-grounded*"; Related Work now bounds explicitly vs CLEF-IP
   (prior-art gold, not parallel) and DAPFAM (family gold = the equivalence we
   reject), and confirms-not-discovers the alignment line of whatdrivesclir /
   crosslingualcost. All 4 mandatory cite clusters added inline (CLEF-IP,
   CLIRMatrix, DAPFAM, whatdrivesclir/crosslingualcost; plus Oard+Saleh&Pecina,
   Bailey, patentembeddings, rankingrobustness, MMTEB-Borda).
3. **CLIR-MRS demoted (done).** §Metrics renames it "table-ordering convenience
   only"; §6.3 leads with per-axis dominance and adds the aggregation-ribbon
   caveat (rank range [1,4], winner-take-all contamination annotated) with
   cp_fig17. Deployment now recommends egemma on per-axis dominance, not the
   composite.
4. **XRC + RRC headline (done).** New first-class metric defs (Eq. xrc/rrc);
   §6.1 cost beat (XRC50 3.5×, censoring discipline: median headline, D90/D95
   lower bounds only) + RRC beat (RRC@100 0.7445, RRC@1000 0.9416, lost 5.84%).
   Threaded into abstract, intro, analysis crux, deployment, conclusion.
5. **Correlations softened (done).** auc_cross~clir kept as the *single* robust
   mechanism (+0.958 on n=7 stated); over-rep~clir and home-adv~rbo demoted to
   descriptive-on-n=9, explicitly "do not survive dropping the collapsers,"
   never in abstract/conclusion as mechanisms.
6. **Two-benchmark split + RBO harmonized (done).** Analysis "availability"
   paragraph split into an alias-graph sentence (own/foreign 0.63–0.82 / 0.35–
   0.47; 42% vs 8–10%; slope −0.57 descriptive n=5) + a cross-lingual sentence
   (49×, 60%), each with its own benchmark label/source. Led by the A6 joint-
   failure thesis (44.4% same-language sibling). Abstract now names both RBO
   ceilings (0.39 alias / 0.19 cross-lingual); intro/conclusion say "the best
   RBO any model achieves" (B2 fixed). availability headline sharpened from
   "mostly availability" to "availability sets the stage; residual encoder bias
   remains."

## Critic points — addressed vs deferred

### Novelty (Reviewer #1)
- C1 CLEF-IP omission, narrow "first", DAPFAM-as-rejected-design: **fixed**.
- C2 directional matrix = CLIRMatrix; RBO = Bailey lineage; AUC = standard,
  claim only same-vs-cross + re-ranker corollary: **fixed inline**.
- C2 CLIR-MRS highest risk → demote to convenience + per-axis dominance +
  aggregation-ribbon range: **fixed** (this was the #1 ranked over-claim).
- C3 "first decomposition" → "confirm on content-controlled corpus" + cites:
  **fixed**.
- C4 QT-vs-DT (Oard, Saleh&Pecina) + patentembeddings overlap: **fixed**.
- The dreamer's "CERC / reading-cost multiplier" upgrade (route 3) is realized
  as XRC; the conformal-guarantee version is flagged as horizon only (honest).

### Correctness (Reviewer #2)
- **B1 (MISMATCH):** fixed (see above).
- **B2 (attribution):** intro + conclusion now "best RBO any model achieves";
  abstract already fine.
- **B3 (Fig 1 MoLIR population):** added a footnote on the collapse paragraph.
- **T1 (home-adv↔availability):** §6.1 hedge clause + Analysis reframe; the
  negative-slope result *strengthens* this rather than papering over it.
- **T2 (corpus composition):** folded into the anisotropy paragraph + caption.
- **T5 (small-n correlations):** all reported "across the nine models"; the two
  fragile ones demoted, the robust one annotated with the n=7 robustness.
- **T6 (stale baseline run):** no number from `20260601-235117_137questions/`
  used; confirmed by trace.

### Cohesion (Reviewer #3)
- #1 two-benchmark fusion paragraph: **split** into two labelled sentences, two
  footnotes, led by the A6 thesis.
- #2 abstract single RBO ceiling: **harmonized** to name both.
- #3 Related Work → Benchmarks transition: future-work disclaimer **moved** into
  the calibration paragraph; §2 now ends with a forward bridge into Benchmarks.
- #4 in-text `\todo`s: **both demoted** to clean sentences + `% TODO` comments.
- #5 cp_fig11 caption over-claim: **rewritten** to describe the bars, with +0.96
  (and the n=7 robustness) as a clearly-labelled text statistic; radars got the
  interpretive clause ("leads on consistency and separability, not raw recall").
- #6 teaser vs leaderboard reorder: **signpost sentence** added near Table 1.
- C5 orphan promise: **softened** to "supporting evidence … human validation
  summarized in the system description."
- "~12% universal-blind" orphan: now **earned** in Analysis (14/16 structure)
  before its use in Deployment.

## Open `\todo{trace:...}` items
**None in-text.** Two items remain as `% TODO` LaTeX comments (invisible to the
reader), exactly as the correctness critic permits (not blocking, not yet under
reports/):
- corpus dedup count 14,401 / GP-EPO-JRC coverage matrix (Benchmarks §3).
- human-eval numbers (8.33/10, 97/100, +4.3pp auto-grader strictness) (C5,
  Benchmarks + Appendix).
- The MMTEB/MIRACL/NeuCLIR domain-transfer number is likewise a `% TODO`
  comment in Limitations (backlogged, not a fabricated number).

## PENDING-EVAL items mentioned honestly (not flagged as missing)
Conformal XRC (Related Work + Limitations), CLIR-MRS external validation
(Limitations), graded ChEBI hop-distance severity law (Results footnote +
Limitations), claim-equivalence spot-check (Limitations). All scoped as future
work, none claimed as results.
