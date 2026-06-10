# Writer notes — round 1 (first full draft)

## Status
Wrote the first complete LaTeX draft of `paper/main.tex` following the story's
section map and C1–C5 contributions. No previous-round critics/reporter exist,
so this round is a clean build from `story.md` + the two `EXECUTIVE_SUMMARY.md`
files + the headline CSV/JSON. Self-lint passed: braces 455/455, all environments
paired, 20/20 `\includegraphics` targets exist, 22/22 `\cite` keys defined.

## Sections written (all 10 + abstract + appendix)
- **Abstract** — deployment problem, two benchmarks + metric family, headline
  numbers (CLIR@10 0.50, home adv +0.55, RBO 0.39, confusion 14–78%,
  r(AUC,CLIR@10)=+0.96, CLIR-MRS 0.71), alignment-not-re-ranking payoff.
- **1 Introduction** — industrial framing, Spanish no-home hook, the two hidden
  failures (inconsistency / confusion), numbered C1–C5, one-line spoiler.
  Teaser = Figure 1 (`cp_fig01_clir_leaderboard.png`).
- **2 Related Work** — six defended boundaries (multilingual/CLIR benchmarks &
  MTEB averaging; cross-lingual RAG content-vs-language confound; calibration/
  conformal/fairness-OT vs ranking robustness; patent-family non-equivalence;
  chemistry/entity models; when-to-translate). Calibration-line metrics framed
  as future work only.
- **3 Benchmarks (C1)** — shared 23,487-doc corpus; alias-graph (132 Q, 24
  compounds, two lenses); cross-lingual (137 Q, 57 original + 80 synthetic,
  Spanish pure query-side); honesty design (no MT docs, no publication_number
  gold); C5 pipeline pointer.
- **4 Metrics (C2)** — CLIR@k/MoLIR + home advantage, directional CLIR, mate-
  retrieval, cross-lingual RBO, language collapse, separability AUC, and the
  CLIR-MRS equation (Eq. 1) + MRS definition.
- **5 Experimental Setup** — 9 models, shared haystack, both lenses / orig-synth
  split, reproduce commands.
- **6 Results** — chem-patents collapse/anisotropy/MT-null/mate beats; alias RBO
  & confusion beats; two leaderboard TABLES (tab:cp_board, tab:ag_board) + two
  radars.
- **7 Analysis (C3)** — availability confound (42% vs 8–10%; 49×; 60%); bias⇒
  inconsistency (r=−0.85 / −0.87); structure-question trap (0.26/51% vs
  0.60/25%); separability deficit (AUC 0.55 vs 0.70; r=+0.96) ⇒ re-ranker
  can't fix.
- **8 Deployment Recommendation (C4)** — deploy embeddinggemma; report
  robustness next to recall; don't naively ensemble (oracle 0.61 / 88% vs RRF
  loses — stated together per narrative risk #6); align not re-rank; budget rule.
- **Limitations** — unnumbered `\section*` per ACL; scale, domain transfer,
  judge dependence, language coverage, out-of-scope.
- **Conclusion** — 4-sentence thesis restatement.
- **Appendix** — reproducibility + C5 validation pointer.

## Figures included (20) + 2 tables
12 chem-patents (`cp_`) + 8 alias-graph (`ag_`); see `figures_manifest.md`.
Leaderboards rendered as booktabs TABLES (numbers from headline CSVs) rather
than as the bar-chart figures, which keeps them precise and citable. The two
leaderboard bar charts and the direction-asymmetry / alias-home figures were
copied to `paper/figures/` but left un-referenced for later rounds.

## Bib entries added to custom.bib (22)
miracl2023, mmteb2025, muennighoff2023mteb, mteb_contamination2025, neuclir2023,
bordirlines2024, xrag2025, nepotism2025, traq2023, conflare2024,
conformalrag2025, fairnessot2023, buyl2022optimal, prime2002, sapbert2021,
paecter2024, kishida2008, labse2022, bgem3_2024, chebi2016, rbo2010, rrf2009.
Several use placeholder author lists / paraphrased titles (xrag2025,
nepotism2025, conformalrag2025, conflare2024 have `Anonymous`/`others`) — FLAGGED
for the fact/novelty critic to verify exact metadata against the arXiv IDs the
story supplied.

## Narrative risks (from story) handled
1. **System-PDF numbers not in reports/** — took option (a): restricted ALL
   load-bearing numbers to the two key_findings files. Corpus dedup count
   (14,401), GP/EPO/JRC matrix, and human-eval numbers (8.33/10, +4.3pp) are
   left as `\todo{trace:...}` rather than printed. See open todos below.
2. **Synthetic-benchmark credibility** — C5 pipeline + human validation kept
   visible (Benchmarks §, Appendix), not buried, but its numbers are todo'd.
3. **General-domain transfer gap** — framed chemistry specificity as the
   contribution; flagged the missing transfer number as a Limitation + todo.
4. **Metric-novelty overlap with calibration line** — Related Work beat 3 draws
   the ranking-vs-calibration boundary in its first sentences.
5. **Two RBO ceilings (0.39 alias / 0.19 chem-patents)** — cited each to its own
   benchmark, never averaged or conflated (abstract, intro, results all keep
   them separate).
6. **"Don't ensemble" vs "oracle headroom real"** — stated together in one
   paragraph (Deployment) so it does not read as contradictory.
7. **MT result is a null (p=0.13)** — reported as "no significant penalty,"
   explicitly "we do not claim it helps."

## Open `\todo{trace:...}` items (3)
1. **Corpus dedup count 14,401 + GP/EPO/JRC coverage matrix** (Benchmarks §) —
   in source system slides, not under reports/. Either dump to reports/ or keep
   as descriptive-only.
2. **Human-eval numbers** (Benchmarks § + Appendix) — mean 8.33/10, 97/100
   reviewed, 0 rejected, auto-grader +4.3pp stricter — source slides only.
3. **General-domain transfer number** (Limitations) — no MMTEB/MIRACL/NeuCLIR
   companion computed; candidate for `needs_eval.md`.

## Suggested follow-ups for the implementer / next round
- Dump the three todo'd quantities to `reports/` so C5 and the corpus paragraph
  can cite a file instead of a `\todo`.
- Verify the placeholder bib metadata (4 entries with Anonymous/others authors).
- Decide whether a general-domain transfer eval goes into `needs_eval.md`.
