# Story (round 6) — CUT-TO-6-PAGES RESTRUCTURE BLUEPRINT

> **This round is NOT a story round.** The science is frozen and submission-ready
> (round-5 reporter: 0 compute, 0 new results, "wind the loop down to polish").
> The thesis, contributions, section spine, and every number are unchanged. The
> SOLE job is **length compliance** for the EMNLP Industry Track CFP +
> the small carried text-cleanup. **NO new claims, NO new analyses, NO new
> figures, NO bib edits** (the bibliography was fixed this session — the writer
> must NOT touch `custom.bib`).

## The constraint that drives this round (verified against the CFP)
- **6 content pages** (7 camera-ready). **References, the `Limitations` section,
  and the `Appendix` do NOT count.**
- Current body: **23 in-body figures + 2 in-body leaderboard tables + ~1200
  lines of dense prose** → compiles to ~14–18 pages, ~3× over.
- Therefore the restructure is mechanical and aggressive: **keep ≤7 floats in the
  6-page body, push the other ~16 figures + the robustness table into the free
  Appendix, and trim prose** (Related Work 7¶→1¶; Metrics paragraph-defs
  compressed; Results/Analysis keep only load-bearing findings). Every moved
  float is still `\ref`'d exactly once from the body so no beat is orphaned.

## Changes since round 5
Round 5 closed the 8-page float overload (29→25 floats, two radars cut, two
stitched panels, appendix `tab:robust` added). That budget is now obsolete: the
real target is **6 pages**, not 8. So round 6 supersedes the round-5 float plan
with a far deeper cut. Round 5 also handed forward a 5-item writer-only cleanup
list (W-M1 tau-bands, W-N2 49x→48.7x, W-N3 L∞ dual notation, W-N1 float order,
W-Guard no-edit note); we **carry W-M1/W-N2/W-N3** into this round (anchors
below). W-N1 (float order) is moot once figures relocate to the appendix.

---

## PART 1 — THE 6-PAGE BODY (what stays, at what length)

The body keeps the full spine but at compressed length. Target rough budget
(6 pages, two-column ACL): Abstract+Intro ~1.25pp · Related Work ~0.3pp ·
Benchmarks ~0.6pp · Metrics ~0.9pp · Setup ~0.2pp · Results ~1.3pp · Analysis
~0.8pp · Deployment ~0.6pp · Conclusion ~0.2pp. Floats eat ~2 of the 6 pages,
so prose must be tight.

### Abstract — **FREEZE (keep verbatim).**
It is already a single dense paragraph and is the strongest compression of the
whole paper. Do not touch except the carried-number cleanup does not affect it.

### Introduction — **KEEP, light trim only.**
Keep the four motivating paragraphs and the **full C1–C5 contributions list**
(the contributions list is load-bearing for an industry submission and must stay
in the body). Trim: the third paragraph ("And both are costly…") can lose its
parenthetical forward-refs. Keep the teaser figure (the only body figure here).
- BODY FIGURE: **Fig 1 / `cp_fig01` (teaser)** — stays. It is the one-glance
  proof of the thesis (overall R@10 sits between easy MoLIR and hard CLIR).

### Related Work — **HARD CUT: 7 paragraphs → 1 tight paragraph.**
This is the single biggest prose win. Collapse the seven `\paragraph` blocks
(multilingual/CLIR benchmarks; cross-lingual RAG; alignment-not-translation;
calibration/conformal; patent-IR/family-non-equivalence; chemistry-IR;
when-to-translate) into **one ~10–12 line paragraph** that keeps every citation
but states each positioning in one clause. Required content to preserve in the
single paragraph (each as a clause, citations intact):
1. Multilingual/CLIR leaderboards report a single averaged score
   (MIRACL/MMTEB/NeuCLIR/MTEB, CLIRMatrix for the directional style); ours is a
   ranking-level robustness suite reported co-equally with recall.
2. Cross-lingual RAG / language-preference work
   (BordIRlines/XRAG/Nepotism) confounds content with language; our parallel
   human-translated patents isolate the language effect (the C1 defense).
3. Alignment-not-translation line: `whatdrivesclir2025` and `crosslingualcost2025`
   reach a close thesis on general benchmarks; **we confirm it on a
   content-controlled patent corpus, we do not claim to discover it.**
   **PROSE NOTE (carry from conductor): `crosslingualcost2025` is
   Arabic–English specific — cite it as prior art on the same-language head
   start, do NOT imply it covers our five languages.** (One added qualifier
   like "(Arabic–English)" at the cite, nothing more.)
4. Patent IR: CLEF-IP is the multilingual-patent precedent but uses prior-art
   gold, not parallel translation, and is not chemistry-grounded; DAPFAM uses
   family-level gold — the design we reject. Keep the
   `publication_number`-non-equivalence justification in ONE clause.
5. Chemistry IR: SapBERT/PaECTER are monolingual; `patentembeddings2026` ranks
   the same model family on **English-only** patents (ordering differs).
6. When-to-translate: a 25-year CLIR question (oard1998/salehpecina2020); our
   budget rule re-derives it for embedding retrieval. RBO-as-query-variant cites
   `bailey2017consistency`.
- **DROP from the body** (move nothing — just delete prose, citations stay in the
  one paragraph): the calibration/conformal paragraph's long hedge about a
  "finite-sample conformal XRC horizon" (keep at most a half-sentence: conformal
  IR is complementary, scores calibration not ranking robustness). The
  `rankingrobustness2026` / `mteb_contamination2025` asides fold into clause 1.
- The closing "Having positioned our four contributions…" bridge sentence → cut
  to one short clause or delete; the transition to Benchmarks is already implied.

### Benchmarks — **KEEP, modest trim.**
Keep all five `\paragraph` blocks (shared corpus, alias-graph, CLIR, honesty-by-
design, generation-pipeline-C5) — they define the two released artifacts (C1)
and are core to an industry submission. Trim: the two `% TODO trace` comment
blocks can stay as comments (invisible). Compress the C5 paragraph to ~3 lines
(the validation numbers already live only in the system description). Keep the
two footnotes that anchor sizes (132/24, 137/57/80, Spanish no-home).

### Metrics — **KEEP all definitions but COMPRESS the `\paragraph` prose.**
This section is a contribution (C2) so the metric **definitions** stay, but each
`\paragraph` is currently a mini-essay. Compress to **definition + one-line
why-it-matters**, drop the discursive justification:
- CLIR@k/MoLIR@k/home advantage — keep (3 lines).
- Directional CLIR & asymmetry — keep, drop the "hub-and-spoke graph" elaboration
  to one clause.
- Mate-retrieval — keep (2 lines).
- Cross-lingual RBO — keep, drop the repeated `bailey2017consistency` lineage
  sentence (already in Related Work now).
- Language collapse — keep (2 lines).
- Separability AUC — keep (2 lines).
- **XRC** (Eq. 1) — keep equation + the headline rule (XRC50 finite, D90/D95
  right-censored lower bounds). Drop the long "monotone-invariant vs AUC"
  justification to a single clause.
- **RRC** (Eq. 2) — keep equation + knee K* + floor L∞ definition. Trim the
  multi-stage-cascade citation sentence to a clause.
- **ARI** (Eq. 3) — keep equation + the "three shares sum to one" sentence. Drop
  the `residualrerank2026` inversion justification to one clause.
- **Degeneracy gate** — keep the one-line criterion (CLIR@10<0.10 flags gte-base
  + e5-large-instruct). **MOVE its figure (`cp_fig20`) to the Appendix**; keep
  the two-criterion footnote OR move it to the appendix (writer's call —
  prefer keeping it as a short footnote).
- **CLIR-MRS/MRS** — keep Eq. 4, one sentence ("table-ordering convenience
  only, not a contribution"). Drop the long footnoted definitions to the
  appendix-or-footnote.

### Experimental Setup — **KEEP, trim Reproducibility.**
Keep Models (the nine) and Protocol. **Compress the Reproducibility paragraph**:
it currently lists ~10 script names inline. Cut to one sentence ("all figures and
numbers regenerate from `run_all.py` and the `extra_*.py` scripts under the two
run families") — the full script list moves to the Appendix Reproducibility note
(already partly there).

### Results — **KEEP load-bearing findings; cut figures hard.**
Keep the prose beats, but most figures relocate. Findings that stay in body prose:
- §6.1 Collapse: teaser (Fig 1) carries it; keep "best CLIR@10 0.50, home
  advantage +0.55" prose. **MOVE `cp_fig02` (home advantage) → Appendix.**
- §6.1 Directional/hub: keep the 2–3 sentence result (en→de hardest 0.12,
  de↔zh most asymmetric +0.23, no single easiest target, asymmetry tracks corpus
  composition). **MOVE `cp_fig03` (directional matrix) → Appendix.**
- §6.1 Cost frontier: **KEEP `cp_fig18` in body** (it is the C4 spine). Keep the
  Pareto-corner prose + the tau-band sentence (with the W-M1 raw bands).
- §6.1 MT-is-safe: keep the one-sentence null (−0.044, p=0.13). **MOVE
  `cp_fig05` (MT penalty) → Appendix.**
- §6.1 RRC/mate/ARI: **KEEP `cp_fig19` (RRC budget) in body** (C2/C4 spine: knee
  K*=5, RRC@100=0.74, L∞=0.058). **MOVE `cp_fig22` (ARI decomposition) and
  `cp_fig06_07_mate` (mate) → Appendix**; keep their numbers in body prose.
- §6.2 two-tax: keep the one-sentence non-redundancy result (ρ=−0.59, n.s.).
  **MOVE `cp_fig21` (two-tax) → Appendix.**
- §6.2 RBO: keep "best cross-lingual RBO 0.39, Chinese odd-one-out" prose.
  **MOVE `ag_fig1` (RBO) → Appendix.**
- §6.2 confusion: **KEEP one alias confusion figure `ag_fig2` in body**
  (the 14–78% confusion result is a core C3 finding and the chemistry hook).
  **MOVE `ag_fig5` (universal attractors) → Appendix**; keep the attractor list
  (polypeptide/methyl/…) as one inline clause.
- §6.3 leaderboards: **KEEP both tables `tab:cp_board` + `tab:ag_board` in body**
  (the two-leaderboard payload is the deliverable). Keep the aggregation-
  sensitivity paragraph prose (rank ranges [1,4]). **MOVE `cp_fig17`
  (aggregation ribbon) → Appendix.**

### Analysis — **KEEP findings; relocate all five figures.**
Keep the prose mechanism story (it is C3, load-bearing) but move every figure:
- Modal failure = same-language sibling (44.4%): keep prose. **MOVE `ag_fig12`
  (joint failure) → Appendix.**
- Availability-vs-residual-bias (slope −0.57; collapse 48.7×, noise 60%): keep
  prose. **MOVE `ag_fig11` (availability residual) + `cp_fig09_10_collapse`
  → Appendix.**
- Structure-questions-are-the-trap (R@10 0.26 vs role 0.60; 14/16 blind core):
  keep prose. **MOVE `ag_fig6` (question type) → Appendix.**
- Bias↔inconsistency descriptive (r=−0.85/−0.87, not robust to dropping
  degenerates): keep the one short paragraph.
- **Separability deficit (the load-bearing mechanism, r=+0.96 sign-robust):**
  **KEEP `cp_fig11` (separability) in body.** **MOVE `ag_fig8` (confusion-is-
  separability) → Appendix**; keep the AUC 0.55-vs-0.70 number inline.

### Deployment Recommendation — **KEEP prose; relocate figures.**
This is C4, the industry payoff — keep all `\paragraph` beats but move figures:
- Deploy embeddinggemma (capability corner) — keep, refs `cp_fig18` (body).
- Per-route router is upside headroom — keep the prose (3/5 routes, corner moves,
  flips on en/fr). **MOVE `cp_fig23` (per-route frontier) → Appendix.** This
  paragraph can shrink ~30%; the thin-sample caveat duplicates the Limitations
  per-route paragraph, so keep it terse here.
- Report-robustness-next-to-recall — keep (3 lines).
- Do-not-reflexively-ensemble — keep prose (oracle 0.61 / 88% vs 76%, RRF loses).
  **MOVE `cp_fig12` (CLIR ensemble) + `ag_fig7` (alias ensemble) → Appendix.**
- Budget-the-re-ranker-by-the-knee — keep, refs `cp_fig19` (body).
- Align-don't-re-rank — keep.
- Budget rule (MT-query/human-corpus) — keep.

### Limitations — **FREEZE (does NOT count toward 6 pages).**
No length pressure. Keep as-is. It already absorbs the per-route thinness, the
small-n caveats, and the forthcoming alignment-causal-probe.

### Conclusion — **KEEP, light trim.**
Keep the single dense paragraph; it can lose one clause but is fine.

---

## PART 2 — BODY FLOATS (the ≤7 that stay) — rationale

| # | float | label | why it earns a body slot |
|---|-------|-------|--------------------------|
| 1 | `cp_fig01` teaser | `fig:teaser` | one-glance proof of the thesis (collapse vs average) |
| 2 | `cp_fig18` cost frontier | `fig:cost_frontier` | the C4 deployment spine (Pareto capability corner, tau-band) |
| 3 | `cp_fig19` RRC budget | `fig:rrc_budget` | the C2/C4 "align-don't-re-rank" budget curve (knee K*, floor L∞) |
| 4 | `cp_fig11` separability | `fig:cp_sep` | the C3 load-bearing mechanism (r=+0.96, sign-robust) |
| 5 | `ag_fig2` confusion both lenses | `fig:ag_conf` | the chemistry confusability trap (14–78%), the only alias hook in body |
| T1 | leaderboard table CP | `tab:cp_board` | deliverable: the cross-lingual benchmark ranking + per-axis numbers |
| T2 | leaderboard table AG | `tab:ag_board` | deliverable: the alias-graph benchmark ranking + per-axis numbers |

That is **5 figures + 2 tables = 7 body floats**. If LaTeX still overflows 6
pages after the prose trim, the **first further cut is `ag_fig2`** (its 14–78%
number survives in prose and Table 2 carries the alias per-axis numbers),
leaving 4 figures + 2 tables. Do not cut below the teaser, cost frontier, RRC
budget, separability, and the two tables — those five carry C1–C4.

## PART 3 — APPENDIX (free) — every relocated float, grouped, each `\ref`'d once from body

Create grouped appendix subsections. Each figure keeps its existing caption and
`\label`; only the `\begin{figure}` block moves. The single body `\ref` already
exists for every one (verified) — keep exactly one body reference each so no
float is orphaned and none triggers an "undefined reference."

**App A — Extended cross-lingual results** (referenced from §6.1 / §7):
- `cp_fig02_home_advantage` (`fig:cp_home`)
- `cp_fig03_directional_clir_matrix` (`fig:cp_dir`)
- `cp_fig05_mt_penalty` (`fig:cp_mt`)
- `cp_fig06_07_mate` (`fig:cp_mate`)
- `cp_fig22_ari_decomposition` (`fig:ari`)
- `cp_fig09_10_collapse` (`fig:cp_collapse`)
- `cp_fig20_degeneracy_gap` (`fig:cp_deg`)

**App B — Extended alias-graph results** (referenced from §6.2 / §7):
- `ag_fig1_cross_lingual_rbo` (`fig:ag_rbo`)
- `ag_fig5_universal_attractors` (`fig:ag_attr`)
- `ag_fig12_joint_failure_modes` (`fig:ag_joint`)
- `ag_fig11_availability_residual` (`fig:ag_avail`)
- `ag_fig6_question_type_effect` (`fig:ag_qtype`)
- `ag_fig8_confusion_is_separability` (`fig:ag_sep`)

**App C — Aggregation, routing, and ensemble** (referenced from §6.3 / §8):
- `cp_fig17_aggregation_ribbon` (`fig:cp_ribbon`)
- `cp_fig21_two_tax` (`fig:two_tax`)
- `cp_fig23_per_route_frontier` (`fig:per_route`)
- `cp_fig12_ensemble_headroom` (`fig:cp_ens`)
- `ag_fig7_ensemble_headroom` (`fig:ag_ens`)

**App D — Robustness ledger** (already present, keep):
- `tab:robust` (`app:robust`) — unchanged.

That is **18 relocated figures + the existing robustness table** in the appendix,
plus the existing Reproducibility / C5-validation notes. Total body floats: 7
(5 fig + 2 tab). Total appendix floats: 18 fig + 1 tab.

> **Writer mechanic:** the cleanest implementation is to leave each figure's
> `\ref` in body prose where it already is, and physically move the
> `\begin{figure}…\end{figure}` block into the appropriate appendix subsection.
> LaTeX resolves the cross-reference regardless of float location. Do a final
> `\ref`-balance check: 23 figure labels + 3 table labels, each referenced ≥1×.

---

## PART 4 — CARRIED CLEANUP (verified anchors; pure text edits, no science change)

1. **W-M1 (correctness — restore raw tau-bands).**
   - L632: `[0.39, 0.43]` → `[0.385, 0.43]`
   - L633: `[0.33, 0.44]` → `[0.33, 0.435]`
   - L657 (`cp_fig18` caption): `[0.33, 0.44]` → `[0.33, 0.435]`
   - L1085 (Deployment): `[0.33, 0.44]` → `[0.33, 0.435]`
   Verified raw: admitted-stable `[0.385,0.43]`, cheapest-bge `[0.33,0.435]`
   (`extra_cost_frontier/tau_sweep_summary.json`).

2. **W-N2 (harmonize 49× → 48.7×).**
   - L955 (prose) and L973 (`cp_fig09_10_collapse` caption): `49\times` →
     `48.7\times`. Source overrep = 48.71 (zh),
     `round06_language_collapse/summary.json` `most_collapsed_language.overrep`.

3. **W-N3 (standardize L∞ dual notation `0.058 (5.84%)`).**
   - Introduce both forms once at L692 (the RRC paragraph): `…the floor
     $L_\infty = 1-\mathrm{RRC@1000} = 0.058$ (5.84\%) of foreign twins…`.
   - Convert bare-percentage sites to `$0.058$`: L1030 (`L_\infty=5.84\%` →
     `L_\infty=0.058`) and L1191 (`L_\infty = 1-\mathrm{RRC@1000}=5.84\%` →
     `L_\infty = 1-\mathrm{RRC@1000}=0.058`).
   - Leave existing `$0.058$` sites unchanged. (0.058 ≡ 5.84% = 1−0.9416.)

4. **PROSE NOTE — `crosslingualcost2025` is Arabic–English specific.** In the
   collapsed Related Work paragraph, when citing `crosslingualcost2025` for the
   same-language head start / directional asymmetry, add a one-word qualifier
   (e.g. "(Arabic–English)") so it is not implied to cover our five languages.
   Cite it as prior art on the head start, not as our-coverage.

5. **W-Guard (no edit, checklist note).** Keep embeddinggemma's L∞=0.058 floor
   textually distinct from the now-tied ARI@100 (gap 0.004, CI straddles 0). Do
   not let the ARI@100 tie wording imply the L∞ floor is also tied.

6. **W-N1 (float order) — moot.** Once `cp_fig22` (ARI) moves to the appendix and
   `cp_fig19` (RRC budget) stays in body, the round-5 float-order concern
   dissolves. No action.

> **Do NOT touch `custom.bib`** — it was fixed this session (real authors/titles).

---

## Open narrative risks (for critics to watch)
1. **Orphaned floats / dangling refs.** After moving 18 figures, every body `\ref`
   must still resolve and every appendix figure must keep exactly one body
   reference. Risk: a relocated figure whose only `\ref` was in prose the writer
   also trimmed. Critic check: 26 labels, each `\ref`'d ≥1×, no "??" in build.
2. **Over-trimming a load-bearing number.** Compressing Results/Analysis prose
   must NOT drop any headline number (CLIR@10 0.50, home +0.55, RBO 0.39/0.19,
   confusion 14–78%, XRC50 3.5×, RRC@100 0.74, L∞ 0.058, r +0.96, MT −0.044).
   These must survive in prose even when their figure leaves the body.
3. **Related Work over-compression dropping a positioning.** The 7→1 collapse
   must keep every citation and the novelty positioning vs CLEF-IP/DAPFAM and the
   alignment-not-translation line (the novelty critic will check C1/C3 defenses
   survive).
4. **6-page overflow after the cut.** If still over, the named first-further-cut
   is `ag_fig2`; do not cut the five spine floats or the two tables.
5. **Accidental science drift.** This is length-only + the carried cleanup. Any
   change to a claim, a number, or an analysis is out of scope this round.
