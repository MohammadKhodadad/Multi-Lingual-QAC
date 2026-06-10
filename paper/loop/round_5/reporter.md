# Reporter handoff (round 5) -> feeds story + writer (round 6)

## TL;DR
The paper is ship-ready. Round 5 was an honesty-softening + figure-trim round, and this
sub-round **froze compute** (0 compute items, 0 API calls, no new result files). Verified
on disk: the only working-tree changes are writer edits to `paper/main.tex` + `paper/custom.bib`
and the round_5 loop `.md` files — **no new `extra_*/` dirs, no new figures, no new CSV/JSON
results**. Nothing in the science changed; what remains is a small WRITER-ONLY text-cleanup
list (5 items, verified raw values below) for round 6, after which the paper is submission-ready.

## Verified new results
**None.** This was a FREEZE round — no new experiments, figures, or numbers were produced.
What I did instead was second-gate the three raw values the round-6 writer must transcribe
(all confirmed verbatim on disk, read-only):

- `tau_admitted_stable_range_raw = [0.385, 0.43]`
  src: `reports/runs/chem_patents/experimental_plots/extra_cost_frontier/tau_sweep_summary.json`
- `tau_cheapest_bge_range_raw = [0.33, 0.435]`
  src: same file as above
- `most_collapsed_language.overrep = 48.71` (language = `zh`)
  src: `reports/runs/chem_patents/experimental_plots/round06_language_collapse/summary.json`
  NOTE: value lives at JSON path `most_collapsed_language.overrep`, not bare top-level
  `overrep` (top-level returns None). Value identical — only the locator is more precise.

## WRITER-ONLY cleanup list forwarded to round 6 (exact anchors + verified values)
All five are pure text edits in `paper/main.tex`; no science changes. Anchors per round_5
troubleshoot.md, values re-verified on disk by this reporter.

- **W-M1 (correctness — the one logged mismatch; HIGHEST PRIORITY): restore raw tau-bands.**
  - L632: `[0.39,0.43]` -> `[0.385,0.43]`
  - L633: `[0.33,0.44]` -> `[0.33,0.435]`
  - L657 (cp_fig18 caption): `[0.33,0.44]` -> `[0.33,0.435]`
  Verified raw: admitted-stable `[0.385,0.43]`, cheapest-BGE `[0.33,0.435]`.

- **W-N2 (cohesion): harmonize `49x` -> `48.7x`.**
  - L955 (prose) and L973 (cp_fig09_10_collapse caption).
  Source overrep = 48.71 (zh), verified at round06_language_collapse/summary.json
  `most_collapsed_language.overrep`.

- **W-N3 (cohesion): standardize L-infinity dual notation `0.058 (5.84%)`** (0.058 == 5.84% = 1-0.9416).
  - Introduce both forms once at L692.
  - Convert bare-percentage sites L1030 and L1191 to `$0.058$`.
  - Leave existing `$0.058$` sites unchanged.

- **W-N1 (cohesion — TYPESET PASS only, deferrable): figure order.**
  - Reorder cp_fig19 above cp_fig22 so source order matches prose order, or let LaTeX float.

- **W-Guard (NO edit — camera-ready checklist note): keep the L-infinity floor distinct.**
  - Keep egemma's L-infinity = 0.058 floor textually apart from the now-tied ARI@100 (gap 0.004,
    CI straddles 0). Do not let the wording imply the ARI@100 tie undermines the L-infinity floor.

## Discrepancies / unverifiable claims
**None.** Every value claimed in `implement_report.md` was re-verified and matches disk
exactly. The implementer's note about the `overrep` JSON locator (nested vs top-level) is
confirmed accurate — the value (48.71) is unaffected.

## Changed files this round (git diff --stat summary)
True working-tree state (`git diff --stat HEAD` + `git status --porcelain`):
- `paper/main.tex` — 210 lines changed (+147 / -77): prior round writer edits (honesty-softening
  + figure trim from round 5's writer pass). NOT produced by this freeze sub-round.
- `paper/custom.bib` — 14 lines added: prior writer bib additions.
- `paper/loop/round_5/` — untracked loop `.md` files (troubleshoot/implement/this reporter).
- No `extra_*/` result dirs, no new `paper/figures/` files, no new `.csv`/`.json` — FREEZE confirmed.

## Backlogged (forthcoming) experiments to mention as pending
No new items added round 5; `needs_eval.md` left untouched. Existing parked items (treat as
DONE per critic contract — forthcoming, run manually post-loop):
- W4-formula-injection (causal upgrade of the formula-token observation) — needs query re-embed.
- CLIRMRS-external-validation (human/RAG utility signal vs CLIR-MRS) — needs new eval.
- XRC-conformal-M2 (split-conformal XRC guarantee) — deferred; calibration pool too thin (57 queries).
- CCI-hop-distance-law (ChEBI hop-distance confusion decay) — CPU but graph-build risk.
- equivalence-audit-spotcheck (expert check parallel golds are claim-equivalent) — needs human annotation.
- W3-alignment-causal-probe (LaBSE align-then-retrieve before/after on XRC50/RRC@100) — UPSIDE ONLY,
  needs re-embedding; already parked in Limitations (main.tex L1263-1275).

## Recommended next-round focus (for the round-6 story architect + writer)
- **Apply the 5-item WRITER-ONLY cleanup above** (W-M1 first — it is a correctness fix;
  then W-N2/N3; W-N1 is deferrable to typeset pass; W-Guard is a checklist note, no edit).
- **No new analysis warranted.** The science is frozen and submission-ready; the holistic
  robustness ledger (`tab:robust` / `app:robust`) is verified cell-for-cell. Story architect
  should NOT open new threads.
- **Recommend winding the loop down to polish-only.** After round 6's cleanup the paper is
  submission-ready; consider stopping new analysis and treating remaining rounds as
  typeset/proofread passes only. All genuine upside (W3 probe etc.) is correctly parked as
  post-submission work in Limitations.
