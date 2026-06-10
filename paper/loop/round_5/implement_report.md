# Implementation report (round 5)

**FREEZE ROUND — 0 compute items executed, 0 API calls.** Confirmed against
`paper/loop/round_5/troubleshoot.md`: the DO-NOW list is explicitly EMPTY ("NONE. No
compute this round."). No code, data, figures, or paper text were changed. The only
actions taken were read-only spot-verifications of the raw values that round 6's writer
will transcribe.

## Done (each: what / command run / output paths / quick verification / api-calls-used)

- **Confirmed FREEZE.** Re-read `paper/loop/round_5/troubleshoot.md`. Triage verdict is
  FREEZE; DO-NOW = NONE; "API cost this round: 0"; all five residual items are WRITER-ONLY
  text edits forwarded to round 6. The candidate CPU-only analysis (LOLO on headline
  separability) is judged ill-defined/noise and explicitly DO-NOT-RUN. The holistic
  robustness ledger (`tab:robust` / `app:robust`) already exists and is verified
  cell-for-cell. No DO-NOW to execute. (api-calls: 0)

- **Spot-verified raw τ-band values (read-only Bash).** Read
  `reports/runs/chem_patents/experimental_plots/extra_cost_frontier/tau_sweep_summary.json`.
  Confirmed verbatim:
  - `tau_admitted_stable_range_raw = [0.385, 0.43]`
  - `tau_cheapest_bge_range_raw   = [0.33, 0.435]`
  These match the troubleshoot.md transcription targets exactly. (api-calls: 0)

- **Spot-verified overrep source value (read-only Bash).** Read
  `reports/runs/chem_patents/experimental_plots/round06_language_collapse/summary.json`.
  Confirmed `most_collapsed_language.overrep = 48.71` (truer source for the 49× → 48.7×
  harmonization). Minor note for round 6: the value lives at JSON path
  `most_collapsed_language.overrep`, not the bare top-level `overrep` named in
  troubleshoot.md — value is identical (48.71), only the locator is more precise. (api-calls: 0)

## Backlogged to needs_eval.md (id + reason + exact command)

- **None.** Troubleshoot.md adds no new backlog item this round. The one high-value future
  move (W3 alignment causal probe) is already parked in the paper's Limitations
  (main.tex L1263–1275) and stays frozen as post-submission upside; it is NOT CPU-only and
  is not appended. `needs_eval.md` is left untouched.

## New figures copied to paper/figures/ (basename -> source)

- **None.** No figure regenerated or copied. No `--evaluate-mteb` run.

## WRITER-ONLY cleanup list forwarded to round 6 (no edits made here)

All five are pure text edits in `paper/main.tex` (anchors per troubleshoot.md §WRITER-ONLY),
forwarded intact with confirmed raw values:
- **W-M1** (correctness, the round's one logged mismatch): restore raw τ-bands —
  L632 `[0.39,0.43]`→`[0.385,0.43]`; L633 `[0.33,0.44]`→`[0.33,0.435]`;
  L657 (cp_fig18 caption) `[0.33,0.44]`→`[0.33,0.435]`. Raw values verified on disk above.
- **W-N2** (cohesion): harmonize `49×`→`48.7×` (source 48.71, verified) at L955 (prose) and
  L973 (cp_fig09_10_collapse caption).
- **W-N3** (cohesion): standardize L∞ dual notation (0.058 ≡ 5.84% = 1−0.9416) — introduce
  both forms once at L692; convert bare-percentage sites L1030 and L1191 to `$0.058$`; leave
  existing `$0.058$` sites unchanged.
- **W-N1** (cohesion, TYPESET PASS only, deferrable): reorder cp_fig19 above cp_fig22 so
  source order matches prose order, or let LaTeX float.
- **W-Guard** (no edit): camera-ready checklist note — keep egemma's L∞=0.058 floor textually
  apart from the now-tied ARI@100 (gap 0.004, CI straddles 0).

## Failures / surprises (verbatim errors, real outcomes)

- Initial probe `d.get('overrep')` on the language-collapse summary returned `None` because
  the value is nested under `most_collapsed_language`, not at top level. Resolved by walking
  the JSON; value confirmed as 48.71. No impact — this is a read-only verification, and the
  number the writer needs is confirmed correct. No other surprises. Nothing backlogged.

## API calls used this round: 0/20
