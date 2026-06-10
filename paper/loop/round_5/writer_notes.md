# Writer notes (round 5)

Round 5 = polish + honesty-correction + figure-trim. NO new claims. Every edit is a
softening, a trim, a cite, or an appendix consolidation of verified resamples.

## Sections touched
- §2/§4 (Metrics, ARI paragraph): R1 rho_k cite.
- §6.1 (cost frontier): C4 tau-band; cp_fig18 caption C4.
- §6.1 (ARI read-off): R3 trim (open at result), C2 tie, R2 fig22 caption.
- §6.1/§Analysis: MERGE 1 (mate pair) and MERGE 2 (collapse pair).
- §6.3 (leaderboards): radar CUT-NOTE clause; deleted the two radar figure blocks.
- §7 (Analysis crux): C1 sign-stability + wide CI; cp_fig11 caption C1; C2 tie + L_inf kept; C3 partial-r caveat.
- §8 (Deployment): C2 tie + C4 band in both the capability-corner and per-route paragraphs.
- §9 (Limitations): one sentence leading to Appendix robustness table.
- Appendix: NEW robustness table (tab:robust, label app:robust).
- custom.bib: residualrerank2026 entry added.

## The four honesty corrections (C1-C4)
- **C1 (separability).** §7 now leads with sign-stability P(r>0)=0.9997 (point r=0.96) and
  presents the n=7 CI [0.73,1.00] as WIDE; "robust" explicitly redefined as sign-robust under
  resampling. cp_fig11 caption carries P(r>0)=0.9997 + wide CI. Abstract/conclusion keep "+0.96,
  robust" without CI (protected surface) — sign-robust meaning now anchored in body+appendix.
- **C2 (ARI tie).** Every ARI@100 site (§6.1 read-off, fig22 caption, §7 crux, §8 capability-corner,
  §8 per-route) now reads as a TIE: egemma 0.229 / qwen3-0.6B 0.233, gap 0.004, 95% CI [-0.174,0.176]
  straddles zero. egemma KEEPS its separate smallest alignment-only floor L_inf=0.058 (next qwen3
  0.073) everywhere — not erased, kept textually distinct from the ARI@100 tie.
- **C3 (collinearity).** §7 now frames separability as a DESCRIPTIVE correlate: partial
  r(auc_cross, CLIR@10 | Recall@10)=+0.29, p=0.57 n.s.; explicit "not an effect net of general
  retrieval capability." Mechanism bridge ("the lever is at the embedding level") kept as qualitative
  reading with the partial-r caveat beside it. No "independent / not a tautology" reading anywhere.
- **C4 (tau-band).** §6.1 + cp_fig18 caption + §8 now state: admitted set stable tau in [0.39,0.43];
  bge-m3 cheapest tau in [0.33,0.44]; flips to granite-278m below ~0.33; only egemma admitted above
  ~0.45; ONLY egemma's max-CLIR corner is tau-invariant. Never "the rule is robust."

## Figure cut (29 -> 26 floats; 25 in-body)
- CUT cp_fig14 + ag_fig10 radars (one-clause CUT-NOTE in leaderboard paragraph; content carried by
  Tables 1-2 + cp_fig17 ribbon).
- REPLACE mate pair (cp_fig06/07) with cp_fig06_07_mate.png (single float, one caption, refs collapsed).
- REPLACE collapse pair (cp_fig09/10) with cp_fig09_10_collapse.png (single float, refs collapsed).
- ADD appendix robustness table (the round's one new float, outside the 8-page body budget).
- Net: 23 figures + 3 tables = 26 floats; appendix table is outside body => 25 in-body floats (target met).

## Cheap residuals
- **R1.** rho_k residualrerank2026 cited once in §4 ARI paragraph with the credit-and-distinguish
  half-clause ("we invert it ... alignment-only floor that rho_k has no analogue for"); bib entry added.
- **R2.** fig22 caption "all nine models" -> "every model (the identity closes for all nine; the
  figure shows the seven non-degenerate)." No value changed.
- **R3.** §6.1 ARI paragraph opens at the result, not the re-definition of the §4 identity.

## Critic points addressed
- correctness D-NEW (fig22 "nine vs seven"): fixed via R2.
- novelty residual #1 (rho_k cite): fixed via R1.
- novelty/cohesion fig22 caption nit: fixed via R2.
- cohesion joint #1 (§6.1 re-definition redundancy): fixed via R3.
- cohesion #2 (float overload 29): fixed via the cut to 26 (25 in-body).
- cohesion #3 (float order cp_fig22 before cp_fig19): DEFERRED — cosmetic, left to LaTeX float;
  did not spend a content edit (story marks it optional/lowest-priority).

## Self-lint (all pass)
- No refs to deleted labels (fig:cp_rank, fig:cp_distractor, fig:cp_radar, fig:ag_radar): 0 hits.
- All 23 \includegraphics targets exist on disk.
- begin/end balance: figure 23/23, table 3/3; brace count 910/910; no env mismatch.
- residualrerank2026: cited 1x, 1 bib entry.
- app:robust: 1 label def, 5 refs (§6.1, §7 x2, §8 implicit via §ref, §9) all resolve.
- Protected surfaces (abstract / intro contributions / conclusion body) scanned clean of ARI@100,
  0.229, tau-band/\tau, partial-r, per-route/router. Abstract/conclusion keep "+0.96, robust" only.

## Open \todo{trace:...} items (carried, NOT new)
- corpus dedup count 14,401 + GP/EPO/JRC coverage matrix (source slides, not under reports/) —
  remains comment-fenced in §3; not in rendered prose.
- human-eval numbers (8.33/10, 97/100, +4.3pp) — comment-fenced in §3 + Appendix; not in prose.
- MMTEB/MIRACL/NeuCLIR transfer number — comment-fenced in Limitations; backlogged.
None are new this round; all three pre-existed and stay fenced per the correctness contract.
