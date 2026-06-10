# Troubleshooting plan (round 5)

**Triage verdict: FREEZE confirmed. ZERO compute this round.** I re-grounded every
on-disk fact the conductor named (Bash read-only). All residual round-5 items are
WRITER-ONLY text edits with no recompute: the M1 τ-band raw values exist verbatim on
disk for transcription, the 49×/48.7× nit has a clear truer source value, and the
L∞ dual-notation is one number in two displays. The robustness ledger (`tab:robust`)
already exists and is verified cell-for-cell; the only candidate CPU-only analysis
(LOLO on the headline separability) is ill-defined and noise, not signal. The dreamer's
FREEZE-and-polish recommendation is **correct as stated**.

---

## DO-NOW (ordered) — each: goal / files / exact commands / inputs(exist?) / outputs / runtime / verify / api-cost

**NONE. No compute this round.**

There is no high-value CPU-only DO-NOW left. Justification (verified on disk):
- The holistic "every load-bearing scalar + resample CI" ledger the conductor asked
  about **already exists** as `tab:robust` / `app:robust` (main.tex: `\label{tab:robust}`
  L1342, `\label{app:robust}` L1308; `\ref`d from the body at L707, L1023, L1042, L1052,
  L1212 = 5 sites). All six rows verified to match
  `reports/runs/chem_patents/experimental_plots/extra_robustness_appendix/robustness_table.csv`
  cell-for-cell (separability r 0.9577 / P(r>0)=0.9997 / CI [0.7301,0.9977]; three XRC50
  medians 3.5/2.0/1.25 with finite CIs and censored-draw fraction 0.0; ARI@100 gap 0.004
  / CI [-0.174,0.1762] / order-prob 0.5191; partial r 0.2948 / p 0.5706 / zero-order
  0.9577). Adding a second ledger or extending this one would duplicate a clean asset
  and risk a new scalar-inconsistency. **Do not extend.**
- **LOLO on the headline separability slope is noise, not signal — DO NOT RUN.** The
  headline r=+0.96 is a correlation **across the 7 non-degenerate encoders**, not across
  languages, so "leave-one-language-out" on *this* scalar is not well-defined — the
  resample unit is the model, and the model-jackknife (n=7→+0.958, n=9→+0.888) plus the
  sign-stability bootstrap (P(r>0)=0.9997, CI [0.73,1.00]) already do the strictly
  stronger job and are already in `tab:robust`. A LOLO would only touch the per-language
  *descriptive* slopes (home-advantage −0.57, n=5), which are deliberately fenced as
  descriptive and kept out of abstract/intro/conclusion. Re-touching an already-fenced
  n=5 slope with an even smaller n would only tempt a new float and invite "why LOLO on
  the language axis but not the encoder axis?" Negative payoff.

**API cost this round: 0.**

---

## WRITER-ONLY (cleanup list — exact line anchors + raw values to transcribe)

All five items are pure text edits; no figure regeneration, no recompute. Anchors verified
against the current `paper/main.tex`.

### W-M1 (correctness, low — the round's one logged MISMATCH). Restore raw τ-bands.
Transcribe these on-disk raw values (verified in
`reports/runs/chem_patents/experimental_plots/extra_cost_frontier/tau_sweep_summary.json`:
`tau_admitted_stable_range_raw = [0.385, 0.43]`, `tau_cheapest_bge_range_raw = [0.33, 0.435]`):
- **L632** (body): `the admitted set is stable for $\tau \in [0.39, 0.43]$` →
  `... $\tau \in [0.385, 0.43]$`
- **L633** (body): `\texttt{bge-m3} remains the cheapest admitted reader for $\tau \in [0.33, 0.44]$` →
  `... $\tau \in [0.33, 0.435]$`
- **L657** (`cp_fig18` caption): `stable for $\tau \in [0.33, 0.44]$ and flipping ...` →
  `stable for $\tau \in [0.33, 0.435]$ and flipping ...`

(Alternative accepted by Reviewer #2: keep the 2dp display and append once "(rounded to
the 0.005 τ-grid; raw $[0.385,0.43]$ / $[0.33,0.435]$)". Restore-raw is shorter and removes
the discrepancy entirely — preferred.)
**DO NOT TOUCH** (all verified correct against source): the granite-flip clause (granite
CLIR@10=0.3285 → admitted at τ≤0.3285 ≈0.33, L634/L657), the "above $\tau{\approx}0.45$ only
embeddinggemma" clause (L636), and the "egemma corner τ-invariant" clause (L637/L658).

### W-N2 (cohesion, trivial). Harmonize 49× → 48.7× to match the rendered panel.
The truer source value is **48.71** (`reports/runs/chem_patents/experimental_plots/round06_language_collapse/summary.json`
→ `"overrep": 48.71`), displayed as **48.7×** on the rendered left panel and in
`experimental_plots/FINDINGS.md` L162 ("zh at **48.7×**"). The "49×" appears **only** in
`key_findings/EXECUTIVE_SUMMARY.md` L44 as a round of 48.71; the paper followed the EXEC
summary. The panel/FINDINGS value is the truer source, so harmonize the two prose sites to
the panel:
- **L955** (Analysis prose): `over-fetch their own language up to $49\times$ the` →
  `... up to $48.7\times$ the`
- **L973** (`cp_fig09_10_collapse` caption): `same-language over-representation by query language, up to $49\times$ the` →
  `... up to $48.7\times$ the`

(Reviewer #3's alternative "$\sim$$49\times$" is also acceptable; matching the panel's
48.7× exactly is cleanest and removes the 0.3 caption-vs-panel gap.)

### W-N3 (cohesion, trivial). Standardize L∞ dual notation.
0.058 and 5.84% are the **same number** (1 − RRC@1000 = 1 − 0.9416 = 0.0584; verified).
Standardize to one running-prose notation:
- **L692** (first body use): write once as `$L_\infty = 1-\mathrm{RRC@1000} = \mathbf{0.058}$ ($5.84\%$)`
  to introduce both forms.
- Convert the two bare-percentage sites to `$0.058$` for single notation thereafter:
  **L1030** (`...the floor $L_\infty=5.84\%$ of foreign twins...` → `$L_\infty=0.058$`) and
  **L1191** (`$L_\infty = 1-\mathrm{RRC@1000}=5.84\%$ of foreign twins...` → `...=0.058$`).
- Leave the existing `$0.058$` sites (L695, L703, L719, L734, L1036, L1088, L1109, L1289)
  unchanged.

### W-N1 (cohesion, trivial, inherited — TYPESET PASS only, defer).
On the final typesetting pass, move the `cp_fig19` (RRC budget) `\begin{figure}` block (≈L728)
above the `cp_fig22` (ARI) block (≈L713) so source order matches prose order (RRC→ARI), **or**
let LaTeX float — Reviewer #3 expects LaTeX to co-locate them anyway. Lowest priority; no
content edit; deferrable.

### W-Guard (no edit — camera-ready guardrail).
Make **no change now**. Add a camera-ready checklist note: any future copy-edit must keep
embeddinggemma's smallest-`$L_\infty=0.058$` floor (the C4 differentiator) textually apart
from the now-tied ARI@100 (egemma 0.229 vs qwen3 0.233, gap 0.004, CI straddles 0). Correct
at all sites today; this only prevents a future regression that would cost C4 its only
non-tied capability-axis differentiator.

---

## BACKLOG-EVAL (exact commands + rationale for needs_eval.md)

**No NEW backlog item this round.** The single high-value future move — the **W3 alignment
causal probe** (fit a per-language alignment map on one model, re-retrieve, recompute XRC50 /
RRC@100 / ARI@100 / L∞; if L∞ drops under alignment while staying flat under re-ranking, the
"alignment-only" adjective becomes causal) — is **already parked** in the paper's Limitations
(main.tex L1263–1275) and named as post-submission UPSIDE by both the story freeze and
Reviewer #1. It requires new retrieval runs + an alignment fit (cannot be CPU-only-from-disk),
the current submission stands without it, and adding it now would re-open a hardening cycle.
Leave it frozen as forthcoming; do not append a new `needs_eval.md` entry. (Other deferred
items — T5 parallel-gold equivalence audit, CTC/CERC/LSR/ELI/ARGF level-2 metrics — likewise
stay post-submission and unchanged.)

---

## Round API budget plan (target 0, cap 20)

**0 API calls.** No LLM-free or LLM-using builds proposed this round. `--evaluate-mteb` not run.

---

## Risks & sequencing notes

- **Sequence:** W-M1 first (the only logged correctness mismatch), then W-N2 and W-N3 (trivial
  harmonizations), then W-N1/W-Guard on the typeset/camera-ready pass. All are independent
  single-token-class edits; none touches a figure, a number's provenance, or a claim surface.
- **Lowest-risk possible round.** Every edit makes display match source more exactly (M1, N2)
  or unifies notation (N3); none can destabilize a result or float. No figure is regenerated.
- **Protect-the-paper:** the paper already stands without any code-switch / new-eval result.
  The dominant risk this round is *over-editing* — adding a "nice-to-have" analysis, a new
  float, or a new claim surface at the exact moment the residual surface is five cosmetic items.
  Decline LOLO, any new metric, any new float, and the optional "ledger-promise" framing
  sentence unless a human explicitly asks. Wind the chain down to polish-only.
