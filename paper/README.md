# Paper: multi-agent iterative writer

This directory holds the EMNLP **industry-track** paper for the Multilingual
Chemistry QAC benchmark, assembled by a closed loop of 9 Claude subagents
sequenced by a conductor slash-command.

## What's here
- `main.tex` + `acl.sty`, `acl_natbib.bst`, `custom.bib`, `anthology.bib.txt` — ACL/EMNLP template.
- `figures/` — figures copied/regenerated for the paper.
- `loop/` — orchestration state and per-round agent artifacts:
  - `state.json` — round counter, open issues, convergence flags.
  - `CHANGELOG.md` — one line per round.
  - `needs_eval.md` — backlog of GPU/eval experiments (never auto-run; treated as done by later critics).
  - `round_NN/` — `story.md`, `draft.tex` snapshot, `critic_{novelty,correctness,cohesion}.md`, `dreamer.md`, `troubleshoot.md`, `implement_report.md`, `reporter.md`.

## The agents (`.claude/agents/`)
`paper-story` → `paper-writer` → (`paper-critic-novelty`, `paper-critic-correctness`, `paper-critic-cohesion`) → `paper-dreamer` → `paper-troubleshooter` → `paper-implementer` → `paper-reporter`, then the reporter feeds back into the next round's story + writer. All run on Opus 4.8 with max reasoning.

## How to run
The loop is driven by slash-commands and is best run in **bypass-permissions** mode (no prompts):

```
/paper-round        # run exactly one round, then stop for inspection
/paper-loop         # run all 20 rounds, checkpointing each round
/paper-loop 5       # run 5 more rounds from the current state
```

Each round git-commits a checkpoint, so the loop is resumable and interruptible.
To resume after an interruption, just run `/paper-loop` again — it reads
`loop/state.json` and continues from `last_round_completed + 1`.

## Hard rules (enforced in agent prompts)
- **No fabricated numbers.** Every figure/number in `main.tex` must trace to a file under `reports/`.
- **Implementer budget.** Prefer 0 API calls; ≤20 requests/round if unavoidable; never run `--evaluate-mteb`. New evals go to `needs_eval.md`.
- **Paper stands without code-switch results** — code-switch is an optional bonus experiment.

## Venue & length (verified against the EMNLP Industry Track CFP)
- Venue: **EMNLP Industry Track**, double-blind. Template: ACL (unmodified) — correct, `[review]` mode on.
- **Page limit: 6 content pages (review), 7 camera-ready.** References, the **Limitations** section, and the **Appendix** do NOT count. Non-conforming papers are rejected without review.
- Strategy: keep ~5–7 floats in the 6-page body; move all other figures/tables to the Appendix (free).

## Compiling
This machine has a local Tectonic install at
`%LOCALAPPDATA%\Programs\Tectonic\tectonic.exe`; new terminals should pick it up
from the user `PATH`.

From `paper/`, compile the short scaffold with:

```powershell
tectonic short_main.tex
```

The output is `paper/short_main.pdf`. The current scaffold intentionally comments
out `\bibliography{custom}` until the short draft contains citations; re-enable it
once section prose starts citing papers.

To compile the full archived draft:

```powershell
tectonic main.tex
```

On a full TeX Live / MiKTeX install, this is equivalent to:

```powershell
latexmk -pdf short_main.tex
latexmk -pdf main.tex
```
