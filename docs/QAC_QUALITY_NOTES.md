# QAC Quality Notes

Living notes for reviewing the current Question-Answer-Context output.

This file tracks the **JRC-Acquis legal/regulatory QA pipeline**, not the older chemistry/patent pipeline.

## Current Snapshot

- Source file reviewed: `data/JRC-ACQUIS/qac/qac.csv`
- Current reviewed run: `22` rows
- Current generation design: **pair-level**, same-language generation from the translated/target side of each sampled pair
- Current linking design: one generated query is attached to **both** documents in the selected pair
- Current domain: legal / regulatory EU documents
- Current validation: language match + faithfulness + legal-domain question-quality checks
- Current reviewed prompt state: legal prompts now explicitly discourage checklist, inventory, deadline-only, amount-only, threshold-only, exact-phrase, joined two-part, and exact-code / exact-value lookup questions; the QA loop now also logs checker-triggered retries
- Current score: `7.2 / 10`

Quick summary from the latest reviewed run:

- Languages present: `cs`, `el`, `es`, `et`, `fi`, `fr`, `hu`, `lt`, `lv`, `pl`, `pt`, `sl`, `sv`
- Average question length: `131.2`
- Median question length: `129`
- Average answer length: `177.5`
- Median answer length: `184.5`
- Heuristic article/label mentions in questions: `0`
- Full yield for the run: `22 / 22`
- Questions with length `>= 180` chars: `3 / 22`
- Checker-triggered retry wins: `6 / 22`
- Main residual failure patterns:
  - condition / inventory-shaped legal questions
  - some remaining broad procedural questions
  - a smaller number of lookup-shaped questions that still pass after rewrite

Important comparison note:

- The previous write-up in this file was based on an older `44`-row review.
- This update folds in that missed review history, but the file is now anchored on the newest `22/22` run.
- The current scoring is intentionally stricter and focuses more heavily on whether each question is a **single sharp legal retrieval need**.

## Scorecard

Latest reviewed run score by perspective:

- Overall: `7.2 / 10`
- Legal-domain fit: `8.5 / 10`
- Faithfulness / grounding: `8.5 / 10`
- Article-label avoidance: `9.5 / 10`
- Language correctness: `8.5 / 10`
- Retrieval usefulness: `7.0 / 10`
- Question sharpness / single-focus: `6.8 / 10`
- Natural query style: `7.0 / 10`
- Resistance to checklist / inventory shape: `6.5 / 10`
- Resistance to timing / value lookup: `6.8 / 10`
- Resistance to exact-formula / exact-phrase lookup: `7.0 / 10`
- Pipeline robustness for this run: `8.5 / 10`

Interpretation:

- `8.5-9.5`: strong
- `7-8`: good / usable
- `6-7`: decent but still a visible weakness
- `<6`: problem area

## Method

Current JRC review method:

1. Run `uv run main.py --source JRC-ACQUIS`.
2. Review `data/JRC-ACQUIS/qac/qac.csv`.
3. Check broad signals:
   - language distribution
   - question/answer length
   - explicit article-number / label usage
   - joined or checklist-shaped questions
   - timing / value / formula lookup behavior
4. Read a manual sample of generated rows.
5. Classify problems into:
   - overly generic
   - article/provision lookup
   - bundled legal conditions
   - deadline/value/formula lookup
   - overly long or multi-clause question shape

## What Looks Good

- The current questions remain much better aligned with **legal/regulatory** text than the earlier chemistry-oriented prompts.
- The obvious `According to Article ...` / label-led failures are still largely gone.
- The newest run reached full yield: `22 / 22` rows were written.
- The tighter generator + checker combination now catches and rewrites several weak first-attempt questions instead of letting them pass immediately.
- The questions are usually generated in the correct target-side language and the answers are generally grounded in the target-side document.
- Many questions now ask about:
  - legal effect
  - scope of a rule
  - what authority or party may act
  - what condition must be met
  - what happens when a requirement is or is not satisfied
- The current corpus cleanup appears to have helped with grounding and general usefulness, and the latest QA iteration materially improved question length and reduced the worst literal lookup shapes.

## Example Strengths

### Good: legal effect / condition / consequence

- `31995D0568__es__fi`
  - Q: `¿Qué efecto tiene la limitación de la competencia de la Comisión para aprobar modificaciones de los Anexos sobre las sustancias que no estén ya reguladas por la legislación comunitaria pertinente?`
  - Why it is good: The retry moved this away from literal scope wording and toward the legal effect of the limitation.

- `32006R1003__es__hu`
  - Q: `¿Qué efecto jurídico tiene el importe fijado en el anexo para la importación de melaza cuando se suspenden los derechos de importación?`
  - Why it is good: It now asks for the legal effect of the amount instead of asking only for the number.

- `32000D0532__et__lt`
  - Q: `Mis õiguslikku tagajärge toob kaasa see, kui jäätmes on ühe või mitme R60/R61 märgistusega reproduktsiooni mõjutava aine üldkontsentratsioon vähemalt 0,5%?`
  - Why it is good: A former threshold lookup was rewritten into a legal-consequence question.

- `31992R0684__nl__pt`
  - Q: `Aplica-se o regulamento a uma transportadora estabelecida num Estado‑membro que utilize autocarros matriculados noutro Estado‑membro?`
  - Why it is good: This is much cleaner than a literal article extraction and asks one direct applicability question.

- `32003R1651__et__pl`
  - Q: `Kas juurdepääsu keeldimise otsuse vastu võib esitada kaebuse Euroopa Ombudsmani poole?`
  - Why it is good: The retry collapsed a broader procedural question into one focused appeal-right question.

## What Still Looks Weak

- The main remaining weakness is now **condition / inventory-shaped legal questions**, not article-label phrasing.
- Some questions still ask for **a bundle of conditions or definitions** instead of one narrower legal point.
- Some still behave like **safe procedural extraction** rather than the strongest semantic retrieval question available.
- A few questions are still a bit broad even after the latest retries.
- The latest prompt + checker pass materially improved output quality, but it did not fully solve condition-inventory shape.

## Example Weaknesses

### Weak: condition / inventory shape

- `31999R2337__fr__it`
  - Why it is weak: It still asks for a bundle of issuance conditions, even though it is cleaner than a duration-only lookup.

- `31967R0422__lt__sl`
  - Why it is weak: It still asks for the full set of conditions for qualifying as a dependent child, which keeps the answer somewhat list-shaped.

### Weak: still somewhat broad / procedural

- `31999D0468__el__lv`
  - Why it is weak: It is usable, but still a bit broad and procedural compared with the sharper legal-effect questions in the same sample.

- `32001R1112__cs__sk`
  - Why it is weak: It is better than the earlier exact-code lookup, but it still remains a weaker exception/period question than the strongest rows.

### Weak: residual multi-condition question

- `31998L0093__et__lv`
  - Why it is weak: It still asks two related obligations together instead of fully isolating one legal point.

## Current Quality Summary

- Domain fit: clearly improved relative to the earlier chemistry-style setup
- Language correctness: good in the reviewed sample
- Faithfulness: generally good
- Article-label avoidance: strong and no longer the main weakness
- Yield stability: improved in the newest run
- Retrieval usefulness: now clearly better than the previous `22`-row sample
- Main remaining issue: some questions are still condition-list or inventory-shaped, even after the stronger retries

Practical judgment:

- The current JRC questions are **usable and meaningfully better than the previous reviewed `22`-row sample**.
- The run is operationally stronger because it reached full `22/22` yield and the checker now visibly rewrites several weak first drafts.
- The remaining issue is no longer mainly raw lookup/value questions; it is more often condition-list shape and a smaller number of broad procedural queries.
- Current score: `7.2 / 10`

## Recommended Next Improvements

1. Reject more condition-list and inventory-style questions when one narrower legal point is available.
2. Reject more procedural questions that are correct but still broader than the strongest available legal-effect query.
3. Keep the current pressure against exact value/code/period lookups; that part improved and should remain in place.
4. Prefer questions about:
   - what triggers a consequence
   - what prevents approval
   - what a missing document/report changes
   - who is exempt or affected
   - when a derogation applies
   - what legal effect follows
5. Penalize questions whose answer naturally becomes a semicolon-separated inventory.

## Review Log

### Review 1

- Result: Initial JRC same-language sample was usable but too citation-heavy and too close to legal lookup.
- Main improvement needed: remove article-label dependence and make questions more semantic.

### Review 2

- Result: Better than Review 1.
- Main improvement: article/provision-label style was reduced significantly.
- Main remaining issue: some questions were still too broad, checklist-like, or deadline/list-oriented.

### Review 3

- Result: Better than Review 2.
- Main improvement: stronger avoidance of article-number phrasing and better legal-domain fit.
- Main remaining issue: some questions were still too broad or too extractive, especially around conditions, exemptions, deadlines, or numeric replacements.

### Review 4

- Result: Better than Review 3.
- Main improvement: article/provision-label phrasing became largely controlled; the run was more consistently about operative meaning and legal effect.
- Main remaining issue: the residual failures concentrated more in long multi-part questions and date/value/inventory-style queries.

### Review 5

- Result: Older larger reviewed run with `44/44` yield.
- Main improvement: question length improved and several earlier dual-step procedural questions were replaced by narrower single-point queries.
- Main remaining issue: residual failures were concentrated in dual-part questions, timing/frequency questions, and inventory-style covered-set questions.

### Review 6

- Result: Earlier `22/22` reviewed run before the latest checker/generation tightening.
- Main improvement: the run was stable, article-label phrasing remained controlled, and several questions were still legally grounded and single-point.
- Main remaining issue: the remaining errors were concentrated in joined/two-part questions, timing/value lookups, exact-formula questions, and long condition inventories.

### Review 7

- Result: Better than Review 6.
- Main improvement: question length dropped materially, several exact value/threshold lookups were rewritten into legal-effect questions, and the new logs show multiple weak first attempts being rejected and repaired.
- Main remaining issue: the weakest rows are now more often condition-list or inventory-shaped questions rather than exact code/value lookups.
