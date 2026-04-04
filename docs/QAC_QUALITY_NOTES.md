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
- Current reviewed prompt state: legal prompts now explicitly discourage checklist, inventory, deadline-only, amount-only, threshold-only, exact-phrase, and joined two-part questions
- Current score: `6.5 / 10`

Quick summary from the latest reviewed run:

- Languages present: `bg`, `de`, `el`, `en`, `es`, `fi`, `hu`, `it`, `lt`, `lv`, `pl`, `sl`
- Average question length: `152.1`
- Average answer length: `196.7`
- Heuristic article/label mentions in questions: `0`
- Full yield for the run: `22 / 22`
- Main residual failure patterns:
  - joined / two-part legal questions
  - timing / value lookup questions
  - exact-formula or exact-phrase lookup questions

Important comparison note:

- The previous write-up in this file was based on an older `44`-row review.
- This update folds in that missed review history, but the file is now anchored on the newest `22/22` run.
- The current scoring is intentionally stricter and focuses more heavily on whether each question is a **single sharp legal retrieval need**.

## Scorecard

Latest reviewed run score by perspective:

- Overall: `6.5 / 10`
- Legal-domain fit: `8.0 / 10`
- Faithfulness / grounding: `8.5 / 10`
- Article-label avoidance: `9.5 / 10`
- Language correctness: `8.5 / 10`
- Retrieval usefulness: `6.5 / 10`
- Question sharpness / single-focus: `6.0 / 10`
- Natural query style: `6.5 / 10`
- Resistance to checklist / inventory shape: `6.0 / 10`
- Resistance to timing / value lookup: `5.5 / 10`
- Resistance to exact-formula / exact-phrase lookup: `5.5 / 10`
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
- The questions are usually generated in the correct target-side language and the answers are generally grounded in the target-side document.
- Many questions now ask about:
  - legal effect
  - scope of a rule
  - what authority or party may act
  - what condition must be met
  - what happens when a requirement is or is not satisfied
- The current corpus cleanup appears to have helped with grounding and general usefulness, even though question shaping still needs work.

## Example Strengths

### Good: legal effect / condition / consequence

- `32003D0467__da__pl`
  - Q: `Co prawnie oznacza umieszczenie państwa członkowskiego na liście uznanych za oficjalnie wolne od gruźlicy?`
  - Why it is good: Narrow, grounded, and about the legal significance of recognition rather than a label lookup.

- `32003R1351__el__et`
  - Q: `Τι γίνεται με τις ποσότητες που προορίζονται για τους μη παραδοσιακούς εισαγωγείς όταν αυτές δεν χορηγούνται;`
  - Why it is good: Consequence-oriented and tied to one concrete legal mechanism.

- `31981R2180__bg__lt`
  - Q: `При какво условие държава-членка може да получи разрешение за приспособяване на броя на местата за свине в рамките на план за развитие на стопанството?`
  - Why it is good: Focuses on the condition that triggers a permission rather than merely asking for a number.

- `31996L0097__it__pl`
  - Q: `Cosa consente la direttiva a un datore di lavoro nei confronti di persone che hanno raggiunto l'età pensionabile prevista da un regime professionale ma non ancora l'età pensionabile legale?`
  - Why it is good: It asks about what the directive permits in one legal situation and yields a grounded answer.

- `52005PC0571__it__sl`
  - Q: `Quale condizione finanziaria deve essere soddisfatta prima che l'aiuto a fondo perduto sia messo a disposizione della Georgia in almeno due rate?`
  - Why it is good: Focused on one precondition for action, which is a strong legal retrieval shape.

## What Still Looks Weak

- The main remaining weakness is now **joined / two-part and lookup-shaped legal questions**, not article-label phrasing.
- Some questions still ask for **too many conditions at once**.
- Some still behave like **literal compliance extraction** rather than strong semantic retrieval.
- Some still ask directly for a **date, duration, value, wording, or full inventory** when a better legal-effect question seems possible.
- The corpus-side cleanup improved the source material more than the prompt changes improved the final question shapes.

## Example Weaknesses

### Weak: joined / two-part legal question

- `32005H0256__fi__pl`
  - Why it is weak: It asks both **who** grants discharge and **on whose recommendation** it happens. One of those should be kept and the other dropped.

- `31996L0021__fi__pt`
  - Why it is weak: It asks both whether non-compliant products may remain on sale and what limiting condition applies, which makes the answer dual-part.

### Weak: timing / value lookup

- `32006HB0001__de__es`
  - Why it is weak: It asks only for the maximum duration of a mandate, which is mainly a value lookup.

- `32006R0807__es__lv`
  - Why it is weak: It asks only for the validity period of export certificates, which is still mostly a timing lookup.

- `52005SC0011__lt__sv`
  - Why it is weak: It asks for the nature and size of corrective measures in a way that leans toward extracting prescribed values and targets.

### Weak: exact-formula / exact-phrase lookup

- `32004R0687__mt__sl`
  - Why it is weak: It asks for the exact English wording the official veterinarian must enter, which is closer to form-text retrieval than to legal-semantic retrieval.

### Weak: long condition inventory

- `31996R1676__de__en`
  - Why it is weak: It asks for the full set of conditions under which a customs valuation method may be allowed, producing a long checklist answer rather than one sharp legal point.

- `31991L0663__lv__ro`
  - Why it is weak: It asks for the whole bundle of circumstances under which vehicles are not treated as a different type, which still pushes the answer toward inventory form.

## Current Quality Summary

- Domain fit: clearly improved relative to the earlier chemistry-style setup
- Language correctness: good in the reviewed sample
- Faithfulness: generally good
- Article-label avoidance: strong and no longer the main weakness
- Yield stability: improved in the newest run
- Retrieval usefulness: usable, but still held back by joined and lookup-shaped questions
- Main remaining issue: some questions are still dual-part, list-shaped, timing/value-oriented, or exact-formula-oriented

Practical judgment:

- The current JRC questions are **usable**, but this newest sample is still not consistently at the level of narrow, high-value semantic legal retrieval questions.
- The run is operationally better because it reached full `22/22` yield.
- The corpus and candidate-pool cleanup helped, but the remaining question-shape issues are still visible enough that the QA score should stay moderate.
- Current score: `6.5 / 10`

## Recommended Next Improvements

1. Reject more dual-part questions joined by `and`, `or`, or equivalent multi-clause legal framing when one sharper sub-question is available.
2. Reject more timing-only, duration-only, effective-date-only, amount-only, and threshold-only questions unless that value is truly the key retrieval target.
3. Reject exact-formula / exact-phrase questions when the stronger retrieval need is about legal effect, authorization, condition, or consequence.
4. Reject more inventory-style questions about covered categories, listed elements, or full sets of conditions when one narrower legal point is available.
5. Prefer questions about:
   - what triggers a consequence
   - what prevents approval
   - what a missing document/report changes
   - who is exempt or affected
   - when a derogation applies
   - what legal effect follows
6. Penalize questions whose answer naturally becomes a semicolon-separated inventory.

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

- Result: Newest current review with `22/22` yield.
- Main improvement: the run was stable, article-label phrasing remained controlled, and several questions were still legally grounded and single-point.
- Main remaining issue: the remaining errors are now concentrated in joined/two-part questions, timing/value lookups, exact-formula questions, and long condition inventories.
