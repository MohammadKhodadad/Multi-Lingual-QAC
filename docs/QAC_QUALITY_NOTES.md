# QAC Quality Notes

Living notes for reviewing the current Question-Answer-Context output.

This file tracks the **JRC-Acquis legal/regulatory QA pipeline**, not the older chemistry/patent pipeline.

## Current Snapshot

- Source file reviewed: `data/JRC-ACQUIS/qac/qac.csv`
- Current reviewed run: `43` rows written from `44` generation units
- Current generation design: **pair-level**, same-language generation from the translated/target side of each sampled pair
- Current linking design: one generated query is attached to **both** documents in the selected pair
- Current domain: legal / regulatory EU documents
- Current validation: language match + faithfulness + legal-domain question-quality checks + legal-shape check
- Current reviewed prompt state: legal prompts now explicitly discourage checklist, inventory, deadline-only, amount-only, threshold-only, exact-phrase, joined two-part, exact-code / exact-value lookup questions, and broad legal-shape failures; the QA loop logs checker-triggered retries
- Current score: `7.3 / 10`

Quick summary from the latest reviewed run:

- Languages present: `bg`, `cs`, `da`, `de`, `el`, `en`, `es`, `et`, `fi`, `fr`, `hu`, `it`, `lt`, `lv`, `nl`, `pl`, `pt`, `sk`, `sl`, `sv`
- Average question length: `138.7`
- Median question length: `134`
- Average answer length: `186.1`
- Median answer length: `170`
- Heuristic article/label mentions in questions: `0`
- Yield for the run: `43 / 44`
- Questions with length `>= 180` chars: `5 / 43`
- Checker-triggered retry wins: `17 / 44`
- Main residual failure patterns:
  - accepted condition-list legal questions
  - accepted document/item inventory questions
  - accepted timing/value/procedural lookup questions

Important comparison note:

- The previous write-up in this file was anchored on the newer `22/22` run.
- This update is now anchored on the latest `43/44` run after the dedicated legal-shape checker was added.
- The current scoring is intentionally stricter and focuses more heavily on whether each question is a **single sharp legal retrieval need**.

## Scorecard

Latest reviewed run score by perspective:

- Overall: `7.3 / 10`
- Legal-domain fit: `8.6 / 10`
- Faithfulness / grounding: `8.5 / 10`
- Article-label avoidance: `9.5 / 10`
- Language correctness: `8.5 / 10`
- Retrieval usefulness: `7.2 / 10`
- Question sharpness / single-focus: `6.9 / 10`
- Natural query style: `7.1 / 10`
- Resistance to checklist / inventory shape: `6.4 / 10`
- Resistance to timing / value lookup: `7.1 / 10`
- Resistance to exact-formula / exact-phrase lookup: `7.3 / 10`
- Pipeline robustness for this run: `8.0 / 10`

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
5. Read terminal retry logs to see which checker rejected each failed draft.
6. Classify problems into:
   - overly generic
   - article/provision lookup
   - bundled legal conditions
   - document / item inventories
   - deadline / value / procedural-limit lookup
   - overly long or multi-clause question shape

## What Looks Good

- The current questions remain much better aligned with **legal/regulatory** text than the earlier chemistry-oriented prompts.
- The obvious `According to Article ...` / label-led failures are still largely gone.
- The newest run reached near-full yield: `43 / 44` rows were written.
- The tighter generator + checker combination now catches and rewrites many weak first-attempt questions instead of letting them pass immediately.
- The new legal-shape checker is now visibly firing on broad legal-shape, condition-list, and definition-inventory failures.
- The questions are usually generated in the correct target-side language and the answers are generally grounded in the target-side document.
- Many questions now ask about:
  - legal effect
  - scope of a rule
  - what authority or party may act
  - what condition must be met
  - what happens when a requirement is or is not satisfied
- The current corpus cleanup appears to have helped with grounding and general usefulness, and the latest QA iteration materially improved repair behavior and reduced some of the worst literal lookup shapes.

## Example Strengths

### Good: legal effect / condition / consequence

- `31985R0223_el`
  - Q: `Πώς επηρεάζει ο κανονισμός την ανάγκη μεταφοράς των όρων της συμφωνίας αλιείας στο εθνικό δίκαιο των κρατών μελών;`
  - Why it is good: The accepted retry asks for legal effect instead of repeating one clause.

- `31999R2771_sl`
  - Q: `Kateri pogoj povzroči izgubo pravice do pomoči za zamrznjeno smetano, kadar po zamrznitvi ni mogoče natančno preveriti vsebnosti maščobe?`
  - Why it is good: It isolates one trigger and one consequence instead of asking for a list of obligations.

- `22006D0646_et`
  - Q: `Kas eksportiva riigi poolt dokumenti valideerides tekib impordil tollivõlg?`
  - Why it is good: The final version focuses on the legal effect of validation rather than the quoted sentence.

- `31998L0008_lt`
  - Q: `Kam priskiriama pareiga padengti direktyvos taikymo tvarkų išlaidas ... ?`
  - Why it is good: It asks who bears the legal burden, not just where the implementation-cost clause appears.

## What Still Looks Weak

- The main remaining weakness is now **accepted condition-list / inventory-shaped legal questions**, not article-label phrasing.
- Some accepted questions still ask for **a bundle of conditions, limits, documents, or included items** instead of one narrower legal point.
- Some still behave like **safe procedural extraction** rather than the strongest semantic retrieval question available.
- The latest prompt + checker pass materially improved output quality, but it did not fully solve accepted condition-inventory shape.

## Example Weaknesses

### Weak: accepted condition / inventory shape

- `31999R0111_cs`
  - Why it is weak: It still asks what conditions a legal entity must satisfy, so the answer becomes a compact checklist.

- `32003R0953_lt`
  - Why it is weak: It still asks what conditions must be met and allows a multi-condition answer with a fallback branch.

- `32005R1307_sv`
  - Why it is weak: It asks which documents must be sent, so the answer is an inventory of required items.

### Weak: accepted procedural / limit lookup

- `32003R0415_da`
  - Why it is weak: It asks for entry and validity limits together, so the answer is still a rule list.

- `31997R2382_it`
  - Why it is weak: It remains close to a form-field instruction lookup even after retry.

### Weak: skipped after repeated rejection

- `52005BP0097_en`
  - Why it is important: The checker correctly kept rejecting it as overly extractive, but the run still lost the row after three attempts, so yield is not fully stable yet.

## Current Quality Summary

- Domain fit: clearly improved relative to the earlier chemistry-style setup
- Language correctness: good in the reviewed sample
- Faithfulness: generally good
- Article-label avoidance: strong and no longer the main weakness
- Yield stability: better than the previous failed `21`-row review, but not full
- Retrieval usefulness: better than the previous reviewed run
- Main remaining issue: some accepted questions are still condition-list, inventory, or procedural-limit shaped even after retries

Practical judgment:

- The current JRC questions are **usable and better than the previous reviewed run**, especially because the new legal-shape checker now visibly catches some structural failures.
- The run is operationally stronger because `17` rows were repaired after visible rejection and only `1` row was finally lost.
- The remaining issue is that too many accepted rows are still condition-list, inventory, or procedural-limit shaped.
- Current score: `7.3 / 10`

## Recommended Next Improvements

1. Reject more accepted condition-list and inventory-style questions when one narrower legal point is available.
2. Reject more document/item-list questions whose answer naturally becomes an inventory.
3. Reject more procedural-limit questions that are correct but still broader than the strongest available legal-effect query.
4. Keep the current pressure against exact value/code/period lookups; that part improved and should remain in place.
5. Prefer questions about:
   - what triggers a consequence
   - what prevents approval
   - what a missing document/report changes
   - who is exempt or affected
   - when a derogation applies
   - what legal effect follows
6. Penalize questions whose answer naturally becomes a semicolon-separated inventory or a pair of parallel limits.

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

### Review 8

- Result: Better than Review 7 overall, but not yet clean.
- Main improvement: the dedicated legal-shape checker is now visibly firing in logs, `17` rows were repaired after retries, and the sample expanded to `43/44` rows across `20` languages.
- Main remaining issue: some accepted rows are still condition-list, document-inventory, or procedural-limit questions, so the new checker is helping but not yet catching the whole residual failure class.
