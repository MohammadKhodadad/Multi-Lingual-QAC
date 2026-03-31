# QAC Quality Notes

Living notes for reviewing the current Question-Answer-Context output.

This file now tracks the **JRC-Acquis legal/regulatory QA pipeline**, not the older chemistry/patent pipeline.

## Current Snapshot

- Source file reviewed: `data/JRC-ACQUIS/qac/qac.csv`
- Current reviewed run: `44` rows
- Current generation design: **pair-level**, same-language generation from the translated/target side of each sampled pair
- Current linking design: one generated query is attached to **both** documents in the selected pair
- Current domain: legal / regulatory EU documents
- Current validation: language match + faithfulness + legal-domain question-quality checks

Quick summary from the latest reviewed run:

- Languages present: `bg`, `cs`, `da`, `de`, `el`, `en`, `es`, `et`, `fr`, `hu`, `it`, `lt`, `lv`, `mt`, `nl`, `pt`, `sk`, `sl`, `sv`
- Median question length: `149`
- 90th percentile question length: `232`
- Median answer length: `168`
- 90th percentile answer length: `328`

## Method

Current JRC review method:

1. Run `uv run main.py --source JRC-ACQUIS`.
2. Review `data/JRC-ACQUIS/qac/qac.csv`.
3. Check broad signals:
   - language distribution
   - question/answer length
   - explicit article-number / label usage
   - generic or checklist-shaped questions
4. Read a manual sample of generated rows.
5. Classify problems into:
   - overly generic
   - article/provision lookup
   - bundled legal conditions
   - deadline/list/certificate lookup
   - overly long question shape

## What Looks Good

- The current questions are much better aligned with **legal/regulatory** text than the earlier chemistry-oriented prompts.
- The earlier obvious `According to Article ...` / label-driven failures are largely gone in the reviewed sample.
- The questions are now usually generated in the correct target-side language and the answer is generally grounded in the target-side document.
- Many questions now ask about:
  - operative meaning
  - legal effect
  - who may act
  - what happens if a condition is or is not met
  - what a rule permits or prevents
- Several examples are strong semantic legal queries rather than clause lookup prompts.

## Example Strengths

### Good: legal effect / authority action / consequence

- `32004D0009__de__en`
  - Q: `What labour and social law matters is the Committee explicitly barred from addressing?`
  - Why it is good: Narrow, document-specific, and clearly about scope/limitation rather than label lookup.

- `32005B0532__de__et`
  - Q: `Mida peab president tegema pärast otsuse vastuvõtmist seoses otsuse ja kaasneva resolutsiooni edastamise ja avaldamisega?`
  - Why it is good: Concrete procedural effect; answerable, grounded, and not generic.

- `31997L0027__en__nl`
  - Q: `Mag een lidstaat een verzoek van de fabrikant om toepassing van de in bijlage IV beschreven procedure weigeren ... ?`
  - Why it is good: Focuses on one operational legal issue and yields a concise yes/no rule with conditions.

- `32005D0880__nl__sk`
  - Q: `Wat gebeurt er als uit de administratieve controle van de jaarlijkse derogatiemelding blijkt dat niet aan de gestelde voorwaarden wordt voldaan?`
  - Why it is good: Good consequence-oriented question; requires understanding what the failed control changes.

- `32006E0418__da__sl`
  - Q: `Hvad skal fremgå af den specifikke finansieringsaftale ... vedrørende synligheden af EU's bidrag?`
  - Why it is good: Focused, grounded, and clearly tied to one identifiable obligation.

## What Still Looks Weak

- Some questions still ask for **too many conditions at once**.
- Some still behave like **literal compliance extraction** rather than strong semantic retrieval.
- Some still ask directly for a **date, amount, threshold, or complete inventory** when a better legal-effect question seems possible.
- A few questions are still long enough that they feel closer to an answer-shaped checklist than a clean query.

## Example Weaknesses

### Weak: bundled conditions / checklist shape

- `31998L0038__bg__es`
  - Why it is weak: The question asks what Member States may and may not do from a given date. It is grounded, but still broad and partly deadline-driven.

- `31999R1215__el__es`
  - Why it is weak: It asks for the conditions under which an authority may withdraw a benefit. This is relevant, but still fairly checklist-like.

- `32001R1093__cs__el`
  - Why it is weak: It asks for multiple possible operations under one certification condition, which makes the answer list-like and easier to solve by extraction.

- `31993R2131__fr__sv`
  - Why it is weak: It asks which tenders are exempt from both publication and the waiting period. This is narrower than before, but still shaped like an exception inventory.

### Weak: date / amount / threshold lookup

- `31998R2305__en__es`
  - Why it is weak: It asks directly when sub-quotas must be adjusted after a quota is exceeded. This is valid, but fairly close to deadline lookup.

- `32003D0560__es__nl`
  - Why it is weak: It asks when the withdrawal takes effect. This is concise, but mostly a date/effective-time lookup.

- `31992L0006__et__pt`
  - Why it is weak: It asks for a specific maximum speed value. The number may matter, but it still leans extractive.

- `31995R2417__el__sl`
  - Why it is weak: It asks what amount replaces another amount. This is a very literal numeric lookup.

## Current Quality Summary

- Domain fit: clearly improved
- Language correctness: good in the reviewed sample
- Faithfulness: generally good
- Article-label avoidance: much better than before
- Retrieval usefulness: improved, but still mixed
- Main remaining issue: some questions are still too broad, too list-like, or too literal

Practical judgment:

- The current JRC questions are **usable and clearly better than the earlier runs**.
- They are not yet consistently at the level of narrow, high-value semantic legal retrieval questions.

## Recommended Next Improvements

1. Reject more aggressively when a question asks for `all conditions`, `all guarantees`, `all exceptions`, or `all consequences`.
2. Reject more deadline-only and amount-only questions unless the number itself is truly the key retrieval target.
3. Prefer questions about:
   - what triggers a consequence
   - what prevents approval
   - what a missing document/report changes
   - who is exempt or affected
   - when a derogation applies
   - what legal effect follows
4. Penalize questions whose answer naturally becomes a semicolon-separated inventory.
5. Keep monitoring question length so long multi-clause questions do not dominate.

## Review Log

### Review 1

- Result: Initial JRC same-language sample was usable but too citation-heavy and too close to legal lookup.
- Main improvement needed: remove article-label dependence and make questions more semantic.

### Review 2

- Result: Better than Review 1.
- Main improvement: article/provision-label style was reduced significantly.
- Main remaining issue: some questions were still too broad, checklist-like, or deadline/list-oriented.

### Review 3

- Result: Better than Review 2 and the current best JRC run so far.
- Main improvement: stronger avoidance of article-number phrasing and better legal-domain fit.
- Main remaining issue: some questions are still too broad or too extractive, especially around full conditions, exemptions, deadlines, or numeric replacements.
