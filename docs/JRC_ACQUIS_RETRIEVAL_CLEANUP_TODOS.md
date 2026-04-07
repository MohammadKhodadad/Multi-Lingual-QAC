# JRC-Acquis Retrieval Cleanup TODOs

This note turns the current JRC corpus review into an implementation backlog.

Goal:
- improve the quality of the JRC retrieval corpus before the next QA-generation round
- reduce low-value legal acts such as pure amendments, corrigenda, and boilerplate-heavy records
- keep the benchmark focused on documents that support stronger retrieval questions
- keep the accepted QA set focused on sharp single-point legal retrieval questions

## Why This Matters

Right now the JRC corpus is usable, but some records are still weaker than we want for retrieval:

- some acts are mainly amendments or procedural updates rather than substantive legal content
- some records are short or too narrow to support strong legal questions
- some documents contain repetitive structure that makes retrieval easier but less meaningful
- whole-document relevance is still coarse, so low-value documents hurt benchmark quality more than they would in chunk-level retrieval

Because of that, it is worth filtering document quality before we sample QA pairs.

## Priority Order

Work these in order:

1. Tune the current multilingual substance thresholds so the QA pool is less skewed by language.
2. Expand JRC relevance mapping from one sampled pair to CELEX-group multilingual relevance sets.
3. Tighten the legal QA checker against accepted condition-list / inventory / procedural-limit question shapes.
4. Add manual accepted vs rejected inspection samples.
5. Add amendment-heavy detection that works across many languages.
6. Decide whether amendment filtering should affect only QA candidates or the broader retrieval corpus.
7. Rebuild and manually inspect examples before the next QA run.

## Already Completed

These cleanup steps are now done:

- dropped `jrcHeader-*` style helper/header pseudo-documents from the corpus
- stopped prepending `header_notes` into retrieval `context`
- added a structured retrieval-text cap so body text dominates over annex/signature tail text
- expanded reference-line cleanup to remove bracketed citation/editorial notes such as `[1] OJ ...`
- tightened the QA-candidate filter using multilingual-safe paragraph/substance heuristics
- added `qa_rejection_reasons` reporting to `document_corpus_stats.json`
- updated the language-pair table to the QA-based directional matrix used for dataset building
- added per-attempt QA rejection logging so retry reasons are visible during JRC generation
- tightened the legal QA generation + quality-check prompts against exact value/code lookup and multi-clause legal questions
- added a dedicated legal-shape checker for broad legal shape, condition-list, menu-of-measures, and definition-inventory failures
- added `corpus_language` and `question_language` columns to the Hugging Face export path
- removed dead same-language prompt branches from `openai_qa.py` so the JRC legal path is explicit

What this means:

- retrieval corpus quality is clearly better than before
- the main remaining issue is now uneven QA filtering across languages plus accepted condition-list / inventory / procedural-limit QA questions, not raw header/boilerplate leakage

## TODOs

### 1. Add title-based exclusion rules for weak act types

TODO:
- add a filter for titles that clearly indicate low-value legal acts

Examples to target:
- amendment / amending / modifying acts
- corrigendum / correction / rectification
- repeal-only or replacement-heavy acts
- purely technical-adaptation acts when they mostly rewrite references rather than introduce substantive rules

Why:
- these documents often generate narrow lookup questions instead of stronger legal retrieval questions
- they are frequently dominated by "replace X with Y" language

Implementation idea:
- add a helper such as `_looks_like_low_value_title(title, lang)`
- run it during document build or QA-candidate selection

### 2. Add body-level amendment-density detection

TODO:
- detect documents whose body is dominated by amendment language even when the title is not explicit

Patterns to look for:
- "is amended as follows"
- "shall be replaced by"
- "is replaced by"
- "paragraph X is deleted"
- "Annex ... is replaced"
- repeated article/paragraph substitution formulas

Why:
- title-only filtering will miss many weak records
- some documents look normal in the title but are mostly patch instructions in the body

Implementation idea:
- count amendment-style lines or phrases in body paragraphs
- drop records when the amendment density crosses a conservative threshold

### 3. Add a document substance score

TODO:
- score documents for how much real retrieval-worthy content they contain

Signals to reward:
- enough meaningful body paragraphs
- continuous prose, not just short edit instructions
- presence of recitals, obligations, definitions, exceptions, conditions, or operative rules
- richer operative sections beyond one or two short articles

Signals to penalize:
- many very short paragraphs
- mostly citations or replacement formulas
- very small operative body
- overly repetitive structure

Why:
- a single yes/no filter is too blunt
- a score gives us a better way to keep borderline but still useful acts

Implementation idea:
- add a helper such as `_score_document_substance(...)`
- use the score to decide QA eligibility

### 4. Tune the QA-candidate floor

Status:
- initial multilingual-safe tightening is already implemented

Current floor:
- minimum chars: `1500`
- minimum body paragraphs: `8`
- minimum operative chars: `1200`
- minimum medium operative paragraphs: `5`
- max short operative ratio: `0.55`

Current issue:
- retention is now uneven across languages, so the next step is tuning rather than simply making the filter harsher

Why:
- some documents are now filtered for the right reason, but the thresholds still appear to interact with language formatting differences

Implementation idea:
- tune the current thresholds using per-language retention review
- keep the retrieval corpus broader if needed, but make the QA pool stricter and more balanced

### 5. Exclude repeated boilerplate from quality scoring

Status:
- retrieval-text leakage from `header_notes` is already fixed
- bracketed editorial citation lines are now filtered during paragraph cleaning

TODO:
- ensure any remaining repeated boilerplate does not help a document pass quality filters

Why:
- we already removed note boilerplate from retrieval `context`, but similar repeated structure should also not count as evidence that a document is substantive

Implementation idea:
- base quality scoring on cleaned body/operative paragraphs, not on metadata or notes

### 6. Add explicit reporting for filtered document classes

Status:
- QA rejection reasons are already reported in `document_corpus_stats.json`

TODO:
- expand reporting to cover more exclusion classes when new filters are added

Useful counters:
- dropped as header/helper
- dropped by title rule
- dropped by amendment-density rule
- dropped by low-substance score
- kept for retrieval but excluded from QA

Why:
- otherwise we will not know whether the new rules are doing useful work or being too aggressive

Implementation idea:
- add fields to `document_corpus_stats.json`
- include a short inspection sample per exclusion type if feasible

### 7. Add manual inspection samples for accepted vs rejected records

TODO:
- export small CSV samples of:
- accepted high-quality documents
- rejected amendment-heavy documents
- rejected corrigenda/rectifications
- rejected low-substance documents

Why:
- this gives us a quick sanity check before another full QA run
- it is easier to tune heuristics from examples than from counters alone

Implementation idea:
- add one inspection file under `data/JRC-ACQUIS/preprocessed/`
- this is now one of the highest-priority remaining tasks because the filter is cleaner but still uneven

### 8. Tighten accepted QA-shape filtering

TODO:
- reject more accepted questions whose best answer naturally becomes:
- a list of conditions
- a list of documents or required items
- a pair of validity / limit rules
- a form-field instruction copied from one clause

Why:
- the newest `43/44` review shows that the legal-shape checker is helping, but many of these rows still pass
- this is now a more important bottleneck than raw corpus dirt

Implementation idea:
- add stricter legal examples and repair hints to the checker for:
- `what conditions must be met`
- `which documents/handlingar/items must be sent`
- `what limits apply`
- `how must X be indicated`
- treat list-shaped answers as a stronger reject signal even when the question is otherwise grounded

### 9. Expand pair-level relevance to CELEX-group multilingual relevance

TODO:
- stop limiting each generated JRC query to only the two documents in the sampled pair
- map each generated query to all cleaned retained translations of the same `celex`
- support at least two explicit modes:
- sampled-group relevance: all translations of the same `celex` present in the sampled benchmark corpus
- full-group relevance: all cleaned retained translations of the same `celex` present in the broader cleaned corpus
- add reporting so we can see how many relevant documents each query gets under each mode

Why:
- the current export already shows that many sampled `celex` acts have more than two retained translations in the sampled corpus
- in the broader cleaned corpus, many sampled acts have roughly `19-22` translations available
- limiting qrels to a single pair undercounts correct cross-lingual retrieval and makes the supervision more arbitrary than the corpus alignment actually supports
- the benchmark unit is really a multilingual legal act, not just one bilingual pair

Implementation idea:
- keep `corpus` rows document-language specific; do not merge several translations into one corpus row
- group cleaned retained corpus rows by `celex`
- during JRC export, replace `linked_corpus_ids_json = [source, target]` with a CELEX-driven relevant-document set
- keep the sampled pair for generation provenance, but decouple relevance mapping from that pair
- add corpus metadata such as `celex_group_size` or `document_group_id` if that helps downstream inspection
- compare metrics between:
- pair-only qrels
- sampled-group qrels
- full-group qrels
- document which mode becomes the default and why

### 10. Re-review retrieval and QA quality after rebuild

TODO:
- after implementing the filters, rebuild and review the resulting corpus again

Review questions:
- are top-of-document previews now mostly substantive?
- do sampled documents support stronger legal questions?
- are we removing too many multilingual documents?
- are any languages disproportionately harmed by the new heuristics?
- are accepted condition-list, inventory, and procedural-limit legal questions still the main remaining failure pattern after the new checker/generation tightening?

Why:
- legal corpora are language-skewed, so a rule that looks good globally may still hurt some languages

## Suggested Next Work Package

Start with this bundle next:

1. tune the current QA thresholds using per-language retention
2. expand JRC qrels from sampled pairs to CELEX-group multilingual relevance sets
3. tighten the legal QA checker against accepted condition-list / inventory / procedural-limit questions
4. add accepted vs rejected inspection samples
5. add conservative amendment-density body detection
6. expand stats counters for the new exclusion reasons
7. review whether the remaining weak QA rows come from weak source documents or from still-fixable condition-list generation

That should give the biggest quality gain with the least risk of overfitting to English-only patterns.

## Success Criteria

We should consider this cleanup successful if the next build shows:

- fewer amendment-heavy acts in the QA candidate pool
- better manual inspection samples
- fewer short or procedural-only QA sources
- stronger question quality in the next JRC QA run
- fewer accepted condition-list / inventory / procedural-limit questions after retries
- more stable multilingual coverage despite stricter filtering
- more than two relevant cleaned documents for many JRC queries where the same `celex` is available in multiple retained languages
- a documented, auditable default relevance mode for JRC (`pair`, `sampled-group`, or `full-group`)
