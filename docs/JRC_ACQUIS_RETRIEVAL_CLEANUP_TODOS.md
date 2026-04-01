# JRC-Acquis Retrieval Cleanup TODOs

This note turns the current JRC corpus review into an implementation backlog.

Goal:
- improve the quality of the JRC retrieval corpus before the next QA-generation round
- reduce low-value legal acts such as pure amendments, corrigenda, and boilerplate-heavy records
- keep the benchmark focused on documents that support stronger retrieval questions

## Why This Matters

Right now the JRC corpus is usable, but some records are still weaker than we want for retrieval:

- some acts are mainly amendments or procedural updates rather than substantive legal content
- some records are short or too narrow to support strong legal questions
- some documents contain repetitive structure that makes retrieval easier but less meaningful
- whole-document relevance is still coarse, so low-value documents hurt benchmark quality more than they would in chunk-level retrieval

Because of that, it is worth filtering document quality before we sample QA pairs.

## Priority Order

Work these in order:

1. Remove obviously weak document types.
2. Add a substance score for the remaining documents.
3. Tighten QA-candidate filtering so weak documents do not enter the QA pool.
4. Add reporting so every build tells us how much was filtered and why.
5. Rebuild and manually inspect examples before the next QA run.

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

### 4. Raise the QA-candidate floor

TODO:
- make the QA-candidate filter stricter than the current minimum

Current floor:
- minimum chars: `1500`
- minimum body paragraphs: `4`

Possible stricter floor:
- require more substantive body paragraphs
- require a minimum number of medium-length paragraphs
- require stronger operative coverage

Why:
- some documents pass the current filter but are still too weak for good question generation

Implementation idea:
- extend `_is_jrc_qa_candidate(...)`
- keep the retrieval corpus broader if needed, but make the QA pool stricter

### 5. Exclude repeated boilerplate from quality scoring

TODO:
- ensure repeated authenticity/translation boilerplate does not help a document pass quality filters

Why:
- we already removed note boilerplate from retrieval `context`, but similar repeated structure should also not count as evidence that a document is substantive

Implementation idea:
- base quality scoring on cleaned body/operative paragraphs, not on metadata or notes

### 6. Add explicit reporting for filtered document classes

TODO:
- write build stats for each exclusion reason

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

### 8. Re-review retrieval quality after rebuild

TODO:
- after implementing the filters, rebuild and review the resulting corpus again

Review questions:
- are top-of-document previews now mostly substantive?
- do sampled documents support stronger legal questions?
- are we removing too many multilingual documents?
- are any languages disproportionately harmed by the new heuristics?

Why:
- legal corpora are language-skewed, so a rule that looks good globally may still hurt some languages

## Suggested First Work Package

Start with this bundle first:

1. title-based weak-act filter
2. amendment-density body filter
3. stronger QA-candidate filter
4. stats counters for what was removed

That should give the biggest quality gain with the least complexity.

## Success Criteria

We should consider this cleanup successful if the next build shows:

- fewer amendment-heavy acts in the QA candidate pool
- better manual inspection samples
- fewer short or procedural-only QA sources
- stronger question quality in the next JRC QA run
- stable multilingual coverage despite stricter filtering
