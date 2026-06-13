# Related Work Improvement Plan

## Goal

Make the Related Work section do more than list adjacent benchmarks. It should
quickly establish the research landscape, show the exact gap this paper fills,
and prepare the reader for the paper's three core ideas:

- public multilingual chemistry retrieval evaluation data is largely missing,
  even though both multilingual retrieval and chemistry NLP benchmarks exist;
- multilingual retrieval evaluation needs language-aware diagnostics;
- chemistry retrieval needs domain-specific relevance and confusability checks;
- LLM-generated benchmark data needs provenance, filtering, and human audit.

The section is already broadly correct. The main opportunity is to make the
argument sharper and more connected to the paper's contribution.

## Current Strengths

- It covers the right major areas: general embedding benchmarks, multilingual
  retrieval benchmarks, chemistry-specific benchmarks, cross-lingual retrieval
  bias, patent retrieval, and chemical ontology resources.
- It correctly distinguishes MTEB from MMTEB and separates general embedding
  evaluation from multilingual/cross-lingual retrieval resources.
- It now positions the paper around multilingual chemistry retrieval rather than
  generic chemistry QA or patent search.
- It avoids overclaiming that existing chemistry benchmarks are only
  single-language QA.

## Main Weaknesses

- The first paragraph is dense and citation-heavy. It names many resources but
  could make the gap clearer earlier.
- The central missing-data claim should be more explicit: existing work gives us
  multilingual retrieval datasets in general domains and chemistry benchmarks in
  mostly non-multilingual settings, but not a clean public multilingual chemistry
  retrieval dataset with query/document language metadata and cross-language
  qrels.
- The section does not yet discuss LLM-assisted benchmark construction, even
  though the paper relies on an LLM-assisted QAC generation and scoring pipeline.
- The chemistry paragraph is useful, but it can more explicitly say what prior
  chemistry benchmarks do not provide: controlled multilingual variants,
  query/document language metadata, cross-language qrels, and chemistry-aware
  hard negatives.
- The patent paragraph comes late and could be tied more directly to why patents
  are a good benchmark substrate: multilingual publication variants,
  technical descriptions, and a long evaluation history.
- The section could end with a stronger synthesis sentence that makes the paper's
  niche obvious.

## Proposed Structure

### 1. Multilingual Retrieval and Embedding Benchmarks

Purpose: establish that retrieval benchmarks exist, but not for this exact
setting.

Keep citations to MTEB, MMTEB, MIRACL, NeuCLIR, and CLIRMatrix. Sharpen the
contrast:

- MTEB/MMTEB support broad embedding comparison.
- MIRACL/NeuCLIR/CLIRMatrix support multilingual or cross-lingual retrieval.
- None are multilingual chemistry retrieval datasets: they do not provide
  chemistry-specific relevance, controlled patent variants, query/document
  language metadata for chemistry evidence, or chemically confusable negatives.

Suggested direction:

> Existing multilingual retrieval benchmarks make model comparison possible
> across languages, but they generally do not ask whether a retriever can find
> chemistry-specific evidence across controlled technical variants while
> avoiding plausible chemical look-alikes.

Stronger missing-data version:

> In other words, multilingual retrieval benchmarks exist, and chemistry
> benchmarks exist, but clean public multilingual chemistry retrieval data remains
> missing.

### 2. Chemistry-Specific NLP and Retrieval Benchmarks

Purpose: show that chemistry NLP has benchmarks, but the multilingual retrieval
gap remains.

Keep ChemTEB, ChemLit-QA, ChemComp, ChemKGMultiHopQA, and ChEmbed. Make the
distinction cleaner:

- ChemTEB and ChEmbed: chemistry embedding/retrieval evaluation.
- ChemLit-QA: expert-validated chemistry QAC/RAG data.
- ChemComp and ChemKGMultiHopQA: chemistry reasoning and retrieval-aware
  scientific QA.
- Our benchmark suite: multilingual chemistry retrieval with source-controlled
  language variants and query/document language metadata.
- Core gap: these resources are valuable chemistry evaluations, but they do not
  directly fill the missing public multilingual chemistry retrieval dataset
  niche.

Suggested direction:

> These resources establish the need for chemistry-aware evaluation, but they do
> not provide two patent-derived multilingual retrieval releases with explicit
> same-language and cross-language relevance judgments.

More direct version:

> Thus, the gap is not a lack of chemistry NLP benchmarks in general; it is the
> lack of public multilingual chemistry retrieval data that lets researchers
> measure whether evidence can be found across languages.

### 3. LLM-Assisted Benchmark Construction

Purpose: justify the QAC generation/scoring/audit pipeline as a benchmark
construction method, not just an implementation detail.

This should be short because page budget is tight. Add one or two sentences,
probably after the chemistry benchmark paragraph.

Suggested wording:

> LLM-assisted benchmark construction is increasingly useful for creating
> domain-specific evaluation data, but generated items require provenance,
> filtering, and human audit before they can support reliable conclusions. Our
> pipeline follows this direction by keeping each generated QAC traceable to a
> source document, language, scorer output, and retrieval relevance judgment.

Before adding this, we should confirm whether `custom.bib` already has a good
citation for LLM-assisted benchmark construction. If not, the sentence can be
written without adding a new citation, or we can add one carefully if space and
accuracy allow.

### 4. Cross-Lingual Retrieval Bias and Same-Language Preference

Purpose: motivate why the evaluation separates same-language and cross-language
gold evidence.

Keep the current paragraph, but tighten it around the paper's diagnostic:

- Prior work shows multilingual retrieval behavior varies by language and can
  prefer same-language matches.
- Our controlled patent variants let us expose this failure mode cleanly.
- This connects directly to the Results heatmaps and home-advantage figure.

Suggested direction:

> This motivates reporting not only aggregate Recall@10, but also whether a model
> retrieves same-language evidence, cross-language evidence, or the right
> evidence when no same-language gold document is available.

### 5. Patent Retrieval and Chemical Hard Negatives

Purpose: justify the use of patents and the auxiliary alias-graph stress test.

Keep NTCIR, CLEF-IP, patent embeddings, ChEBI, and CEAR. Improve the transition:

- Patent retrieval has a long evaluation history.
- Patent publications are useful here because they provide multilingual
  technical variants at scale.
- Chemical ontologies and aliases let us define plausible but wrong neighbors.

Suggested direction:

> We build on this patent retrieval history, but use patents as a controlled
> multilingual substrate for chemistry retrieval evaluation rather than as a
> generic prior-art search benchmark.

## Recommended Revision Strategy

1. Keep the section short: around four paragraphs is enough.
2. Do not add many new citations unless they directly support LLM-assisted
   benchmark construction.
3. Reduce citation-list feel by making each paragraph end with a gap statement.
4. Use the same terminology as the rest of the paper:
   - `benchmark suite` for both releases together;
   - `benchmark releases` for Google Patents and EPO individually;
   - `human audit` for the 97-item reviewed sample;
   - `auxiliary alias-graph stress test` for the chemistry hard-negative
     diagnostic.
5. Avoid claiming that the paper is the first overall chemistry retrieval
   benchmark. The safer claim is that it contributes two public multilingual
   chemistry-patent retrieval benchmark releases with language-aware and
   chemistry-aware diagnostics.

## Concrete Edit Plan

1. Rewrite paragraph 1 to separate general embedding benchmarks from
   multilingual retrieval resources and state the missing multilingual chemistry
   retrieval dataset gap directly.
2. Rewrite paragraph 2 to group chemistry resources by role and end with the
   multilingual retrieval data gap.
3. Add a short LLM-assisted benchmark construction sentence or mini-paragraph.
4. Tighten paragraph 3 so it directly motivates same-language versus
   cross-language reporting.
5. Tighten paragraph 4 so patents are framed as the benchmark substrate and
   ChEBI/CEAR motivate the auxiliary hard-negative diagnostic.
6. End with a synthesis sentence that states the paper's exact position:

> In combination, prior work gives strong general retrieval benchmarks,
> chemistry-aware evaluation resources, and patent retrieval infrastructure; this
> paper connects them in a multilingual chemistry retrieval benchmark suite with
> auditable QAC construction, language-aware qrels, and chemistry-confusability
> diagnostics.

Alternative stronger ending:

> In combination, prior work gives strong multilingual retrieval benchmarks and
> strong chemistry-specific evaluations, but not public multilingual chemistry
> retrieval data. This paper fills that gap with two patent-derived benchmark
> releases and diagnostics that make cross-language and chemistry-confusability
> failures measurable.

## Expected Outcome

After revision, the Related Work section should feel less like background and
more like a compact argument for why this paper is needed. It should make a
reviewer think:

> The individual pieces exist, but this exact combination of multilingual
> chemistry retrieval, patent-derived controlled variants, QAC provenance, and
> chemistry-aware confusability diagnostics is new and useful.
