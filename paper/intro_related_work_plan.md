# Introduction and Related Work Combination Plan

## Goal

Combine the current Introduction and Related Work into one compact section:

```latex
\section{Introduction and Related Work}
```

The goal is not to introduce new claims or add new literature. The goal is to
reuse the material already present in `01_introduction.tex` and
`02_related_work.tex`, reduce repetition, and make the first body section more
efficient for a short ACL/EMNLP-style paper.

## Standard Approach

For a short paper, combining Introduction and Related Work is acceptable when the
paper needs to save space and the related work can be used to motivate the
problem rather than appear as a separate survey. The combined section should not
read like two sections pasted together. It should read as one argument:

1. State the practical problem.
2. Explain why existing evaluation is insufficient.
3. Position the closest benchmark and retrieval literature.
4. Introduce the paper's benchmark suite and contributions.

In this format, related work citations should appear where they support the
motivation. We should avoid long literature inventory paragraphs. Each citation
group should answer a specific question:

- What existing benchmark/evaluation line is relevant?
- What does it already cover?
- What gap remains for multilingual chemistry retrieval?

## Target Story

The section should tell this story:

Technical chemistry search crosses language boundaries, but multilingual
retrieval evaluation often hides the failure that matters most: whether the
system retrieves the right evidence across languages rather than an easier
same-language match. Existing multilingual retrieval benchmarks and
chemistry-specific benchmarks are valuable, but they do not jointly provide
controlled multilingual chemistry retrieval data with language-aware relevance
judgments and chemically plausible hard negatives. Patent publication variants
give a practical source of comparable multilingual technical evidence, and
chemical ontologies/aliases make it possible to test chemistry-specific
confusability. X-ChemIR fills this evaluation gap with two patent-grounded QAC
retrieval releases, an auxiliary alias-graph stress test, and diagnostics that
connect benchmark results to deployment decisions.

## Proposed Paragraph Structure

### Paragraph 1: Practical Problem and Stakes

Purpose: open with the user/workflow problem.

Use material from `01_introduction.tex`, especially lines 86--90.

What it should say:

- Chemistry search is multilingual in practice.
- A query may be in one language while useful evidence is in another.
- This is not only translation: chemistry has aliases, abbreviations, formulas,
  compound families, process descriptions, and near-neighbor concepts.
- In high-trust workflows such as patent search, RAG, and decision support,
  cross-language failure and chemical confusion are practical risks.

Avoid:

- Overexplaining translation costs here.
- Listing too many languages.
- Saying too much about benchmark construction before the problem is clear.

Possible shape:

```latex
Technical chemistry search rarely respects a single language boundary: a user
may query in one language while the strongest evidence appears in another.
The risk is not only mistranslation; chemistry retrieval must also distinguish
aliases, compound families, and near-neighbor concepts that are semantically
close but wrong for the question. In high-trust workflows such as patent search,
RAG, and technical decision support, a plausible but chemically wrong document is
not a harmless ranking error.
```

### Paragraph 2: Why Existing Retrieval Evaluation Is Not Enough

Purpose: connect the problem to dense multilingual retrieval and aggregate
metrics.

Use material from `01_introduction.tex`, especially the dense-retrieval paragraph
and current X-ChemIR intro paragraph.

What it should say:

- Dense multilingual retrieval is attractive because it promises one index across
  languages.
- Prior cross-lingual retrieval and multilingual RAG work shows language
  alignment, translation effects, and same-language preference matter.
- Aggregate Recall@k can hide cross-language failure.
- The right evaluation should separate same-language evidence, cross-language
  evidence, and chemical confusability.

Relevant citations already available:

- Dense retrieval: `\citep{reimers2019sentencebert,karpukhin2020dpr}`
- Multilingual encoders: `\citep{labse2022,bgem3_2024}`
- Cross-lingual behavior: `\citep{whatdrivesclir2025,crosslingualcost2025,bordirlines2024,xrag2025,nepotism2025}`

Avoid:

- Repeating the full results here.
- Explaining every metric (`LT`, `k^\star`, `XRC`, `RRC`, `ARI`) before the
  evaluation section.

Possible shape:

```latex
Dense retrieval with neural sentence and passage embeddings is attractive in
this setting because multilingual encoders promise a single retrieval index
across languages ... Yet prior work on cross-lingual retrieval and multilingual
RAG shows that retrieval behavior depends on language alignment, translation
effects, and same-language preference ... Aggregate Recall@k can therefore
overstate readiness when a model succeeds on easier same-language matches but
misses cross-language evidence or chemically confusable cases.
```

### Paragraph 3: Benchmark Landscape and Missing Combination

Purpose: merge the strongest Related Work positioning into the introduction.

Use material from `02_related_work.tex`, especially lines 3--14 and 35--40.

What it should say:

- General embedding and multilingual retrieval benchmarks exist: MTEB, MMTEB,
  MIRACL, NeuCLIR, CLIRMatrix.
- Chemistry-specific resources exist: ChemTEB, ChemLit-QA, ChemComp,
  ChemKGMultiHopQA, ChEmbed.
- These lines of work do not jointly provide multilingual chemistry retrieval
  data with controlled technical variants, explicit same/cross-language
  relevance judgments, and chemically plausible hard negatives.
- ChEBI/alias resources and CEAR establish useful chemistry structure, but the
  paper uses this structure for retrieval evaluation rather than entity/role
  extraction.

Relevant citations already available:

- General/multilingual retrieval benchmarks:
  `\citep{miracl2023,neuclir2023,muennighoff2023mteb,mmteb2025,clirmatrix2020}`
- Chemistry benchmarks:
  `\citep{chemteb2024,chemlitqa2024,khodadad2026chemcomp,astaraki2026iterativerag,chembed2025}`
- Ontology resources:
  `\citep{chebi2016,cear2024}`

Avoid:

- A full separate survey tone.
- Repeating the entire contribution list.
- Saying prior work is inadequate overall. It is not; it covers adjacent needs.

Possible shape:

```latex
Existing benchmark lines cover important parts of this space. General embedding
and multilingual retrieval benchmarks ... enable systematic comparison across
models and languages, while chemistry-specific resources ... extend evaluation
to chemical embeddings, RAG, and scientific reasoning. What remains missing is
their combination: multilingual chemistry retrieval data with controlled
technical variants, explicit same-language and cross-language relevance
judgments, and chemically plausible hard negatives. Chemical ontology and alias
resources such as ChEBI make such hard negatives possible, complementing
ontology-grounded extraction work such as CEAR.
```

### Paragraph 4: Why Patents Are the Substrate

Purpose: explain why patent data appears without making the paper sound like it
is only about patent search.

Use material from `01_introduction.tex` lines 92--97 and `02_related_work.tex`
lines 27--33.

What it should say:

- Patent retrieval has a long evaluation history.
- Patent representation/embedding work remains active.
- Patent publications often contain related technical disclosures across
  languages.
- Therefore patents are useful as controlled multilingual technical evidence, but
  the broader task remains multilingual chemistry retrieval.

Relevant citations already available:

- Patent retrieval: `\citep{prime2002,clefip2013}`
- Patent embeddings: `\citep{paecter2024,patentembeddings2026}`

Avoid:

- Overusing “patent” in the first paragraph.
- Saying the paper is “not patent search alone” too many times.
- Introducing detailed corpus statistics here.

Possible shape:

```latex
Patents are a practical substrate for this evaluation. Patent retrieval has a
long evaluation history ..., and recent work has revisited patent representation
learning and embedding evaluation .... Because patent publications often contain
related technical disclosures across languages, they provide comparable
multilingual technical evidence beyond prior-art search alone.
```

### Paragraph 5: What This Paper Contributes

Purpose: end the combined section with a concise contribution paragraph.

Use material from `01_introduction.tex` lines 105--112, but make it cleaner and
less crowded.

What it should say:

- The paper presents X-ChemIR.
- It contains two patent-grounded multilingual chemistry QAC retrieval releases:
  Google Patents and EPO.
- It includes an LLM-assisted QAC generation and human-audit pipeline.
- It adds language-aware retrieval diagnostics and an auxiliary alias-graph
  stress test.
- It evaluates eight multilingual embedding models and connects results to
  deployment decisions.

Avoid:

- Listing every metric in the introduction. The metric names can appear later in
  Evaluation/Results.
- Packing too many parentheticals into one sentence.
- Repeating the numerical findings already stated in the abstract.

Possible shape:

```latex
We present X-ChemIR, a patent-grounded benchmark suite for cross-lingual
chemistry retrieval. The suite includes two public QAC retrieval releases derived
from Google Patents and EPO data, a reproducible LLM-assisted QAC generation and
human-audit pipeline, language-aware retrieval diagnostics that separate
same-language from cross-language evidence, and an auxiliary alias-graph stress
test for chemistry-confusability. We evaluate eight off-the-shelf multilingual
embedding models and use the results to show why aggregate recall is not a
sufficient deployment dashboard.
```

## Recommended Final Length

The combined section should be about 5 concise paragraphs.

Target length:

- Around 650--800 words if page budget is comfortable.
- Around 500--650 words if page budget is tight.

Because this paper is currently space-constrained, the first implementation
should aim for the shorter side: roughly 5 paragraphs, each 4--6 lines in the
compiled ACL format.

## What To Cut From the Original Sections

Cut or avoid:

- Repeated statements that aggregate recall hides cross-language failure.
- Repeated explanations that patents are only the substrate, not the whole task.
- A long standalone literature survey.
- Detailed metric names in the contribution paragraph.
- Dataset statistics, which belong in the benchmark/evaluation sections.
- Full result numbers in the introduction, since they already appear in the
  abstract and Results.

Keep:

- The practical multilingual chemistry retrieval problem.
- The chemistry-specific risk of plausible but wrong near-neighbor documents.
- The gap in existing multilingual/chemistry benchmarks.
- The reason patents are useful.
- The high-level contribution list.

## Implementation Checklist

1. Copy no text mechanically. Rewrite into one flow.
2. Keep citations attached to the claim they support.
3. Use “benchmark suite” for X-ChemIR as a whole.
4. Use “benchmark releases” when naming Google Patents and EPO individually.
5. Use “auxiliary alias-graph stress test” for the chemistry-confusability
   diagnostic.
6. Avoid “we do X” in literature-positioning paragraphs except in the final
   contribution paragraph.
7. After drafting, compile and check whether the combined section saves space
   relative to separate Introduction + Related Work.
