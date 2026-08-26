# Response to reviewer: confidence intervals, significance tests, and per-route counts

**Reviewer comment.** *"The test set is small and the paper reports no confidence intervals or significance
tests for the primary Recall@10 comparisons. Per-route counts are also absent, despite substantial imbalance
in language availability."*

Everything below is produced by one reproducible, CPU-only script that re-aggregates the cached per-query
rankings (no embedding model is re-run):

```
python reports/runs/final_plots/code/claimF_significance.py
```

Outputs: `data/F_*.csv` (every number), `tables/F_*.tex` (paste-ready booktabs), `candidates/claimF_F2_tax_forest.*`
and `candidates/claimF_F4_route_counts.*` (figures). All numbers were re-derived by four independent
verification scripts written from scratch against the raw rankings; every point estimate, count, CI, p-value,
and odds ratio reconciled exactly.

---

## 1. The primary Recall@10 comparisons now carry 95% CIs and significance tests

### 1a. Headline table with CIs (`tables/F_headline_ci_GP.tex`, `..._EPO.tex`; `data/F_t1_headline_ci.csv`)
Every model's language-balanced Recall@10 and CLIR@10 now reports a **95% bias-corrected-and-accelerated (BCa)
bootstrap CI** from a **language-stratified, patent-family-clustered** bootstrap (B=10,000). Example (Google Patents):

| Model | R@10 [95% CI] | CLIR@10 [95% CI] |
|---|---|---|
| embeddinggemma | 0.573 [0.535, 0.611] | 0.541 [0.500, 0.580] |
| bge-m3 | 0.510 [0.471, 0.547] | 0.467 [0.427, 0.505] |
| e5-large-instruct† | 0.213 [0.190, 0.238] | 0.091 [0.071, 0.115] |

### 1b. The core claim — same-language vs cross-language Recall@10 — is now a paired significance test
This is the "do a statistical test" ask. For each model we test **MoLIR@10 (same-language gold) vs CLIR@10
(cross-language gold)**, paired per query on the **both-gold domain** (queries that have both a same- and a
cross-language gold: **GP n=261, EPO n=198**). We report the balanced gap Δ with a **paired cluster-bootstrap
95% CI**, a **family-level sign-flip permutation test** (Smucker et al. 2007), win/tie/loss, and a rank-biserial
effect size. (`tables/F_tax_GP.tex`, `..._EPO.tex`; `data/F_t2_tax.csv`; Fig. `claimF_F2_tax_forest`.)

Google Patents — **the cross-lingual tax is significant for all 8 models** (permutation p ≤ 1×10⁻⁴, floored;
independent Wilcoxon cross-check gives p = 10⁻⁶…10⁻²⁰):

| Model | MoLIR | CLIR | Gap Δ [95% CI] | W/T/L |
|---|---|---|---|---|
| embeddinggemma | 0.74 | 0.57 | **+0.172 [+0.125, +0.238]** | 57/200/4 |
| bge-m3 | 0.64 | 0.42 | +0.212 [+0.157, +0.275] | 72/186/3 |
| nomic-v2-moe | 0.65 | 0.35 | +0.305 [+0.238, +0.382] | 91/167/3 |
| granite-278m | 0.54 | 0.39 | +0.147 [+0.091, +0.218] | 46/204/11 |

EPO — significant for every model **except LaBSE**, which we disclose: Δ = +0.046 **[−0.006, +0.099]**,
p = 0.20 (n.s.). The CIs make this honest gradation visible instead of hiding it in a point estimate.

### 1c. Model-vs-model leaderboard significance (`data/F_t3_friedman.csv`, `data/F_t3_pairwise.csv`)
A **Friedman omnibus** (blocked by query) rejects equality of the 8 models on both metrics and benchmarks
(GP Recall@10: χ²=809, df=7, p≈10⁻¹⁷⁰). Two pre-registered pairwise families follow — **winner-vs-rest** and
**adjacent-rank** — each tested with the paired sign-flip permutation + cluster-bootstrap CI, **Holm-corrected**
within the family (full 28-pair matrix with Benjamini-Hochberg in the CSV). Result: **embeddinggemma is
significantly the best model** (beats all 7 others, Holm p=0.0007, CIs exclude 0), while some **adjacent
mid-table pairs are statistically indistinguishable** (e.g. bge-m3 vs qwen3-0.6B: +0.017 [−0.009, +0.042]).

### 1d. Global, language-adjusted tax test (`data/F_t5_global_tax.csv`)
A **cluster-robust (patent-family) binomial GLM** on gold-pair hits, `hit ~ route_type + query_language`, gives
the odds of retrieving a same-language gold document vs a cross-language one, controlling for query language.
Same-language gold documents are **1.3–3.3× more likely** to land in the top-10 (all p<0.001 for non-degenerate
models), corroborating the paired test. LaBSE-on-EPO is again the lone exception (OR=1.16, p=0.27) — internally
consistent with 1b.

---

## 2. Per-route counts (`tables/F_route_counts_GP.tex`, `..._EPO.tex`; `data/F_t4_routes.csv`; Fig. `claimF_F4_route_counts`)

A **route = (query language → document language)**. Because each patent has ≤1 version per language,
`n_gold_pairs ≡ n_queries` on every route. The count matrix makes the "substantial imbalance" explicit
(Google Patents; **diagonal = same-language, 261 pairs; off-diagonal = cross-language, 1023 pairs**):

| q ↓ / d → | EN | DE | ES | FR | ZH |
|---|---|---|---|---|---|
| **EN** | **119** | 11 | 40 | 80 | 36 |
| **DE** | 112 | **13** | 24 | 90 | 39 |
| **ES** | 103 | 23 | **27** | 78 | 31 |
| **FR** | 97 | 13 | 19 | **80** | 32 |
| **ZH** | 92 | 10 | 28 | 65 | **22** |

English and French gold columns are near-full; German/Spanish/Chinese are sparse (some routes have only 10–13
queries). This is exactly why the headline numbers are **language-balanced (macro) means, not pooled** — pooling
would let the English-dense routes dominate. The route table also reports Recall@10 with **Wilson** CIs per cell
(for embeddinggemma, e5, and the all-model mean); we deliberately report **no p-value** on any per-route or
per-language cell (see caveat 3). EPO is perfectly balanced (each patent has en/de/fr → every route = its
query-language count: en 72, de 58, fr 68).

---

## 3. Honesty caveats (all disclosed in captions/prose)

1. **Permutation p is at its resolution floor** (1/(B+1) ≈ 1×10⁻⁴, slightly conservative). Report headline
   significance as "p ≤ 10⁻⁴ (permutation)", not an exact tiny value; the 10⁻¹¹…10⁻²⁰ figures come only from the
   parametric Wilcoxon cross-check.
2. **The tax claim is scoped to the both-gold domain** (261/524 GP; 198 EPO) — queries that actually have both a
   same- and a cross-language gold. Stated explicitly wherever the gap is reported.
3. **Tiny-N languages** in the GP paired domain (de=13, zh=22, es=27) are reported with N + CI only; we run **no
   per-language / per-route significance test** (numeric backstop: no p when n<30 or #families<10 or #non-ties<10).
4. **LaBSE-on-EPO is the one non-significant tax case** — disclosed, not smoothed over.
5. **e5-large-instruct is degenerate** (CLIR ≈ 0.09, language-siloing); its large apparent gap is an artifact,
   flagged with † and greyed in the figure, but kept in the Holm burden (conservative).
6. **Winner-vs-rest is post-selection** (winner chosen by point estimate); the selection-free ordering evidence is
   the adjacent-rank family + the Friedman gate. Immaterial here given the margins.
7. **LT% differs from Table 1 by design**: here CLIR and MoLIR are both on the paired both-gold domain; the paper's
   `main_table` divides full-524 CLIR by 261-domain MoLIR. Pick one and state it.

---

## 4. Ready-to-paste "Statistical analysis" paragraph (LaTeX)

```latex
\paragraph{Statistical analysis.}
We treat the query as the sampling unit and report language-balanced (macro) means, i.e.\ the mean of the
per-language means, so that the unequal per-language query counts (Table~\ref{tab:route_counts}) cannot bias the
headline. All 95\% confidence intervals come from a language-stratified, patent-family--clustered bootstrap
(\num{10000} resamples; BCa intervals \citep{efron1987bca}); two queries derived from the same patent family share
their gold set and are resampled together \citep{field2007clustered}. For the same-vs.-cross comparison
(MoLIR@10 vs.\ CLIR@10) we test, per model, the paired per-query gap on the both-gold domain (queries with both a
same- and a cross-language relevant document; $n{=}261$ GP, $n{=}198$ EPO) using a family-level sign-flip
permutation test \citep{smucker2007comparison} with $p{=}(1+b)/(B+1)$ \citep{phipson2010permutation}, reporting
win/tie/loss and the matched-pairs rank-biserial effect size \citep{kerby2014simple}. Model comparisons use a
Friedman omnibus \citep{friedman1937,demsar2006} followed by paired permutation tests on the balanced difference
with Holm correction \citep{holm1979} within each pre-registered family. Per-route recall uses Wilson score
intervals \citep{wilson1927}; we report no significance test on cells with fewer than 30 queries. A cluster-robust
binomial GLM \citep{cameron2015practitioner} on gold-pair hits ($\text{hit}\sim\text{route\_type}+\text{query\_language}$,
family-clustered SEs) gives the language-adjusted odds of same- vs.\ cross-language gold retrieval.
```

Key BibTeX handles the paragraph expects: `efron1987bca`, `field2007clustered`, `smucker2007comparison`,
`phipson2010permutation`, `kerby2014simple`, `friedman1937`, `demsar2006`, `holm1979`, `wilson1927`,
`cameron2015practitioner` (full author-year-title list in the panel plan / `code/claimF_significance.py` header).
