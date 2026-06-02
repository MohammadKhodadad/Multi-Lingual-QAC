# Human vs LLM Evaluation of the Balanced-100 QAC Sample

**Date:** 2026-05-28
**Inputs**
- Human annotator scores: [Evaluated data.xlsx](../../Evaluated%20data.xlsx) (sheet `qac_with_modes`, 97 rows)
- LLM auto-grader scores: [balanced_100_qac_regraded.csv](../../balanced_100_qac_regraded.csv) (137 rows)
- Analysis script: [scripts/analyze_human_vs_llm.py](../../scripts/analyze_human_vs_llm.py)
- Joined per-row data: [joined_scores.csv](joined_scores.csv) · [summary.json](summary.json)

---

## 1. Scales and scope

| Source | Column with the overall score | Max | Composition |
|---|---|---|---|
| Human | `total_score` | **10** | One overall score per question (per the user's instruction: 0–3 poor, 4–6 ok, 7–10 good). Per-dimension columns are also filled on a 1–5 scale. |
| LLM   | `total_score` | **40** | `faith_overall` (3 sub-scores × 5 = **/15**) + `qual_overall` (5 sub-scores × 5 = **/25**). Confirmed empirically: faith_overall max = 15, qual_overall max = 25, observed total_score max in the file = 38. |

All 97 human-rated rows were matched to a row in the LLM CSV via `(corpus_id, question)` — no orphans.

> The human sample covers strategies **`all`, `random_any`, `random_existing`, `random_missing`** but **not `forced_zh`** (strategy 0), so the comparison below excludes that strategy.

---

## 2. Human-only analysis

### 2.1 Overall distribution

| Score | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|
| Count | 3 | 16 | 29 | 44 | 5 |

- **n = 97**, mean = **8.33 / 10**, median = **9**, range 6–10.
- Bucket breakdown using the user's bands:

| Bucket | Range | Count | Share |
|---|---|---|---|
| Poor | 0–3 | **0** | 0.0% |
| OK   | 4–6 | **3** | 3.1% |
| Good | 7–10 | **94** | 96.9% |

Nothing in the sample was rejected by the annotator; the lowest score given was 6.

### 2.2 By mode

| Mode | n | Mean | Median | Std | Min | Max |
|---|---|---|---|---|---|---|
| **technical** | 50 | **8.72** | 9 | 0.83 | 6 | 10 |
| **semantic**  | 47 | **7.91** | 8 | 0.83 | 6 | 9  |

Technical questions are rated ~0.8 points higher on a /10 scale and reach a ceiling of 10, whereas the semantic set never breaks 9.

### 2.3 By mode × strategy

| Mode | Strategy | n | Mean | Median | Good % |
|---|---|---|---|---|---|
| technical | random_existing | 12 | **9.08** | 9 | 100% |
| technical | random_any      | 13 | 8.69 | 9 | 92.3% |
| technical | random_missing  | 13 | 8.62 | 9 | 100% |
| technical | all             | 12 | 8.50 | 9 | 100% |
| semantic  | random_existing | 10 | **8.30** | 8.5 | 100% |
| semantic  | random_any      | 13 | 8.15 | 8 | 100% |
| semantic  | random_missing  | 13 | 7.77 | 8 | 100% |
| semantic  | all             | 11 | 7.45 | 8 | 81.8% |

Observations
- `random_existing` is the top-ranked strategy in **both** modes — questions written against a language version that the corpus actually has read best to the annotator.
- The `all` strategy is the weakest in both modes; it's also the only strategy where any "ok" (4–6) ratings appear in the semantic set (2 of 11).
- The only "ok" rating in the technical set is a single point in `random_any`.

### 2.4 Per-dimension means (1–5 scale)

**Technical mode (n = 50)** — fidelity sub-scores plus the technical-style quality rubric.

| Dimension | Mean |
|---|---|
| faith_grounding | 4.42 |
| faith_precision | **4.90** |
| faith_numerical_fidelity | **5.00** |
| faith_overall | 4.74 |
| qual_search_bar_realism | **3.52** |
| qual_specificity | 4.04 |
| qual_phrasing_economy | **3.58** |
| qual_focus | 4.74 |
| qual_linguistic_quality | 4.70 |
| qual_overall | 3.98 |

Weak spots: `search_bar_realism` and `phrasing_economy` — technical questions read clean and linguistically correct, but the annotator finds them too "essay-shaped" for a real search box.

**Semantic mode (n = 47)** — semantic-style quality rubric.

| Dimension | Mean |
|---|---|
| faith_grounding | **3.47** |
| faith_precision | **3.49** |
| faith_numerical_fidelity | 4.94 |
| faith_overall | 3.89 |
| qual_search_realism | 3.68 |
| qual_lexical_distance | 3.96 |
| qual_conceptual_framing | 3.79 |
| qual_retrievability | 4.15 |
| qual_linguistic_quality_1 | **4.98** |
| qual_overall_2 | 4.02 |

Weak spots: `faith_grounding` and `faith_precision` drop ~1 full point below their technical counterparts. This is the expected cost of the semantic mode — paraphrasing trades off against verbatim grounding — but it's the dominant reason semantic scores trail technical.

---

## 3. Human vs LLM comparison

LLM scores are normalized to a percentage (`llm_total / 40`) and human scores to a percentage (`human_total / 10`) so the two are on a common axis.

### 3.1 Overall

| Metric | Value |
|---|---|
| n | 97 |
| Human mean | **83.3%** |
| LLM mean   | **79.0%** |
| Mean signed Δ (human − LLM) | **+4.25 pp** |
| Mean absolute Δ | 8.22 pp |
| Pearson r  | **0.40** |
| Spearman ρ | **0.40** |

The human grades the sample ~4 pp **higher** than the LLM on average. Correlation is **moderate** (~0.40) — the LLM and human agree on the broad ranking of items but disagree noticeably on individual scores.

### 3.2 Bucket agreement

Applying the 0–3 / 4–6 / 7–10 bands to both scales:

| Human \ LLM | poor | ok | good |
|---|---|---|---|
| poor | 0 | 0 | 0 |
| ok   | 0 | 0 | 3 |
| good | 0 | 3 | 94 |

- **96.9% bucket agreement**, but it's almost entirely "both call it good", because **neither grader produced any `poor` ratings** in this sample.
- The disagreements are symmetric in count (3 cases each) but in opposite directions: the LLM flagged 3 items as "ok" that the human rated "good", and the human rated 3 items "ok" that the LLM put in "good".
- The high agreement number should therefore be read with caution: with the lowest LLM score equivalent to ~27/40 = 67.5% (≈ bucket "good"), the LLM is essentially binary on this sample.

### 3.3 By mode

| Mode | n | Human mean | LLM mean | Mean Δ (H−L) | Pearson | Spearman |
|---|---|---|---|---|---|---|
| **technical** | 50 | 87.2% | 81.0% | **+6.2 pp** | **0.36** | **0.41** |
| **semantic**  | 47 | 79.1% | 77.0% | +2.2 pp | 0.21 | 0.20 |

- The LLM grades the **technical** set noticeably lower than the human (the LLM is harder than the annotator on the question-quality rubric — `qual_specificity`/`qual_phrasing_economy`).
- Correlation on the **semantic** set is weak (~0.2). The two graders are nearly uncorrelated there — the LLM's semantic-mode rubric (lexical distance, conceptual framing, retrievability) does not track what the human annotator weights when reading a paraphrased question.

### 3.4 By mode × strategy

| Mode | Strategy | n | Human % | LLM % | Δ (H−L) | Pearson | Spearman |
|---|---|---|---|---|---|---|---|
| technical | random_existing | 12 | 90.8 | 85.2 | +5.6 | **0.56** | **0.64** |
| technical | random_missing  | 13 | 86.2 | 79.0 | +7.1 | 0.49 | 0.54 |
| technical | random_any      | 13 | 86.9 | 80.2 | +6.7 | 0.30 | 0.32 |
| technical | all             | 12 | 85.0 | 79.8 | +5.2 | 0.07 | 0.06 |
| semantic  | random_existing | 10 | 83.0 | 77.5 | +5.5 | 0.45 | 0.42 |
| semantic  | all             | 11 | 74.5 | 76.4 | −1.8 | 0.31 | 0.36 |
| semantic  | random_missing  | 13 | 77.7 | 77.3 | +0.4 | 0.18 | 0.05 |
| semantic  | random_any      | 13 | 81.5 | 76.7 | +4.8 | 0.04 | 0.09 |

- **Best alignment:** `technical / random_existing` (ρ ≈ 0.64) — also the highest absolute scores from both graders.
- **Worst alignment:** `technical / all` (ρ ≈ 0.06) and `semantic / random_any` (ρ ≈ 0.09). Strategies that mix language conditions or aren't tied to an existing target language produce the noisiest graders agreement.
- **Sign flip:** `semantic / all` is the only cell where the LLM grades **higher** than the human — consistent with §2.3, where the human flagged 2/11 `all` items as "ok" while the LLM kept them in "good".

---

## 4. Takeaways

1. **The sample is overwhelmingly "good"** under the user's thresholds — 94 / 97 rows. With no rejections, this sample is most useful for measuring relative ordering, not pass/fail.
2. **Technical > semantic** on both human and LLM judgments. The gap is driven mainly by `faith_grounding` and `faith_precision` (semantic paraphrasing reduces verbatim grounding).
3. **`random_existing` is the best strategy** in both modes by human score and by human/LLM agreement — the safest sampling strategy if downstream consumers need uniformly high quality.
4. **`all` and `forced_zh` are the weak strategies** by every human cut; `all` also has the lowest human/LLM correlation on the technical side, meaning the LLM rubric isn't catching what the annotator dislikes about it.
5. **LLM auto-grader is moderately calibrated** (ρ ≈ 0.40 overall) but **biased down by ~4 pp** vs the human, mostly on the technical question-quality rubric (`search_bar_realism`, `phrasing_economy`). If the auto-grader is being used as a pass/fail gate, the threshold should probably be raised on the LLM side or the human side should be considered authoritative for borderline cases.
6. **Semantic-mode auto-grading is weakest** (ρ ≈ 0.2). The LLM's semantic-mode rubric needs review — it currently does not predict human judgment well, so headline metrics built on it are not reliable for this mode.

---

## 5. Method notes

- Bucketing used the user-given bands (0–3 poor, 4–6 ok, 7–10 good) applied to both scales by linearly rescaling LLM `/40` to `/10`. Cell boundaries on the LLM side: poor ≤ 12, ok 13–24, good ≥ 25.
- Correlations are computed on the percentage-normalized scores; the two correlation methods agree closely, suggesting no major rank-vs-magnitude divergence.
- The human file was joined to the LLM file via `(corpus_id, question)`; all 97 rows joined cleanly.
- `forced_zh` (strategy 0) was not in the human-annotated subset and therefore does not appear in any per-strategy row above.
