# Data Audit Report

Generated: 2026-05-27 20:39

## Summary

| Source | Documents | Total Rows (all langs) | Languages | Date Range | IPC Available | File |
|--------|-----------|----------------------|-----------|------------|---------------|------|
| Google Patents | 10,628 | 23,387 | en, fr, es, de | 1999-03-18 .. 2025-10-22 | Yes | `multilingual_corpus.csv` |
| EPO | 3,773 | 11,315 | en, fr, de | 2026-04-22 .. 2026-05-27 | Yes | `multilingual_corpus.csv` |

## Google Patents

- **File**: `data/google_patents/multilingual_corpus.csv`
- **Documents**: 10,628 (each counted once regardless of how many languages)
- **Total rows**: 23,387 (one row per document-language pair)
- **Date range**: 1999-03-18 to 2025-10-22

### Language Distribution

| Language | Rows |
|----------|------|
| de | 1,601 |
| en | 10,827 |
| es | 2,154 |
| fr | 8,805 |

### Parallel Text Coverage

Documents with text in both row-language and column-language:

| | **de** |  **en** |  **es** |  **fr** |
|---|---|---|---|---|
| **de** | 1,601 | 1,601 | 0 | 1,601 |
| **en** | 1,601 | 10,827 | 2,149 | 8,800 |
| **es** | 0 | 2,149 | 2,154 | 127 |
| **fr** | 1,601 | 8,800 | 127 | 8,805 |

### Field Coverage

| Field | Rows with content | % |
|-------|------------------|---|
| abstract | 23,387 | 100.0% |
| first_claim | 0 | 0.0% |
| description | 0 | 0.0% |

### Country Codes

- `CR`: 12 rows
- `EP`: 4,525 rows
- `ES`: 164 rows
- `MX`: 3,876 rows
- `WO`: 14,810 rows

### IPC Class Distribution (top-level)

| IPC Class | Documents |
|-----------|----------|
| A61 | 12,743 |
| C08 | 5,320 |
| C07 | 4,985 |
| C12 | 4,140 |
| C22 | 2,604 |
| C09 | 2,301 |
| B01 | 2,047 |
| H01 | 1,954 |
| C25 | 1,622 |
| C23 | 1,274 |
| C10 | 1,147 |
| C21 | 1,146 |
| C01 | 1,139 |
| C04 | 910 |
| G01 | 846 |
| B32 | 789 |
| C02 | 783 |
| A23 | 774 |
| A01 | 725 |
| C11 | 618 |

## EPO

- **File**: `data/EPO/multilingual_corpus.csv`
- **Documents**: 3,773 (each counted once regardless of how many languages)
- **Total rows**: 11,315 (one row per document-language pair)
- **Date range**: 2026-04-22 to 2026-05-27

### Language Distribution

| Language | Rows |
|----------|------|
| de | 3,770 |
| en | 3,773 |
| fr | 3,772 |

### Parallel Text Coverage

Documents with text in both row-language and column-language:

| | **de** |  **en** |  **fr** |
|---|---|---|---|
| **de** | 3,770 | 3,770 | 3,769 |
| **en** | 3,770 | 3,773 | 3,772 |
| **fr** | 3,769 | 3,772 | 3,772 |

### Field Coverage

| Field | Rows with content | % |
|-------|------------------|---|
| abstract | 0 | 0.0% |
| first_claim | 11,315 | 100.0% |
| description | 3,773 | 33.3% |

### Country Codes

- `EP`: 11,315 rows

### IPC Class Distribution (top-level)

| IPC Class | Documents |
|-----------|----------|
| A61 | 4,212 |
| H01 | 3,484 |
| G01 | 1,580 |
| C08 | 1,354 |
| C07 | 1,308 |
| B01 | 1,019 |
| C12 | 714 |
| C09 | 584 |
| B29 | 432 |
| C10 | 373 |
| C22 | 366 |
| A23 | 348 |
| C23 | 281 |
| A01 | 268 |
| C11 | 242 |
| C01 | 211 |
| H10 | 198 |
| C04 | 196 |
| B32 | 195 |
| B60 | 188 |

## Cross-Source Overlap

Dedup key: `country_code + bare doc-number` (e.g. `EP_4634118`).

| Metric | Count |
|--------|-------|
| EPO-only documents | 3,773 |
| Google Patents-only documents | 10,628 |
| Duplicates (same patent in both) | 0 |
| **Merged total (deduplicated)** | **14,401** |

No duplicates found (the dedup script may have already removed them).

## Gaps Across Sources

### Language Gaps

- Languages in Google Patents only: es

### Date Coverage Gaps

- Google Patents: 1999-03-18 to 2025-10-22
- EPO: 2026-04-22 to 2026-05-27
- **Gap**: no date overlap between the two sources.

### IPC Coverage Gaps

- IPC classes in Google Patents only: C02, C21, C25
- IPC classes in EPO only: B29, B60, H10
