# Data Audit Report

Generated: 2026-05-27 18:53

## Summary

| Source | Unique Docs | Total Rows | Languages | Date Range | IPC Available | File |
|--------|------------|------------|-----------|------------|---------------|------|
| Google Patents | 531 | 1,110 | fr, en, de, zh, es | 2025-10-02 .. 2025-10-22 | Yes | `multilingual_corpus.csv` |
| EPO | 3,876 | 11,625 | en, fr, de | 2026-04-15 .. 2026-05-20 | No (not in CSV) | `multilingual_corpus.csv` |
| USPTO | — | — | — | — | — | *Not yet ingested* |

## Google Patents

- **File**: `data/google_patents/multilingual_corpus.csv`
- **Unique documents**: 531
- **Total rows**: 1,110
- **Date range**: 2025-10-02 to 2025-10-22

### Language Distribution

| Language | Rows |
|----------|------|
| de | 127 |
| en | 420 |
| es | 12 |
| fr | 531 |
| zh | 20 |

### Field Coverage

| Field | Rows with content | % |
|-------|------------------|---|
| abstract | 1,110 | 100.0% |
| first_claim | 0 | 0.0% |
| description | 0 | 0.0% |

### Country Codes

- `EP`: 962 rows
- `WO`: 148 rows

### IPC Class Distribution (top-level)

| IPC Class | Documents |
|-----------|----------|
| A61 | 135 |
| C08 | 114 |
| C07 | 92 |
| C12 | 83 |
| B01 | 49 |
| C09 | 48 |
| H01 | 35 |
| G01 | 35 |
| C23 | 25 |
| C22 | 25 |
| C25 | 22 |
| C10 | 21 |
| C01 | 19 |
| B60 | 19 |
| C11 | 16 |
| B29 | 16 |
| C04 | 16 |
| B32 | 12 |
| A23 | 12 |
| C03 | 12 |

## EPO

- **File**: `data/EPO/multilingual_corpus.csv`
- **Unique documents**: 3,876
- **Total rows**: 11,625
- **Date range**: 2026-04-15 to 2026-05-20

### Language Distribution

| Language | Rows |
|----------|------|
| de | 3,873 |
| en | 3,876 |
| fr | 3,876 |

### Field Coverage

| Field | Rows with content | % |
|-------|------------------|---|
| abstract | 0 | 0.0% |
| first_claim | 11,625 | 100.0% |
| description | 3,876 | 33.3% |

### Country Codes

- `EP`: 11,625 rows

### IPC Class Distribution

Not available in current CSV output. The EPO XML parser extracts IPC/CPC codes
at parse time but `build_row_for_language` does not persist them to the corpus CSV.
Chemistry filtering uses prefixes: C, A01N, A23L, A61K, A61P, B01D, B01F, B01J, B01L, C25, G01N, H01M.

## USPTO

No data ingested yet. No USPTO loader or data files present in the project.

## Cross-Source Overlap

Dedup key: `country_code + bare doc-number` (e.g. `EP_4634118`).

| Metric | Count |
|--------|-------|
| EPO-only documents | 3,876 |
| Google Patents-only documents | 531 |
| Duplicates (same patent in both) | 0 |
| **Merged total (deduplicated)** | **4,407** |

No duplicates found (the dedup script may have already removed them).

## Gaps Across Sources

### Language Gaps

- Languages in Google Patents only: es, zh

### Date Coverage Gaps

- Google Patents: 2025-10-02 to 2025-10-22
- EPO: 2026-04-15 to 2026-05-20
- **Gap**: no date overlap between the two sources.

### IPC Coverage Gaps

- Google Patents has IPC data; EPO does not (not persisted to CSV).
- Google Patents IPC classes: A23, A61, B01, B29, B32, B60, C01, C03, C04, C07, C08, C09, C10, C11, C12, C22, C23, C25, G01, H01
