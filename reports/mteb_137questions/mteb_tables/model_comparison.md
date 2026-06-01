# MTEB Model Comparison

## Leaderboard

### Overview

- Dataset: `MehdiAstaraki/multi-lingual-qac-chem-patents`
- Models compared: `4`
- Best model by `recall_at_10`: `BAAI/bge-m3` (0.7166)

### Ranking

| Rank | Model | Main score | Recall@10 | Recall@100 | MAP@10 | MAP@100 | MAP | nDCG@10 | nDCG@100 | Same-lang irr@100 | Time (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `BAAI/bge-m3` | **0.7166** | **0.7166** | **0.9088** | **0.5738** | **0.5894** | **0.5904** | **0.6390** | **0.6919** | 0.2215 | 40.9 |
| 2 | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 0.5888 | 0.5888 | 0.8248 | 0.4368 | 0.4536 | 0.4551 | 0.5034 | 0.5682 | **0.2004** | 17.5 |
| 3 | `intfloat/multilingual-e5-large` | 0.5450 | 0.5450 | 0.8491 | 0.3751 | 0.3962 | 0.3977 | 0.4573 | 0.5390 | 0.5193 | 39.8 |
| 4 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 0.4951 | 0.4951 | 0.8285 | 0.3450 | 0.3686 | 0.3699 | 0.4136 | 0.5031 | 0.2061 | 16.7 |

### Metric Winners

| Metric | Best model | Score |
| --- | --- | ---: |
| `Main score` | `BAAI/bge-m3` | 0.7166 |
| `Recall@10` | `BAAI/bge-m3` | 0.7166 |
| `Recall@100` | `BAAI/bge-m3` | 0.9088 |
| `MAP@10` | `BAAI/bge-m3` | 0.5738 |
| `MAP@100` | `BAAI/bge-m3` | 0.5894 |
| `MAP` | `BAAI/bge-m3` | 0.5904 |
| `nDCG@10` | `BAAI/bge-m3` | 0.6390 |
| `nDCG@100` | `BAAI/bge-m3` | 0.6919 |
| `Same-lang irr@100` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 0.2004 |
