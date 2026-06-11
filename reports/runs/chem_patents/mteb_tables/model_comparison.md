# MTEB Model Comparison

## Leaderboard

### Overview

- Dataset: `MehdiAstaraki/multi-lingual-qac-chem-patents`
- Models compared: `9`
- Best model by `recall_at_10`: `google/embeddinggemma-300m` (0.5739)

### Ranking

| Rank | Model | Main score | Recall@10 | Recall@100 | MAP@10 | MAP@100 | MAP | nDCG@10 | nDCG@100 | Same-lang irr@100 | Time (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `google/embeddinggemma-300m` | **0.5739** | **0.5739** | **0.7584** | **0.4445** | **0.4589** | **0.4595** | **0.5032** | **0.5540** | 0.3692 | 334.1 |
| 2 | `BAAI/bge-m3` | 0.5091 | 0.5091 | 0.6869 | 0.3861 | 0.3974 | 0.3987 | 0.4456 | 0.4927 | 0.2961 | 630.2 |
| 3 | `Qwen/Qwen3-Embedding-0.6B` | 0.4959 | 0.4959 | 0.7150 | 0.3758 | 0.3917 | 0.3930 | 0.4324 | 0.4921 | 0.3350 | 289.6 |
| 4 | `nomic-ai/nomic-embed-text-v2-moe` | 0.4669 | 0.4669 | 0.6668 | 0.3612 | 0.3745 | 0.3755 | 0.4172 | 0.4705 | 0.3751 | 271.0 |
| 5 | `ibm-granite/granite-embedding-278m-multilingual` | 0.4154 | 0.4154 | 0.6212 | 0.3160 | 0.3297 | 0.3312 | 0.3633 | 0.4180 | 0.2663 | 58.0 |
| 6 | `sentence-transformers/LaBSE` | 0.3010 | 0.3010 | 0.4787 | 0.1981 | 0.2092 | 0.2109 | 0.2505 | 0.2977 | **0.2429** | 140.5 |
| 7 | `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR` | 0.2377 | 0.2377 | 0.4113 | 0.1628 | 0.1727 | 0.1741 | 0.1994 | 0.2446 | 0.3381 | 196.0 |
| 8 | `intfloat/multilingual-e5-large-instruct` | 0.2183 | 0.2183 | 0.3772 | 0.1547 | 0.1636 | 0.1658 | 0.2065 | 0.2473 | 0.9734 | 189.9 |
| 9 | `Alibaba-NLP/gte-multilingual-base` | 0.0048 | 0.0048 | 0.0224 | 0.0014 | 0.0019 | 0.0021 | 0.0028 | 0.0073 | 0.6766 | 73.2 |

### Metric Winners

| Metric | Best model | Score |
| --- | --- | ---: |
| `Main score` | `google/embeddinggemma-300m` | 0.5739 |
| `Recall@10` | `google/embeddinggemma-300m` | 0.5739 |
| `Recall@100` | `google/embeddinggemma-300m` | 0.7584 |
| `MAP@10` | `google/embeddinggemma-300m` | 0.4445 |
| `MAP@100` | `google/embeddinggemma-300m` | 0.4589 |
| `MAP` | `google/embeddinggemma-300m` | 0.4595 |
| `nDCG@10` | `google/embeddinggemma-300m` | 0.5032 |
| `nDCG@100` | `google/embeddinggemma-300m` | 0.5540 |
| `Same-lang irr@100` | `sentence-transformers/LaBSE` | 0.2429 |
