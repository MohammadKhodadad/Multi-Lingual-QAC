# MTEB Model Comparison

## Leaderboard

### Overview

- Dataset: `MehdiAstaraki/multi-lingual-qac-chem-patents`
- Models compared: `9`
- Best model by `recall_at_10`: `google/embeddinggemma-300m` (0.5438)

### Ranking

| Rank | Model | Main score | Recall@10 | Recall@100 | MAP@10 | MAP@100 | MAP | nDCG@10 | nDCG@100 | Same-lang irr@100 | Time (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `google/embeddinggemma-300m` | **0.5438** | **0.5438** | **0.7238** | **0.4020** | **0.4138** | **0.4146** | **0.4614** | **0.5088** | 0.3568 | 327.0 |
| 2 | `BAAI/bge-m3` | 0.4757 | 0.4757 | 0.6606 | 0.3295 | 0.3408 | 0.3416 | 0.3906 | 0.4392 | 0.2760 | 606.5 |
| 3 | `Qwen/Qwen3-Embedding-0.6B` | 0.4684 | 0.4684 | 0.6837 | 0.3331 | 0.3447 | 0.3455 | 0.3904 | 0.4446 | 0.3257 | 441.4 |
| 4 | `nomic-ai/nomic-embed-text-v2-moe` | 0.4428 | 0.4428 | 0.6642 | 0.3354 | 0.3480 | 0.3493 | 0.3907 | 0.4467 | 0.3518 | 312.4 |
| 5 | `ibm-granite/granite-embedding-278m-multilingual` | 0.3589 | 0.3589 | 0.6010 | 0.2727 | 0.2883 | 0.2898 | 0.3166 | 0.3800 | 0.2429 | 56.7 |
| 6 | `sentence-transformers/LaBSE` | 0.2774 | 0.2774 | 0.4672 | 0.1818 | 0.1920 | 0.1931 | 0.2281 | 0.2775 | **0.2402** | 133.4 |
| 7 | `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR` | 0.2117 | 0.2117 | 0.4246 | 0.1421 | 0.1545 | 0.1548 | 0.1759 | 0.2306 | 0.3231 | 329.2 |
| 8 | `intfloat/multilingual-e5-large-instruct` | 0.1776 | 0.1776 | 0.3431 | 0.1122 | 0.1194 | 0.1213 | 0.1519 | 0.1928 | 0.9530 | 343.9 |
| 9 | `Alibaba-NLP/gte-multilingual-base` | 0.0037 | 0.0037 | 0.0182 | 0.0018 | 0.0022 | 0.0023 | 0.0028 | 0.0063 | 0.6223 | 73.5 |

### Metric Winners

| Metric | Best model | Score |
| --- | --- | ---: |
| `Main score` | `google/embeddinggemma-300m` | 0.5438 |
| `Recall@10` | `google/embeddinggemma-300m` | 0.5438 |
| `Recall@100` | `google/embeddinggemma-300m` | 0.7238 |
| `MAP@10` | `google/embeddinggemma-300m` | 0.4020 |
| `MAP@100` | `google/embeddinggemma-300m` | 0.4138 |
| `MAP` | `google/embeddinggemma-300m` | 0.4146 |
| `nDCG@10` | `google/embeddinggemma-300m` | 0.4614 |
| `nDCG@100` | `google/embeddinggemma-300m` | 0.5088 |
| `Same-lang irr@100` | `sentence-transformers/LaBSE` | 0.2402 |
