# Question-level analysis (MehdiAstaraki/multi-lingual-qac-alias-graph, `multilingual`, Recall@10 / MRR@10)

## Dataset structure
- Queries with relevance judgements: 132
- Queries by language: de=27, en=27, es=25, fr=27, zh=26
- Relevant (query, doc) pairs: 14423 (avg 109.27/query)
- Models analysed: Alibaba-NLP/gte-multilingual-base, BAAI/bge-m3, cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR, google/embeddinggemma-300m, ibm-granite/granite-embedding-278m-multilingual, intfloat/multilingual-e5-large-instruct, nomic-ai/nomic-embed-text-v2-moe, Qwen/Qwen3-Embedding-0.6B, sentence-transformers/LaBSE

## 1) Recall@10 by query language
| Group | n | Alibaba-NLP/gte-multilingual-base | BAAI/bge-m3 | cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR | google/embeddinggemma-300m | ibm-granite/granite-embedding-278m-multilingual | intfloat/multilingual-e5-large-instruct | nomic-ai/nomic-embed-text-v2-moe | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/LaBSE |
|---|---|---|---|---|---|---|---|---|---|---|
| de | 27 | 0.001 | 0.052 | 0.051 | 0.076 | 0.064 | 0.030 | 0.059 | 0.060 | 0.053 |
| en | 27 | 0.017 | 0.060 | 0.057 | 0.085 | 0.054 | 0.047 | 0.072 | 0.073 | 0.071 |
| es | 25 | 0.002 | 0.057 | 0.025 | 0.092 | 0.056 | 0.022 | 0.065 | 0.065 | 0.043 |
| fr | 27 | 0.002 | 0.063 | 0.056 | 0.080 | 0.062 | 0.029 | 0.046 | 0.060 | 0.053 |
| zh | 26 | 0.008 | 0.038 | 0.058 | 0.071 | 0.044 | 0.017 | 0.043 | 0.063 | 0.059 |

##    MRR@10 by query language
| Group | n | Alibaba-NLP/gte-multilingual-base | BAAI/bge-m3 | cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR | google/embeddinggemma-300m | ibm-granite/granite-embedding-278m-multilingual | intfloat/multilingual-e5-large-instruct | nomic-ai/nomic-embed-text-v2-moe | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/LaBSE |
|---|---|---|---|---|---|---|---|---|---|---|
| de | 27 | 0.119 | 0.753 | 0.316 | 0.678 | 0.630 | 0.431 | 0.736 | 0.605 | 0.527 |
| en | 27 | 0.183 | 0.800 | 0.558 | 0.706 | 0.659 | 0.685 | 0.827 | 0.830 | 0.599 |
| es | 25 | 0.063 | 0.710 | 0.374 | 0.823 | 0.564 | 0.277 | 0.822 | 0.728 | 0.518 |
| fr | 27 | 0.120 | 0.728 | 0.512 | 0.751 | 0.585 | 0.589 | 0.712 | 0.819 | 0.606 |
| zh | 26 | 0.113 | 0.515 | 0.441 | 0.673 | 0.409 | 0.254 | 0.350 | 0.665 | 0.572 |

## 5) Cross-lingual targets: same- vs cross-language (mean Recall@10)
| Target | Alibaba-NLP/gte-multilingual-base | BAAI/bge-m3 | cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR | google/embeddinggemma-300m | ibm-granite/granite-embedding-278m-multilingual | intfloat/multilingual-e5-large-instruct | nomic-ai/nomic-embed-text-v2-moe | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/LaBSE |
|---|---|---|---|---|---|---|---|---|---|
| same-language target | 0.047 | 0.125 | 0.081 | 0.131 | 0.112 | 0.174 | 0.142 | 0.101 | 0.091 |
| cross-language target | 0.000 | 0.046 | 0.043 | 0.077 | 0.050 | 0.009 | 0.047 | 0.057 | 0.054 |

## 6) Language-pair Recall@10 matrix — google/embeddinggemma-300m (best model)
Rows = query language, Cols = relevant-doc language; cell = fraction of those
relevant docs retrieved in the top 10 (n = #relevant pairs).

| q\d | de | en | es | fr | zh |
|---|---|---|---|---|---|
| **de** | 0.12 (209) | 0.01 (1334) | 0.02 (201) | 0.02 (1143) | 0.17 (23) |
| **en** | 0.03 (209) | 0.03 (1334) | 0.04 (201) | 0.02 (1143) | 0.22 (23) |
| **es** | 0.06 (206) | 0.02 (1292) | 0.08 (193) | 0.02 (1109) | 0.22 (23) |
| **fr** | 0.04 (209) | 0.02 (1334) | 0.02 (201) | 0.04 (1143) | 0.17 (23) |
| **zh** | 0.05 (207) | 0.02 (1315) | 0.03 (201) | 0.02 (1124) | 0.22 (23) |

