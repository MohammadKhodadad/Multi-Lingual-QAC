# Question-level analysis (MehdiAstaraki/multi-lingual-qac-chem-patents, `multilingual`, Recall@10 / MRR@10)

## Dataset structure
- Queries with relevance judgements: 137
- Original: 57  |  synthetic-translation: 80
- Queries by language: de=27, en=27, es=34, fr=27, zh=22
- Relevant (query, doc) pairs: 322 (avg 2.35/query)
- Models analysed: BAAI/bge-m3, cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR, google/embeddinggemma-300m, ibm-granite/granite-embedding-278m-multilingual, intfloat/multilingual-e5-large-instruct, nomic-ai/nomic-embed-text-v2-moe, Qwen/Qwen3-Embedding-0.6B, sentence-transformers/LaBSE

## 1) Recall@10 by query language
| Group | n | BAAI/bge-m3 | cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR | google/embeddinggemma-300m | ibm-granite/granite-embedding-278m-multilingual | intfloat/multilingual-e5-large-instruct | nomic-ai/nomic-embed-text-v2-moe | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/LaBSE |
|---|---|---|---|---|---|---|---|---|---|
| de | 27 | 0.457 | 0.272 | 0.574 | 0.377 | 0.154 | 0.451 | 0.488 | 0.228 |
| en | 27 | 0.407 | 0.179 | 0.463 | 0.302 | 0.290 | 0.389 | 0.414 | 0.222 |
| es | 34 | 0.485 | 0.142 | 0.505 | 0.338 | 0.029 | 0.466 | 0.485 | 0.235 |
| fr | 27 | 0.407 | 0.191 | 0.463 | 0.284 | 0.284 | 0.463 | 0.321 | 0.284 |
| zh | 20 | 0.458 | 0.225 | 0.533 | 0.325 | 0.075 | 0.300 | 0.450 | 0.392 |

##    MRR@10 by query language
| Group | n | BAAI/bge-m3 | cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR | google/embeddinggemma-300m | ibm-granite/granite-embedding-278m-multilingual | intfloat/multilingual-e5-large-instruct | nomic-ai/nomic-embed-text-v2-moe | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/LaBSE |
|---|---|---|---|---|---|---|---|---|---|
| de | 27 | 0.411 | 0.276 | 0.580 | 0.375 | 0.216 | 0.518 | 0.448 | 0.285 |
| en | 27 | 0.328 | 0.197 | 0.455 | 0.353 | 0.289 | 0.371 | 0.392 | 0.235 |
| es | 34 | 0.456 | 0.137 | 0.400 | 0.337 | 0.029 | 0.445 | 0.417 | 0.203 |
| fr | 27 | 0.388 | 0.190 | 0.483 | 0.285 | 0.395 | 0.521 | 0.368 | 0.246 |
| zh | 20 | 0.358 | 0.210 | 0.367 | 0.238 | 0.032 | 0.239 | 0.367 | 0.348 |

## 4) Recall@10 by query origin (original vs synthetic-translation)
| Group | n | BAAI/bge-m3 | cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR | google/embeddinggemma-300m | ibm-granite/granite-embedding-278m-multilingual | intfloat/multilingual-e5-large-instruct | nomic-ai/nomic-embed-text-v2-moe | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/LaBSE |
|---|---|---|---|---|---|---|---|---|---|
| original | 55 | 0.436 | 0.203 | 0.506 | 0.333 | 0.321 | 0.455 | 0.421 | 0.267 |
| synthetic-translation | 80 | 0.450 | 0.194 | 0.506 | 0.321 | 0.056 | 0.400 | 0.442 | 0.263 |

## 5) Cross-lingual targets: same- vs cross-language (mean Recall@10)
| Target | BAAI/bge-m3 | cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR | google/embeddinggemma-300m | ibm-granite/granite-embedding-278m-multilingual | intfloat/multilingual-e5-large-instruct | nomic-ai/nomic-embed-text-v2-moe | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/LaBSE |
|---|---|---|---|---|---|---|---|---|
| same-language target | 0.582 | 0.309 | 0.655 | 0.455 | 0.618 | 0.655 | 0.564 | 0.400 |
| cross-language target | 0.404 | 0.163 | 0.459 | 0.290 | 0.067 | 0.359 | 0.391 | 0.222 |

## 6) Language-pair Recall@10 matrix — google/embeddinggemma-300m (best model)
Rows = query language, Cols = relevant-doc language; cell = fraction of those
relevant docs retrieved in the top 10 (n = #relevant pairs).

| q\d | de | en | fr | zh |
|---|---|---|---|---|
| **de** | 0.71 (7) | 0.57 (21) | 0.63 (27) | 0.00 (8) |
| **en** | 0.14 (7) | 0.67 (21) | 0.44 (27) | 0.00 (8) |
| **es** | 0.50 (12) | 0.44 (27) | 0.59 (34) | 0.15 (13) |
| **fr** | 0.22 (9) | 0.42 (19) | 0.63 (27) | 0.00 (8) |
| **zh** | 0.50 (4) | 0.53 (17) | 0.55 (20) |  -  |

