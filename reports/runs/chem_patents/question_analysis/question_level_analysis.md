# Question-level analysis (MehdiAstaraki/multi-lingual-qac-chem-patents, `multilingual`, Recall@10 / MRR@10)

## Dataset structure
- Queries with relevance judgements: 137
- Original: 57  |  synthetic-translation: 80
- Queries by language: de=27, en=27, es=34, fr=27, zh=22
- Relevant (query, doc) pairs: 322 (avg 2.35/query)
- Models analysed: Alibaba-NLP/gte-multilingual-base, BAAI/bge-m3, cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR, google/embeddinggemma-300m, ibm-granite/granite-embedding-278m-multilingual, intfloat/multilingual-e5-large-instruct, nomic-ai/nomic-embed-text-v2-moe, Qwen/Qwen3-Embedding-0.6B, sentence-transformers/LaBSE

## 1) Recall@10 by query language
| Group | n | Alibaba-NLP/gte-multilingual-base | BAAI/bge-m3 | cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR | google/embeddinggemma-300m | ibm-granite/granite-embedding-278m-multilingual | intfloat/multilingual-e5-large-instruct | nomic-ai/nomic-embed-text-v2-moe | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/LaBSE |
|---|---|---|---|---|---|---|---|---|---|---|
| de | 27 | 0.000 | 0.506 | 0.296 | 0.636 | 0.426 | 0.154 | 0.488 | 0.562 | 0.241 |
| en | 27 | 0.000 | 0.420 | 0.179 | 0.488 | 0.340 | 0.290 | 0.401 | 0.463 | 0.247 |
| es | 34 | 0.000 | 0.525 | 0.162 | 0.544 | 0.392 | 0.029 | 0.485 | 0.534 | 0.245 |
| fr | 27 | 0.019 | 0.432 | 0.204 | 0.488 | 0.309 | 0.284 | 0.475 | 0.333 | 0.296 |
| zh | 22 | 0.000 | 0.485 | 0.235 | 0.568 | 0.311 | 0.167 | 0.333 | 0.424 | 0.386 |

##    MRR@10 by query language
| Group | n | Alibaba-NLP/gte-multilingual-base | BAAI/bge-m3 | cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR | google/embeddinggemma-300m | ibm-granite/granite-embedding-278m-multilingual | intfloat/multilingual-e5-large-instruct | nomic-ai/nomic-embed-text-v2-moe | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/LaBSE |
|---|---|---|---|---|---|---|---|---|---|---|
| de | 27 | 0.000 | 0.429 | 0.276 | 0.580 | 0.443 | 0.216 | 0.524 | 0.459 | 0.293 |
| en | 27 | 0.000 | 0.328 | 0.197 | 0.455 | 0.353 | 0.289 | 0.371 | 0.399 | 0.278 |
| es | 34 | 0.000 | 0.456 | 0.141 | 0.431 | 0.377 | 0.029 | 0.445 | 0.446 | 0.204 |
| fr | 27 | 0.019 | 0.388 | 0.190 | 0.483 | 0.306 | 0.395 | 0.521 | 0.368 | 0.246 |
| zh | 22 | 0.000 | 0.436 | 0.214 | 0.452 | 0.294 | 0.137 | 0.301 | 0.379 | 0.325 |

## 4) Recall@10 by query origin (original vs synthetic-translation)
| Group | n | Alibaba-NLP/gte-multilingual-base | BAAI/bge-m3 | cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR | google/embeddinggemma-300m | ibm-granite/granite-embedding-278m-multilingual | intfloat/multilingual-e5-large-instruct | nomic-ai/nomic-embed-text-v2-moe | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/LaBSE |
|---|---|---|---|---|---|---|---|---|---|---|
| original | 57 | 0.009 | 0.462 | 0.219 | 0.547 | 0.363 | 0.322 | 0.474 | 0.453 | 0.287 |
| synthetic-translation | 80 | 0.000 | 0.485 | 0.206 | 0.542 | 0.356 | 0.075 | 0.421 | 0.479 | 0.271 |

## 5) Cross-lingual targets: same- vs cross-language (mean Recall@10)
| Target | Alibaba-NLP/gte-multilingual-base | BAAI/bge-m3 | cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR | google/embeddinggemma-300m | ibm-granite/granite-embedding-278m-multilingual | intfloat/multilingual-e5-large-instruct | nomic-ai/nomic-embed-text-v2-moe | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/LaBSE |
|---|---|---|---|---|---|---|---|---|---|
| same-language target | 0.018 | 0.596 | 0.316 | 0.667 | 0.456 | 0.632 | 0.667 | 0.561 | 0.404 |
| cross-language target | 0.000 | 0.437 | 0.179 | 0.502 | 0.328 | 0.077 | 0.381 | 0.433 | 0.238 |

## 6) Language-pair Recall@10 matrix — google/embeddinggemma-300m (best model)
Rows = query language, Cols = relevant-doc language; cell = fraction of those
relevant docs retrieved in the top 10 (n = #relevant pairs).

| q\d | de | en | fr | zh |
|---|---|---|---|---|
| **de** | 0.71 (7) | 0.57 (21) | 0.63 (27) | 0.62 (8) |
| **en** | 0.14 (7) | 0.67 (21) | 0.44 (27) | 0.25 (8) |
| **es** | 0.50 (12) | 0.44 (27) | 0.59 (34) | 0.46 (13) |
| **fr** | 0.22 (9) | 0.42 (19) | 0.63 (27) | 0.25 (8) |
| **zh** | 0.50 (4) | 0.53 (19) | 0.59 (22) | 1.00 (2) |

