# Progressive code-switching — retrieval decay
- **28** base documents, ladder depth 0..5, haystack = **23725** docs (base publications removed).
- `Alibaba-NLP/gte-multilingual-base`: recall@10 0.01 → 0.01 (Δ=+0.00); mean cosine 0.770 → 0.769.
- `BAAI/bge-m3`: recall@10 0.66 → 0.64 (Δ=+0.02); mean cosine 0.574 → 0.568.
- `Qwen/Qwen3-Embedding-0.6B`: recall@10 0.63 → 0.57 (Δ=+0.05); mean cosine 0.557 → 0.550.
- `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR`: recall@10 0.41 → 0.37 (Δ=+0.04); mean cosine 0.751 → 0.746.
- `google/embeddinggemma-300m`: recall@10 0.72 → 0.70 (Δ=+0.01); mean cosine 0.460 → 0.453.
- `ibm-granite/granite-embedding-278m-multilingual`: recall@10 0.55 → 0.54 (Δ=+0.01); mean cosine 0.717 → 0.714.
- `intfloat/multilingual-e5-large-instruct`: recall@10 0.34 → 0.35 (Δ=-0.01); mean cosine 0.833 → 0.833.
- `nomic-ai/nomic-embed-text-v2-moe`: recall@10 0.63 → 0.59 (Δ=+0.04); mean cosine 0.509 → 0.504.
- `sentence-transformers/LaBSE`: recall@10 0.37 → 0.37 (Δ=+0.01); mean cosine 0.516 → 0.508.

## Per-step cosine drop by swap mode
- `Alibaba-NLP/gte-multilingual-base`: B=0.0014, C=0.0008, D=0.0001, F=-0.0006
- `BAAI/bge-m3`: B=-0.0008, C=0.0027, D=0.0003, F=0.0025
- `Qwen/Qwen3-Embedding-0.6B`: B=-0.0010, C=-0.0003, D=0.0022, F=0.0030
- `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR`: B=-0.0003, C=0.0010, D=0.0011, F=0.0017
- `google/embeddinggemma-300m`: B=0.0020, C=0.0007, D=0.0015, F=0.0012
- `ibm-granite/granite-embedding-278m-multilingual`: B=-0.0007, C=0.0005, D=0.0007, F=0.0011
- `intfloat/multilingual-e5-large-instruct`: B=-0.0015, C=-0.0010, D=0.0002, F=0.0008
- `nomic-ai/nomic-embed-text-v2-moe`: B=-0.0029, C=0.0002, D=0.0009, F=0.0037
- `sentence-transformers/LaBSE`: B=-0.0002, C=0.0013, D=0.0016, F=0.0025
