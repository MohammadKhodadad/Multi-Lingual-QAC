# Progressive code-switching — retrieval decay
- **28** base documents, ladder depth 0..5, haystack = **23426** docs (base publications removed).
- `Alibaba-NLP/gte-multilingual-base`: recall@10 0.01 → 0.01 (Δ=+0.00); mean cosine 0.770 → 0.769.

## Per-step cosine drop by swap mode
- `Alibaba-NLP/gte-multilingual-base`: B=0.0014, C=0.0008, D=0.0001, F=-0.0006
