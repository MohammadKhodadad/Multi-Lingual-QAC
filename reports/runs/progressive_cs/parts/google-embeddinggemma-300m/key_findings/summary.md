# Progressive code-switching — retrieval decay
- **28** base documents, ladder depth 0..5, haystack = **23426** docs (base publications removed).
- `google/embeddinggemma-300m`: recall@10 0.72 → 0.70 (Δ=+0.01); mean cosine 0.460 → 0.453.

## Per-step cosine drop by swap mode
- `google/embeddinggemma-300m`: B=0.0020, C=0.0007, D=0.0015, F=0.0012
