# Progressive code-switching — retrieval decay
- **28** base documents, ladder depth 0..5, haystack = **23426** docs (base publications removed).
- `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR`: recall@10 0.41 → 0.37 (Δ=+0.04); mean cosine 0.751 → 0.746.

## Per-step cosine drop by swap mode
- `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR`: B=-0.0003, C=0.0010, D=0.0011, F=0.0017
