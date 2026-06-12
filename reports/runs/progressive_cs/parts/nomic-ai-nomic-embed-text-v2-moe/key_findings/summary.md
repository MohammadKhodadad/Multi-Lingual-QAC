# Progressive code-switching — retrieval decay
- **28** base documents, ladder depth 0..5, haystack = **23426** docs (base publications removed).
- `nomic-ai/nomic-embed-text-v2-moe`: recall@10 0.63 → 0.59 (Δ=+0.04); mean cosine 0.509 → 0.504.

## Per-step cosine drop by swap mode
- `nomic-ai/nomic-embed-text-v2-moe`: B=-0.0029, C=0.0002, D=0.0009, F=0.0037
