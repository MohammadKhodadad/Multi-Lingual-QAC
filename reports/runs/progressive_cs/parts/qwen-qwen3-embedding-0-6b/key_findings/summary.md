# Progressive code-switching — retrieval decay
- **28** base documents, ladder depth 0..5, haystack = **23426** docs (base publications removed).
- `Qwen/Qwen3-Embedding-0.6B`: recall@10 0.63 → 0.57 (Δ=+0.05); mean cosine 0.557 → 0.550.

## Per-step cosine drop by swap mode
- `Qwen/Qwen3-Embedding-0.6B`: B=-0.0010, C=-0.0003, D=0.0022, F=0.0030
