"""
Debug script: saves the exact prompts that would be sent to the LLM
for the first document in technical mode, strategy 1. No API calls made.
"""
from pathlib import Path
import random

from multilingual_qa import (
    load_multilingual_corpus,
    pick_target_languages,
    _pick_context,
    _build_all_passages_text,
    _load_generation_prompt,
    _load_faithfulness_prompt,
    _load_quality_prompt,
    STRATEGY_RANDOM_ANY,
    MODE_TECHNICAL,
)

random.seed(42)

# Output directory
out_dir = Path("data/google_patents/qac/debug")
out_dir.mkdir(parents=True, exist_ok=True)

# Load corpus
corpus_path = Path("data/google_patents/multilingual_corpus.csv")
groups = load_multilingual_corpus(corpus_path)
pub_num = list(groups.keys())[0]
rows = groups[pub_num]

# Pick target language (forced to French for debugging)
target_lang = "fr"

# Build all passages
all_passages = _build_all_passages_text(rows)

# Pick context row (for metadata only)
context_row, context_text = _pick_context(rows, target_lang)

# ---- GENERATION ----
gen_prompt = _load_generation_prompt(MODE_TECHNICAL, target_lang)
gen_file = out_dir / "1_generation_prompt.txt"
gen_file.write_text(
    f"=== SYSTEM MESSAGE ===\n\n{gen_prompt}\n\n"
    f"=== USER MESSAGE ===\n\n{all_passages}\n",
    encoding="utf-8",
)

# ---- FAITHFULNESS ----
faith_prompt = _load_faithfulness_prompt()
fake_qa = [
    {"question": "<Q1 would be here>", "answer": "<A1 would be here>"},
    {"question": "<Q2 would be here>", "answer": "<A2 would be here>"},
    {"question": "<Q3 would be here>", "answer": "<A3 would be here>"},
]
candidates = "\n\n".join(
    f"Candidate {i}:\n  Question: {qa['question']}\n  Answer: {qa['answer']}"
    for i, qa in enumerate(fake_qa)
)
faith_user = f"{all_passages}\n\n{candidates}"
faith_file = out_dir / "2_faithfulness_prompt.txt"
faith_file.write_text(
    f"=== SYSTEM MESSAGE ===\n\n{faith_prompt}\n\n"
    f"=== USER MESSAGE ===\n\n{faith_user}\n",
    encoding="utf-8",
)

# ---- QUALITY ----
qual_prompt = _load_quality_prompt(MODE_TECHNICAL)
qual_user = f"{all_passages}\n\n{candidates}"
qual_file = out_dir / "3_quality_prompt.txt"
qual_file.write_text(
    f"=== SYSTEM MESSAGE ===\n\n{qual_prompt}\n\n"
    f"=== USER MESSAGE ===\n\n{qual_user}\n",
    encoding="utf-8",
)

print(f"Document: {pub_num}")
print(f"Available languages: {[r['language'] for r in rows]}")
print(f"Target language: {target_lang}")
print(f"Context row language: {context_row.get('language')}")
print()
print(f"Saved full prompts to:")
print(f"  {gen_file}  ({gen_file.stat().st_size} bytes)")
print(f"  {faith_file}  ({faith_file.stat().st_size} bytes)")
print(f"  {qual_file}  ({qual_file.stat().st_size} bytes)")
print()
print("No API calls were made.")

