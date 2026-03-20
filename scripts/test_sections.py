"""Quick smoke-test for cross-lingual section extraction.

Run with:  uv run python scripts/test_sections.py
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.multi_lingual_qac.dataloaders.wikidata import _parse_sections

DATA_DIR = Path("data/WIKIDATA/prepared/pages")
LANGS_TO_TEST = ["en", "de", "fr", "fa", "ar", "zh", "ja"]
RECORDS_PER_LANG = 3


def load_records(lang: str, n: int) -> list[dict]:
    path = DATA_DIR / f"{lang}.jsonl.gz"
    if not path.exists():
        return []
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            records.append(json.loads(line))
            if len(records) >= n:
                break
    return records


def show_sections(lang: str) -> None:
    records = load_records(lang, RECORDS_PER_LANG)
    if not records:
        print(f"[{lang}] No data found.\n")
        return

    print(f"=== {lang} ({len(records)} records) ===")
    for rec in records:
        sections = _parse_sections(rec["raw_wikitext"])
        print(f"  {rec['wikidata_label']} ({rec['qid']}) -> {len(sections)} sections")
        for s in sections:
            norm = s["title_normalized"] if s["title_normalized"] else "?"
            snippet = s["content"][:70].replace("\n", " ")
            print(f"    [L{s['level']}] {repr(s['title_raw'])} -> {norm} | {snippet}")
    print()


def cross_lang_alignment_demo() -> None:
    """Show which sections align across languages for shared QIDs."""
    print("=== Cross-language section alignment demo ===")

    # Build: qid -> lang -> [title_normalized]
    by_qid: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for lang in LANGS_TO_TEST:
        for rec in load_records(lang, 10):
            qid = rec["qid"]
            sections = _parse_sections(rec["raw_wikitext"])
            known = [s["title_normalized"] for s in sections if s["title_normalized"]]
            by_qid[qid][lang] = known

    for qid, langs in list(by_qid.items())[:5]:
        all_keys: set[str] = set()
        for keys in langs.values():
            all_keys.update(keys)
        lang_labels = [f"{la}({len(keys)})" for la, keys in langs.items()]
        aligned = [k for k in all_keys if sum(1 for keys in langs.values() if k in keys) > 1]
        print(f"  QID {qid}: langs={lang_labels}")
        print(f"    Aligned section keys ({len(aligned)}): {aligned}")
    print()


if __name__ == "__main__":
    for lang in LANGS_TO_TEST:
        show_sections(lang)
    cross_lang_alignment_demo()
