"""
Measure how much of "case B" can be fixed with the ChEBI Wikipedia name cache.

Case B = the translated Chinese (zh) docs at the tail of the Google-Patents
corpus that contain >=1 Latin word (>=2 letters) in title+abstract. A Latin span
is "fixable" if it matches a ChEBI concept (via the same name index the
alias-graph builder uses) whose cache entry carries a Chinese (`zh`) name -- i.e.
the Latin term could be swapped for its Chinese ChEBI name.

Read-only. Reuses src.alias_graph.matching so the match universe is identical to
the production pipeline. Re-run after extending wiki_names_cache.json to see the
delta. Outputs land in reports/runs/zh_latin_fixability/.

    python reports/runs/zh_latin_fixability/experimental_codes/measure_case_b_fixability.py
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from src.alias_graph.chebi import load_chebi_graph  # noqa: E402
from src.alias_graph.matching import (  # noqa: E402
    build_name_index,
    scan_corpus,
    _normalize,
)
from src.alias_graph.wikidata_names import _load_cache  # noqa: E402

CORPUS = REPO / "data" / "google_patents" / "multilingual_corpus.csv"
CHEBI_DIR = REPO / "data" / "chebi"
CACHE = CHEBI_DIR / "wiki_names_cache.json"
OUT_DIR = REPO / "reports" / "runs" / "zh_latin_fixability"

_TAG = re.compile(r"<[^>]+>")
_WORD = re.compile(r"[A-Za-z]+")

# Categories for the UNfixable, non-ChEBI Latin remainder (illustrative buckets).
ELEMENTS = {e.lower() for e in (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I "
    "Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt "
    "Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm"
).split()}
UNITS = {
    "wt", "vol", "mol", "mmol", "mass", "cm", "mm", "nm", "um", "pm", "kg", "mg",
    "ng", "pg", "ml", "ul", "ppm", "ppb", "rpm", "kpa", "mpa", "gpa", "pa", "hz",
    "khz", "mhz", "kj", "kv", "mv", "mah", "wh", "bar", "atm", "mw", "da", "kda",
    "iu", "rh", "od", "meq",
}
ACR = {
    "sup", "sub", "seq", "id", "rna", "dna", "mrna", "sirna", "trna", "cdna",
    "dpp", "kras", "uv", "ir", "nmr", "hplc", "pcr", "ph", "fc", "sd", "sem",
    "lc", "ms", "gc", "tlc", "formula", "sequence",
}
ROMAN = {"ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii"}
PROSE = {
    "to", "by", "from", "and", "or", "of", "the", "an", "in", "on", "for",
    "with", "is", "are", "be", "as", "at", "that", "this", "less", "than",
    "more", "most", "not", "based", "whether", "such", "which", "into", "out",
    "over", "under", "between", "within", "can", "may", "will", "would", "each",
    "any", "all", "some", "no", "nor", "but", "if", "then", "so", "weight",
    "volume", "parts", "part", "amount", "content", "total", "wherein", "said",
    "having",
}


def _fixfield(row) -> str:
    title = str(row["title"]) if pd.notna(row["title"]) else ""
    abstract = str(row["abstract"]) if pd.notna(row["abstract"]) else ""
    return _TAG.sub(" ", f"{title} {abstract}")


def _latin_types(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text) if len(w) >= 2}


def _covered_tokens(doc_id: str, match_info: dict) -> set[str]:
    """Latin token-types that participate in a matched concept name for this doc."""
    cov: set[str] = set()
    for _cid, (surface, _lang) in match_info.get(doc_id, {}).items():
        cov.update(_normalize(surface))
    return cov


def _category(token: str) -> str:
    if token in UNITS:
        return "unit"
    if token in ELEMENTS:
        return "element symbol"
    if token in ROMAN or token in ACR:
        return "acronym / markup / roman"
    if token in PROSE:
        return "english prose (needs MT)"
    return "other non-chemical Latin"


def main() -> None:
    df = pd.read_csv(CORPUS)
    zh = df[df["language"] == "zh"].copy()
    zh["fixfield"] = zh.apply(_fixfield, axis=1)
    zh["ltypes"] = zh["fixfield"].apply(_latin_types)
    case_b = zh[zh["ltypes"].map(len) > 0].copy()
    rows = case_b.to_dict("records")
    n_b = len(rows)

    graph = load_chebi_graph(CHEBI_DIR, "full")
    cache = _load_cache(CACHE)
    zh_concepts = {c for c, v in cache.items() if isinstance(v, dict) and v.get("zh")}

    index_any = build_name_index(graph, {})
    # zh-restricted index: keep a Latin name only if one of its concepts has a zh title.
    fix_latin = {
        n: (ids & zh_concepts) for n, ids in index_any.latin.items()
    }
    fix_latin = {n: ids for n, ids in fix_latin.items() if ids}
    index_zh = build_name_index(graph, {})
    index_zh.latin = fix_latin
    index_zh.cjk = {}
    index_zh.surface = {n: index_any.surface.get(n, n) for n in fix_latin}
    index_zh.langs = {n: index_any.langs.get(n, set()) for n in fix_latin}

    _, any_match, _ = scan_corpus(rows, index_any, field="fixfield")
    _, zh_match, _ = scan_corpus(rows, index_zh, field="fixfield")

    n_any_doc = sum(1 for r in rows if any_match.get(r["id"]))
    n_fix_doc = sum(1 for r in rows if zh_match.get(r["id"]))
    n_full_doc = 0
    n_types_total = 0
    n_types_fix = 0
    cat_types: Counter = Counter()
    cat_occ: Counter = Counter()
    cat_examples: dict[str, set[str]] = {}
    per_doc: list[dict] = []

    for r in rows:
        did = r["id"]
        ltypes = set(r["ltypes"])
        occ = Counter(w.lower() for w in _WORD.findall(r["fixfield"]) if len(w) >= 2)
        fix_cov = _covered_tokens(did, zh_match) & ltypes
        any_cov = _covered_tokens(did, any_match) & ltypes
        n_types_total += len(ltypes)
        n_types_fix += len(fix_cov)
        fully = bool(ltypes) and ltypes <= fix_cov
        if fully:
            n_full_doc += 1
        for t in ltypes:
            if t in fix_cov:
                continue
            if t in any_cov:
                cat = "chemical (ChEBI match, no Chinese Wikipedia article)"
            else:
                cat = _category(t)
            cat_types[cat] += 1
            cat_occ[cat] += occ[t]
            cat_examples.setdefault(cat, set())
            if len(cat_examples[cat]) < 12:
                cat_examples[cat].add(t)
        fix_terms = sorted(
            {
                index_zh.surface.get(s, s)
                for cid, (s, _l) in zh_match.get(did, {}).items()
            }
        )
        per_doc.append({
            "id": did,
            "n_latin_types": len(ltypes),
            "maps_to_chebi": len(any_cov),
            "zh_fixable": len(fix_cov),
            "fully_fixable": fully,
            "example_fixable_terms": "; ".join(fix_terms[:6]),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def pct(x: int, n: int) -> str:
        return f"{100 * x / n:.0f}%" if n else "n/a"

    summary = [
        "# Case-B fixability via the ChEBI Wikipedia name cache",
        "",
        f"- ChEBI concepts in cache with a Chinese name: **{len(zh_concepts)}**",
        f"- Case B (zh docs with >=1 Latin word): **{n_b}** of {len(zh)}",
        "",
        "| Definition of \"fixed\" | Count | % of case B |",
        "|---|---|---|",
        f"| >=1 non-Chinese word fixable (concept has zh) | {n_fix_doc} | {pct(n_fix_doc, n_b)} |",
        f"| Fully cleaned (all Latin words replaceable) | {n_full_doc} | {pct(n_full_doc, n_b)} |",
        f"| Latin word-TYPES fixable | {n_types_fix}/{n_types_total} | {pct(n_types_fix, n_types_total)} |",
        f"| Hard ceiling: >=1 word maps to ANY ChEBI concept | {n_any_doc} | {pct(n_any_doc, n_b)} |",
        "",
        f"**Hard ceiling**: {n_b - n_any_doc}/{n_b} "
        f"({pct(n_b - n_any_doc, n_b)}) case-B docs contain NO ChEBI-nameable Latin "
        "word (only units / element symbols / acronyms / English prose), so the "
        "\">=1 fixable\" rate cannot exceed the ceiling no matter how complete the cache is.",
        "",
        "## Unfixable Latin token-types by category",
        "",
        "| Category | Token-types | Occurrences | Examples |",
        "|---|---|---|---|",
    ]
    for cat, _ in cat_types.most_common():
        ex = ", ".join(sorted(cat_examples.get(cat, []))[:10])
        summary.append(f"| {cat} | {cat_types[cat]} | {cat_occ[cat]} | {ex} |")
    summary_text = "\n".join(summary) + "\n"
    (OUT_DIR / "case_b_summary.md").write_text(summary_text, encoding="utf-8")

    with (OUT_DIR / "case_b_per_doc.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_doc[0].keys()))
        w.writeheader()
        w.writerows(per_doc)

    with (OUT_DIR / "unfixable_breakdown.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "n_token_types", "n_occurrences", "examples"])
        for cat, _ in cat_types.most_common():
            w.writerow([
                cat, cat_types[cat], cat_occ[cat],
                "; ".join(sorted(cat_examples.get(cat, []))[:12]),
            ])

    print(summary_text)
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
