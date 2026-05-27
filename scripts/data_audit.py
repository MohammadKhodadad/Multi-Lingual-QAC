"""M-0.1.1 Per-source data audit.

Reads the EPO and Google Patents multilingual corpus CSVs (and optionally the
Google Patents raw NDJSON for IPC/CPC extraction), computes per-source stats,
cross-source overlaps and gaps, and writes a markdown report.

Usage:
    uv run python scripts/data_audit.py \
        --epo-csv data/EPO/multilingual_corpus.csv \
        --gp-csv data/google_patents/multilingual_corpus.csv \
        --gp-ndjson data/google_patents/chemistry_patents.ndjson \
        --output reports/data_audit.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def _dedup_key(publication_number: str, country_code: str) -> str:
    m = re.search(r"(\d{5,})", publication_number)
    bare = m.group(1) if m else publication_number
    return f"{country_code}_{bare}"


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _unique_pubs(rows: List[Dict[str, str]]) -> Set[str]:
    return {_dedup_key(r["publication_number"], r["country_code"]) for r in rows}


def _format_date(d: str) -> str:
    if len(d) == 8 and d.isdigit():
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    return d


def _load_ipc_from_ndjson(path: Path, pub_nums_needed: Set[str]) -> Dict[str, List[str]]:
    """Build publication_number -> list of IPC top-level codes from NDJSON."""
    pub_ipc: Dict[str, List[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pub = rec.get("publication_number", "")
            if pub not in pub_nums_needed:
                continue
            codes: List[str] = []
            for ipc_entry in rec.get("ipc", []) or []:
                code = (ipc_entry.get("code") or "").strip()
                if code:
                    top = code[:3].rstrip()
                    if top and top not in codes:
                        codes.append(top)
            if codes:
                pub_ipc[pub] = codes
    return pub_ipc


def _source_stats(
    rows: List[Dict[str, str]],
    source_name: str,
    ipc_map: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    pubs = set()
    lang_counter: Counter = Counter()
    country_counter: Counter = Counter()
    dates: List[str] = []
    has_abstract = 0
    has_first_claim = 0
    has_description = 0

    for r in rows:
        pub_key = _dedup_key(r["publication_number"], r["country_code"])
        pubs.add(pub_key)
        lang_counter[r["language"]] += 1
        country_counter[r["country_code"]] += 1
        if r.get("publication_date", "").strip():
            dates.append(r["publication_date"].strip())
        if r.get("abstract", "").strip():
            has_abstract += 1
        if r.get("first_claim", "").strip():
            has_first_claim += 1
        if r.get("description", "").strip():
            has_description += 1

    ipc_counter: Counter = Counter()
    ipc_available = False
    if ipc_map:
        ipc_available = True
        pub_nums_in_rows = {r["publication_number"] for r in rows}
        seen_pubs_ipc: Set[str] = set()
        for pub_num in pub_nums_in_rows:
            if pub_num in seen_pubs_ipc:
                continue
            seen_pubs_ipc.add(pub_num)
            for code in ipc_map.get(pub_num, []):
                ipc_counter[code] += 1

    sorted_dates = sorted(dates) if dates else []
    return {
        "source": source_name,
        "unique_pubs": len(pubs),
        "total_rows": len(rows),
        "languages": dict(lang_counter.most_common()),
        "country_codes": dict(country_counter.most_common()),
        "date_min": _format_date(sorted_dates[0]) if sorted_dates else "N/A",
        "date_max": _format_date(sorted_dates[-1]) if sorted_dates else "N/A",
        "has_abstract": has_abstract,
        "has_first_claim": has_first_claim,
        "has_description": has_description,
        "ipc_available": ipc_available,
        "ipc_distribution": dict(ipc_counter.most_common(20)) if ipc_counter else {},
    }


def _overlap_analysis(
    epo_rows: List[Dict[str, str]],
    gp_rows: List[Dict[str, str]],
) -> Dict[str, Any]:
    epo_keys = _unique_pubs(epo_rows)
    gp_keys = _unique_pubs(gp_rows)
    overlap = sorted(epo_keys & gp_keys)
    epo_only = len(epo_keys - gp_keys)
    gp_only = len(gp_keys - epo_keys)

    details: List[Dict[str, Any]] = []
    if overlap:
        epo_by_key = defaultdict(list)
        gp_by_key = defaultdict(list)
        for r in epo_rows:
            epo_by_key[_dedup_key(r["publication_number"], r["country_code"])].append(r)
        for r in gp_rows:
            gp_by_key[_dedup_key(r["publication_number"], r["country_code"])].append(r)

        for key in overlap[:20]:
            er = epo_by_key[key]
            gr = gp_by_key[key]
            details.append({
                "key": key,
                "epo_langs": sorted({r["language"] for r in er}),
                "gp_langs": sorted({r["language"] for r in gr}),
                "epo_has_abstract": any(r.get("abstract", "").strip() for r in er),
                "gp_has_abstract": any(r.get("abstract", "").strip() for r in gr),
                "epo_has_claims": any(r.get("first_claim", "").strip() for r in er),
                "gp_has_claims": any(r.get("first_claim", "").strip() for r in gr),
            })

    return {
        "overlap_count": len(overlap),
        "epo_only": epo_only,
        "gp_only": gp_only,
        "merged_total": epo_only + gp_only + len(overlap),
        "details": details,
    }


def _render_markdown(
    epo_stats: Optional[Dict[str, Any]],
    gp_stats: Optional[Dict[str, Any]],
    overlap: Optional[Dict[str, Any]],
    epo_path: Optional[Path],
    gp_path: Optional[Path],
    gp_ndjson_path: Optional[Path],
) -> str:
    lines: List[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"# Data Audit Report")
    lines.append(f"")
    lines.append(f"Generated: {now}")
    lines.append("")

    # --- Summary table ---
    lines.append("## Summary")
    lines.append("")
    lines.append("| Source | Unique Docs | Total Rows | Languages | Date Range | IPC Available | File |")
    lines.append("|--------|------------|------------|-----------|------------|---------------|------|")
    for stats, path in [
        (gp_stats, gp_path),
        (epo_stats, epo_path),
    ]:
        if stats:
            langs = ", ".join(stats["languages"].keys())
            dr = f"{stats['date_min']} .. {stats['date_max']}"
            ipc = "Yes" if stats["ipc_available"] else "No (not in CSV)"
            fname = path.name if path else "N/A"
            lines.append(
                f"| {stats['source']} | {stats['unique_pubs']:,} | {stats['total_rows']:,} "
                f"| {langs} | {dr} | {ipc} | `{fname}` |"
            )
    lines.append("| USPTO | — | — | — | — | — | *Not yet ingested* |")
    lines.append("")

    # --- Per-source detail ---
    for stats, path in [
        (gp_stats, gp_path),
        (epo_stats, epo_path),
    ]:
        if not stats:
            continue
        lines.append(f"## {stats['source']}")
        lines.append("")
        lines.append(f"- **File**: `{path}`")
        lines.append(f"- **Unique documents**: {stats['unique_pubs']:,}")
        lines.append(f"- **Total rows**: {stats['total_rows']:,}")
        lines.append(f"- **Date range**: {stats['date_min']} to {stats['date_max']}")
        lines.append("")

        lines.append("### Language Distribution")
        lines.append("")
        lines.append("| Language | Rows |")
        lines.append("|----------|------|")
        for lang, count in sorted(stats["languages"].items()):
            lines.append(f"| {lang} | {count:,} |")
        lines.append("")

        lines.append("### Field Coverage")
        lines.append("")
        total = stats["total_rows"]
        lines.append(f"| Field | Rows with content | % |")
        lines.append(f"|-------|------------------|---|")
        for field, count in [
            ("abstract", stats["has_abstract"]),
            ("first_claim", stats["has_first_claim"]),
            ("description", stats["has_description"]),
        ]:
            pct = f"{count / total * 100:.1f}%" if total else "0%"
            lines.append(f"| {field} | {count:,} | {pct} |")
        lines.append("")

        lines.append("### Country Codes")
        lines.append("")
        for cc, count in sorted(stats["country_codes"].items()):
            lines.append(f"- `{cc}`: {count:,} rows")
        lines.append("")

        if stats["ipc_available"] and stats["ipc_distribution"]:
            lines.append("### IPC Class Distribution (top-level)")
            lines.append("")
            lines.append("| IPC Class | Documents |")
            lines.append("|-----------|----------|")
            for code, count in sorted(stats["ipc_distribution"].items(), key=lambda x: -x[1]):
                lines.append(f"| {code} | {count:,} |")
            lines.append("")
        elif not stats["ipc_available"]:
            lines.append("### IPC Class Distribution")
            lines.append("")
            lines.append("Not available in current CSV output. The EPO XML parser extracts IPC/CPC codes")
            lines.append("at parse time but `build_row_for_language` does not persist them to the corpus CSV.")
            lines.append("Chemistry filtering uses prefixes: C, A01N, A23L, A61K, A61P, B01D, B01F, B01J, B01L, C25, G01N, H01M.")
            lines.append("")

    # --- USPTO placeholder ---
    lines.append("## USPTO")
    lines.append("")
    lines.append("No data ingested yet. No USPTO loader or data files present in the project.")
    lines.append("")

    # --- Cross-source overlap ---
    if overlap:
        lines.append("## Cross-Source Overlap")
        lines.append("")
        lines.append(f"Dedup key: `country_code + bare doc-number` (e.g. `EP_4634118`).")
        lines.append("")
        lines.append(f"| Metric | Count |")
        lines.append(f"|--------|-------|")
        lines.append(f"| EPO-only documents | {overlap['epo_only']:,} |")
        lines.append(f"| Google Patents-only documents | {overlap['gp_only']:,} |")
        lines.append(f"| Duplicates (same patent in both) | {overlap['overlap_count']:,} |")
        lines.append(f"| **Merged total (deduplicated)** | **{overlap['merged_total']:,}** |")
        lines.append("")

        if overlap["details"]:
            lines.append("### Duplicate Details")
            lines.append("")
            lines.append("| Doc Key | EPO langs | GP langs | EPO abstract | GP abstract | EPO claims | GP claims |")
            lines.append("|---------|-----------|----------|-------------|-------------|------------|-----------|")
            for d in overlap["details"]:
                lines.append(
                    f"| {d['key']} "
                    f"| {','.join(d['epo_langs'])} "
                    f"| {','.join(d['gp_langs'])} "
                    f"| {'Yes' if d['epo_has_abstract'] else 'No'} "
                    f"| {'Yes' if d['gp_has_abstract'] else 'No'} "
                    f"| {'Yes' if d['epo_has_claims'] else 'No'} "
                    f"| {'Yes' if d['gp_has_claims'] else 'No'} |"
                )
            lines.append("")
        elif overlap["overlap_count"] == 0:
            lines.append("No duplicates found (the dedup script may have already removed them).")
            lines.append("")

    # --- Gaps ---
    if epo_stats and gp_stats:
        lines.append("## Gaps Across Sources")
        lines.append("")

        epo_langs = set(epo_stats["languages"].keys())
        gp_langs = set(gp_stats["languages"].keys())
        epo_only_langs = sorted(epo_langs - gp_langs)
        gp_only_langs = sorted(gp_langs - epo_langs)
        lines.append("### Language Gaps")
        lines.append("")
        if epo_only_langs:
            lines.append(f"- Languages in EPO only: {', '.join(epo_only_langs)}")
        if gp_only_langs:
            lines.append(f"- Languages in Google Patents only: {', '.join(gp_only_langs)}")
        if not epo_only_langs and not gp_only_langs:
            lines.append("- Both sources cover the same languages.")
        lines.append("")

        lines.append("### Date Coverage Gaps")
        lines.append("")
        lines.append(f"- Google Patents: {gp_stats['date_min']} to {gp_stats['date_max']}")
        lines.append(f"- EPO: {epo_stats['date_min']} to {epo_stats['date_max']}")
        epo_max = epo_stats["date_max"].replace("-", "")
        gp_min = gp_stats["date_min"].replace("-", "")
        gp_max = gp_stats["date_max"].replace("-", "")
        epo_min = epo_stats["date_min"].replace("-", "")
        if gp_max < epo_min or epo_max < gp_min:
            lines.append(f"- **Gap**: no date overlap between the two sources.")
        else:
            lines.append(f"- Date ranges overlap.")
        lines.append("")

        lines.append("### IPC Coverage Gaps")
        lines.append("")
        if gp_stats["ipc_available"] and not epo_stats["ipc_available"]:
            lines.append("- Google Patents has IPC data; EPO does not (not persisted to CSV).")
            lines.append(f"- Google Patents IPC classes: {', '.join(sorted(gp_stats['ipc_distribution'].keys()))}")
        elif gp_stats["ipc_available"] and epo_stats["ipc_available"]:
            gp_ipc = set(gp_stats["ipc_distribution"].keys())
            epo_ipc = set(epo_stats["ipc_distribution"].keys())
            gp_only_ipc = sorted(gp_ipc - epo_ipc)
            epo_only_ipc = sorted(epo_ipc - gp_ipc)
            if gp_only_ipc:
                lines.append(f"- IPC classes in Google Patents only: {', '.join(gp_only_ipc)}")
            if epo_only_ipc:
                lines.append(f"- IPC classes in EPO only: {', '.join(epo_only_ipc)}")
            if not gp_only_ipc and not epo_only_ipc:
                lines.append("- Both sources cover the same IPC classes.")
        else:
            lines.append("- IPC comparison not possible (one or both sources lack IPC data).")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="M-0.1.1 Per-source data audit")
    parser.add_argument("--epo-csv", type=Path, default=None, help="EPO multilingual corpus CSV")
    parser.add_argument("--gp-csv", type=Path, default=None, help="Google Patents multilingual corpus CSV")
    parser.add_argument("--gp-ndjson", type=Path, default=None, help="Google Patents raw NDJSON (for IPC extraction)")
    parser.add_argument("--output", type=Path, default=Path("reports/data_audit.md"), help="Output markdown path")
    args = parser.parse_args()

    epo_rows = _load_csv(args.epo_csv) if args.epo_csv and args.epo_csv.exists() else None
    gp_rows = _load_csv(args.gp_csv) if args.gp_csv and args.gp_csv.exists() else None

    gp_ipc_map: Optional[Dict[str, List[str]]] = None
    if gp_rows and args.gp_ndjson and args.gp_ndjson.exists():
        gp_pub_nums = {r["publication_number"] for r in gp_rows}
        print(f"Loading IPC from {args.gp_ndjson} for {len(gp_pub_nums)} GP publications...")
        gp_ipc_map = _load_ipc_from_ndjson(args.gp_ndjson, gp_pub_nums)
        print(f"  IPC data found for {len(gp_ipc_map)} publications.")

    epo_stats = _source_stats(epo_rows, "EPO") if epo_rows else None
    gp_stats = _source_stats(gp_rows, "Google Patents", ipc_map=gp_ipc_map) if gp_rows else None

    overlap = None
    if epo_rows and gp_rows:
        overlap = _overlap_analysis(epo_rows, gp_rows)

    md = _render_markdown(
        epo_stats, gp_stats, overlap,
        epo_path=args.epo_csv, gp_path=args.gp_csv, gp_ndjson_path=args.gp_ndjson,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    print(f"\nAudit report written to {args.output}")


if __name__ == "__main__":
    main()
