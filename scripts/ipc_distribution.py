"""Plot IPC subclass distribution across sources and flag under-represented classes.

Usage:
    uv run python scripts/ipc_distribution.py \
        --epo-csv data/EPO/multilingual_corpus.csv \
        --gp-csv data/google_patents/multilingual_corpus.csv \
        --output-plot reports/ipc_distribution.png \
        --output-md reports/ipc_analysis.md
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CHEMISTRY_CLASSES = {
    "C01": "Inorganic chemistry",
    "C02": "Water treatment",
    "C03": "Glass / mineral fibres",
    "C04": "Cements / ceramics",
    "C05": "Fertilisers",
    "C06": "Explosives / matches",
    "C07": "Organic chemistry",
    "C08": "Polymers",
    "C09": "Dyes / adhesives / coatings",
    "C10": "Petroleum / gas / coke",
    "C11": "Fats / detergents / candles",
    "C12": "Biochemistry / fermentation",
    "C13": "Sugar industry",
    "C14": "Skins / hides",
    "C21": "Iron / steel metallurgy",
    "C22": "Non-ferrous metallurgy / alloys",
    "C23": "Coating / surface treatment",
    "C25": "Electrolytic processes",
    "A61": "Pharmaceuticals / medical",
    "A23": "Food / beverages",
    "A01": "Agriculture / forestry",
    "B01": "Separation / mixing / catalysis",
}


def _extract_ipc_counts(
    path: Path,
    source_name: str,
) -> Tuple[Counter, int]:
    """Count unique docs per top-level IPC class (first 3 chars)."""
    counter: Counter = Counter()
    seen: Set[str] = set()
    total_docs = 0
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pub = row["publication_number"]
            if pub in seen:
                continue
            seen.add(pub)
            total_docs += 1
            for code in (row.get("ipc_codes") or "").split("|"):
                code = code.strip()
                if not code:
                    continue
                top = code[:3].rstrip()
                if top:
                    counter[top] += 1
    return counter, total_docs


def _render_plot(
    gp_counts: Counter,
    epo_counts: Counter,
    output_path: Path,
    top_n: int = 25,
) -> None:
    all_classes = sorted(
        set(gp_counts) | set(epo_counts),
        key=lambda c: gp_counts.get(c, 0) + epo_counts.get(c, 0),
        reverse=True,
    )[:top_n]
    all_classes = list(reversed(all_classes))

    gp_vals = [gp_counts.get(c, 0) for c in all_classes]
    epo_vals = [epo_counts.get(c, 0) for c in all_classes]

    labels = []
    for c in all_classes:
        desc = CHEMISTRY_CLASSES.get(c, "")
        labels.append(f"{c} ({desc})" if desc else c)

    colors_chem = []
    for c in all_classes:
        if c in CHEMISTRY_CLASSES:
            colors_chem.append(True)
        else:
            colors_chem.append(False)

    fig, ax = plt.subplots(figsize=(12, max(8, len(all_classes) * 0.35)))
    y = range(len(all_classes))

    bars_gp = ax.barh(y, gp_vals, height=0.7, label="Google Patents", color="#4285F4", alpha=0.85)
    bars_epo = ax.barh(y, epo_vals, height=0.7, left=gp_vals, label="EPO", color="#EA4335", alpha=0.85)

    for i, c in enumerate(all_classes):
        total = gp_vals[i] + epo_vals[i]
        ax.text(total + 50, i, f"{total:,}", va="center", fontsize=8, color="#333")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    for i, is_chem in enumerate(colors_chem):
        if is_chem:
            ax.get_yticklabels()[i].set_fontweight("bold")

    ax.set_xlabel("Number of Documents")
    ax.set_title("IPC Subclass Distribution (Chemistry Patent Corpus)")
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def _render_analysis_md(
    gp_counts: Counter,
    epo_counts: Counter,
    gp_docs: int,
    epo_docs: int,
    output_path: Path,
    min_docs: int,
) -> None:
    combined = Counter()
    combined.update(gp_counts)
    combined.update(epo_counts)
    total_docs = gp_docs + epo_docs

    lines: List[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append("# IPC Subclass Distribution Analysis")
    lines.append(f"\nGenerated: {now}")
    lines.append(f"\nTotal documents: {total_docs:,} (Google Patents: {gp_docs:,}, EPO: {epo_docs:,})")

    lines.append("\n## Full IPC Class Table")
    lines.append("")
    lines.append("| IPC | Description | Combined | Google Patents | EPO | % of Total |")
    lines.append("|-----|------------|----------|---------------|-----|-----------|")
    for code, count in combined.most_common():
        desc = CHEMISTRY_CLASSES.get(code, "")
        gp = gp_counts.get(code, 0)
        epo = epo_counts.get(code, 0)
        pct = f"{count / total_docs * 100:.1f}%"
        lines.append(f"| {code} | {desc} | {count:,} | {gp:,} | {epo:,} | {pct} |")

    # Domain balance
    core_chem = sum(combined.get(f"C{i:02d}", 0) for i in range(1, 26))
    pharma = combined.get("A61", 0)
    other = sum(combined.values()) - core_chem - pharma

    lines.append("\n## Domain Balance")
    lines.append("")
    lines.append(f"| Category | Documents | % |")
    lines.append(f"|----------|----------|---|")
    total_ipc_hits = sum(combined.values())
    for cat, val in [("Core chemistry (C01-C25)", core_chem),
                     ("Pharma / medical (A61)", pharma),
                     ("Other (non-chemistry IPC)", other)]:
        pct = f"{val / total_ipc_hits * 100:.1f}%" if total_ipc_hits else "0%"
        lines.append(f"| {cat} | {val:,} | {pct} |")

    # Under-represented
    chem_codes = {c for c in CHEMISTRY_CLASSES if c.startswith("C")}
    under = []
    for code in sorted(chem_codes):
        count = combined.get(code, 0)
        if count < min_docs:
            under.append((code, CHEMISTRY_CLASSES.get(code, ""), count))

    lines.append(f"\n## Under-Represented Chemistry Classes (< {min_docs:,} docs)")
    lines.append("")
    if under:
        lines.append("These IPC classes should be prioritised for additional collection:\n")
        lines.append("| IPC | Description | Current Docs | Gap to Threshold |")
        lines.append("|-----|------------|-------------|-----------------|")
        for code, desc, count in under:
            gap = min_docs - count
            lines.append(f"| **{code}** | {desc} | {count:,} | {gap:,} needed |")
    else:
        lines.append(f"All chemistry IPC classes have >= {min_docs:,} documents.")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Analysis written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="IPC subclass distribution plot + analysis")
    parser.add_argument("--epo-csv", type=Path, default=None)
    parser.add_argument("--gp-csv", type=Path, default=None)
    parser.add_argument("--output-plot", type=Path, default=Path("reports/ipc_distribution.png"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/ipc_analysis.md"))
    parser.add_argument("--min-docs", type=int, default=500, help="Threshold below which a chemistry IPC class is flagged as under-represented")
    args = parser.parse_args()

    gp_counts, gp_docs = Counter(), 0
    epo_counts, epo_docs = Counter(), 0

    if args.gp_csv and args.gp_csv.exists():
        gp_counts, gp_docs = _extract_ipc_counts(args.gp_csv, "Google Patents")
        print(f"Google Patents: {gp_docs:,} docs, {len(gp_counts)} IPC classes")
    if args.epo_csv and args.epo_csv.exists():
        epo_counts, epo_docs = _extract_ipc_counts(args.epo_csv, "EPO")
        print(f"EPO: {epo_docs:,} docs, {len(epo_counts)} IPC classes")

    _render_plot(gp_counts, epo_counts, args.output_plot)
    _render_analysis_md(gp_counts, epo_counts, gp_docs, epo_docs, args.output_md, args.min_docs)


if __name__ == "__main__":
    main()
