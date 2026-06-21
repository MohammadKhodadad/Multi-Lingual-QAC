#!/usr/bin/env python3
"""Reproduce the human-annotator score-distribution bar chart.

Reads the per-annotator overall scores (`total_score`, on a /10 scale) from
``Evaluated data_by_annotator.xlsx`` (sheet ``qac_with_modes``) and renders the
score distribution. Stdlib-only xlsx reader (openpyxl is not installed in the
venv). CPU-only; no models loaded.

Usage:
    .venv/bin/python scripts/plot_human_score_distribution.py
"""
from __future__ import annotations

import collections
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "Evaluated data_by_annotator.xlsx"
OUT_DIR = ROOT / "reports" / "human_eval"
SHEET = "qac_with_modes"
SCORE_COL = "total_score"

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RNS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

BAR_COLOR = "#1f5b8a"
BG = "#fbf8ef"
INK = "#33444f"


def _colnum(ref: str) -> int:
    letters = re.match(r"([A-Z]+)", ref).group(1)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_sheet(xlsx_path: Path, sheet_name: str) -> list[list[str | None]]:
    """Return the named sheet as a list of rows (list of cell strings)."""
    z = zipfile.ZipFile(xlsx_path)
    shared: list[str] = []
    if "xl/sharedStrings.xml" in z.namelist():
        sroot = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in sroot.findall(f"{NS}si"):
            shared.append("".join(t.text or "" for t in si.iter(f"{NS}t")))
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    rid2tgt = {r.get("Id"): r.get("Target") for r in rels.findall(f"{RNS}Relationship")}
    target = None
    for s in wb.find(f"{NS}sheets"):
        if s.get("name") == sheet_name:
            target = rid2tgt[s.get(RID)].lstrip("/")
            break
    if target is None:
        raise KeyError(f"sheet {sheet_name!r} not found")
    root = ET.fromstring(z.read(target))
    rows: list[dict[int, str | None]] = []
    for row in root.iter(f"{NS}row"):
        cells: dict[int, str | None] = {}
        for c in row.findall(f"{NS}c"):
            t, v, iss = c.get("t"), c.find(f"{NS}v"), c.find(f"{NS}is")
            if t == "s" and v is not None:
                val = shared[int(v.text)]
            elif t == "inlineStr" and iss is not None:
                val = "".join(x.text or "" for x in iss.iter(f"{NS}t"))
            elif v is not None:
                val = v.text
            else:
                val = None
            cells[_colnum(c.get("r"))] = val
        rows.append(cells)
    width = max((max(c) + 1 if c else 0) for c in rows)
    return [[c.get(i) for i in range(width)] for c in rows]


def main() -> None:
    rows = read_sheet(XLSX, SHEET)
    header = rows[0]
    j = header.index(SCORE_COL)
    scores = [int(float(r[j])) for r in rows[1:] if j < len(r) and r[j] not in (None, "")]
    counts = collections.Counter(scores)
    xs = list(range(min(counts), max(counts) + 1))  # 6..10, include empty bins
    ys = [counts.get(x, 0) for x in xs]
    n = sum(ys)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bars = ax.bar(xs, ys, width=0.6, color=BAR_COLOR, zorder=3)
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.8, str(y), ha="center", va="bottom",
                fontsize=13, color=INK)

    ax.set_title(f"Score distribution (n = {n})", fontsize=16,
                 fontweight="bold", color=INK, pad=14)
    ax.set_xlabel("Score (/10)", fontsize=13, fontweight="bold", color=INK)
    ax.set_xticks(xs)
    ax.set_ylim(0, 58)
    ax.set_yticks(range(0, 51, 5))
    ax.tick_params(colors=INK, labelsize=12, length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK)
    ax.margins(x=0.04)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"score_distribution.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
        print(f"wrote {out}")
    print(f"distribution: {dict(sorted(counts.items()))}  n={n}  mean={sum(scores)/n:.2f}")


if __name__ == "__main__":
    main()
