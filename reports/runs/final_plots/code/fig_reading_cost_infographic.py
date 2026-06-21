"""Reproducible rebuild of the XRC/RRC/ARI 'reading cost' infographic.

Faithful re-creation of the one-off slide figure (Screenshot 2026-06-10): a single
vertical ranked-list bar on a log-compressed rank axis, with the same-language gold
(rank 2) and cross-language gold (rank 7) marked, the XRC reading-cost multiplier as
a bracket, and the three RRC/ARI coverage zones colour-coded down the list.
Labels use the paper's same-/cross-language terminology.

Numbers are embeddinggemma's real values on the pooled GP+EPO both-gold domain
(n=459), from E_ari_decomposition.csv / E_cost_frontier.csv: RRC@100=78%,
+15% deeper pool, L_inf=7%, ARI@100=0.32, XRC=5x. The two ranked dots illustrate
that XRC at the median same-language depth (rank 2 -> rank 10); the band
percentages are the real aggregate statistics.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# real values: embeddinggemma, pooled GP+EPO (E_ari_decomposition / E_cost_frontier).
# Ranks illustrate the reported XRC=5x at the median same-language depth (2 -> 10).
HOME_RANK, FOREIGN_RANK = 2, 10
XRC = 5
RRC100, DEEPER, LINF, ARI = 78, 15, 7, 0.32

# palette
BG = "#0c1a18"
GREEN = "#1b917a"     # rank 1-100   = RRC@100 "reachable" zone
ORANGE = "#9a6717"    # rank 100-1000
RED = "#8c3030"       # rank > 1000
INK = "#eaf0ee"
SUB = "#90a1a1"
LEAD = "#6f8587"

BAR_L, BAR_R = 4.35, 5.65
TOP, BOTTOM = 1, 1500


def main():
    fig, ax = plt.subplots(figsize=(11.8, 6.9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_yscale("log")
    ax.set_ylim(TOP, BOTTOM)
    ax.invert_yaxis()  # rank 1 at the top
    ax.axis("off")

    # ---- the ranked-list bar, coloured by coverage zone -------------------
    for y0, y1, c in [(1, 100, GREEN), (100, 1000, ORANGE), (1000, BOTTOM, RED)]:
        ax.fill_betweenx([y0, y1], BAR_L, BAR_R, color=c, zorder=2)

    # rank gridlines + left-side rank labels (band boundaries)
    for r in (1, 100, 1000):
        ax.plot([BAR_L, BAR_R], [r, r], color=BG, lw=1.2, alpha=0.55, zorder=3)
        ax.text(BAR_L - 0.18, r, str(r), color=SUB, fontsize=12,
                ha="right", va="center")

    # ---- home copy / foreign twin markers ---------------------------------
    ax.plot(BAR_L, HOME_RANK, "o", ms=12, mfc=BG, mec=INK, mew=2.0, zorder=6)
    ax.plot(BAR_L, FOREIGN_RANK, "o", ms=12, mfc=INK, mec=INK, zorder=6)
    for r, lab in [(HOME_RANK, f"Same-language gold · rank {HOME_RANK}"),
                   (FOREIGN_RANK, f"Cross-language gold · rank {FOREIGN_RANK}")]:
        ax.plot([3.65, BAR_L], [r, r], ls="--", lw=1.0, color=LEAD, zorder=4)
        ax.text(3.6, r, lab, color=INK, fontsize=12.5, ha="right", va="center")

    # ---- XRC bracket spanning rank 2..7 -----------------------------------
    bx = 0.85
    ax.plot([bx, bx], [HOME_RANK, FOREIGN_RANK], color=SUB, lw=1.6, zorder=4)
    for r in (HOME_RANK, FOREIGN_RANK):
        ax.plot([bx, bx + 0.18], [r, r], color=SUB, lw=1.6, zorder=4)
    gm = np.sqrt(HOME_RANK * FOREIGN_RANK)
    ax.text(bx - 0.22, gm, f"XRC\n≈ {XRC:g}×", color=INK, fontsize=14,
            ha="right", va="center", fontweight="bold", linespacing=1.3)

    # ---- right-side zone callouts -----------------------------------------
    def callout(y_anchor, swatch, headline, sub):
        ax.plot([BAR_R, 6.25], [y_anchor, y_anchor], ls="--", lw=1.0,
                color=LEAD, zorder=4)
        ax.scatter(6.45, y_anchor, s=150, marker="s", color=swatch,
                   edgecolors="none", zorder=5)
        ax.text(6.7, y_anchor, headline, color=INK, fontsize=14.5,
                ha="left", va="center", fontweight="bold")
        ax.annotate(sub, xy=(6.7, y_anchor), xytext=(0, -16),
                    textcoords="offset points", color=SUB, fontsize=11.5, ha="left")

    callout(88, GREEN, f"{RRC100}% re-ranker can reach", "RRC@100")
    callout(380, ORANGE, f"+{DEEPER}% need a deeper pool", "RRC@1000 − RRC@100")
    callout(1050, RED, f"{LINF}% — alignment only", "L∞ floor")
    ax.annotate(f"ARI@100 ≈ {ARI:g}", xy=(6.7, 1050), xytext=(0, -36),
                textcoords="offset points", color=SUB, fontsize=12, ha="left")

    # ---- titles / footers (figure coords, outside the plot area) ----------
    fig.subplots_adjust(top=0.85, bottom=0.09, left=0.04, right=0.98)
    fig.text(0.51, 0.955, "Ranked search results", color=INK, fontsize=18,
             ha="center", va="center", fontweight="bold")
    fig.text(0.51, 0.905, "rank 1 = top of the list (scale compressed)",
             color=SUB, fontsize=12, ha="center", va="center")
    fig.text(0.51, 0.035, "deeper into the list ↓", color=SUB, fontsize=12.5,
             ha="center", va="center")

    out = Path(__file__).resolve().parents[1] / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = out / f"fig_reading_cost_infographic.{ext}"
        fig.savefig(p, dpi=200, facecolor=BG)
        print("wrote", p)


if __name__ == "__main__":
    main()
