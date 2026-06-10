"""
EXTRA (round-4) — DO-NOW-4 (OPTIONAL): pre-stitched 2-panel PNGs for the two float merges.

Produces a single 2-up PNG for each of the two merges the writer is cutting the float count with:
  cp_fig06_07_mate.png   <- cp_fig06_mate_retrieval.png + cp_fig07_first_foreign_rank.png
  cp_fig09_10_collapse.png <- cp_fig09_language_collapse.png + cp_fig10_distractor_language.png

CRITICAL (per troubleshooter): LOAD the two already-rendered, correctness-verified per-panel PNGs and
place them side by side with imread/imshow. Do NOT re-plot from the per-query CSVs — that risks the
merged panel diverging from the verified key_findings figures. This script touches NO data.

The writer may instead use a LaTeX subfigure (preferred default); this stitch is the fallback if
subfigure layout fights the column width.

Outputs DIRECTLY into paper/figures/ (the merged assets are paper-bound):
    paper/figures/cp_fig06_07_mate.png
    paper/figures/cp_fig09_10_collapse.png

Run:  /home/mehdi/Projects/Multi-Lingual-QAC/.venv/bin/python \
        reports/runs/chem_patents/experimental_codes/stitch_merged_panels.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[4]
FIG = REPO / "paper" / "figures"
DPI = 140

MERGES = [
    ("cp_fig06_07_mate.png",
     ["cp_fig06_mate_retrieval.png", "cp_fig07_first_foreign_rank.png"]),
    ("cp_fig09_10_collapse.png",
     ["cp_fig09_language_collapse.png", "cp_fig10_distractor_language.png"]),
]


def stitch(out_name: str, panels: list[str]) -> None:
    imgs = []
    for p in panels:
        src = FIG / p
        if not src.is_file():
            raise FileNotFoundError(f"missing source panel: {src}")
        imgs.append(mpimg.imread(src))
    # width ratios proportional to each panel's pixel width so neither is squeezed
    widths = [im.shape[1] for im in imgs]
    heights = [im.shape[0] for im in imgs]
    fig_h = 5.4
    fig_w = fig_h * sum(widths) / max(heights)
    fig, axes = plt.subplots(1, len(imgs), figsize=(fig_w, fig_h),
                             gridspec_kw={"width_ratios": widths})
    for ax, im in zip(axes, imgs):
        ax.imshow(im)
        ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
    fig.savefig(FIG / out_name, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[stitch] wrote {FIG / out_name}  <- {panels}  (sizes {list(zip(widths, heights))})")


def main() -> None:
    for out_name, panels in MERGES:
        stitch(out_name, panels)
    print(f"[stitch] done -> {FIG}")


if __name__ == "__main__":
    main()
