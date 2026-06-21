"""Explanatory figure for the coverage-depth metrics k_same, k_cross, k*_80.

Shows the empirical gold-coverage curves (fraction of queries whose gold is found
within the top-k) for same-language gold, cross-language gold, and BOTH (joint),
for the headline model embeddinggemma over the pooled GP+EPO both-gold domain
(n=459). The three metrics are exactly where each curve crosses the 80% target:
k_same=36, k_cross=121, k*_80=147 (matches the main table).

The plot visualizes the inversion that defines the metric: Recall@k fixes the
depth (k=10) and reads coverage UPWARD; coverage-depth fixes coverage (80%) and
reads the depth ACROSS.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import claimE_metrics
import fp_common as fp

MODEL = "google/embeddinggemma-300m"
TAU = 0.80
C_SAME, C_CROSS, C_JOINT = "#4c72b0", "#c44e52", "#7a4fa3"


def _domain_ranks():
    gp = claimE_metrics._first_ranks()
    epo = fp.epo_first_ranks()
    dom = pd.concat([gp, epo], ignore_index=True)
    dom = dom[(dom["n_gold_same"] > 0) & (dom["n_gold_cross"] > 0)]
    g = dom[dom["model"] == MODEL]
    same = g["first_same_rank"].to_numpy(float)
    cross = g["first_cross_rank"].to_numpy(float)
    return same, cross, np.maximum(same, cross), len(g)


def _cross80(arr):
    for k in range(1, 1001):
        if np.mean(arr <= k) >= TAU:
            return k
    return 1001


def main():
    same, cross, joint, n = _domain_ranks()
    ks = np.arange(1, 1001)
    cov = lambda a: np.array([np.mean(a <= k) for k in ks])
    cov_same, cov_cross, cov_joint = cov(same), cov(cross), cov(joint)
    k_same, k_cross, k_star = _cross80(same), _cross80(cross), _cross80(joint)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.plot(ks, cov_same, color=C_SAME, lw=2.4, label=r"same-language gold  ($k_{\mathrm{same}}$)")
    ax.plot(ks, cov_cross, color=C_CROSS, lw=2.4, label=r"cross-language gold  ($k_{\mathrm{cross}}$)")
    ax.plot(ks, cov_joint, color=C_JOINT, lw=2.4, ls="--",
            label=r"both golds  ($k^{\star}_{80}$)")

    ax.set_xscale("log")
    ax.set_xlim(1, 1000)
    ax.set_ylim(0, 1.0)

    # 80% coverage target
    ax.axhline(TAU, color="#444", lw=1.2, ls=":")
    ax.text(1.05, TAU + 0.015, "80% coverage target", fontsize=10, color="#444")

    # drop-lines + dots at each crossing; values are read from the color-coded box
    # (top-left) because k_cross and k_star sit too close on the log axis to label
    # at the tick positions without overlap.
    for k, c in [(k_same, C_SAME), (k_cross, C_CROSS), (k_star, C_JOINT)]:
        ax.plot([k, k], [0, TAU], color=c, lw=1.3, ls="--", alpha=0.8)
        ax.plot(k, TAU, "o", color=c, ms=7, zorder=5)
    for yy, c, txt in [(0.74, C_SAME, rf"$k_{{\mathrm{{same}}}}={k_same}$"),
                       (0.665, C_CROSS, rf"$k_{{\mathrm{{cross}}}}={k_cross}$"),
                       (0.59, C_JOINT, rf"$k^{{\star}}_{{80}}={k_star}$")]:
        ax.text(1.15, yy, txt, color=c, fontsize=11.5, fontweight="bold", va="center")

    # Recall@10 reference: fix depth, read coverage up
    ax.axvline(10, color="#999", lw=1.0, ls="-", alpha=0.6)
    ax.annotate("Recall@10\nfixes depth,\nreads coverage ↑",
                xy=(10, 0.30), xytext=(2.0, 0.13), fontsize=9, color="#666",
                arrowprops=dict(arrowstyle="->", color="#999", lw=1.0))

    # coverage-depth direction: fix 80%, read depth across
    ax.annotate("coverage depth: fix 80%, read depth →",
                xy=(k_cross, TAU), xytext=(150, 0.55), fontsize=9.5, color="#333")

    # cross-lingual penalty bracket between k_same and k_cross
    ax.annotate("", xy=(k_same, 0.86), xytext=(k_cross, 0.86),
                arrowprops=dict(arrowstyle="<->", color="#c44e52", lw=1.4))
    ax.text(np.sqrt(k_same * k_cross), 0.885, "cross-lingual penalty",
            ha="center", fontsize=9.5, color="#c44e52", fontweight="bold")

    ax.set_xlabel("retrieval depth  $k$  (top-$k$ read; log scale)", fontsize=12, labelpad=8)
    ax.set_ylabel("coverage: fraction of queries with gold within top-$k$", fontsize=11.5)
    ax.set_title(f"Reading depth to reach 80% gold coverage  (embeddinggemma, n={n})",
                 fontsize=13.5, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10.5, framealpha=0.95)
    ax.grid(True, which="both", axis="both", alpha=0.18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(bottom=0.12)

    out = fp.FINAL / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = out / f"fig_k_explainer.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print("wrote", p)
    print(f"k_same={k_same}  k_cross={k_cross}  k*={k_star}  (table: 36/121/147)")


if __name__ == "__main__":
    main()
