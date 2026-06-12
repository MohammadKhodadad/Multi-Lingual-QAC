"""
EXTRA (round-2 additive analysis) — the DEG gate + the TWO-TAX non-redundancy (DO-NOW-3).

(a) DEG gate (M1) — define "degenerate" operationally so the paper's 4 load-bearing "non-degenerate"
    uses + the WTA footnote are anchored to a single criterion. Emits three candidate rules and the
    membership of each:
      * DEG_strict   = (clir_at_10 < 0.10) AND (RRC_at_1000 < 0.10)  -> flags only gte-base
      * DEG_clir_only= (clir_at_10 < 0.10)                            -> flags {gte-base, e5-large-instruct}
      * DEG_paper    = DEG_strict OR (clir_at_10 < 0.10)              -> == DEG_clir_only here
    RECOMMENDED single clean criterion: DEG = clir_at_10 < 0.10 (flags exactly {gte, e5}; clean gap
    to SapBERT at 0.1788). The mate-hit@1000 floor is a corroborating signal for the caption, NOT the
    gate (the AND-gate misses e5, which has RRC@1000 0.6277 >= 0.10).

(b) TWO-TAX non-redundancy (M4) — the cross-lingual cost has two line-items measured by two different
    benchmarks:
      * reading-cost tax   = XRC50          (chem-patents: how deep you must read for the foreign twin)
      * confusability tax  = confusion_rate (alias-graph: a look-alike compound out-ranks the gold)
    If these were redundant, one benchmark would be padding. We test it with the cross-model Spearman
    rank correlation, joined on the shared `short` key, on (i) all 9 and (ii) the n=7 non-degenerate
    set (dropping {gte, e5}, which co-inflate both taxes and would manufacture a spurious correlation).
    The "non-redundant" claim rests on the n=7 rho being WEAK. The script gates the prose on the
    actual value.

Reads ONLY on-disk CSVs (no parquet, no API, no eval):
    chem  : experimental_plots/extra_xrc_reading_cost/xrc_per_model.csv   (clir_at_10, XRC50, short)
    chem  : experimental_plots/extra_xrc_reading_cost/rrc_per_model.csv   (RRC_at_1000, short)
    alias : reports/runs/alias_graph/experimental_plots/extra_confusion_severity/severity_split.csv

Writes to a NEW dir:
    reports/runs/chem_patents/experimental_plots/extra_two_tax_degeneracy/
Outputs: deg_flags.csv, two_tax_table.csv, summary.json, degeneracy_gap.png, two_tax_scatter.png
Paper figure target names: cp_fig20_degeneracy_gap.png, cp_fig21_two_tax.png

Run:  /home/mehdi/Projects/Multi-Lingual-QAC/.venv/bin/python \
        reports/runs/chem_patents/experimental_codes/extra_two_tax_degeneracy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SLUG = "extra_two_tax_degeneracy"
SEED = 20260610
DEG_CLIR_CUTOFF = 0.10
DEG_RRC_CUTOFF = 0.10


def main() -> None:
    from scipy.stats import spearmanr

    C.set_style()
    import matplotlib.pyplot as plt

    out = C.round_dir(SLUG)
    xrc_path = C.PLOTS_DIR / "extra_xrc_reading_cost" / "xrc_per_model.csv"
    rrc_path = C.PLOTS_DIR / "extra_xrc_reading_cost" / "rrc_per_model.csv"
    alias_path = (C.REPO_ROOT / "reports" / "runs" / "alias_graph" / "experimental_plots"
                  / "extra_confusion_severity" / "severity_split.csv")
    for p in (xrc_path, rrc_path, alias_path):
        if not p.is_file():
            raise FileNotFoundError(f"missing input CSV: {p}")

    xrc = pd.read_csv(xrc_path)
    rrc = pd.read_csv(rrc_path)
    alias = pd.read_csv(alias_path)
    xrc["XRC50f"] = pd.to_numeric(xrc["XRC50"], errors="coerce")

    # ---- DEG gate (join clir + rrc on short) ----
    deg = xrc[["short", "clir_at_10"]].merge(rrc[["short", "RRC_at_1000"]], on="short", how="inner")
    assert len(deg) == len(C.MODEL_ORDER), (
        f"expected {len(C.MODEL_ORDER)} models after clir/rrc join, got {len(deg)} "
        "(gte-multilingual-base is excluded as a loading artifact)"
    )
    deg["DEG_strict"] = (deg["clir_at_10"] < DEG_CLIR_CUTOFF) & (deg["RRC_at_1000"] < DEG_RRC_CUTOFF)
    deg["DEG_clir_only"] = deg["clir_at_10"] < DEG_CLIR_CUTOFF
    deg["DEG_paper"] = deg["DEG_strict"] | (deg["clir_at_10"] < DEG_CLIR_CUTOFF)
    deg = deg.sort_values("clir_at_10", ascending=False).reset_index(drop=True)
    deg.to_csv(out / "deg_flags.csv", index=False)

    strict_members = sorted(deg[deg["DEG_strict"]]["short"].tolist())
    clir_members = sorted(deg[deg["DEG_clir_only"]]["short"].tolist())
    paper_members = sorted(deg[deg["DEG_paper"]]["short"].tolist())
    deg_set = set(clir_members)  # the recommended single criterion

    # ---- two-tax table: join XRC50 (reading-cost tax) with confusion_rate (confusability tax) ----
    tt = (xrc[["short", "XRC50f", "clir_at_10"]]
          .merge(alias[["short", "confusion_rate", "sibling_win_rate"]], on="short", how="inner"))
    assert len(tt) == len(C.MODEL_ORDER), (
        f"expected {len(C.MODEL_ORDER)} models after chem/alias join on short, got {len(tt)} "
        "(gte-multilingual-base is excluded as a loading artifact)"
    )
    tt = tt.rename(columns={"XRC50f": "reading_cost_tax", "confusion_rate": "confusability_tax"})
    tt["DEG"] = tt["short"].isin(deg_set)
    tt = tt.sort_values("confusability_tax").reset_index(drop=True)
    tt[["short", "reading_cost_tax", "confusability_tax", "sibling_win_rate", "clir_at_10", "DEG"]] \
        .to_csv(out / "two_tax_table.csv", index=False)

    # ---- correlations: all-9 and n=7 non-degenerate. reading_cost_tax (XRC50) is NaN for gte
    #      (blank XRC50) -> for the all-9 Spearman drop NaN rows; the honest claim is the n=7 set. ----
    all9 = tt.dropna(subset=["reading_cost_tax", "confusability_tax"])
    rho9, p9 = spearmanr(all9["reading_cost_tax"].to_numpy(), all9["confusability_tax"].to_numpy())
    n7 = tt[~tt["DEG"]].dropna(subset=["reading_cost_tax", "confusability_tax"])
    rho7, p7 = spearmanr(n7["reading_cost_tax"].to_numpy(), n7["confusability_tax"].to_numpy())
    nonredundant = bool(abs(rho7) < 0.6)  # the threshold the troubleshoot plan names

    # ---- figure 1: degeneracy gap (clir_at_10 bar, sorted desc, cutoff at 0.10) ----
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    d = deg.sort_values("clir_at_10", ascending=False)
    xs = np.arange(len(d))
    colors = ["#cc6677" if v < DEG_CLIR_CUTOFF else "#4477aa" for v in d["clir_at_10"]]
    ax.bar(xs, d["clir_at_10"], color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(DEG_CLIR_CUTOFF, color="#cc6677", ls="--", lw=1.1)
    ax.text(len(d) - 0.5, DEG_CLIR_CUTOFF + 0.005, f"DEG cutoff {DEG_CLIR_CUTOFF}",
            ha="right", va="bottom", fontsize=8, color="#cc6677")
    for x, v in zip(xs, d["clir_at_10"]):
        ax.text(x, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels(d["short"], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("CLIR@10 (cross-lingual recall@10)")
    ax.set_title("Degeneracy gap: CLIR@10 with the 0.10 cutoff\n"
                 "(red = degenerate; clean gap SapBERT 0.179 vs e5 0.077)")
    fig.tight_layout(); fig.savefig(out / "degeneracy_gap.png"); plt.close(fig)

    # ---- figure 2: two-tax scatter (x=XRC50 reading-cost tax, y=confusion_rate confusability tax) ----
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    name2model = {C.short(m): m for m in C.MODEL_ORDER}
    for r in tt.itertuples():
        if not np.isfinite(r.reading_cost_tax):
            continue  # gte (blank XRC50) cannot be placed on the reading-cost axis
        col = C.MODEL_COLOR[name2model[r.short]]
        mk = "X" if r.DEG else "o"
        edge = "#d62728" if r.DEG else "black"
        ax.scatter(r.reading_cost_tax, r.confusability_tax, s=120, color=col, marker=mk,
                   zorder=3, edgecolor=edge, linewidth=1.4 if r.DEG else 0.5)
        ax.annotate(r.short, (r.reading_cost_tax, r.confusability_tax), fontsize=7.5,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("reading-cost tax  =  XRC50  (median reading-depth multiplier, log)")
    ax.set_ylabel("confusability tax  =  alias-graph confusion_rate")
    ax.set_title(f"Two taxes of cross-linguality (n=7 non-deg Spearman rho = {rho7:.2f})\n"
                 "(X / red edge = degenerate, excluded from the n=7 correlation)")
    fig.tight_layout(); fig.savefig(out / "two_tax_scatter.png"); plt.close(fig)

    # ---- join sanity values for the report ----
    eg = tt[tt.short == "embeddinggemma"].iloc[0]
    gr = tt[tt.short == "granite-278m"].iloc[0]

    summary = {
        "deg_gate": {
            "DEG_strict_clir<0.10_AND_rrc1000<0.10": strict_members,
            "DEG_clir_only_clir<0.10": clir_members,
            "DEG_paper_strict_OR_clir<0.10": paper_members,
            "RECOMMENDED": "DEG = clir_at_10 < 0.10",
            "recommended_members": clir_members,
            "matches_paper_exclusions_{gte,e5}":
                set(clir_members) == {"gte-base", "e5-large-instruct"},
            "note": ("AND-gate flags ONLY gte (e5 RRC@1000=0.6277 >= 0.10); the clean single "
                     "criterion clir_at_10<0.10 flags exactly {gte, e5} matching the paper."),
        },
        "two_tax": {
            "reading_cost_tax": "XRC50 (chem-patents)",
            "confusability_tax": "confusion_rate (alias-graph)",
            "spearman_rho_all9_finite": round(float(rho9), 4),
            "spearman_p_all9": round(float(p9), 4),
            "n_all9_finite": int(len(all9)),
            "spearman_rho_n7_nondeg": round(float(rho7), 4),
            "spearman_p_n7": round(float(p7), 4),
            "n7_models": n7["short"].tolist(),
            "nonredundant_supported_abs_rho_lt_0.6": nonredundant,
            "prose_guidance": (
                "n=7 rho weak (|rho|<0.6) => two taxes NON-REDUNDANT, both benchmarks necessary"
                if nonredundant else
                "n=7 |rho|>=0.6 => soften to the minimal C-P4 sentence; non-redundancy NOT supported"),
            "join_sanity_egemma_XRC50_3.5_conf_0.068":
                [float(eg.reading_cost_tax), float(eg.confusability_tax)],
            "join_sanity_granite_XRC50_1.25_conf_0.182":
                [float(gr.reading_cost_tax), float(gr.confusability_tax)],
        },
    }
    C.jdump(summary, out / "summary.json")

    print(f"[{SLUG}] DEG flags (sorted by CLIR@10 desc):")
    print(deg[["short", "clir_at_10", "RRC_at_1000", "DEG_strict", "DEG_clir_only"]].to_string(index=False))
    print(f"\n[{SLUG}] DEG_strict members    : {strict_members}")
    print(f"[{SLUG}] DEG_clir_only members : {clir_members}  "
          f"(matches paper {{gte,e5}}: {set(clir_members) == {'gte-base','e5-large-instruct'}})")
    print(f"\n[{SLUG}] two-tax table:")
    print(tt[["short", "reading_cost_tax", "confusability_tax", "DEG"]].to_string(index=False))
    print(f"\n[{SLUG}] Spearman rho(reading-cost tax, confusability tax):")
    print(f"          all-9 finite (n={len(all9)}): rho={rho9:.4f} p={p9:.4f}")
    print(f"          n=7 non-deg            : rho={rho7:.4f} p={p7:.4f}  "
          f"-> non-redundant {'SUPPORTED' if nonredundant else 'NOT supported'}")
    print(f"[{SLUG}] wrote -> {out}")


if __name__ == "__main__":
    main()
