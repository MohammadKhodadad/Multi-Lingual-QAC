"""
EXTRA (round-1 additive analysis) — B1 fix: directional CLIR hub-and-spoke re-read.

The correctness critic flagged ONE mismatch (B1): round02's "English is the easiest target" is FALSE.
This script produces the replacement numbers for the writer, reading ONLY two on-disk CSVs (no parquet,
no embeddings, no API):
    reports/runs/chem_patents/experimental_plots/round02_directional_clir/pair_recall.csv
    reports/runs/chem_patents/experimental_plots/round02_directional_clir/asymmetry.csv

Aggregation matches round02's `mp` matrix exactly (8 reliable models, gte-base excluded; matrix-cell
mean = unweighted mean of per-model cell recalls per (qlang,dlang); column mean over qlang rows,
diagonal included). This reproduces the critic's recomputation: en 0.367 / fr 0.375 / zh 0.350 /
de 0.309  ->  fr (0.375) > en (0.367), so English is NOT the easiest target.

Outputs -> reports/runs/chem_patents/experimental_plots/extra_directional_hub/
    hub_scores.csv (target lang, mean-incoming recall [incl & excl diagonal], corpus share),
    summary.json (hardest directed edge, most asymmetric pair, hub ranking).

Run: /home/mehdi/Projects/Multi-Lingual-QAC/.venv/bin/python \
       reports/runs/chem_patents/experimental_codes/extra_directional_hub.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SLUG = "extra_directional_hub"
DEGENERATE_SHORT = "gte-base"
# corpus base share (round06_language_collapse summary.json, authoritative on disk)
CORPUS_SHARE = {"en": 0.461, "fr": 0.3749, "es": 0.0917, "de": 0.0682, "zh": 0.0043}


def main() -> None:
    out = C.round_dir(SLUG)
    src = C.PLOTS_DIR / "round02_directional_clir"
    pr = pd.read_csv(src / "pair_recall.csv")
    asym = pd.read_csv(src / "asymmetry.csv")

    pool = pr[pr["short"] != DEGENERATE_SHORT].copy()  # 8 reliable models

    # matrix cell = unweighted mean of per-model cell recalls (mirrors round02 mp)
    cell = pool.groupby(["qlang", "dlang"])["recall_at_10"].mean().reset_index()

    # hub score = mean INCOMING recall per target language (column mean over qlang rows)
    incl = cell.groupby("dlang")["recall_at_10"].mean()            # includes diagonal (matches critic)
    cross_cell = cell[cell.qlang != cell.dlang]
    excl = cross_cell.groupby("dlang")["recall_at_10"].mean()      # cross-only (true CLIR hub)

    hub = pd.DataFrame({
        "target_lang": incl.index,
        "mean_incoming_recall_incl_diag": incl.values.round(4),
        "mean_incoming_recall_cross_only": [round(float(excl.get(l, np.nan)), 4) for l in incl.index],
        "corpus_share": [CORPUS_SHARE.get(l, np.nan) for l in incl.index],
    }).sort_values("mean_incoming_recall_incl_diag", ascending=False).reset_index(drop=True)
    hub.to_csv(out / "hub_scores.csv", index=False)

    # hardest directed edge (lowest cell recall, cross-language only)
    hardest = cross_cell.sort_values("recall_at_10").iloc[0]
    easiest = cross_cell.sort_values("recall_at_10", ascending=False).iloc[0]

    # most asymmetric pair (largest |recall_xy - recall_yx|) from asymmetry.csv
    asym = asym.assign(absgap=asym["asymmetry_xy_minus_yx"].abs()).sort_values("absgap", ascending=False)
    top_asym = asym.iloc[0]

    # gate: fr > en on the incl-diagonal hub score
    fr = float(incl["fr"]); en = float(incl["en"])
    gate_fr_gt_en = fr > en

    summary = {
        "B1_replacement": {
            "claim_to_drop": "English is the easiest target (round02 L217-218).",
            "correct_hub_scores_incl_diag": {l: round(float(incl[l]), 3) for l in ["en", "fr", "zh", "de"]},
            "fr_gt_en_gate": gate_fr_gt_en,
            "easiest_target_is": str(incl.idxmax()),
            "hardest_target_is": str(incl.idxmin()),
        },
        "hardest_directed_edge": {
            "dir": f"{hardest.qlang}->{hardest.dlang}",
            "recall_at_10": round(float(hardest.recall_at_10), 3),
        },
        "easiest_cross_lingual_edge": {
            "dir": f"{easiest.qlang}->{easiest.dlang}",
            "recall_at_10": round(float(easiest.recall_at_10), 3),
        },
        "most_asymmetric_pair": {
            "pair": f"{top_asym['from']}<->{top_asym['to']}",
            "recall_from_to": round(float(top_asym["recall_xy"]), 3),
            "recall_to_from": round(float(top_asym["recall_yx"]), 3),
            "gap": round(float(top_asym["asymmetry_xy_minus_yx"]), 3),
        },
        "corpus_composition_caveat": (
            "Directional asymmetry partly reflects corpus language composition "
            f"(en {CORPUS_SHARE['en']:.0%} / zh {CORPUS_SHARE['zh']:.1%} of documents), not only encoder "
            "behaviour: the hub score correlates with how many target-language documents exist to find."),
        "writer_replacement_sentence": (
            f"the hardest direction is {hardest.qlang}->{hardest.dlang} (R@10 "
            f"{float(hardest.recall_at_10):.2f}) and the most asymmetric pair is "
            f"{top_asym['from']}<->{top_asym['to']} (gap {float(top_asym['asymmetry_xy_minus_yx']):+.2f}); "
            "no single language is a clean 'easiest target' (fr 0.375 ~ en 0.367), and what asymmetry "
            "exists partly tracks corpus composition."),
    }
    C.jdump(summary, out / "summary.json")

    print(f"[{SLUG}] hub scores (target lang, mean incoming recall):")
    print(hub.to_string(index=False))
    print(f"\n[{SLUG}] GATE fr({fr:.3f}) > en({en:.3f}): {gate_fr_gt_en}")
    print(f"[{SLUG}] hardest edge: {summary['hardest_directed_edge']['dir']} "
          f"= {summary['hardest_directed_edge']['recall_at_10']}")
    print(f"[{SLUG}] most asymmetric pair: {summary['most_asymmetric_pair']['pair']} "
          f"gap {summary['most_asymmetric_pair']['gap']:+}")
    print(f"[{SLUG}] wrote -> {out}")


if __name__ == "__main__":
    main()
