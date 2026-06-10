"""
EXTRA (round-4 additive analysis) — tau-SENSITIVITY of the cost frontier (DO-NOW-3 / dreamer A3).

Replaces the "tau=0.40 untuned constant" footnote with a STATED stability interval, pre-empting the
"you tuned the threshold" objection. INLINE result (no body float).

Sweeps the CLIR@10 admission threshold tau over {0.30, 0.35, 0.40, 0.45, 0.50}. For each tau:
  * admitted set = {short : clir_at_10 >= tau}
  * cheapest-admitted = argmin XRC50 over admitted with FINITE XRC (matches extra_cost_frontier's
    sort_values("XRC50f") read-off)
Then reports the three keys:
  (i)   tau_admitted_stable_range  — tau-range over which admitted == {bge-m3, qwen3, embeddinggemma}
  (ii)  tau_cheapest_bge_range     — tau-range over which cheapest-admitted == bge-m3
  (iii) egemma_corner_tau_invariant — embeddinggemma is the unique global max-CLIR@10 corner for ALL tau

HONEST FINDING (verified by the troubleshooter, reproduced here): the admitted set and "cheapest =
bge-m3" are NARROWER than the dreamer assumed. tau=0.30 admits granite (XRC50 1.25) which becomes
cheapest (the recommendation FLIPS); tau>=~0.45 admits only embeddinggemma. The tau-invariance of
embeddinggemma's max-CLIR corner is the clean, unconditional part.

Reads ONLY the on-disk CSV emitted by extra_xrc_reading_cost (no parquet, no API, no eval):
    reports/runs/chem_patents/experimental_plots/extra_xrc_reading_cost/xrc_per_model.csv

Writes to the existing cost_frontier dir alongside the tau=0.40 read-off (does NOT overwrite
cost_frontier.csv / cost_frontier.png / summary.json):
    reports/runs/chem_patents/experimental_plots/extra_cost_frontier/tau_sweep.csv
    reports/runs/chem_patents/experimental_plots/extra_cost_frontier/tau_sweep_summary.json

Run:  /home/mehdi/Projects/Multi-Lingual-QAC/.venv/bin/python \
        reports/runs/chem_patents/experimental_codes/extra_tau_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SLUG = "extra_cost_frontier"  # write alongside the tau=0.40 read-off
TAU_GRID = [0.30, 0.35, 0.40, 0.45, 0.50]
FINE_GRID = [round(t, 3) for t in np.arange(0.30, 0.501, 0.005)]  # 0.005 steps for boundary detection


def admit(xrc: pd.DataFrame, tau: float):
    """admitted set + cheapest finite-XRC admitted model at threshold tau (matches cost_frontier)."""
    admitted = xrc[xrc["clir_at_10"] >= tau].copy()
    admitted_short = admitted.sort_values("clir_at_10", ascending=False)["short"].tolist()
    fin = admitted[np.isfinite(admitted["XRC50f"])].sort_values("XRC50f")
    if len(fin):
        cheapest = str(fin.iloc[0]["short"])
        cheapest_xrc = float(fin.iloc[0]["XRC50f"])
    else:
        cheapest, cheapest_xrc = "", float("nan")
    return admitted_short, cheapest, cheapest_xrc


def contiguous_range(taus, flags):
    """Return (lo, hi) of the maximal contiguous True-run that contains tau=0.40, else None."""
    pairs = list(zip(taus, flags))
    # find the run containing 0.40
    anchor = 0.40
    if anchor not in taus:
        return None
    ai = taus.index(anchor)
    if not flags[ai]:
        return None
    lo = anchor
    i = ai - 1
    while i >= 0 and flags[i]:
        lo = taus[i]; i -= 1
    hi = anchor
    i = ai + 1
    while i < len(taus) and flags[i]:
        hi = taus[i]; i += 1
    return (lo, hi)


def main() -> None:
    out = C.round_dir(SLUG)
    src = C.PLOTS_DIR / "extra_xrc_reading_cost" / "xrc_per_model.csv"
    if not src.is_file():
        raise FileNotFoundError(f"missing input CSV: {src}")
    xrc = pd.read_csv(src)
    xrc["XRC50f"] = pd.to_numeric(xrc["XRC50"], errors="coerce")

    # unique global max-CLIR@10 corner (tau-invariant by construction)
    top_clir_model = str(xrc.sort_values("clir_at_10", ascending=False).iloc[0]["short"])

    rows = []
    for tau in TAU_GRID:
        adm, cheap, cheap_xrc = admit(xrc, tau)
        rows.append({
            "tau": tau,
            "n_admitted": len(adm),
            "admitted_set": ";".join(sorted(adm)),
            "cheapest_admitted": cheap,
            "cheapest_XRC50": cheap_xrc,
            "max_clir_corner": top_clir_model,
        })
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out / "tau_sweep.csv", index=False)

    # ---- verify tau=0.40 reproduces the existing cost_frontier summary ----
    adm40, cheap40, _ = admit(xrc, 0.40)
    target_set = {"bge-m3", "qwen3-0.6B", "embeddinggemma"}
    gate_set_40 = set(adm40) == target_set
    gate_cheap_40 = cheap40 == "bge-m3"

    # ---- key (i): tau-range where admitted == {bge-m3, qwen3, embeddinggemma} ----
    fine_admit_sets = {t: set(admit(xrc, t)[0]) for t in FINE_GRID}
    fine_cheapest = {t: admit(xrc, t)[1] for t in FINE_GRID}
    set_flags = [fine_admit_sets[t] == target_set for t in FINE_GRID]
    set_range = contiguous_range(FINE_GRID, set_flags)

    # ---- key (ii): tau-range where cheapest-admitted == bge-m3 ----
    bge_flags = [fine_cheapest[t] == "bge-m3" for t in FINE_GRID]
    bge_range = contiguous_range(FINE_GRID, bge_flags)

    # ---- key (iii): egemma max-CLIR corner invariant for all tau ----
    egemma_invariant = bool(top_clir_model == "embeddinggemma")

    summary = {
        "method": "sweep CLIR@10 admission threshold tau; admitted={short: clir_at_10>=tau}; "
                  "cheapest=argmin finite XRC50 over admitted (matches extra_cost_frontier read-off).",
        "tau_grid": TAU_GRID,
        "fine_grid_step": 0.005,
        "tau_sweep_rows": rows,
        "verify_tau040_admitted_set": sorted(adm40),
        "verify_tau040_admitted_set_matches_cost_frontier": bool(gate_set_40),
        "verify_tau040_cheapest_is_bge_m3": bool(gate_cheap_40),
        "tau_admitted_stable_range": (
            f"[{set_range[0]:.3f}, {set_range[1]:.3f}]" if set_range else "none containing 0.40"),
        "tau_admitted_stable_range_raw": list(set_range) if set_range else None,
        "tau_cheapest_bge_range": (
            f"[{bge_range[0]:.3f}, {bge_range[1]:.3f}]" if bge_range else "none containing 0.40"),
        "tau_cheapest_bge_range_raw": list(bge_range) if bge_range else None,
        "egemma_corner_tau_invariant": egemma_invariant,
        "unique_max_clir_corner": top_clir_model,
        "HONEST_NARROW_BAND": (
            "cheapest admitted reader = bge-m3 holds only for tau in the bge-range above; below it "
            "granite-278m (lower-recall but cheaper-to-read, XRC50 1.25 < bge-m3 2.0) enters the "
            "admitted set and becomes cheapest (the recommendation FLIPS at tau=0.30); at the high "
            "end only embeddinggemma is admitted. embeddinggemma is the unique max-CLIR@10 corner for "
            "ALL tau (tau-invariant). Writer: do NOT overstate robustness — the rule is tau-sensitive "
            "at the low end; only the egemma corner is unconditional."),
    }
    C.jdump(summary, out / "tau_sweep_summary.json")

    print(f"[{SLUG}/tau_sweep] tau-sensitivity of the cost frontier:")
    print(sweep.to_string(index=False))
    print(f"\n[{SLUG}/tau_sweep] VERIFY tau=0.40 admitted={sorted(adm40)} "
          f"(matches cost_frontier set: {gate_set_40}); cheapest={cheap40} (=bge-m3: {gate_cheap_40})")
    print(f"[{SLUG}/tau_sweep] (i)  admitted=={sorted(target_set)} stable over tau "
          f"{summary['tau_admitted_stable_range']}")
    print(f"[{SLUG}/tau_sweep] (ii) cheapest==bge-m3 over tau {summary['tau_cheapest_bge_range']}")
    print(f"[{SLUG}/tau_sweep] (iii) egemma max-CLIR corner tau-invariant: {egemma_invariant} "
          f"(corner={top_clir_model})")
    print(f"[{SLUG}/tau_sweep] wrote -> {out}/tau_sweep.csv, {out}/tau_sweep_summary.json")


if __name__ == "__main__":
    main()
