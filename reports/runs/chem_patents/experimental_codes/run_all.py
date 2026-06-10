"""
Run the full 10-round chem-patents CLIR analysis end to end, then build the curated key_findings/.

Rounds have cross-dependencies (Round 10 reads Rounds 1/3/5/8), so they run in order. Each runs in its
own subprocess for isolation.

    python reports/runs/chem_patents/experimental_codes/run_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "round01_clir_leaderboard.py",
    "round02_directional_clir.py",
    "round03_mt_penalty.py",
    "round04_mate_retrieval.py",
    "round05_consistency.py",
    "round06_language_collapse.py",
    "round07_distractor_dominance.py",
    "round08_separability.py",
    "round09_ensemble.py",
    "round10_robustness_synthesis.py",
    "build_key_findings.py",
]


def main() -> None:
    for name in SCRIPTS:
        print(f"\n===== {name} =====")
        r = subprocess.run([sys.executable, str(HERE / name)])
        if r.returncode != 0:
            print(f"!! {name} failed (exit {r.returncode})")
            sys.exit(r.returncode)
    print("\nAll rounds complete. See experimental_plots/FINDINGS.md and key_findings/EXECUTIVE_SUMMARY.md")


if __name__ == "__main__":
    main()
