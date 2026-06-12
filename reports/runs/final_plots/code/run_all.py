"""
Regenerate the entire final paper-figure suite from local artifacts only.

    cd reports/runs/final_plots/code
    ../../../../.venv/bin/python run_all.py

No embedding model is loaded and no network is touched. Every figure renders to candidates/
(PNG 300 dpi + PDF) and its plotted numbers to data/. The curation step (copying winners into
main/ and appendix/) is done by curate.py.
"""
from __future__ import annotations

import fp_common as fp
import claimA_tech_semantic as A
import claimB_source_transfer as B
import claimC_per_language as C
import claimD_alias_graph as D
import claimE_metrics
import claimE_xrc_rrc_ari as E


def main():
    fp.set_style()
    import numpy as np
    np.random.seed(0)
    claimE_metrics.compute()  # recompute XRC/RRC/ARI on the 524-query gold before claim E plots
    for mod in (A, B, C, D, E):
        mod.main()
    print("\nAll candidates regenerated in", fp.CAND)


if __name__ == "__main__":
    main()
