"""One-off pilot: stream BDDS item 1573 (1978, 96.5 MB) end-to-end.

Bypasses `ingest_n_batches` (which always picks the newest unprocessed item)
so we can validate the streaming + parsing + filtering pipeline against a
small known-good archive before touching the real 16 GB weekly fronts.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.multi_lingual_qac.config import PipelinePaths
from src.multi_lingual_qac.dataloaders.epo_bdds import (
    Manifest,
    _build_session,
    ingest_one_batch,
    list_items,
)


def main() -> None:
    paths = PipelinePaths.from_project_root(PROJECT_ROOT)

    pilot_dir = paths.epo_data_dir / "pilot_smallest"
    if pilot_dir.exists():
        shutil.rmtree(pilot_dir)
    pilot_dir.mkdir(parents=True)
    manifest_path = pilot_dir / "manifest.json"
    corpus_path = pilot_dir / "multilingual_corpus.csv"

    session = _build_session()
    try:
        items = list_items(session=session)
        target_item_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1573
        item = next((it for it in items if it.item_id == target_item_id), None)
        if item is None:
            sys.exit(f"Could not find target item_id={target_item_id} in BDDS product 32 listing")

        print(f"Pilot item: {item}")
        manifest = Manifest.load(manifest_path)
        stats = ingest_one_batch(
            item,
            manifest=manifest,
            corpus_path=corpus_path,
            session=session,
        )
    finally:
        session.close()

    print()
    print("=== PILOT RESULT ===")
    print(f"  corpus: {corpus_path}")
    if corpus_path.exists():
        size = corpus_path.stat().st_size
        with corpus_path.open() as f:
            lines = sum(1 for _ in f)
        print(f"    {size} bytes, {lines} lines (header + rows)")
    print(f"  manifest: {manifest_path}")
    print(f"  stats: {stats}")


if __name__ == "__main__":
    main()
