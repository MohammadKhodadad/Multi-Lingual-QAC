"""
Populate ``wiki_names_cache.json`` with Wikipedia names for *all* ChEBI terms.

The cache is normally filled lazily, only for concepts that turn up in the
corpus (~3k of ~205k ChEBI terms). This driver fetches the Wikipedia titles for
every node in the ChEBI graph so downstream name matching can draw on the full
ontology. It is a thin wrapper over the existing incremental fetcher: the heavy
lifting (batching, retry/backoff, atomic incremental writes, ``{}`` for
no-article hits) lives in :mod:`src.alias_graph.wikidata_names`.

Resumable: only ids absent from the cache are queried, so re-running the same
command continues from wherever a previous run stopped.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Sequence

from src.alias_graph.chebi import load_chebi_graph
from src.alias_graph.wikidata_names import (
    _BATCH_SIZE,
    _SLEEP_BETWEEN_BATCHES,
    DEFAULT_LANGS,
    _load_cache,
    fetch_wikipedia_names,
)


def cache_all_chebi_wiki(
    chebi_cache_dir: Path | str,
    *,
    variant: str = "full",
    langs: Sequence[str] = DEFAULT_LANGS,
) -> None:
    """Fetch Wikipedia names for every ChEBI term into the shared name cache."""
    chebi_cache_dir = Path(chebi_cache_dir)
    cache_path = chebi_cache_dir / "wiki_names_cache.json"

    graph = load_chebi_graph(chebi_cache_dir, variant)
    all_ids = list(graph.nodes())

    cache = _load_cache(cache_path)
    missing = [cid for cid in all_ids if cid not in cache]
    n_batches = ceil(len(missing) / _BATCH_SIZE)
    # ~1s sleep + ~1-2.5s per query round-trip => budget a 2-3.5s/batch range.
    eta_lo = n_batches * (_SLEEP_BETWEEN_BATCHES + 1.0) / 3600
    eta_hi = n_batches * (_SLEEP_BETWEEN_BATCHES + 2.5) / 3600

    print(
        f"ChEBI {variant} graph: {len(all_ids)} terms; cache has {len(cache)} "
        f"({len(missing)} missing).\n"
        f"Fetching {len(missing)} ids in {n_batches} batches of {_BATCH_SIZE} "
        f"for langs {tuple(langs)}; est. {eta_lo:.1f}-{eta_hi:.1f} h.\n"
        f"Resumable -- safe to interrupt and re-run."
    )
    if not missing:
        print("Nothing to do: every ChEBI term is already in the cache.")
        return

    fetch_wikipedia_names(all_ids, langs=langs, cache_path=cache_path)

    final = _load_cache(cache_path)
    with_any = sum(1 for v in final.values() if v)
    with_zh = sum(1 for v in final.values() if isinstance(v, dict) and v.get("zh"))
    print(
        f"Done. Cache now holds {len(final)} entries: "
        f"{with_any} with >=1 Wikipedia name, {with_zh} with a Chinese name."
    )
