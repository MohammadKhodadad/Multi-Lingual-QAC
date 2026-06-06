"""
Alias-Graph Retrieval benchmark builder.

For each ChEBI concept that genuinely appears in the corpus (gold documents), we
surround it with hard negatives -- documents that mention a *taxonomic neighbor*
of the concept (chemically similar, but a different concept) and do not mention
the concept itself. A concept is kept only if it has at least ``min_gold`` gold
documents and at least ``min_neg`` hard-negative documents. Each kept concept is
written to its own CSV (gold + hard-negative rows, role-labeled), and a manifest
records the concept's multilingual name set (the retrieval query) plus counts.

Pipeline: read corpus -> load ChEBI graph -> (KG-only scan to find concepts that
appear) -> fetch Wikipedia names for those concepts -> rebuild index + rescan ->
assemble gold/hard-negatives -> write per-concept CSVs + manifest.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx

from src.alias_graph.chebi import load_chebi_graph, taxonomic_neighbors
from src.alias_graph.wikidata_names import (
    DEFAULT_LANGS,
    fetch_wikipedia_names,
)
from src.alias_graph.matching import (
    build_name_index,
    prune_names,
    scan_corpus,
)

# Root of the ChEBI structural (actual-molecule) subtree. Restricting main
# concepts to its descendants keeps real chemical entities and drops role /
# group / atom / application classes whose names are ordinary words.
MOLECULAR_ENTITY = "CHEBI:23367"

# Patent description fields can exceed the default csv field-size limit.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

CORPUS_FIELDS: Tuple[str, ...] = (
    "id", "language", "title", "abstract", "description", "first_claim",
    "context", "publication_number", "country_code", "publication_date",
    "source", "ipc_codes",
)
EXTRA_FIELDS: Tuple[str, ...] = (
    "role", "concept_chebi_id", "concept_name",
    "matched_chebi_id", "matched_name", "matched_lang", "relation",
)
OUTPUT_FIELDS: Tuple[str, ...] = CORPUS_FIELDS + EXTRA_FIELDS


def _slug(name: str, maxlen: int = 60) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_name).strip("-").lower()
    return s[:maxlen] or "concept"


def _read_corpus(path: Path) -> List[dict]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _concept_name_set(
    graph: nx.DiGraph,
    cid: str,
    wiki_names: Dict[str, Dict[str, str]],
) -> Dict[str, List[str]]:
    """Multilingual name set (the query) for a concept, grouped by source."""
    data = graph.nodes[cid]
    name_set: Dict[str, List[str]] = {}
    chebi_names = []
    if data.get("name"):
        chebi_names.append(data["name"])
    chebi_names.extend(data.get("synonyms", ()))
    if chebi_names:
        name_set["chebi"] = chebi_names
    for lang, title in wiki_names.get(cid, {}).items():
        name_set.setdefault(lang, []).append(title)
    return name_set


def _neighbor_relations(graph: nx.DiGraph, cid: str) -> Dict[str, str]:
    """Map each taxonomic neighbor id -> relation, parent/child before sibling."""
    nb = taxonomic_neighbors(graph, cid)
    rel: Dict[str, str] = {}
    for nid in nb["sibling"]:
        rel[nid] = "sibling"
    for nid in nb["child"]:
        rel[nid] = "child"
    for nid in nb["parent"]:
        rel[nid] = "parent"
    return rel


def build_alias_graph(
    corpus_csv: Path,
    output_dir: Path,
    chebi_cache_dir: Path,
    *,
    variant: str = "full",
    langs: Sequence[str] = DEFAULT_LANGS,
    use_wikipedia: bool = True,
    wiki_cache_path: Optional[Path] = None,
    min_gold: int = 2,
    min_neg: int = 3,
    max_concepts: Optional[int] = None,
    match_field: str = "context",
    min_name_len: int = 4,
    max_concepts_per_name: int = 3,
    max_df_ratio: float = 0.02,
    molecular_only: bool = True,
    leaf_only: bool = True,
) -> dict:
    """Build the benchmark; returns a summary dict."""
    corpus_csv = Path(corpus_csv)
    output_dir = Path(output_dir)
    chebi_cache_dir = Path(chebi_cache_dir)
    wiki_cache_path = Path(wiki_cache_path) if wiki_cache_path else chebi_cache_dir / "wiki_names_cache.json"

    print(f"Reading corpus: {corpus_csv}")
    rows = _read_corpus(corpus_csv)
    doc_by_id = {r["id"]: r for r in rows}
    print(f"  {len(rows)} documents")

    graph = load_chebi_graph(chebi_cache_dir, variant)

    mol_entity_set = None
    if molecular_only:
        if MOLECULAR_ENTITY in graph:
            mol_entity_set = nx.ancestors(graph, MOLECULAR_ENTITY) | {MOLECULAR_ENTITY}
            print(f"  restricting to {len(mol_entity_set)} molecular-entity concepts")
        else:
            print(f"  warning: {MOLECULAR_ENTITY} absent from {variant} graph; no molecular filter")

    # Pass 1: KG-only scan to discover which concepts occur (so we only ask
    # Wikidata about those) and to measure per-name document frequency.
    print("Scanning corpus for ChEBI names (KG only) ...")
    kg_index = build_name_index(
        graph, {}, min_len=min_name_len, max_concepts_per_name=max_concepts_per_name
    )
    print(f"  name index: {kg_index.n_names()} names")
    concept_to_docs, _, name_doc_freq = scan_corpus(rows, kg_index, field=match_field)
    print(f"  concepts found in corpus: {len(concept_to_docs)}")

    # Names that behave like corpus stopwords (common words masquerading as
    # aliases, e.g. "para", "groupe") are pruned so only specific names match.
    df_ceiling = max_df_ratio * len(rows)
    stop_grams = {g for g, n in name_doc_freq.items() if n > df_ceiling}
    if stop_grams:
        examples = sorted(stop_grams, key=lambda g: -name_doc_freq[g])[:8]
        print(f"  pruning {len(stop_grams)} stopword names (df > {df_ceiling:.0f}); e.g. {examples}")

    wiki_names: Dict[str, Dict[str, str]] = {}
    if use_wikipedia and concept_to_docs:
        wiki_names = fetch_wikipedia_names(
            list(concept_to_docs.keys()), langs=langs, cache_path=wiki_cache_path
        )

    # Pass 2: final index (with Wikipedia names if enabled), stopwords removed.
    index = (
        build_name_index(
            graph, wiki_names, min_len=min_name_len, max_concepts_per_name=max_concepts_per_name
        )
        if wiki_names
        else kg_index
    )
    prune_names(index, stop_grams)
    print("Re-scanning corpus (Wikipedia names folded in, stopwords pruned) ...")
    concept_to_docs, match_info, _ = scan_corpus(rows, index, field=match_field)
    print(f"  concepts found in corpus: {len(concept_to_docs)}")

    # Candidate main concepts: specific molecular entities (leaves of the is_a
    # graph -- not broad classes) with enough gold docs, most-attested first.
    def _is_candidate(cid: str) -> bool:
        if len(concept_to_docs[cid]) < min_gold:
            return False
        if mol_entity_set is not None and cid not in mol_entity_set:
            return False
        if leaf_only and graph.in_degree(cid) > 0:
            return False
        return True

    candidates = sorted(
        (cid for cid in concept_to_docs if _is_candidate(cid)),
        key=lambda c: len(concept_to_docs[c]),
        reverse=True,
    )
    kind = "leaf molecular" if leaf_only else "molecular"
    print(f"Candidate concepts ({kind}, >= {min_gold} gold docs): {len(candidates)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[dict] = []

    for cid in candidates:
        if max_concepts is not None and len(manifest) >= max_concepts:
            break
        gold_docs = concept_to_docs[cid]
        relations = _neighbor_relations(graph, cid)

        # Hard negatives: docs mentioning a neighbor but NOT the main concept.
        hard_neg: Dict[str, Tuple[str, str]] = {}  # doc_id -> (neighbor_id, relation)
        neighbors_in_corpus: Set[str] = set()
        for nid, rel in relations.items():
            nid_docs = concept_to_docs.get(nid)
            if not nid_docs:
                continue
            neighbors_in_corpus.add(nid)
            for doc_id in nid_docs - gold_docs:
                hard_neg.setdefault(doc_id, (nid, rel))

        if len(hard_neg) < min_neg:
            continue

        concept_name = graph.nodes[cid].get("name", cid)
        name_set = _concept_name_set(graph, cid, wiki_names)

        # Write the per-concept CSV (gold first, then hard negatives).
        out_path = output_dir / f"{cid.replace(':', '_')}__{_slug(concept_name)}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for doc_id in sorted(gold_docs):
                row = doc_by_id[doc_id]
                surf, mlang = match_info[doc_id][cid]
                writer.writerow({
                    **{k: row.get(k, "") for k in CORPUS_FIELDS},
                    "role": "gold", "concept_chebi_id": cid, "concept_name": concept_name,
                    "matched_chebi_id": cid, "matched_name": surf,
                    "matched_lang": mlang, "relation": "self",
                })
            for doc_id in sorted(hard_neg):
                row = doc_by_id[doc_id]
                nid, rel = hard_neg[doc_id]
                surf, mlang = match_info[doc_id][nid]
                writer.writerow({
                    **{k: row.get(k, "") for k in CORPUS_FIELDS},
                    "role": "hard_negative", "concept_chebi_id": cid, "concept_name": concept_name,
                    "matched_chebi_id": nid, "matched_name": surf,
                    "matched_lang": mlang, "relation": rel,
                })

        gold_pubs = {doc_by_id[d]["publication_number"] for d in gold_docs}
        gold_langs = sorted({doc_by_id[d]["language"] for d in gold_docs})
        manifest.append({
            "chebi_id": cid,
            "name": concept_name,
            "name_set": name_set,
            "query_names": sorted({n for names in name_set.values() for n in names}),
            "n_gold_docs": len(gold_docs),
            "n_gold_pubs": len(gold_pubs),
            "gold_langs": gold_langs,
            "n_hard_neg_docs": len(hard_neg),
            "n_neighbors_total": len(relations),
            "n_neighbors_in_corpus": len(neighbors_in_corpus),
            "csv_path": str(out_path.relative_to(output_dir)),
        })

    _write_manifest(output_dir, manifest)

    summary = {
        "corpus": str(corpus_csv),
        "documents": len(rows),
        "variant": variant,
        "use_wikipedia": use_wikipedia,
        "concepts_in_corpus": len(concept_to_docs),
        "candidates": len(candidates),
        "concepts_written": len(manifest),
        "output_dir": str(output_dir),
    }
    print(
        f"Wrote {len(manifest)} concept files -> {output_dir}\n"
        f"  manifest: {output_dir / 'manifest.csv'}"
    )
    return summary


def _write_manifest(output_dir: Path, manifest: List[dict]) -> None:
    json_path = output_dir / "manifest.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    csv_path = output_dir / "manifest.csv"
    cols = [
        "chebi_id", "name", "n_gold_docs", "n_gold_pubs", "gold_langs",
        "n_hard_neg_docs", "n_neighbors_total", "n_neighbors_in_corpus", "csv_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for entry in manifest:
            row = dict(entry)
            row["gold_langs"] = "|".join(entry["gold_langs"])
            writer.writerow(row)
