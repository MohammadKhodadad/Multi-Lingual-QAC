"""
EPO BDDS streaming ingester for chemistry multilingual patents.

Downloads the next ~10 GB worth of EP full-text data (BDDS product 32) directly
from publication-bdds.apps.epo.org over HTTP Range requests, processes inner
zip/tar entries one at a time (peak working disk: ~MB, not GB), filters to
chemistry documents that exist in >=2 of {en, fr, de}, and appends rows to
data/EPO/multilingual_corpus.csv. A manifest tracks BDDS itemIds so the bulk
files can be discarded after processing without losing dedup state.

Discovery API (anonymous):
  GET https://publication-bdds.apps.epo.org/bdds/bdds-bff-service/prod/api/public/products/32

Download URL pattern (anonymous, Accept-Ranges: bytes):
  .../api/public/products/{productId}/delivery/{deliveryId}/item/{itemId}/download
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import requests
import remotezip
from tqdm import tqdm

from src.multi_lingual_qac.dataloaders.epo_xml import (
    AUXILIARY_XML_RE,
    build_row_for_language,
    language_has_substantive_text,
    parse_epo_patent_bytes,
)


API_BASE = "https://publication-bdds.apps.epo.org/bdds/bdds-bff-service/prod"
PRODUCT_ID = 32
TARGET_LANGUAGES: Tuple[str, ...] = ("en", "fr", "de")
MIN_LANG_COUNT = 2
USER_AGENT = "multi-lingual-qac/0.1 (+research)"

CSV_FIELDNAMES: Tuple[str, ...] = (
    "id",
    "language",
    "title",
    "abstract",
    "description",
    "first_claim",
    "context",
    "publication_number",
    "country_code",
    "publication_date",
    "source",
)


# --------------------------------------------------------------------------- #
# Item discovery
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ItemRef:
    item_id: int
    item_name: str
    delivery_id: int
    delivery_name: str
    file_size_str: str
    file_checksum_sha1: str
    item_published_at: str
    download_url: str

    @property
    def archive_kind(self) -> str:
        name = self.item_name.lower()
        if name.endswith(".zip"):
            return "zip"
        if name.endswith(".tar"):
            return "tar"
        if name.endswith((".tar.gz", ".tgz")):
            return "tar_gz"
        return "unknown"


def _build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json, application/octet-stream;q=0.9, */*;q=0.5",
    })
    return s


def list_items(
    product_id: int = PRODUCT_ID,
    *,
    session: Optional[requests.Session] = None,
) -> List[ItemRef]:
    """Discover every item in every delivery of a public BDDS product, newest first."""
    owns_session = session is None
    if session is None:
        session = _build_session()
    try:
        url = f"{API_BASE}/api/public/products/{product_id}"
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    finally:
        if owns_session:
            session.close()

    items: List[ItemRef] = []
    for delivery in data.get("deliveries", []):
        deliv_id = delivery["deliveryId"]
        deliv_name = delivery.get("deliveryName", "")
        for it in delivery.get("items", []):
            items.append(ItemRef(
                item_id=it["itemId"],
                item_name=it["itemName"],
                delivery_id=deliv_id,
                delivery_name=deliv_name,
                file_size_str=it.get("fileSize", ""),
                file_checksum_sha1=it.get("fileChecksum", ""),
                item_published_at=it.get("itemPublicationDatetime", ""),
                download_url=(
                    f"{API_BASE}/api/public/products/{product_id}"
                    f"/delivery/{deliv_id}/item/{it['itemId']}/download"
                ),
            ))

    items.sort(key=lambda x: x.item_published_at, reverse=True)
    return items


def select_next_items(
    items: Iterable[ItemRef],
    processed_ids: Iterable[int],
    n: int,
) -> List[ItemRef]:
    seen = set(processed_ids)
    out: List[ItemRef] = []
    for it in items:
        if it.item_id in seen:
            continue
        out.append(it)
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

@dataclass
class Manifest:
    path: Path
    schema_version: int = 1
    processed_items: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    processed_pubs: set = field(default_factory=set)
    last_ingest_at: str = ""

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        path = Path(path)
        if not path.exists():
            return cls(path=path)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            path=path,
            schema_version=int(data.get("schema_version", 1)),
            processed_items={str(k): v for k, v in data.get("processed_items", {}).items()},
            processed_pubs=set(data.get("processed_publication_numbers", [])),
            last_ingest_at=data.get("last_ingest_at", ""),
        )

    def is_item_processed(self, item_id: int) -> bool:
        return str(item_id) in self.processed_items

    def mark_item_processed(self, item: ItemRef, stats: Dict[str, Any]) -> None:
        self.processed_items[str(item.item_id)] = {
            "item_name": item.item_name,
            "delivery_id": item.delivery_id,
            "delivery_name": item.delivery_name,
            "file_size_str": item.file_size_str,
            "file_checksum_sha1": item.file_checksum_sha1,
            "item_published_at": item.item_published_at,
            "processed_at": _now_iso(),
            **stats,
        }
        self.last_ingest_at = _now_iso()

    def add_pubs(self, pubs: Iterable[str]) -> None:
        self.processed_pubs.update(pubs)

    def save_atomic(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {
            "schema_version": self.schema_version,
            "processed_items": self.processed_items,
            "processed_publication_numbers": sorted(self.processed_pubs),
            "last_ingest_at": self.last_ingest_at,
        }
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


# --------------------------------------------------------------------------- #
# In-memory accumulator: pub_num -> {lang: row_dict}
# --------------------------------------------------------------------------- #

class MultilingualAccumulator:
    """Aggregates parsed records into per-(pub, lang) rows.

    A single publication may appear in multiple XMLs within one item (different
    kind codes: A1 application, A2 search report, B1 grant). We keep the row
    whose abstract+claim text is richest for each language.
    """

    def __init__(self, *, target_languages: Iterable[str]):
        self.target_languages = tuple(target_languages)
        self._pubs: Dict[str, Dict[str, Dict[str, str]]] = {}
        self._chemistry_label: Dict[str, str] = {}
        self._record_for_filter: Dict[str, Dict[str, Any]] = {}

    def add(self, record: Dict[str, Any]) -> None:
        pub = record["publication_number"]
        if not pub:
            return

        # Track chemistry label at the publication level — strongest signal wins.
        existing_label = self._chemistry_label.get(pub, "")
        new_label = record["chemistry"]["label"]
        if _chemistry_rank(new_label) > _chemistry_rank(existing_label):
            self._chemistry_label[pub] = new_label

        # Keep a reference record so we can re-check language coverage on the
        # full multi-XML view of the publication.
        self._record_for_filter[pub] = _merge_records(
            self._record_for_filter.get(pub),
            record,
        )

        lang_rows = self._pubs.setdefault(pub, {})
        for lang in self.target_languages:
            new_row = build_row_for_language(record, lang)
            if new_row is None:
                continue
            existing = lang_rows.get(lang)
            if existing is None or _row_richness(new_row) > _row_richness(existing):
                lang_rows[lang] = new_row

    def materialize(
        self,
        *,
        min_langs: int,
        chemistry_strict: bool,
    ) -> Iterator[Dict[str, str]]:
        """Yield rows that pass chemistry and multilingual filters."""
        for pub, lang_rows in self._pubs.items():
            label = self._chemistry_label.get(pub, "not_chemistry")
            if label == "not_chemistry":
                continue
            if chemistry_strict and label != "chemistry_core":
                continue

            merged_record = self._record_for_filter[pub]
            covered = [
                lang for lang in self.target_languages
                if language_has_substantive_text(merged_record, lang)
            ]
            if len(covered) < min_langs:
                continue

            for lang in self.target_languages:
                row = lang_rows.get(lang)
                if row is None:
                    continue
                if lang not in covered:
                    continue
                yield row

    def stats(self) -> Dict[str, int]:
        return {
            "pubs_seen": len(self._pubs),
            "chemistry_pubs": sum(
                1 for label in self._chemistry_label.values()
                if label != "not_chemistry"
            ),
        }


_CHEMISTRY_RANK = {"not_chemistry": 0, "chemistry_related": 1, "chemistry_core": 2}


def _chemistry_rank(label: str) -> int:
    return _CHEMISTRY_RANK.get(label, 0)


def _row_richness(row: Dict[str, str]) -> int:
    """Higher = richer text content for ranking duplicate (pub, lang) rows."""
    return (
        len(row.get("abstract") or "")
        + len(row.get("first_claim") or "")
        + len(row.get("description") or "") // 4
    )


def _merge_records(
    existing: Optional[Dict[str, Any]],
    new: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine localized blocks from two XMLs of the same publication."""
    if existing is None:
        return dict(new)
    merged = dict(existing)
    for field_name in ("title_localized", "abstract_localized", "first_claim_localized", "description_localized"):
        merged[field_name] = _merge_localized_blocks(
            existing.get(field_name, []),
            new.get(field_name, []),
        )
    # Prefer the newer kind/date if it's "later" alphabetically (B1 > A2 > A1).
    if new.get("kind", "") > existing.get("kind", ""):
        merged["kind"] = new["kind"]
        merged["publication_date"] = new.get("publication_date") or existing.get("publication_date", "")
    # Union classification codes.
    for field_name in ("ipc_codes", "cpc_codes"):
        merged[field_name] = sorted(set(existing.get(field_name, [])) | set(new.get(field_name, [])))
    return merged


def _merge_localized_blocks(
    existing: List[Dict[str, str]],
    new: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Union of localized blocks; on language collision the longer text wins."""
    by_lang: Dict[str, str] = {}
    for block in existing + new:
        lang = (block.get("language") or "").lower()
        text = block.get("text") or ""
        if not text:
            continue
        if lang not in by_lang or len(text) > len(by_lang[lang]):
            by_lang[lang] = text
    return [{"language": lang, "text": text} for lang, text in by_lang.items()]


# --------------------------------------------------------------------------- #
# Streaming extraction
# --------------------------------------------------------------------------- #

# Safety: refuse to load any single nested archive larger than this into memory.
# Recent BDDS inner-zips run a few hundred MB at most; this guards against
# pathological files. Tunable via env var if needed.
_MAX_NESTED_ARCHIVE_BYTES = int(os.environ.get("EPO_MAX_NESTED_BYTES", str(2 * 1024 * 1024 * 1024)))


def stream_item_xmls(
    item: ItemRef,
    *,
    session: requests.Session,
    progress: Optional[Callable[[str], None]] = None,
) -> Iterator[Tuple[bytes, str]]:
    """Yield (xml_bytes, xml_name) for every patent XML inside a BDDS item.

    The outer archive is streamed via HTTP Range (zip) or sequential GET (tar)
    so the full file is never held on disk. Inner archives are loaded into
    memory, which is safe because BDDS inner zips are typically <100 MB.
    """
    kind = item.archive_kind
    if kind == "zip":
        yield from _stream_remote_zip(item.download_url, session, progress=progress)
    elif kind == "tar":
        yield from _stream_remote_tar(item.download_url, session, progress=progress)
    elif kind == "tar_gz":
        yield from _stream_remote_tar(item.download_url, session, mode="r|gz", progress=progress)
    else:
        raise ValueError(f"Unknown archive kind for item {item.item_id}: {item.item_name}")


def _stream_remote_zip(
    url: str,
    session: requests.Session,
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> Iterator[Tuple[bytes, str]]:
    with remotezip.RemoteZip(url, session=session) as rz:
        for info in rz.infolist():
            if getattr(info, "is_dir", lambda: False)() or info.filename.endswith("/"):
                continue
            if progress:
                progress(info.filename)
            data = _read_zip_member(rz, info)
            yield from _yield_xmls_recursive(data, info.filename)


def _stream_remote_tar(
    url: str,
    session: requests.Session,
    *,
    mode: str = "r|",
    progress: Optional[Callable[[str], None]] = None,
) -> Iterator[Tuple[bytes, str]]:
    with session.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        # tarfile in streaming mode needs a non-seekable file-like; resp.raw works.
        with tarfile.open(fileobj=resp.raw, mode=mode) as tf:
            for member in tf:
                if not member.isfile():
                    continue
                if progress:
                    progress(member.name)
                f = tf.extractfile(member)
                if f is None:
                    continue
                data = f.read()
                yield from _yield_xmls_recursive(data, member.name)


def _read_zip_member(rz: "remotezip.RemoteZip", info: "zipfile.ZipInfo") -> bytes:
    size = getattr(info, "file_size", 0)
    if size and size > _MAX_NESTED_ARCHIVE_BYTES:
        raise RuntimeError(
            f"BDDS entry {info.filename!r} is {size} bytes uncompressed, "
            f"exceeds the in-memory cap of {_MAX_NESTED_ARCHIVE_BYTES} bytes. "
            "Set EPO_MAX_NESTED_BYTES if you have the headroom."
        )
    return rz.read(info)


def _yield_xmls_recursive(data: bytes, name: str) -> Iterator[Tuple[bytes, str]]:
    """Recursively dispatch a bytes blob (xml / zip / tar) into XML payloads."""
    lower = name.lower()
    if lower.endswith(".xml"):
        if AUXILIARY_XML_RE.search(lower):
            return
        yield data, name
        return
    if lower.endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist():
                    if info.is_dir() or info.filename.endswith("/"):
                        continue
                    inner = zf.read(info)
                    yield from _yield_xmls_recursive(inner, f"{name}::{info.filename}")
        except zipfile.BadZipFile:
            return
        return
    if lower.endswith((".tar", ".tar.gz", ".tgz")):
        mode = "r:gz" if lower.endswith((".tar.gz", ".tgz")) else "r:"
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode=mode) as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    inner = f.read()
                    yield from _yield_xmls_recursive(inner, f"{name}::{member.name}")
        except tarfile.TarError:
            return
        return
    # Unknown extension - silently skip.


# --------------------------------------------------------------------------- #
# Single-batch orchestrator
# --------------------------------------------------------------------------- #

def _append_rows(
    corpus_path: Path,
    rows: Iterable[Dict[str, str]],
    *,
    previously_processed_pubs: set,
) -> Tuple[int, set]:
    """Append rows to the corpus CSV.

    Skips any row whose publication_number was already committed in a *previous*
    batch (cross-batch dedup). Rows for the same publication in different
    languages within the current batch are all written — multilingual coverage
    is the entire point of the pipeline.
    """
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not corpus_path.exists()
    new_pubs: set = set()
    appended = 0
    with corpus_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_FIELDNAMES), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            pub = row["publication_number"]
            if pub in previously_processed_pubs:
                continue
            writer.writerow(row)
            appended += 1
            new_pubs.add(pub)
    return appended, new_pubs


def ingest_one_batch(
    item: ItemRef,
    *,
    manifest: Manifest,
    corpus_path: Path,
    session: requests.Session,
    target_languages: Tuple[str, ...] = TARGET_LANGUAGES,
    min_langs: int = MIN_LANG_COUNT,
    chemistry_strict: bool = False,
) -> Dict[str, Any]:
    """Stream one BDDS item end-to-end: parse, filter, append, commit manifest."""
    if manifest.is_item_processed(item.item_id):
        return {"skipped": True, "reason": "already_processed"}

    print(
        f"[ingest] item_id={item.item_id} name={item.item_name} "
        f"size={item.file_size_str} published_at={item.item_published_at}"
    )

    accumulator = MultilingualAccumulator(target_languages=target_languages)
    stats = {
        "xml_files_seen": 0,
        "xml_parse_errors": 0,
        "xml_non_patent": 0,
    }

    pbar = tqdm(desc=f"  parse {item.item_name}", unit="xml")
    try:
        for xml_bytes, xml_name in stream_item_xmls(item, session=session):
            stats["xml_files_seen"] += 1
            try:
                record = parse_epo_patent_bytes(xml_bytes, xml_name=xml_name)
            except ET.ParseError:
                stats["xml_parse_errors"] += 1
                pbar.update(1)
                continue
            except ValueError:
                # Non-`<ep-patent-document>` root or missing identifiers.
                stats["xml_non_patent"] += 1
                pbar.update(1)
                continue
            accumulator.add(record)
            pbar.update(1)
    finally:
        pbar.close()

    rows_to_write = list(accumulator.materialize(
        min_langs=min_langs,
        chemistry_strict=chemistry_strict,
    ))

    # Snapshot the previously-processed pubs BEFORE writing — this batch's own
    # rows must not dedup against each other (we want all language rows).
    previously_processed = set(manifest.processed_pubs)
    appended, new_pubs = _append_rows(
        corpus_path,
        rows_to_write,
        previously_processed_pubs=previously_processed,
    )

    acc_stats = accumulator.stats()
    batch_stats = {
        **stats,
        "pubs_seen": acc_stats["pubs_seen"],
        "chemistry_pubs_kept": acc_stats["chemistry_pubs"],
        "multilingual_pubs_kept": len(new_pubs),
        "rows_appended": appended,
    }

    manifest.mark_item_processed(item, batch_stats)
    manifest.add_pubs(new_pubs)
    manifest.save_atomic()

    print(
        f"  -> parsed {stats['xml_files_seen']} XMLs, kept {acc_stats['chemistry_pubs']} chemistry pubs, "
        f"appended {appended} rows ({len(new_pubs)} new pubs) to {corpus_path}"
    )
    return batch_stats


# --------------------------------------------------------------------------- #
# N-batch orchestrator
# --------------------------------------------------------------------------- #

def ingest_n_batches(
    n: int,
    *,
    manifest_path: Path,
    corpus_path: Path,
    target_languages: Tuple[str, ...] = TARGET_LANGUAGES,
    min_langs: int = MIN_LANG_COUNT,
    chemistry_strict: bool = False,
    product_id: int = PRODUCT_ID,
) -> List[Dict[str, Any]]:
    """Process the N newest unprocessed BDDS items in sequence."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    manifest = Manifest.load(manifest_path)

    session = _build_session()
    try:
        all_items = list_items(product_id, session=session)
        next_items = select_next_items(
            all_items,
            processed_ids={int(k) for k in manifest.processed_items},
            n=n,
        )

        if not next_items:
            print(f"[epo_bdds] No new items to process. {len(manifest.processed_items)} already in manifest.")
            return []

        print(f"[epo_bdds] {len(next_items)} item(s) queued:")
        for it in next_items:
            print(f"  - itemId={it.item_id:>5} {it.item_name:<40} {it.file_size_str:>10}  ({it.item_published_at})")

        results: List[Dict[str, Any]] = []
        for i, item in enumerate(next_items, 1):
            print(f"\n[epo_bdds] Batch {i}/{len(next_items)}")
            try:
                result = ingest_one_batch(
                    item,
                    manifest=manifest,
                    corpus_path=corpus_path,
                    session=session,
                    target_languages=target_languages,
                    min_langs=min_langs,
                    chemistry_strict=chemistry_strict,
                )
                results.append(result)
            except Exception as exc:
                # Don't poison the manifest with a half-processed item: bail.
                print(f"[epo_bdds] ERROR processing item {item.item_id}: {exc}", file=sys.stderr)
                raise
        return results
    finally:
        session.close()
