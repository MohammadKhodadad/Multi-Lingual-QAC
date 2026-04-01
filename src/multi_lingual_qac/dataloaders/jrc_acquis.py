from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import re
import shutil
import sys
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional
from urllib.parse import urljoin
from urllib.request import urlopen

from tqdm import tqdm

JRC_ACQUIS_LANGS = (
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "hu",
    "it",
    "lt",
    "lv",
    "mt",
    "nl",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "sv",
)

JRC_ACQUIS_CORPUS_INDEX_URL = "https://wt-public.emm4u.eu/Acquis/JRC-Acquis.3.0/corpus/"
RE_MULTISPACE = re.compile(r"\s+")
RE_CELEX = re.compile(r"\b[0-9A-Z][0-9A-Z()_-]{5,}\b")
RE_ARCHIVE_NAME = re.compile(r"\bjrc-([a-z]{2})\.tgz\b", re.IGNORECASE)
GENERIC_JRC_TITLE_RE = re.compile(r"^JRC-ACQUIS\s+[0-9A-Z()_-]+\s+\w+", re.IGNORECASE)
RE_JRC_PLUS_PREFIX = re.compile(r"^(?:\+\+\+\+\s*)+")
RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
RE_SPACE_AFTER_OPEN = re.compile(r"([(\[])\s+")
RE_SPACE_BEFORE_CLOSE = re.compile(r"\s+([)\]])")
RE_JRC_SECTION_HEADING = re.compile(r"^[IVXLC]+\s*\([^)]+\)$", re.IGNORECASE)
RE_JRC_TITLE_WRAPPER = re.compile(r"^[IVXLC]+\s*\([^)]+\)\s+[A-ZÀ-ÖØ-Þ]+", re.IGNORECASE)
RE_JRC_ARTICLE_HEADING = re.compile(
    r"^(?:"
    r"(?:art(?:icle|ikel|ikla|ikkel|igo|icolo|icolul|ikulu|ículo|iculo|ykuł|ykul)"
    r"|čl(?:ánek|anok)?|člen|член|άρθρο|αρθρο|straipsnis|pants)"
    r"\s+(?:\d+(?:[./-][\w]+)?|[ivxlcdm]+|one|first|premier|primo|primero|pierwszy|prv[yý]|els[őo]|pirmas|pirmais)"
    r"|(?:\d+(?:[./-][\w]+)?\s+(?:cikk|artikla|straipsnis|pants))"
    r")\b",
    re.IGNORECASE,
)
RE_JRC_ARTICLE_TOKEN = re.compile(
    r"^(?:art(?:icle|ikel|ikla|ikkel|igo|icolo|icolul|ikulu|ículo|iculo|ykuł|ykul)"
    r"|čl(?:ánek|anok)?|člen|член|άρθρο|αρθρο|cikk|straipsnis|pants)\b",
    re.IGNORECASE,
)
RE_JRC_ARTIFACT_LINE = re.compile(r"^(?:\*{5,}|\[pic\])", re.IGNORECASE)
RE_JRC_REFERENCE_LINE = re.compile(r"^\(\d+\)\s+[A-ZÀ-ÖØ-Þ]{1,6}\b")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _clean_text(text: str) -> str:
    return RE_MULTISPACE.sub(" ", (text or "")).strip()


def _normalize_jrc_text(text: str) -> str:
    text = (text or "").replace("\ufeff", "").strip()
    text = RE_JRC_PLUS_PREFIX.sub("", text).strip()
    text = _clean_text(text)
    text = RE_SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = RE_SPACE_AFTER_OPEN.sub(r"\1", text)
    text = RE_SPACE_BEFORE_CLOSE.sub(r"\1", text)
    return text.strip()


def _looks_like_article_heading(text: str, lang: str = "") -> bool:
    candidate = _normalize_jrc_text(text)
    if not candidate:
        return False
    match = RE_JRC_ARTICLE_HEADING.match(candidate)
    if not match:
        return False
    suffix = candidate[match.end():].strip(" .:-")
    if not suffix:
        return True
    return len(suffix) <= 40 and len(suffix.split()) <= 6


def _looks_like_institution_heading(text: str) -> bool:
    candidate = _normalize_jrc_text(text)
    if not candidate:
        return False
    letters = [ch for ch in candidate if ch.isalpha()]
    if len(letters) < 10:
        return False
    upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
    return upper_ratio >= 0.8 and len(candidate.split()) >= 2


def _looks_like_adoption_formula(text: str) -> bool:
    candidate = _normalize_jrc_text(text)
    if not candidate:
        return False
    letters = [ch for ch in candidate if ch.isalpha()]
    if len(letters) < 8:
        return False
    upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
    return upper_ratio >= 0.7 and candidate[-1:] in {":", ".", "-"}


def _looks_like_reference_line(text: str) -> bool:
    candidate = _normalize_jrc_text(text)
    return bool(RE_JRC_REFERENCE_LINE.match(candidate))


def _clean_jrc_paragraphs(paragraphs: list[str], lang: str) -> tuple[list[str], bool]:
    cleaned: list[str] = []
    changed = False
    for paragraph in paragraphs:
        normalized = _normalize_jrc_text(paragraph)
        if not normalized:
            changed = True
            continue
        if normalized != paragraph.strip():
            changed = True
        if RE_JRC_ARTIFACT_LINE.match(normalized):
            changed = True
            continue
        if _looks_like_reference_line(normalized):
            changed = True
            continue
        if RE_JRC_SECTION_HEADING.match(normalized):
            changed = True
            continue
        cleaned.append(normalized)
    return cleaned, changed


def _trim_jrc_to_operative_body(paragraphs: list[str], lang: str) -> tuple[list[str], bool]:
    for idx, paragraph in enumerate(paragraphs):
        if not _looks_like_article_heading(paragraph, lang):
            continue
        remaining = paragraphs[idx:]
        if idx >= 1 and len(remaining) >= 2:
            remaining_chars = sum(len(part) for part in remaining)
            if remaining_chars >= 80:
                return remaining, True
    return paragraphs, False


def _is_aggregate_corpus_xml(source_name: str, archive_lang: str = "") -> bool:
    """
    Skip top-level TEI corpus wrappers like `bg/jrc-bg.xml` that reference
    document entities (`&teiHeader;`, `&jrc...;`) instead of containing one
    concrete document body.
    """
    name = Path(source_name).name.lower()
    if name in {"jrcheader.xml", "jrcheader.xml"}:
        return True
    if archive_lang and name == f"jrc-{archive_lang}.xml":
        return True
    return bool(re.fullmatch(r"jrc-[a-z]{2}\.xml", name))


def _requested_jrc_languages(languages: Optional[Iterable[str]]) -> tuple[str, ...]:
    if languages is None:
        return JRC_ACQUIS_LANGS
    requested = tuple(lang.lower() for lang in languages)
    valid = tuple(lang for lang in requested if lang in JRC_ACQUIS_LANGS)
    if not valid:
        raise ValueError(
            "No valid JRC-Acquis languages requested. "
            f"Supported: {', '.join(JRC_ACQUIS_LANGS)}"
        )
    return valid


def count_jrc_acquis_input_files(input_dir: Path) -> int:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        return 0
    archives = list(input_dir.glob("jrc-*.tgz")) + list(input_dir.glob("jrc-*.tar.gz"))
    xmls = list(input_dir.rglob("*.xml"))
    return len(archives) + len(xmls)


def _fetch_jrc_archive_urls(index_url: str = JRC_ACQUIS_CORPUS_INDEX_URL) -> dict[str, str]:
    with urlopen(index_url) as response:
        html = response.read().decode("utf-8", errors="replace")

    urls: dict[str, str] = {}
    for match in RE_ARCHIVE_NAME.finditer(html):
        lang = match.group(1).lower()
        if lang in JRC_ACQUIS_LANGS:
            filename = f"jrc-{lang}.tgz"
            urls[lang] = urljoin(index_url, filename)
    return urls


def download_jrc_acquis_archives(
    input_dir: Path,
    *,
    languages: Optional[Iterable[str]] = None,
    overwrite: bool = False,
    index_url: str = JRC_ACQUIS_CORPUS_INDEX_URL,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    requested = _requested_jrc_languages(languages)
    archive_urls = _fetch_jrc_archive_urls(index_url=index_url)
    missing = [lang for lang in requested if lang not in archive_urls]
    if missing:
        raise ValueError(
            f"Missing download URLs for JRC-Acquis languages: {missing} from {index_url}"
        )

    stats = {
        "downloaded": 0,
        "skipped_existing": 0,
        "archives": [],
    }
    for lang in requested:
        filename = f"jrc-{lang}.tgz"
        dest = input_dir / filename
        if dest.exists() and not overwrite:
            stats["skipped_existing"] += 1
            stats["archives"].append(filename)
            continue

        tmp = dest.with_suffix(dest.suffix + ".part")
        with urlopen(archive_urls[lang]) as response, tmp.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        tmp.replace(dest)
        stats["downloaded"] += 1
        stats["archives"].append(filename)

    return stats


def _iter_elements(root: ET.Element, name: str) -> Iterator[ET.Element]:
    target = name.lower()
    for elem in root.iter():
        if _local_name(elem.tag).lower() == target:
            yield elem


def _first_attr(elem: ET.Element, *keys: str) -> str:
    for key in keys:
        if key in elem.attrib and elem.attrib[key]:
            return str(elem.attrib[key]).strip()
        xml_key = f"{{http://www.w3.org/XML/1998/namespace}}{key}"
        if xml_key in elem.attrib and elem.attrib[xml_key]:
            return str(elem.attrib[xml_key]).strip()
    return ""


def _guess_celex(root: ET.Element, source_name: str) -> str:
    for elem in root.iter():
        for candidate in (
            _first_attr(elem, "celex"),
            _first_attr(elem, "id"),
            _first_attr(elem, "n"),
        ):
            if candidate and RE_CELEX.fullmatch(candidate):
                return candidate

    for text in (_clean_text("".join(elem.itertext())) for elem in _iter_elements(root, "idno")):
        match = RE_CELEX.search(text)
        if match:
            return match.group(0)

    match = RE_CELEX.search(source_name)
    return match.group(0) if match else Path(source_name).stem


def _guess_language(root: ET.Element, archive_lang: str) -> str:
    candidates = [
        _first_attr(root, "lang"),
        _first_attr(root, "language"),
        _first_attr(root, "id"),
    ]
    for elem in root.iter():
        candidates.extend(
            [
                _first_attr(elem, "lang"),
                _first_attr(elem, "language"),
            ]
        )
        if len(candidates) > 50:
            break

    for value in candidates:
        code = value.lower().strip()
        if code in JRC_ACQUIS_LANGS:
            return code
    return archive_lang


def _extract_title(root: ET.Element) -> str:
    for tag_name in ("title", "head"):
        for elem in _iter_elements(root, tag_name):
            text = _clean_text("".join(elem.itertext()))
            if text:
                return text
    return ""


def _extract_header_notes(root: ET.Element) -> list[str]:
    notes: list[str] = []
    seen: set[str] = set()
    for elem in _iter_elements(root, "note"):
        text = _normalize_jrc_text("".join(elem.itertext()))
        if not text:
            continue
        lowered = text.lower()
        if "http://" in lowered or "https://" in lowered:
            continue
        if text in seen:
            continue
        seen.add(text)
        notes.append(text)
    return notes


def _is_header_helper_record(*, celex: str, source_name: str, title: str) -> bool:
    celex_norm = celex.strip().lower()
    source_norm = source_name.replace("\\", "/").strip().lower()
    title_norm = _normalize_jrc_text(title).lower()
    if celex_norm.startswith("jrcheader"):
        return True
    if source_norm.endswith("jrcheader.xml") or "/jrcheader" in source_norm:
        return True
    if "multilingual parallel corpus" in title_norm and "jrc-acquis" in title_norm:
        return True
    return False


def _derive_document_title(
    raw_title: str,
    body_paragraphs: list[str],
    celex: str,
    *,
    lang: str,
) -> str:
    raw_title = _normalize_jrc_text(raw_title)

    title_parts: list[str] = []
    for para in body_paragraphs[:8]:
        candidate = _normalize_jrc_text(para)
        if not candidate:
            continue
        if _looks_like_reference_line(candidate):
            continue
        if title_parts and (
            _looks_like_article_heading(candidate, lang)
            or _looks_like_institution_heading(candidate)
            or _looks_like_adoption_formula(candidate)
        ):
            break
        if title_parts and len(candidate) > 220:
            break
        title_parts.append(candidate)
        if candidate.startswith("(") and ")" in candidate:
            break
        if sum(len(part) for part in title_parts) >= 320:
            break

    derived_title = _clean_text(" ".join(title_parts))
    raw_title_is_wrapper = bool(RE_JRC_TITLE_WRAPPER.match(raw_title))
    if derived_title and (not raw_title or GENERIC_JRC_TITLE_RE.match(raw_title) or raw_title_is_wrapper):
        return derived_title[:500]
    if raw_title and not raw_title_is_wrapper:
        return raw_title[:500]
    if derived_title:
        return derived_title[:500]
    return raw_title or celex


def _extract_eurovoc(root: ET.Element) -> list[str]:
    values: list[str] = []
    for elem in _iter_elements(root, "classCode"):
        text = _clean_text("".join(elem.itertext()))
        scheme = _first_attr(elem, "scheme")
        if text and ("eurovoc" in scheme.lower() or not scheme):
            values.append(text)
    return sorted(set(values))


def _section_from_ancestors(
    elem: ET.Element,
    parent_map: Dict[ET.Element, ET.Element],
) -> str:
    cursor: Optional[ET.Element] = elem
    while cursor is not None:
        local = _local_name(cursor.tag).lower()
        type_attr = _first_attr(cursor, "type").lower()
        id_attr = _first_attr(cursor, "id").lower()
        joined = f"{local} {type_attr} {id_attr}"
        if "annex" in joined:
            return "annex"
        if "sign" in joined:
            return "signature"
        if "title" in joined or "head" in joined:
            return "title"
        cursor = parent_map.get(cursor)
    return "body"


def _extract_paragraphs(root: ET.Element) -> list[dict[str, str]]:
    parent_map = {child: parent for parent in root.iter() for child in parent}
    records: list[dict[str, str]] = []

    paragraph_elems = list(_iter_elements(root, "p"))
    if not paragraph_elems:
        paragraph_elems = list(_iter_elements(root, "seg"))

    for elem in paragraph_elems:
        text = _clean_text("".join(elem.itertext()))
        if not text:
            continue
        records.append(
            {
                "n": _first_attr(elem, "n"),
                "section": _section_from_ancestors(elem, parent_map),
                "text": text,
            }
        )
    return records


def _parse_xml_record(
    xml_bytes: bytes,
    *,
    archive_lang: str,
    source_name: str,
    source_archive: str,
) -> dict[str, Any]:
    root = ET.fromstring(xml_bytes)
    paragraphs = _extract_paragraphs(root)
    section_counts = Counter(p["section"] for p in paragraphs)
    return {
        "celex": _guess_celex(root, source_name=source_name),
        "language": _guess_language(root, archive_lang=archive_lang),
        "title": _extract_title(root),
        "header_notes": _extract_header_notes(root),
        "source_archive": source_archive,
        "source_name": source_name,
        "paragraph_count": len(paragraphs),
        "section_counts": dict(sorted(section_counts.items())),
        "eurovoc_ids": _extract_eurovoc(root),
        "paragraphs": paragraphs,
    }


def _safe_parse_xml_record(
    xml_bytes: bytes,
    *,
    archive_lang: str,
    source_name: str,
    source_archive: str,
) -> Optional[dict[str, Any]]:
    if not xml_bytes or not xml_bytes.strip():
        return None
    try:
        return _parse_xml_record(
            xml_bytes,
            archive_lang=archive_lang,
            source_name=source_name,
            source_archive=source_archive,
        )
    except ET.ParseError:
        return None


def _iter_archive_records(
    archive_path: Path,
    *,
    archive_lang: str,
) -> Iterator[dict[str, Any]]:
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith(".xml"):
                continue
            if _is_aggregate_corpus_xml(member.name, archive_lang=archive_lang):
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            record = _safe_parse_xml_record(
                extracted.read(),
                archive_lang=archive_lang,
                source_name=member.name,
                source_archive=str(archive_path),
            )
            if record is not None:
                yield record


def _process_archive_to_temp(
    archive_path_str: str,
    archive_lang: str,
    temp_dir_str: str,
) -> dict[str, Any]:
    archive_path = Path(archive_path_str)
    temp_dir = Path(temp_dir_str)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_jsonl = temp_dir / f"{archive_path.stem}.jsonl"

    count = 0
    language_counts: Counter[str] = Counter()
    paragraph_counts: Counter[str] = Counter()

    with temp_jsonl.open("w", encoding="utf-8") as out:
        for record in _iter_archive_records(archive_path, archive_lang=archive_lang):
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            language_counts[record["language"]] += 1
            paragraph_counts[record["language"]] += int(record["paragraph_count"])

    return {
        "archive_name": archive_path.name,
        "temp_jsonl": str(temp_jsonl),
        "documents_loaded": count,
        "languages": dict(sorted(language_counts.items())),
        "paragraphs_by_language": dict(sorted(paragraph_counts.items())),
    }


def _iter_xml_records(input_dir: Path) -> Iterator[dict[str, Any]]:
    for xml_path in sorted(input_dir.rglob("*.xml")):
        archive_lang = xml_path.parent.name.lower()
        if archive_lang not in JRC_ACQUIS_LANGS:
            archive_lang = ""
        if _is_aggregate_corpus_xml(str(xml_path.relative_to(input_dir)), archive_lang=archive_lang):
            continue
        record = _safe_parse_xml_record(
            xml_path.read_bytes(),
            archive_lang=archive_lang,
            source_name=str(xml_path.relative_to(input_dir)),
            source_archive=str(xml_path),
        )
        if record is not None:
            yield record


def iter_jrc_acquis_raw_records(
    input_dir: Path,
    *,
    languages: Optional[Iterable[str]] = None,
) -> Iterator[dict[str, Any]]:
    allowed = {lang.lower() for lang in languages} if languages else None
    input_dir = Path(input_dir)

    archives = sorted(input_dir.glob("jrc-*.tgz")) + sorted(input_dir.glob("jrc-*.tar.gz"))
    if archives:
        for archive_path in tqdm(archives, desc="JRC archives", unit="archive"):
            archive_lang = archive_path.stem.replace("jrc-", "").replace(".tar", "").lower()
            if allowed and archive_lang not in allowed:
                continue
            yield from _iter_archive_records(archive_path, archive_lang=archive_lang)
        return

    for record in _iter_xml_records(input_dir):
        if allowed and record["language"] not in allowed:
            continue
        yield record


def load_jrc_acquis_raw(
    *,
    input_dir: Path,
    output_dir: Path,
    languages: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    workers: int = 1,
) -> dict[str, Any]:
    """
    Read raw JRC-Acquis monolingual XML archives or extracted XML files and write
    an inspectable JSONL dump plus corpus stats.

    Expected raw inputs:
    - `data/JRC-ACQUIS/input/jrc-<lang>.tgz` archives from the official corpus page, or
    - extracted XML files under `input_dir`.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_jsonl = output_dir / "raw_documents.jsonl"
    stats_json = output_dir / "raw_load_stats.json"

    count = 0
    language_counts: Counter[str] = Counter()
    paragraph_counts: Counter[str] = Counter()
    archive_counts: Counter[str] = Counter()

    allowed = {lang.lower() for lang in languages} if languages else None
    archives = sorted(input_dir.glob("jrc-*.tgz")) + sorted(input_dir.glob("jrc-*.tar.gz"))
    selected_archives = []
    for archive_path in archives:
        archive_lang = archive_path.stem.replace("jrc-", "").replace(".tar", "").lower()
        if allowed and archive_lang not in allowed:
            continue
        selected_archives.append((archive_path, archive_lang))

    if selected_archives and workers > 1 and limit is None:
        temp_dir = output_dir / "_jrc_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            with raw_jsonl.open("w", encoding="utf-8") as out, ProcessPoolExecutor(
                max_workers=workers
            ) as pool, tqdm(total=len(selected_archives), desc="JRC archives", unit="archive") as archive_pbar, tqdm(
                desc="Load JRC XML documents",
                unit="doc",
            ) as doc_pbar:
                futures = {
                    pool.submit(
                        _process_archive_to_temp,
                        str(archive_path),
                        archive_lang,
                        str(temp_dir),
                    ): (archive_path, archive_lang)
                    for archive_path, archive_lang in selected_archives
                }
                for future in as_completed(futures):
                    result = future.result()
                    temp_jsonl = Path(result["temp_jsonl"])
                    if temp_jsonl.exists():
                        with temp_jsonl.open("r", encoding="utf-8") as fh:
                            shutil.copyfileobj(fh, out)
                        temp_jsonl.unlink()
                    count += int(result["documents_loaded"])
                    for lang, value in result["languages"].items():
                        language_counts[lang] += int(value)
                    for lang, value in result["paragraphs_by_language"].items():
                        paragraph_counts[lang] += int(value)
                    archive_counts[result["archive_name"]] += int(result["documents_loaded"])
                    archive_pbar.update(1)
                    doc_pbar.update(int(result["documents_loaded"]))
                    archive_pbar.set_postfix(lang=futures[future][1], docs=result["documents_loaded"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        progress_total = limit if limit is not None else None
        with raw_jsonl.open("w", encoding="utf-8") as out, tqdm(
            total=progress_total,
            desc="Load JRC XML documents",
            unit="doc",
        ) as pbar:
            for record in iter_jrc_acquis_raw_records(input_dir, languages=languages):
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                language_counts[record["language"]] += 1
                paragraph_counts[record["language"]] += int(record["paragraph_count"])
                archive_counts[Path(record["source_archive"]).name] += 1
                pbar.update(1)
                pbar.set_postfix(lang=record["language"], celex=record["celex"][:12])
                if limit is not None and count >= limit:
                    break

    if count == 0:
        raise ValueError(
            f"No JRC-Acquis XML inputs found in {input_dir}. "
            "Expected jrc-<lang>.tgz archives or extracted XML files."
        )

    stats = {
        "documents_loaded": count,
        "languages": dict(sorted(language_counts.items())),
        "paragraphs_by_language": dict(sorted(paragraph_counts.items())),
        "source_files": dict(sorted(archive_counts.items())),
        "raw_jsonl": str(raw_jsonl),
        "workers": workers if selected_archives and workers > 1 and limit is None else 1,
    }
    stats_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def count_jrc_acquis_prepared_records(prepared_dir: Path) -> int:
    stats_path = Path(prepared_dir) / "raw_load_stats.json"
    if not stats_path.exists():
        return 0
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return int(stats.get("documents_loaded", 0))


JRC_DOCUMENT_FIELDNAMES = [
    "id",
    "language",
    "title",
    "header_notes",
    "abstract",
    "context",
    "generation_context",
    "operative_context",
    "annex_context",
    "signature_context",
    "celex",
    "source",
    "paragraph_count",
    "body_paragraph_count",
    "eurovoc_ids",
    "source_archive",
]


JRC_PAIR_FIELDNAMES = [
    "pair_id",
    "celex",
    "lang_a",
    "corpus_id_a",
    "lang_b",
    "corpus_id_b",
]

JRC_QA_MIN_CHARS = 1500
JRC_QA_MAX_CHARS = 30000
JRC_QA_MIN_BODY_PARAGRAPHS = 4
JRC_INSPECTION_SAMPLE_PER_LANGUAGE = 5
JRC_RETRIEVAL_BODY_MAX_CHARS = 8000
JRC_RETRIEVAL_ANNEX_MAX_CHARS = 2000
JRC_RETRIEVAL_SIGNATURE_MAX_CHARS = 800
JRC_RETRIEVAL_TOTAL_MAX_CHARS = 12000
JRC_INSPECTION_FIELDNAMES = [
    *JRC_DOCUMENT_FIELDNAMES,
    "celex_language_count",
    "qa_candidate",
]


def _iter_batches(items: Iterable[Any], batch_size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _take_paragraph_budget(
    paragraphs: list[str],
    *,
    max_chars: int,
) -> list[str]:
    selected: list[str] = []
    used_chars = 0
    for paragraph in paragraphs:
        if used_chars + len(paragraph) > max_chars and selected:
            break
        selected.append(paragraph)
        used_chars += len(paragraph)
        if used_chars >= max_chars:
            break
    return selected


def _build_jrc_document_entry(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    celex = str(row.get("celex", "")).strip()
    lang = str(row.get("language", "")).strip().lower()
    source_name = str(row.get("source_name", ""))
    paragraphs = row.get("paragraphs") or []
    if not celex or not lang or not paragraphs:
        return None
    if _is_header_helper_record(
        celex=celex,
        source_name=source_name,
        title=str(row.get("title", "")),
    ):
        return None

    raw_body_paragraphs = [
        p.get("text", "")
        for p in paragraphs
        if p.get("section") == "body" and _normalize_jrc_text(p.get("text", ""))
    ]
    if not raw_body_paragraphs:
        raw_body_paragraphs = [
            p.get("text", "") for p in paragraphs if _normalize_jrc_text(p.get("text", ""))
        ]
    if not raw_body_paragraphs:
        return None
    raw_annex_paragraphs = [
        p.get("text", "")
        for p in paragraphs
        if p.get("section") == "annex" and _normalize_jrc_text(p.get("text", ""))
    ]
    raw_signature_paragraphs = [
        p.get("text", "")
        for p in paragraphs
        if p.get("section") == "signature" and _normalize_jrc_text(p.get("text", ""))
    ]

    cleaned_body_paragraphs, cleaned_changed = _clean_jrc_paragraphs(raw_body_paragraphs, lang)
    if not cleaned_body_paragraphs:
        return None
    cleaned_annex_paragraphs, _ = _clean_jrc_paragraphs(raw_annex_paragraphs, lang)
    cleaned_signature_paragraphs, _ = _clean_jrc_paragraphs(raw_signature_paragraphs, lang)
    header_notes = [
        _normalize_jrc_text(str(note))
        for note in (row.get("header_notes") or [])
        if _normalize_jrc_text(str(note))
    ]

    title = _derive_document_title(
        str(row.get("title", "")),
        cleaned_body_paragraphs,
        celex,
        lang=lang,
    )
    operative_paragraphs, trimmed_to_operative = _trim_jrc_to_operative_body(cleaned_body_paragraphs, lang)

    retrieval_body = _take_paragraph_budget(
        cleaned_body_paragraphs,
        max_chars=JRC_RETRIEVAL_BODY_MAX_CHARS,
    )
    retrieval_annex = _take_paragraph_budget(
        cleaned_annex_paragraphs,
        max_chars=JRC_RETRIEVAL_ANNEX_MAX_CHARS,
    )
    retrieval_signature = _take_paragraph_budget(
        cleaned_signature_paragraphs,
        max_chars=JRC_RETRIEVAL_SIGNATURE_MAX_CHARS,
    )

    full_parts = []
    if title:
        full_parts.append(title)
    full_parts.extend(retrieval_body)
    full_parts.extend(retrieval_annex)
    full_parts.extend(retrieval_signature)
    full_text = "\n\n".join(part for part in full_parts if part).strip()
    if len(full_text) > JRC_RETRIEVAL_TOTAL_MAX_CHARS:
        full_text = full_text[:JRC_RETRIEVAL_TOTAL_MAX_CHARS].rsplit("\n\n", 1)[0].strip() or full_text[
            :JRC_RETRIEVAL_TOTAL_MAX_CHARS
        ].strip()
    operative_text = "\n\n".join(operative_paragraphs).strip()
    annex_text = "\n\n".join(cleaned_annex_paragraphs).strip()
    signature_text = "\n\n".join(cleaned_signature_paragraphs).strip()
    if not full_text:
        return None

    article_start_idx = 0
    for idx, paragraph in enumerate(cleaned_body_paragraphs):
        if _looks_like_article_heading(paragraph, lang):
            article_start_idx = idx
            break

    preamble_paragraphs = cleaned_body_paragraphs[:article_start_idx] if article_start_idx > 0 else []
    preamble_candidates = [
        paragraph
        for paragraph in preamble_paragraphs
        if not _looks_like_institution_heading(paragraph)
        and not _looks_like_adoption_formula(paragraph)
        and not _looks_like_reference_line(paragraph)
        and len(paragraph) >= 60
        and len(paragraph.split()) >= 8
    ]

    selected_preamble: list[str] = []
    preamble_chars = 0
    for paragraph in reversed(preamble_candidates):
        if len(selected_preamble) >= 6:
            break
        if preamble_chars + len(paragraph) > 1800 and selected_preamble:
            break
        selected_preamble.append(paragraph)
        preamble_chars += len(paragraph)
    selected_preamble.reverse()

    operative_focus: list[str] = []
    operative_focus_chars = 0
    for paragraph in operative_paragraphs:
        if len(operative_focus) >= 18:
            break
        if operative_focus_chars + len(paragraph) > 3200 and operative_focus:
            break
        operative_focus.append(paragraph)
        operative_focus_chars += len(paragraph)

    annex_focus: list[str] = []
    annex_focus_chars = 0
    for paragraph in cleaned_annex_paragraphs:
        if len(annex_focus) >= 8:
            break
        if annex_focus_chars + len(paragraph) > 1800 and annex_focus:
            break
        annex_focus.append(paragraph)
        annex_focus_chars += len(paragraph)

    generation_parts: list[str] = []
    if title:
        generation_parts.append(title)
    generation_parts.extend(selected_preamble)
    generation_parts.extend(operative_focus or operative_paragraphs[:10])
    generation_parts.extend(annex_focus)
    generation_context = "\n\n".join(part for part in generation_parts if part).strip() or operative_text or full_text

    corpus_id = f"{celex}_{lang}"
    eurovoc_ids = "|".join(row.get("eurovoc_ids") or [])

    return {
        "celex": celex,
        "lang": lang,
        "corpus_id": corpus_id,
        "text_length": len(full_text),
        "cleaned_formatting": cleaned_changed,
        "trimmed_to_operative": trimmed_to_operative,
        "full_row": {
            "id": corpus_id,
            "language": lang,
            "title": title,
            "header_notes": "\n\n".join(header_notes),
            "abstract": generation_context[:800],
            "context": full_text,
            "generation_context": generation_context,
            "operative_context": operative_text,
            "annex_context": annex_text,
            "signature_context": signature_text,
            "celex": celex,
            "source": "jrc-acquis",
            "paragraph_count": str(int(row.get("paragraph_count", len(paragraphs)))),
            "body_paragraph_count": str(len(cleaned_body_paragraphs)),
            "eurovoc_ids": eurovoc_ids,
            "source_archive": str(row.get("source_archive", "")),
        },
        "mteb_row": {"_id": corpus_id, "title": title, "text": full_text},
    }


def _build_jrc_document_batch(lines: list[str]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    language_counts: Counter[str] = Counter()
    body_chars_by_lang: Counter[str] = Counter()
    docs_short = 0
    docs_long = 0
    formatting_cleaned = 0
    operative_trimmed = 0

    for line in lines:
        record = _build_jrc_document_entry(json.loads(line))
        if record is None:
            continue
        records.append(record)
        lang = record["lang"]
        text_length = int(record["text_length"])
        language_counts[lang] += 1
        body_chars_by_lang[lang] += text_length
        formatting_cleaned += int(bool(record.get("cleaned_formatting")))
        operative_trimmed += int(bool(record.get("trimmed_to_operative")))
        if text_length < 1500:
            docs_short += 1
        if text_length > 30000:
            docs_long += 1

    return {
        "records": records,
        "language_counts": dict(language_counts),
        "body_chars_by_lang": dict(body_chars_by_lang),
        "docs_short": docs_short,
        "docs_long": docs_long,
        "formatting_cleaned": formatting_cleaned,
        "operative_trimmed": operative_trimmed,
    }


def _build_jrc_pair_batch(items: list[tuple[str, list[tuple[str, str]]]]) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    pair_count = 0
    celex_multilingual = 0
    languages_per_celex: Counter[int] = Counter()

    for celex, docs in items:
        deduped_docs = sorted(set(docs))
        languages_per_celex[len(deduped_docs)] += 1
        if len(deduped_docs) >= 2:
            celex_multilingual += 1
        for i in range(len(deduped_docs)):
            lang_a, cid_a = deduped_docs[i]
            for j in range(i + 1, len(deduped_docs)):
                lang_b, cid_b = deduped_docs[j]
                rows.append(
                    {
                        "pair_id": f"{celex}__{lang_a}__{lang_b}",
                        "celex": celex,
                        "lang_a": lang_a,
                        "corpus_id_a": cid_a,
                        "lang_b": lang_b,
                        "corpus_id_b": cid_b,
                    }
                )
                pair_count += 1

    return {
        "rows": rows,
        "pair_count": pair_count,
        "celex_multilingual": celex_multilingual,
        "languages_per_celex": dict(languages_per_celex),
    }


def _is_jrc_qa_candidate(row: dict[str, str]) -> bool:
    context = row.get("context", "") or ""
    body_paragraph_count = int(row.get("body_paragraph_count", "0") or 0)
    return (
        len(context) >= JRC_QA_MIN_CHARS
        and len(context) <= JRC_QA_MAX_CHARS
        and body_paragraph_count >= JRC_QA_MIN_BODY_PARAGRAPHS
    )


def _set_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def build_jrc_acquis_document_corpus(
    *,
    raw_jsonl_path: Path,
    preprocessed_dir: Path,
    full_output_path: Path,
    output_path: Path,
    workers: int = 1,
) -> dict[str, Any]:
    """
    Build a document-level multilingual corpus from `raw_documents.jsonl`.

    Outputs:
    - `corpus_full.csv`: one row per `(celex, language)` with full document text
    - `corpus.csv`: MTEB-style document corpus (`_id`, `title`, `text`)
    - `document_pairs_all.csv`: all undirected language pairs for the same `celex`
    - `document_corpus_stats.json`: summary stats for inspection
    """
    raw_jsonl_path = Path(raw_jsonl_path)
    preprocessed_dir = Path(preprocessed_dir)
    full_output_path = Path(full_output_path)
    output_path = Path(output_path)

    if not raw_jsonl_path.is_file():
        raise ValueError(f"JRC-Acquis raw JSONL not found: {raw_jsonl_path}")

    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    full_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(workers or 1))

    pairs_path = preprocessed_dir / "document_pairs_all.csv"
    stats_path = preprocessed_dir / "document_corpus_stats.json"

    per_celex: dict[str, list[tuple[str, str]]] = {}
    language_counts: Counter[str] = Counter()
    body_chars_by_lang: Counter[str] = Counter()
    doc_count = 0
    docs_short = 0
    docs_long = 0
    formatting_cleaned = 0
    operative_trimmed = 0
    doc_batch_size = 500
    pair_batch_size = 200

    with (
        raw_jsonl_path.open("r", encoding="utf-8") as src,
        full_output_path.open("w", encoding="utf-8", newline="") as full_fh,
        output_path.open("w", encoding="utf-8", newline="") as mteb_fh,
        tqdm(desc="Build JRC document corpus", unit="doc") as pbar,
    ):
        full_writer = csv.DictWriter(full_fh, fieldnames=JRC_DOCUMENT_FIELDNAMES)
        full_writer.writeheader()
        mteb_writer = csv.DictWriter(mteb_fh, fieldnames=["_id", "title", "text"])
        mteb_writer.writeheader()

        doc_batches = _iter_batches(src, doc_batch_size)
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results_iter = executor.map(_build_jrc_document_batch, doc_batches)
                for result in results_iter:
                    records = result["records"]
                    for record in records:
                        full_writer.writerow(record["full_row"])
                        mteb_writer.writerow(record["mteb_row"])
                        per_celex.setdefault(record["celex"], []).append(
                            (record["lang"], record["corpus_id"])
                        )
                    for lang, count in result["language_counts"].items():
                        language_counts[lang] += int(count)
                    for lang, chars in result["body_chars_by_lang"].items():
                        body_chars_by_lang[lang] += int(chars)
                    doc_count += len(records)
                    docs_short += int(result["docs_short"])
                    docs_long += int(result["docs_long"])
                    formatting_cleaned += int(result["formatting_cleaned"])
                    operative_trimmed += int(result["operative_trimmed"])
                    pbar.update(len(records))
                    if records:
                        last = records[-1]
                        pbar.set_postfix(lang=last["lang"], celex=last["celex"][:12])
        else:
            for batch in doc_batches:
                result = _build_jrc_document_batch(batch)
                records = result["records"]
                for record in records:
                    full_writer.writerow(record["full_row"])
                    mteb_writer.writerow(record["mteb_row"])
                    per_celex.setdefault(record["celex"], []).append((record["lang"], record["corpus_id"]))
                for lang, count in result["language_counts"].items():
                    language_counts[lang] += int(count)
                for lang, chars in result["body_chars_by_lang"].items():
                    body_chars_by_lang[lang] += int(chars)
                doc_count += len(records)
                docs_short += int(result["docs_short"])
                docs_long += int(result["docs_long"])
                formatting_cleaned += int(result["formatting_cleaned"])
                operative_trimmed += int(result["operative_trimmed"])
                pbar.update(len(records))
                if records:
                    last = records[-1]
                    pbar.set_postfix(lang=last["lang"], celex=last["celex"][:12])

    pair_count = 0
    celex_multilingual = 0
    languages_per_celex: Counter[int] = Counter()
    pair_items = list(sorted(per_celex.items()))
    with pairs_path.open("w", encoding="utf-8", newline="") as pairs_fh, tqdm(
        total=len(pair_items),
        desc="Build JRC document pairs",
        unit="celex",
    ) as pair_pbar:
        pair_writer = csv.DictWriter(pairs_fh, fieldnames=JRC_PAIR_FIELDNAMES)
        pair_writer.writeheader()

        pair_batches = _iter_batches(pair_items, pair_batch_size)
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results_iter = executor.map(_build_jrc_pair_batch, pair_batches)
                for result in results_iter:
                    for row in result["rows"]:
                        pair_writer.writerow(row)
                    pair_count += int(result["pair_count"])
                    celex_multilingual += int(result["celex_multilingual"])
                    for n_langs, count in result["languages_per_celex"].items():
                        languages_per_celex[int(n_langs)] += int(count)
                    batch_celex = sum(int(count) for count in result["languages_per_celex"].values())
                    pair_pbar.update(batch_celex)
        else:
            for batch in pair_batches:
                result = _build_jrc_pair_batch(batch)
                for row in result["rows"]:
                    pair_writer.writerow(row)
                pair_count += int(result["pair_count"])
                celex_multilingual += int(result["celex_multilingual"])
                for n_langs, count in result["languages_per_celex"].items():
                    languages_per_celex[int(n_langs)] += int(count)
                pair_pbar.update(len(batch))

    multilingual_celexes = {
        celex for celex, docs in per_celex.items() if len(set(docs)) >= 2
    }
    celex_language_counts = {
        celex: len(set(docs)) for celex, docs in per_celex.items() if celex in multilingual_celexes
    }
    multilingual_full_path = preprocessed_dir / "corpus_multilingual_full.csv"
    multilingual_mteb_path = preprocessed_dir / "corpus_multilingual.csv"
    qa_candidates_path = preprocessed_dir / "corpus_qa_candidates.csv"
    inspection_sample_path = preprocessed_dir / "inspection_sample.csv"

    multilingual_docs_written = 0
    qa_candidates_written = 0
    multilingual_docs_by_language: Counter[str] = Counter()
    qa_candidates_by_language: Counter[str] = Counter()
    inspection_rows_written = 0
    inspection_rows_by_language: Counter[str] = Counter()

    with (
        full_output_path.open("r", encoding="utf-8", newline="") as full_in,
        multilingual_full_path.open("w", encoding="utf-8", newline="") as multilingual_full_fh,
        multilingual_mteb_path.open("w", encoding="utf-8", newline="") as multilingual_mteb_fh,
        qa_candidates_path.open("w", encoding="utf-8", newline="") as qa_fh,
        inspection_sample_path.open("w", encoding="utf-8", newline="") as inspection_fh,
        tqdm(total=doc_count, desc="Build JRC derived views", unit="doc") as derived_pbar,
    ):
        _set_csv_field_size_limit()
        reader = csv.DictReader(full_in)
        multilingual_full_writer = csv.DictWriter(
            multilingual_full_fh, fieldnames=JRC_DOCUMENT_FIELDNAMES
        )
        multilingual_full_writer.writeheader()
        multilingual_mteb_writer = csv.DictWriter(
            multilingual_mteb_fh, fieldnames=["_id", "title", "text"]
        )
        multilingual_mteb_writer.writeheader()
        qa_writer = csv.DictWriter(qa_fh, fieldnames=JRC_DOCUMENT_FIELDNAMES)
        qa_writer.writeheader()
        inspection_writer = csv.DictWriter(
            inspection_fh, fieldnames=JRC_INSPECTION_FIELDNAMES
        )
        inspection_writer.writeheader()

        for row in reader:
            derived_pbar.update(1)
            celex = row.get("celex", "")
            if celex not in multilingual_celexes:
                continue

            multilingual_full_writer.writerow({field: row.get(field, "") for field in JRC_DOCUMENT_FIELDNAMES})
            multilingual_mteb_writer.writerow(
                {
                    "_id": row.get("id", ""),
                    "title": row.get("title", ""),
                    "text": row.get("context", ""),
                }
            )

            lang = row.get("language", "")
            multilingual_docs_written += 1
            multilingual_docs_by_language[lang] += 1

            is_qa_candidate = _is_jrc_qa_candidate(row)
            if is_qa_candidate:
                qa_writer.writerow({field: row.get(field, "") for field in JRC_DOCUMENT_FIELDNAMES})
                qa_candidates_written += 1
                qa_candidates_by_language[lang] += 1
                if inspection_rows_by_language[lang] < JRC_INSPECTION_SAMPLE_PER_LANGUAGE:
                    inspection_writer.writerow(
                        {
                            **{field: row.get(field, "") for field in JRC_DOCUMENT_FIELDNAMES},
                            "celex_language_count": str(celex_language_counts.get(celex, 0)),
                            "qa_candidate": "1",
                        }
                    )
                    inspection_rows_written += 1
                    inspection_rows_by_language[lang] += 1

    avg_chars_by_lang = {
        lang: round(body_chars_by_lang[lang] / count, 2)
        for lang, count in sorted(language_counts.items())
        if count
    }
    stats = {
        "documents_written": doc_count,
        "celex_total": len(per_celex),
        "celex_multilingual": celex_multilingual,
        "pairs_written": pair_count,
        "languages": dict(sorted(language_counts.items())),
        "multilingual_docs_written": multilingual_docs_written,
        "multilingual_docs_by_language": dict(sorted(multilingual_docs_by_language.items())),
        "qa_candidates_written": qa_candidates_written,
        "qa_candidates_by_language": dict(sorted(qa_candidates_by_language.items())),
        "inspection_rows_written": inspection_rows_written,
        "inspection_rows_by_language": dict(sorted(inspection_rows_by_language.items())),
        "avg_body_chars_by_language": avg_chars_by_lang,
        "languages_per_celex_distribution": dict(sorted(languages_per_celex.items())),
        "docs_under_1500_chars": docs_short,
        "docs_over_30000_chars": docs_long,
        "docs_with_formatting_cleaned": formatting_cleaned,
        "docs_trimmed_to_operative_body": operative_trimmed,
        "corpus_full_csv": str(full_output_path),
        "corpus_csv": str(output_path),
        "pairs_csv": str(pairs_path),
        "multilingual_full_csv": str(multilingual_full_path),
        "multilingual_corpus_csv": str(multilingual_mteb_path),
        "qa_candidates_csv": str(qa_candidates_path),
        "inspection_sample_csv": str(inspection_sample_path),
        "qa_filter": {
            "min_chars": JRC_QA_MIN_CHARS,
            "max_chars": JRC_QA_MAX_CHARS,
            "min_body_paragraphs": JRC_QA_MIN_BODY_PARAGRAPHS,
        },
        "workers": workers,
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def parse_jrc_acquis_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Load raw JRC-Acquis XML archives into an inspectable JSONL dump."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "data" / "JRC-ACQUIS" / "input",
        help="Directory containing jrc-<lang>.tgz archives or extracted XML files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "JRC-ACQUIS" / "prepared",
        help="Directory to write raw_documents.jsonl and raw_load_stats.json.",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        metavar="LANG",
        help=f"Optional ISO language subset. Supported: {', '.join(JRC_ACQUIS_LANGS)}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of XML documents to load for inspection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_jrc_acquis_args()
    stats = load_jrc_acquis_raw(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        languages=args.languages,
        limit=args.limit,
    )
    print("Loaded raw JRC-Acquis documents:")
    print("  Input:", args.input_dir)
    print("  Output:", stats["raw_jsonl"])
    print("  Documents:", stats["documents_loaded"])
    print("  Languages:", stats["languages"])


__all__ = [
    "JRC_ACQUIS_LANGS",
    "count_jrc_acquis_input_files",
    "count_jrc_acquis_prepared_records",
    "download_jrc_acquis_archives",
    "iter_jrc_acquis_raw_records",
    "load_jrc_acquis_raw",
    "main",
    "parse_jrc_acquis_args",
]


if __name__ == "__main__":
    main()
