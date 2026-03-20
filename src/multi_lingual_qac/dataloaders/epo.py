from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import html
import json
import os
import re
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterator, List

from tqdm import tqdm

from src.multi_lingual_qac.constants import DEFAULT_LANGS

CHEMISTRY_CLASSIFICATION_PREFIXES = [
    "C",
    "A01N",
    "A23L",
    "A61K",
    "A61P",
    "B01D",
    "B01F",
    "B01J",
    "B01L",
    "C25",
    "G01N",
    "H01M",
]

CHEMISTRY_KEYWORDS = [
    "adhesive",
    "antibody",
    "battery",
    "biomarker",
    "catalyst",
    "cell culture",
    "chemical",
    "chemistry",
    "coating",
    "composition",
    "compound",
    "crystal form",
    "detergent",
    "drug",
    "electrolyte",
    "excipient",
    "fermentation",
    "formulation",
    "inhibitor",
    "material",
    "molecule",
    "nanoparticle",
    "peptide",
    "pharmaceutical",
    "pharmaceutically",
    "polymer",
    "protein",
    "resin",
    "semiconductor composition",
    "slurry",
    "solvent",
    "surfactant",
    "synthesis",
    "therapeutic",
]

CLASSIFICATION_CODE_RE = re.compile(r"([A-HY]\d{2}[A-Z]?\s*\d+(?:/\d+)?)")
AUXILIARY_XML_RE = re.compile(r"__(?:TOC|SL\d+)\.xml$", re.IGNORECASE)
DESCRIPTION_MAX_CHARS = 2000
FIRST_CLAIM_MAX_CHARS = 1500
MIN_ABSTRACT_WORDS = 50
PREFERRED_TEXT_LANGUAGES = ["fr", "de", "en"]


def clean_text(s: str) -> str:
    """Decode entities and normalize patent text for CSV output."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = html.unescape(s)
    s = s.replace("\ufeff", " ").replace("\u00ad", "").replace("\xa0", " ")
    s = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([(\[{])\s+", r"\1", s)
    s = re.sub(r"\s+([)\]}])", r"\1", s)
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def word_count(text: str) -> int:
    return len((text or "").split())


def iter_epo_zip_files(input_dir: Path) -> Iterator[Path]:
    """Yield patent zip files from EPO package folders."""
    input_dir = Path(input_dir)
    for zip_path in sorted(input_dir.rglob("*.zip")):
        if zip_path.is_file():
            yield zip_path


def count_epo_xml_files(xml_dir: Path) -> int:
    """Count extracted XML files in the EPO XML directory."""
    xml_dir = Path(xml_dir)
    if not xml_dir.exists():
        return 0
    return sum(1 for path in xml_dir.glob("*.xml") if path.is_file())


def iter_epo_patent_xml_files(xml_dir: Path) -> Iterator[Path]:
    """Yield extracted EPO patent XML files, excluding package TOC XMLs."""
    xml_dir = Path(xml_dir)
    if not xml_dir.exists():
        return
    for xml_path in sorted(xml_dir.glob("*.xml")):
        if xml_path.is_file() and not AUXILIARY_XML_RE.search(xml_path.name):
            yield xml_path


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = value.strip("._")
    return value or "unknown"


def _build_xml_output_name(zip_path: Path, input_dir: Path, member_name: str) -> str:
    rel_zip = zip_path.relative_to(input_dir)
    package_name = rel_zip.parts[0] if len(rel_zip.parts) >= 1 else "package"
    doc_bucket = rel_zip.parts[2] if len(rel_zip.parts) >= 3 else "doc"
    zip_stem = zip_path.stem
    member_stem = Path(member_name).stem
    suffix = Path(member_name).suffix.lower() or ".xml"

    name_parts = [_slugify(package_name), _slugify(doc_bucket), _slugify(zip_stem)]
    if member_stem.lower() != zip_stem.lower():
        name_parts.append(_slugify(member_stem))
    return "__".join(name_parts) + suffix


def _dedupe_output_path(output_dir: Path, file_name: str) -> Path:
    candidate = output_dir / file_name
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    index = 2
    while True:
        deduped = output_dir / f"{stem}__{index}{suffix}"
        if not deduped.exists():
            return deduped
        index += 1


def extract_epo_xml_files(
    input_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> Dict[str, int]:
    """
    Extract XML files from EPO patent zip archives.

    Output filenames include package and document bucket context so XML files from
    different folders do not overwrite each other.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise ValueError(f"EPO input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_files = list(iter_epo_zip_files(input_dir))

    stats = {
        "zip_files": len(zip_files),
        "xml_files": 0,
        "skipped_existing": 0,
        "bad_zips": 0,
    }

    for zip_path in tqdm(zip_files, desc="Extract EPO XMLs", unit="zip"):
        try:
            with zipfile.ZipFile(zip_path) as archive:
                xml_members = [
                    member
                    for member in archive.namelist()
                    if member.lower().endswith(".xml") and not member.endswith("/")
                ]
                for member_name in xml_members:
                    output_name = _build_xml_output_name(zip_path, input_dir, member_name)
                    output_path = output_dir / output_name
                    if output_path.exists() and not overwrite:
                        stats["skipped_existing"] += 1
                        continue
                    if output_path.exists() and overwrite:
                        final_output_path = output_path
                    else:
                        final_output_path = _dedupe_output_path(output_dir, output_name)

                    final_output_path.write_bytes(archive.read(member_name))
                    stats["xml_files"] += 1
        except zipfile.BadZipFile:
            stats["bad_zips"] += 1
            tqdm.write(f"  Skipping bad zip: {zip_path}")

    return stats


def _normalize_code(raw_text: str) -> str:
    raw_text = clean_text(raw_text)
    match = CLASSIFICATION_CODE_RE.search(raw_text)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()


def _normalized_prefix(value: str) -> str:
    return value.upper().replace(" ", "")


def _extract_title_localized(root: ET.Element) -> List[Dict[str, str]]:
    titles: List[Dict[str, str]] = []
    title_block = root.find(".//B540")
    if title_block is None:
        return titles

    current_lang = ""
    for child in title_block:
        if child.tag == "B541":
            current_lang = clean_text(child.text or "").lower()
        elif child.tag == "B542":
            text = clean_text(child.text or "")
            if current_lang and text:
                titles.append({"language": current_lang, "text": text})
    return titles


def _truncate_text(text: str, *, max_chars: int) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    snippet = text[:max_chars].rsplit(" ", 1)[0].strip()
    return snippet or text[:max_chars].strip()


def _extract_text_blocks(root: ET.Element, tag_name: str) -> List[Dict[str, str]]:
    blocks: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for node in root.findall(f".//{tag_name}"):
        language = clean_text(node.attrib.get("lang", "")).lower()
        text = clean_text(" ".join("".join(node.itertext()).split()))
        if not text:
            continue
        key = (language, text)
        if key in seen:
            continue
        seen.add(key)
        blocks.append({"language": language, "text": text})
    return blocks


def _get_text_for_language(blocks: List[Dict[str, str]], language: str) -> str:
    language = (language or "").lower()
    for block in blocks:
        if block["language"] == language and block["text"]:
            return block["text"]
    return ""


def _choose_best_text(
    blocks: List[Dict[str, str]],
    *,
    target_language: str,
    source_language: str,
    min_words: int = 0,
) -> str:
    if not blocks:
        return ""
    candidates = [
        block for block in blocks
        if block["text"] and word_count(block["text"]) >= min_words
    ]
    if not candidates:
        return ""

    target_language = (target_language or "").lower()
    source_language = (source_language or "").lower()
    for candidate_language in [
        target_language,
        *PREFERRED_TEXT_LANGUAGES,
        source_language,
        "",
    ]:
        if not candidate_language:
            continue
        for block in candidates:
            if block["language"] == candidate_language:
                return block["text"]
    return candidates[0]["text"]


def _choose_preferred_corpus_language(record: Dict[str, Any]) -> str:
    abstract_blocks = record["abstract_localized"]
    for language in PREFERRED_TEXT_LANGUAGES:
        if word_count(_get_text_for_language(abstract_blocks, language)) >= MIN_ABSTRACT_WORDS:
            return language

    source_language = (record.get("source_language") or "").lower()
    if word_count(_get_text_for_language(abstract_blocks, source_language)) >= MIN_ABSTRACT_WORDS:
        return source_language

    for block in abstract_blocks:
        if word_count(block["text"]) >= MIN_ABSTRACT_WORDS:
            return block["language"] or source_language or "en"
    return ""


def _choose_title_for_language(record: Dict[str, Any], language: str) -> str:
    title_text = _get_text_for_language(record["title_localized"], language)
    if title_text:
        return title_text
    for fallback_language in PREFERRED_TEXT_LANGUAGES:
        title_text = _get_text_for_language(record["title_localized"], fallback_language)
        if title_text:
            return title_text
    return record["title"]


def _extract_first_claim_text(root: ET.Element) -> List[Dict[str, str]]:
    claims_by_lang: Dict[str, str] = {}
    for claims_node in root.findall(".//claims"):
        language = clean_text(claims_node.attrib.get("lang", "")).lower()
        claim_node = claims_node.find(".//claim")
        if claim_node is None:
            continue
        text = clean_text(" ".join("".join(claim_node.itertext()).split()))
        if text and language not in claims_by_lang:
            claims_by_lang[language] = _truncate_text(text, max_chars=FIRST_CLAIM_MAX_CHARS)
    return [{"language": lang, "text": text} for lang, text in claims_by_lang.items()]


def _extract_classification_codes(root: ET.Element, tag_name: str) -> List[str]:
    codes: List[str] = []
    seen: set[str] = set()
    for node in root.findall(f".//{tag_name}"):
        code = _normalize_code(node.findtext("text", default=""))
        if code and code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def _extract_party_names(root: ET.Element, xpath: str) -> List[str]:
    names: List[str] = []
    seen: set[str] = set()
    for node in root.findall(xpath):
        name = clean_text(node.findtext("snm", default=""))
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _extract_designated_states(root: ET.Element) -> List[str]:
    states: List[str] = []
    seen: set[str] = set()
    for node in root.findall(".//B840/ctry"):
        state = clean_text(node.text or "").upper()
        if state and state not in seen:
            seen.add(state)
            states.append(state)
    return states


def _extract_priority_numbers(root: ET.Element) -> List[str]:
    numbers: List[str] = []
    seen: set[str] = set()
    for node in root.findall(".//B300/B310"):
        number = clean_text("".join(node.itertext()))
        if number and number not in seen:
            seen.add(number)
            numbers.append(number)
    return numbers


def _extract_priority_dates(root: ET.Element) -> List[str]:
    dates: List[str] = []
    seen: set[str] = set()
    for node in root.findall(".//B300/B320/date"):
        date_value = clean_text(node.text or "")
        if date_value and date_value not in seen:
            seen.add(date_value)
            dates.append(date_value)
    return dates


def _has_chemistry_classification(codes: List[str]) -> List[str]:
    matches: List[str] = []
    prefixes = [_normalized_prefix(prefix) for prefix in CHEMISTRY_CLASSIFICATION_PREFIXES]
    for code in codes:
        normalized = _normalized_prefix(code)
        if any(normalized.startswith(prefix) for prefix in prefixes):
            matches.append(code)
    return matches


def _keyword_hits(texts: List[str]) -> List[str]:
    haystack = " ".join(clean_text(text).lower() for text in texts if text)
    return [keyword for keyword in CHEMISTRY_KEYWORDS if keyword in haystack]


def analyze_epo_chemistry(record: Dict[str, Any]) -> Dict[str, Any]:
    """Score whether a parsed EPO record is chemistry-related."""
    ipc_matches = _has_chemistry_classification(record.get("ipc_codes", []))
    cpc_matches = _has_chemistry_classification(record.get("cpc_codes", []))
    keyword_hits = _keyword_hits([title["text"] for title in record.get("title_localized", [])])

    score = 0
    reasons: List[str] = []
    if ipc_matches:
        score += 2
        reasons.append(f"IPC match: {', '.join(ipc_matches[:5])}")
    if cpc_matches:
        score += 2
        reasons.append(f"CPC match: {', '.join(cpc_matches[:5])}")
    if keyword_hits:
        score += 1
        reasons.append(f"Title keywords: {', '.join(keyword_hits[:8])}")

    if ipc_matches or cpc_matches:
        label = "chemistry_core"
    elif keyword_hits:
        label = "chemistry_related"
    else:
        label = "not_chemistry"

    return {
        "score": score,
        "label": label,
        "keep": label != "not_chemistry",
        "reasons": reasons,
    }


def parse_epo_patent_xml(xml_path: Path) -> Dict[str, Any]:
    """Parse a single EPO patent XML into a normalized metadata record."""
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()

    if root.tag != "ep-patent-document":
        raise ValueError(f"Unsupported EPO XML root tag: {root.tag}")

    title_localized = _extract_title_localized(root)
    source_language = clean_text(root.attrib.get("lang", "")).lower()
    english_title = next((item["text"] for item in title_localized if item["language"] == "en"), "")
    primary_title = english_title or (title_localized[0]["text"] if title_localized else "")
    primary_language = "en" if english_title else (title_localized[0]["language"] if title_localized else source_language)

    record = {
        "xml_file": xml_path.name,
        "document_id": clean_text(root.attrib.get("id", "")),
        "publication_number": clean_text(root.attrib.get("doc-number", "")),
        "application_number": clean_text(root.findtext(".//B200/B210", default="")),
        "country_code": clean_text(root.attrib.get("country", "")) or clean_text(root.findtext(".//B100/B190", default="")),
        "publication_date": clean_text(root.attrib.get("date-publ", "")) or clean_text(root.findtext(".//B140/date", default="")),
        "filing_date": clean_text(root.findtext(".//B220/date", default="")),
        "priority_dates": _extract_priority_dates(root),
        "priority_numbers": _extract_priority_numbers(root),
        "kind": clean_text(root.attrib.get("kind", "")),
        "source_language": source_language or primary_language,
        "title": primary_title,
        "title_localized": title_localized,
        "abstract_localized": _extract_text_blocks(root, "abstract"),
        "description_localized": _extract_text_blocks(root, "description"),
        "first_claim_localized": _extract_first_claim_text(root),
        "ipc_codes": _extract_classification_codes(root, "classification-ipcr"),
        "cpc_codes": _extract_classification_codes(root, "classification-cpc"),
        "applicants": _extract_party_names(root, ".//B710/B711"),
        "inventors": _extract_party_names(root, ".//B720/B721"),
        "representatives": _extract_party_names(root, ".//B740/B741"),
        "designated_states": _extract_designated_states(root),
    }

    if not record["document_id"] or not record["publication_number"]:
        raise ValueError(f"Missing essential patent identifiers in {xml_path.name}")
    if not record["title_localized"] and not record["title"]:
        raise ValueError(f"Missing title data in {xml_path.name}")

    record["chemistry"] = analyze_epo_chemistry(record)
    return record


def _build_context(record: Dict[str, Any], title_text: str, language: str) -> str:
    parts = []
    if title_text:
        parts.append(f"Title ({language}): {title_text}")
    abstract_text = _choose_best_text(
        record["abstract_localized"],
        target_language=language,
        source_language=record["source_language"],
    )
    if abstract_text:
        parts.append(f"Abstract: {abstract_text}")
    description_text = _choose_best_text(
        record["description_localized"],
        target_language=language,
        source_language=record["source_language"],
    )
    if description_text:
        parts.append(
            f"Description: {_truncate_text(description_text, max_chars=DESCRIPTION_MAX_CHARS)}"
        )
    first_claim_text = _choose_best_text(
        record["first_claim_localized"],
        target_language=language,
        source_language=record["source_language"],
    )
    if first_claim_text:
        parts.append(f"First claim: {first_claim_text}")
    english_title = next(
        (item["text"] for item in record["title_localized"] if item["language"] == "en"),
        "",
    )
    if english_title and language != "en":
        parts.append(f"English title: {english_title}")
    if record["ipc_codes"]:
        parts.append(f"IPC: {', '.join(record['ipc_codes'][:10])}")
    if record["cpc_codes"]:
        parts.append(f"CPC: {', '.join(record['cpc_codes'][:10])}")
    if record["applicants"]:
        parts.append(f"Applicants: {', '.join(record['applicants'][:5])}")
    if record["inventors"]:
        parts.append(f"Inventors: {', '.join(record['inventors'][:5])}")
    if record["priority_numbers"]:
        parts.append(f"Priority numbers: {', '.join(record['priority_numbers'][:5])}")
    if record["chemistry"]["reasons"]:
        parts.append(f"Chemistry signal: {'; '.join(record['chemistry']['reasons'])}")
    return "\n\n".join(parts).strip()


def _record_to_rows(record: Dict[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    titles = record["title_localized"] or [{
        "language": record["source_language"] or "unknown",
        "text": record["title"],
    }]
    for title_entry in titles:
        language = title_entry["language"] or record["source_language"] or "unknown"
        title_text = title_entry["text"]
        abstract_text = _choose_best_text(
            record["abstract_localized"],
            target_language=language,
            source_language=record["source_language"],
        )
        description_text = _choose_best_text(
            record["description_localized"],
            target_language=language,
            source_language=record["source_language"],
        )
        first_claim_text = _choose_best_text(
            record["first_claim_localized"],
            target_language=language,
            source_language=record["source_language"],
        )
        rows.append({
            "id": f"{record['document_id'] or record['publication_number']}_{language}",
            "language": language,
            "title": title_text,
            "abstract": abstract_text,
            "description": _truncate_text(description_text, max_chars=DESCRIPTION_MAX_CHARS) if description_text else "",
            "first_claim": first_claim_text,
            "context": _build_context(record, title_text, language),
            "publication_number": record["publication_number"],
            "country_code": record["country_code"],
            "publication_date": record["publication_date"],
            "source": "epo",
            "document_id": record["document_id"],
            "application_number": record["application_number"],
            "filing_date": record["filing_date"],
            "kind": record["kind"],
            "source_language": record["source_language"],
            "available_title_languages": ",".join(
                item["language"] for item in record["title_localized"] if item["language"]
            ),
            "title_localized_json": json.dumps(record["title_localized"], ensure_ascii=False),
            "ipc_codes": "|".join(record["ipc_codes"]),
            "cpc_codes": "|".join(record["cpc_codes"]),
            "applicants": "|".join(record["applicants"]),
            "inventors": "|".join(record["inventors"]),
            "representatives": "|".join(record["representatives"]),
            "priority_numbers": "|".join(record["priority_numbers"]),
            "priority_dates": "|".join(record["priority_dates"]),
            "designated_states": "|".join(record["designated_states"]),
            "chemistry_label": record["chemistry"]["label"],
            "chemistry_score": str(record["chemistry"]["score"]),
            "chemistry_reasons": " | ".join(record["chemistry"]["reasons"]),
            "xml_file": record["xml_file"],
        })
    return rows


def _record_to_corpus_row(record: Dict[str, Any]) -> Dict[str, str] | None:
    language = _choose_preferred_corpus_language(record)
    if not language:
        return None

    title_text = _choose_title_for_language(record, language)
    abstract_text = _choose_best_text(
        record["abstract_localized"],
        target_language=language,
        source_language=record["source_language"],
        min_words=MIN_ABSTRACT_WORDS,
    )
    if not title_text or not abstract_text:
        return None

    description_text = _choose_best_text(
        record["description_localized"],
        target_language=language,
        source_language=record["source_language"],
    )
    first_claim_text = _choose_best_text(
        record["first_claim_localized"],
        target_language=language,
        source_language=record["source_language"],
    )

    return {
        "id": f"{record['document_id'] or record['publication_number']}_{language}",
        "language": language,
        "title": title_text,
        "abstract": abstract_text,
        "description": _truncate_text(description_text, max_chars=DESCRIPTION_MAX_CHARS) if description_text else "",
        "first_claim": first_claim_text,
        "context": _build_context(record, title_text, language),
        "publication_number": record["publication_number"],
        "country_code": record["country_code"],
        "publication_date": record["publication_date"],
        "source": "epo",
        "document_id": record["document_id"],
        "application_number": record["application_number"],
        "filing_date": record["filing_date"],
        "kind": record["kind"],
        "source_language": record["source_language"],
        "available_title_languages": ",".join(
            item["language"] for item in record["title_localized"] if item["language"]
        ),
        "title_localized_json": json.dumps(record["title_localized"], ensure_ascii=False),
        "ipc_codes": "|".join(record["ipc_codes"]),
        "cpc_codes": "|".join(record["cpc_codes"]),
        "applicants": "|".join(record["applicants"]),
        "inventors": "|".join(record["inventors"]),
        "representatives": "|".join(record["representatives"]),
        "priority_numbers": "|".join(record["priority_numbers"]),
        "priority_dates": "|".join(record["priority_dates"]),
        "designated_states": "|".join(record["designated_states"]),
        "chemistry_label": record["chemistry"]["label"],
        "chemistry_score": str(record["chemistry"]["score"]),
        "chemistry_reasons": " | ".join(record["chemistry"]["reasons"]),
        "xml_file": record["xml_file"],
    }


def _process_epo_xml_file(xml_path: str) -> Dict[str, Any]:
    try:
        record = parse_epo_patent_xml(Path(xml_path))
    except (ET.ParseError, ValueError) as exc:
        return {
            "status": "error",
            "xml_file": Path(xml_path).name,
            "error": str(exc),
        }

    all_rows = _record_to_rows(record)
    corpus_row = _record_to_corpus_row(record) if record["chemistry"]["keep"] else None
    return {
        "status": "ok",
        "all_rows": all_rows,
        "corpus_row": corpus_row,
    }


def build_epo_corpus(
    xml_dir: Path,
    preprocessed_dir: Path,
    full_output_path: Path,
    output_path: Path,
    *,
    batch_mode: bool = False,
) -> Dict[str, int]:
    """
    Parse extracted EPO XMLs into a full parsed table and a chemistry-focused corpus.
    """
    xml_dir = Path(xml_dir)
    preprocessed_dir = Path(preprocessed_dir)
    full_output_path = Path(full_output_path)
    output_path = Path(output_path)

    if not xml_dir.exists():
        raise ValueError(f"EPO XML directory not found: {xml_dir}")

    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    full_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
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
        "document_id",
        "application_number",
        "filing_date",
        "kind",
        "source_language",
        "available_title_languages",
        "title_localized_json",
        "ipc_codes",
        "cpc_codes",
        "applicants",
        "inventors",
        "representatives",
        "priority_numbers",
        "priority_dates",
        "designated_states",
        "chemistry_label",
        "chemistry_score",
        "chemistry_reasons",
        "xml_file",
    ]

    all_rows: List[Dict[str, str]] = []
    corpus_rows: List[Dict[str, str]] = []
    stats = {
        "xml_files": 0,
        "candidate_xml_files": 0,
        "documents_parsed": 0,
        "documents_kept": 0,
        "all_rows": 0,
        "corpus_rows": 0,
        "parse_errors": 0,
        "skipped_auxiliary": 0,
        "workers": 1,
    }

    xml_paths = sorted(xml_dir.glob("*.xml"))
    patent_xml_paths: List[Path] = []
    for xml_path in xml_paths:
        stats["candidate_xml_files"] += 1
        if not xml_path.is_file() or AUXILIARY_XML_RE.search(xml_path.name):
            stats["skipped_auxiliary"] += 1
            continue
        patent_xml_paths.append(xml_path)

    if batch_mode and patent_xml_paths:
        available_cpus = os.cpu_count() or 1
        workers = max(1, min(len(patent_xml_paths), max(1, available_cpus // 2), 8))
        stats["workers"] = workers
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_process_epo_xml_file, str(xml_path))
                for xml_path in patent_xml_paths
            ]
            progress = tqdm(as_completed(futures), total=len(futures), desc="Build EPO corpus", unit="xml")
            for future in progress:
                result = future.result()
                if result["status"] != "ok":
                    stats["parse_errors"] += 1
                    tqdm.write(
                        f"  Skipping non-patent or malformed XML: {result['xml_file']}"
                    )
                    continue

                stats["xml_files"] += 1
                stats["documents_parsed"] += 1
                all_rows.extend(result["all_rows"])
                if result["corpus_row"]:
                    stats["documents_kept"] += 1
                    corpus_rows.append(result["corpus_row"])
    else:
        for xml_path in tqdm(patent_xml_paths, desc="Build EPO corpus", unit="xml"):
            try:
                record = parse_epo_patent_xml(xml_path)
            except (ET.ParseError, ValueError):
                stats["parse_errors"] += 1
                tqdm.write(f"  Skipping non-patent or malformed XML: {xml_path.name}")
                continue

            stats["xml_files"] += 1
            rows = _record_to_rows(record)
            stats["documents_parsed"] += 1
            all_rows.extend(rows)
            if record["chemistry"]["keep"]:
                kept_row = _record_to_corpus_row(record)
                if kept_row:
                    stats["documents_kept"] += 1
                    corpus_rows.append(kept_row)

    stats["all_rows"] = len(all_rows)
    stats["corpus_rows"] = len(corpus_rows)

    all_records_path = preprocessed_dir / "all_epo_records.csv"
    with all_records_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    with full_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(corpus_rows)

    mteb_fieldnames = ["_id", "title", "text"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=mteb_fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                "_id": row["id"],
                "title": row["title"],
                "text": row["context"],
            }
            for row in corpus_rows
        )

    return stats
