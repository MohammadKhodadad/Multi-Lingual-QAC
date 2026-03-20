from __future__ import annotations

import csv
import gzip
import json
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, unquote, urlencode
from urllib.request import Request, urlopen

from tqdm import tqdm

from src.multi_lingual_qac.constants import DEFAULT_LANGS, DEFAULT_WIKIDATA_ENTITY_TARGET

# Canonical chemistry section keys mapped to known titles across all 16 target languages.
# A section whose title (lowercased, stripped) matches any alias maps to that canonical key.
# "lead" is special: the article introduction before any section header.
CHEMISTRY_SECTION_TITLES: dict[str, list[str]] = {
    "lead": [],  # always the intro text before first ==
    "properties": [
        # en
        "properties", "physical properties", "chemical properties",
        "physicochemical properties", "physical and chemical properties",
        "characteristics", "structure and properties",
        # de
        "eigenschaften", "physikalische eigenschaften", "chemische eigenschaften",
        # fr
        "propriétés", "propriétés physiques", "propriétés chimiques",
        "caractéristiques",
        # es
        "propiedades", "propiedades físicas", "propiedades químicas",
        "características",
        # pt
        "propriedades", "propriedades físicas", "propriedades químicas",
        # it
        "proprietà", "proprietà fisiche", "proprietà chimiche",
        # nl
        "eigenschappen", "fysische eigenschappen", "chemische eigenschappen",
        # ru
        "свойства", "физические свойства", "химические свойства",
        # pl
        "właściwości", "właściwości fizyczne", "właściwości chemiczne",
        # tr
        "özellikler", "fiziksel özellikler", "kimyasal özellikler",
        # ar
        "خصائص", "الخصائص", "خصائص فيزيائية", "خصائص كيميائية",
        # fa
        "خواص", "ویژگی‌ها", "خواص فیزیکی", "خواص شیمیایی",
        # zh
        "性质", "物理性质", "化学性质", "特性",
        # ja
        "性質", "物性", "化学的性質", "物理的性質",
        # ko
        "성질", "물리적 성질", "화학적 성질", "특성",
        # hi
        "गुण", "भौतिक गुण", "रासायनिक गुण",
    ],
    "history": [
        "history", "historical background", "discovery",
        "geschichte", "entdeckung",
        "histoire", "découverte",
        "historia", "descubrimiento",
        "história", "descoberta",
        "storia", "scoperta",
        "geschiedenis", "ontdekking",
        "история", "открытие",
        "historia", "odkrycie",
        "tarih", "keşif",
        "تاريخ", "التاريخ", "الاكتشاف",
        "تاریخ", "کشف",
        "历史", "发现",
        "歴史", "発見",
        "역사", "발견",
        "इतिहास",
    ],
    "uses": [
        "uses", "applications", "use", "usage",
        "verwendung", "anwendung", "anwendungen",
        "utilisations", "applications", "usages",
        "usos", "aplicaciones",
        "usos", "aplicações",
        "usi", "applicazioni",
        "toepassingen", "gebruik",
        "применение", "использование",
        "zastosowanie", "zastosowania",
        "kullanım", "kullanım alanları",
        "استخدامات", "تطبيقات",
        "کاربردها", "استفاده",
        "用途", "应用",
        "用途", "利用",
        "용도", "사용",
        "उपयोग",
    ],
    "synthesis": [
        "synthesis", "production", "preparation", "manufacture",
        "synthese", "herstellung", "produktion",
        "synthèse", "préparation", "production",
        "síntesis", "producción", "preparación",
        "síntese", "produção",
        "sintesi", "produzione",
        "synthese", "productie",
        "синтез", "производство", "получение",
        "synteza", "produkcja",
        "sentez", "üretim",
        "تخليق", "إنتاج", "تحضير",
        "سنتز", "تولید",
        "合成", "制备", "生产",
        "合成", "製造",
        "합성", "제조",
        "संश्लेषण",
    ],
    "safety": [
        "safety", "hazards", "toxicity", "health effects", "safety and hazards",
        "sicherheit", "gefährdung", "toxizität",
        "sécurité", "dangers", "toxicité",
        "seguridad", "peligros", "toxicidad",
        "segurança", "perigos",
        "sicurezza", "pericoli",
        "veiligheid", "gevaren",
        "безопасность", "токсичность", "опасность",
        "bezpieczeństwo", "toksyczność",
        "güvenlik", "tehlikeler",
        "سلامة", "مخاطر", "سمية",
        "ایمنی", "خطرات",
        "安全", "危险", "毒性",
        "安全性", "毒性",
        "안전", "위험",
        "सुरक्षा",
    ],
    "structure": [
        "structure", "molecular structure", "crystal structure", "structure and bonding",
        "struktur", "molekülstruktur", "kristallstruktur",
        "structure", "structure moléculaire",
        "estructura", "estructura molecular",
        "estrutura", "estrutura molecular",
        "struttura",
        "structuur",
        "структура",
        "struktura",
        "yapı", "moleküler yapı",
        "بنية", "البنية",
        "ساختار",
        "结构", "分子结构",
        "構造",
        "구조",
        "संरचना",
    ],
    "occurrence": [
        "occurrence", "natural occurrence", "abundance", "sources",
        "vorkommen", "natürliches vorkommen",
        "occurrence", "présence naturelle",
        "ocurrencia", "fuentes naturales",
        "ocorrência",
        "occorrenza",
        "voorkomen",
        "нахождение в природе", "распространённость",
        "występowanie",
        "doğal kaynaklar",
        "الوجود", "المصادر الطبيعية",
        "وجود در طبیعت",
        "自然界", "存在",
        "存在",
        "존재",
        "प्राकृतिक स्रोत",
    ],
}

# Invert for fast lookup: alias -> canonical key
_SECTION_ALIAS_MAP: dict[str, str] = {
    alias.lower().strip(): canonical
    for canonical, aliases in CHEMISTRY_SECTION_TITLES.items()
    for alias in aliases
}


def _normalize_section_title(raw_title: str) -> str | None:
    """Return the canonical section key for a raw section title, or None if unknown."""
    return _SECTION_ALIAS_MAP.get(raw_title.lower().strip())


_WIKITEXT_STRIP_RE: list[tuple] = []


def _strip_wikitext(text: str) -> str:
    """Best-effort wikitext markup stripping for human-readable section content."""
    import re
    # Remove <ref>...</ref> blocks (including multiline)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<ref[^/]*/\s*>", "", text, flags=re.IGNORECASE)
    # Remove {{templates}} — nested templates need multiple passes
    for _ in range(4):
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    # [[File:...]] / [[Image:...]] — remove entirely
    text = re.sub(r"\[\[(?:File|Image|Datei|Fichier|Archivo|Ficheiro|Bestand):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    # [[Link|Label]] → Label,  [[Link]] → Link
    text = re.sub(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]", r"\1", text)
    # External links [url label] → label, or drop bare [url]
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)
    # Bold/italic markup
    text = re.sub(r"'{2,3}", "", text)
    # HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # HTML entities
    text = re.sub(r"&[a-z]+;", " ", text)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _parse_sections(wikitext: str) -> list[dict[str, Any]]:
    """
    Split Wikipedia wikitext into sections using `== Title ==` header markers.
    Returns a list of dicts:

        [{"title_raw": str, "title_normalized": str | None, "level": int, "content": str}]

    The lead (introduction before the first header) is entry 0 with
    title_raw="" and title_normalized="lead".

    Content is stripped of the most common wikitext markup so it is
    human-readable without a full wikitext parser.
    """
    import re
    sections: list[dict[str, Any]] = []
    header_re = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)

    last_end = 0
    last_title_raw = ""
    last_title_normalized: str | None = "lead"
    last_level = 1

    for match in header_re.finditer(wikitext):
        raw_content = wikitext[last_end:match.start()]
        content = _strip_wikitext(raw_content)
        if content:
            sections.append({
                "title_raw": last_title_raw,
                "title_normalized": last_title_normalized,
                "level": last_level,
                "content": content,
            })
        last_title_raw = match.group(2)
        last_title_normalized = _normalize_section_title(match.group(2))
        last_level = len(match.group(1))
        last_end = match.end()

    trailing = _strip_wikitext(wikitext[last_end:])
    if trailing:
        sections.append({
            "title_raw": last_title_raw,
            "title_normalized": last_title_normalized,
            "level": last_level,
            "content": trailing,
        })

    return sections


SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIPEDIA_API_TEMPLATE = "https://{lang}.wikipedia.org/w/api.php"
HTTP_TIMEOUT_SECONDS = 120
MAX_HTTP_RETRIES = 5
SPARQL_PAGE_SIZE = 500
SPARQL_MIN_SITELINKS = 5
SITELINK_BATCH_SIZE = 200
WIKIPEDIA_BATCH_SIZE = 10

# These roots intentionally blend core chemistry with chemistry-adjacent concepts
# so the acquisition stage preserves enough material for later QA-oriented filtering.
CHEMISTRY_SEED_CLASSES = [
    {"qid": "Q11173", "label": "chemical compound"},
    {"qid": "Q11344", "label": "chemical element"},
    {"qid": "Q81869", "label": "chemical reaction"},
    {"qid": "Q12140", "label": "medication"},
    {"qid": "Q159226", "label": "catalyst"},
    {"qid": "Q146505", "label": "solvent"},
    {"qid": "Q811430", "label": "polymer"},
]

USER_AGENT = "multi-lingual-qac/0.1 (wikidata-wikipedia-acquisition)"


def _batched(items: Iterable[Any], batch_size: int) -> list[list[Any]]:
    batch: list[Any] = []
    batches: list[list[Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches


def _ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _http_get_json(url: str, *, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request_headers = {
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    if headers:
        request_headers.update(headers)

    last_error: Exception | None = None
    for attempt in range(1, MAX_HTTP_RETRIES + 1):
        try:
            request = Request(url, headers=request_headers)
            with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network variability
            last_error = exc
            if attempt == MAX_HTTP_RETRIES:
                break
            wait = 2 ** attempt
            tqdm.write(f"  [retry {attempt}/{MAX_HTTP_RETRIES - 1}] {type(exc).__name__}: {exc}. Waiting {wait}s...")
            time.sleep(wait)
    assert last_error is not None
    raise last_error


def _run_sparql(query: str) -> list[dict[str, str]]:
    params = urlencode({"format": "json", "query": query})
    url = f"{SPARQL_ENDPOINT}?{params}"
    payload = _http_get_json(
        url,
        headers={
            "Accept": "application/sparql-results+json",
        },
    )
    return payload.get("results", {}).get("bindings", [])


def _extract_binding_value(binding: dict[str, Any], key: str) -> str:
    return binding.get(key, {}).get("value", "")


def _qid_from_uri(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def _candidate_query(seed_qid: str, *, limit: int, offset: int) -> str:
    # Use only P31 (instance of) without the expensive P279* (subclass chain)
    # to avoid WDQS timeouts. We accept slight recall loss in exchange for
    # reliable, fast queries.
    return f"""
SELECT DISTINCT ?item ?itemLabel ?itemDescription ?sitelinks
WHERE {{
  ?item wdt:P31 wd:{seed_qid} .
  ?item wikibase:sitelinks ?sitelinks .
  FILTER(?sitelinks >= {SPARQL_MIN_SITELINKS})
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
ORDER BY DESC(?sitelinks) ?item
LIMIT {limit}
OFFSET {offset}
""".strip()


def _sitelink_query(qids: list[str], languages: tuple[str, ...]) -> str:
    values = " ".join(f"wd:{qid}" for qid in qids)
    lang_values = " ".join(f'"{lang}"' for lang in languages)
    return f"""
PREFIX schema: <http://schema.org/>

SELECT ?item ?lang ?article
WHERE {{
  VALUES ?item {{ {values} }}
  VALUES ?lang {{ {lang_values} }}
  ?article schema:about ?item ;
           schema:inLanguage ?lang ;
           schema:isPartOf ?site .
  FILTER(CONTAINS(STR(?site), "wikipedia.org"))
}}
ORDER BY ?item ?lang
""".strip()


def _decode_article_title(article_url: str) -> str:
    encoded_title = article_url.rsplit("/wiki/", 1)[-1]
    return unquote(encoded_title).replace("_", " ")


def _discover_candidates(
    *,
    target_entities: int,
) -> dict[str, dict[str, Any]]:
    candidate_target = max(target_entities * 3, 30_000)
    per_seed_target = max(candidate_target // max(len(CHEMISTRY_SEED_CLASSES), 1), SPARQL_PAGE_SIZE)
    candidates: dict[str, dict[str, Any]] = {}

    seed_bar = tqdm(CHEMISTRY_SEED_CLASSES, desc="Discover seed classes", unit="class")
    for seed in seed_bar:
        seed_bar.set_postfix(seed=seed["label"], entities=len(candidates))
        collected_for_seed = 0
        offset = 0

        page_bar = tqdm(
            desc=f"  {seed['label']} pages",
            unit="page",
            leave=False,
        )
        while collected_for_seed < per_seed_target and len(candidates) < candidate_target:
            rows = _run_sparql(_candidate_query(seed["qid"], limit=SPARQL_PAGE_SIZE, offset=offset))
            if not rows:
                break

            for row in rows:
                qid = _qid_from_uri(_extract_binding_value(row, "item"))
                candidate = candidates.setdefault(
                    qid,
                    {
                        "qid": qid,
                        "label": _extract_binding_value(row, "itemLabel"),
                        "description": _extract_binding_value(row, "itemDescription"),
                        "sitelinks": int(_extract_binding_value(row, "sitelinks") or 0),
                        "seed_classes": set(),
                    },
                )
                candidate["sitelinks"] = max(
                    candidate["sitelinks"],
                    int(_extract_binding_value(row, "sitelinks") or 0),
                )
                if not candidate["label"]:
                    candidate["label"] = _extract_binding_value(row, "itemLabel")
                if not candidate["description"]:
                    candidate["description"] = _extract_binding_value(row, "itemDescription")
                candidate["seed_classes"].add(seed["label"])
                collected_for_seed += 1

            page_bar.update(len(rows))
            page_bar.set_postfix(unique=len(candidates))
            offset += SPARQL_PAGE_SIZE
            time.sleep(0.1)

            if len(rows) < SPARQL_PAGE_SIZE:
                break

        page_bar.close()
        seed_bar.set_postfix(seed=seed["label"], entities=len(candidates))

    for candidate in candidates.values():
        candidate["seed_classes"] = sorted(candidate["seed_classes"])

    tqdm.write(f"Discovered {len(candidates)} unique candidate entities.")
    return candidates


def _attach_target_language_sitelinks(
    candidates: dict[str, dict[str, Any]],
    *,
    languages: tuple[str, ...],
) -> None:
    qids = sorted(candidates)
    for batch in tqdm(_batched(qids, SITELINK_BATCH_SIZE), desc="Resolve Wikidata sitelinks"):
        rows = _run_sparql(_sitelink_query(batch, languages))
        for row in rows:
            qid = _qid_from_uri(_extract_binding_value(row, "item"))
            lang = _extract_binding_value(row, "lang")
            article_url = _extract_binding_value(row, "article")
            if not qid or not lang or not article_url:
                continue
            candidates[qid].setdefault("language_pages", {})[lang] = {
                "title": _decode_article_title(article_url),
                "url": article_url,
            }
        time.sleep(0.1)

    for candidate in candidates.values():
        candidate["language_pages"] = candidate.get("language_pages", {})
        candidate["available_languages"] = sorted(candidate["language_pages"])
        candidate["target_language_count"] = len(candidate["available_languages"])
        candidate["has_fa"] = "fa" in candidate["language_pages"]


def _select_entities(
    candidates: dict[str, dict[str, Any]],
    *,
    target_entities: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        (candidate for candidate in candidates.values() if candidate["target_language_count"] > 0),
        key=lambda candidate: (
            candidate["target_language_count"],
            1 if candidate["has_fa"] else 0,
            candidate["sitelinks"],
            candidate["qid"],
        ),
        reverse=True,
    )
    return ranked[:target_entities]


def _coverage_report(
    selected_entities: list[dict[str, Any]],
    *,
    languages: tuple[str, ...],
    candidate_count: int,
    target_entities: int,
) -> dict[str, Any]:
    pages_per_language = {lang: 0 for lang in languages}
    coverage_distribution = Counter()
    for entity in selected_entities:
        coverage_distribution[entity["target_language_count"]] += 1
        for lang in entity["available_languages"]:
            pages_per_language[lang] += 1

    return {
        "target_entities": target_entities,
        "candidate_entities": candidate_count,
        "selected_entities": len(selected_entities),
        "pages_per_language": pages_per_language,
        "coverage_distribution": dict(sorted(coverage_distribution.items(), reverse=True)),
        "entities_with_fa": sum(1 for entity in selected_entities if entity["has_fa"]),
        "languages": list(languages),
    }


def _write_entities_csv(path: Path, selected_entities: list[dict[str, Any]], languages: tuple[str, ...]) -> None:
    fieldnames = [
        "qid",
        "label",
        "description",
        "seed_classes",
        "sitelinks",
        "target_language_count",
        "available_languages",
        "has_fa",
    ] + [f"has_{lang}" for lang in languages]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for entity in selected_entities:
            row = {
                "qid": entity["qid"],
                "label": entity["label"],
                "description": entity["description"],
                "seed_classes": "|".join(entity["seed_classes"]),
                "sitelinks": entity["sitelinks"],
                "target_language_count": entity["target_language_count"],
                "available_languages": "|".join(entity["available_languages"]),
                "has_fa": entity["has_fa"],
            }
            for lang in languages:
                row[f"has_{lang}"] = lang in entity["language_pages"]
            writer.writerow(row)


def _extract_revision_content(page: dict[str, Any]) -> str:
    revisions = page.get("revisions") or []
    if not revisions:
        return ""
    revision = revisions[0]
    slots = revision.get("slots") or {}
    main_slot = slots.get("main") or {}
    return (
        main_slot.get("content")
        or main_slot.get("*")
        or revision.get("content")
        or revision.get("*")
        or ""
    )


def _wikipedia_api_query(lang: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"{WIKIPEDIA_API_TEMPLATE.format(lang=lang)}?{urlencode(params, doseq=True)}"
    return _http_get_json(url)


def _fetch_pages_for_language(
    lang: str,
    entities: list[dict[str, Any]],
    *,
    output_path: Path,
) -> dict[str, Any]:
    missing_pages = 0
    pages_written = 0
    optional_field_counts: dict[str, int] = {
        "has_extract": 0,
        "has_categories": 0,
        "has_wikitext": 0,
        "has_sections": 0,
    }

    records_by_title = {
        entity["language_pages"][lang]["title"]: entity
        for entity in entities
        if lang in entity["language_pages"]
    }
    requested_titles = list(records_by_title)

    with gzip.open(output_path, "wt", encoding="utf-8") as fh:
        for batch in tqdm(_batched(requested_titles, WIKIPEDIA_BATCH_SIZE), desc=f"Fetch {lang} pages"):
            response = _wikipedia_api_query(
                lang,
                {
                    "action": "query",
                    "format": "json",
                    "formatversion": "2",
                    "redirects": "1",
                    "titles": "|".join(batch),
                    "prop": "info|extracts|revisions|categories",
                    "inprop": "url",
                    "explaintext": "1",
                    "exsectionformat": "plain",
                    "rvslots": "main",
                    "rvprop": "ids|timestamp|content",
                    "cllimit": "max",
                },
            )

            query = response.get("query", {})
            normalized_map = {
                item.get("from", ""): item.get("to", "")
                for item in query.get("normalized", [])
            }
            redirect_map = {
                item.get("from", ""): item.get("to", "")
                for item in query.get("redirects", [])
            }
            pages_by_title = {
                page.get("title", ""): page
                for page in query.get("pages", [])
                if "missing" not in page
            }

            for requested_title in batch:
                normalized_title = normalized_map.get(requested_title, requested_title)
                final_title = redirect_map.get(normalized_title, normalized_title)
                page = pages_by_title.get(final_title) or pages_by_title.get(requested_title)
                entity = records_by_title[requested_title]
                if not page:
                    missing_pages += 1
                    continue

                revision = (page.get("revisions") or [{}])[0]
                raw_extract = page.get("extract", "")
                categories = [
                    category.get("title", "").removeprefix("Category:")
                    for category in page.get("categories", [])
                ]
                raw_wikitext = _extract_revision_content(page)
                sections = _parse_sections(raw_wikitext) if raw_wikitext else []

                has_extract = bool(raw_extract)
                has_categories = bool(categories)
                has_wikitext = bool(raw_wikitext)
                has_sections = bool(sections)

                if has_extract:
                    optional_field_counts["has_extract"] += 1
                if has_categories:
                    optional_field_counts["has_categories"] += 1
                if has_wikitext:
                    optional_field_counts["has_wikitext"] += 1
                if has_sections:
                    optional_field_counts["has_sections"] += 1

                record = {
                    "qid": entity["qid"],
                    "lang": lang,
                    "wiki_title": page.get("title", requested_title),
                    "wiki_url": page.get("fullurl") or entity["language_pages"][lang]["url"],
                    "pageid": page.get("pageid"),
                    "revision_id": revision.get("revid"),
                    "revision_timestamp": revision.get("timestamp"),
                    "wikidata_label": entity["label"],
                    "wikidata_description": entity["description"],
                    "seed_classes": entity["seed_classes"],
                    "raw_extract": raw_extract,
                    "has_extract": has_extract,
                    "sections": sections,
                    "has_sections": has_sections,
                    "categories": categories,
                    "has_categories": has_categories,
                    "raw_wikitext": raw_wikitext,
                    "has_wikitext": has_wikitext,
                    "raw_page": page,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                pages_written += 1

            time.sleep(0.05)

    return {
        "pages_written": pages_written,
        "missing_pages": missing_pages,
        "optional_field_counts": optional_field_counts,
    }


def prepare_wikidata_source(
    *,
    prepared_dir: Path,
    raw_pages_dir: Path,
    languages: tuple[str, ...] = tuple(DEFAULT_LANGS),
    target_entities: int = DEFAULT_WIKIDATA_ENTITY_TARGET,
    overwrite: bool = False,
) -> dict[str, int]:
    if overwrite:
        _ensure_empty_dir(prepared_dir)
    else:
        prepared_dir.mkdir(parents=True, exist_ok=True)
    raw_pages_dir.mkdir(parents=True, exist_ok=True)

    tqdm.write(f"[1/4] Discovering chemistry entity candidates (target pool: {target_entities * 3:,})...")
    candidates = _discover_candidates(target_entities=target_entities)
    tqdm.write(f"[2/4] Resolving sitelinks for {len(candidates):,} candidates across {len(languages)} languages...")
    _attach_target_language_sitelinks(candidates, languages=languages)
    tqdm.write(f"[3/4] Selecting top {target_entities:,} entities by cross-language coverage...")
    selected_entities = _select_entities(candidates, target_entities=target_entities)
    tqdm.write(f"      Selected {len(selected_entities):,} entities. Building coverage report...")
    coverage = _coverage_report(
        selected_entities,
        languages=languages,
        candidate_count=len(candidates),
        target_entities=target_entities,
    )

    _write_entities_csv(prepared_dir / "entities.csv", selected_entities, languages)
    tqdm.write(f"[4/4] Fetching Wikipedia pages for {len(selected_entities):,} entities...")

    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in selected_entities:
        for lang in entity["available_languages"]:
            by_language[lang].append(entity)

    pages_fetched = 0
    missing_pages = 0
    optional_fields_by_language: dict[str, dict[str, int]] = {}
    active_languages = [lang for lang in languages if by_language.get(lang)]
    lang_bar = tqdm(active_languages, desc="Fetch Wikipedia pages", unit="lang")
    for lang in lang_bar:
        lang_entities = by_language.get(lang, [])
        lang_bar.set_postfix(lang=lang, entities=len(lang_entities))
        lang_stats = _fetch_pages_for_language(
            lang,
            lang_entities,
            output_path=raw_pages_dir / f"{lang}.jsonl.gz",
        )
        pages_fetched += lang_stats["pages_written"]
        missing_pages += lang_stats["missing_pages"]
        optional_fields_by_language[lang] = {
            "pages_fetched": lang_stats["pages_written"],
            **lang_stats["optional_field_counts"],
        }
        lang_bar.set_postfix(lang=lang, fetched=pages_fetched, missing=missing_pages)

    coverage["optional_fields_by_language"] = optional_fields_by_language

    prepare_stats = {
        "candidate_entities": len(candidates),
        "selected_entities": len(selected_entities),
        "target_entities": target_entities,
        "pages_fetched": pages_fetched,
        "missing_pages": missing_pages,
        "languages_with_pages": sum(1 for count in coverage["pages_per_language"].values() if count > 0),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    (prepared_dir / "coverage_report.json").write_text(
        json.dumps(coverage, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (prepared_dir / "prepare_stats.json").write_text(
        json.dumps(prepare_stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "candidate_entities": len(candidates),
        "selected_entities": len(selected_entities),
        "pages_fetched": pages_fetched,
        "missing_pages": missing_pages,
        "languages_with_pages": prepare_stats["languages_with_pages"],
    }


def count_wikidata_prepared_records(prepared_dir: Path) -> int:
    stats_path = prepared_dir / "prepare_stats.json"
    if not stats_path.exists():
        return 0
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return int(stats.get("pages_fetched", 0))


__all__ = [
    "prepare_wikidata_source",
    "count_wikidata_prepared_records",
]
