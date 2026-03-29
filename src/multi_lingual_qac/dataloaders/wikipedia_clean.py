"""
Multilingual Wikipedia plaintext cleanup and chunking for Wikidata-derived extracts.

MediaWiki `extract` is already plain text; this removes typical editorial noise,
drops low-value paragraphs, and chunks with sentence-aware splitting where possible.
Chemistry-like tokens (formulas, units, stereo labels) are treated conservatively.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Set

from src.multi_lingual_qac.constants import DEFAULT_LANGS

# Vocab is defined for each language in DEFAULT_LANGS (en, de, fr, es, ja, ko, zh, ru, pt, it, nl, ar, fa, tr, pl, hi).

# -----------------------------------------------------------------------------
# Language-specific vocab (BAD sections: native only + en fallback in _lang_vocab)
# -----------------------------------------------------------------------------

BAD_SECTION_TITLES: Dict[str, Set[str]] = {
    "en": {
        "see also",
        "references",
        "notes",
        "further reading",
        "external links",
        "bibliography",
        "sources",
        "citations",
    },
    "fr": {
        "voir aussi",
        "références",
        "notes",
        "liens externes",
        "bibliographie",
        "sources",
    },
    "de": {
        "siehe auch",
        "einzelnachweise",
        "literatur",
        "Weblinks",
        "weblinks",
        "anmerkungen",
        "quellen",
    },
    "es": {
        "véase también",
        "referencias",
        "notas",
        "enlaces externos",
        "bibliografía",
        "fuentes",
    },
    "it": {
        "vedi anche",
        "riferimenti",
        "note",
        "collegamenti esterni",
        "bibliografia",
        "fonti",
    },
    "pt": {
        "ver também",
        "referências",
        "notas",
        "ligações externas",
        "bibliografia",
        "fontes",
    },
    "nl": {
        "zie ook",
        "referenties",
        "noten",
        "externe links",
        "bibliografie",
        "bronnen",
    },
    "pl": {
        "zobacz też",
        "bibliografia",
        "linki zewnętrzne",
        "przypisy",
        "uwagi",
    },
    "ru": {
        "см. также",
        "примечания",
        "литература",
        "ссылки",
        "источники",
    },
    "ja": {
        "関連項目",
        "脚注",
        "参考文献",
        "外部リンク",
        "出典",
        "さらに読む",
    },
    "ko": {
        "같이 보기",
        "각주",
        "참고 문헌",
        "참고문헌",
        "외부 링크",
        "바깥 고리",
        "문헌",
    },
    "zh": {
        "参见",
        "另见",
        "参考文献",
        "参考资料",
        "外部链接",
        "外部連結",
        "注释",
        "註釋",
        "延伸阅读",
        "延伸閱讀",
    },
    "ar": {
        "انظر أيضًا",
        "انظر ايضا",
        "مراجع",
        "هوامش",
        "وصلات خارجية",
        "ملاحظات",
        "المصادر",
    },
    "fa": {
        "منابع",
        "پانویس",
        "پیوند به بیرون",
        "یادداشت",
        "پانویس‌ها",
        "همچنین ببینید",
    },
    "tr": {
        "ayrıca bakınız",
        "kaynakça",
        "dış bağlantılar",
        "notlar",
        "konu dışı",
        "bibliyografya",
    },
    "hi": {
        "सन्दर्भ",
        "संदर्भ",
        "टिप्पणियाँ",
        "टिप्पणी",
        "बाहरी कड़ियाँ",
        "बाहरी स्रोत",
        "इन्हें भी देखें",
        "सम्बद्ध पठन",
    },
}

USEFUL_CHEM_SECTION_HINTS: Dict[str, Set[str]] = {
    "en": {
        "properties",
        "structure",
        "synthesis",
        "reactions",
        "uses",
        "occurrence",
        "production",
        "history",
        "mechanism",
        "safety",
        "toxicity",
        "applications",
    },
    "fr": {
        "propriétés",
        "structure",
        "synthèse",
        "réactions",
        "utilisations",
        "occurrence",
        "production",
        "histoire",
        "mécanisme",
        "sécurité",
        "toxicité",
    },
    "de": {
        "eigenschaften",
        "struktur",
        "synthese",
        "reaktionen",
        "verwendung",
        "vorkommen",
        "herstellung",
        "geschichte",
        "mechanismus",
        "sicherheit",
        "toxizität",
    },
    "es": {
        "propiedades",
        "estructura",
        "síntesis",
        "reacciones",
        "usos",
        "ocurrencia",
        "producción",
        "historia",
        "mecanismo",
        "seguridad",
        "toxicidad",
    },
    "it": {
        "proprietà",
        "struttura",
        "sintesi",
        "reazioni",
        "usi",
        "occorrenza",
        "produzione",
        "storia",
        "meccanismo",
        "sicurezza",
        "tossicità",
    },
    "pt": {
        "propriedades",
        "estrutura",
        "síntese",
        "reações",
        "usos",
        "ocorrência",
        "produção",
        "história",
        "mecanismo",
        "segurança",
        "toxicidade",
    },
    "nl": {
        "eigenschappen",
        "structuur",
        "synthese",
        "reacties",
        "gebruik",
        "productie",
        "geschiedenis",
        "mechanisme",
        "veiligheid",
        "toxiciteit",
    },
    "pl": {
        "właściwości",
        "struktura",
        "synteza",
        "reakcje",
        "zastosowania",
        "występowanie",
        "produkcja",
        "historia",
        "mechanizm",
        "bezpieczeństwo",
        "toksyczność",
    },
    "ru": {
        "свойства",
        "структура",
        "синтез",
        "реакции",
        "применение",
        "получение",
        "история",
        "механизм",
        "безопасность",
        "токсичность",
    },
    "ja": {
        "性質",
        "物理化学的性質",
        "構造",
        "合成",
        "反応",
        "用途",
        "製法",
        "歴史",
        "機構",
        "安全性",
        "毒性",
    },
    "ko": {
        "성질",
        "구조",
        "합성",
        "반응",
        "용도",
        "생산",
        "역사",
        "기전",
        "안전성",
        "독성",
    },
    "zh": {
        "性质",
        "性質",
        "结构",
        "結構",
        "合成",
        "反应",
        "反應",
        "用途",
        "制备",
        "製備",
        "历史",
        "歷史",
        "机理",
        "機理",
        "安全",
        "毒性",
    },
    "ar": {
        "الخصائص",
        "البنية",
        "التركيب",
        "التفاعلات",
        "الاستخدامات",
        "الإنتاج",
        "التاريخ",
        "آلية",
        "السلامة",
        "السمية",
    },
    "fa": {
        "خواص",
        "ساختار",
        "سنتز",
        "واکنش",
        "کاربرد",
        "تولید",
        "تاریخ",
        "مکانیسم",
        "ایمنی",
        "سمیت",
    },
    "tr": {
        "özellikler",
        "yapı",
        "sentez",
        "tepkimeler",
        "kullanım",
        "üretim",
        "tarih",
        "mekanizma",
        "güvenlik",
        "toksisite",
    },
    "hi": {
        "गुण",
        "संरचना",
        "संश्लेषण",
        "अभिक्रियाएँ",
        "अभिक्रिया",
        "उपयोग",
        "उत्पादन",
        "इतिहास",
        "तंत्र",
        "सुरक्षा",
        "विषाक्तता",
    },
}

CHEMISTRY_KEYWORDS: Dict[str, Set[str]] = {
    "en": {
        "acid",
        "base",
        "compound",
        "molecule",
        "reaction",
        "synthesis",
        "catalyst",
        "solvent",
        "element",
        "ion",
        "salt",
        "organic",
        "inorganic",
        "polymer",
        "alkane",
        "alkene",
        "aromatic",
        "ketone",
        "aldehyde",
        "ester",
        "amine",
        "protein",
        "enzyme",
        "metal",
        "oxidation",
        "reduction",
        "ph",
        "molar",
    },
    "fr": {
        "acide",
        "base",
        "composé",
        "molécule",
        "réaction",
        "synthèse",
        "catalyseur",
        "solvant",
        "élément",
        "ion",
        "sel",
        "organique",
        "inorganique",
        "polymère",
        "protéine",
        "enzyme",
        "métal",
        "oxydation",
        "réduction",
        "ph",
        "molaire",
    },
    "de": {
        "säure",
        "base",
        "verbindung",
        "molekül",
        "reaktion",
        "synthese",
        "katalysator",
        "lösungsmittel",
        "element",
        "ion",
        "salz",
        "organisch",
        "anorganisch",
        "polymer",
        "protein",
        "enzym",
        "metall",
        "oxidation",
        "reduktion",
        "ph",
        "molar",
    },
    "es": {
        "ácido",
        "base",
        "compuesto",
        "molécula",
        "reacción",
        "síntesis",
        "catalizador",
        "solvente",
        "elemento",
        "ión",
        "sal",
        "orgánico",
        "inorgánico",
        "polímero",
        "proteína",
        "enzima",
        "metal",
        "oxidación",
        "reducción",
        "ph",
        "molar",
    },
    "it": {
        "acido",
        "base",
        "composto",
        "molecola",
        "reazione",
        "sintesi",
        "catalizzatore",
        "solvente",
        "elemento",
        "ione",
        "sale",
        "organico",
        "inorganico",
        "polimero",
        "proteina",
        "enzima",
        "metallo",
        "ossidazione",
        "riduzione",
        "ph",
        "molare",
    },
    "pt": {
        "ácido",
        "base",
        "composto",
        "molécula",
        "reação",
        "síntese",
        "catalisador",
        "solvente",
        "elemento",
        "íon",
        "sal",
        "orgânico",
        "inorgânico",
        "polímero",
        "proteína",
        "enzima",
        "metal",
        "oxidação",
        "redução",
        "ph",
        "molar",
    },
    "nl": {
        "zuur",
        "base",
        "verbinding",
        "molecuul",
        "reactie",
        "synthese",
        "katalysator",
        "oplosmiddel",
        "element",
        "ion",
        "zout",
        "organisch",
        "anorganisch",
        "polymeer",
        "eiwit",
        "enzym",
        "metaal",
        "oxidatie",
        "reductie",
        "ph",
        "molaire",
    },
    "pl": {
        "kwas",
        "zasada",
        "związek",
        "cząsteczka",
        "reakcja",
        "synteza",
        "katalizator",
        "rozpuszczalnik",
        "pierwiastek",
        "jon",
        "sól",
        "organiczny",
        "nieorganiczny",
        "polimer",
        "białko",
        "enzym",
        "metal",
        "utlenianie",
        "redukcja",
        "ph",
        "molowy",
    },
    "ru": {
        "кислота",
        "основание",
        "соединение",
        "молекула",
        "реакция",
        "синтез",
        "катализатор",
        "растворитель",
        "элемент",
        "ион",
        "соль",
        "органический",
        "неорганический",
        "полимер",
        "белок",
        "фермент",
        "металл",
        "окисление",
        "восстановление",
        "редукция",
        "ph",
        "молярн",
    },
    "ja": {
        "酸",
        "塩基",
        "反応",
        "分子",
        "イオン",
        "触媒",
        "溶媒",
        "酸化",
        "還元",
        "有機",
        "無機",
        "重合",
        "タンパク質",
        "酵素",
        "金属",
        "化合物",
        "合成",
    },
    "ko": {
        "산",
        "염기",
        "반응",
        "분자",
        "이온",
        "촉매",
        "용매",
        "산화",
        "환원",
        "유기",
        "무기",
        "중합",
        "단백질",
        "효소",
        "금속",
        "화합물",
        "합성",
    },
    "zh": {
        "酸",
        "碱",
        "盐",
        "反应",
        "反應",
        "分子",
        "离子",
        "離子",
        "催化剂",
        "催化劑",
        "溶剂",
        "溶劑",
        "氧化",
        "还原",
        "還原",
        "有机",
        "有機",
        "无机",
        "無機",
        "聚合",
        "蛋白",
        "酶",
        "金属",
        "金屬",
        "化合物",
    },
    "ar": {
        "حمض",
        "قاعدة",
        "تفاعل",
        "جزيء",
        "أيون",
        "محفز",
        "مذيب",
        "أكسدة",
        "اختزال",
        "عضوي",
        "غير عضوي",
        "بوليمر",
        "بروتين",
        "إنزيم",
        "فلز",
        "مركب",
        "ph",
    },
    "fa": {
        "اسید",
        "باز",
        "واکنش",
        "مولکول",
        "یون",
        "کاتالیزور",
        "حلال",
        "اکسایش",
        "کاهش",
        "آلی",
        "غیر آلی",
        "پلیمر",
        "پروتئین",
        "آنزیم",
        "فلز",
        "ترکیب",
        "ph",
    },
    "tr": {
        "asit",
        "baz",
        "tepkime",
        "molekül",
        "iyon",
        "katalizör",
        "çözücü",
        "oksidasyon",
        "indirgenme",
        "organik",
        "anorganik",
        "polimer",
        "protein",
        "enzim",
        "metal",
        "bileşik",
        "ph",
    },
    "hi": {
        "अम्ल",
        "क्षार",
        "अभिक्रिया",
        "अणु",
        "आयन",
        "उत्प्रेरक",
        "विलायक",
        "ऑक्सीकरण",
        "अपचयन",
        "कार्बनिक",
        "अकार्बनिक",
        "बहुलक",
        "प्रोटीन",
        "एंजाइम",
        "धातु",
        "यौगिक",
        "ph",
    },
}

# Case-fold section titles / keywords (Latin scripts); CJK/Arabic largely unchanged.
for _k, _v in list(BAD_SECTION_TITLES.items()):
    BAD_SECTION_TITLES[_k] = {x.lower() for x in _v}
for _k, _v in list(USEFUL_CHEM_SECTION_HINTS.items()):
    USEFUL_CHEM_SECTION_HINTS[_k] = {x.lower() for x in _v}
for _k, _v in list(CHEMISTRY_KEYWORDS.items()):
    CHEMISTRY_KEYWORDS[_k] = {x.lower() for x in _v}

for _table, _label in (
    (BAD_SECTION_TITLES, "BAD_SECTION_TITLES"),
    (USEFUL_CHEM_SECTION_HINTS, "USEFUL_CHEM_SECTION_HINTS"),
    (CHEMISTRY_KEYWORDS, "CHEMISTRY_KEYWORDS"),
):
    _missing = [c for c in DEFAULT_LANGS if c not in _table]
    if _missing:
        raise ValueError(f"wikipedia_clean.{_label} missing DEFAULT_LANGS: {_missing}")

# -----------------------------------------------------------------------------
# Regexes (conservative for chemistry notation)
# -----------------------------------------------------------------------------

RE_MULTISPACE = re.compile(r"[ \t]+")
RE_MULTI_NEWLINES = re.compile(r"\n{3,}")
RE_CITATION_NUM = re.compile(r"\[(?:\d{1,3}|[a-z])\]")
RE_CITATION_NEEDED = re.compile(
    r"\[(?:citation needed|clarification needed|dubious|who\?|when\?|where\?|according to whom\?)\]",
    re.IGNORECASE,
)
RE_EMPTY_PARENS = re.compile(r"\(\s*\)")
RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,;:.!?])")

_DISAMBIG_PATTERNS: List[re.Pattern[str]] = [
    re.compile(
        r"\b(?:may refer to|can refer to|this disambiguation page lists)\b",
        re.IGNORECASE,
    ),
    re.compile(r"曖昧さ回避"),
    re.compile(r"消歧义|消歧義"),
    re.compile(r"同类索引"),
    re.compile(r"동음이의어|모호한 표기"),
    re.compile(r"Begriffsklärung", re.IGNORECASE),
    re.compile(r"desambiguaci[oó]n", re.IGNORECASE),
    re.compile(r"homonymie", re.IGNORECASE),
    re.compile(r"disambigua", re.IGNORECASE),
    re.compile(r"صفحة توضيح|تداخل لغوي"),
    re.compile(r"ابهام‌زدایی|صفحهٔ ابهام‌زدایی"),
    re.compile(r"این یک صفحهٔ ابهام‌زدایی است"),
    re.compile(r"यह एक द्विअर्थी शब्द का", re.IGNORECASE),
]

RE_CHEM_FORMULA = re.compile(r"\b(?:[A-Z][a-z]?\d{0,3}){2,}(?:[+-]\d*)?\b")
RE_COMPLEX_ION = re.compile(r"\[[A-Za-z0-9()+\-.,\s]{2,}\](?:\d*[+-])?")
RE_STEREO = re.compile(r"\((?:R|S|E|Z|D|L|±)\)")
RE_CHEM_NAME_NUM = re.compile(
    r"\b\d+(?:,\d+)*-(?:[A-Za-zÀ-ÿα-ωΑ-Ω][A-Za-zÀ-ÿα-ωΑ-Ω\-]*)\b"
)
RE_UNITS = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?(?:°C|K|atm|bar|Pa|kPa|MPa|mol|mmol|μmol|M|mM|μM|nm|µm|mg|g|kg|mL|L|wt%|%)\b",
    re.IGNORECASE,
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？؛])\s+")


def _text_looks_disambig(text: str) -> bool:
    return any(rx.search(text) for rx in _DISAMBIG_PATTERNS)


def _lang_vocab(mapping: Dict[str, Set[str]], lang: Optional[str]) -> Set[str]:
    """Return language-specific set, falling back to English, then empty."""
    code = (lang or "en").lower()
    if code in mapping:
        return mapping[code]
    return mapping.get("en", set())


def _chem_vocab(lang: Optional[str]) -> Set[str]:
    """Local chemistry keywords plus English (loanwords / Latin in non-English wikis)."""
    return _lang_vocab(CHEMISTRY_KEYWORDS, lang) | CHEMISTRY_KEYWORDS.get("en", set())


def _useful_section_vocab(lang: Optional[str]) -> Set[str]:
    """Local useful headings plus English section hints."""
    return _lang_vocab(USEFUL_CHEM_SECTION_HINTS, lang) | USEFUL_CHEM_SECTION_HINTS.get(
        "en", set()
    )


def _normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = RE_MULTISPACE.sub(" ", text)
    text = RE_MULTI_NEWLINES.sub("\n\n", text)
    return text.strip()


def _looks_like_chemistry(text: str, lang: Optional[str]) -> bool:
    if not text:
        return False
    text_l = text.lower()
    kw = _chem_vocab(lang)
    keyword_hits = sum(1 for k in kw if k in text_l)
    pattern_hits = 0
    for rx in (RE_CHEM_FORMULA, RE_COMPLEX_ION, RE_STEREO, RE_CHEM_NAME_NUM, RE_UNITS):
        if rx.search(text):
            pattern_hits += 1
    return keyword_hits >= 1 or pattern_hits >= 1


def _heading_norm(s: str) -> str:
    return _normalize_unicode(s).lower().strip(" .:;-–—")


def _uses_compact_script_heading_heuristic(s: str) -> bool:
    """Scripts where headings are often unspaced or single-token (CJK, Arabic, Devanagari)."""
    for ch in s:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:  # CJK
            return True
        if 0x3040 <= o <= 0x30FF:  # Kana
            return True
        if 0xAC00 <= o <= 0xD7AF:  # Hangul syllables
            return True
        if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F:  # Arabic
            return True
        if 0x0900 <= o <= 0x097F:  # Devanagari (hi)
            return True
    return False


def _is_probable_heading(paragraph: str) -> bool:
    p = paragraph.strip()
    if not p or "\n" in p:
        return False
    if len(p) > 80:
        return False
    if p.endswith((".", "!", "?", ";", "。", "！", "？")):
        return False
    if _uses_compact_script_heading_heuristic(p):
        return 1 <= len(p) <= 50
    words = p.split()
    return 1 <= len(words) <= 10


def _clean_inline_noise(text: str) -> str:
    text = RE_CITATION_NUM.sub("", text)
    text = RE_CITATION_NEEDED.sub("", text)
    text = RE_SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = RE_EMPTY_PARENS.sub("", text)
    text = RE_MULTISPACE.sub(" ", text)
    return text.strip()


def _is_low_value_paragraph(paragraph: str, lang: Optional[str]) -> bool:
    p = paragraph.strip()
    if not p:
        return True
    if _text_looks_disambig(p):
        return True
    if len(p) < 40 and not _looks_like_chemistry(p, lang):
        return True
    alnum_count = sum(ch.isalnum() for ch in p)
    if alnum_count < 10 and not _looks_like_chemistry(p, lang):
        return True
    return False


def _truncate_bad_terminal_sections(paragraphs: list[str], lang: Optional[str]) -> list[str]:
    bad_titles = _lang_vocab(BAD_SECTION_TITLES, lang)
    for i, p in enumerate(paragraphs):
        if _is_probable_heading(p) and _heading_norm(p) in bad_titles:
            return paragraphs[:i]
    return paragraphs


def clean_wikipedia_text(
    text: str,
    *,
    lang: Optional[str] = "en",
    keep_useful_headings: bool = True,
) -> str:
    """
    Clean MediaWiki plaintext extract for multilingual chemistry retrieval.

    Removes citation noise and low-value lines, drops trailing reference-style
    sections when headings are detected, keeps likely chemistry headings.
    """
    text = _normalize_unicode(text)
    if not text:
        return ""

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return ""

    paragraphs = _truncate_bad_terminal_sections(paragraphs, lang)
    useful_headings = _useful_section_vocab(lang)
    bad_titles = _lang_vocab(BAD_SECTION_TITLES, lang)

    cleaned: list[str] = []
    for para in paragraphs:
        para = _clean_inline_noise(para)
        if not para:
            continue

        if _is_probable_heading(para):
            hn = _heading_norm(para)
            if hn in bad_titles:
                break
            if keep_useful_headings and (
                hn in useful_headings or _looks_like_chemistry(para, lang)
            ):
                cleaned.append(para.strip())
            elif keep_useful_headings and len(para) <= 60:
                cleaned.append(para.strip())
            continue

        if _is_low_value_paragraph(para, lang):
            continue
        cleaned.append(para)

    deduped: list[str] = []
    prev: Optional[str] = None
    for item in cleaned:
        if item != prev:
            deduped.append(item)
        prev = item

    out = "\n\n".join(deduped).strip()
    out = RE_MULTI_NEWLINES.sub("\n\n", out)
    return out


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    paragraph = paragraph.strip()
    if len(paragraph) <= max_chars:
        return [paragraph]

    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]
    if len(sentences) > 1:
        out: list[str] = []
        buf = ""
        for sent in sentences:
            candidate = f"{buf} {sent}".strip() if buf else sent
            if len(candidate) <= max_chars:
                buf = candidate
            else:
                if buf:
                    out.append(buf)
                if len(sent) <= max_chars:
                    buf = sent
                else:
                    pieces = re.split(r"(?<=[,;:])\s+", sent)
                    inner = ""
                    for piece in pieces:
                        cand2 = f"{inner} {piece}".strip() if inner else piece
                        if len(cand2) <= max_chars:
                            inner = cand2
                        else:
                            if inner:
                                out.append(inner)
                            if len(piece) <= max_chars:
                                inner = piece
                            else:
                                for i in range(0, len(piece), max_chars):
                                    chunk = piece[i : i + max_chars].strip()
                                    if chunk:
                                        out.append(chunk)
                                inner = ""
                    buf = inner

        if buf:
            out.append(buf)
        return out

    return [
        paragraph[i : i + max_chars].strip()
        for i in range(0, len(paragraph), max_chars)
        if paragraph[i : i + max_chars].strip()
    ]


def chunk_plain_text_multilingual(
    text: str,
    *,
    max_chars: int,
    min_chars: int,
) -> list[str]:
    """
    Chunk cleaned plaintext; merge paragraphs up to max_chars with min_chars floor.
    Long paragraphs are split on sentence boundaries when possible.
    """
    text = _normalize_unicode(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    buf = ""

    for para in paragraphs:
        for part in _split_long_paragraph(para, max_chars=max_chars):
            candidate = f"{buf}\n\n{part}".strip() if buf else part

            if len(candidate) <= max_chars:
                buf = candidate
                continue

            if buf:
                if len(buf) >= min_chars:
                    chunks.append(buf)
                elif chunks:
                    chunks[-1] = (chunks[-1] + "\n\n" + buf).strip()
                else:
                    chunks.append(buf)

            buf = part

    if buf:
        if len(buf) >= min_chars:
            chunks.append(buf)
        elif chunks:
            chunks[-1] = (chunks[-1] + "\n\n" + buf).strip()
        else:
            chunks.append(buf)

    return chunks


def chemistry_relevance_score(text: str, *, lang: Optional[str] = "en") -> float:
    """Heuristic chemistry relevance in [0, 1] (optional filtering)."""
    if not text:
        return 0.0
    text_l = text.lower()
    kw = _chem_vocab(lang)
    keyword_hits = sum(1 for k in kw if k in text_l)
    pattern_hits = sum(
        1
        for rx in (
            RE_CHEM_FORMULA,
            RE_COMPLEX_ION,
            RE_STEREO,
            RE_CHEM_NAME_NUM,
            RE_UNITS,
        )
        if rx.search(text)
    )
    score = min(1.0, 0.08 * keyword_hits + 0.18 * pattern_hits)
    return round(score, 4)


__all__ = [
    "clean_wikipedia_text",
    "chunk_plain_text_multilingual",
    "chemistry_relevance_score",
]
