"""Najdi Phonetic G2P & Tashkeel Engine for natural White Saudi TTS synthesis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .lexicon import NajdiLexicon, get_default_lexicon, strip_diacritics

# Arabic-Indic digit translation table
_ARABIC_INDIC_TRANS = str.maketrans("٠١٢٣٤٥٦٧٨٩٫", "0123456789.")

# Long IDs, URLs, UUIDs, IBANs, tokens, and phone numbers (privacy-clean generic patterns)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_TOKEN_ID_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{10,}|gh[pousr]_[A-Za-z0-9]{12,}|0x[0-9a-fA-F]{8,}|SA\d{22}|[A-Za-z0-9_-]{16,})\b"
)
_PHONE_RE = re.compile(r"(?<!\d)(?:\+\d{1,3}[\s-]?)?(?:0?5\d{8}|\d{10,15})(?!\d)")
_LONG_DIGITS_RE = re.compile(r"\b\d{8,}\b")

# Date patterns: YYYY-MM-DD or DD/MM/YYYY
_ISO_DATE_RE = re.compile(r"\b(19\d{2}|20\d{2})-(0?[1-9]|1[0-2])-(0?[1-9]|[12]\d|3[01])\b")
_SLASH_DATE_RE = re.compile(r"\b(0?[1-9]|[12]\d|3[01])/(0?[1-9]|1[0-2])/(19\d{2}|20\d{2})\b")

# Time pattern: e.g. "الساعة 3:30 العصر", "4:45 PM", "9:15 ص", "10:30"
_TIME_RE = re.compile(
    r"(?P<prefix>الساعة\s+|السَّاعَة\s+)?"
    r"\b(?P<hour>[01]?\d|2[0-3]):(?P<minute>[0-5]\d)\b"
    r"(?:\s*(?P<meridiem>AM|PM|am|pm|ص|م|صباحاً|صباحا|مساءً|مساء|الصبح|الظهر|العصر|المغرب|العشاء|الفجر|بالليل))?"
)

# Currency & Percentage patterns
_PREFIX_USD_RE = re.compile(r"(?<!\w)\$\s*(?P<amount>\d+(?:,\d{3})*(?:\.\d{1,2})?)\b")
_CURRENCY_RE = re.compile(
    r"\b(?P<amount>\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*"
    r"(?P<unit>ريال(?:\s+سعودي)?|ريالات|ر\.س|SAR|هللة|هللات|دولار(?:\s+أمريكي)?|دولارات|USD)(?!\w)",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"\b(?P<amount>\d+(?:\.\d+)?)\s*[%٪]")
_NUMBER_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")

# English acronym & tech vocabulary phonetic map (Modern Saudi pronunciation)
_ENGLISH_PHONETIC_MAP: dict[str, str] = {
    "api": "إِي بِي آيْ",
    "apis": "إِي بِي آيْزْ",
    "tts": "تِي تِي إِسْ",
    "stt": "إِسْ تِي تِي",
    "asr": "إِي إِسْ آرْ",
    "ai": "إِي آيْ",
    "llm": "إِلْ إِلْ إِمْ",
    "llms": "إِلْ إِلْ إِمْزْ",
    "mcp": "إِمْ سِي بِي",
    "cli": "سِي إِلْ آيْ",
    "sdk": "إِسْ دِي كَيْ",
    "ide": "آيْ دِي إِي",
    "json": "جَيْسُونْ",
    "sql": "إِسْ كْيُو إِلْ",
    "url": "يُو آرْ إِلْ",
    "pdf": "بِي دِي إِفْ",
    "csv": "سِي إِسْ فِي",
    "otp": "أُو تِي بِي",
    "ui": "يُو آيْ",
    "ux": "يُو إِكْسْ",
    "gpu": "جِي بِي يُو",
    "cpu": "سِي بِي يُو",
    "ram": "رَامْ",
    "ssd": "إِسْ إِسْ دِي",
    "usb": "يُو إِسْ بِي",
    "wifi": "وَايْ فَايْ",
    "wi-fi": "وَايْ فَايْ",
    "pr": "بِي آرْ",
    "ci": "سِي آيْ",
    "cd": "سِي دِي",
    "ssh": "إِسْ إِسْ إِتْشْ",
    "https": "إِتْشْ تِي تِي بِي إِسْ",
    "http": "إِتْشْ تِي تِي بِي",
    "dns": "دِي إِنْ إِسْ",
    "vpn": "فِي بِي إِنْ",
    "vip": "فِي آيْ بِي",
    "kpi": "كَيْ بِي آيْ",
    "roi": "آرْ أُو آيْ",
    "vat": "ضَرِيبَةْ الْقِيمَة الْمُضَافَة",
    "iban": "آيْبَانْ",
    "usdt": "يُو إِسْ دِي تِي",
    "mlx": "إِمْ إِلْ إِكْسْ",
    "ok": "أُوكَيْ",
    "okay": "أُوكَيْ",
    "github": "قِيتْ هَبْ",
    "git": "قِيتْ",
    "python": "بَايْثُونْ",
    "openai": "أُوبِنْ إِي آيْ",
    "whisper": "وِيسْبَرْ",
    "whatsapp": "وَاتْسَابْ",
    "iphone": "آيْفُونْ",
    "ios": "آيْ أُو إِسْ",
    "macos": "مَاكْ أُو إِسْ",
    "app": "تَطْبِيقْ",
}
_LATIN_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9-]{1,15}\b")

# Tokenization preserving punctuation (including Arabic ، ؛ ؟) & whitespace
_TOKEN_SPLIT_RE = re.compile(r"([\u0621-\u064A\u0671\u064B-\u065F\u0670]+)")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

# MSA formal case tanween at word boundaries (excluding natural conversational fatha-tanween adverbs)
_FORMAL_TANWEEN_RE = re.compile(r"[\u064C\u064D](?=$|[\s،.؟!,:؛)\]])")

# High-frequency Urban Najdi / White Saudi imperfective verb stems (kasrah on تـ/يـ/نـ, fathah on أَ)
# Maps bare stem (after ت/ي/ن/أ) -> vocalized stem with pausal sukun
_IMPERFECTIVE_STEMS: dict[str, str] = {
    "عرف": "عْرِفْ",
    "قول": "قُولْ",
    "قدر": "قْدَرْ",
    "شوف": "شُوفْ",
    "سمع": "سْمَعْ",
    "كتب": "كْتِبْ",
    "قرا": "قْرَا",
    "فهم": "فْهَمْ",
    "حسب": "حْسِبْ",
    "دخل": "دْخُلْ",
    "طلع": "طْلَعْ",
    "رجع": "رْجَعْ",
    "ضبط": "ضْبِطْ",
    "لحق": "لْحَقْ",
    "صير": "صِيرْ",
    "روح": "رُوحْ",
    "مشي": "مْشِي",
    "جي": "جِي",
    "بي": "بِي",
    "شتغل": "شْتَغِلْ",
    "ستفيد": "سْتَفِيدْ",
    "ستمر": "سْتَمِرّ",
    "نتهي": "نْتَهِي",
    "ختصر": "خْتَصِرْ",
    "عتمد": "عْتَمِدْ",
    "نتظر": "نْتَظِرْ",
    "توقع": "تَوَقَّعْ",
    "تكلم": "تْكَلَّمْ",
    "تأكد": "تَأَكَّدْ",
    "تاكد": "تَأَكَّدْ",
    "تحسن": "تْحَسَّنْ",
    "تغير": "تْغَيَّرْ",
    "شيك": "شَيِّكْ",
    "خلص": "خَلِّصْ",
    "جهز": "جَهِّزْ",
    "رتب": "رَتِّبْ",
    "سوي": "سَوِّي",
    "كمل": "كَمِّلْ",
    "حول": "حَوِّلْ",
    "حدث": "حَدِّثْ",
    "حمل": "حَمِّلْ",
    "نزل": "نَزِّلْ",
    "جرب": "جَرِّبْ",
    "وضح": "وَضِّحْ",
    "وفر": "وَفِّرْ",
    "سهل": "سَهِّلْ",
    "ساعد": "سَاعِدْ",
    "راجع": "رَاجِعْ",
    "ناسب": "نَاسِبْ",
    "عالج": "عَالِجْ",
    "حفظ": "حْفَظْ",
    "ربط": "رْبِطْ",
    "ضيف": "ضِيفْ",
    "ختار": "خْتَارْ",
    "لقى": "لْقَى",
    "نفع": "نْفَعْ",
    "ضمن": "ضْمَنْ",
    "فرق": "فْرِقْ",
}

_ENCLITICS: tuple[tuple[str, str], ...] = (
    ("هم", "هُمْ"),
    ("كم", "كُمْ"),
    ("ها", "هَا"),
    ("نا", "نَا"),
    ("ني", "نِي"),
    ("ون", "ُونْ"),
    ("ين", "ِينْ"),
    ("ك", "ِكْ"),
    ("ه", "ِهْ"),
)

_PROCLITICS: tuple[tuple[str, str], ...] = (
    ("وال", "وَالْ"),
    ("فال", "فَالْ"),
    ("بال", "بِالْ"),
    ("للا", "لِلْا"),
    ("لل", "لِلْ"),
    ("ال", "الْ"),
    ("و", "وَ"),
    ("ف", "فَ"),
    ("ب", "بِ"),
)

_MONTHS_AR: dict[int, str] = {
    1: "يَنَايِرْ",
    2: "فِبْرَايِرْ",
    3: "مَارِسْ",
    4: "أَبْرِيلْ",
    5: "مَايُو",
    6: "يُونْيُو",
    7: "يُولْيُو",
    8: "أَغُسْطُسْ",
    9: "سِبْتَمْبَرْ",
    10: "أُكْتُوبَرْ",
    11: "نُوفَمْبَرْ",
    12: "دِيسَمْبَرْ",
}

_ONES_MASC = (
    "صِفِرْ",
    "وَاحِدْ",
    "اِثْنَيْنْ",
    "ثَلَاثَة",
    "أَرْبَعَة",
    "خَمْسَة",
    "سِتَّة",
    "سَبْعَة",
    "ثَمَانْيَة",
    "تِسْعَة",
)

_ONES_COUNT = (
    "صِفِرْ",
    "وَاحِدْ",
    "اِثْنَيْنْ",
    "ثَلَاثْ",
    "أَرْبَعْ",
    "خَمْسْ",
    "سِتّ",
    "سَبْعْ",
    "ثَمَانْ",
    "تِسْعْ",
)

_TEENS_SAUDI = (
    "عَشْرَة",
    "أَحْدَعْشْ",
    "اِثْنَعْشْ",
    "ثَلْطَعْشْ",
    "أَرْبَعْطَعْشْ",
    "خَمْسْطَعْشْ",
    "سِتَّعْشْ",
    "سَبْعْطَعْشْ",
    "ثَمَنْطَعْشْ",
    "تِسْعْطَعْشْ",
)

_TENS_SAUDI = (
    "",
    "عَشْرَة",
    "عِشْرِينْ",
    "ثَلَاثِينْ",
    "أَرْبَعِينْ",
    "خَمْسِينْ",
    "سِتِّينْ",
    "سَبْعِينْ",
    "ثَمَانِينْ",
    "تِسْعِينْ",
)

_HUNDREDS_SAUDI = (
    "",
    "مِيَّة",
    "مِيتَيْنْ",
    "ثَلَاثْمِيَّة",
    "أَرْبَعْمِيَّة",
    "خَمْسْمِيَّة",
    "سِتْمِيَّة",
    "سَبْعْمِيَّة",
    "ثَمَانْمِيَّة",
    "تِسْعْمِيَّة",
)

_HOUR_WORDS = {
    0: "اِثْنَعْشْ",
    1: "وَحْدَة",
    2: "ثِنْتَيْنْ",
    3: "ثَلَاثْ",
    4: "أَرْبَعْ",
    5: "خَمْسْ",
    6: "سِتّ",
    7: "سَبْعْ",
    8: "ثَمَانْ",
    9: "تِسْعْ",
    10: "عَشْرْ",
    11: "أَحْدَعْشْ",
    12: "اِثْنَعْشْ",
}


def number_to_saudi_words(n: int, *, counted_noun: bool = False) -> str:
    """Convert an integer (0..999,999,999) into natural Modern White Saudi spoken Arabic."""
    if n < 0:
        return f"سَالِبْ {number_to_saudi_words(-n, counted_noun=counted_noun)}"
    if n == 0:
        return _ONES_MASC[0]
    if n < 10:
        return _ONES_COUNT[n] if counted_noun else _ONES_MASC[n]
    if n < 20:
        if n == 10 and counted_noun:
            return "عَشْرْ"
        return _TEENS_SAUDI[n - 10]
    if n < 100:
        tens, rem = divmod(n, 10)
        if rem == 0:
            return _TENS_SAUDI[tens]
        return f"{_ONES_MASC[rem]} وَ{_TENS_SAUDI[tens]}"
    if n < 1000:
        hundreds, rem = divmod(n, 100)
        head = _HUNDREDS_SAUDI[hundreds]
        if rem == 0:
            return head
        return f"{head} وَ{number_to_saudi_words(rem, counted_noun=counted_noun)}"
    if n < 1_000_000:
        thousands, rem = divmod(n, 1000)
        if thousands == 1:
            head = "أَلْفْ"
        elif thousands == 2:
            head = "أَلْفَيْنْ"
        elif 3 <= thousands <= 10:
            head = f"{_ONES_COUNT[thousands]} آلَافْ"
        else:
            head = f"{number_to_saudi_words(thousands)} أَلْفْ"
        if rem == 0:
            return head
        return f"{head} وَ{number_to_saudi_words(rem, counted_noun=counted_noun)}"

    millions, rem = divmod(n, 1_000_000)
    if millions == 1:
        head = "مَلْيُونْ"
    elif millions == 2:
        head = "مَلْيُونَيْنْ"
    elif 3 <= millions <= 10:
        head = f"{_ONES_COUNT[millions]} مَلَايِينْ"
    else:
        head = f"{number_to_saudi_words(millions)} مَلْيُونْ"
    if rem == 0:
        return head
    return f"{head} وَ{number_to_saudi_words(rem, counted_noun=counted_noun)}"


def _format_counted_currency(amount: int, singular: str, dual: str, plural: str) -> str:
    if amount == 0:
        return f"صِفِرْ {singular}"
    if amount == 1:
        return f"{singular} وَاحِدْ"
    if amount == 2:
        return dual
    if 3 <= amount <= 10:
        return f"{number_to_saudi_words(amount, counted_noun=True)} {plural}"
    return f"{number_to_saudi_words(amount)} {singular}"


def _convert_currency(amount_str: str, unit_raw: str) -> str:
    clean_num = amount_str.replace(",", "")
    unit_low = unit_raw.strip().lower()
    is_usd = unit_low in ("usd", "$") or unit_low.startswith("دولار")
    is_halalah_only = unit_low.startswith("هلل")

    if is_halalah_only:
        val = int(float(clean_num))
        return _format_counted_currency(val, "هَلَلَة", "هَلَلْتَيْنْ", "هَلَلَاتْ")

    if "." in clean_num:
        whole_part, frac_part = clean_num.split(".", 1)
        whole = int(whole_part)
        frac = int((frac_part + "0")[:2])
    else:
        whole = int(clean_num)
        frac = 0

    if is_usd:
        main_phrase = _format_counted_currency(whole, "دُولَارْ", "دُولَارَيْنْ", "دُولَارَاتْ")
        if frac == 0:
            return main_phrase
        sub_phrase = _format_counted_currency(frac, "سِنْتْ", "سِنْتَيْنْ", "سِنْتَاتْ")
        return f"{main_phrase} وَ{sub_phrase}"

    main_phrase = _format_counted_currency(whole, "رِيَالْ", "رِيَالَيْنْ", "رِيَالَاتْ")
    if frac == 0:
        return main_phrase
    sub_phrase = _format_counted_currency(frac, "هَلَلَة", "هَلَلْتَيْنْ", "هَلَلَاتْ")
    return f"{main_phrase} وَ{sub_phrase}"


def _resolve_period(hour24: int, meridiem: str | None) -> str:
    if not meridiem:
        if hour24 == 0:
            return "بِاللَّيْلْ"
        if 13 <= hour24 <= 14:
            return "الظُّهْرْ"
        if 15 <= hour24 <= 17:
            return "الْعَصْرْ"
        if hour24 == 18:
            return "الْمَغْرِبْ"
        if hour24 >= 19:
            return "الْمَسَاءْ"
        return ""

    m = meridiem.strip().lower()
    explicit_map = {
        "الصبح": "الصُّبْحْ",
        "الظهر": "الظُّهْرْ",
        "العصر": "الْعَصْرْ",
        "المغرب": "الْمَغْرِبْ",
        "العشاء": "الْعِشَاءْ",
        "الفجر": "الْفَجْرْ",
        "بالليل": "بِاللَّيْلْ",
    }
    if m in explicit_map:
        return explicit_map[m]

    h12 = hour24 % 12 or 12
    if m in ("am", "ص", "صباحاً", "صباحا"):
        if h12 == 12:
            return "بِاللَّيْلْ"
        if 1 <= h12 <= 3:
            return "الْفَجْرْ"
        return "الصُّبْحْ"
    if m in ("pm", "م", "مساءً", "مساء"):
        if h12 == 12 or h12 in (1, 2):
            return "الظُّهْرْ"
        if 3 <= h12 <= 5:
            return "الْعَصْرْ"
        if h12 == 6:
            return "الْمَغْرِبْ"
        return "الْمَسَاءْ"
    return ""


def _convert_time(match: re.Match[str]) -> str:
    has_prefix = bool(match.group("prefix"))
    hour_raw = int(match.group("hour"))
    minute = int(match.group("minute"))
    meridiem = match.group("meridiem")

    display_hour = hour_raw
    if minute in (40, 45):
        display_hour = (hour_raw + 1) % 24

    h12 = display_hour % 12 or 12
    hour_word = _HOUR_WORDS[h12]

    if minute == 0:
        core = hour_word
    elif minute == 15:
        core = f"{hour_word} وَرُبْعْ"
    elif minute == 20:
        core = f"{hour_word} وَثُلْثْ"
    elif minute == 30:
        core = f"{hour_word} وَنُصّ"
    elif minute == 40:
        core = f"{hour_word} إِلَّا ثُلْثْ"
    elif minute == 45:
        core = f"{hour_word} إِلَّا رُبْعْ"
    elif minute == 1:
        core = f"{hour_word} وَدَقِيقَة"
    elif minute == 2:
        core = f"{hour_word} وَدَقِيقْتَيْنْ"
    elif 3 <= minute <= 10:
        core = f"{hour_word} وَ{number_to_saudi_words(minute, counted_noun=True)} دَقَايِقْ"
    else:
        core = f"{hour_word} وَ{number_to_saudi_words(minute)} دَقِيقَة"

    period = _resolve_period(hour_raw, meridiem)
    prefix = "السَّاعَة " if has_prefix else ""
    suffix = f" {period}" if period else ""
    return f"{prefix}{core}{suffix}"


def _convert_date(day: int, month: int, year: int) -> str:
    day_word = "أَوَّلْ" if day == 1 else number_to_saudi_words(day)
    month_word = _MONTHS_AR.get(month, str(month))
    year_word = number_to_saudi_words(year)
    return f"{day_word} {month_word} {year_word}"


def _convert_decimal_or_int(raw: str) -> str:
    clean = raw.replace(",", "")
    if "." not in clean:
        return number_to_saudi_words(int(clean))
    left, right = clean.split(".", 1)
    whole = int(left)
    if right in ("5", "50"):
        if whole == 0:
            return "نُصّ"
        return f"{number_to_saudi_words(whole)} وَنُصّ"
    if right == "25":
        if whole == 0:
            return "رُبْعْ"
        return f"{number_to_saudi_words(whole)} وَرُبْعْ"
    if right == "75":
        return f"{number_to_saudi_words(whole)} وَثَلَاثْ أَرْبَاعْ"
    frac_trimmed = right.rstrip("0") or "0"
    return f"{number_to_saudi_words(whole)} فَاصْلَة {number_to_saudi_words(int(frac_trimmed))}"


@dataclass(slots=True)
class PhoneticResult:
    """Structured output of the NajdiPhoneticsEngine."""

    original: str
    speakable: str
    skipped_ids: int = 0
    lexicon_hits: int = 0
    rule_hits: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "speakable": self.speakable,
            "skipped_ids": self.skipped_ids,
            "lexicon_hits": self.lexicon_hits,
            "rule_hits": self.rule_hits,
        }


class NajdiPhoneticsEngine:
    """Morpho-phonetic G2P & tashkeel processor for Modern White Saudi TTS."""

    def __init__(self, lexicon: NajdiLexicon | None = None) -> None:
        self.lexicon = lexicon or get_default_lexicon()

    def _vocalize_imperfective(self, plain: str) -> str | None:
        """Apply Urban Najdi kasrah on imperfective prefixes (تِـ/يِـ/نِـ) & pausal sukun."""
        if len(plain) < 3:
            return None

        future_ba = False
        work = plain
        if work.startswith("ب") and len(work) >= 4 and work[1] in ("ت", "ي", "ن"):
            future_ba = True
            work = work[1:]

        prefix_char = work[0]
        if prefix_char not in ("ت", "ي", "ن", "أ", "ا"):
            return None

        rest = work[1:]
        enclitic_voc = ""
        stem_candidate = rest

        if stem_candidate not in _IMPERFECTIVE_STEMS:
            for enc_plain, enc_v in _ENCLITICS:
                if rest.endswith(enc_plain) and len(rest) - len(enc_plain) >= 2:
                    cand = rest[: -len(enc_plain)]
                    if cand in _IMPERFECTIVE_STEMS:
                        stem_candidate = cand
                        enclitic_voc = enc_v
                        break
            else:
                return None

        voc_stem = _IMPERFECTIVE_STEMS[stem_candidate]
        if prefix_char in ("أ", "ا"):
            voc_prefix = "بَأَ" if future_ba else "أَ"
        else:
            # Signature Najdi/White Saudi kasrah on mudara'ah prefix
            voc_prefix = f"بِ{prefix_char}ْ" if future_ba else f"{prefix_char}ِ"

        combined = f"{voc_prefix}{voc_stem}"
        if enclitic_voc:
            # Replace terminal sukun before vowel-initial enclitics like ـون / ـين
            if enclitic_voc[0] in ("\u064F", "\u0650") and combined.endswith("ْ"):
                combined = combined[:-1] + enclitic_voc
            else:
                combined = combined + enclitic_voc
        return combined

    def _vocalize_with_clitics(self, word: str) -> tuple[str, bool, bool]:
        """Return (vocalized_word, lexicon_hit, rule_hit)."""
        direct = self.lexicon.vocalize_token(word)
        if direct is not None:
            return direct, True, False

        plain = strip_diacritics(word)
        imp = self._vocalize_imperfective(plain)
        if imp is not None:
            return imp, False, True

        # Try proclitic stripping (و، ف، ب، ال، وال، بال، لل)
        for proc_plain, proc_voc in _PROCLITICS:
            if plain.startswith(proc_plain) and len(plain) - len(proc_plain) >= 2:
                remainder = plain[len(proc_plain) :]
                rem_voc = self.lexicon.vocalize_token(remainder)
                if rem_voc is not None:
                    if proc_plain.endswith("ال") and rem_voc.startswith("ال"):
                        return proc_voc[:-3] + rem_voc, True, False
                    return proc_voc + rem_voc, True, False
                rem_imp = self._vocalize_imperfective(remainder)
                if rem_imp is not None and proc_plain in ("و", "ف"):
                    return proc_voc + rem_imp, False, True

        # Apply conversational pausal sukun cleanup if word carries formal MSA tanween
        cleaned = _FORMAL_TANWEEN_RE.sub("ْ", word)
        if cleaned != word:
            return cleaned, False, True
        return word, False, False

    def prepare_tts(
        self,
        text: str,
        *,
        skip_long_ids: bool = True,
        long_id_placeholder: str = "",
        expand_numbers: bool = True,
        expand_acronyms: bool = True,
    ) -> PhoneticResult:
        """Transform text into Najdi/White Saudi phonetically tuned TTS input."""
        if not text or not text.strip():
            return PhoneticResult(original=text, speakable=text)

        work = text.translate(_ARABIC_INDIC_TRANS)
        skipped_ids = 0

        if skip_long_ids:
            for pattern in (_URL_RE, _UUID_RE, _TOKEN_ID_RE, _PHONE_RE, _LONG_DIGITS_RE):
                work, n = pattern.subn(long_id_placeholder, work)
                skipped_ids += n

        if expand_numbers:
            # 1. Dates before slash/hyphen numbers
            work = _ISO_DATE_RE.sub(
                lambda m: _convert_date(int(m.group(3)), int(m.group(2)), int(m.group(1))),
                work,
            )
            work = _SLASH_DATE_RE.sub(
                lambda m: _convert_date(int(m.group(1)), int(m.group(2)), int(m.group(3))),
                work,
            )
            # 2. Times (3:30, 4:45 PM, etc.)
            work = _TIME_RE.sub(_convert_time, work)
            # 3. Currencies ($100, 150 ريال, 45.50 SAR)
            work = _PREFIX_USD_RE.sub(lambda m: _convert_currency(m.group("amount"), "USD"), work)
            work = _CURRENCY_RE.sub(
                lambda m: _convert_currency(m.group("amount"), m.group("unit")),
                work,
            )
            # 4. Percentages (25%)
            work = _PERCENT_RE.sub(
                lambda m: f"{_convert_decimal_or_int(m.group('amount'))} بِالْمِيَّة",
                work,
            )
            # 5. Standalone numbers
            work = _NUMBER_RE.sub(lambda m: _convert_decimal_or_int(m.group(0)), work)

        if expand_acronyms:
            def _replace_latin(match: re.Match[str]) -> str:
                raw_tok = match.group(0)
                mapped = _ENGLISH_PHONETIC_MAP.get(raw_tok.lower())
                return mapped if mapped is not None else raw_tok

            work = _LATIN_WORD_RE.sub(_replace_latin, work)

        # Strip formal MSA dammatan/kasratan case endings
        work = _FORMAL_TANWEEN_RE.sub("ْ", work)

        # Multi-word longest-match phrase vocalization + single-token morpho-phonetics
        lex_hits = 0
        rule_hits = 0
        multi_idx = self.lexicon.multiword_index

        # Split by whitespace while keeping punctuation attached cleanly
        raw_tokens = work.split()
        out_tokens: list[str] = []
        i = 0
        n_toks = len(raw_tokens)

        while i < n_toks:
            tok = raw_tokens[i]
            # Extract leading/trailing non-Arabic punctuation around token
            m_first = _TOKEN_SPLIT_RE.search(tok)
            if m_first is not None:
                first_plain = strip_diacritics(m_first.group(1))
                candidates = multi_idx.get(first_plain)
                matched_span = 0
                if candidates:
                    for phrase_parts, entry in candidates:
                        span_len = len(phrase_parts)
                        if i + span_len <= n_toks:
                            window_plains: list[str] = []
                            ok = True
                            for k in range(span_len):
                                sub_m = _TOKEN_SPLIT_RE.search(raw_tokens[i + k])
                                if sub_m is None:
                                    ok = False
                                    break
                                window_plains.append(strip_diacritics(sub_m.group(1)))
                            if ok and tuple(window_plains) == phrase_parts:
                                lead = tok[: m_first.start()]
                                last_tok = raw_tokens[i + span_len - 1]
                                last_m = _TOKEN_SPLIT_RE.search(last_tok)
                                trail = last_tok[last_m.end() :] if last_m else ""
                                out_tokens.append(f"{lead}{entry.vocalized}{trail}")
                                lex_hits += 1
                                matched_span = span_len
                                break
                if matched_span:
                    i += matched_span
                    continue

            # Single-token processing (may contain punctuation like "أبشر،")
            parts = _TOKEN_SPLIT_RE.split(tok)
            rebuilt: list[str] = []
            for idx, seg in enumerate(parts):
                if idx % 2 == 1 and seg:
                    # Already carries explicit short vowels/sukun from number/time/homograph expansion?
                    if any(ch in "\u064E\u064F\u0650\u0652" for ch in seg):
                        rebuilt.append(seg)
                    else:
                        voc, is_lex, is_rule = self._vocalize_with_clitics(seg)
                        if is_lex:
                            lex_hits += 1
                        elif is_rule:
                            rule_hits += 1
                        rebuilt.append(voc)
                else:
                    rebuilt.append(seg)
            out_tokens.append("".join(rebuilt))
            i += 1

        speakable = _MULTI_SPACE_RE.sub(" ", " ".join(out_tokens)).strip()
        # Clean stray spaces before punctuation when IDs are omitted
        speakable = re.sub(r"\s+([،.؟!,:؛])", r"\1", speakable)
        return PhoneticResult(
            original=text,
            speakable=speakable,
            skipped_ids=skipped_ids,
            lexicon_hits=lex_hits,
            rule_hits=rule_hits,
        )


_DEFAULT_ENGINE: NajdiPhoneticsEngine | None = None


def to_speakable(
    text: str,
    *,
    skip_long_ids: bool = True,
    expand_numbers: bool = True,
    expand_acronyms: bool = True,
    lexicon: NajdiLexicon | None = None,
) -> str:
    """Convenience function returning Najdi/White Saudi speakable text in < 0.3ms."""
    global _DEFAULT_ENGINE
    if lexicon is not None:
        return NajdiPhoneticsEngine(lexicon=lexicon).prepare_tts(
            text,
            skip_long_ids=skip_long_ids,
            expand_numbers=expand_numbers,
            expand_acronyms=expand_acronyms,
        ).speakable
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = NajdiPhoneticsEngine()
    return _DEFAULT_ENGINE.prepare_tts(
        text,
        skip_long_ids=skip_long_ids,
        expand_numbers=expand_numbers,
        expand_acronyms=expand_acronyms,
    ).speakable
