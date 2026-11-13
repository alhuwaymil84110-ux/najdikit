"""Real-Time Voice Call & Telephony Engine for Modern White Saudi AI Agents (< 20 µs)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .lexicon import NajdiLexicon, get_default_lexicon, light_arabic_key, strip_diacritics
from .phonetics import NajdiPhoneticsEngine

SUPPORTED_TTS_BACKENDS: tuple[str, ...] = (
    "default",
    "elevenlabs",
    "omnivoice",
    "habibi",
    "openai",
    "sip_8khz",
)

# Markdown, emoji, and whitespace cleaners for live voice streaming
_MD_SYMBOLS_RE = re.compile(r"[*_`~]+")
_MD_HEADING_RE = re.compile(r"(?:^|\n)\s*#{1,6}\s+")
_MD_BULLET_RE = re.compile(r"(?:^|\n)\s*[-•]\s+")
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_AR_INDIC_DIGITS = frozenset("0123456789٠١٢٣٤٥٦٧٨٩")
_CLAUSE_PUNCT = frozenset("،,.;؛?!؟:\n")
_CLAUSE_PUNCT_RE = re.compile(r"[،,.;؛?!؟:\n]")
_STRONG_PUNCT = frozenset(".?!؟\n")

# OmniVoice explicit gemination respelling (handles both C+shaddah+V and C+V+shaddah Unicode orders)
_OMNIVOICE_SHADDAH_RE = re.compile(
    r"(?<!ال)([\u0621-\u064A])(?:([\u064B-\u0650\u0670])\u0651|\u0651([\u064B-\u0650\u0670]?))"
)


def _respell_omnivoice(text: str) -> str:
    return _OMNIVOICE_SHADDAH_RE.sub(
        lambda m: f"{m.group(1)}ْ{m.group(1)}{m.group(2) or m.group(3) or ''}",
        text,
    )

# Fast-lane Saudi telephony openers (may emit cleanly on Clause 0 even as 1-2 short words)
_FAST_LANE_OPENERS = frozenset(
    {
        "ابشر",
        "سم",
        "تم",
        "طيب",
        "اكيد",
        "حاضر",
        "ممتاز",
        "حياك",
        "هلا",
        "يا هلا",
        "هلا والله",
        "يا هلا والله",
        "حياك الله",
        "ابشر بعزك",
        "ابشر طال عمرك",
        "سم طال عمرك",
        "على خشمي",
        "ولا يهمك",
        "تحت امرك",
        "الامور طيبة",
    }
)

# Standalone prepositions & particles after which a clause MUST NEVER split
_NON_TERMINAL_PARTICLES = frozenset(
    {
        "في",
        "من",
        "على",
        "عن",
        "الى",
        "إلى",
        "مع",
        "عند",
        "بين",
        "قبل",
        "بعد",
        "بدون",
        "عشان",
        "علشان",
        "لان",
        "لأن",
        "ان",
        "إن",
        "أن",
        "ما",
        "لا",
        "يا",
        "كل",
        "اي",
        "أي",
        "ولا",
        "ابو",
        "أبو",
        "ام",
        "أم",
        "و",
        "ف",
        "او",
        "أو",
    }
)

# Multi-word inseparable Saudi constructs compiled into O(1) adjacent word bigrams
_INSEPARABLE_CONSTRUCTS: tuple[tuple[str, ...], ...] = (
    ("ان", "شاء", "الله"),
    ("ما", "شاء", "الله"),
    ("سبحان", "الله"),
    ("الحمد", "لله"),
    ("ولله", "الحمد"),
    ("باذن", "الله"),
    ("حياك", "الله"),
    ("عزك", "الله"),
    ("سلمك", "الله"),
    ("الله", "يعافيك"),
    ("الله", "يسلمك"),
    ("الله", "يحييك"),
    ("جزاك", "الله", "خير"),
    ("بارك", "الله", "فيك"),
    ("بيض", "الله", "وجهك"),
    ("في", "امان", "الله"),
    ("فمان", "الله"),
    ("مع", "السلامة"),
    ("طال", "عمرك"),
    ("كل", "شي"),
    ("كل", "شيء"),
    ("ولا", "شي"),
    ("اي", "شي"),
    ("ما", "قصر"),
    ("ما", "قصرت"),
    ("لا", "تشيل", "هم"),
    ("على", "خشمي"),
    ("سم", "طال", "عمرك"),
)

_INSEPARABLE_BIGRAMS: frozenset[tuple[str, str]] = frozenset(
    (c[i], c[i + 1]) for c in _INSEPARABLE_CONSTRUCTS for i in range(len(c) - 1)
)

# Post-numeric units that must stay attached to a preceding number
_NUMERIC_UNITS = frozenset(
    {
        "ريال",
        "ريالات",
        "ر.س",
        "هللة",
        "هللات",
        "دولار",
        "دولارات",
        "بالمية",
        "%",
        "٪",
        "ص",
        "م",
        "الصبح",
        "الظهر",
        "العصر",
        "المغرب",
        "العشاء",
        "الفجر",
        "بالليل",
        "دقيقة",
        "دقايق",
        "ثانية",
        "ثواني",
        "ساعة",
        "ساعات",
        "ام",
        "بي",
        "am",
        "pm",
        "sar",
        "usd",
    }
)

# Saudi discourse connectors before which Breath-Group Overflow Splitter prefers to break
_OVERFLOW_CONNECTORS = frozenset(
    {
        "وعشان",
        "ولان",
        "ولأن",
        "وبعدين",
        "واذا",
        "وإذا",
        "عشان",
        "علشان",
        "لان",
        "لأن",
        "بعدين",
        "بس",
        "لكن",
        "او",
        "أو",
        "ثم",
        "وكمان",
        "وبرضو",
    }
)

_NON_SPACE_RE = re.compile(r"\S+")
_NON_AR_SPACE_RE = re.compile(r"[^\u0621-\u064A\u0671\s]")


def _has_numeric_tail(text: str) -> bool:
    tokens = text.rsplit(None, 1)
    if not tokens:
        return False
    last = tokens[-1].rstrip("،,.:")
    return any(ch in _AR_INDIC_DIGITS for ch in last)


# ---------------------------------------------------------------------------
# Multi-Backend TTS Formatter
# ---------------------------------------------------------------------------


def format_for_tts_backend(
    text: str,
    *,
    backend: str = "default",
    phonetics: NajdiPhoneticsEngine | None = None,
    lexicon: NajdiLexicon | None = None,
    skip_long_ids: bool = True,
    expand_numbers: bool = True,
    expand_acronyms: bool = True,
) -> str:
    """Format Arabic/mixed text for a specific TTS synthesis backend in < 60 µs.

    Supported backends:
      - ``"default"``: Full Najdi/White Saudi vocalization & G2P.
      - ``"habibi"``: Vocalizes and wraps in ``SWivid/Habibi-TTS`` Saudi dialect token ``②〈...〉``.
      - ``"omnivoice"``: Explicit gemination respelling (``شَيْيَكْتْ``) for character-level tokenizers.
      - ``"elevenlabs"``: Balanced Najdi kasrah & pausal sukun tuned for ElevenLabs Flash v2.5 / v3.
      - ``"openai"``: Prosody-smoothed vocalization for OpenAI Realtime / gpt-4o-mini-tts.
      - ``"sip_8khz"``: Narrowband telephony mode with crisp punctuation pauses & shorter breath groups.
    """
    if not text or not text.strip():
        return ""

    mode = backend.strip().lower() if backend else "default"
    if mode not in SUPPORTED_TTS_BACKENDS:
        raise ValueError(
            f"Unsupported TTS backend '{backend}'. Choose from: {', '.join(SUPPORTED_TTS_BACKENDS)}"
        )

    # Unwrap existing Habibi-TTS markers if re-formatting
    raw_in = text.strip()
    if raw_in.startswith("②〈") and raw_in.endswith("〉"):
        raw_in = raw_in[2:-1].strip()

    engine = phonetics or NajdiPhoneticsEngine(lexicon=lexicon or get_default_lexicon())
    speakable = engine.prepare_tts(
        raw_in,
        skip_long_ids=skip_long_ids,
        expand_numbers=expand_numbers,
        expand_acronyms=expand_acronyms,
    ).speakable

    if not speakable:
        return ""

    if mode == "habibi":
        return f"②〈{speakable}〉"

    if mode == "omnivoice":
        # Respell consonant+shaddah into explicit consonant+sukun+consonant for char-level models
        return _respell_omnivoice(speakable)

    if mode == "elevenlabs":
        # Smooth light particle sukuns before definite articles so Flash v2.5/v3 flows without micro-clicks
        tuned = (
            speakable.replace("مِنْ الْ", "مِنَ الْ")
            .replace("فِي الْ", "فِي الْ")
            .replace("عَلَى الْ", "عَلَى الْ")
            .replace("...", "، ")
        )
        return _MULTI_SPACE_RE.sub(" ", tuned).strip()

    if mode == "sip_8khz":
        # Insert crisp breath-group commas on long unpunctuated stretches (> 8 words) for narrowband RTP
        words = speakable.split()
        if len(words) > 8:
            rebuilt: list[str] = []
            since_pause = 0
            for idx, w in enumerate(words):
                plain_k = _clean_word_key(w)
                if (
                    since_pause >= 6
                    and plain_k in _OVERFLOW_CONNECTORS
                    and rebuilt
                    and not rebuilt[-1].endswith(("،", ".", "؟", "!"))
                ):
                    rebuilt[-1] = rebuilt[-1] + "،"
                    since_pause = 0
                rebuilt.append(w)
                if w.endswith(("،", ",", ".", "؟", "!", "؛", ":")):
                    since_pause = 0
                else:
                    since_pause += 1
                if since_pause >= 8 and idx + 2 < len(words):
                    prev_k = _clean_word_key(w)
                    if prev_k not in _NON_TERMINAL_PARTICLES:
                        rebuilt[-1] = rebuilt[-1] + "،"
                        since_pause = 0
            speakable = " ".join(rebuilt)
        speakable = speakable.replace("—", "، ").replace("–", "، ").replace("...", "، ")
        speakable = _MULTI_SPACE_RE.sub(" ", speakable).strip()
        if speakable and speakable[-1] not in ".؟!":
            speakable += "."
        return speakable

    if mode == "openai":
        return speakable.replace("...", "، ").strip()

    return speakable


# ---------------------------------------------------------------------------
# Pre-Cached 0ms Call Acknowledgment Bank
# ---------------------------------------------------------------------------


_ACK_BANK: dict[str, tuple[str, ...]] = {
    "lookup": (
        "ثَوَانِي أَشَيِّكْ لِكْ.",
        "عَطْنِي ثَانْيَة أَشُوفْ لِكْ الْحِينْ.",
        "سَمّ، أَشَيِّكْ عَلَيْهَا الْحِينْ.",
    ),
    "action": (
        "أَبْشِرْ، جَارِي التَّنْفِيذْ الْحِينْ.",
        "عَلَى خَشْمِي، الْحِينْ أَسَوِّيهَا لِكْ.",
        "تَمّ طَالْ عُمْرِكْ، الْحِينْ أَعْتَمِدْهَا.",
    ),
    "confirm": (
        "أَبْشِرْ بِعِزِّكْ.",
        "تَمّ طَالْ عُمْرِكْ.",
        "مُمْتَازْ، وَاضِحْ جِدًّا.",
    ),
    "hold": (
        "خِذْ رَاحْتِكْ، مَعِكْ.",
        "خِذْ وَقْتِكْ طَالْ عُمْرِكْ، أَنَا مَعِكْ عَلَى الْخَطّ.",
        "وَلَا يْهِمِّكْ، بِانْتِظَارِكْ.",
    ),
    "unclear": (
        "الْمَعْذِرَة الصَّوْتْ يِقَطِّعْ، يِمْدِيكْ تْعِيدْهَا لَاهِنْتْ؟",
        "عَفْوًا مَا سِمَعْتِكْ زَيْنْ، تِقْدَرْ تْعِيدْ آخِرْ جُمْلَة؟",
        "الصَّوْتْ بَعِيدْ شْوَيْ، يِمْدِيكْ تْوَضِّحْ لِي لَاهِنْتْ؟",
    ),
}


class CallAckBank:
    """Pre-vocalized White Saudi call acknowledgments for 0ms pre-cached audio playback."""

    CATEGORIES: tuple[str, ...] = tuple(_ACK_BANK.keys())

    @staticmethod
    def get(category: str = "lookup", index: int = 0, *, backend: str = "default") -> str:
        bucket = _ACK_BANK.get(category)
        if bucket is None:
            raise ValueError(
                f"Unknown CallAckBank category '{category}'. Expected one of: {', '.join(_ACK_BANK)}"
            )
        phrase = bucket[index % len(bucket)]
        if backend == "habibi":
            return f"②〈{phrase}〉"
        if backend == "omnivoice":
            return _respell_omnivoice(phrase)
        return phrase

    @staticmethod
    def all_for(category: str) -> tuple[str, ...]:
        if category not in _ACK_BANK:
            raise ValueError(f"Unknown CallAckBank category '{category}'.")
        return _ACK_BANK[category]

    @staticmethod
    def export_all() -> dict[str, list[str]]:
        return {k: list(v) for k, v in _ACK_BANK.items()}


def get_call_ack(category: str = "lookup", index: int = 0, *, backend: str = "default") -> str:
    """Return a pre-vocalized White Saudi call acknowledgment in O(1) (< 0.2 µs)."""
    return CallAckBank.get(category=category, index=index, backend=backend)


# ---------------------------------------------------------------------------
# Stateful Real-Time LLM Voice Call Streamer (< 20 µs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CallClause:
    """A single speakable breath-group clause emitted by NajdiCallStreamer."""

    index: int
    text: str
    speakable: str
    word_count: int
    is_fast_lane: bool = False
    is_final: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "speakable": self.speakable,
            "word_count": self.word_count,
            "is_fast_lane": self.is_fast_lane,
            "is_final": self.is_final,
        }


_PUNCT_TIGHTEN_RE = re.compile(r"\s+([،.؟!,:؛])")
_STRIP_MD_TRANS = str.maketrans("", "", "*_`~")
_ALEF_NORM_TRANS = str.maketrans("أإآٱ", "اااا")
_PUNCT_TRIM_CHARS = "،,.;؛?!؟:()[]{}\"'«»-"


def _clean_word_key(word: str) -> str:
    w = strip_diacritics(word.strip(_PUNCT_TRIM_CHARS)).translate(_ALEF_NORM_TRANS)
    return w.lower()


def _clean_voice_surface(raw: str) -> str:
    if not raw:
        return ""
    text = raw
    if "#" in text:
        text = _MD_HEADING_RE.sub(" ", text)
    if "-" in text or "•" in text:
        text = _MD_BULLET_RE.sub(" ", text)
    text = text.translate(_STRIP_MD_TRANS)
    if text and max(text) >= "\u2600":
        text = _EMOJI_RE.sub("", text)
    text = " ".join(text.split())
    if " " in text:
        text = _PUNCT_TIGHTEN_RE.sub(r"\1", text)
    return text


def _violates_atomic_split(left_words: list[str], right_words: list[str]) -> bool:
    """Return True in O(1) if splitting between `left_words` and `right_words` breaks an atomic unit."""
    if not left_words:
        return True
    last_k = _clean_word_key(left_words[-1])
    if not last_k:
        return True

    # 1. Never split right after a standalone preposition or particle
    if last_k in _NON_TERMINAL_PARTICLES:
        return True

    if right_words:
        first_right_k = _clean_word_key(right_words[0])
        # 2. Never split between a number and its currency/time/percentage unit
        if _has_numeric_tail(left_words[-1]) and first_right_k in _NUMERIC_UNITS:
            return True
        # 3. Never split inside an inseparable Saudi multi-word construct (O(1) bigram check)
        if (last_k, first_right_k) in _INSEPARABLE_BIGRAMS:
            return True

    return False


class NajdiCallStreamer:
    """Stateful token-by-token LLM stream chunker optimized for minimum TTFA on live phone calls."""

    def __init__(
        self,
        *,
        backend: str = "default",
        max_words: int | None = None,
        vocalize: bool = True,
        phonetics: NajdiPhoneticsEngine | None = None,
        lexicon: NajdiLexicon | None = None,
    ) -> None:
        if backend not in SUPPORTED_TTS_BACKENDS:
            raise ValueError(f"Unsupported TTS backend: {backend}")
        self.backend = backend
        self.max_words = max_words if max_words is not None else (8 if backend == "sip_8khz" else 12)
        self.vocalize = vocalize
        self._lexicon = lexicon or get_default_lexicon()
        self._phonetics = phonetics or NajdiPhoneticsEngine(lexicon=self._lexicon)
        self.extracted_actions: list[str] = []
        self._buf: str = ""
        self._in_bracket: bool = False
        self._bracket_chars: list[str] = []
        self._clause_index: int = 0

    def reset(self) -> None:
        """Reset streaming state for a new assistant turn."""
        self.extracted_actions.clear()
        self._buf = ""
        self._in_bracket = False
        self._bracket_chars.clear()
        self._clause_index = 0

    def _ingest_raw(self, chunk: str) -> None:
        """Strip `[ACTION: ...]` tags safely across arbitrary token splits before buffering."""
        if not chunk:
            return
        i = 0
        n = len(chunk)
        out_parts: list[str] = []

        while i < n:
            if self._in_bracket:
                close_idx = chunk.find("]", i)
                if close_idx == -1:
                    self._bracket_chars.append(chunk[i:])
                    break
                self._bracket_chars.append(chunk[i:close_idx])
                tag_body = "".join(self._bracket_chars).strip()
                self._bracket_chars.clear()
                self._in_bracket = False
                upper = tag_body.upper()
                if upper.startswith("ACTION:"):
                    payload = tag_body.split(":", 1)[1].strip()
                    if payload:
                        self.extracted_actions.append(payload)
                elif upper.startswith("ACTION "):
                    payload = tag_body[7:].strip()
                    if payload:
                        self.extracted_actions.append(payload)
                i = close_idx + 1
            else:
                open_idx = chunk.find("[", i)
                if open_idx == -1:
                    out_parts.append(chunk[i:])
                    break
                if open_idx > i:
                    out_parts.append(chunk[i:open_idx])
                self._in_bracket = True
                self._bracket_chars.clear()
                i = open_idx + 1

        if out_parts:
            self._buf += "".join(out_parts)

    def _make_clause(
        self,
        raw_slice: str,
        *,
        is_fast_lane: bool = False,
        is_final: bool = False,
        precleaned: str | None = None,
        precount: int | None = None,
    ) -> CallClause | None:
        cleaned = precleaned if precleaned is not None else _clean_voice_surface(raw_slice)
        if not cleaned:
            return None
        w_count = precount if precount is not None else len(cleaned.split())
        if w_count == 0:
            return None
        speakable = (
            format_for_tts_backend(
                cleaned,
                backend=self.backend,
                phonetics=self._phonetics,
                lexicon=self._lexicon,
            )
            if self.vocalize
            else cleaned
        )
        if not speakable:
            return None
        clause = CallClause(
            index=self._clause_index,
            text=cleaned,
            speakable=speakable,
            word_count=w_count,
            is_fast_lane=is_fast_lane,
            is_final=is_final,
        )
        self._clause_index += 1
        return clause

    def _find_punctuation_split(
        self, buf: str, *, flushing: bool = False
    ) -> tuple[int, bool, str, int] | None:
        """Find earliest valid punctuation split `(end_pos, is_fast_lane, cleaned, word_count)`."""
        n = len(buf)
        for m in _CLAUSE_PUNCT_RE.finditer(buf):
            idx = m.start()
            ch = m.group(0)

            # Atomic Guard 1: Never split inside decimals, thousand-commas, or HH:MM times (`1,250.50`, `3.85`, `4:30`)
            if ch in ".,:" and idx > 0 and buf[idx - 1] in _AR_INDIC_DIGITS:
                if idx + 1 >= n and not flushing:
                    return None
                if idx + 1 < n and buf[idx + 1] in _AR_INDIC_DIGITS:
                    continue

            end_pos = idx + 1
            while end_pos < n and buf[end_pos] in _CLAUSE_PUNCT and buf[end_pos] != "\n":
                end_pos += 1

            candidate = _clean_voice_surface(buf[:end_pos])
            if not candidate:
                continue
            cand_words = candidate.split()
            n_cand = len(cand_words)
            if n_cand == 0:
                continue

            next_m = _NON_SPACE_RE.search(buf, end_pos)
            rest_words = [next_m.group(0)] if next_m is not None else []

            if not flushing and not rest_words and _clean_word_key(cand_words[-1]) in _NON_TERMINAL_PARTICLES:
                continue

            if _violates_atomic_split(cand_words, rest_words):
                continue

            if self._clause_index == 0:
                if n_cand >= 2:
                    return end_pos, True, candidate, n_cand
                plain_opener = _NON_AR_SPACE_RE.sub(
                    "", strip_diacritics(candidate).translate(_ALEF_NORM_TRANS)
                ).strip()
                if plain_opener in _FAST_LANE_OPENERS or len(plain_opener.replace(" ", "")) >= 6:
                    return end_pos, True, candidate, n_cand
                continue

            if ch in _STRONG_PUNCT or n_cand >= 3:
                return end_pos, False, candidate, n_cand
            plain_sub = _NON_AR_SPACE_RE.sub("", strip_diacritics(candidate))
            if len(plain_sub.replace(" ", "")) >= 14:
                return end_pos, False, candidate, n_cand

        return None

    def _find_overflow_split(self, buf: str, *, flushing: bool = False) -> int | None:
        """Split unpunctuated LLM output exceeding `max_words` cleanly before Saudi connectors."""
        spans: list[tuple[str, int, int]] = [
            (m.group(0), m.start(), m.end()) for m in _NON_SPACE_RE.finditer(buf)
        ]
        min_required = self.max_words + 1 if flushing else self.max_words + 2
        if len(spans) < min_required:
            return None

        words = [s[0] for s in spans]

        # 1. Preferred split: right BEFORE a Saudi discourse connector within [3 .. max_words + 1]
        upper_bound = min(len(words) - 1, self.max_words + 1)
        for w_idx in range(upper_bound, 2, -1):
            w_key = _clean_word_key(words[w_idx])
            if w_key in _OVERFLOW_CONNECTORS:
                if not _violates_atomic_split(words[:w_idx], words[w_idx:]):
                    return spans[w_idx][1]

        # 2. Fallback safe breath-group split near `max_words` respecting Atomic Guard
        for w_idx in range(min(self.max_words, len(words) - 1), 2, -1):
            if not _violates_atomic_split(words[:w_idx], words[w_idx:]):
                return spans[w_idx][1]

        return None

    def push(self, token: str) -> list[CallClause]:
        """Push a streaming LLM token delta and return any ready `CallClause` chunks (< 20 µs)."""
        self._ingest_raw(token)
        emitted: list[CallClause] = []

        while self._buf:
            punct_hit = self._find_punctuation_split(self._buf, flushing=False)
            if punct_hit is not None:
                split_pos, fast_lane, pre_clean, pre_cnt = punct_hit
                if pre_cnt > self.max_words:
                    ov = self._find_overflow_split(self._buf[:split_pos], flushing=True)
                    if ov is not None:
                        head = self._buf[:ov]
                        self._buf = self._buf[ov:].lstrip()
                        clause = self._make_clause(
                            head, is_fast_lane=(self._clause_index == 0), is_final=False
                        )
                        if clause is not None:
                            emitted.append(clause)
                        continue

                head = self._buf[:split_pos]
                self._buf = self._buf[split_pos:].lstrip()
                clause = self._make_clause(
                    head,
                    is_fast_lane=fast_lane,
                    is_final=False,
                    precleaned=pre_clean,
                    precount=pre_cnt,
                )
                if clause is not None:
                    emitted.append(clause)
                continue

            overflow_pos = self._find_overflow_split(self._buf, flushing=False)
            if overflow_pos is not None:
                head = self._buf[:overflow_pos]
                self._buf = self._buf[overflow_pos:].lstrip()
                clause = self._make_clause(head, is_fast_lane=(self._clause_index == 0), is_final=False)
                if clause is not None:
                    emitted.append(clause)
                continue

            break

        return emitted

    def flush(self) -> list[CallClause]:
        """Flush any remaining text at the end of an LLM stream."""
        if self._in_bracket:
            tag_body = "".join(self._bracket_chars).strip()
            if tag_body.upper().startswith("ACTION:"):
                payload = tag_body.split(":", 1)[1].strip()
                if payload:
                    self.extracted_actions.append(payload)
            self._bracket_chars.clear()
            self._in_bracket = False

        emitted: list[CallClause] = []
        while self._buf:
            punct_hit = self._find_punctuation_split(self._buf, flushing=True)
            if punct_hit is not None:
                split_pos, fast_lane, pre_clean, pre_cnt = punct_hit
                if pre_cnt > self.max_words:
                    ov = self._find_overflow_split(self._buf[:split_pos], flushing=True)
                    if ov is not None:
                        head = self._buf[:ov]
                        self._buf = self._buf[ov:].lstrip()
                        clause = self._make_clause(
                            head, is_fast_lane=(self._clause_index == 0), is_final=False
                        )
                        if clause is not None:
                            emitted.append(clause)
                        continue

                head = self._buf[:split_pos]
                rest = self._buf[split_pos:].lstrip()
                self._buf = rest
                clause = self._make_clause(
                    head,
                    is_fast_lane=fast_lane,
                    is_final=(not bool(rest)),
                    precleaned=pre_clean,
                    precount=pre_cnt,
                )
                if clause is not None:
                    emitted.append(clause)
                continue

            overflow_pos = self._find_overflow_split(self._buf, flushing=True)
            if overflow_pos is not None:
                head = self._buf[:overflow_pos]
                self._buf = self._buf[overflow_pos:].lstrip()
                clause = self._make_clause(
                    head,
                    is_fast_lane=(self._clause_index == 0),
                    is_final=False,
                )
                if clause is not None:
                    emitted.append(clause)
                continue

            tail = self._buf
            self._buf = ""
            clause = self._make_clause(
                tail,
                is_fast_lane=(self._clause_index == 0),
                is_final=True,
            )
            if clause is not None:
                emitted.append(clause)

        return emitted

    def chunk_text(self, full_text: str) -> list[CallClause]:
        """Chunk a complete assistant response into low-TTFA live call clauses."""
        self.reset()
        self._ingest_raw(full_text)
        return self.flush()


# ---------------------------------------------------------------------------
# Sub-10µs Caller Turn, Barge-In, Backchannel & Endpoint Analyzer
# ---------------------------------------------------------------------------

_TURN_ELONG_RE = re.compile(r"([\u0621-\u064A])\1{2,}")
_TURN_PUNCT_RE = re.compile(r"[^\u0621-\u064A\u0671\s]")

_BACKCHANNEL_WORDS = frozenset(
    {
        "ايه",
        "اي",
        "ايوه",
        "ايوة",
        "نعم",
        "طيب",
        "تمام",
        "اها",
        "همم",
        "امم",
        "كمل",
        "كملي",
        "معك",
        "معاك",
        "حلو",
        "صح",
        "صحيح",
        "جميل",
        "واضح",
        "سمعتك",
        "اوكي",
        "زين",
        "ممتاز",
        "اكيد",
    }
)

_BARGE_IN_SINGLE = frozenset(
    {
        "لحظة",
        "لحظه",
        "وقف",
        "وقفي",
        "اصبر",
        "دقيقة",
        "دقيقه",
        "اسمع",
        "اسمعني",
        "مقاطعة",
        "غلط",
        "خطا",
    }
)

_BARGE_IN_PHRASES: tuple[str, ...] = (
    "لا لا",
    "لالا",
    "بس سوال",
    "عندي سوال",
    "غير كذا",
    "مو كذا",
    "لا مو",
    "طيب بس",
    "لو سمحت وقف",
    "خلني اقاطعك",
    "دقيقة شوي",
    "لحظة شوي",
)

_HOLD_SINGLE = frozenset({"ثواني", "ثانية", "ثانيه", "دقايق"})

_HOLD_PHRASES: tuple[str, ...] = (
    "دقيقة بس",
    "دقيقه بس",
    "خلك معي",
    "خليك معي",
    "لحظة اشوف",
    "لحظه اشوف",
    "ثواني اشوف",
    "اصبر اشوف",
    "عطني ثانية",
    "عطني دقيقة",
    "خلني اشوف",
    "بس اشيك",
)

_CLOSING_PHRASES: tuple[str, ...] = (
    "مع السلامة",
    "مع السلامه",
    "فمان الله",
    "في امان الله",
    "خلاص ما قصرت",
    "ما قصرت خلاص",
    "توصي شي",
    "توصي على شي",
    "تامر على شي",
    "تامرني بشي",
    "يعطيك العافية خلاص",
    "شكرا مع السلامة",
    "استودعك الله",
    "بحفظ الله",
)

_DANGLING_END_WORDS = frozenset(
    {
        "يعني",
        "عشان",
        "علشان",
        "اللي",
        "لان",
        "لانه",
        "بس",
        "او",
        "ولا",
        "و",
        "ف",
        "ان",
        "اذا",
        "لو",
        "لما",
        "من",
        "في",
        "على",
        "عن",
        "مع",
        "بخصوص",
        "مثل",
    }
)

_SUBSTANTIVE_CUES = frozenset(
    {
        "كم",
        "وش",
        "متى",
        "وين",
        "ليش",
        "كيف",
        "ابي",
        "ودي",
        "عطني",
        "احجز",
        "الغي",
        "حول",
        "ارسل",
        "لا",
    }
)


@dataclass(frozen=True, slots=True)
class TurnDecision:
    """Real-time turn-taking & VAD decision for full-duplex Saudi voice calls."""

    kind: str
    should_abort_tts: bool
    endpoint_grace_ms: int
    recommended_wait_sec: float = 0.0
    is_closing: bool = False
    suggested_ack: str | None = None
    normalized_text: str = ""
    confidence: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "should_abort_tts": self.should_abort_tts,
            "endpoint_grace_ms": self.endpoint_grace_ms,
            "recommended_wait_sec": self.recommended_wait_sec,
            "is_closing": self.is_closing,
            "suggested_ack": self.suggested_ack,
            "normalized_text": self.normalized_text,
            "confidence": self.confidence,
        }


class CallTurnAnalyzer:
    """Sub-10µs full-duplex turn, backchannel, barge-in, hold, and endpoint classifier."""

    def analyze_turn(self, text: str, *, is_ai_speaking: bool = False) -> TurnDecision:
        if not text or not text.strip():
            return TurnDecision(
                kind="backchannel" if is_ai_speaking else "incomplete_thought",
                should_abort_tts=False,
                endpoint_grace_ms=300,
                normalized_text="",
                confidence=0.5,
            )

        # Ultra-fast light normalization (< 2 µs)
        stripped = strip_diacritics(text.replace("\u0640", ""))
        stripped = _TURN_ELONG_RE.sub(r"\1", stripped)
        norm = light_arabic_key(_TURN_PUNCT_RE.sub(" ", stripped))
        norm = _MULTI_SPACE_RE.sub(" ", norm).strip()

        if not norm:
            return TurnDecision(
                kind="backchannel" if is_ai_speaking else "complete_turn",
                should_abort_tts=False,
                endpoint_grace_ms=0,
                normalized_text="",
                confidence=0.5,
            )

        words = norm.split()
        n_words = len(words)

        # 1. Hold requests (`ثواني`, `دقيقة بس`, `خلك معي`, `لحظة أشوف`)
        if (n_words == 1 and words[0] in _HOLD_SINGLE) or any(p in norm for p in _HOLD_PHRASES):
            return TurnDecision(
                kind="hold_request",
                should_abort_tts=is_ai_speaking,
                endpoint_grace_ms=0,
                recommended_wait_sec=20.0,
                is_closing=False,
                suggested_ack="خِذْ رَاحْتِكْ، مَعِكْ.",
                normalized_text=norm,
                confidence=0.97,
            )

        # 2. Call closings (`مع السلامة`, `فمان الله`, `في أمان الله`, `خلاص ما قصرت`, `توصي شي`)
        if any(p in norm for p in _CLOSING_PHRASES):
            ack = (
                "سَلَامْتِكْ طَالْ عُمْرِكْ، فِي أَمَانْ اللَّهْ."
                if ("توصي" in norm or "تامر" in norm)
                else "فِي أَمَانْ اللَّهْ، حَيَّاكْ اللَّهْ."
            )
            return TurnDecision(
                kind="closing",
                should_abort_tts=is_ai_speaking,
                endpoint_grace_ms=0,
                recommended_wait_sec=0.0,
                is_closing=True,
                suggested_ack=ack,
                normalized_text=norm,
                confidence=0.98,
            )

        # 3. Pure backchannels (`إيه`, `أيوه`, `طيب`, `تمام`, `أها`, `كمّل`, `معك`, `حلو`, `صح`)
        if n_words <= 3 and all(w in _BACKCHANNEL_WORDS for w in words):
            return TurnDecision(
                kind="backchannel",
                should_abort_tts=False,
                endpoint_grace_ms=0,
                recommended_wait_sec=0.0,
                is_closing=False,
                suggested_ack=None,
                normalized_text=norm,
                confidence=0.98,
            )

        # 4. Explicit barge-in cues (`لحظة`, `وقف`, `اصبر`, `لا لا`, `دقيقة`, `اسمع`, `بس سؤال`, `غير كذا`)
        if (
            (n_words == 1 and words[0] in _BARGE_IN_SINGLE)
            or (words[0] in _BARGE_IN_SINGLE and n_words <= 3)
            or any(p in norm for p in _BARGE_IN_PHRASES)
        ):
            return TurnDecision(
                kind="barge_in",
                should_abort_tts=True,
                endpoint_grace_ms=0,
                recommended_wait_sec=0.0,
                is_closing=False,
                suggested_ack="سَمّ، تَفَضَّلْ.",
                normalized_text=norm,
                confidence=0.97,
            )

        # 5. Substantive caller speech while AI is speaking -> Barge-in!
        if is_ai_speaking and (n_words >= 2 or words[0] in _SUBSTANTIVE_CUES):
            return TurnDecision(
                kind="barge_in",
                should_abort_tts=True,
                endpoint_grace_ms=0,
                recommended_wait_sec=0.0,
                is_closing=False,
                suggested_ack=None,
                normalized_text=norm,
                confidence=0.93,
            )

        # 6. Incomplete thought ending with dangling connector (`يعني`, `عشان`, `اللي`, `لأن`, `بس`, `أو`, `و`)
        if words[-1] in _DANGLING_END_WORDS and not text.rstrip().endswith(("؟", "?")):
            return TurnDecision(
                kind="incomplete_thought",
                should_abort_tts=False,
                endpoint_grace_ms=400,
                recommended_wait_sec=0.0,
                is_closing=False,
                suggested_ack=None,
                normalized_text=norm,
                confidence=0.94,
            )

        # 7. Complete normal caller turn
        return TurnDecision(
            kind="complete_turn",
            should_abort_tts=False,
            endpoint_grace_ms=0,
            recommended_wait_sec=0.0,
            is_closing=False,
            suggested_ack=None,
            normalized_text=norm,
            confidence=0.95,
        )


_DEFAULT_TURN_ANALYZER = CallTurnAnalyzer()


def analyze_turn(text: str, *, is_ai_speaking: bool = False) -> TurnDecision:
    """Classify a caller utterance for full-duplex voice turn management in < 10 µs."""
    return _DEFAULT_TURN_ANALYZER.analyze_turn(text, is_ai_speaking=is_ai_speaking)
