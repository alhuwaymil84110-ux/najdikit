"""Saudi Dialect Purity, MSA-Leakage & Voice-Call Suitability Scorer (< 30 µs).

Inspired by Saudi-Dialect-ALLaM alignment criteria and powered by a single-pass
compiled TokenAhoCorasick automaton.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .lexicon import NajdiLexicon, get_default_lexicon, strip_diacritics
from .packs import TokenAhoCorasick

_FORMAL_TANWEEN_RE = re.compile(r"[\u0621-\u064A]+[\u064C\u064D](?=$|[\s،.؟!,:؛)\]])")
_MD_CALL_PENALTY_RE = re.compile(r"(\*\*|__|```|(?:^|\n)\s*(?:[-*•]|\d+\.)\s+|#{1,6}\s+)")
_RAW_URL_UUID_RE = re.compile(
    r"https?://\S+|www\.\S+|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_PAUSE_SPLIT_RE = re.compile(r"[،,.;؛?!؟:\n]+")
_WORD_RE = re.compile(r"[\u0621-\u064A\u0671\u064B-\u065F\u0670]+|[A-Za-z0-9]+")

# Additional stiff/translated MSA markers beyond lexicon anti_confusion_rules
_EXTRA_STIFF_MSA: tuple[tuple[str, str], ...] = (
    ("سوف اقوم", "بـ / الحين أسوي"),
    ("ساقوم", "بـ / الحين أسوي"),
    ("سوف نقوم", "بنقوم / الحين نسوي"),
    ("سوف", "راح / بـ"),
    ("بامكانك", "يمديك / تقدر"),
    ("بامكانكم", "يمديكم / تقدرون"),
    ("الذي", "اللي"),
    ("التي", "اللي"),
    ("الذين", "اللي"),
    ("اللذان", "اللي"),
    ("اللتان", "اللي"),
    ("ماذا", "وش"),
    ("لماذا", "ليش"),
    ("حسنا", "طيب / تمام"),
    ("حيث ان", "لأن / عشان"),
    ("نظرا لان", "عشان"),
    ("علاوة على ذلك", "وفوق كذا"),
    ("فيما يتعلق", "بخصوص"),
    ("يتوجب عليك", "لازم / يحتاج"),
    ("ينبغي عليك", "يفضل / أحسن"),
    ("هل تود", "ودّك / تبي"),
    ("لقد قمت", "سويت / خلصت"),
    ("في الوقت الراهن", "الحين / حالياً"),
    ("لا داعي للقلق", "لا تشيل هم"),
    ("سأوافيك", "بعطيك خبر"),
    ("ساوافيك", "بعطيك خبر"),
    ("يرجى العلم", "ترا / للعلم"),
)

# Additional foreign dialectal markers (Levantine / Egyptian / Maghrebi / Iraqi)
_EXTRA_FOREIGN_DIALECT: tuple[tuple[str, str, str], ...] = (
    ("بدي", "levantine", "أبي / ودّي"),
    ("بدك", "levantine", "تبي / ودّك"),
    ("بده", "levantine", "يبي / ودّه"),
    ("بدنا", "levantine", "نبي / ودّنا"),
    ("هلق", "levantine", "الحين"),
    ("هلأ", "levantine", "الحين"),
    ("عشان هيك", "levantine", "عشان كذا"),
    ("هيك", "levantine", "كذا"),
    ("شو", "levantine", "وش"),
    ("منيح", "levantine", "زين / ممتاز"),
    ("كتير", "levantine", "مرّة / كثير"),
    ("امبارح", "levantine", "أمس"),
    ("مبارح", "levantine", "أمس"),
    ("عايز", "egyptian", "أبي / ودّي"),
    ("عاوز", "egyptian", "أبي / ودّي"),
    ("ازاي", "egyptian", "وشلون / كيف"),
    ("دلوقتي", "egyptian", "الحين"),
    ("لسه", "egyptian", "للحين / تو"),
    ("كويس اوي", "egyptian", "ممتاز مرّة"),
    ("بزاف", "maghrebi", "مرّة / واجد"),
    ("برشا", "maghrebi", "مرّة / واجد"),
    ("شنو", "other_dialect", "وش"),
)

# Core White Saudi & Urban Najdi positive markers
_CORE_SAUDI_ANCHORS: tuple[str, ...] = (
    "أبشر",
    "أبشر بعزك",
    "حياك الله",
    "يا هلا",
    "هلا والله",
    "الحين",
    "وش",
    "وشلون",
    "وشلونك",
    "يمديك",
    "يمدينا",
    "ودّي",
    "أبي",
    "تبي",
    "نبي",
    "يبون",
    "توّي",
    "شيكت",
    "أشيك",
    "بشيك",
    "نشيك",
    "عشان",
    "اللي",
    "هذي",
    "كذا",
    "زين",
    "تمام",
    "لاهنت",
    "ما قصر",
    "ما قصرت",
    "طال عمرك",
    "عزّك الله",
    "على خشمي",
    "سم",
    "ترا",
    "عساه",
    "لا تشيل هم",
    "الأمور طيبة",
    "خذ راحتك",
    "في أمان الله",
    "فمان الله",
    "توصي على شي",
    "توصي شي",
    "كل شي",
    "ولا شي",
    "ثواني",
    "عطني",
    "خلنا",
    "بعدين",
    "للحين",
    "مضبوط",
    "شغّال",
    "بيض الله وجهك",
    "ما عليك زود",
    "سلمت",
)


@dataclass(frozen=True, slots=True)
class _ScoreCue:
    kind: str  # "saudi" | "msa" | "foreign"
    origin: str
    replacement: str
    weight: float = 1.0


@dataclass(slots=True)
class DialectScoreResult:
    """Comprehensive Saudi dialect alignment & live-call suitability assessment."""

    saudi_alignment_score: float
    msa_leakage_score: float
    foreign_dialect_score: float
    call_suitability_score: float
    flagged_spans: list[dict[str, Any]] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    saudi_hits: int = 0
    word_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "saudi_alignment_score": self.saudi_alignment_score,
            "msa_leakage_score": self.msa_leakage_score,
            "foreign_dialect_score": self.foreign_dialect_score,
            "call_suitability_score": self.call_suitability_score,
            "flagged_spans": list(self.flagged_spans),
            "suggestions": list(self.suggestions),
            "saudi_hits": self.saudi_hits,
            "word_count": self.word_count,
        }


class NajdiDialectScorer:
    """Sub-30µs dialect purity & voice-call suitability evaluator."""

    def __init__(self, lexicon: NajdiLexicon | None = None) -> None:
        self.lexicon = lexicon or get_default_lexicon()
        self._automaton: TokenAhoCorasick[_ScoreCue] = TokenAhoCorasick(light_normalize=True)
        self._build_automaton()

    def _build_automaton(self) -> None:
        ac = self._automaton
        seen_patterns: set[str] = set()

        # 1. Anti-confusion rules from lexicon (highest diagnostic priority)
        for rule in self.lexicon.anti_confusion_rules:
            key = strip_diacritics(rule.non_saudi).strip()
            if not key or key in seen_patterns:
                continue
            seen_patterns.add(key)
            is_msa = "msa" in rule.origin.lower()
            cue = _ScoreCue(
                kind="msa" if is_msa else "foreign",
                origin=rule.origin,
                replacement=rule.saudi_white,
                weight=1.35 if " " in key else 1.0,
            )
            ac.add_phrase(rule.non_saudi, cue)

        # 2. Extra stiff MSA markers
        for phrase, repl in _EXTRA_STIFF_MSA:
            if phrase not in seen_patterns:
                seen_patterns.add(phrase)
                ac.add_phrase(
                    phrase,
                    _ScoreCue(
                        kind="msa",
                        origin="stiff_msa",
                        replacement=repl,
                        weight=1.25 if " " in phrase else 1.0,
                    ),
                )

        # 3. Extra foreign dialect markers
        for phrase, origin, repl in _EXTRA_FOREIGN_DIALECT:
            if phrase not in seen_patterns:
                seen_patterns.add(phrase)
                ac.add_phrase(
                    phrase,
                    _ScoreCue(
                        kind="foreign",
                        origin=origin,
                        replacement=repl,
                        weight=1.25 if " " in phrase else 1.0,
                    ),
                )

        # 4. Authentic White Saudi & Najdi anchors (core + telephony + conversational entries)
        for anchor in _CORE_SAUDI_ANCHORS:
            plain = strip_diacritics(anchor).strip()
            if plain and plain not in seen_patterns:
                seen_patterns.add(plain)
                ac.add_phrase(
                    anchor,
                    _ScoreCue(
                        kind="saudi",
                        origin="white_saudi",
                        replacement="",
                        weight=1.3 if " " in plain else 1.0,
                    ),
                )

        for entry in self.lexicon.entries:
            plain = strip_diacritics(entry.term).strip()
            if not plain or plain in seen_patterns:
                continue
            if entry.category in (
                "greetings_courtesy",
                "verbs_desire_cognition",
                "connectors_discourse",
                "idioms_expressions",
                "telephony_openers",
                "telephony_etiquette",
                "telephony_closers",
                "spontaneous_conversation",
            ):
                seen_patterns.add(plain)
                ac.add_phrase(
                    plain,
                    _ScoreCue(
                        kind="saudi",
                        origin=entry.pack_id,
                        replacement="",
                        weight=1.2 if " " in plain else 0.9,
                    ),
                )

        ac.compile()

    def score_dialect(self, text: str) -> DialectScoreResult:
        """Evaluate Saudi alignment, MSA leakage, foreign dialect drift, and call suitability (< 30 µs)."""
        if not text or not text.strip():
            return DialectScoreResult(
                saudi_alignment_score=0.0,
                msa_leakage_score=0.0,
                foreign_dialect_score=0.0,
                call_suitability_score=0.0,
            )

        words = _WORD_RE.findall(text)
        n_words = len(words)
        if n_words == 0:
            return DialectScoreResult(
                saudi_alignment_score=0.0,
                msa_leakage_score=0.0,
                foreign_dialect_score=0.0,
                call_suitability_score=0.0,
            )

        matches = self._automaton.select_non_overlapping(text)

        saudi_weight = 0.0
        saudi_hits = 0
        msa_weight = 0.0
        foreign_weight = 0.0
        flagged_spans: list[dict[str, Any]] = []
        suggestions: list[str] = []
        seen_suggestions: set[str] = set()

        for m in matches:
            cue = m.value
            if cue.kind == "saudi":
                saudi_hits += 1
                saudi_weight += cue.weight
            elif cue.kind == "msa":
                msa_weight += cue.weight
                flagged_spans.append(
                    {
                        "category": "msa_leakage",
                        "origin": cue.origin,
                        "text": m.matched_text,
                        "span": [m.start_char, m.end_char],
                        "replacement": cue.replacement,
                    }
                )
                if cue.replacement and m.matched_text not in seen_suggestions:
                    seen_suggestions.add(m.matched_text)
                    suggestions.append(f'استبدل "{m.matched_text}" بـ "{cue.replacement}"')
            elif cue.kind == "foreign":
                foreign_weight += cue.weight
                flagged_spans.append(
                    {
                        "category": "foreign_dialect",
                        "origin": cue.origin,
                        "text": m.matched_text,
                        "span": [m.start_char, m.end_char],
                        "replacement": cue.replacement,
                    }
                )
                if cue.replacement and m.matched_text not in seen_suggestions:
                    seen_suggestions.add(m.matched_text)
                    suggestions.append(f'استبدل "{m.matched_text}" ({cue.origin}) بـ "{cue.replacement}"')

        # Check formal MSA case tanween (ٌ / ٍ) if present
        if "\u064C" in text or "\u064D" in text:
            for tm in _FORMAL_TANWEEN_RE.finditer(text):
                msa_weight += 1.0
                flagged_spans.append(
                    {
                        "category": "msa_leakage",
                        "origin": "formal_tanween",
                        "text": tm.group(0),
                        "span": [tm.start(), tm.end()],
                        "replacement": strip_diacritics(tm.group(0)),
                    }
                )
            suggestions.append("احذف تنوين الإعراب الفصيح (ٌ / ٍ) والتزم بالتسكين الوقفي الطبيعي.")

        # Normalize scores to 0..100
        denom = max(3.0, n_words * 0.45)
        msa_score = min(100.0, round((msa_weight / denom) * 100.0, 1))
        foreign_score = min(100.0, round((foreign_weight / denom) * 100.0, 1))

        # Neutral Arabic words shared across White Saudi & MSA get a healthy baseline (72.0)
        # boosted by positive Saudi anchors and reduced by MSA stiffness or foreign dialect drift.
        anchor_boost = min(28.0, (saudi_weight / max(2.0, n_words * 0.30)) * 28.0)
        drift_penalty = (msa_score * 0.68) + (foreign_score * 0.85)
        saudi_score = max(0.0, min(100.0, round(72.0 + anchor_boost - drift_penalty, 1)))

        # Call suitability score: starts from Saudi alignment + checks spoken breath/formatting hygiene
        call_penalty = 0.0
        if any(c in text for c in "*_`#\n-") and _MD_CALL_PENALTY_RE.search(text):
            call_penalty += 25.0
            suggestions.append("أزل رموز التنسيق (Markdown) والقوائم النقطية لتناسب النطق الصوتي المباشر.")

        if ("http" in text or "www." in text or "-" in text) and _RAW_URL_UUID_RE.search(text):
            call_penalty += 25.0
            suggestions.append("تجنب تضمين الروابط أو المعرفات الطويلة في النص المنطوق للمكالمة.")

        clauses = [seg.strip() for seg in _PAUSE_SPLIT_RE.split(text) if seg.strip()]
        max_clause_words = max((len(seg.split()) for seg in clauses), default=n_words)
        if max_clause_words > 18:
            call_penalty += min(25.0, (max_clause_words - 18) * 2.5)
            suggestions.append("قسّم الجمل الطويلة بعلامات وقف قصيرة (،) لتقليل زمن الاستجابة الصوتي.")

        call_score = max(0.0, min(100.0, round(saudi_score - call_penalty, 1)))

        return DialectScoreResult(
            saudi_alignment_score=saudi_score,
            msa_leakage_score=msa_score,
            foreign_dialect_score=foreign_score,
            call_suitability_score=call_score,
            flagged_spans=flagged_spans,
            suggestions=suggestions,
            saudi_hits=saudi_hits,
            word_count=n_words,
        )


_DEFAULT_SCORER: NajdiDialectScorer | None = None


def score_dialect(text: str, *, lexicon: NajdiLexicon | None = None) -> DialectScoreResult:
    """Evaluate Saudi dialect alignment, MSA leakage, foreign drift & call suitability in < 30 µs."""
    global _DEFAULT_SCORER
    if lexicon is not None:
        return NajdiDialectScorer(lexicon=lexicon).score_dialect(text)
    if _DEFAULT_SCORER is None:
        _DEFAULT_SCORER = NajdiDialectScorer()
    return _DEFAULT_SCORER.score_dialect(text)
