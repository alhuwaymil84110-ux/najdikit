"""Najdi STT Prompt Builder, Anti-Hallucination Filter, and Transcript Post-Processor."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from .lexicon import NajdiLexicon, get_default_lexicon, strip_diacritics
from .morphology import NajdiMorphAnalyzer, get_default_morph_analyzer

_ELONGATION_RE = re.compile(r"([\u0621-\u064A])\1{2,}")
_TATWEEL_RE = re.compile(r"\u0640+")
_REPEAT_PHRASE_RE = re.compile(r"\b(\S+(?:\s+\S+){0,2})(?:\s+\1){3,}\b")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_PUNCT_SPACE_RE = re.compile(r"\s+([،.؟!,:؛])")
_AR_WORD_RE = re.compile(r"([\u0621-\u064A\u0671\u064B-\u065F\u0670]+)")

_DOMAIN_SEEDS: dict[str, str] = {
    "general": (
        "حياك الله، أبشر، شيكت على الموضوع الحين وكل شي تمام ولله الحمد. "
        "وش ودّك نسوي بعدين؟ يمديك ترسل التفاصيل لاهنت عشان نعتمدها إن شاء الله."
    ),
    "tech": (
        "شيكت على السيرفر واللوق في الداشبورد، الكود نظيف والاختبارات شغّالة بدون أخطاء. "
        "يمدينا نرفع التحديث ونعتمد طلب الدمج الحين عبر API و GitHub."
    ),
    "finance": (
        "راجعنا الفاتورة وكشف الحساب والتسوية المالية بالريال والهللة، شامله القيمة المضافة. "
        "الرصيد مضبوط والحوالة البنكية وصلت عبر مدى وسداد."
    ),
    "customer_care": (
        "يا هلا ومسهلا، أبشر بعزك ولا تشيل هم. الحين أشيك لك على الطلب وأعطيك خبر أول بأول، "
        "وإذا احتجت أي خدمة إضافية آمرني طال عمرك."
    ),
    "telephony": (
        "يا هلا والله حياك الله، صوتك واضح طال عمرك. سم، ثواني أشيك لك على الموضوع الحين، "
        "خذ راحتك وأنا معك على الخط، إذا توصي على شي آمرني في أمان الله."
    ),
}


@runtime_checkable
class PersonalIntentAdapter(Protocol):
    """Protocol allowing external private systems to inject user-specific vocabulary/intents safely."""

    def adapt_transcript(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | dict[str, Any]:
        """Transform a normalized transcript or attach custom intent metadata."""
        ...


AdapterHook = PersonalIntentAdapter | Callable[[str], str]


class DictionaryPersonalAdapter:
    """Ready-to-use runtime adapter backed by an external dictionary or JSON file."""

    def __init__(
        self,
        replacements: Mapping[str, str] | None = None,
        intent_aliases: Mapping[str, str] | None = None,
    ) -> None:
        self.replacements: dict[str, str] = dict(replacements or {})
        self.intent_aliases: dict[str, str] = dict(intent_aliases or {})
        # Sort longest phrases first for deterministic replacement
        self._sorted_pairs: list[tuple[re.Pattern[str], str]] = [
            (re.compile(rf"(?<!\w){re.escape(src)}(?!\w)"), dst)
            for src, dst in sorted(self.replacements.items(), key=lambda kv: len(kv[0]), reverse=True)
            if src
        ]

    @classmethod
    def from_json_file(cls, path: str | os.PathLike[str]) -> "DictionaryPersonalAdapter":
        resolved = Path(os.path.expanduser(str(path)))
        with resolved.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if "replacements" in data or "intent_aliases" in data:
            return cls(
                replacements=data.get("replacements", {}),
                intent_aliases=data.get("intent_aliases", {}),
            )
        return cls(replacements=data)

    def adapt_transcript(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        out = text
        applied = 0
        for pattern, dst in self._sorted_pairs:
            out, count = pattern.subn(dst, out)
            applied += count

        matched_intent: str | None = None
        for trigger, intent_name in self.intent_aliases.items():
            if trigger and trigger in out:
                matched_intent = intent_name
                break

        return {
            "text": out,
            "adapter_replacements": applied,
            "matched_intent": matched_intent,
            "metadata": metadata or {},
        }


@dataclass(slots=True)
class TranscriptCleanResult:
    """Result returned by NajdiSTTEnhancer.postprocess_transcript."""

    raw: str
    text: str
    was_hallucination: bool = False
    removed_hallucinations: list[str] = field(default_factory=list)
    normalized_count: int = 0
    matched_intent: str | None = None
    adapter_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "text": self.text,
            "was_hallucination": self.was_hallucination,
            "removed_hallucinations": list(self.removed_hallucinations),
            "normalized_count": self.normalized_count,
            "matched_intent": self.matched_intent,
            "adapter_applied": self.adapter_applied,
        }


class NajdiSTTEnhancer:
    """Contextual prompt generator and deterministic transcript cleaner for Saudi ASR."""

    def __init__(
        self,
        lexicon: NajdiLexicon | None = None,
        *,
        personal_adapter_hook: AdapterHook | None = None,
        morph: NajdiMorphAnalyzer | None = None,
    ) -> None:
        self.lexicon = lexicon or get_default_lexicon()
        self.morph = morph or (
            NajdiMorphAnalyzer(lexicon=self.lexicon)
            if lexicon is not None
            else get_default_morph_analyzer()
        )
        self.personal_adapter_hook = personal_adapter_hook

        env_adapter_path = os.environ.get("NAJDIKIT_PERSONAL_ADAPTER_JSON", "").strip()
        if self.personal_adapter_hook is None and env_adapter_path:
            self.personal_adapter_hook = DictionaryPersonalAdapter.from_json_file(env_adapter_path)

        # Pre-compile hallucination patterns (case-insensitive, punctuation-tolerant)
        self._hallucination_patterns: list[tuple[str, re.Pattern[str]]] = [
            (
                phrase,
                re.compile(
                    rf"(?:^|[\s،.؟!,:;\-–—]){re.escape(phrase)}(?:$|[\s،.؟!,:;\-–—])",
                    re.IGNORECASE,
                ),
            )
            for phrase in sorted(self.lexicon.hallucination_blacklist, key=len, reverse=True)
        ]

        # Multi-word orthographic canonicalization rules
        self._multiword_fixes: tuple[tuple[re.Pattern[str], str], ...] = (
            (re.compile(r"\b(?:ان شاء الله|انشاءالله|إنشالله|انشالله|إن شاءالله)\b"), "إن شاء الله"),
            (re.compile(r"\b(?:ماشاء الله|ماشاءالله|مشاءالله|ماشالله)\b"), "ما شاء الله"),
            (re.compile(r"\bجزاك الله خيرا\b"), "جزاك الله خير"),
            (re.compile(r"\bلا هنت\b"), "لاهنت"),
            (re.compile(r"\bوش لونك\b"), "وشلونك"),
            (re.compile(r"\bوش لون\b"), "وشلون"),
            (re.compile(r"\b(?P<p>و?)(?:كل شيء|كلشئ)\b"), r"\g<p>كل شي"),
            (re.compile(r"\bولا شيء\b"), "ولا شي"),
            (re.compile(r"\bأول شيء\b"), "أول شي"),
            (re.compile(r"\bأهم شيء\b"), "أهم شي"),
            (re.compile(r"\bفي المائة\b"), "بالمية"),
            (re.compile(r"\bفي المئة\b"), "بالمية"),
        )

    def build_stt_prompt(
        self,
        *,
        domain: str = "general",
        extra_terms: Sequence[str] | None = None,
        previous_transcript: str | None = None,
        max_chars: int = 420,
    ) -> dict[str, Any]:
        """Build `initial_prompt` and `carry_prompt` optimized for Whisper / Scribe ASR.

        Uses natural in-style Modern White Saudi text (avoiding imperative instructions
        that cause Whisper prompt-echo hallucinations) plus optional sliding carry context.
        """
        seed = _DOMAIN_SEEDS.get(domain, _DOMAIN_SEEDS["general"])
        terms_clean = [t.strip() for t in (extra_terms or ()) if t and t.strip()]

        parts: list[str] = [seed]
        if terms_clean:
            parts.append("مصطلحات السياق: " + "، ".join(terms_clean) + ".")

        initial_prompt = " ".join(parts).strip()
        if len(initial_prompt) > max_chars:
            initial_prompt = initial_prompt[:max_chars].rsplit(" ", 1)[0]

        carry_prompt = initial_prompt
        if previous_transcript and previous_transcript.strip():
            cleaned_prev = self.postprocess_transcript(previous_transcript).text
            if cleaned_prev:
                tail_budget = max(120, max_chars - len(seed) - 10)
                tail = cleaned_prev[-tail_budget:]
                if len(cleaned_prev) > tail_budget and " " in tail:
                    tail = tail.split(" ", 1)[1]
                carry_prompt = f"{seed} {tail}".strip()
                if len(carry_prompt) > max_chars:
                    carry_prompt = carry_prompt[-max_chars:]

        keyterms = [
            "أبشر",
            "الحين",
            "وشلون",
            "يمديك",
            "ودّي",
            "توّي",
            "شيكت",
            "لاهنت",
            "عشان",
            "اللي",
            "هذي",
            "ما قصر",
            *terms_clean,
        ]
        return {
            "domain": domain,
            "initial_prompt": initial_prompt,
            "carry_prompt": carry_prompt,
            "keyterms": list(dict.fromkeys(keyterms)),
        }

    def postprocess_transcript(
        self,
        text: str,
        *,
        personal_adapter_hook: AdapterHook | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TranscriptCleanResult:
        """Clean ASR hallucinations, normalize White Saudi spelling, and invoke optional adapter."""
        if not text or not text.strip():
            return TranscriptCleanResult(raw=text, text="")

        work = _TATWEEL_RE.sub("", text).strip()
        removed_hallucinations: list[str] = []

        # 1. Remove known silence/subtitle hallucinations
        for label, pattern in self._hallucination_patterns:
            if pattern.search(work):
                removed_hallucinations.append(label)
                work = pattern.sub(" ", work).strip()

        # Compress pathological ASR repetition loops on silence (4+ identical n-grams)
        work = _REPEAT_PHRASE_RE.sub(r"\1", work)
        work = re.sub(r"^[\s،.؟!,:;\-–—]+|[\s،.؟!,:;\-–—]+$", lambda m: m.group(0).strip() if m.group(0).strip() in "؟!." else "", work).strip()

        if not work and removed_hallucinations:
            return TranscriptCleanResult(
                raw=text,
                text="",
                was_hallucination=True,
                removed_hallucinations=removed_hallucinations,
            )

        # 2. Compress vocal elongations ("هلااااا" -> "هلا", "مررررة" -> "مرّة")
        work = _ELONGATION_RE.sub(r"\1", work)

        # 3. Single-pass O(N) multi-word orthographic canonicalization via compiled Aho-Corasick
        work, normalized_count = self.lexicon.ortho_phrase_automaton.replace_all(work)

        # 4. Token-level variant & clitic-inflected stem normalization
        morph = self.morph

        def _norm_word(match: re.Match[str]) -> str:
            nonlocal normalized_count
            tok = match.group(1)
            mapped = morph.normalize_inflected(tok)
            if mapped != tok:
                normalized_count += 1
                return mapped
            return tok

        work = _AR_WORD_RE.sub(_norm_word, work)
        work = _PUNCT_SPACE_RE.sub(r"\1", _MULTI_SPACE_RE.sub(" ", work)).strip()

        # 5. Optional privacy-isolated PersonalIntentAdapter hook
        hook = personal_adapter_hook or self.personal_adapter_hook
        matched_intent: str | None = None
        adapter_applied = False
        if hook is not None and work:
            adapter_applied = True
            if isinstance(hook, PersonalIntentAdapter):
                adapted = hook.adapt_transcript(work, metadata=metadata)
            else:
                adapted = hook(work)

            if isinstance(adapted, dict):
                work = str(adapted.get("text", work)).strip()
                matched_intent = adapted.get("matched_intent")
            else:
                work = str(adapted).strip()

        return TranscriptCleanResult(
            raw=text,
            text=work,
            was_hallucination=bool(removed_hallucinations and not work),
            removed_hallucinations=removed_hallucinations,
            normalized_count=normalized_count,
            matched_intent=matched_intent,
            adapter_applied=adapter_applied,
        )


_DEFAULT_STT: NajdiSTTEnhancer | None = None


def postprocess_transcript(
    text: str,
    *,
    personal_adapter_hook: AdapterHook | None = None,
    metadata: dict[str, Any] | None = None,
) -> TranscriptCleanResult:
    """Convenience function to clean and normalize an STT transcript in < 0.3ms."""
    global _DEFAULT_STT
    if _DEFAULT_STT is None:
        _DEFAULT_STT = NajdiSTTEnhancer()
    return _DEFAULT_STT.postprocess_transcript(
        text,
        personal_adapter_hook=personal_adapter_hook,
        metadata=metadata,
    )


def build_stt_prompt(
    *,
    domain: str = "general",
    extra_terms: Sequence[str] | None = None,
    previous_transcript: str | None = None,
) -> dict[str, Any]:
    """Convenience function to build Whisper / Scribe initial & carry prompts."""
    global _DEFAULT_STT
    if _DEFAULT_STT is None:
        _DEFAULT_STT = NajdiSTTEnhancer()
    return _DEFAULT_STT.build_stt_prompt(
        domain=domain,
        extra_terms=extra_terms,
        previous_transcript=previous_transcript,
    )
