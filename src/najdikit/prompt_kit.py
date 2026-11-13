"""Positive System Grounding Builder for authentic Modern White Saudi LLM interaction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from .lexicon import NajdiLexicon, get_default_lexicon

SUPPORTED_PROFILES: tuple[str, ...] = (
    "modern_white",
    "executive",
    "voice_assistant",
    "technical",
)

_PROFILE_DIRECTIVES: dict[str, dict[str, str]] = {
    "modern_white": {
        "title": "الأسلوب السعودي الأبيض العصري (Modern White Saudi)",
        "tone": (
            "تحدث بلهجة سعودية بيضاء عصرية وواضحة تجمع بين اللباقة العملية والبساطة الطبيعية، "
            "مفهومة ومألوفة لكل المتحدثين بدون تصنع أو مبالغة محلية مغلقة وبدون فصحى مترجمة متكلفة."
        ),
        "anchors": (
            "استخدم الروابط والأفعال الطبيعية بسلاسة في موضعها: "
            "(أبشر، حياك الله، الحين، وش، عشان، اللي، هذي، يمديك، ودّي، توّي، شيكت، تمام، زين، لاهنت، ما قصرت)."
        ),
        "cadence": (
            "ابدأ بالخلاصة المفيدة مباشرة، واجعل الجمل رشيقة ومترابطة كمتحدث سعودي متمكن وودود."
        ),
    },
    "executive": {
        "title": "الأسلوب التنفيذي الرصين (Executive White Saudi)",
        "tone": (
            "اعتمد لهجة سعودية بيضاء مهنية رصينة تجمع بين الوقار العملي والإيجاز الحاسم، "
            "مناسبة للاجتماعات والتقارير واتخاذ القرار."
        ),
        "anchors": (
            "استخدم عبارات مهنية راقية باعتدال: "
            "(سم، أبشر، طال عمرك، اعتمدنا، الزبدة، الأمور طيبة، شيكت على المؤشرات، لاهنت، شاكر ومقدر)."
        ),
        "cadence": (
            "قدم النتيجة أو التوصية في أول سطر دون مقدمات إنشائية أو حشو."
        ),
    },
    "voice_assistant": {
        "title": "أسلوب المساعد الصوتي الحي (Real-Time Saudi Voice Persona)",
        "tone": (
            "تحدث بجمل قصيرة ومنطوقة بلهجة سعودية بيضاء ونجدية حضرية سلسة ومريحة للأذن، "
            "مكتوبة كما تُنطق في الحوار الطبيعي دون رموز تنسيق أو قوائم نقطية طويلة."
        ),
        "anchors": (
            "اكتب الأرقام والأوقات بصيغة منطوقة طبيعية عند الحاجة (مية وخمسين ريال، ثلاث ونص العصر)، "
            "واستخدم ردوداً صوتية دافئة وموجزة مثل: (أبشر، تم، عطني ثانية أشيك لك، كل شي جاهز الحين)."
        ),
        "cadence": (
            "اجعل كل رد بين جملة إلى ثلاث جمل قصيرة يسهل نطقها في نفس واحد واضح."
        ),
    },
    "technical": {
        "title": "أسلوب المهندس التقني (Technical White Saudi)",
        "tone": (
            "تحدث كمهندس برمجيات سعودي محترف يشرح الحلول التقنية بوضوح مباشر وبلهجة بيضاء عصرية."
        ),
        "anchors": (
            "ادمج المصطلحات البرمجية الدقيقة في سياق عربي سعودي طبيعي: "
            "(شيكت على السجلات، الكود نظيف، نرفع التحديث، يمدينا نختصر الاستعلام، الأداء ممتاز)."
        ),
        "cadence": (
            "شخّص السبب الجذري أولاً ثم اعرض الحل العملي والخطوة التالية بوضوح."
        ),
    },
}

_FEW_SHOT_CALIBRATION: tuple[tuple[str, str], ...] = (
    (
        "سوف أقوم بالتحقق من الأمر حالاً وسأوافيك بالتفاصيل.",
        "أبشر، بشيك على الموضوع الحين وأعطيك خبر بالتفاصيل.",
    ),
    (
        "هل بإمكانك تزويدي بالملف لكي نتمكن من البدء؟",
        "يمديك ترسل لي الملف لاهنت عشان نبدأ؟",
    ),
    (
        "لا داعي للقلق، كل شيء يسير على ما يرام حتى هذه اللحظة.",
        "لا تشيل هم، الأمور طيبة وكل شي ماشي تمام للحين.",
    ),
)


@dataclass(frozen=True, slots=True)
class GroundingBundle:
    """Generated system prompt grounding block with metadata."""

    profile: str
    compact: bool
    system_prompt: str
    estimated_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "compact": self.compact,
            "system_prompt": self.system_prompt,
            "estimated_tokens": self.estimated_tokens,
        }


class NajdiPromptKit:
    """Builds affirmative, token-efficient LLM grounding prompts and harmonizes dialect drift."""

    def __init__(self, lexicon: NajdiLexicon | None = None) -> None:
        self.lexicon = lexicon or get_default_lexicon()
        self._drift_patterns: list[tuple[re.Pattern[str], str]] = [
            (
                re.compile(rf"(?<!\w){re.escape(rule.non_saudi)}(?!\w)"),
                rule.saudi_white,
            )
            for rule in sorted(
                self.lexicon.anti_confusion_rules,
                key=lambda r: len(r.non_saudi),
                reverse=True,
            )
        ]

    def build_grounding(
        self,
        *,
        profile: str = "modern_white",
        compact: bool = False,
        domain_terms: Sequence[str] | None = None,
        include_examples: bool = True,
    ) -> GroundingBundle:
        """Build a positive system grounding prompt for Claude, GPT, Gemini, Qwen, or Grok."""
        selected = profile if profile in _PROFILE_DIRECTIVES else "modern_white"
        spec = _PROFILE_DIRECTIVES[selected]

        lines: list[str] = [
            f"## الهوية اللغوية: {spec['title']}",
            f"- النبرة: {spec['tone']}",
            f"- الإيقاع: {spec['cadence']}",
        ]

        if not compact:
            lines.append(f"- المرتكزات اللغوية: {spec['anchors']}")
            lines.append(
                "- الاتساق اللهجي: التزم حصرياً بالبيضاء السعودية المتزنة، واستخدم مقابلاتها الأصيلة "
                "(أبي/ودّي، الحين، وش، عشان، كذا، زين، للحين، أمس) بثبات طبيعي طوال الحوار."
            )

        clean_terms = [t.strip() for t in (domain_terms or ()) if t and t.strip()]
        if clean_terms:
            lines.append(f"- مصطلحات المجال المعتمدة: {'، '.join(clean_terms)}.")

        if include_examples and not compact:
            lines.append("- أمثلة معايرة الأسلوب:")
            for stiff, natural in _FEW_SHOT_CALIBRATION:
                lines.append(f'  • بدل: "{stiff}" ← قل: "{natural}"')

        prompt_text = "\n".join(lines)
        # Heuristic token estimate for Arabic/mixed UTF-8 (~3.4 chars/token)
        est_tokens = max(1, round(len(prompt_text) / 3.4))
        return GroundingBundle(
            profile=selected,
            compact=compact,
            system_prompt=prompt_text,
            estimated_tokens=est_tokens,
        )

    def harmonize_text(self, text: str) -> dict[str, Any]:
        """Replace common non-Saudi / stiff-MSA drift phrases with natural White Saudi equivalents."""
        out = text
        replacements = 0
        for pattern, target in self._drift_patterns:
            out, n = pattern.subn(target, out)
            replacements += n
        return {
            "original": text,
            "harmonized": out,
            "replacements": replacements,
        }


_DEFAULT_KIT: NajdiPromptKit | None = None


def build_system_grounding(
    *,
    profile: str = "modern_white",
    compact: bool = False,
    domain_terms: Sequence[str] | None = None,
    include_examples: bool = True,
) -> GroundingBundle:
    """Convenience function to build a positive White Saudi system grounding block."""
    global _DEFAULT_KIT
    if _DEFAULT_KIT is None:
        _DEFAULT_KIT = NajdiPromptKit()
    return _DEFAULT_KIT.build_grounding(
        profile=profile,
        compact=compact,
        domain_terms=domain_terms,
        include_examples=include_examples,
    )
