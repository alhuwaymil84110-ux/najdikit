"""Optional Local MLX / OpenAI-Compatible Contextual Disambiguator with instant deterministic fallback."""

from __future__ import annotations

import importlib.util
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .phonetics import NajdiPhoneticsEngine, to_speakable


def is_mlx_available() -> bool:
    """Return True if Apple Silicon `mlx` is installed in the current Python environment."""
    return importlib.util.find_spec("mlx") is not None


# Fast contextual window heuristics used when no local LLM server is active
_HOMOGRAPH_CONTEXT_RULES: tuple[tuple[str, tuple[str, ...], str, str], ...] = (
    # (ambiguous_token, following_or_preceding_cues, contextual_vocalized, default_vocalized)
    ("شغل", ("السيرفر", "الفحص", "السكربت", "النظام", "الجهاز", "المحرك"), "شَغِّلْ", "شُغْلْ"),
    ("وده", ("لقسم", "للمكتب", "للعميل", "للفرع", "معك"), "وَدَّهْ", "وِدَّهْ"),
    ("ودّه", ("لقسم", "للمكتب", "للعميل", "للفرع", "معك"), "وَدَّهْ", "وِدَّهْ"),
    ("قفل", ("التذكرة", "الباب", "الملف", "الصفحة", "الحساب", "الطلب"), "قَفِّلْ", "قُفْلْ"),
    ("حول", ("المبلغ", "الرصيد", "الحوالة", "النص", "الصيغة"), "حَوِّلْ", "حَوْلْ"),
)


@dataclass(slots=True)
class DisambiguationResult:
    """Output of contextual disambiguation with provenance metadata."""

    original: str
    speakable: str
    backend: str  # "local_mlx_http" | "deterministic_fallback"
    used_fallback: bool
    mlx_installed: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "speakable": self.speakable,
            "backend": self.backend,
            "used_fallback": self.used_fallback,
            "mlx_installed": self.mlx_installed,
            "error": self.error,
        }


class OptionalMLXDisambiguator:
    """Disambiguates homographs via an OpenAI-compatible local MLX server or deterministic fallback."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout_sec: float | None = None,
        phonetics: NajdiPhoneticsEngine | None = None,
    ) -> None:
        env_url = os.environ.get("NAJDIKIT_LOCAL_LLM_URL", "").strip()
        self.base_url = (base_url if base_url is not None else env_url).rstrip("/")
        self.model = (
            model
            or os.environ.get("NAJDIKIT_LOCAL_LLM_MODEL", "").strip()
            or "mlx-community/Qwen2.5-7B-Instruct-4bit"
        )
        env_timeout = os.environ.get("NAJDIKIT_LOCAL_LLM_TIMEOUT_SEC", "").strip()
        self.timeout_sec = (
            timeout_sec
            if timeout_sec is not None
            else (float(env_timeout) if env_timeout else 1.5)
        )
        self.phonetics = phonetics or NajdiPhoneticsEngine()

    def _deterministic_disambiguate(self, text: str) -> str:
        tokens = text.split()
        adjusted: list[str] = list(tokens)
        for idx, tok in enumerate(tokens):
            neighbor_window = set(tokens[max(0, idx - 2) : min(len(tokens), idx + 3)])
            for target, cues, cue_voc, _default_voc in _HOMOGRAPH_CONTEXT_RULES:
                if tok == target and any(cue in neighbor_window for cue in cues):
                    adjusted[idx] = cue_voc
                    break
        rejoined = " ".join(adjusted)
        return self.phonetics.prepare_tts(rejoined).speakable

    def disambiguate(
        self,
        text: str,
        *,
        prefer_neural: bool = True,
    ) -> DisambiguationResult:
        """Disambiguate & vocalize text, falling back deterministically if local MLX is unavailable."""
        mlx_flag = is_mlx_available()
        if not text or not text.strip():
            return DisambiguationResult(
                original=text,
                speakable=text,
                backend="deterministic_fallback",
                used_fallback=False,
                mlx_installed=mlx_flag,
            )

        if not prefer_neural or not self.base_url:
            return DisambiguationResult(
                original=text,
                speakable=self._deterministic_disambiguate(text),
                backend="deterministic_fallback",
                used_fallback=bool(prefer_neural and not self.base_url),
                mlx_installed=mlx_flag,
            )

        endpoint = (
            f"{self.base_url}/chat/completions"
            if not self.base_url.endswith("/chat/completions")
            else self.base_url
        )
        baseline = self._deterministic_disambiguate(text)
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": max(128, len(text) * 3),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "أنت مدقق صوتي متخصص في اللهجة السعودية البيضاء والنجدية الحضرية. "
                        "فك اللبس السياقي للكلمات المشتركة لفظياً مع الحفاظ على كسرة حرف المضارعة "
                        "وتسكين أواخر الكلمات الوقفي، وأرجع النص المشكول فقط دون أي شرح."
                    ),
                },
                {
                    "role": "user",
                    "content": f"النص الأصلي: {text}\nالتشكيل المبدئي: {baseline}",
                },
            ],
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            choices = body.get("choices") or []
            content = (
                choices[0].get("message", {}).get("content", "").strip()
                if choices
                else ""
            )
            if content:
                return DisambiguationResult(
                    original=text,
                    speakable=content,
                    backend="local_mlx_http",
                    used_fallback=False,
                    mlx_installed=mlx_flag,
                )
            return DisambiguationResult(
                original=text,
                speakable=baseline,
                backend="deterministic_fallback",
                used_fallback=True,
                mlx_installed=mlx_flag,
                error="Empty completion from local MLX endpoint",
            )
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
            return DisambiguationResult(
                original=text,
                speakable=baseline,
                backend="deterministic_fallback",
                used_fallback=True,
                mlx_installed=mlx_flag,
                error=type(exc).__name__,
            )


def disambiguate_speakable(
    text: str,
    *,
    base_url: str | None = None,
) -> DisambiguationResult:
    """Convenience wrapper for optional local MLX disambiguation with safe fallback."""
    if not base_url and not os.environ.get("NAJDIKIT_LOCAL_LLM_URL"):
        return DisambiguationResult(
            original=text,
            speakable=to_speakable(text),
            backend="deterministic_fallback",
            used_fallback=True,
            mlx_installed=is_mlx_available(),
        )
    return OptionalMLXDisambiguator(base_url=base_url).disambiguate(text)
