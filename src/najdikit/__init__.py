"""NajdiKit: Modern White Saudi & Urban Najdi Computational Linguistics & Voice AI Toolkit."""

from __future__ import annotations

from .lexicon import (
    AntiConfusionRule,
    LexiconEntry,
    NajdiLexicon,
    get_default_lexicon,
    light_arabic_key,
    strip_diacritics,
)
from .neural_mlx import (
    DisambiguationResult,
    OptionalMLXDisambiguator,
    disambiguate_speakable,
    is_mlx_available,
)
from .phonetics import (
    NajdiPhoneticsEngine,
    PhoneticResult,
    number_to_saudi_words,
    to_speakable,
)
from .prompt_kit import (
    SUPPORTED_PROFILES,
    GroundingBundle,
    NajdiPromptKit,
    build_system_grounding,
)
from .stt_enhancer import (
    DictionaryPersonalAdapter,
    NajdiSTTEnhancer,
    PersonalIntentAdapter,
    TranscriptCleanResult,
    build_stt_prompt,
    postprocess_transcript,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "AntiConfusionRule",
    "DictionaryPersonalAdapter",
    "DisambiguationResult",
    "GroundingBundle",
    "LexiconEntry",
    "NajdiLexicon",
    "NajdiPhoneticsEngine",
    "NajdiPromptKit",
    "NajdiSTTEnhancer",
    "OptionalMLXDisambiguator",
    "PersonalIntentAdapter",
    "PhoneticResult",
    "SUPPORTED_PROFILES",
    "TranscriptCleanResult",
    "build_stt_prompt",
    "build_system_grounding",
    "disambiguate_speakable",
    "get_default_lexicon",
    "is_mlx_available",
    "light_arabic_key",
    "number_to_saudi_words",
    "postprocess_transcript",
    "strip_diacritics",
    "to_speakable",
]
