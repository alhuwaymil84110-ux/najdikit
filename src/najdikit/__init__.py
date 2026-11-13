"""NajdiKit: Modern White Saudi & Urban Najdi Computational Linguistics & Real-Time Voice AI Toolkit."""

from __future__ import annotations

from .lexicon import (
    AntiConfusionRule,
    LexiconEntry,
    NajdiLexicon,
    get_default_lexicon,
    light_arabic_key,
    strip_diacritics,
)
from .morphology import (
    IMPERFECTIVE_STEMS,
    PERFECTIVE_STEMS,
    MorphAnalysis,
    NajdiMorphAnalyzer,
    get_default_morph_analyzer,
)
from .neural_mlx import (
    DisambiguationResult,
    OptionalMLXDisambiguator,
    disambiguate_speakable,
    is_mlx_available,
)
from .packs import (
    AutomatonMatch,
    DialectPack,
    DialectPackRegistry,
    PhraseRadixTrie,
    TokenAhoCorasick,
    get_default_pack_registry,
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
from .scoring import (
    DialectScoreResult,
    NajdiDialectScorer,
    score_dialect,
)
from .stt_enhancer import (
    DictionaryPersonalAdapter,
    NajdiSTTEnhancer,
    PersonalIntentAdapter,
    TranscriptCleanResult,
    build_stt_prompt,
    postprocess_transcript,
)
from .voice_call import (
    SUPPORTED_TTS_BACKENDS,
    CallAckBank,
    CallClause,
    CallTurnAnalyzer,
    NajdiCallStreamer,
    TurnDecision,
    analyze_turn,
    format_for_tts_backend,
    get_call_ack,
)

__version__ = "2.0.0"

__all__ = [
    "__version__",
    "AntiConfusionRule",
    "AutomatonMatch",
    "CallAckBank",
    "CallClause",
    "CallTurnAnalyzer",
    "DialectPack",
    "DialectPackRegistry",
    "DialectScoreResult",
    "DictionaryPersonalAdapter",
    "DisambiguationResult",
    "GroundingBundle",
    "IMPERFECTIVE_STEMS",
    "LexiconEntry",
    "MorphAnalysis",
    "NajdiCallStreamer",
    "NajdiDialectScorer",
    "NajdiLexicon",
    "NajdiMorphAnalyzer",
    "NajdiPhoneticsEngine",
    "NajdiPromptKit",
    "NajdiSTTEnhancer",
    "OptionalMLXDisambiguator",
    "PERFECTIVE_STEMS",
    "PersonalIntentAdapter",
    "PhoneticResult",
    "PhraseRadixTrie",
    "SUPPORTED_PROFILES",
    "SUPPORTED_TTS_BACKENDS",
    "TokenAhoCorasick",
    "TranscriptCleanResult",
    "TurnDecision",
    "analyze_turn",
    "build_stt_prompt",
    "build_system_grounding",
    "disambiguate_speakable",
    "format_for_tts_backend",
    "get_call_ack",
    "get_default_lexicon",
    "get_default_morph_analyzer",
    "get_default_pack_registry",
    "is_mlx_available",
    "light_arabic_key",
    "number_to_saudi_words",
    "postprocess_transcript",
    "score_dialect",
    "strip_diacritics",
    "to_speakable",
]
