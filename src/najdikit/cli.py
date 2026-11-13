"""Command-line interface and warm JSON-stdio worker for NajdiKit v2.0."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import Any, Sequence

from .lexicon import NajdiLexicon, get_default_lexicon
from .morphology import NajdiMorphAnalyzer, get_default_morph_analyzer
from .neural_mlx import OptionalMLXDisambiguator
from .packs import DialectPackRegistry, get_default_pack_registry
from .phonetics import NajdiPhoneticsEngine
from .prompt_kit import SUPPORTED_PROFILES, NajdiPromptKit
from .scoring import NajdiDialectScorer
from .stt_enhancer import DictionaryPersonalAdapter, NajdiSTTEnhancer
from .voice_call import (
    SUPPORTED_TTS_BACKENDS,
    CallAckBank,
    CallTurnAnalyzer,
    NajdiCallStreamer,
    format_for_tts_backend,
)


def run_microbenchmark(iterations: int = 1000) -> dict[str, Any]:
    """Run deterministic microsecond latency benchmarks across all v2.0 NajdiKit engines."""
    lex = get_default_lexicon()
    morph = NajdiMorphAnalyzer(lexicon=lex)
    phonetics = NajdiPhoneticsEngine(lexicon=lex, morph=morph)
    stt = NajdiSTTEnhancer(lexicon=lex, morph=morph)
    prompt_kit = NajdiPromptKit(lexicon=lex)
    streamer = NajdiCallStreamer(vocalize=False, phonetics=phonetics, lexicon=lex)
    streamer_voc = NajdiCallStreamer(vocalize=True, phonetics=phonetics, lexicon=lex)
    turn_analyzer = CallTurnAnalyzer()
    scorer = NajdiDialectScorer(lexicon=lex)

    sample_tts = (
        "حياك الله، أبشر شيكت على الفاتورة الحين بمبلغ 150.50 ريال قبل الساعة 4:45 العصر عبر API."
    )
    sample_stt = "ان شاء الله بشيك ع الموضوع الحيين وكل شيء تمام لا هنت. اشتركوا في القناة"
    sample_call = (
        "أبشر طال عمرك، شيكت لك على الفاتورة بمبلغ 1,250.50 ريال، وبنرسلها الساعة 4:30 العصر [ACTION: send_receipt]."
    )
    sample_score = "أبشر طال عمرك، الحين أشيك لك على الفاتورة وأعطيك خبر بالتفاصيل."

    # Warm-up
    for _ in range(30):
        lex.lookup("يمديك")
        morph._analyze_uncached("وبنشيكها")
        phonetics.prepare_tts(sample_tts)
        streamer.chunk_text(sample_call)
        streamer_voc.chunk_text(sample_call)
        turn_analyzer.analyze_turn("طيب كمّل", is_ai_speaking=True)
        stt.postprocess_transcript(sample_stt)
        scorer.score_dialect(sample_score)
        prompt_kit.build_grounding(profile="voice_call_realtime")

    def _measure(fn: Any) -> dict[str, float]:
        samples_us: list[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            fn()
            t1 = time.perf_counter_ns()
            samples_us.append((t1 - t0) / 1000.0)
        samples_us.sort()
        n = len(samples_us)
        mean_us = statistics.fmean(samples_us)
        p50_us = samples_us[n // 2]
        p95_us = samples_us[min(n - 1, int(n * 0.95))]
        p99_us = samples_us[min(n - 1, int(n * 0.99))]
        return {
            "mean_us": round(mean_us, 2),
            "mean_ms": round(mean_us / 1000.0, 4),
            "p50_us": round(p50_us, 2),
            "p50_ms": round(p50_us / 1000.0, 4),
            "p95_us": round(p95_us, 2),
            "p95_ms": round(p95_us / 1000.0, 4),
            "p99_us": round(p99_us, 2),
            "p99_ms": round(p99_us / 1000.0, 4),
        }

    return {
        "iterations": iterations,
        "lexicon_entries": len(lex),
        "loaded_packs": list(lex.loaded_pack_ids),
        "operations": {
            "lexicon_lookup": _measure(lambda: lex.lookup("يمديك")),
            "morphology_analyze": _measure(lambda: morph._analyze_uncached("وبنشيكها")),
            "call_turn_analyze": _measure(
                lambda: turn_analyzer.analyze_turn("طيب كمّل", is_ai_speaking=True)
            ),
            "call_stream_chunk": _measure(lambda: streamer.chunk_text(sample_call)),
            "call_stream_vocalized": _measure(lambda: streamer_voc.chunk_text(sample_call)),
            "dialect_score": _measure(lambda: scorer.score_dialect(sample_score)),
            "phonetics_speakable": _measure(lambda: phonetics.prepare_tts(sample_tts)),
            "stt_postprocess": _measure(lambda: stt.postprocess_transcript(sample_stt)),
            "stt_prompt_build": _measure(lambda: stt.build_stt_prompt(domain="telephony")),
            "system_grounding": _measure(
                lambda: prompt_kit.build_grounding(profile="voice_call_realtime")
            ),
        },
    }


_WORKER_CACHE: dict[str, tuple[NajdiPhoneticsEngine, NajdiSTTEnhancer, NajdiPromptKit, NajdiDialectScorer, CallTurnAnalyzer]] = {}


def _get_worker_engines(
    custom_lex: str | None,
) -> tuple[NajdiLexicon, NajdiPhoneticsEngine, NajdiSTTEnhancer, NajdiPromptKit, NajdiDialectScorer, CallTurnAnalyzer]:
    key = custom_lex or "__default__"
    lex = get_default_lexicon(custom_lex) if custom_lex else get_default_lexicon()
    cached = _WORKER_CACHE.get(key)
    if cached is None:
        morph = NajdiMorphAnalyzer(lexicon=lex) if custom_lex else get_default_morph_analyzer()
        ph = NajdiPhoneticsEngine(lexicon=lex, morph=morph)
        stt = NajdiSTTEnhancer(lexicon=lex, morph=morph)
        kit = NajdiPromptKit(lexicon=lex)
        scorer = NajdiDialectScorer(lexicon=lex)
        turn = CallTurnAnalyzer()
        cached = (ph, stt, kit, scorer, turn)
        _WORKER_CACHE[key] = cached
    return lex, cached[0], cached[1], cached[2], cached[3], cached[4]


def _handle_worker_request(req: dict[str, Any]) -> dict[str, Any]:
    action = req.get("action", "")
    params = req.get("params") or {}
    custom_lex = params.get("custom_lexicon_path") or None
    lex, phonetics, stt_default, kit, scorer, turn_analyzer = _get_worker_engines(custom_lex)

    if action == "tts_prepare":
        text = str(params.get("text", ""))
        skip_ids = bool(params.get("skip_long_ids", True))
        expand_numbers = bool(params.get("expand_numbers", True))
        expand_acronyms = bool(params.get("expand_acronyms", True))
        backend = str(params.get("backend", "default") or "default")
        use_mlx = bool(params.get("use_mlx", False))
        if use_mlx:
            dis = OptionalMLXDisambiguator(
                base_url=params.get("local_llm_url"),
                phonetics=phonetics,
            ).disambiguate(text)
            out_mlx = dis.to_dict()
            if backend != "default":
                out_mlx["speakable"] = format_for_tts_backend(
                    out_mlx["speakable"],
                    backend=backend,
                    phonetics=phonetics,
                    lexicon=lex,
                    skip_long_ids=False,
                    expand_numbers=False,
                    expand_acronyms=False,
                )
            out_mlx["backend"] = backend
            return out_mlx
        res = phonetics.prepare_tts(
            text,
            skip_long_ids=skip_ids,
            expand_numbers=expand_numbers,
            expand_acronyms=expand_acronyms,
        )
        out = res.to_dict()
        if backend != "default":
            out["speakable"] = format_for_tts_backend(
                text,
                backend=backend,
                phonetics=phonetics,
                lexicon=lex,
                skip_long_ids=skip_ids,
                expand_numbers=expand_numbers,
                expand_acronyms=expand_acronyms,
            )
        out["backend"] = backend
        return out

    if action == "call_stream_chunk":
        text = str(params.get("text", ""))
        backend = str(params.get("backend", "default") or "default")
        max_words_raw = params.get("max_words")
        max_words = int(max_words_raw) if max_words_raw is not None else None
        vocalize = bool(params.get("vocalize", True))
        streamer = NajdiCallStreamer(
            backend=backend,
            max_words=max_words,
            vocalize=vocalize,
            phonetics=phonetics,
            lexicon=lex,
        )
        clauses = streamer.chunk_text(text)
        out_call: dict[str, Any] = {
            "backend": backend,
            "clause_count": len(clauses),
            "clauses": [c.to_dict() for c in clauses],
            "extracted_actions": list(streamer.extracted_actions),
        }
        ack_cat = params.get("ack_category")
        if ack_cat:
            out_call["ack"] = CallAckBank.get(str(ack_cat), backend=backend)
        return out_call

    if action == "call_turn_analyze":
        text = str(params.get("text", ""))
        is_ai_speaking = bool(params.get("is_ai_speaking", False))
        decision = turn_analyzer.analyze_turn(text, is_ai_speaking=is_ai_speaking)
        return decision.to_dict()

    if action == "stt_clean":
        text = str(params.get("text", ""))
        adapter_path = params.get("personal_adapter_json")
        if adapter_path:
            hook = DictionaryPersonalAdapter.from_json_file(adapter_path)
            enhancer = NajdiSTTEnhancer(lexicon=lex, morph=phonetics.morph, personal_adapter_hook=hook)
        else:
            enhancer = stt_default
        cleaned = enhancer.postprocess_transcript(text)
        out = cleaned.to_dict()
        if params.get("include_prompt"):
            out["stt_prompt"] = enhancer.build_stt_prompt(
                domain=str(params.get("domain", "general")),
                extra_terms=params.get("extra_terms"),
                previous_transcript=cleaned.text,
            )
        return out

    if action == "dialect_score":
        text = str(params.get("text", ""))
        return scorer.score_dialect(text).to_dict()

    if action == "style_grounding":
        bundle = kit.build_grounding(
            profile=str(params.get("profile", "modern_white")),
            compact=bool(params.get("compact", False)),
            domain_terms=params.get("domain_terms"),
            include_examples=bool(params.get("include_examples", True)),
        )
        out = bundle.to_dict()
        sample_text = params.get("harmonize_text")
        if sample_text:
            out["harmonized_sample"] = kit.harmonize_text(str(sample_text))
        return out

    if action == "lexicon_lookup":
        query = str(params.get("query", "")).strip()
        category = params.get("category") or None
        register = params.get("register") or None
        pack_id = params.get("pack_id") or None
        limit = int(params.get("limit", 15))
        exact = lex.lookup(query) if query else None
        anti = lex.lookup_anti_confusion(query) if query else None
        matches = lex.search(query, category=category, register=register, pack_id=pack_id, limit=limit)
        return {
            "query": query,
            "exact_match": exact.to_dict() if exact else None,
            "anti_confusion": anti.to_dict() if anti else None,
            "matches": [m.to_dict() for m in matches],
            "stats": lex.stats(),
        }

    raise ValueError(f"Unsupported worker action: {action}")


def _run_stdio_worker() -> int:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        req_id = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            result = _handle_worker_request(req)
            resp = {"id": req_id, "ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001
            resp = {"id": req_id, "ok": False, "error": str(exc)}
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="najdikit",
        description=(
            "NajdiKit v2.0: Microsecond Real-Time Voice Calls, Gulf/Najdi Morphology, "
            "Multi-Backend TTS, STT Post-Processing, Modular Packs & Dialect Scorer."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. speakable
    p_speak = sub.add_parser("speakable", help="Prepare text with Najdi/White Saudi tashkeel & G2P for TTS.")
    p_speak.add_argument("text", help="Input Arabic/mixed text to vocalize.")
    p_speak.add_argument(
        "--backend",
        default="default",
        choices=list(SUPPORTED_TTS_BACKENDS),
        help="Target TTS backend format (default, elevenlabs, omnivoice, habibi, openai, sip_8khz).",
    )
    p_speak.add_argument("--keep-ids", action="store_true", help="Do not strip long IDs, URLs, or phone numbers.")
    p_speak.add_argument("--no-numbers", action="store_true", help="Disable number/currency/time expansion.")
    p_speak.add_argument("--no-acronyms", action="store_true", help="Disable English acronym expansion.")
    p_speak.add_argument("--mlx", action="store_true", help="Use optional local MLX server if configured.")
    p_speak.add_argument("--custom-lexicon", default=None, help="Path to optional custom JSON lexicon overlay.")
    p_speak.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # 2. call-chunks
    p_chunks = sub.add_parser(
        "call-chunks",
        help="Split & vocalize LLM text into low-TTFA live voice call clauses while extracting [ACTION] tags.",
    )
    p_chunks.add_argument("text", help="Assistant response text to chunk for live voice streaming.")
    p_chunks.add_argument("--backend", default="default", choices=list(SUPPORTED_TTS_BACKENDS))
    p_chunks.add_argument("--max-words", type=int, default=None, help="Max words per breath group.")
    p_chunks.add_argument("--no-vocalize", action="store_true", help="Skip phonetic vocalization on chunks.")
    p_chunks.add_argument("--ack", default=None, choices=list(CallAckBank.CATEGORIES), help="Include a 0ms pre-cached call ack.")
    p_chunks.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # 3. turn-check
    p_turn = sub.add_parser(
        "turn-check",
        help="Classify caller speech into backchannel, barge_in, hold_request, closing, incomplete_thought, or complete_turn.",
    )
    p_turn.add_argument("text", help="Caller speech transcript chunk.")
    p_turn.add_argument("--ai-speaking", action="store_true", help="Whether AI TTS is currently playing.")
    p_turn.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # 4. normalize
    p_norm = sub.add_parser("normalize", help="Clean ASR hallucinations & normalize White Saudi spelling.")
    p_norm.add_argument("text", help="Raw transcript or colloquial text.")
    p_norm.add_argument("--adapter-json", default=None, help="Path to optional personal adapter JSON.")
    p_norm.add_argument("--custom-lexicon", default=None, help="Path to optional custom JSON lexicon overlay.")
    p_norm.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # 5. stt-prompt
    p_stt = sub.add_parser("stt-prompt", help="Generate Whisper / Scribe contextual initial & carry prompts.")
    p_stt.add_argument(
        "--domain",
        default="general",
        choices=["general", "tech", "finance", "customer_care", "telephony"],
        help="Target conversational domain.",
    )
    p_stt.add_argument("--terms", default="", help="Comma-separated domain terms to inject.")
    p_stt.add_argument("--prev", default="", help="Previous transcript chunk for carry_prompt continuity.")
    p_stt.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # 6. score
    p_score = sub.add_parser(
        "score",
        help="Score Saudi dialect alignment, MSA leakage, foreign dialect drift & call suitability.",
    )
    p_score.add_argument("text", help="Arabic text to score.")
    p_score.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # 7. morph
    p_morph = sub.add_parser(
        "morph",
        help="Decompose Gulf & Najdi proclitics, stem, enclitics, and stem-inherited vocalization.",
    )
    p_morph.add_argument("token", help="Single inflected Arabic word (e.g. وبنشيكها, هالموضوع, بمكالماتهم).")
    p_morph.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # 8. packs
    p_packs = sub.add_parser("packs", help="List active modular DialectPacks and lexicon statistics.")
    p_packs.add_argument("--load-dir", default=None, help="Optional external directory of JSON packs to inspect.")
    p_packs.add_argument("--pack-id", default=None, help="Inspect a single pack by ID.")
    p_packs.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # 9. grounding
    p_gnd = sub.add_parser("grounding", help="Build positive White Saudi system grounding prompt for LLMs.")
    p_gnd.add_argument("--profile", default="modern_white", choices=list(SUPPORTED_PROFILES))
    p_gnd.add_argument("--compact", action="store_true", help="Emit compact token-saving prompt.")
    p_gnd.add_argument("--terms", default="", help="Comma-separated domain terms.")
    p_gnd.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # 10. lookup
    p_look = sub.add_parser("lookup", help="Look up or search entries in the White Saudi & Najdi lexicon.")
    p_look.add_argument("query", nargs="?", default="", help="Word or phrase to look up.")
    p_look.add_argument("--category", default=None, help="Filter by category.")
    p_look.add_argument("--register", default=None, help="Filter by register.")
    p_look.add_argument("--pack-id", default=None, help="Filter by pack ID (e.g. core, telephony_calls).")
    p_look.add_argument("--limit", type=int, default=10, help="Max search results.")
    p_look.add_argument("--stats", action="store_true", help="Show lexicon statistics.")
    p_look.add_argument("--custom-lexicon", default=None, help="Path to optional custom JSON lexicon overlay.")
    p_look.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # 11. bench
    p_bench = sub.add_parser("bench", help="Run microsecond latency & throughput benchmarks.")
    p_bench.add_argument("--iterations", type=int, default=1000, help="Iterations per operation.")
    p_bench.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # 12. internal stdio worker for MCP / OpenClaw bridge
    sub.add_parser("worker", help="Run persistent line-delimited JSON worker over stdio.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "worker":
        return _run_stdio_worker()

    if args.command == "speakable":
        lex = NajdiLexicon(args.custom_lexicon) if args.custom_lexicon else get_default_lexicon()
        engine = NajdiPhoneticsEngine(lexicon=lex)
        if args.mlx:
            res = OptionalMLXDisambiguator(phonetics=engine).disambiguate(args.text)
            payload_mlx = res.to_dict()
            if args.backend != "default":
                payload_mlx["speakable"] = format_for_tts_backend(
                    payload_mlx["speakable"],
                    backend=args.backend,
                    phonetics=engine,
                    lexicon=lex,
                    skip_long_ids=False,
                    expand_numbers=False,
                    expand_acronyms=False,
                )
            payload_mlx["backend"] = args.backend
            if args.json:
                print(json.dumps(payload_mlx, ensure_ascii=False, indent=2))
            else:
                print(payload_mlx["speakable"])
            return 0

        out = engine.prepare_tts(
            args.text,
            skip_long_ids=not args.keep_ids,
            expand_numbers=not args.no_numbers,
            expand_acronyms=not args.no_acronyms,
        )
        payload_out = out.to_dict()
        if args.backend != "default":
            payload_out["speakable"] = format_for_tts_backend(
                args.text,
                backend=args.backend,
                phonetics=engine,
                lexicon=lex,
                skip_long_ids=not args.keep_ids,
                expand_numbers=not args.no_numbers,
                expand_acronyms=not args.no_acronyms,
            )
        payload_out["backend"] = args.backend
        if args.json:
            print(json.dumps(payload_out, ensure_ascii=False, indent=2))
        else:
            print(payload_out["speakable"])
        return 0

    if args.command == "call-chunks":
        streamer = NajdiCallStreamer(
            backend=args.backend,
            max_words=args.max_words,
            vocalize=not args.no_vocalize,
        )
        clauses = streamer.chunk_text(args.text)
        payload_chunks: dict[str, Any] = {
            "backend": args.backend,
            "clause_count": len(clauses),
            "clauses": [c.to_dict() for c in clauses],
            "extracted_actions": list(streamer.extracted_actions),
        }
        if args.ack:
            payload_chunks["ack"] = CallAckBank.get(args.ack, backend=args.backend)
        if args.json:
            print(json.dumps(payload_chunks, ensure_ascii=False, indent=2))
        else:
            for c in clauses:
                tag = "[FAST-LANE]" if c.is_fast_lane else f"[CLAUSE {c.index}]"
                print(f"{tag} {c.speakable}")
            if streamer.extracted_actions:
                print(f"ACTIONS: {', '.join(streamer.extracted_actions)}")
        return 0

    if args.command == "turn-check":
        decision = CallTurnAnalyzer().analyze_turn(args.text, is_ai_speaking=args.ai_speaking)
        if args.json:
            print(json.dumps(decision.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(
                f"kind={decision.kind} | abort_tts={decision.should_abort_tts} | "
                f"grace_ms={decision.endpoint_grace_ms} | wait_sec={decision.recommended_wait_sec}"
            )
            if decision.suggested_ack:
                print(f"ack: {decision.suggested_ack}")
        return 0

    if args.command == "normalize":
        lex = NajdiLexicon(args.custom_lexicon) if args.custom_lexicon else get_default_lexicon()
        hook = DictionaryPersonalAdapter.from_json_file(args.adapter_json) if args.adapter_json else None
        enhancer = NajdiSTTEnhancer(lexicon=lex, personal_adapter_hook=hook)
        res = enhancer.postprocess_transcript(args.text)
        if args.json:
            print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(res.text)
        return 0

    if args.command == "stt-prompt":
        terms = [t.strip() for t in args.terms.split(",") if t.strip()] if args.terms else None
        enhancer = NajdiSTTEnhancer()
        bundle = enhancer.build_stt_prompt(
            domain=args.domain,
            extra_terms=terms,
            previous_transcript=args.prev or None,
        )
        if args.json:
            print(json.dumps(bundle, ensure_ascii=False, indent=2))
        else:
            print(bundle["carry_prompt"] if args.prev else bundle["initial_prompt"])
        return 0

    if args.command == "score":
        score_res = NajdiDialectScorer().score_dialect(args.text)
        if args.json:
            print(json.dumps(score_res.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(
                f"Saudi Alignment: {score_res.saudi_alignment_score}/100 | "
                f"MSA Leakage: {score_res.msa_leakage_score}/100 | "
                f"Foreign Dialect: {score_res.foreign_dialect_score}/100 | "
                f"Call Suitability: {score_res.call_suitability_score}/100"
            )
            for s in score_res.suggestions:
                print(f"  - {s}")
        return 0

    if args.command == "morph":
        analysis = get_default_morph_analyzer().analyze(args.token)
        if args.json:
            print(json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(
                f"{analysis.token} -> {analysis.vocalized or analysis.normalized} "
                f"[proclitics={list(analysis.proclitics)}, stem={analysis.stem}, enclitics={list(analysis.enclitics)}]"
            )
        return 0

    if args.command == "packs":
        reg = DialectPackRegistry() if args.load_dir else get_default_pack_registry()
        if args.load_dir:
            reg.load_packs_dir(args.load_dir)
        lex = NajdiLexicon(pack_registry=reg) if args.load_dir else get_default_lexicon()
        packs_list = reg.list_packs()
        if args.pack_id:
            packs_list = [p for p in packs_list if p["id"] == args.pack_id]
        payload_packs = {
            "total_lexicon_entries": len(lex),
            "pack_counts": lex.pack_counts(),
            "packs": packs_list,
        }
        if args.json:
            print(json.dumps(payload_packs, ensure_ascii=False, indent=2))
        else:
            print(f"Total Active Lexicon Entries: {len(lex)} (core={lex.pack_counts().get('core', 0)})")
            for p in packs_list:
                print(f"  - {p['id']} (v{p['version']}, priority={p['priority']}): {p['entry_count']} entries — {p['name']}")
        return 0

    if args.command == "grounding":
        terms = [t.strip() for t in args.terms.split(",") if t.strip()] if args.terms else None
        bundle = NajdiPromptKit().build_grounding(
            profile=args.profile,
            compact=args.compact,
            domain_terms=terms,
        )
        if args.json:
            print(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(bundle.system_prompt)
        return 0

    if args.command == "lookup":
        lex = NajdiLexicon(args.custom_lexicon) if args.custom_lexicon else get_default_lexicon()
        if args.stats and not args.query:
            stats = lex.stats()
            if args.json:
                print(json.dumps(stats, ensure_ascii=False, indent=2))
            else:
                print(f"Entries: {stats['entry_count']} | Variants: {stats['variant_count']} | Anti-Confusion: {stats['anti_confusion_count']}")
                for cat, cnt in stats["categories"].items():
                    print(f"  - {cat}: {cnt}")
            return 0

        matches = lex.search(
            args.query,
            category=args.category,
            register=args.register,
            pack_id=args.pack_id,
            limit=args.limit,
        )
        anti = lex.lookup_anti_confusion(args.query) if args.query else None
        if args.json:
            payload = {
                "query": args.query,
                "anti_confusion": anti.to_dict() if anti else None,
                "results": [m.to_dict() for m in matches],
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            if anti:
                print(f"[Anti-Confusion] '{anti.non_saudi}' ({anti.origin}) -> '{anti.saudi_white}' | {anti.note}")
            if not matches:
                print("No matching entries found.")
            for m in matches:
                print(f"{m.term} ({m.vocalized}) [{m.pack_id}:{m.category}/{m.pos}] — {m.meaning_ar} | {m.meaning_en}")
                if m.example:
                    print(f"  مثال: {m.example}")
        return 0

    if args.command == "bench":
        report = run_microbenchmark(iterations=args.iterations)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(f"NajdiKit v2.0 Benchmark ({report['iterations']} iterations | {report['lexicon_entries']} lexicon entries)")
            print("-" * 76)
            print(f"{'Operation':<26} {'Mean (µs)':>11} {'Mean (ms)':>11} {'P50 (µs)':>11} {'P95 (µs)':>11}")
            print("-" * 76)
            for op_name, metrics in report["operations"].items():
                print(
                    f"{op_name:<26} {metrics['mean_us']:>11.2f} {metrics['mean_ms']:>11.4f} "
                    f"{metrics['p50_us']:>11.2f} {metrics['p95_us']:>11.2f}"
                )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
