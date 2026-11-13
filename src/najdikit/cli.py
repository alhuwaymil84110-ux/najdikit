"""Command-line interface and warm JSON-stdio worker for NajdiKit."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import Any, Sequence

from .lexicon import NajdiLexicon, get_default_lexicon
from .neural_mlx import OptionalMLXDisambiguator
from .phonetics import NajdiPhoneticsEngine
from .prompt_kit import SUPPORTED_PROFILES, NajdiPromptKit
from .stt_enhancer import DictionaryPersonalAdapter, NajdiSTTEnhancer


def run_microbenchmark(iterations: int = 1000) -> dict[str, Any]:
    """Run deterministic latency benchmarks across all core NajdiKit engines."""
    lex = get_default_lexicon()
    phonetics = NajdiPhoneticsEngine(lexicon=lex)
    stt = NajdiSTTEnhancer(lexicon=lex)
    prompt_kit = NajdiPromptKit(lexicon=lex)

    sample_tts = (
        "حياك الله، أبشر شيكت على الفاتورة الحين بمبلغ 150.50 ريال قبل الساعة 4:45 العصر عبر API."
    )
    sample_stt = "ان شاء الله بشيك ع الموضوع الحيين وكل شيء تمام لا هنت. اشتركوا في القناة"

    # Warm-up
    for _ in range(25):
        lex.lookup("يمديك")
        phonetics.prepare_tts(sample_tts)
        stt.postprocess_transcript(sample_stt)
        prompt_kit.build_grounding(profile="modern_white")

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
        "operations": {
            "lexicon_lookup": _measure(lambda: lex.lookup("يمديك")),
            "phonetics_speakable": _measure(lambda: phonetics.prepare_tts(sample_tts)),
            "stt_postprocess": _measure(lambda: stt.postprocess_transcript(sample_stt)),
            "stt_prompt_build": _measure(lambda: stt.build_stt_prompt(domain="tech")),
            "system_grounding": _measure(lambda: prompt_kit.build_grounding(profile="modern_white")),
        },
    }


def _handle_worker_request(req: dict[str, Any]) -> dict[str, Any]:
    action = req.get("action", "")
    params = req.get("params") or {}
    custom_lex = params.get("custom_lexicon_path") or None
    lex = get_default_lexicon(custom_lex) if custom_lex else get_default_lexicon()

    if action == "tts_prepare":
        text = str(params.get("text", ""))
        skip_ids = bool(params.get("skip_long_ids", True))
        expand_numbers = bool(params.get("expand_numbers", True))
        expand_acronyms = bool(params.get("expand_acronyms", True))
        use_mlx = bool(params.get("use_mlx", False))
        if use_mlx:
            dis = OptionalMLXDisambiguator(
                base_url=params.get("local_llm_url"),
                phonetics=NajdiPhoneticsEngine(lexicon=lex),
            ).disambiguate(text)
            return dis.to_dict()
        res = NajdiPhoneticsEngine(lexicon=lex).prepare_tts(
            text,
            skip_long_ids=skip_ids,
            expand_numbers=expand_numbers,
            expand_acronyms=expand_acronyms,
        )
        return res.to_dict()

    if action == "stt_clean":
        text = str(params.get("text", ""))
        adapter_path = params.get("personal_adapter_json")
        hook = DictionaryPersonalAdapter.from_json_file(adapter_path) if adapter_path else None
        enhancer = NajdiSTTEnhancer(lexicon=lex, personal_adapter_hook=hook)
        cleaned = enhancer.postprocess_transcript(text)
        out = cleaned.to_dict()
        if params.get("include_prompt"):
            out["stt_prompt"] = enhancer.build_stt_prompt(
                domain=str(params.get("domain", "general")),
                extra_terms=params.get("extra_terms"),
                previous_transcript=cleaned.text,
            )
        return out

    if action == "style_grounding":
        kit = NajdiPromptKit(lexicon=lex)
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
        limit = int(params.get("limit", 15))
        exact = lex.lookup(query) if query else None
        anti = lex.lookup_anti_confusion(query) if query else None
        matches = lex.search(query, category=category, register=register, limit=limit)
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
            "NajdiKit: Sub-millisecond Modern White Saudi & Urban Najdi "
            "Phonetics, STT Post-Processing, Lexicon, and LLM Grounding Toolkit."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. speakable
    p_speak = sub.add_parser("speakable", help="Prepare text with Najdi/White Saudi tashkeel & G2P for TTS.")
    p_speak.add_argument("text", help="Input Arabic/mixed text to vocalize.")
    p_speak.add_argument("--keep-ids", action="store_true", help="Do not strip long IDs, URLs, or phone numbers.")
    p_speak.add_argument("--no-numbers", action="store_true", help="Disable number/currency/time expansion.")
    p_speak.add_argument("--no-acronyms", action="store_true", help="Disable English acronym expansion.")
    p_speak.add_argument("--mlx", action="store_true", help="Use optional local MLX server if configured.")
    p_speak.add_argument("--custom-lexicon", default=None, help="Path to optional custom JSON lexicon overlay.")
    p_speak.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # 2. normalize
    p_norm = sub.add_parser("normalize", help="Clean ASR hallucinations & normalize White Saudi spelling.")
    p_norm.add_argument("text", help="Raw transcript or colloquial text.")
    p_norm.add_argument("--adapter-json", default=None, help="Path to optional personal adapter JSON.")
    p_norm.add_argument("--custom-lexicon", default=None, help="Path to optional custom JSON lexicon overlay.")
    p_norm.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # 3. stt-prompt
    p_stt = sub.add_parser("stt-prompt", help="Generate Whisper / Scribe contextual initial & carry prompts.")
    p_stt.add_argument(
        "--domain",
        default="general",
        choices=["general", "tech", "finance", "customer_care"],
        help="Target conversational domain.",
    )
    p_stt.add_argument("--terms", default="", help="Comma-separated domain terms to inject.")
    p_stt.add_argument("--prev", default="", help="Previous transcript chunk for carry_prompt continuity.")
    p_stt.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # 4. grounding
    p_gnd = sub.add_parser("grounding", help="Build positive White Saudi system grounding prompt for LLMs.")
    p_gnd.add_argument("--profile", default="modern_white", choices=list(SUPPORTED_PROFILES))
    p_gnd.add_argument("--compact", action="store_true", help="Emit compact token-saving prompt.")
    p_gnd.add_argument("--terms", default="", help="Comma-separated domain terms.")
    p_gnd.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # 5. lookup
    p_look = sub.add_parser("lookup", help="Look up or search entries in the White Saudi & Najdi lexicon.")
    p_look.add_argument("query", nargs="?", default="", help="Word or phrase to look up.")
    p_look.add_argument("--category", default=None, help="Filter by category.")
    p_look.add_argument("--register", default=None, help="Filter by register.")
    p_look.add_argument("--limit", type=int, default=10, help="Max search results.")
    p_look.add_argument("--stats", action="store_true", help="Show lexicon statistics.")
    p_look.add_argument("--custom-lexicon", default=None, help="Path to optional custom JSON lexicon overlay.")
    p_look.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # 6. bench
    p_bench = sub.add_parser("bench", help="Run latency & throughput benchmarks.")
    p_bench.add_argument("--iterations", type=int, default=1000, help="Iterations per operation.")
    p_bench.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # 7. internal stdio worker for MCP / OpenClaw bridge
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
            if args.json:
                print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
            else:
                print(res.speakable)
            return 0
        out = engine.prepare_tts(
            args.text,
            skip_long_ids=not args.keep_ids,
            expand_numbers=not args.no_numbers,
            expand_acronyms=not args.no_acronyms,
        )
        if args.json:
            print(json.dumps(out.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(out.speakable)
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

        matches = lex.search(args.query, category=args.category, register=args.register, limit=args.limit)
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
                print(f"{m.term} ({m.vocalized}) [{m.category}/{m.pos}] — {m.meaning_ar} | {m.meaning_en}")
                if m.example:
                    print(f"  مثال: {m.example}")
        return 0

    if args.command == "bench":
        report = run_microbenchmark(iterations=args.iterations)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(f"NajdiKit Benchmark ({report['iterations']} iterations | {report['lexicon_entries']} lexicon entries)")
            print("-" * 74)
            print(f"{'Operation':<24} {'Mean (µs)':>11} {'Mean (ms)':>11} {'P50 (µs)':>11} {'P95 (µs)':>11}")
            print("-" * 74)
            for op_name, metrics in report["operations"].items():
                print(
                    f"{op_name:<24} {metrics['mean_us']:>11.2f} {metrics['mean_ms']:>11.4f} "
                    f"{metrics['p50_us']:>11.2f} {metrics['p95_us']:>11.2f}"
                )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
