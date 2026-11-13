#!/usr/bin/env python3
"""Comprehensive microsecond latency & scalability benchmark for NajdiKit v2.0 (< 0.1ms gate)."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from najdikit import (  # noqa: E402
    CallTurnAnalyzer,
    NajdiCallStreamer,
    NajdiDialectScorer,
    NajdiLexicon,
    NajdiMorphAnalyzer,
    NajdiPhoneticsEngine,
    NajdiPromptKit,
    NajdiSTTEnhancer,
    TokenAhoCorasick,
    format_for_tts_backend,
    get_call_ack,
)

LOOKUP_CORPUS = (
    "أبشر",
    "يمديك",
    "ودّي",
    "شيكت",
    "الحين",
    "طال عمرك",
    "على خشمي",
    "خذ راحتك",
    "التسوية",
    "السيرفر",
)

MORPH_CORPUS = (
    "وبنشيكها",
    "بيعتمدونها",
    "خلصتوها",
    "هالفاتورة",
    "بمكالماتهم",
    "هالموضوع",
    "هالشغلة",
    "وذلحين",
)

CALL_STREAM_CORPUS = (
    "أبشر طال عمرك، شيكت لك على الفاتورة بمبلغ 1,250.50 ريال، وبنرسلها الساعة 4:30 العصر [ACTION: send_receipt].",
    "هلا والله، الأمور طيبة وكل شي جاهز الحين، إذا توصي على شي ثاني علمني.",
    "سم، ثواني أشيك لك على حالة الشحنة في النظام [ACTION: lookup_order] وأعطيك الخبر الأكيد.",
)

TURN_CORPUS = (
    ("طيب كمّل", True),
    ("أيوه معك", True),
    ("لحظة وقف شوي", True),
    ("ثواني أشوف الرقم", False),
    ("خلاص ما قصرت في أمان الله", False),
    ("أبي أستفسر عن الفاتورة اللي", False),
    ("كم رصيد الحساب الحين؟", False),
)

TTS_CORPUS = (
    "حياك الله، أبشر شيكت على هالفاتورة الحين بمبلغ 150.50 ريال قبل الساعة 4:45 العصر.",
    "يمديك تراجع الكود وترفع التحديث عبر API على GitHub؟ نسبة النجاح 100%.",
    "توّي مخلص الاجتماع الساعة 10:30 الصبح، والأمور طيبة ولله الحمد.",
    "تعرف وشلون نقدر نختصر وقت الاستجابة عشان يصير أسرع من 25%؟",
    "المبلغ الإجمالي $250 بتاريخ 2026-11-13 شامل القيمة المضافة.",
)

STT_CORPUS = (
    "ان شاء الله بشيك ع الموضوع الحيين وكل شيء تمام لا هنت. اشتركوا في القناة",
    "ماشاءالله الشغل مرتب مررررة، هاذي النسخة اللي نبيها علشان نعتمدها.",
    "وش لونك اليوم؟ ودي ارسل لك التقرير ذلحين باذن الله.",
    "ترجمة نانسي قنقر",
    "توي شيكت على السيرفر والامور طيبة جزاك الله خيرا.",
)

SCORE_CORPUS = (
    "أبشر طال عمرك، الحين أشيك لك على الفاتورة وأعطيك خبر بالتفاصيل.",
    "سوف أقوم بالتحقق من الأمر الذي ذكرته حسناً وسأوافيك بالتفاصيل.",
    "أنا بدي أشوف الموضوع هلق عشان هيك خبرني شو صار.",
)


def measure_corpus(fn: Callable[[int], Any], rounds: int) -> dict[str, float]:
    samples_us: list[float] = []
    for i in range(rounds):
        t0 = time.perf_counter_ns()
        fn(i)
        t1 = time.perf_counter_ns()
        samples_us.append((t1 - t0) / 1000.0)

    samples_us.sort()
    n = len(samples_us)
    mean_us = statistics.fmean(samples_us)
    p50_us = samples_us[n // 2]
    p95_us = samples_us[min(n - 1, int(n * 0.95))]
    p99_us = samples_us[min(n - 1, int(n * 0.99))]
    ops_per_sec = 1_000_000.0 / mean_us if mean_us > 0 else 0.0
    return {
        "mean_us": round(mean_us, 2),
        "mean_ms": round(mean_us / 1000.0, 4),
        "p50_us": round(p50_us, 2),
        "p50_ms": round(p50_us / 1000.0, 4),
        "p95_us": round(p95_us, 2),
        "p95_ms": round(p95_us / 1000.0, 4),
        "p99_us": round(p99_us, 2),
        "p99_ms": round(p99_us / 1000.0, 4),
        "throughput_ops_sec": round(ops_per_sec, 1),
    }


def _build_10k_automaton(lex: NajdiLexicon) -> TokenAhoCorasick[str]:
    ac: TokenAhoCorasick[str] = TokenAhoCorasick(light_normalize=True)
    for entry in lex.entries:
        ac.add_phrase(entry.term, entry.vocalized)
    idx = 0
    while ac.pattern_count < 10_500:
        ac.add_phrase(f"مصطلح تجريبي رقم {idx} في السياق", f"val_{idx}")
        idx += 1
    ac.compile()
    return ac


def run_suite(rounds: int = 2000, budget_ms: float = 0.1) -> dict[str, Any]:
    t_load0 = time.perf_counter_ns()
    lex = NajdiLexicon()
    morph = NajdiMorphAnalyzer(lexicon=lex)
    phonetics = NajdiPhoneticsEngine(lexicon=lex, morph=morph)
    stt = NajdiSTTEnhancer(lexicon=lex, morph=morph)
    prompt_kit = NajdiPromptKit(lexicon=lex)
    streamer_chunk = NajdiCallStreamer(vocalize=False, phonetics=phonetics, lexicon=lex)
    streamer_voc = NajdiCallStreamer(vocalize=True, phonetics=phonetics, lexicon=lex)
    turn_analyzer = CallTurnAnalyzer()
    scorer = NajdiDialectScorer(lexicon=lex)
    cold_init_ms = round((time.perf_counter_ns() - t_load0) / 1_000_000.0, 3)

    ac_10k = _build_10k_automaton(lex)

    profiles = ("modern_white", "executive", "voice_assistant", "voice_call_realtime", "technical")
    domains = ("general", "tech", "finance", "customer_care", "telephony")
    backends = ("default", "elevenlabs", "habibi", "omnivoice", "openai", "sip_8khz")

    # Warm-up pass
    for i in range(40):
        lex.lookup(LOOKUP_CORPUS[i % len(LOOKUP_CORPUS)])
        morph._analyze_uncached(MORPH_CORPUS[i % len(MORPH_CORPUS)])
        streamer_chunk.chunk_text(CALL_STREAM_CORPUS[i % len(CALL_STREAM_CORPUS)])
        streamer_voc.chunk_text(CALL_STREAM_CORPUS[i % len(CALL_STREAM_CORPUS)])
        t_txt, t_spk = TURN_CORPUS[i % len(TURN_CORPUS)]
        turn_analyzer.analyze_turn(t_txt, is_ai_speaking=t_spk)
        phonetics.prepare_tts(TTS_CORPUS[i % len(TTS_CORPUS)])
        stt.postprocess_transcript(STT_CORPUS[i % len(STT_CORPUS)])
        scorer.score_dialect(SCORE_CORPUS[i % len(SCORE_CORPUS)])
        ac_10k.select_non_overlapping(CALL_STREAM_CORPUS[i % len(CALL_STREAM_CORPUS)])

    results = {
        "lexicon_lookup": measure_corpus(
            lambda i: lex.lookup(LOOKUP_CORPUS[i % len(LOOKUP_CORPUS)]), rounds
        ),
        "call_ack_bank_0ms": measure_corpus(
            lambda i: get_call_ack("lookup", i), rounds
        ),
        "call_turn_analyze": measure_corpus(
            lambda i: turn_analyzer.analyze_turn(
                TURN_CORPUS[i % len(TURN_CORPUS)][0],
                is_ai_speaking=TURN_CORPUS[i % len(TURN_CORPUS)][1],
            ),
            rounds,
        ),
        "morphology_analyze": measure_corpus(
            lambda i: morph._analyze_uncached(MORPH_CORPUS[i % len(MORPH_CORPUS)]), rounds
        ),
        "aho_corasick_10k_scan": measure_corpus(
            lambda i: ac_10k.select_non_overlapping(
                CALL_STREAM_CORPUS[i % len(CALL_STREAM_CORPUS)]
            ),
            rounds,
        ),
        "dialect_score": measure_corpus(
            lambda i: scorer.score_dialect(SCORE_CORPUS[i % len(SCORE_CORPUS)]), rounds
        ),
        "call_stream_chunk": measure_corpus(
            lambda i: streamer_chunk.chunk_text(CALL_STREAM_CORPUS[i % len(CALL_STREAM_CORPUS)]),
            rounds,
        ),
        "call_stream_vocalized": measure_corpus(
            lambda i: streamer_voc.chunk_text(CALL_STREAM_CORPUS[i % len(CALL_STREAM_CORPUS)]),
            rounds,
        ),
        "phonetics_prepare_tts": measure_corpus(
            lambda i: phonetics.prepare_tts(TTS_CORPUS[i % len(TTS_CORPUS)]), rounds
        ),
        "tts_multi_backend": measure_corpus(
            lambda i: format_for_tts_backend(
                TTS_CORPUS[i % len(TTS_CORPUS)],
                backend=backends[i % len(backends)],
                phonetics=phonetics,
                lexicon=lex,
            ),
            rounds,
        ),
        "stt_postprocess": measure_corpus(
            lambda i: stt.postprocess_transcript(STT_CORPUS[i % len(STT_CORPUS)]), rounds
        ),
        "stt_prompt_build": measure_corpus(
            lambda i: stt.build_stt_prompt(domain=domains[i % len(domains)]), rounds
        ),
        "prompt_grounding_build": measure_corpus(
            lambda i: prompt_kit.build_grounding(profile=profiles[i % len(profiles)]), rounds
        ),
    }

    sub_gates = {
        "call_turn_analyze_lt_10us": results["call_turn_analyze"]["mean_us"] < 10.0,
        "morphology_analyze_lt_10us": results["morphology_analyze"]["mean_us"] < 10.0,
        "call_stream_chunk_lt_20us": results["call_stream_chunk"]["mean_us"] < 20.0,
        "dialect_score_lt_30us": results["dialect_score"]["mean_us"] < 30.0,
        "aho_corasick_10k_lt_40us": results["aho_corasick_10k_scan"]["mean_us"] < 40.0,
    }
    within_budget = all(m["mean_ms"] < budget_ms for m in results.values()) and all(sub_gates.values())

    return {
        "version": "2.0.0",
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "lexicon_entries": len(lex),
        "loaded_packs": list(lex.loaded_pack_ids),
        "aho_corasick_scale_patterns": ac_10k.pattern_count,
        "cold_init_ms": cold_init_ms,
        "rounds_per_op": rounds,
        "target_budget_ms": budget_ms,
        "sub_gates": sub_gates,
        "all_within_budget": within_budget,
        "benchmarks": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NajdiKit v2.0 performance benchmark suite.")
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--budget-ms", type=float, default=0.1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_suite(rounds=args.rounds, budget_ms=args.budget_ms)
    if args.json or os.environ.get("NAJDIKIT_BENCH_JSON") == "1":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            f"NajdiKit v{report['version']} Performance Suite | Python {report['python_version']} | "
            f"{report['lexicon_entries']} entries ({len(report['loaded_packs'])} packs) | "
            f"10K Automaton: {report['aho_corasick_scale_patterns']} patterns | Cold init: {report['cold_init_ms']} ms"
        )
        print("=" * 92)
        print(
            f"{'Engine Operation':<28} {'Mean (µs)':>10} {'Mean (ms)':>10} "
            f"{'P50 (µs)':>10} {'P95 (µs)':>10} {'Ops/sec':>16}"
        )
        print("-" * 92)
        for name, m in report["benchmarks"].items():
            print(
                f"{name:<28} {m['mean_us']:>10.2f} {m['mean_ms']:>10.4f} "
                f"{m['p50_us']:>10.2f} {m['p95_us']:>10.2f} {m['throughput_ops_sec']:>16,.0f}"
            )
        print("=" * 92)
        status = (
            f"PASS (< {report['target_budget_ms']}ms + all v2 microsecond sub-gates met)"
            if report["all_within_budget"]
            else "FAIL (exceeded latency gate)"
        )
        print(f"Latency Gate: {status}")

    return 0 if report["all_within_budget"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
