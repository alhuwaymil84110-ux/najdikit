#!/usr/bin/env python3
"""Comprehensive latency & throughput benchmark for NajdiKit (< 0.3ms target gate)."""

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
    NajdiLexicon,
    NajdiPhoneticsEngine,
    NajdiPromptKit,
    NajdiSTTEnhancer,
)

LOOKUP_CORPUS = (
    "أبشر",
    "يمديك",
    "ودّي",
    "شيكت",
    "الحين",
    "ما قصر",
    "وشلونك",
    "لاهنت",
    "التسوية",
    "السيرفر",
)

TTS_CORPUS = (
    "حياك الله، أبشر شيكت على الفاتورة الحين بمبلغ 150.50 ريال قبل الساعة 4:45 العصر.",
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


def run_suite(rounds: int = 2000, budget_ms: float = 0.3) -> dict[str, Any]:
    t_load0 = time.perf_counter_ns()
    lex = NajdiLexicon()
    phonetics = NajdiPhoneticsEngine(lexicon=lex)
    stt = NajdiSTTEnhancer(lexicon=lex)
    prompt_kit = NajdiPromptKit(lexicon=lex)
    cold_init_ms = round((time.perf_counter_ns() - t_load0) / 1_000_000.0, 3)

    # Warm-up pass
    for i in range(40):
        lex.lookup(LOOKUP_CORPUS[i % len(LOOKUP_CORPUS)])
        phonetics.prepare_tts(TTS_CORPUS[i % len(TTS_CORPUS)])
        stt.postprocess_transcript(STT_CORPUS[i % len(STT_CORPUS)])
        stt.build_stt_prompt(domain="tech")
        prompt_kit.build_grounding(profile="modern_white")

    profiles = ("modern_white", "executive", "voice_assistant", "technical")
    domains = ("general", "tech", "finance", "customer_care")

    results = {
        "lexicon_lookup": measure_corpus(
            lambda i: lex.lookup(LOOKUP_CORPUS[i % len(LOOKUP_CORPUS)]), rounds
        ),
        "phonetics_prepare_tts": measure_corpus(
            lambda i: phonetics.prepare_tts(TTS_CORPUS[i % len(TTS_CORPUS)]), rounds
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

    within_budget = all(m["mean_ms"] < budget_ms for m in results.values())
    return {
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "lexicon_entries": len(lex),
        "cold_init_ms": cold_init_ms,
        "rounds_per_op": rounds,
        "target_budget_ms": budget_ms,
        "all_within_budget": within_budget,
        "benchmarks": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NajdiKit performance benchmark suite.")
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--budget-ms", type=float, default=0.3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_suite(rounds=args.rounds, budget_ms=args.budget_ms)
    if args.json or os.environ.get("NAJDIKIT_BENCH_JSON") == "1":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            f"NajdiKit Performance Suite | Python {report['python_version']} | "
            f"{report['lexicon_entries']} entries | Cold init: {report['cold_init_ms']} ms"
        )
        print("=" * 88)
        print(
            f"{'Engine Operation':<26} {'Mean (µs)':>10} {'Mean (ms)':>10} "
            f"{'P50 (µs)':>10} {'P95 (µs)':>10} {'Ops/sec':>14}"
        )
        print("-" * 88)
        for name, m in report["benchmarks"].items():
            print(
                f"{name:<26} {m['mean_us']:>10.2f} {m['mean_ms']:>10.4f} "
                f"{m['p50_us']:>10.2f} {m['p95_us']:>10.2f} {m['throughput_ops_sec']:>14,.0f}"
            )
        print("=" * 88)
        status = "PASS (< 0.3ms target met)" if report["all_within_budget"] else "FAIL (exceeded budget)"
        print(f"Latency Gate: {status}")

    return 0 if report["all_within_budget"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
