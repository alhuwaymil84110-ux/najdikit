"""Comprehensive v2.0 tests for Real-Time Voice Calls, Gulf Morphology, Domain Packs, Scale & Dialect Scorer."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from najdikit import (  # noqa: E402
    CallAckBank,
    CallTurnAnalyzer,
    DialectPack,
    DialectPackRegistry,
    NajdiCallStreamer,
    NajdiMorphAnalyzer,
    PhraseRadixTrie,
    TokenAhoCorasick,
    analyze_turn,
    build_system_grounding,
    format_for_tts_backend,
    get_call_ack,
    get_default_lexicon,
    postprocess_transcript,
    score_dialect,
    to_speakable,
)


class TestScalablePacksAndAutomata(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lex = get_default_lexicon()

    def test_v2_01_total_lexicon_exceeds_1250_with_zero_duplicates(self) -> None:
        self.assertGreaterEqual(len(self.lex), 1250)
        terms = [e.term for e in self.lex.entries]
        self.assertEqual(len(terms), len(set(terms)), "Duplicate terms found across core and packs")

    def test_v2_02_four_builtin_packs_loaded_with_counts(self) -> None:
        pack_counts = self.lex.pack_counts()
        for pack_id in (
            "core",
            "telephony_calls",
            "sada_conversational",
            "commerce_fintech",
            "service_appointments",
        ):
            self.assertIn(pack_id, pack_counts)
            self.assertGreaterEqual(pack_counts[pack_id], 95)

    def test_v2_03_live_call_signature_phrases_present(self) -> None:
        phrases = (
            "طال عمرك",
            "عزّك الله",
            "على خشمي",
            "سم طال عمرك",
            "خذ راحتك",
            "الخط يقطّع",
            "صوتك واضح",
            "ثواني أشيك لك",
            "توصي على شي",
            "في أمان الله",
        )
        for p in phrases:
            hit = self.lex.lookup(p)
            self.assertIsNotNone(hit, f"Missing live-call phrase: {p}")
            self.assertTrue(hit.vocalized)

    def test_v2_04_aho_corasick_10000_scale_under_40us(self) -> None:
        ac: TokenAhoCorasick[str] = TokenAhoCorasick(light_normalize=True)
        for e in self.lex.entries:
            ac.add_phrase(e.term, e.vocalized)
        for idx in range(9500):
            ac.add_phrase(f"عبارة قياسية للتوسع رقم {idx}", f"V_{idx}")
        ac.compile()
        self.assertGreaterEqual(ac.pattern_count, 10_000)

        sample = "أبشر طال عمرك، ثواني أشيك لك على الموضوع وفي أمان الله."
        for _ in range(25):
            ac.select_non_overlapping(sample)

        samples_us: list[float] = []
        for _ in range(300):
            t0 = time.perf_counter_ns()
            matches = ac.select_non_overlapping(sample)
            t1 = time.perf_counter_ns()
            samples_us.append((t1 - t0) / 1000.0)
        self.assertGreaterEqual(len(matches), 3)
        samples_us.sort()
        p50_us = samples_us[len(samples_us) // 2]
        self.assertLess(p50_us, 40.0, f"10K Aho-Corasick P50 {p50_us:.2f}us exceeded 40us")

    def test_v2_05_phrase_radix_trie_greedy_longest_match(self) -> None:
        trie: PhraseRadixTrie[str] = PhraseRadixTrie()
        trie.insert(("سم",), "SHORT")
        trie.insert(("سم", "طال", "عمرك"), "LONG")
        hit = trie.match_longest(["سم", "طال", "عمرك", "الحين"], 0)
        self.assertEqual(hit, (3, "LONG"))
        hit_short = trie.match_longest(["سم", "أبشر"], 0)
        self.assertEqual(hit_short, (1, "SHORT"))

    def test_v2_06_dialect_pack_registry_custom_dir_and_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_path = Path(tmpdir) / "healthcare_pack.json"
            pack_path.write_text(
                json.dumps(
                    {
                        "id": "healthcare_custom",
                        "name": "Healthcare Triage Pack",
                        "version": "2.0.0",
                        "domain": "healthcare",
                        "priority": 90,
                        "entries": [
                            {
                                "term": "تحليل شامل",
                                "vocalized": "تَحْلِيلْ شَامِلْ",
                                "category": "medical",
                                "meaning_ar": "فحص مخبري متكامل",
                                "meaning_en": "Comprehensive lab panel",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            old_env = os.environ.get("NAJDIKIT_PACKS_DIR")
            os.environ["NAJDIKIT_PACKS_DIR"] = tmpdir
            try:
                reg = DialectPackRegistry(load_builtins=True, load_env_dir=True)
                ids = [p["id"] for p in reg.list_packs()]
                self.assertIn("healthcare_custom", ids)
                snap = reg.snapshot()
                self.assertIsInstance(snap, tuple)
                self.assertEqual(snap[-1].id, "healthcare_custom")

                # Unregister restores snapshot cleanly
                self.assertTrue(reg.unregister_pack("healthcare_custom"))
                self.assertIsNone(reg.get_pack("healthcare_custom"))
            finally:
                if old_env is None:
                    os.environ.pop("NAJDIKIT_PACKS_DIR", None)
                else:
                    os.environ["NAJDIKIT_PACKS_DIR"] = old_env

    def test_v2_07_register_programmatic_dialect_pack(self) -> None:
        reg = DialectPackRegistry(load_builtins=False, load_env_dir=False)
        pack = DialectPack.from_dict(
            {
                "id": "logistics_mini",
                "name": "Logistics Mini",
                "entries": [{"term": "بوليصة شحن", "vocalized": "بُولِيصَةْ شَحِنْ"}],
            }
        )
        reg.register_pack(pack)
        self.assertEqual(len(reg.list_packs()), 1)


class TestGulfNajdiMorphology(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.morph = NajdiMorphAnalyzer()

    def test_v2_08_demonstrative_hal_proclitic_sun_and_moon_letters(self) -> None:
        a1 = self.morph.analyze("هالموضوع")
        self.assertEqual(a1.proclitics, ("هال",))
        self.assertEqual(a1.vocalized, "هَاالْمَوْضُوعْ")

        a2 = self.morph.analyze("هالشغلة")
        self.assertEqual(a2.proclitics, ("هال",))
        self.assertEqual(a2.vocalized, "هَاالشَّغْلَةْ")

        a3 = self.morph.analyze("هالفاتورة")
        self.assertEqual(a3.proclitics, ("هال",))
        self.assertEqual(a3.vocalized, "هَاالْفَاتُورَةْ")

    def test_v2_09_imperfective_future_ba_and_compound_enclitics(self) -> None:
        a1 = self.morph.analyze("وبنشيكها")
        self.assertEqual(a1.proclitics, ("و", "ب"))
        self.assertEqual(a1.enclitics, ("ها",))
        self.assertEqual(a1.vocalized, "وَبِنْشَيِّكْهَا")

        a2 = self.morph.analyze("بيعتمدونها")
        self.assertEqual(a2.proclitics, ("ب",))
        self.assertEqual(a2.enclitics, ("ونها",))
        self.assertEqual(a2.vocalized, "بِيِعْتَمِدُونْهَا")

    def test_v2_10_perfective_plural_object_suffixes(self) -> None:
        a1 = self.morph.analyze("خلصتوها")
        self.assertEqual(a1.enclitics, ("توها",))
        self.assertEqual(a1.vocalized, "خَلَّصْتُوهَا")

        a2 = self.morph.analyze("جربتيها")
        self.assertEqual(a2.enclitics, ("تيها",))
        self.assertEqual(a2.vocalized, "جَرَّبْتِيهَا")

    def test_v2_11_feminine_plural_and_possessive_clitics(self) -> None:
        a1 = self.morph.analyze("بمكالماتهم")
        self.assertEqual(a1.proclitics, ("ب",))
        self.assertEqual(a1.stem, "مكالمة")
        self.assertEqual(a1.enclitics, ("ات", "هم"))
        self.assertEqual(a1.vocalized, "بِمُكَالَمَاتْهُمْ")

        a2 = self.morph.analyze("فاتورتها")
        self.assertEqual(a2.stem, "فاتورة")
        self.assertEqual(a2.enclitics, ("ها",))
        self.assertEqual(a2.vocalized, "فَاتُورَتْهَا")

    def test_v2_12_morphology_integrated_into_phonetics_and_stt(self) -> None:
        spoken = to_speakable("وبنشيكها وبيعتمدونها على هالفاتورة بمكالماتهم بعد ما خلصتوها")
        self.assertIn("وَبِنْشَيِّكْهَا", spoken)
        self.assertIn("وَبِيِعْتَمِدُونْهَا", spoken)
        self.assertIn("هَاالْفَاتُورَةْ", spoken)
        self.assertIn("بِمُكَالَمَاتْهُمْ", spoken)
        self.assertIn("خَلَّصْتُوهَا", spoken)

        cleaned = postprocess_transcript("شيكت على هالفاتوره وذلحين بنعتمدها")
        self.assertIn("هالفاتورة", cleaned.text)
        self.assertIn("والحين", cleaned.text)

    def test_v2_13_morphology_latency_under_10us(self) -> None:
        for _ in range(30):
            self.morph._analyze_uncached("وبنشيكها")
        samples_us = []
        for _ in range(400):
            t0 = time.perf_counter_ns()
            self.morph._analyze_uncached("وبنشيكها")
            t1 = time.perf_counter_ns()
            samples_us.append((t1 - t0) / 1000.0)
        samples_us.sort()
        self.assertLess(samples_us[len(samples_us) // 2], 10.0)


class TestRealTimeVoiceCallEngine(unittest.TestCase):
    def test_v2_14_first_clause_fast_lane_on_saudi_openers(self) -> None:
        for opener in ("أبشر، ", "سم، ", "هلا والله، "):
            streamer = NajdiCallStreamer()
            clauses = streamer.push(opener)
            self.assertEqual(len(clauses), 1, f"Opener {opener!r} did not emit immediately on Clause 0")
            self.assertTrue(clauses[0].is_fast_lane)
            self.assertEqual(clauses[0].index, 0)

    def test_v2_15_atomic_guard_never_splits_decimals_currencies_or_times(self) -> None:
        streamer = NajdiCallStreamer()
        emitted = []
        # Token stream splits right across thousands comma, decimal point, and time colon!
        tokens = [
            "أبشر، ",
            "المبلغ الإجمالي ",
            "1,",
            "250.",
            "50 ريال ",
            "وسعر الصرف 3.",
            "85 ",
            "موعدنا الساعة 4:",
            "30 العصر.",
        ]
        for tok in tokens:
            emitted.extend(streamer.push(tok))
        emitted.extend(streamer.flush())

        self.assertEqual(len(emitted), 2)
        self.assertEqual(emitted[0].text, "أبشر،")
        # Entire second clause keeps 1,250.50, 3.85, and 4:30 intact!
        self.assertIn("1,250.50 ريال", emitted[1].text)
        self.assertIn("3.85", emitted[1].text)
        self.assertIn("4:30 العصر.", emitted[1].text)

    def test_v2_16_atomic_guard_inseparable_saudi_constructs_and_prepositions(self) -> None:
        streamer = NajdiCallStreamer(max_words=5)
        text = "بنخلص الترتيبات كلها اليوم إن شاء الله وبعدين نرسل لك التأكيد في أمان الله"
        clauses = streamer.chunk_text(text)
        joined_clauses = [c.text for c in clauses]
        for c_txt in joined_clauses:
            self.assertFalse(c_txt.endswith("إن"), f"Split inside إن شاء الله: {joined_clauses}")
            self.assertFalse(c_txt.endswith("إن شاء"), f"Split inside إن شاء الله: {joined_clauses}")
            self.assertFalse(c_txt.endswith("في"), f"Split after preposition في: {joined_clauses}")
            self.assertFalse(c_txt.endswith("في أمان"), f"Split inside في أمان الله: {joined_clauses}")

    def test_v2_17_breath_group_overflow_splits_before_saudi_connectors(self) -> None:
        streamer = NajdiCallStreamer(max_words=10, vocalize=False)
        long_unpunctuated = (
            "شيكت لك الحين على حالة الطلب في المستودع الرئيسي بالرياض "
            "وعشان يوصلك بدري حجزنا لك أقرب مندوب توصيل اليوم"
        )
        clauses = streamer.chunk_text(long_unpunctuated)
        self.assertGreaterEqual(len(clauses), 2)
        self.assertTrue(clauses[1].text.startswith("وعشان"))

    def test_v2_18_sip_8khz_enforces_shorter_8_word_max(self) -> None:
        streamer = NajdiCallStreamer(backend="sip_8khz")
        self.assertEqual(streamer.max_words, 8)
        clauses = streamer.chunk_text(
            "شيكت لك الحين على الفاتورة في النظام الأساسي وبنرسل لك رسالة نصية بالتأكيد فوراً"
        )
        self.assertGreaterEqual(len(clauses), 2)

    def test_v2_19_inline_action_tag_and_markdown_emoji_stripper_across_split_tokens(self) -> None:
        streamer = NajdiCallStreamer()
        tokens = [
            "**أبشر**، ",
            "جاري تحويلك ",
            "[ACT",
            "ION: ",
            "transfer_vip_queue",
            "] ",
            "للقسم المختص 😊 ",
            "[ACTION: log_ticket].",
        ]
        out = []
        for t in tokens:
            out.extend(streamer.push(t))
        out.extend(streamer.flush())

        full_spoken = " ".join(c.speakable for c in out)
        full_text = " ".join(c.text for c in out)
        self.assertEqual(streamer.extracted_actions, ["transfer_vip_queue", "log_ticket"])
        self.assertNotIn("ACTION", full_spoken)
        self.assertNotIn("[", full_spoken)
        self.assertNotIn("]", full_spoken)
        self.assertNotIn("*", full_text)
        self.assertNotIn("😊", full_text)

    def test_v2_20_turn_analyzer_backchannel_never_aborts_tts(self) -> None:
        analyzer = CallTurnAnalyzer()
        for word in ("إيه", "أيوه", "طيب", "تمام", "أها", "كمّل", "معك", "حلو", "صح"):
            dec = analyzer.analyze_turn(word, is_ai_speaking=True)
            self.assertEqual(dec.kind, "backchannel", f"Expected backchannel for {word}")
            self.assertFalse(dec.should_abort_tts, f"Backchannel {word} must NOT abort AI TTS")

    def test_v2_21_turn_analyzer_barge_in_aborts_tts(self) -> None:
        for cue in ("لحظة", "وقف", "اصبر", "لا لا", "دقيقة", "اسمع", "بس سؤال", "غير كذا"):
            dec = analyze_turn(cue, is_ai_speaking=True)
            self.assertEqual(dec.kind, "barge_in", f"Expected barge_in for {cue}")
            self.assertTrue(dec.should_abort_tts)

        # Substantive multi-word caller interruption while AI is speaking
        substantive = analyze_turn("أبي أغير العنوان لحي الملقا", is_ai_speaking=True)
        self.assertEqual(substantive.kind, "barge_in")
        self.assertTrue(substantive.should_abort_tts)

    def test_v2_22_turn_analyzer_hold_closing_incomplete_and_complete(self) -> None:
        for hold in ("ثواني", "دقيقة بس", "خلك معي", "لحظة أشوف"):
            d_hold = analyze_turn(hold)
            self.assertEqual(d_hold.kind, "hold_request")
            self.assertEqual(d_hold.recommended_wait_sec, 20.0)
            self.assertEqual(d_hold.suggested_ack, "خِذْ رَاحْتِكْ، مَعِكْ.")

        for close_p in ("مع السلامة", "فمان الله", "في أمان الله", "خلاص ما قصرت", "توصي شي"):
            d_close = analyze_turn(close_p)
            self.assertEqual(d_close.kind, "closing")
            self.assertTrue(d_close.is_closing)
            self.assertTrue(d_close.suggested_ack)

        for dangling in ("أبي أسأل عن الفاتورة اللي", "يعني", "طلبت أمس عشان", "بس", "و"):
            d_inc = analyze_turn(dangling)
            self.assertEqual(d_inc.kind, "incomplete_thought")
            self.assertEqual(d_inc.endpoint_grace_ms, 400)

        d_comp = analyze_turn("كم رصيد البطاقة الحين؟")
        self.assertEqual(d_comp.kind, "complete_turn")
        self.assertEqual(d_comp.endpoint_grace_ms, 0)

    def test_v2_23_call_ack_bank_zero_ms_categories(self) -> None:
        for cat in ("lookup", "action", "confirm", "hold", "unclear"):
            ack = get_call_ack(cat)
            self.assertTrue(ack)
            self.assertGreaterEqual(len(CallAckBank.all_for(cat)), 3)
        habibi_ack = get_call_ack("lookup", backend="habibi")
        self.assertTrue(habibi_ack.startswith("②〈") and habibi_ack.endswith("〉"))

    def test_v2_24_multi_backend_tts_formatters(self) -> None:
        habibi = format_for_tts_backend("أبشر شيكت الحين", backend="habibi")
        self.assertTrue(habibi.startswith("②〈") and habibi.endswith("〉"))
        self.assertIn("أَبْشِرْ", habibi)

        omni = format_for_tts_backend("شيكت على 100 ريال", backend="omnivoice")
        self.assertIn("شَيْيَكْت", omni)
        self.assertNotIn("شَيَّكْت", omni)

        el = format_for_tts_backend("تعرف وشلون نشوف الطلب...", backend="elevenlabs")
        self.assertIn("تِعْرِفْ", el)
        self.assertNotIn("...", el)

        sip = format_for_tts_backend("أبشر طال عمرك كل شي جاهز", backend="sip_8khz")
        self.assertTrue(sip.endswith("."))

        oai = format_for_tts_backend("أبشر الحين...", backend="openai")
        self.assertIn("أَبْشِرْ", oai)


class TestDialectScorerAndRealtimePrompt(unittest.TestCase):
    def test_v2_25_score_dialect_pure_white_saudi_high_alignment(self) -> None:
        res = score_dialect("أبشر طال عمرك، الحين أشيك لك على الفاتورة وأعطيك خبر.")
        self.assertGreaterEqual(res.saudi_alignment_score, 90.0)
        self.assertEqual(res.msa_leakage_score, 0.0)
        self.assertEqual(res.foreign_dialect_score, 0.0)
        self.assertGreaterEqual(res.call_suitability_score, 90.0)
        self.assertEqual(res.flagged_spans, [])

    def test_v2_26_score_dialect_detects_stiff_msa_leakage(self) -> None:
        res = score_dialect("سوف أقوم بالتحقق من الأمر الذي ذكرته حسناً وهذا نظامٌ سريعٍ.")
        self.assertGreaterEqual(res.msa_leakage_score, 60.0)
        self.assertLess(res.saudi_alignment_score, 40.0)
        self.assertGreaterEqual(len(res.flagged_spans), 3)
        self.assertGreaterEqual(len(res.suggestions), 2)

    def test_v2_27_score_dialect_detects_foreign_dialect_drift(self) -> None:
        res = score_dialect("أنا بدي أشوف الموضوع هلق عشان هيك خبرني شو صار.")
        self.assertGreaterEqual(res.foreign_dialect_score, 60.0)
        self.assertLess(res.saudi_alignment_score, 30.0)
        origins = {s["origin"] for s in res.flagged_spans}
        self.assertIn("levantine", origins)

    def test_v2_28_score_dialect_penalizes_markdown_and_urls_for_calls(self) -> None:
        res = score_dialect("أبشر، راجع **التفاصيل** على https://example.org/doc\n- البند الأول")
        self.assertLess(res.call_suitability_score, res.saudi_alignment_score - 30.0)

    def test_v2_29_voice_call_realtime_grounding_profile(self) -> None:
        bundle = build_system_grounding(profile="voice_call_realtime", domain_terms=["تأكيد الحجز"])
        self.assertEqual(bundle.profile, "voice_call_realtime")
        self.assertIn("TTFA", bundle.system_prompt)
        self.assertIn("[ACTION:", bundle.system_prompt)
        self.assertIn("تأكيد الحجز", bundle.system_prompt)

    def test_v2_30_streamer_latency_under_20us_and_turn_under_10us(self) -> None:
        streamer = NajdiCallStreamer(vocalize=False)
        analyzer = CallTurnAnalyzer()
        sample = "أبشر طال عمرك، شيكت لك على الطلب بمبلغ 350.50 ريال، وبنرسلها الحين."
        for _ in range(30):
            streamer.chunk_text(sample)
            analyzer.analyze_turn("أيوه معك", is_ai_speaking=True)

        chunk_us = []
        turn_us = []
        for _ in range(300):
            t0 = time.perf_counter_ns()
            streamer.chunk_text(sample)
            t1 = time.perf_counter_ns()
            chunk_us.append((t1 - t0) / 1000.0)

            t2 = time.perf_counter_ns()
            analyzer.analyze_turn("أيوه معك", is_ai_speaking=True)
            t3 = time.perf_counter_ns()
            turn_us.append((t3 - t2) / 1000.0)

        chunk_us.sort()
        turn_us.sort()
        self.assertLess(chunk_us[len(chunk_us) // 2], 20.0)
        self.assertLess(turn_us[len(turn_us) // 2], 10.0)

    def test_v2_31_scorer_latency_under_30us(self) -> None:
        sample = "أبشر طال عمرك، الحين أشيك لك على الفاتورة وأعطيك خبر بالتفاصيل."
        for _ in range(30):
            score_dialect(sample)
        samples_us = []
        for _ in range(300):
            t0 = time.perf_counter_ns()
            score_dialect(sample)
            t1 = time.perf_counter_ns()
            samples_us.append((t1 - t0) / 1000.0)
        samples_us.sort()
        self.assertLess(samples_us[len(samples_us) // 2], 30.0)

    def test_v2_32_aho_corasick_replace_all_non_overlapping_longest_wins(self) -> None:
        ac: TokenAhoCorasick[str] = TokenAhoCorasick(light_normalize=True)
        ac.add_phrase("ان شاء", "SHORT_WRONG")
        ac.add_phrase("ان شاء الله", "إن شاء الله")
        ac.compile()
        out, n = ac.replace_all("بنخلصها ان شاء الله اليوم")
        self.assertEqual(out, "بنخلصها إن شاء الله اليوم")
        self.assertEqual(n, 1)

    def test_v2_33_unterminated_action_tag_on_stream_eof_never_leaks(self) -> None:
        streamer = NajdiCallStreamer()
        clauses = streamer.push("أبشر، جاري التحويل [ACTION: hangup_clean")
        clauses.extend(streamer.flush())
        spoken = " ".join(c.speakable for c in clauses)
        self.assertNotIn("[", spoken)
        self.assertNotIn("ACTION", spoken)
        self.assertEqual(streamer.extracted_actions, ["hangup_clean"])


if __name__ == "__main__":
    unittest.main()
