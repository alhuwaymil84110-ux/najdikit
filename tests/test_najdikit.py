"""Comprehensive unit & integration test suite for NajdiKit core engines."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from najdikit import (  # noqa: E402
    DictionaryPersonalAdapter,
    NajdiLexicon,
    NajdiPhoneticsEngine,
    NajdiPromptKit,
    NajdiSTTEnhancer,
    OptionalMLXDisambiguator,
    PersonalIntentAdapter,
    build_stt_prompt,
    build_system_grounding,
    number_to_saudi_words,
    postprocess_transcript,
    strip_diacritics,
    to_speakable,
)


class TestLexiconAtlas(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lex = NajdiLexicon()

    def test_01_lexicon_exceeds_500_entries(self) -> None:
        self.assertGreaterEqual(len(self.lex), 500)

    def test_02_core_categories_populated(self) -> None:
        cats = self.lex.categories()
        expected = {
            "greetings_courtesy",
            "verbs_desire_cognition",
            "verbs_action_motion",
            "connectors_discourse",
            "everyday_urban",
            "idioms_expressions",
            "tech_workplace",
            "finance_commerce",
        }
        self.assertTrue(expected.issubset(set(cats.keys())))
        for name in expected:
            self.assertGreater(cats[name], 15, f"Category {name} has too few entries")

    def test_03_signature_saudi_terms_present(self) -> None:
        required_terms = [
            "أبي",
            "يبون",
            "ودّي",
            "توّي",
            "شيكت",
            "أبشر",
            "ما قصر",
            "يمديك",
            "عساه",
            "ترا",
            "وشلونك",
            "اللي",
            "هذي",
            "الحين",
            "عشان",
            "خلاص",
            "زين",
            "لاهنت",
        ]
        for term in required_terms:
            entry = self.lex.lookup(term)
            self.assertIsNotNone(entry, f"Missing signature term: {term}")

    def test_04_variant_and_unvocalized_lookup(self) -> None:
        e1 = self.lex.lookup("ذلحين")
        self.assertIsNotNone(e1)
        self.assertEqual(e1.term, "الحين")

        e2 = self.lex.lookup("ودي")
        self.assertIsNotNone(e2)
        self.assertEqual(e2.term, "ودّي")

        e3 = self.lex.lookup("علشان")
        self.assertIsNotNone(e3)
        self.assertEqual(e3.term, "عشان")

    def test_05_anti_confusion_lookup_and_drift_detection(self) -> None:
        rule = self.lex.lookup_anti_confusion("بدي")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.saudi_white, "أبي")

        drift = self.lex.detect_dialect_drift("أنا بدي أشوف الموضوع هلق لو سمحت")
        found_terms = {d["non_saudi"] for d in drift}
        self.assertIn("بدي", found_terms)
        self.assertIn("هلق", found_terms)

    def test_06_search_filtering_by_category_and_english(self) -> None:
        finance_hits = self.lex.search("settlement", category="finance_commerce")
        self.assertTrue(any(h.term == "التسوية" for h in finance_hits))

    def test_07_custom_lexicon_overlay_without_mutating_default(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as tmp:
            json.dump(
                {
                    "entries": [
                        {
                            "term": "مجبوس",
                            "vocalized": "مَجْبُوسْ",
                            "category": "custom",
                            "meaning_ar": "طبق أرز خليجي",
                            "meaning_en": "Spiced rice dish",
                            "variants": ["مكبوس"],
                        }
                    ],
                    "orthographic_map": {"يبيلها": "يبي لها"},
                },
                tmp,
                ensure_ascii=False,
            )
            tmp_path = Path(tmp.name)

        try:
            custom_lex = NajdiLexicon(custom_lexicon_path=tmp_path)
            hit = custom_lex.lookup("مكبوس")
            self.assertIsNotNone(hit)
            self.assertEqual(hit.vocalized, "مَجْبُوسْ")
            self.assertEqual(custom_lex.normalize_token("يبيلها"), "يبي لها")
            # Default lexicon remains untouched
            self.assertIsNone(self.lex.lookup("مجبوس"))
        finally:
            tmp_path.unlink(missing_ok=True)


class TestPhoneticsEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = NajdiPhoneticsEngine()

    def test_08_najdi_imperfective_kasrah_prefix(self) -> None:
        self.assertEqual(to_speakable("تعرف"), "تِعْرِفْ")
        self.assertEqual(to_speakable("يقول"), "يِقُولْ")
        self.assertEqual(to_speakable("نقدر"), "نِقْدَرْ")
        self.assertEqual(to_speakable("تشوف"), "تِشُوفْ")

    def test_09_first_person_singular_retains_fathah(self) -> None:
        self.assertEqual(to_speakable("أقدر"), "أَقْدَرْ")
        self.assertEqual(to_speakable("أبي"), "أَبِي")
        self.assertEqual(to_speakable("أشوف"), "أَشُوفْ")

    def test_10_conversational_pausal_sukun_phrases(self) -> None:
        out = to_speakable("أبشر كل شي تمام الحين")
        self.assertIn("أَبْشِرْ", out)
        self.assertIn("كِلْ شَيْ", out)
        self.assertIn("تَمَامْ", out)
        self.assertIn("الْحِينْ", out)

    def test_11_formal_msa_tanween_stripped_adverbs_preserved(self) -> None:
        out = to_speakable("هذا نظامٌ سريعٍ فعلاً")
        self.assertNotIn("ٌ", out)
        self.assertNotIn("ٍ", out)
        self.assertIn("فِعْلًا", out)

    def test_12_shaddah_accuracy_on_core_words(self) -> None:
        out = to_speakable("ودّي أقول إن الفريق ما قصر وتوّي شيكت")
        self.assertIn("وِدِّي", out)
        self.assertIn("مَا قَصَّرْ", out)
        self.assertIn("شَيَّكْتْ", out)

    def test_13_saudi_currency_singular_dual_plural_and_halalah(self) -> None:
        self.assertIn("رِيَالْ وَاحِدْ", to_speakable("1 ريال"))
        self.assertIn("رِيَالَيْنْ", to_speakable("2 ريال"))
        self.assertIn("خَمْسْ رِيَالَاتْ", to_speakable("5 ريال"))
        combo = to_speakable("المبلغ 150.50 ريال")
        self.assertIn("مِيَّة وَخَمْسِينْ رِيَالْ وَخَمْسِينْ هَلَلَة", combo)

    def test_14_usd_and_percentage_expansion(self) -> None:
        usd = to_speakable("التكلفة $100 وزادت السرعة 25%")
        self.assertIn("مِيَّة دُولَارْ", usd)
        self.assertIn("خَمْسَة وَعِشْرِينْ بِالْمِيَّة", usd)

    def test_15_arabic_indic_digits_supported(self) -> None:
        indic = to_speakable("الإجمالي ١٥٠ ر.س بنسبة ٢٠٪")
        self.assertIn("مِيَّة وَخَمْسِينْ رِيَالْ", indic)
        self.assertIn("عِشْرِينْ بِالْمِيَّة", indic)

    def test_16_saudi_time_expressions_half_quarter_illa_rub(self) -> None:
        t1 = to_speakable("الساعة 3:30 العصر")
        self.assertIn("السَّاعَة ثَلَاثْ وَنُصّ الْعَصْرْ", t1)

        t2 = to_speakable("4:45 PM")
        self.assertIn("خَمْسْ إِلَّا رُبْعْ الْعَصْرْ", t2)

        t3 = to_speakable("9:15 ص")
        self.assertIn("تِسْعْ وَرُبْعْ الصُّبْحْ", t3)

        t4 = to_speakable("2:20 الظهر")
        self.assertIn("ثِنْتَيْنْ وَثُلْثْ الظُّهْرْ", t4)

    def test_17_iso_and_slash_dates(self) -> None:
        d1 = to_speakable("التاريخ 2026-11-13")
        self.assertIn("ثَلْطَعْشْ نُوفَمْبَرْ أَلْفَيْنْ وَسِتَّة وَعِشْرِينْ", d1)

        d2 = to_speakable("موعدنا 01/05/2026")
        self.assertIn("أَوَّلْ مَايُو أَلْفَيْنْ وَسِتَّة وَعِشْرِينْ", d2)

    def test_18_english_tech_acronyms_expanded(self) -> None:
        out = to_speakable("ربطنا API مع TTS عبر MCP")
        self.assertIn("إِي بِي آيْ", out)
        self.assertIn("تِي تِي إِسْ", out)
        self.assertIn("إِمْ سِي بِي", out)

    def test_19_skip_long_ids_urls_uuids_and_phones(self) -> None:
        raw = (
            "أبشر الطلب 550e8400-e29b-41d4-a716-446655440000 على الرابط "
            "https://example.org/orders/9988776655 ورقم الجوال 0501234567 جاهز."
        )
        res = self.engine.prepare_tts(raw, skip_long_ids=True)
        self.assertGreaterEqual(res.skipped_ids, 3)
        self.assertNotIn("550e8400", res.speakable)
        self.assertNotIn("https://", res.speakable)
        self.assertNotIn("0501234567", res.speakable)
        self.assertIn("أَبْشِرْ", res.speakable)
        self.assertIn("جَاهِزْ", res.speakable)

    def test_20_keep_ids_option_preserves_tokens(self) -> None:
        raw = "الرمز 1234567890"
        res = self.engine.prepare_tts(raw, skip_long_ids=False, expand_numbers=False)
        self.assertIn("1234567890", res.speakable)

    def test_21_clitic_attached_words_vocalized(self) -> None:
        out = to_speakable("وتشوفها بالدوام")
        self.assertIn("وَتِشُوفْهَا", out)

    def test_22_number_to_saudi_words_ranges(self) -> None:
        self.assertEqual(strip_diacritics(number_to_saudi_words(0)), "صفر")
        self.assertEqual(strip_diacritics(number_to_saudi_words(12)), "اثنعش")
        self.assertEqual(strip_diacritics(number_to_saudi_words(200)), "ميتين")
        self.assertEqual(strip_diacritics(number_to_saudi_words(2000)), "ألفين")
        self.assertEqual(strip_diacritics(number_to_saudi_words(5000)), "خمس آلاف")
        self.assertEqual(strip_diacritics(number_to_saudi_words(1_000_000)), "مليون")


class TestSTTEnhancerAndPersonalHook(unittest.TestCase):
    def test_23_pure_silence_hallucination_becomes_empty(self) -> None:
        res = postprocess_transcript("ترجمة نانسي قنقر")
        self.assertEqual(res.text, "")
        self.assertTrue(res.was_hallucination)
        self.assertGreaterEqual(len(res.removed_hallucinations), 1)

    def test_24_trailing_youtube_hallucination_stripped_speech_kept(self) -> None:
        res = postprocess_transcript("أبشر طال عمرك الحين أرسل الملف. اشتركوا في القناة")
        self.assertEqual(res.text, "أبشر طال عمرك الحين أرسل الملف.")
        self.assertFalse(res.was_hallucination)
        self.assertIn("اشتركوا في القناة", res.removed_hallucinations)

    def test_25_orthographic_normalization_consistency(self) -> None:
        raw = "ان شاء الله بشيك على الموضوع الحيين علشان يضبط كل شيء لا هنت"
        res = postprocess_transcript(raw)
        self.assertEqual(
            res.text,
            "إن شاء الله بشيك على الموضوع الحين عشان يضبط كل شي لاهنت",
        )
        self.assertGreaterEqual(res.normalized_count, 4)

    def test_26_elongation_and_repetition_loop_cleanup(self) -> None:
        res = postprocess_transcript("هلااااا والله الشغل مررررة ممتاز تمام تمام تمام تمام تمام")
        self.assertIn("هلا والله", res.text)
        self.assertNotIn("مررررة", res.text)
        self.assertNotIn("تمام تمام تمام تمام", res.text)

    def test_27_personal_intent_adapter_protocol_class(self) -> None:
        class SampleWorkspaceAdapter:
            def adapt_transcript(
                self, text: str, metadata: dict[str, Any] | None = None
            ) -> dict[str, Any]:
                updated = text.replace("مشروع ألفا", "AlphaCore")
                return {
                    "text": updated,
                    "matched_intent": "deploy_alpha" if "AlphaCore" in updated else None,
                }

        adapter = SampleWorkspaceAdapter()
        self.assertIsInstance(adapter, PersonalIntentAdapter)
        res = postprocess_transcript("ارفع تحديث مشروع ألفا الحيين", personal_adapter_hook=adapter)
        self.assertEqual(res.text, "ارفع تحديث AlphaCore الحين")
        self.assertEqual(res.matched_intent, "deploy_alpha")
        self.assertTrue(res.adapter_applied)

    def test_28_dictionary_personal_adapter_from_json_file(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as tmp:
            json.dump(
                {
                    "replacements": {"البيئة التجريبية": "Staging-Cluster"},
                    "intent_aliases": {"Staging-Cluster": "check_staging"},
                },
                tmp,
                ensure_ascii=False,
            )
            adapter_path = Path(tmp.name)

        try:
            adapter = DictionaryPersonalAdapter.from_json_file(adapter_path)
            enhancer = NajdiSTTEnhancer(personal_adapter_hook=adapter)
            res = enhancer.postprocess_transcript("شيكت على البيئة التجريبية ذلحين")
            self.assertEqual(res.text, "شيكت على Staging-Cluster الحين")
            self.assertEqual(res.matched_intent, "check_staging")
        finally:
            adapter_path.unlink(missing_ok=True)

    def test_29_stt_initial_and_carry_prompt_generation(self) -> None:
        p = build_stt_prompt(
            domain="finance",
            extra_terms=["تسوية", "مطابقة"],
            previous_transcript="راجعنا كشف الحساب أمس وكل شيء تمام",
        )
        self.assertIn("تسوية", p["initial_prompt"])
        self.assertIn("كشف الحساب", p["carry_prompt"])
        self.assertIn("كل شي تمام", p["carry_prompt"])
        self.assertLessEqual(len(p["carry_prompt"]), 430)


class TestPromptKitAndNeuralMLX(unittest.TestCase):
    def test_30_positive_system_grounding_profiles(self) -> None:
        for prof in ("modern_white", "executive", "voice_assistant", "technical"):
            bundle = build_system_grounding(profile=prof, domain_terms=["واجهة برمجية"])
            self.assertEqual(bundle.profile, prof)
            self.assertIn("واجهة برمجية", bundle.system_prompt)
            self.assertGreater(bundle.estimated_tokens, 20)

    def test_31_compact_grounding_saves_tokens(self) -> None:
        full = build_system_grounding(profile="modern_white", compact=False)
        compact = build_system_grounding(profile="modern_white", compact=True)
        self.assertLess(compact.estimated_tokens, full.estimated_tokens)

    def test_32_harmonize_non_saudi_and_stiff_msa_drift(self) -> None:
        kit = NajdiPromptKit()
        out = kit.harmonize_text("بدي أشوف التقرير هلق عشان نعتمد")
        self.assertIn("أبي", out["harmonized"])
        self.assertIn("الحين", out["harmonized"])
        self.assertGreaterEqual(out["replacements"], 2)

    def test_33_mlx_homograph_disambiguation_and_offline_fallback(self) -> None:
        dis = OptionalMLXDisambiguator(base_url="http://127.0.0.1:9", timeout_sec=0.05)
        res = dis.disambiguate("شغل السيرفر وقفل التذكرة الحين")
        self.assertEqual(res.backend, "deterministic_fallback")
        self.assertTrue(res.used_fallback)
        self.assertIn("شَغِّلْ", res.speakable)
        self.assertIn("قَفِّلْ", res.speakable)

    def test_34_mlx_openai_compatible_local_http_endpoint(self) -> None:
        class MockMLXHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                _ = self.rfile.read(length)
                reply = {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "شَغِّلْ السَّيْرْفَرْ الْحِينْ",
                            }
                        }
                    ]
                }
                raw = json.dumps(reply, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

        server = HTTPServer(("127.0.0.1", 0), MockMLXHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()

        try:
            dis = OptionalMLXDisambiguator(base_url=f"http://127.0.0.1:{port}/v1", timeout_sec=2.0)
            res = dis.disambiguate("شغل السيرفر الحين")
            self.assertEqual(res.backend, "local_mlx_http")
            self.assertFalse(res.used_fallback)
            self.assertEqual(res.speakable, "شَغِّلْ السَّيْرْفَرْ الْحِينْ")
        finally:
            server.server_close()


if __name__ == "__main__":
    unittest.main()
