---
name: najdikit
description: Prepare natural Modern White Saudi & Urban Najdi Arabic TTS speech (phonetic G2P & tashkeel), clean Whisper/Scribe STT transcripts, look up Saudi dialect vocabulary, and generate positive LLM system grounding prompts.
---

# NajdiKit Skill (`najdikit`)

Use **NajdiKit** whenever your workflow involves **Modern White Saudi** or **Urban Najdi** Arabic (`اللهجة السعودية البيضاء العصرية والنجدية الحضرية`) across voice synthesis (`TTS`), speech recognition (`STT`), dialect-accurate conversational agents, or lexical lookup.

## Available Tools

### 1. `najdi_tts_prepare`
Transforms raw Arabic or mixed Arabic/English text into phonetically tuned input for TTS engines (`ElevenLabs`, `OmniVoice`, `OpenAI Audio`, etc.):
- Applies Urban Najdi/White Saudi imperfective prefix kasrah (`تِعْرِفْ`, `يِقُولْ`, `نِقْدَرْ`, `تِشُوفْ`).
- Applies natural conversational pausal sukun (`أَبْشِرْ`, `كِلْ شَيْ`, `الْحِينْ`, `تَمَامْ`) instead of stiff MSA case endings.
- Expands currencies (`150.50 ريال` → `مِيَّة وَخَمْسِينْ رِيَالْ وَخَمْسِينْ هَلَلَة`), percentages (`25%` → `خَمْسَة وَعِشْرِينْ بِالْمِيَّة`), times (`4:45 العصر` → `خَمْسْ إِلَّا رُبْعْ الْعَصْرْ`), dates, and English acronyms (`API` → `إِي بِي آيْ`).
- Automatically omits unreadable UUIDs, URLs, long reference IDs, and phone numbers when `skip_long_ids: true`.

### 2. `najdi_stt_clean`
Cleans raw ASR transcripts from `Whisper` or `Scribe` and generates anti-hallucination conditioning prompts:
- Strips silence/subtitle hallucinations (`ترجمة نانسي`, `اشتركوا في القناة`, `Amara.org`, repetition loops).
- Normalizes White Saudi orthography (`انشاءالله` → `إن شاء الله`, `الحيين`/`ذلحين` → `الحين`, `علشان` → `عشان`, `كل شيء` → `كل شي`).
- Supports optional runtime `PersonalIntentAdapter` JSON overlays without storing any personal data in the repository.

### 3. `najdi_style_grounding`
Builds concise, affirmative System Prompt Grounding blocks (`modern_white`, `executive`, `voice_assistant`, `technical`) so LLMs (`Claude`, `GPT`, `Gemini`, `Qwen`, `Grok`) converse in authentic Modern White Saudi without drifting into Levantine, Egyptian, or translated stiff MSA.

### 4. `najdi_lexicon_lookup`
Queries the 600+ entry Modern White Saudi & Urban Najdi atlas (`white_saudi_lexicon.json`) for vocalizations, IPA hints, meanings, examples, and dialect anti-confusion checks.

## CLI Quick Reference

```bash
python3 -m najdikit speakable "أبشر، شيكت على الفاتورة الحين بمبلغ 150 ريال"
python3 -m najdikit normalize "ان شاء الله بشيك ع الموضوع الحيين لا هنت"
python3 -m najdikit stt-prompt --domain tech --terms "داشبورد,سيرفر"
python3 -m najdikit grounding --profile modern_white
python3 -m najdikit lookup "يمديك"
python3 -m najdikit bench
```
