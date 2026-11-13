---
name: najdikit
description: Real-time Modern White Saudi & Urban Najdi voice-call streaming (low-TTFA chunking, full-duplex barge-in/backchannel turn analysis), multi-backend TTS phonetics (ElevenLabs, Habibi-TTS, OmniVoice, OpenAI, SIP 8kHz), STT cleaning, Gulf morphology, dialect scoring, and 1,250+ modular lexicon lookup.
---

# NajdiKit v2.0 Skill (`najdikit`)

Use **NajdiKit v2.0** whenever your workflow involves **Modern White Saudi** or **Urban Najdi** Arabic (`اللهجة السعودية البيضاء العصرية والنجدية الحضرية`) across **real-time voice calls & telephony**, speech synthesis (`TTS`), speech recognition (`STT`), dialect purity evaluation, or conversational LLM grounding.

## Available Tools (7 Tools)

### 1. `najdi_tts_prepare`
Transforms raw Arabic or mixed Arabic/English text into phonetically tuned input for TTS engines (`backend`: `default`, `elevenlabs`, `habibi`, `omnivoice`, `openai`, `sip_8khz`):
- Applies Urban Najdi/White Saudi imperfective prefix kasrah (`تِعْرِفْ`, `يِقُولْ`, `نِقْدَرْ`, `تِشُوفْ`) and Gulf/Najdi clitic stem inheritance (`وَبِنْشَيِّكْهَا`, `هَاالْفَاتُورَةْ`, `بِمُكَالَمَاتْهُمْ`).
- Wraps `SWivid/Habibi-TTS` Saudi dialect tokens (`②〈...〉`) in `habibi` mode and respells gemination (`شَيْيَكْتْ`) in `omnivoice` mode.
- Expands currencies, percentages, Saudi time phrases (`4:45 العصر` → `خَمْسْ إِلَّا رُبْعْ الْعَصْرْ`), dates, and English acronyms while stripping unreadable UUIDs/URLs/phones.

### 2. `najdi_call_stream_chunk`
Stateful low-TTFA voice-call chunker for live phone conversations:
- **First-Clause Fast Lane**: Emits Clause 0 immediately on Saudi openers (`أبشر،`, `سم،`, `هلا والله،`) so audio playback begins while the LLM is still generating.
- **Atomic Guard**: Never splits inside decimals/currencies/times (`1,250.50 ريال`, `4:30`) or inseparable Saudi constructs (`إن شاء الله`, `طال عمرك`, `في أمان الله`).
- **Breath-Group Overflow Splitter & Action Extractor**: Breaks long unpunctuated output cleanly before Saudi connectors (`وعشان`, `وبعدين`, `بس`) and strips `[ACTION: ...]` tags without leaking brackets to TTS.

### 3. `najdi_call_turn_analyze`
Sub-`10 µs` full-duplex caller turn & VAD decision engine:
- Distinguishes `"backchannel"` (`طيب`, `إيه`, `أيوه`, `كمّل` → `should_abort_tts: false`) from `"barge_in"` (`لحظة`, `وقف`, `لا لا` → `should_abort_tts: true`).
- Detects `"hold_request"` (`ثواني`, `خلك معي` → `recommended_wait_sec: 20.0`), `"closing"` (`في أمان الله`, `توصي شي`), and `"incomplete_thought"` (trailing `يعني`, `عشان`, `اللي` → `endpoint_grace_ms: 400`).

### 4. `najdi_stt_clean`
Cleans raw ASR transcripts from `Whisper` or `Scribe` in single-pass `O(N)` time:
- Strips silence/subtitle hallucinations (`ترجمة نانسي`, `اشتركوا في القناة`, repetition loops).
- Normalizes White Saudi & Gulf clitic orthography (`انشاءالله` → `إن شاء الله`, `وذلحين` → `والحين`, `كل شيء` → `كل شي`) with optional runtime `PersonalIntentAdapter` hooks.

### 5. `najdi_dialect_score`
Evaluates `saudi_alignment_score` (`0–100`), `msa_leakage_score`, `foreign_dialect_score`, and `call_suitability_score` in `< 30 µs` with exact flagged spans and natural White Saudi replacement suggestions.

### 6. `najdi_style_grounding`
Builds concise, affirmative System Prompt Grounding blocks (`modern_white`, `executive`, `voice_assistant`, `voice_call_realtime`, `technical`) so LLMs speak authentic Modern White Saudi naturally.

### 7. `najdi_lexicon_lookup`
Queries the `1,250+` entry Modern White Saudi & Urban Najdi atlas across `core` and modular domain packs (`telephony_calls`, `sada_conversational`, `commerce_fintech`, `service_appointments`).

## CLI Quick Reference

```bash
python3 -m najdikit speakable --backend habibi "أبشر، شيكت على هالفاتورة بمبلغ 150 ريال"
python3 -m najdikit call-chunks "أبشر، شيكت لك على الطلب بمبلغ 1,250.50 ريال [ACTION: send_sms]."
python3 -m najdikit turn-check --ai-speaking "طيب كمّل"
python3 -m najdikit score "سوف أقوم بالتحقق من الأمر الذي ذكرته حسناً"
python3 -m najdikit morph "وبنشيكها"
python3 -m najdikit packs
python3 -m najdikit grounding --profile voice_call_realtime
python3 -m najdikit bench
```
