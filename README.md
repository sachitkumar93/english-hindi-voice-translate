# Real-Time Voice-to-Voice Translation Pipeline
### English → Hindi | Live Audio → ASR → Translation → TTS

A real-time pipeline that captures live English speech, transcribes it, translates it to Hindi, and plays back natural Hindi audio — end to end in ~2 seconds on a laptop.

```
🎙 MIC → [VAD] → [AudioBuffer] → [Whisper ASR] → [Groq LLM] → [edge-tts] → 🔊 SPEAKER
```

---

## Demo

```
EN: Can we move the meeting to 5 PM tomorrow?
HI: क्या हम मीटिंग को कल शाम 5 बजे कर सकते हैं?
Playing...  ⏱ 1847ms

EN: Push the changes to GitHub and deploy on AWS.
HI: GitHub पर बदलाव करें और AWS पर deploy करें।
Playing...  ⏱ 1723ms

EN: Sir, could you please review this report by EOB?
HI: सर, क्या आप इस रिपोर्ट को दिन के अंत तक देख सकते हैं?
Playing...  ⏱ 1915ms
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         AUDIO THREAD                            │
│  Microphone → sounddevice callback (30ms frames) → VAD check    │
└─────────────────────────┬───────────────────────────────────────┘
                          │ speech frames only
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AUDIO BUFFER                               │
│  Accumulates frames → detects silence → flushes utterance       │
└─────────────────────────┬───────────────────────────────────────┘
                          │ complete utterance (bytes)
                          │ asyncio.run_coroutine_threadsafe()
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ASYNC PIPELINE                              │
│                                                                 │
│  [faster-whisper ASR] → English text + confidence score         │
│          ↓                                                      │
│  [Confidence Gate] → discard if score < 0.55                    │
│          ↓                                                      │
│  [Filler Removal] → strip uh/um/basically/you know              │
│          ↓                                                      │
│  [Groq LLM Translation] → Hindi text                            │
│          ↓                                                      │
│  [edge-tts] → MP3 audio bytes                                   │
│          ↓                                                      │
│  [PlaybackQueue] → decoded PCM → speaker                        │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decision: Thread → Async Bridge
`sounddevice` audio callbacks run in a dedicated OS thread. The ASR/translation/TTS pipeline is async. These two worlds are bridged with `asyncio.run_coroutine_threadsafe()` — the audio thread schedules async work on the event loop without blocking either side.

---

## Tech Stack

| Stage | Component | Why |
|---|---|---|
| Voice Activity Detection | webrtcvad | Google's WebRTC VAD — fast, accurate, no ML overhead |
| ASR | faster-whisper (base, int8) | 4x faster than vanilla Whisper, runs locally, no API cost |
| Translation | Groq llama-3.3-70b-versatile | ~400ms from India vs ~1500ms with OpenAI US servers |
| TTS | Microsoft edge-tts (hi-IN-SwaraNeural) | Free, no API key, neural voice quality |
| Audio I/O | sounddevice + pydub | Separate input/output streams for barge-in support |

---

## Edge Cases Handled

### Filler Word Removal
Words like "uh", "um", "basically", "you know", "I mean" are stripped before translation — they have no semantic content and produce unnatural Hindi output.

```
Input:  "Uh, so basically, we need to reschedule."
Output: "हमें फिर से शेड्यूल करना है।"
```

### Named Entity Preservation
Technical brand names are explicitly protected — a translation model would otherwise render "Docker" or "GitHub" as meaningless Hindi phonetics.

```
Input:  "Push the code to GitHub and deploy on AWS."
Output: "GitHub पर code डालें और AWS पर deploy करें।"
```

### Time & Date Localization
Times and dates are converted to natural Hindi expressions, not transliterated.

```
"5 PM"      → "शाम 5 बजे"
"10 AM"     → "सुबह 10 बजे"
"tomorrow"  → "कल"
"Monday"    → "सोमवार"
```

### Abbreviation Expansion
Common workplace abbreviations are expanded naturally.

```
"EOB"  → "दिन के अंत तक"
"ETA"  → "पहुँचने का अनुमानित समय"
"ASR"  → "ऑटोमेटेड स्पीच रिकग्निशन"
```

### Formal vs Casual Tone Detection
The LLM detects formality from vocabulary and applies the correct Hindi register.

```
"Sir, could you please review this?" → आप form (formal)
"Hey bro, send me that link."       → तुम form (casual)
```

### Low Confidence Filtering
Whisper produces per-word probability scores. Utterances below 0.55 average confidence are discarded — this filters background noise and TTS audio leaking back into the microphone.

### Barge-in Interruption
If the user speaks while TTS is playing, the current audio cuts off immediately and the new utterance is processed. Uses a dedicated `sd.OutputStream` so stopping output never affects the microphone `sd.InputStream`.

### Domain Vocabulary Priming
Whisper's `initial_prompt` is seeded with technical vocabulary to prevent common mishearings (e.g. "pull request" → "full request").

---

## Latency

Measured on MacBook Air, India, hitting overseas APIs:

```
VAD + buffering:   ~0ms    (real-time, no network)
Whisper ASR:       ~500ms  (local inference, beam_size=1)
Groq translation:  ~400ms  (vs ~1500ms with OpenAI from India)
edge-tts:          ~900ms  (Microsoft neural TTS, network)
─────────────────────────────────────────────────────
Total:             ~1.8s   consistent, no spikes
```

### Optimization History
| Change | Before | After |
|---|---|---|
| beam_size=1 (ASR) | 4000ms spikes | ~500ms consistent |
| OpenAI → Groq (Translation) | ~1500ms | ~400ms |
| condition_on_previous_text=False | Hallucinations | Clean output |

---

## Project Structure

```
voice_translator/
├── config.py          # All constants and tuning knobs
├── audio_buffer.py    # VAD + utterance boundary detection
├── asr.py             # faster-whisper speech recognition
├── translator.py      # Filler removal + Groq LLM translation
├── tts.py             # edge-tts Hindi speech synthesis
├── playback.py        # Audio output queue + barge-in handling
└── main.py            # Pipeline orchestrator
```

Each file has exactly one responsibility. Swapping any component (e.g. Whisper → Deepgram, edge-tts → Google WaveNet) requires touching only that file.

---

## Setup

### Prerequisites
```bash
brew install portaudio ffmpeg
```

### Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables
```bash
export OPENAI_API_KEY="sk-..."    # optional, not used in current stack
export GROQ_API_KEY="gsk-..."     # required for translation
```

### Audio Device Configuration
If using Bluetooth headphones, Mac requires separate input/output devices. Find your device numbers:
```bash
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```

Update `main.py`:
```python
sd.default.device = (INPUT_DEVICE_NUMBER, OUTPUT_DEVICE_NUMBER)
```

### Run
```bash
python main.py
```

---

## Production Improvements

### Latency
- **Deepgram streaming API** — partial transcripts every ~200ms instead of waiting for silence, reduces perceived latency by ~1s
- **TTS streaming** — start playing first audio chunk while rest is still generating, hides synthesis latency
- **GPU inference** — Whisper on NVIDIA GPU drops from ~500ms to ~50ms
- **Regional API endpoints** — Groq/OpenAI have regional endpoints that cut network RTT by 100-200ms from India

### Reliability
- **Confidence calibration per speaker** — baseline confidence adapts to the user's voice over time
- **Dynamic vocabulary injection** — domain glossary built from recent conversation, not hardcoded
- **Retry on low confidence** — instead of skipping, prompt user to repeat if confidence < threshold
- **Speaker diarization** — multiple speakers handled independently with per-speaker context windows

### Scale
- **Streaming translation** — translate sentence-by-sentence as ASR produces partial results
- **Context persistence** — conversation history stored across sessions, not just last 4 turns
- **Tone profile per conversation** — detect formality once at conversation start, not per utterance

---

## Known Limitations

- **Bluetooth HFP conflict** — using the same Bluetooth device for mic input and audio output on macOS triggers HFP mode which breaks audio quality. Workaround: use built-in mic for input, Bluetooth for output only.
- **edge-tts network dependency** — TTS requires internet access to Microsoft servers. Offline alternative: Coqui TTS with a Hindi model (higher latency, lower quality).
- **Whisper base model** — adequate for clear speech, struggles with heavy accents or noisy environments. Upgrade path: `small` or `medium` model at the cost of +200-500ms latency.
- **Single speaker** — pipeline assumes one speaker. Multi-speaker scenarios cause context confusion.