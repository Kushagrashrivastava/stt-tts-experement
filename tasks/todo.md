# STT-TTS Round-Trip Pipeline (with Noise Reduction)

## Goal

Build a noise-resilient voice pipeline: Mic -> Denoise -> STT (Whisper) -> TTS (Coqui) -> WAV.
The system should isolate the speaker's voice from background noise before transcription.

## Plan

- [x] **1. `record_audio.py`** — Record mic input via `sounddevice`, save 16kHz mono WAV
- [x] **2. `denoise_audio.py`** — Spectral noise reduction (`noisereduce`) + Silero VAD to keep only speech segments
- [x] **3. `audio_to_text.py`** — Whisper STT to transcribe cleaned WAV to text
- [x] **4. `pipeline.py`** — End-to-end script chaining: record -> denoise -> STT -> TTS
- [x] **5. `requirements.txt`** — Add `sounddevice`, `noisereduce`, `openai-whisper`
- [x] **6. Install & verify** — All deps installed, all imports verified (including SSL fix for Silero VAD on macOS)

## Output Files

| File | Description |
|------|-------------|
| `outputs/recorded_raw.wav` | Raw mic recording (with noise) |
| `outputs/recorded_clean.wav` | After noise reduction + VAD (voice only) |
| `outputs/output.wav` | Final TTS output from transcribed text |

## Review

- All 4 scripts created: `record_audio.py`, `denoise_audio.py`, `audio_to_text.py`, `pipeline.py`
- All imports verified: sounddevice, noisereduce, Silero VAD, Whisper, Coqui TTS
- SSL cert issue on macOS handled in `denoise_audio.py` (certifi fallback + unverified context)
- Mac mic detected: "MacBook Air Microphone"
- Ready for live test: `python pipeline.py` (requires mic access, must be run in terminal)
