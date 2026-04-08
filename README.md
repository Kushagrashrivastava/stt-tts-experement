# STT-TTS: Noise-Resilient Voice Round-Trip Pipeline

A local pipeline that records speech from your microphone, removes background noise, transcribes it to text, and synthesizes clean speech back — all running offline on your machine.

Built to test how well a voicebot pipeline holds up in noisy environments.

## How It Works

```
Microphone → Raw WAV → Noise Reduction → VAD → Cleaned WAV → Whisper STT → Text → Coqui TTS → Output WAV
```

**Three layers of noise handling:**

1. **Spectral Noise Reduction** (`noisereduce`) — removes stationary background noise (fans, AC, ambient hum) via spectral gating
2. **Voice Activity Detection** (Silero VAD) — detects speech segments by pitch and energy, discards silence and noise-only gaps
3. **Whisper** — trained on noisy real-world audio, handles residual noise during transcription

## Project Structure

```
STT-TTS/
├── pipeline.py          # Full end-to-end pipeline (run this)
├── record_audio.py      # Mic recording → WAV
├── denoise_audio.py     # Noise reduction + VAD
├── audio_to_text.py     # Whisper STT
├── text_to_audio.py     # Coqui TTS
├── requirements.txt
└── outputs/
    ├── recorded_raw.wav     # Raw mic recording (with noise)
    ├── recorded_clean.wav   # After denoising (voice only)
    └── output.wav           # Final TTS-generated speech
```

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install "coqui-tts[codec,languages]"
```

## Usage

### Full pipeline (recommended)

```bash
source .venv/bin/activate
python pipeline.py
```

This will:
1. Record from your mic (press **Enter** to stop)
2. Denoise the recording and strip non-speech segments
3. Transcribe the cleaned audio to text via Whisper
4. Generate speech from the transcription via Coqui TTS

Compare `outputs/recorded_raw.wav`, `outputs/recorded_clean.wav`, and `outputs/output.wav` to evaluate the pipeline.

### Individual scripts

```bash
# Record from microphone
python record_audio.py

# Denoise an existing recording
python denoise_audio.py

# Transcribe a WAV file to text
python audio_to_text.py

# Generate speech from text
python text_to_audio.py
```

## Tech Stack

| Component | Library | Role |
|-----------|---------|------|
| Recording | `sounddevice` | Mic capture, 16kHz mono WAV |
| Noise Reduction | `noisereduce` | Spectral gating to remove stationary noise |
| Voice Activity Detection | Silero VAD | Keeps only speech segments |
| Speech-to-Text | OpenAI Whisper (`base`) | Local offline transcription |
| Text-to-Speech | Coqui TTS (FastPitch) | Neural speech synthesis |
