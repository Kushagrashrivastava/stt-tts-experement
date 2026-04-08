"""
End-to-end voice round-trip pipeline:
  1. Record from mic         -> outputs/recorded_raw.wav
  2. Denoise + VAD           -> outputs/recorded_clean.wav
  3. Whisper STT             -> text
  4. Coqui TTS               -> outputs/output.wav
"""

from record_audio import record_audio
from denoise_audio import denoise_audio
from audio_to_text import transcribe_audio
from text_to_audio import generate_audio


def run_pipeline():
    print("=" * 50)
    print("  STEP 1 — Record from microphone")
    print("=" * 50)
    raw_path = record_audio("outputs/recorded_raw.wav")

    print("\n" + "=" * 50)
    print("  STEP 2 — Denoise + Voice Activity Detection")
    print("=" * 50)
    clean_path = denoise_audio(raw_path, "outputs/recorded_clean.wav")

    print("\n" + "=" * 50)
    print("  STEP 3 — Speech-to-Text (Whisper)")
    print("=" * 50)
    text = transcribe_audio(clean_path)

    print("\n" + "=" * 50)
    print("  STEP 4 — Text-to-Speech (Coqui TTS)")
    print("=" * 50)
    output_path = generate_audio(text, "outputs/output.wav")

    print("\n" + "=" * 50)
    print("  SUMMARY")
    print("=" * 50)
    print(f"  Raw recording : {raw_path}")
    print(f"  Cleaned audio : {clean_path}")
    print(f"  Transcription : {text}")
    print(f"  TTS output    : {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()
