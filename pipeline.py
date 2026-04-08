"""
End-to-end voice round-trip pipeline:
  1. Record from mic         -> outputs/recorded_raw.wav
  2. Denoise + VAD           -> outputs/recorded_clean.wav
  3. Whisper STT             -> text
  3.5 LLM correction         -> corrected text
  4. Coqui TTS               -> outputs/output.wav
"""

from record_audio import record_audio
from denoise_audio import denoise_audio
from audio_to_text import transcribe_audio, save_transcription
from correct_text import correct_transcription, save_correction
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
    raw_text = transcribe_audio(clean_path)
    save_transcription(raw_text, "outputs/transcription_raw.txt")

    print("\n" + "=" * 50)
    print("  STEP 3.5 — LLM Correction (GPT-4o-mini)")
    print("=" * 50)
    corrected_text = correct_transcription(raw_text)
    txt_path = save_correction(corrected_text, "outputs/transcription.txt")

    print("\n" + "=" * 50)
    print("  STEP 4 — Text-to-Speech (Coqui TTS)")
    print("=" * 50)
    output_path = generate_audio(corrected_text, "outputs/output.wav")

    print("\n" + "=" * 50)
    print("  SUMMARY")
    print("=" * 50)
    print(f"  Raw recording   : {raw_path}")
    print(f"  Cleaned audio   : {clean_path}")
    print(f"  Raw transcript  : {raw_text}")
    print(f"  Corrected text  : {corrected_text}")
    print(f"  Transcription   : {txt_path}")
    print(f"  TTS output      : {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()
