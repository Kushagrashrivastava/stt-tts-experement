import whisper

MODEL_SIZE = "base"


def transcribe_audio(file_path: str = "outputs/recorded_clean.wav") -> str:
    """Transcribe a WAV file to text using OpenAI Whisper (local)."""
    print(f"Loading Whisper '{MODEL_SIZE}' model...")
    model = whisper.load_model(MODEL_SIZE)

    print(f"Transcribing {file_path}...")
    result = model.transcribe(file_path, fp16=False)
    text = result["text"].strip()

    print(f"Transcription: {text}")
    return text


if __name__ == "__main__":
    transcribe_audio()
