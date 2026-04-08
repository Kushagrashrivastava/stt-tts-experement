from pathlib import Path

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


def save_transcription(text: str, output_path: str = "outputs/transcription.txt") -> str:
    """Save transcription text to a .txt file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(text, encoding="utf-8")
    print(f"Saved transcription to {output_path}")
    return output_path


if __name__ == "__main__":
    text = transcribe_audio()
    save_transcription(text)
