from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = (
    "You are a transcription correction assistant. "
    "Fix ONLY misheard words, spelling errors, and grammar mistakes "
    "in the following speech-to-text transcription. "
    "Do NOT change the meaning, rephrase sentences, add new ideas, "
    "or remove any content. Return ONLY the corrected text."
)


def correct_transcription(text: str) -> str:
    """Send raw transcription to GPT-4o-mini for error correction."""
    client = OpenAI()

    print("Sending transcription to GPT-4o-mini for correction...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
    )

    corrected = response.choices[0].message.content.strip()
    print(f"  Original : {text}")
    print(f"  Corrected: {corrected}")
    return corrected


def save_correction(text: str, output_path: str = "outputs/transcription.txt") -> str:
    """Save corrected text to a file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(text, encoding="utf-8")
    print(f"Saved corrected transcription to {output_path}")
    return output_path


if __name__ == "__main__":
    raw = Path("outputs/transcription.txt").read_text(encoding="utf-8").strip()
    corrected = correct_transcription(raw)
    save_correction(corrected)
