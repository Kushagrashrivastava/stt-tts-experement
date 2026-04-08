import torch
from TTS.api import TTS
import gradio as gr
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_audio(
    text: str = "A journey of a thousand miles begins with a single step",
    file_path: str = "outputs/output.wav",
) -> str:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    tts = TTS(model_name="tts_models/en/ljspeech/fast_pitch").to(device)
    tts.tts_to_file(text=text, file_path=file_path)
    return file_path


if __name__ == "__main__":
    print(generate_audio())
