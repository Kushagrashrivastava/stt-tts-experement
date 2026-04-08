import ssl
import os
import torch
import numpy as np
import scipy.io.wavfile as wav
import noisereduce as nr
from pathlib import Path

# macOS Python often ships without root certs — use certifi if available
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    ssl._create_default_https_context = ssl._create_unverified_context

SAMPLE_RATE = 16000


def _load_wav(file_path: str) -> np.ndarray:
    rate, data = wav.read(file_path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32767.0
    if rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE}Hz, got {rate}Hz")
    return data


def _reduce_noise(audio: np.ndarray) -> np.ndarray:
    """Spectral-gating noise reduction — removes stationary background noise."""
    return nr.reduce_noise(y=audio, sr=SAMPLE_RATE, prop_decrease=0.8)


def _vad_filter(audio: np.ndarray) -> np.ndarray:
    """Use Silero VAD to keep only speech segments."""
    model, utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", trust_repo=True
    )
    get_speech_timestamps, _, read_audio, _, _ = utils

    tensor = torch.from_numpy(audio).float()
    timestamps = get_speech_timestamps(tensor, model, sampling_rate=SAMPLE_RATE)

    if not timestamps:
        print("  VAD found no speech — returning full audio.")
        return audio

    speech_chunks = [audio[ts["start"]: ts["end"]] for ts in timestamps]
    filtered = np.concatenate(speech_chunks)

    original_dur = len(audio) / SAMPLE_RATE
    filtered_dur = len(filtered) / SAMPLE_RATE
    print(f"  VAD: {original_dur:.1f}s -> {filtered_dur:.1f}s (kept speech only)")
    return filtered


def denoise_audio(
    input_path: str = "outputs/recorded_raw.wav",
    output_path: str = "outputs/recorded_clean.wav",
) -> str:
    """Two-stage cleaning: spectral noise reduction, then VAD filtering."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path}")
    audio = _load_wav(input_path)

    print("Stage 1: Spectral noise reduction...")
    audio = _reduce_noise(audio)

    print("Stage 2: Voice Activity Detection...")
    audio = _vad_filter(audio)

    audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    wav.write(output_path, SAMPLE_RATE, audio_int16)
    print(f"Saved cleaned audio to {output_path}")
    return output_path


if __name__ == "__main__":
    denoise_audio()
