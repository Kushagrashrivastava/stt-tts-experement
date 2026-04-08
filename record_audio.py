import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path

SAMPLE_RATE = 16000
CHANNELS = 1


def record_audio(file_path: str = "outputs/recorded_raw.wav") -> str:
    """Record from the default microphone until the user presses Enter."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Recording at {SAMPLE_RATE}Hz... Press Enter to stop.\n")
    frames: list[np.ndarray] = []

    def callback(indata, frame_count, time_info, status):
        if status:
            print(f"  ⚠ {status}")
        frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=callback,
    )
    with stream:
        input()

    audio = np.concatenate(frames, axis=0).flatten()
    audio_int16 = np.int16(audio * 32767)
    wav.write(file_path, SAMPLE_RATE, audio_int16)

    duration = len(audio) / SAMPLE_RATE
    print(f"Saved {duration:.1f}s of audio to {file_path}")
    return file_path


if __name__ == "__main__":
    record_audio()
