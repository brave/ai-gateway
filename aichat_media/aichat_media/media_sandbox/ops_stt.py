"""STT WAV decode ops for the media sandbox (PCM WAV only, stdlib wave + numpy)."""

from __future__ import annotations

import base64
import io
import logging
import wave

import numpy as np

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000


def _resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    new_len = max(1, round(audio.size * target_sr / orig_sr))
    old_x = np.arange(audio.size, dtype=np.float64)
    new_x = np.linspace(0.0, audio.size - 1, new_len)
    return np.interp(new_x, old_x, audio.astype(np.float64)).astype(np.float32)


def op_stt_decode(
    *,
    audio_b64: str,
    max_duration_seconds: float,
    **_ignored,
) -> dict:
    """
    Validate + decode a PCM WAV upload: header checks, mono downmix, resample
    to TARGET_SAMPLE_RATE. Returns base64 float32 mono buffer + duration.
    """
    audio_bytes = base64.b64decode(audio_b64)

    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            if wf.getcomptype() != "NONE":
                raise ValueError(
                    "Only PCM WAV is supported (compressed WAVE formats are not allowed)"
                )
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            if framerate <= 0 or n_frames < 0:
                raise ValueError("Invalid WAV parameters")
            duration_sec = n_frames / framerate
            if duration_sec > max_duration_seconds:
                raise ValueError(
                    f"Audio duration ({duration_sec:.1f}s) exceeds maximum "
                    f"({max_duration_seconds:.0f}s)"
                )
            raw = wf.readframes(n_frames)
    # wave raises bare RuntimeError when chunk-skip seeks past EOF on
    # malformed input; fold it into the friendly error.
    except (wave.Error, EOFError, RuntimeError) as e:
        raise ValueError(
            "Only PCM WAV is supported (OpenAI-compatible uploads like mp3 require a "
            f"full decoder). {e}"
        ) from e

    if sampwidth == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        # 32-bit PCM integer
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float64)
        audio = (audio / 2147483648.0).astype(np.float32)
    else:
        raise ValueError(
            f"Unsupported WAV sample width {sampwidth} bytes; use 8/16/32-bit PCM"
        )

    if n_channels > 1:
        if audio.size % n_channels != 0:
            raise ValueError("Corrupt WAV frame buffer")
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    duration_ms = float(len(audio) / framerate * 1000.0) if framerate else 0.0
    audio = _resample_linear(audio, framerate, TARGET_SAMPLE_RATE)
    if audio.size == 0:
        raise ValueError("Empty audio")

    pcm = audio.astype(np.float32)
    return {
        "pcm_b64": base64.b64encode(pcm.tobytes()).decode("ascii"),
        "n_samples": int(pcm.size),
        "sample_rate": TARGET_SAMPLE_RATE,
        "duration_ms": duration_ms,
    }


def preload() -> None:
    # Warm numpy's C extensions before the audit hook is installed.
    np.zeros(1, dtype=np.float32)


OP_HANDLERS = {
    "stt_decode": op_stt_decode,
}
