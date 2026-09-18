"""Validation helpers for STT multipart uploads (PCM WAV only)."""

from __future__ import annotations

import os

from fastapi import UploadFile

_RIFF_MAGIC = b"RIFF"
_WAVE_MAGIC = b"WAVE"
_MIN_RIFF_WAVE_LEN = 12


def sanitize_stt_filename(filename: str | None) -> str:
    """
    Return a safe basename for logging/metadata. Rejects null bytes; strips path
    components so uploads cannot smuggle traversal segments in the filename.
    """
    if not filename:
        return "audio.wav"
    if "\x00" in filename:
        raise ValueError("Invalid filename")
    # Normalize path separators before basename (clients may send backslashes).
    name = os.path.basename(filename.replace("\\", "/"))
    if not name or name in (".", ".."):
        return "audio.wav"
    return name


def assert_pcm_wav_riff_header(audio_bytes: bytes) -> None:
    """Reject non-WAV payloads regardless of declared Content-Type."""
    if len(audio_bytes) < _MIN_RIFF_WAVE_LEN:
        raise ValueError(
            "Invalid audio file: expected PCM WAV (RIFF/WAVE header missing)"
        )
    if audio_bytes[:4] != _RIFF_MAGIC or audio_bytes[8:12] != _WAVE_MAGIC:
        raise ValueError(
            "Invalid audio file: expected PCM WAV (RIFF/WAVE header missing)"
        )


def validate_pcm_wav_upload(
    audio_bytes: bytes,
    *,
    max_bytes: int,
) -> None:
    """
    Cheap in-process pre-checks (size + RIFF/WAVE magic). Full WAV header
    validation and decoding happen in the media sandbox (stt_decode op).
    """
    if not audio_bytes:
        raise ValueError("empty audio file")
    if len(audio_bytes) > max_bytes:
        max_mb = max_bytes / (1024 * 1024)
        raise ValueError(f"Audio file exceeds maximum upload size ({max_mb:.0f} MB)")

    assert_pcm_wav_riff_header(audio_bytes)


async def read_stt_upload_limited(upload: UploadFile, max_bytes: int) -> bytes:
    """Read upload body, failing before buffering more than max_bytes."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            max_mb = max_bytes / (1024 * 1024)
            raise ValueError(
                f"Audio file exceeds maximum upload size ({max_mb:.0f} MB)"
            )
        chunks.append(chunk)
    return b"".join(chunks)
