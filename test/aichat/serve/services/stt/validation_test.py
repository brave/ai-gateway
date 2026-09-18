"""Unit tests for STT upload validation."""

import io
import wave

import pytest

from aichat.serve.services.stt.validation import (
    assert_pcm_wav_riff_header,
    sanitize_stt_filename,
    validate_pcm_wav_upload,
)


def _tiny_wav(duration_frames: int = 80, framerate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        silence = (0).to_bytes(2, byteorder="little", signed=True) * duration_frames
        wf.writeframes(silence)
    return buf.getvalue()


class TestSanitizeSttFilename:
    def test_basename_strips_path(self):
        assert sanitize_stt_filename("../../etc/passwd") == "passwd"
        assert sanitize_stt_filename("..\\..\\clip.wav") == "clip.wav"

    def test_null_byte_rejected(self):
        with pytest.raises(ValueError, match="Invalid filename"):
            sanitize_stt_filename("a\x00b.wav")

    def test_default_when_empty(self):
        assert sanitize_stt_filename(None) == "audio.wav"
        assert sanitize_stt_filename("") == "audio.wav"


class TestPcmWavValidation:
    def test_riff_wave_required(self):
        with pytest.raises(ValueError, match="RIFF/WAVE"):
            assert_pcm_wav_riff_header(b"not wav")
        with pytest.raises(ValueError, match="RIFF/WAVE"):
            assert_pcm_wav_riff_header(b"RIFF\x00\x00\x00\x00MP3 ")

    def test_accepts_valid_wav(self):
        assert_pcm_wav_riff_header(_tiny_wav())

    def test_max_bytes(self):
        data = _tiny_wav()
        with pytest.raises(ValueError, match="maximum upload size"):
            validate_pcm_wav_upload(data, max_bytes=len(data) - 1)
