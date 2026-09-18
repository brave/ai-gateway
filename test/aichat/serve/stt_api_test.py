"""Tests for STT / transcriptions API (WAV → Triton passthrough)."""

import io
import json
import os
import wave

# Settings modules load on import; CI/local bare pytest needs these.
os.environ.setdefault("ADMIN_AUTH_TOKEN", "test-admin")
os.environ.setdefault("RATE_LIMITING_ENABLED", "false")
os.environ.setdefault("ENV", "test")
os.environ.setdefault("X_BRAVE_KEY_CURRENT", "test-brave-key")
os.environ.setdefault("MASTER_SERVICES_KEY_SEED", "test-seed")
os.environ.setdefault("NEAR_API_KEY", "test-near")

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse, Response

from aichat.serve.stt_api import v1_audio_transcriptions


def _tiny_wav_16k_mono_int16() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        silence = (0).to_bytes(2, byteorder="little", signed=True) * 80
        wf.writeframes(silence)
    return buf.getvalue()


@contextmanager
def _patch_stt_settings(
    *, models, max_stt_upload_size_mb=25, max_stt_audio_duration_seconds=600
):
    with (
        patch("aichat.serve.stt_api.model_settings") as mock_model_settings,
        patch(
            "aichat.serve.stt_api.external_service_settings"
        ) as mock_external_settings,
    ):
        mock_model_settings.models = models
        mock_external_settings.max_stt_upload_size_mb = max_stt_upload_size_mb
        mock_external_settings.max_stt_audio_duration_seconds = (
            max_stt_audio_duration_seconds
        )
        yield mock_model_settings, mock_external_settings


class TestSttAPI:
    @pytest.fixture
    def mock_request(self):
        request = MagicMock(spec=Request)
        return request

    @pytest.fixture
    def mock_wav_file(self):
        wav = _tiny_wav_16k_mono_int16()
        f = MagicMock()
        f.filename = "clip.wav"
        f.read = AsyncMock(side_effect=[wav, b""])
        return f

    @pytest.mark.asyncio
    async def test_missing_model(self, mock_request, mock_wav_file):
        resp = await v1_audio_transcriptions(
            mock_request,
            file=mock_wav_file,
            model=None,
        )
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_model_not_stt(self, mock_request, mock_wav_file):
        with _patch_stt_settings(
            models={
                "parakeet": {"type": "speech_to_text"},
                "other": {"type": "llm"},
            }
        ):
            resp = await v1_audio_transcriptions(
                mock_request,
                file=mock_wav_file,
                model="other",
            )
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_success_json(self, mock_request, mock_wav_file):
        with (
            _patch_stt_settings(models={"parakeet": {"type": "speech_to_text"}}),
            patch(
                "aichat.serve.stt_api.transcribe_upload",
                new_callable=AsyncMock,
            ) as mock_tx,
        ):
            mock_tx.return_value = ("hello world", 250.0)
            resp = await v1_audio_transcriptions(
                mock_request,
                file=mock_wav_file,
                model="parakeet",
                response_format="json",
            )
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 200
        assert json.loads(resp.body.decode()) == {"text": "hello world"}
        mock_tx.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_srt_timestamp_over_one_minute(self, mock_request, mock_wav_file):
        with (
            _patch_stt_settings(models={"parakeet": {"type": "speech_to_text"}}),
            patch(
                "aichat.serve.stt_api.transcribe_upload",
                new_callable=AsyncMock,
            ) as mock_tx,
        ):
            mock_tx.return_value = ("one two five seconds", 125_000.0)
            resp = await v1_audio_transcriptions(
                mock_request,
                file=mock_wav_file,
                model="parakeet",
                response_format="srt",
            )
        assert isinstance(resp, Response)
        assert resp.status_code == 200
        body = resp.body.decode()
        assert "00:02:05,000" in body
        assert "00:00:75" not in body

    @pytest.mark.asyncio
    async def test_oversized_upload_rejected(self, mock_request):
        wav = _tiny_wav_16k_mono_int16()
        f = MagicMock()
        f.filename = "clip.wav"
        f.read = AsyncMock(side_effect=[wav, b""])
        with _patch_stt_settings(
            models={"parakeet": {"type": "speech_to_text"}},
            max_stt_upload_size_mb=0,
        ):
            resp = await v1_audio_transcriptions(
                mock_request,
                file=f,
                model="parakeet",
            )
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 400
        assert "maximum upload size" in resp.body.decode()

    @pytest.mark.asyncio
    async def test_invalid_filename_null_byte(self, mock_request, mock_wav_file):
        mock_wav_file.filename = "bad\x00name.wav"
        with _patch_stt_settings(models={"parakeet": {"type": "speech_to_text"}}):
            resp = await v1_audio_transcriptions(
                mock_request,
                file=mock_wav_file,
                model="parakeet",
            )
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 400
        assert "Invalid filename" in resp.body.decode()

    @pytest.mark.asyncio
    async def test_non_wav_rejected(self, mock_request):
        f = MagicMock()
        f.filename = "x.mp3"
        f.read = AsyncMock(side_effect=[b"not wav data", b""])
        with _patch_stt_settings(models={"parakeet": {"type": "speech_to_text"}}):
            resp = await v1_audio_transcriptions(
                mock_request,
                file=f,
                model="parakeet",
                response_format="json",
            )
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 400
        body = resp.body.decode()
        assert "RIFF/WAVE" in body or "PCM WAV" in body
