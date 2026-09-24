import asyncio
import base64
import io
import os
import platform
import wave

import numpy as np
from aichat_media.media_sandbox import (
    SandboxOpError,
    SandboxWorkerError,
    get_pool,
    reset_pool,
)
from pypdf import PdfWriter

IS_LINUX = platform.system() == "Linux"


async def _run(coro):
    return await coro


def _make_wav_bytes(
    duration_frames: int = 160,
    framerate: int = 16000,
    channels: int = 1,
    sampwidth: int = 2,
    amplitude: int = 1000,
) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        frame = b"".join(
            (amplitude if (i // channels) % 2 else -amplitude).to_bytes(
                sampwidth, byteorder="little", signed=True
            )
            for i in range(duration_frames * channels)
        )
        wf.writeframes(frame)
    return buf.getvalue()


def _stt_decode(pool, wav_bytes: bytes, **overrides) -> dict:
    return asyncio.run(
        pool.call(
            "stt_decode",
            {
                "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
                "max_duration_seconds": 600.0,
                **overrides,
            },
        )
    )


def _make_pdf_bytes(num_pages: int) -> bytes:
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(612, 792)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


async def _analyze(pool, pdf_bytes: bytes, **overrides) -> dict:
    return await pool.call(
        "pdf_analyze",
        {
            "pdf_b64": base64.b64encode(pdf_bytes).decode("ascii"),
            "encoding_name": "cl100k_base",
            "max_extraction_tokens": 100_000,
            "max_allowed_pages": 85,
            **overrides,
        },
    )


class TestWorkerRoundTrip:
    def setup_method(self):
        reset_pool()

    def teardown_method(self):
        reset_pool()

    def test_ping(self):
        pool = get_pool()
        result = asyncio.run(pool.call("ping", {}))
        assert result == {"pong": True}

    def test_pdf_analyze_blank_pages(self):
        pool = get_pool()
        result = asyncio.run(_analyze(pool, _make_pdf_bytes(3)))
        assert result["passthrough"] is None
        assert result["total_pages"] == 3
        assert result["estimated_tokens"] == 0

    def test_unknown_op_is_op_error(self):
        pool = get_pool()
        try:
            asyncio.run(pool.call("does_not_exist", {}))
        except SandboxOpError as e:
            assert "unknown op" in str(e)
        else:
            raise AssertionError("expected SandboxOpError")

    def test_probe_ops_not_registered_without_flag(self):
        pool = get_pool()
        try:
            asyncio.run(pool.call("probe_env", {}))
        except SandboxOpError as e:
            assert "unknown op" in str(e)
        else:
            raise AssertionError("probe ops registered without opt-in flag")

    def test_malformed_pdf_raises_op_error(self):
        pool = get_pool()
        garbage = base64.b64encode(b"not a pdf at all").decode()
        try:
            asyncio.run(
                pool.call(
                    "pdf_analyze",
                    {
                        "pdf_b64": garbage,
                        "encoding_name": "cl100k_base",
                        "max_extraction_tokens": 1000,
                        "max_allowed_pages": 85,
                    },
                )
            )
        except SandboxOpError:
            pass
        else:
            raise AssertionError("expected SandboxOpError")

    def test_worker_recovers_after_op_error(self):
        async def scenario():
            pool = get_pool()
            garbage = base64.b64encode(b"junk").decode()
            for _ in range(2):
                try:
                    await pool.call(
                        "pdf_analyze",
                        {
                            "pdf_b64": garbage,
                            "encoding_name": "cl100k_base",
                            "max_extraction_tokens": 100,
                            "max_allowed_pages": 85,
                        },
                    )
                except SandboxOpError:
                    pass
            return await _analyze(pool, _make_pdf_bytes(1))

        result = asyncio.run(scenario())
        assert result["total_pages"] == 1


class TestSandboxEnforcement:
    def setup_method(self):
        os.environ["MEDIA_SANDBOX_ENABLE_PROBES"] = "1"
        reset_pool()

    def teardown_method(self):
        reset_pool()
        os.environ.pop("MEDIA_SANDBOX_ENABLE_PROBES", None)

    def test_network_connect_denied(self):
        if not IS_LINUX:
            import pytest

            pytest.skip("Landlock/audit-hook net denial only on Linux")
        pool = get_pool()
        try:
            asyncio.run(pool.call("probe_net", {}))
        except SandboxOpError as e:
            assert "blocked in sandbox" in str(e)
        else:
            raise AssertionError("network access was not blocked")

    def test_env_scrubbed(self):
        pool = get_pool()
        result = asyncio.run(pool.call("probe_env", {}))
        assert "SECRET" not in result["env_keys"]

    def test_cpu_bomb_killed_loudly(self):
        async def scenario():
            pool = get_pool()
            try:
                await pool.call("spin_cpu", {"seconds": 60})
            except SandboxWorkerError:
                pass
            else:
                raise AssertionError("cpu bomb did not fail")
            return await pool.call("ping", {})

        result = asyncio.run(scenario())
        assert result == {"pong": True}


class TestSttDecodeOp:
    def setup_method(self):
        reset_pool()

    def teardown_method(self):
        reset_pool()

    def test_mono_16k_decode(self):
        pool = get_pool()
        wav = _make_wav_bytes(duration_frames=16000, framerate=16000)
        result = _stt_decode(pool, wav)
        assert result["sample_rate"] == 16000
        assert result["n_samples"] == 16000
        assert result["duration_ms"] == 1000.0
        pcm = base64.b64decode(result["pcm_b64"])
        audio = np.frombuffer(pcm, dtype=np.float32)
        assert audio.size == 16000
        assert np.abs(audio).max() > 0.0

    def test_stereo_downmix_and_resample(self):
        pool = get_pool()
        wav = _make_wav_bytes(duration_frames=8000, framerate=48000, channels=2)
        result = _stt_decode(pool, wav)
        assert result["sample_rate"] == 16000
        assert result["n_samples"] == round(8000 * 16000 / 48000)

    def test_compressed_wav_rejected(self):
        pool = get_pool()
        try:
            _stt_decode(pool, b"RIFF\x24\x00\x00\x00WAVEjunkjunk")
        except SandboxOpError as e:
            assert "PCM WAV" in str(e)
        else:
            raise AssertionError("compressed/garbage WAV accepted")

    def test_over_duration_rejected(self):
        pool = get_pool()
        wav = _make_wav_bytes(duration_frames=16000, framerate=16000)  # 1s
        try:
            _stt_decode(pool, wav, max_duration_seconds=0.5)
        except SandboxOpError as e:
            assert "duration" in str(e)
        else:
            raise AssertionError("over-duration WAV accepted")

    def test_truncated_data_rejected_or_handled(self):
        pool = get_pool()
        wav = _make_wav_bytes(duration_frames=16000, framerate=16000)
        truncated = wav[: len(wav) // 2]
        try:
            result = _stt_decode(pool, truncated)
            # wave may zero-pad truncated frames; accept decode but sanity check
            assert result["n_samples"] >= 0
        except SandboxOpError as e:
            assert "WAV" in str(e) or "duration" in str(e)

    def test_worker_recovers_after_stt_error(self):
        async def scenario():
            pool = get_pool()
            garbage = base64.b64encode(b"junk").decode()
            for _ in range(2):
                try:
                    await pool.call(
                        "stt_decode",
                        {"audio_b64": garbage, "max_duration_seconds": 600.0},
                    )
                except SandboxOpError:
                    pass
            return await pool.call(
                "stt_decode",
                {
                    "audio_b64": base64.b64encode(
                        _make_wav_bytes(duration_frames=160)
                    ).decode("ascii"),
                    "max_duration_seconds": 600.0,
                },
            )

        result = asyncio.run(scenario())
        assert result["n_samples"] == 160
