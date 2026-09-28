import base64
from typing import Any

import numpy as np

from aichat.serve.backend.litellm import get_global_router
from aichat.serve.external_service_settings import external_service_settings
from aichat.serve.media_client import call_stt_decode

TARGET_SAMPLE_RATE = 16000


async def decode_wav_pcm_to_mono_16k_float32(
    audio_bytes: bytes,
) -> tuple[np.ndarray, float]:
    """
    Decode a PCM WAV upload in the media sandbox (validate + mono + resample
    to TARGET_SAMPLE_RATE). Returns (audio, duration_ms); duration_ms is
    pre-resample (original sample rate).
    """
    result = await call_stt_decode(
        {
            "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
            "max_duration_seconds": float(
                external_service_settings.max_stt_audio_duration_seconds
            ),
        }
    )
    pcm = base64.b64decode(result["pcm_b64"])
    audio = np.frombuffer(pcm, dtype=np.float32)
    if audio.size != result["n_samples"]:
        raise ValueError("sandbox decode returned inconsistent sample count")
    return audio, float(result["duration_ms"])


def _triton_response_to_dict(triton_response: Any) -> dict:
    if isinstance(triton_response, dict):
        return triton_response
    json_fn = getattr(triton_response, "json", None)
    if callable(json_fn):
        return json_fn()
    raise ValueError(f"Unexpected Triton response type: {type(triton_response)}")


def _extract_transcription(triton_data: dict) -> str:
    """Parakeet: one BYTES tensor `transcription`, shape [1], data [str]."""
    outputs = triton_data.get("outputs") or []
    if not outputs:
        raise ValueError("No outputs in Triton response")
    out = next((o for o in outputs if o.get("name") == "transcription"), outputs[0])
    data = out.get("data")
    if data is None:
        raise ValueError("No data in transcription output")
    if isinstance(data, str):
        return data
    if isinstance(data, list) and len(data) > 0:
        el = data[0]
        if isinstance(el, bytes):
            return el.decode("utf-8", errors="replace")
        return str(el)
    raise ValueError(f"Unexpected transcription data: {type(data)!r}")


async def transcribe_audio(model: str, audio: np.ndarray) -> str:
    """Run Parakeet on Triton via router passthrough. model is the aichat model id."""
    router = get_global_router()
    flat = np.ascontiguousarray(audio.reshape(1, -1), dtype=np.float32)
    n = int(flat.size)
    triton_request = {
        "model": model,
        "json": {
            "inputs": [
                {
                    "name": "audio_signal",
                    "shape": [1, n],
                    "datatype": "FP32",
                    "data": flat.flatten().tolist(),
                }
            ],
        },
    }
    triton_response = await router.allm_passthrough_route(**triton_request)
    triton_data = _triton_response_to_dict(triton_response)
    return _extract_transcription(triton_data)


async def transcribe_upload(
    model: str, audio_bytes: bytes, filename: str
) -> tuple[str, float]:
    """
    Decode WAV PCM upload and transcribe. Returns (transcript, duration_ms).
    duration_ms is based on decoded length at TARGET_SAMPLE_RATE after resampling.
    """
    del filename  # extension hint not used; WAV-only
    audio, _pre_resample_duration_ms = await decode_wav_pcm_to_mono_16k_float32(
        audio_bytes
    )
    duration_ms = float(len(audio) / TARGET_SAMPLE_RATE * 1000.0)
    text = await transcribe_audio(model, audio)
    if not text or not text.strip():
        raise RuntimeError("Transcription returned empty text")
    return text.strip(), duration_ms
