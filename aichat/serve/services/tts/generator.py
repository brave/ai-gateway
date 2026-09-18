from collections.abc import AsyncIterator

import httpx

from aichat.serve.backend.litellm import get_global_router
from aichat.serve.services.model_settings import model_settings


async def generate_speech(**kwargs):
    """Passthrough to litellm router's aspeech method."""
    router = get_global_router()
    return await router.aspeech(**kwargs)


async def stream_speech(model: str, body: dict) -> AsyncIterator[bytes]:
    """Proxy streaming TTS audio bytes from the upstream model address."""
    model_config = model_settings.models[model]
    address = model_config["address"].rstrip("/")
    url = f"{address}/audio/speech"
    api_key = model_config.get("api_key", "empty")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    client = httpx.AsyncClient(timeout=600)
    upstream = await client.send(
        client.build_request("POST", url, json=body, headers=headers),
        stream=True,
    )
    if upstream.status_code != 200:
        error_body = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        raise RuntimeError(
            f"Upstream speech failed ({upstream.status_code}): {error_body[:200]!r}"
        )

    async def stream_and_close() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return stream_and_close()
