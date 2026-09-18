"""Optional live test against Bedrock Mantle; skipped without AWS credentials."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_BEDROCK_MANTLE_INTEGRATION") != "1",
    reason="Set RUN_BEDROCK_MANTLE_INTEGRATION=1 and AWS creds to run",
)


@pytest.mark.asyncio
async def test_bedrock_mantle_chat_completion():
    from aichat.serve.backend.litellm import get_global_router
    from aichat.serve.services.bedrock_mantle_auth import (
        get_bedrock_mantle_bearer_token,
    )

    router = get_global_router()
    response = await router.acompletion(
        model="bedrock-openai.gpt-5.5",
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        max_tokens=16,
        api_key=get_bedrock_mantle_bearer_token(),
    )
    content = response.choices[0].message.content
    assert content
