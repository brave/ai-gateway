import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from aichat.prompts.page_text import PageTextContentPart
from aichat.protocol.open_ai_protocol import TextContentPart, UserMessage
from aichat.serve import open_ai_adapter
from aichat.serve.ml_injection_classifier import (
    confidence_bucket,
    ml_injection_classifier_inference,
    shadow_ml_injection_classifier_scan,
    slice_text_for_context_limit,
)
from aichat.serve.utils import DEFAULT_TOKENIZER


def test_brave_page_texts_collects_each_page_part_in_order():
    message = UserMessage(
        content=[
            PageTextContentPart(type="brave-page-text", text="first page"),
            TextContentPart(type="text", text="user question"),
            PageTextContentPart(type="brave-page-text", text="second page"),
        ]
    )
    assert open_ai_adapter._brave_page_texts_from_message(message) == [
        "first page",
        "second page",
    ]


@pytest.mark.parametrize(
    ("confidence", "bucket"),
    [
        (0.0, "0.0-0.1"),
        (0.09, "0.0-0.1"),
        (0.1, "0.1-0.2"),
        (0.87, "0.8-0.9"),
        (0.99, "0.9-1.0"),
        (1.0, "0.9-1.0"),
    ],
)
def test_confidence_bucket(confidence, bucket):
    assert confidence_bucket(confidence) == bucket


def test_slice_short_text_unchanged():
    assert slice_text_for_context_limit("hello", 100) == ["hello"]


def test_slice_long_text_overlaps():
    text = "word " * 5000
    max_tokens = 100
    chunks = slice_text_for_context_limit(text, max_tokens, overlap_ratio=0.05)
    assert len(chunks) > 1
    assert all(len(DEFAULT_TOKENIZER.encode(c)) <= max_tokens for c in chunks)
    encoded = DEFAULT_TOKENIZER.encode(text)
    stride = int(max_tokens * 0.95)
    assert chunks[0] == DEFAULT_TOKENIZER.decode(encoded[:max_tokens])
    assert chunks[1] == DEFAULT_TOKENIZER.decode(encoded[stride : stride + max_tokens])


def _security_settings(**overrides):
    defaults = {
        "shadow_ml_injection_classifier_timeout_seconds": 5.0,
    }
    defaults.update(overrides)
    return type("S", (), defaults)()


def _model_settings(models: dict, triaging: dict | None = None):
    return type(
        "S",
        (),
        {
            "models": models,
            "model_triaging": (
                triaging
                if triaging is not None
                else {"injection_classifier": next(iter(models))}
            ),
        },
    )()


@pytest.mark.asyncio
async def test_ml_injection_classifier_inference_parses_confidence(monkeypatch):
    monkeypatch.setattr(
        "aichat.serve.ml_injection_classifier.model_settings",
        _model_settings({"injection-cls": {}}),
    )
    response = AsyncMock()
    response.json = lambda: {"outputs": [{"name": "CONFIDENCE", "data": [0.87]}]}
    router = AsyncMock(allm_passthrough_route=AsyncMock(return_value=response))
    with patch(
        "aichat.serve.ml_injection_classifier.get_global_router",
        return_value=router,
    ):
        assert (
            await ml_injection_classifier_inference("injection-cls", "page body")
            == 0.87
        )
    call = router.allm_passthrough_route.await_args.kwargs
    assert call["json"]["outputs"] == [{"name": "CONFIDENCE"}]


@pytest.mark.asyncio
async def test_shadow_scan_times_out_per_chunk(monkeypatch):
    monkeypatch.setattr(
        "aichat.serve.ml_injection_classifier.model_settings",
        _model_settings({"injection-cls": {}}),
    )
    monkeypatch.setattr(
        "aichat.serve.ml_injection_classifier.security_settings",
        _security_settings(
            shadow_prompt_injection_context_limit=10_000,
            shadow_ml_injection_classifier_timeout_seconds=0.05,
        ),
    )

    async def slow_call(**kwargs):
        await asyncio.sleep(5)
        return AsyncMock(json=lambda: {"outputs": [{"data": [0.5]}]})()

    with patch(
        "aichat.serve.ml_injection_classifier.get_global_router",
        return_value=AsyncMock(allm_passthrough_route=slow_call),
    ):
        with patch(
            "aichat.serve.ml_injection_classifier.SHADOW_ML_INJECTION_CLASSIFIER_CONFIDENCE_TOTAL"
        ) as counter:
            await shadow_ml_injection_classifier_scan(["page body"])
        counter.labels.assert_not_called()


@pytest.mark.asyncio
async def test_shadow_scan_runs_chunks_concurrently(monkeypatch):
    monkeypatch.setattr(
        "aichat.serve.ml_injection_classifier.security_settings",
        _security_settings(shadow_prompt_injection_context_limit=10),
    )
    monkeypatch.setattr(
        "aichat.serve.ml_injection_classifier.model_settings",
        _model_settings({"injection-cls": {}}),
    )
    calls: list[str] = []

    async def fake_inference(_model: str, text: str):
        calls.append(text)
        await asyncio.sleep(0)
        return 0.1

    with patch(
        "aichat.serve.ml_injection_classifier.ml_injection_classifier_inference",
        fake_inference,
    ):
        await shadow_ml_injection_classifier_scan(["aaaaa", "bbbbb"])
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_shadow_scan_records_max_confidence_per_page(monkeypatch):
    monkeypatch.setattr(
        "aichat.serve.ml_injection_classifier.security_settings",
        _security_settings(shadow_prompt_injection_context_limit=10_000),
    )
    monkeypatch.setattr(
        "aichat.serve.ml_injection_classifier.model_settings",
        _model_settings({"injection-cls": {}}),
    )
    monkeypatch.setattr(
        "aichat.serve.ml_injection_classifier.slice_text_for_context_limit",
        lambda text, _limit: ["chunk-a", "chunk-b"] if text == "long-page" else [text],
    )
    scores = {"chunk-a": 0.2, "chunk-b": 0.9, "short-page": 0.1}

    async def fake_inference(_model: str, text: str):
        return scores[text]

    with (
        patch(
            "aichat.serve.ml_injection_classifier.ml_injection_classifier_inference",
            fake_inference,
        ),
        patch(
            "aichat.serve.ml_injection_classifier.SHADOW_ML_INJECTION_CLASSIFIER_CONFIDENCE_TOTAL"
        ) as counter,
    ):
        await shadow_ml_injection_classifier_scan(["long-page", "short-page"])

    assert counter.labels.call_count == 2
    counter.labels.assert_any_call(bucket=confidence_bucket(0.9))
    counter.labels.assert_any_call(bucket=confidence_bucket(0.1))


@pytest.mark.asyncio
async def test_shadow_scan_three_pages_each_get_max(monkeypatch):
    monkeypatch.setattr(
        "aichat.serve.ml_injection_classifier.security_settings",
        _security_settings(shadow_prompt_injection_context_limit=10_000),
    )
    monkeypatch.setattr(
        "aichat.serve.ml_injection_classifier.model_settings",
        _model_settings({"injection-cls": {}}),
    )

    async def fake_inference(_model: str, text: str):
        return {"page-a": 0.3, "page-b": 0.8, "page-c": 0.5}[text]

    with (
        patch(
            "aichat.serve.ml_injection_classifier.ml_injection_classifier_inference",
            fake_inference,
        ),
        patch(
            "aichat.serve.ml_injection_classifier.SHADOW_ML_INJECTION_CLASSIFIER_CONFIDENCE_TOTAL"
        ) as counter,
    ):
        await shadow_ml_injection_classifier_scan(["page-a", "page-b", "page-c"])

    assert counter.labels.call_count == 3
    for score in (0.3, 0.8, 0.5):
        counter.labels.assert_any_call(bucket=confidence_bucket(score))
