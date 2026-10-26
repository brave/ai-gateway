"""ML prompt-injection classifier (Triton passthrough)."""

import asyncio

from aichat.llm.metrics import SHADOW_ML_INJECTION_CLASSIFIER_CONFIDENCE_TOTAL
from aichat.serve.backend.litellm import get_global_router
from aichat.serve.services.model_settings import model_settings
from aichat.serve.services.security_settings import security_settings
from aichat.serve.utils import DEFAULT_TOKENIZER

INJECTION_CLASSIFIER_TRIAGE_CATEGORY = "injection_classifier"

SLICE_OVERLAP_RATIO = 0.05

_CONFIDENCE_BUCKETS = (
    "0.0-0.1",
    "0.1-0.2",
    "0.2-0.3",
    "0.3-0.4",
    "0.4-0.5",
    "0.5-0.6",
    "0.6-0.7",
    "0.7-0.8",
    "0.8-0.9",
    "0.9-1.0",
)


def confidence_bucket(confidence: float) -> str:
    score = max(0.0, min(1.0, float(confidence)))
    if score >= 1.0:
        return "0.9-1.0"
    return _CONFIDENCE_BUCKETS[int(score * 10)]


def slice_text_for_context_limit(
    text: str,
    max_tokens: int,
    overlap_ratio: float = SLICE_OVERLAP_RATIO,
) -> list[str]:
    """Split long text into overlapping windows using ``DEFAULT_TOKENIZER`` (tiktoken)."""
    if max_tokens <= 0:
        return [text]

    encoded = DEFAULT_TOKENIZER.encode(text)
    if len(encoded) <= max_tokens:
        return [text]

    stride = max(1, int(max_tokens * (1 - overlap_ratio)))
    chunks: list[str] = []
    start = 0
    while start < len(encoded):
        end = min(start + max_tokens, len(encoded))
        chunks.append(DEFAULT_TOKENIZER.decode(encoded[start:end]))
        if end >= len(encoded):
            break
        start += stride
    return chunks


async def ml_injection_classifier_inference(model_id: str, text: str) -> float | None:
    """Return injection confidence in [0, 1], or None on failure."""
    if not text.strip():
        return None
    if model_id not in model_settings.models:
        return None

    router = get_global_router()
    request_data = {
        "model": model_id,
        "json": {
            "inputs": [
                {
                    "name": "TEXT",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [[text]],
                }
            ],
            "outputs": [{"name": "CONFIDENCE"}],
        },
    }

    try:
        response = await router.allm_passthrough_route(**request_data)
        body = response.json() if hasattr(response, "json") else response
        return float(body["outputs"][0]["data"][0])
    except Exception:
        return None


async def shadow_ml_injection_classifier_scan(page_texts: list[str]) -> None:
    """Scan each webpage (one per ``brave-page-text`` part) in shadow mode.

    Long pages are chunked by token limit; only the maximum chunk confidence per
    page is recorded. Multiple pages in one request are scanned concurrently.
    """
    if not page_texts:
        return

    model_id = model_settings.model_triaging.get(INJECTION_CLASSIFIER_TRIAGE_CATEGORY)
    if not model_id or model_id not in model_settings.models:
        return

    limit = security_settings.shadow_prompt_injection_context_limit
    timeout = security_settings.shadow_ml_injection_classifier_timeout_seconds
    indexed_chunks: list[tuple[int, str]] = []
    for page_index, text in enumerate(page_texts):
        for chunk in slice_text_for_context_limit(text, limit):
            indexed_chunks.append((page_index, chunk))

    if not indexed_chunks:
        return

    async def _infer_chunk(chunk: str) -> float | None:
        try:
            if timeout > 0:
                return await asyncio.wait_for(
                    ml_injection_classifier_inference(model_id, chunk),
                    timeout=timeout,
                )
            return await ml_injection_classifier_inference(model_id, chunk)
        except TimeoutError:
            return None

    results = await asyncio.gather(
        *[_infer_chunk(chunk) for _, chunk in indexed_chunks],
        return_exceptions=True,
    )

    page_max_confidence: dict[int, float] = {}
    for (page_index, _), result in zip(indexed_chunks, results, strict=True):
        if not isinstance(result, (int, float)):
            continue
        prior = page_max_confidence.get(page_index)
        page_max_confidence[page_index] = (
            result if prior is None else max(prior, result)
        )

    for page_index in sorted(page_max_confidence):
        SHADOW_ML_INJECTION_CLASSIFIER_CONFIDENCE_TOTAL.labels(
            bucket=confidence_bucket(page_max_confidence[page_index]),
        ).inc()
