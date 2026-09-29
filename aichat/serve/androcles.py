import asyncio
import logging

from aichat.serve.backend.litellm import get_global_router

# Index of each label in the Androcles model output
ANDROCLES_TRIAGE_INDICES = {
    "coding": 3,
    "language": 10,
    "vision": 8,
}

# Threshold values for task classification decisions
ANDROCLES_TRIAGE_THRESHOLDS = {
    "coding": 0.9,
    "language": 0.9,
    "vision": 0.9,
}

logger = logging.getLogger(__name__)


def _coerce_probability(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(
            f"Unexpected Androcles probability type: {type(value).__name__}"
        )
    return float(value)


def _coerce_probability_vector(values: list) -> list[float]:
    return [_coerce_probability(value) for value in values]


def _androcles_batch_probs_rows(probs_list: list, batch_size: int) -> list[list[float]]:
    """Normalize Triton output to one probability vector per batch row."""
    if not isinstance(probs_list, list) or not probs_list:
        raise ValueError("Empty or invalid Androcles batch output")
    if isinstance(probs_list[0], list):
        if len(probs_list) != batch_size:
            raise ValueError(
                f"Unexpected Androcles batch row count: got {len(probs_list)} "
                f"expected {batch_size}"
            )
        rows = probs_list
    elif len(probs_list) % batch_size != 0:
        raise ValueError(
            f"Unexpected Androcles batch flat length: got {len(probs_list)} "
            f"for batch_size {batch_size}"
        )
    else:
        row_len = len(probs_list) // batch_size
        rows = [
            list(probs_list[i * row_len : (i + 1) * row_len]) for i in range(batch_size)
        ]
    return [_coerce_probability_vector(row) for row in rows]


async def androcles_inference(
    input_text,
    *,
    timeout_seconds: float | None = None,
) -> list | None:
    """Send inference request to Androcles model endpoint.

    Args:
        input_text: Text input to send to model (str or list of Content objects)
        timeout_seconds: Optional wall-clock limit; None or <= 0 means no timeout.

    Returns:
        Raw probability vector from the model output layer. None if the request fails.
    """
    if isinstance(input_text, list):
        text_parts = [
            item.text for item in input_text if hasattr(item, "text") and item.text
        ]
        input_text = " ".join(text_parts) if text_parts else ""

    if not isinstance(input_text, str) or not input_text.strip():
        return None

    router = get_global_router()

    request_data = {
        "model": "androcles",
        "json": {
            "inputs": [
                {
                    "name": "text_input",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [[input_text]],
                }
            ]
        },
    }

    async def _call() -> list[float]:
        response = await router.allm_passthrough_route(**request_data)
        data = response.json()["outputs"][0]["data"]
        if not isinstance(data, list):
            raise TypeError(f"Unexpected Androcles output type: {type(data).__name__}")
        return _coerce_probability_vector(data)

    try:
        if timeout_seconds is not None and timeout_seconds > 0:
            return await asyncio.wait_for(_call(), timeout=timeout_seconds)
        return await _call()
    except TimeoutError:
        logger.warning("Androcles inference timed out after %.3fs", timeout_seconds)
        return None
    except Exception as e:
        logger.error(f"Error parsing Androcles response: {e}, skipping..")
        return None


async def androcles_inference_batch(
    texts: list[str],
    *,
    timeout_seconds: float | None = None,
) -> list[list | None]:
    """Send a batched inference request to the Androcles model endpoint.

    Mirrors ``androcles_inference`` but sends all non-empty texts in one
    Triton round-trip. Returns one entry per input (``None`` for blank text or
    on failure), in the same order as ``texts``.
    """
    out: list[list | None] = [None] * len(texts)
    indexed: list[tuple[int, str]] = []
    for i, t in enumerate(texts):
        if isinstance(t, str) and t.strip():
            indexed.append((i, t.strip()))
    if not indexed:
        return out

    cleaned = [t for _, t in indexed]
    router = get_global_router()

    request_data = {
        "model": "androcles",
        "json": {
            "inputs": [
                {
                    "name": "text_input",
                    "shape": [len(cleaned), 1],
                    "datatype": "BYTES",
                    "data": [cleaned],
                }
            ]
        },
    }

    async def _call() -> list:
        response = await router.allm_passthrough_route(**request_data)
        outputs = response.json()["outputs"]
        if not outputs:
            raise ValueError("No outputs in Androcles batch response")
        probs_list = outputs[0]["data"]
        return _androcles_batch_probs_rows(probs_list, len(cleaned))

    try:
        if timeout_seconds is not None and timeout_seconds > 0:
            probs_list = await asyncio.wait_for(_call(), timeout=timeout_seconds)
        else:
            probs_list = await _call()
    except TimeoutError:
        logger.warning(
            "Androcles batch inference timed out after %.3fs", timeout_seconds
        )
        return out
    except Exception as e:
        logger.error(f"Error parsing Androcles batch response: {e}, skipping..")
        return out

    for (i, _), probs in zip(indexed, probs_list, strict=True):
        out[i] = probs
    return out


def task_type_from_androcles_probabilities(
    probabilities: list[float] | None,
) -> str | None:
    if probabilities is None:
        return None
    for task in ("vision", "language", "coding"):
        index = ANDROCLES_TRIAGE_INDICES[task]
        if index >= len(probabilities):
            continue
        if probabilities[index] > ANDROCLES_TRIAGE_THRESHOLDS[task]:
            return task
    return None
