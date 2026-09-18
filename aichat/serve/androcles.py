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

    async def _call() -> list:
        response = await router.allm_passthrough_route(**request_data)
        return response.json()["outputs"][0]["data"]

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
