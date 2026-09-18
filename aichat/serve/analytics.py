import logging
import time

import httpx
from prometheus_client import REGISTRY, Counter, Histogram

from aichat.serve.external_service_settings import external_service_settings

logger = logging.getLogger(__name__)

# Prometheus metrics for analytics
ANALYTICS_REQUESTS_TOTAL = Counter(
    name="analytics_requests_total",
    documentation="Total number of analytics requests sent",
    labelnames=("model", "status"),
    registry=REGISTRY,
)

ANALYTICS_REQUEST_DURATION = Histogram(
    name="analytics_request_duration_seconds",
    documentation="Duration of analytics requests",
    labelnames=("model", "status"),
    registry=REGISTRY,
)

ANALYTICS_REQUEST_ERRORS = Counter(
    name="analytics_request_errors_total",
    documentation="Total number of failed analytics requests",
    labelnames=("model", "error_type"),
    registry=REGISTRY,
)


def _extract_text_from_content(content: str | list | None) -> str:
    # Messages come in not as plain text but as a list of dictionaries, e.g.:
    #
    #   [
    #     {'text': 'hello', 'type': 'text'},
    #     {'type': 'text', 'text': 'This is the text of a web page: <page>some page context</page>.'},
    #   ]
    #
    # We need to normalise them to just the text combined, e.g.:
    #
    #   hello\n\nThis is the text of a web page: <page>some page context</page>.
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "\n\n".join(parts)

    return ""


async def send_analytics_request(
    messages: list[dict] | None = None,
    model: str = "",
):
    """
    Send analytics data to external API without waiting for response.
    This runs as a background task after the main response is sent.
    """
    if not external_service_settings.analytics_enabled:
        return

    start_time = time.time()

    analytics_api_url = external_service_settings.analytics_model_address or None

    if not analytics_api_url:
        logger.debug("Analytics API URL not configured, skipping analytics request")
        ANALYTICS_REQUESTS_TOTAL.labels(model=model, status="skipped").inc()
        return

    try:
        # Extract the text from the last user chat message
        text = ""
        if messages:
            for message in reversed(messages):
                if message["role"] == "user":
                    text = _extract_text_from_content(message.get("content"))
                    break

        # Prepare analytics payload
        analytics_data = {
            "text": text,
            "model": model,
        }

        if not text:
            logger.debug(
                "No text found in the conversation, skipping analytics request"
            )
            ANALYTICS_REQUESTS_TOTAL.labels(model=model, status="skipped").inc()
            return

        async with httpx.AsyncClient() as httpx_client:
            # Send fire-and-forget request with short timeout
            response = await httpx_client.post(
                analytics_api_url,
                json=analytics_data,
                timeout=1,  # Short timeout to avoid hanging
            )

        # Track success metrics
        duration = time.time() - start_time
        status = "success" if response.status_code < 400 else "http_error"

        ANALYTICS_REQUESTS_TOTAL.labels(model=model, status=status).inc()
        ANALYTICS_REQUEST_DURATION.labels(model=model, status=status).observe(duration)

        if response.status_code >= 400:
            ANALYTICS_REQUEST_ERRORS.labels(
                model=model, error_type=f"http_{response.status_code}"
            ).inc()
            logger.info(f"Analytics request failed with HTTP {response.status_code}")
        else:
            logger.debug("Analytics data sent successfully")

    except httpx.TimeoutException:
        logger.info("Analytics request timed out")

    except httpx.RequestError as e:
        duration = time.time() - start_time
        ANALYTICS_REQUESTS_TOTAL.labels(model=model, status="network_error").inc()
        ANALYTICS_REQUEST_DURATION.labels(model=model, status="network_error").observe(
            duration
        )
        ANALYTICS_REQUEST_ERRORS.labels(model=model, error_type="network").inc()
        logger.warning(f"Analytics request network error: {e}")

    except Exception as e:
        duration = time.time() - start_time
        ANALYTICS_REQUESTS_TOTAL.labels(model=model, status="error").inc()
        ANALYTICS_REQUEST_DURATION.labels(model=model, status="error").observe(duration)
        ANALYTICS_REQUEST_ERRORS.labels(model=model, error_type="unknown").inc()
        # Log but don't raise - this should never affect the main response
        logger.warning(f"Failed to send analytics data: {e}")
