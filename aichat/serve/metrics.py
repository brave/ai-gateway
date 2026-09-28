from collections.abc import Callable
from logging import Logger

from fastapi import FastAPI
from prometheus_client import (
    REGISTRY,
    Counter,
    Histogram,
)
from prometheus_client.utils import INF
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_fastapi_instrumentator.metrics import Info

from aichat.serve.api_version import api_version_from_path

PDF_FILE_PART_ENCOUNTERED = Counter(
    name="pdf_file_part_encountered_total",
    documentation="Total number of PDF file parts encountered, before any pypdf processing",
    registry=REGISTRY,
)

MEDIA_REQUEST_RETRY_TOTAL = Counter(
    name="media_request_retry_total",
    documentation="Total number of ai-gateway-media-processor requests retried after a ConnectError, by path and outcome",
    labelnames=("path", "outcome"),
    registry=REGISTRY,
)


def metrics(logger: Logger) -> Callable[[Info], None]:
    TOTAL = Counter(
        name="api_requests_total",
        documentation="Total number of requests by method, status and handler.",
        labelnames=("method", "code", "handler", "model", "host", "api_version"),
        registry=REGISTRY,
    )

    CONTENT_PART_COUNTS = Counter(
        name="content_part_total",
        documentation="Total number of content block types in V2 request messages.",
        labelnames=("model", "host", "content_type", "api_version"),
        registry=REGISTRY,
    )

    LATENCY = Histogram(
        name="request_duration_seconds",
        documentation=(
            "Latency with only few buckets by handler. "
            "Made to be only used if aggregation by handler is important. "
        ),
        buckets=(
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
            25.0,
            50.0,
            75.0,
            100.0,
            INF,
        ),
        labelnames=("method", "code", "handler", "model", "host", "api_version"),
        registry=REGISTRY,
    )

    LATENCY_WITHOUT_STREAMING = Histogram(
        name="request_first_response_duration_seconds",
        documentation=(
            "Latency with only few buckets by handler. "
            "Excluding any streaming latency."
            "Made to be only used if aggregation by handler is important. "
        ),
        buckets=(
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
            25.0,
            50.0,
            75.0,
            100.0,
            INF,
        ),
        labelnames=("method", "code", "handler", "model", "host", "api_version"),
        registry=REGISTRY,
    )

    def instrumentation(info: Info) -> None:
        model = "unknown"
        api_version = "unknown"

        model = getattr(info.request.state, "model", "unknown")

        try:
            api_version = info.request.state.api_version
        except AttributeError:
            api_version = api_version_from_path(info.modified_handler or "")

        host = info.request.headers.get("host")

        try:
            messages = getattr(info.request.state, "messages", None)
            if messages:
                for message in reversed(messages):
                    if getattr(message, "role", None) != "user":
                        break
                    content = getattr(message, "content", None)
                    if isinstance(content, list):
                        for part in content:
                            content_type = getattr(part, "type", None)
                            if content_type:
                                CONTENT_PART_COUNTS.labels(
                                    model, host, content_type[:200], api_version
                                ).inc()
        except AttributeError as e:
            logger.warning(
                "Failed to retrieve messages from request state for metrics. Error: %s",
                e,
            )

        TOTAL.labels(
            info.method,
            info.modified_status,
            info.modified_handler,
            model,
            host,
            api_version,
        ).inc()
        LATENCY.labels(
            info.method,
            info.modified_status,
            info.modified_handler,
            model,
            host,
            api_version,
        ).observe(info.modified_duration)
        LATENCY_WITHOUT_STREAMING.labels(
            info.method,
            info.modified_status,
            info.modified_handler,
            model,
            host,
            api_version,
        ).observe(info.modified_duration_without_streaming)

    return instrumentation


def instrument(app: FastAPI, logger: Logger) -> None:
    instrumentator = Instrumentator(
        excluded_handlers=[
            "/metrics",
            "/models$",
            ".*docs.*",
            "/openapi.json",
        ],
        should_ignore_untemplated=True,
        should_group_status_codes=False,
        should_instrument_requests_inprogress=True,
        inprogress_name="in_flight_requests",
        inprogress_labels=True,
    )
    instrumentator.add(metrics(logger)).instrument(app).expose(app)
