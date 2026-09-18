import functools
import time

import httpx
from prometheus_client import REGISTRY, Counter, Gauge, Histogram

# Single Gauge for both active and idle connections, with a "state" label
HTTPX_CLIENT_CONNECTIONS = Gauge(
    name="httpx_client_connections",
    documentation="Number of active and idle connections in the connection pool",
    labelnames=[
        "host",
        "state",
    ],  # "state" will be either "active" or "idle"
    registry=REGISTRY,
)

# Counter for the number of outgoing HTTP requests
HTTPX_CLIENT_REQUESTS = Counter(
    name="httpx_client_requests",
    documentation="Number of outgoing HTTP requests",
    labelnames=("host", "method", "path", "status_code"),
    registry=REGISTRY,
)

# Histogram for the latency of outgoing HTTP requests
HTTPX_CLIENT_LATENCY = Histogram(
    name="httpx_client_latency_seconds",
    documentation="Latency of outgoing HTTP requests",
    labelnames=("host", "method", "path", "status_code"),
    registry=REGISTRY,
)

# Counter for the number of HTTP errors
HTTPX_CLIENT_HTTP_ERRORS = Counter(
    name="httpx_client_http_errors",
    documentation="Number of HTTP errors",
    labelnames=("host", "method", "path", "error"),
    registry=REGISTRY,
)


def _normalize_metric_path(path: str, max_segments: int = 5) -> str:
    """Normalize URL path for Prometheus labels to prevent high-cardinality explosion."""
    segments = path.rstrip("/").split("/")
    if len(segments) > max_segments:
        return "/".join(segments[:max_segments])
    return path


class InstrumentedHTTPTransport(httpx.AsyncHTTPTransport):
    def __init__(self, limits):
        super().__init__(limits=limits)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        start_time = time.time()
        method = request.method
        host = request.url.host
        path = _normalize_metric_path(request.url.path)

        try:
            # Call the base class method to handle the request
            response = await super().handle_async_request(request)
            end_time = time.time()
            status_code = response.status_code

            # Record latency
            HTTPX_CLIENT_LATENCY.labels(
                host=host, method=method, path=path, status_code=status_code
            ).observe(end_time - start_time)

            # Increment request counter
            HTTPX_CLIENT_REQUESTS.labels(
                host=host, method=method, path=path, status_code=status_code
            ).inc()

            # Update the connection pool gauge
            self._update_connection_gauge()

            original_aclose = response.aclose
            response.aclose = functools.partial(
                self._instrumented_aclose, original_aclose
            )

            return response

        except httpx.HTTPError as e:
            end_time = time.time()

            # Safely get the response and status code if available
            response = getattr(e, "response", None)
            status = (
                response.status_code
                if response and hasattr(response, "status_code")
                else "unknown"
            )

            # Record latency with 'error' status code
            HTTPX_CLIENT_LATENCY.labels(
                host=host, method=method, path=path, status_code=status
            ).observe(end_time - start_time)

            # Increment the request counter with 'error' status code
            HTTPX_CLIENT_REQUESTS.labels(
                host=host, method=method, path=path, status_code=status
            ).inc()

            # Increment the error counter
            HTTPX_CLIENT_HTTP_ERRORS.labels(
                host=host, method=method, path=path, error=e.__class__.__name__
            ).inc()

            # Update the connection pool gauge
            self._update_connection_gauge()

            # Re-raise the exception to allow the calling code to handle it
            raise

    async def _instrumented_aclose(self, original_close, *args, **kwargs):
        """
        Updates the connection gauge after performing the original aclose
        """
        await original_close(*args, **kwargs)
        self._update_connection_gauge()

    def _update_connection_gauge(self):
        """
        This method updates the connection gauge for both active and idle connections
        using a single Prometheus gauge with a "state" label.
        """
        connection_counts = self._get_connection_counts()

        # Update the Prometheus gauge for each host, distinguishing between active and idle
        for host, (active_count, idle_count) in connection_counts.items():
            HTTPX_CLIENT_CONNECTIONS.labels(host=host, state="active").set(active_count)
            HTTPX_CLIENT_CONNECTIONS.labels(host=host, state="idle").set(idle_count)

    def _get_connection_counts(self):
        """
        Counts active and idle connections for all hosts.
        """
        # This method accesses internal attributes to count active connections
        # Based on the __repr__ method of the ConnectionPool class in httpcore,
        # which is used by httpx for connection pooling.
        # https://github.com/encode/httpcore/blob/4ee1ca25b6b242caec2d96782ed25e81ad7b4ecb/httpcore/_async/connection_pool.py#L326-L346

        connection_counts = {}

        with self._pool._optional_thread_lock:
            for connection in self._pool._connections:
                host = connection._origin.host
                is_idle = connection.is_idle()

                if host not in connection_counts:
                    connection_counts[host] = [0, 0]  # [active, idle]

                if is_idle:
                    connection_counts[host][1] += 1  # Increment idle count
                else:
                    connection_counts[host][0] += 1  # Increment active count

        return connection_counts
