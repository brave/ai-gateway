import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps

import blake3
import httpx
from fastapi import HTTPException, Request

from aichat.serve import internal_client
from aichat.serve.internal_settings import internal_settings
from aichat.serve.rate_limiting_settings import rate_limiting_settings
from aichat.serve.services.model_settings import model_settings
from aichat.serve.utils import get_real_ip

logger = logging.getLogger(__name__)

_internal_salt_cache: dict[int, tuple[str, str | None]] = {}


def hash_ip_with_salt(ip_str: str, salt: str) -> str:
    """BLAKE3-hash an IP with the given salt. Used to keep raw IPs off the
    wire when calling aichat-internal."""
    if ip_str == "UNKNOWN":  # local testing
        ip_str = "127.0.0.1"
    return blake3.blake3(salt.encode("utf-8") + ip_str.encode("utf-8")).hexdigest()


async def _fetch_internal_salts(
    httpx_client: httpx.AsyncClient,
) -> tuple[str, str | None] | None:
    ttl = rate_limiting_settings.salt_cache_ttl_seconds
    local_epoch = int(time.time() // ttl)
    cached = _internal_salt_cache.get(local_epoch)
    if cached is not None:
        return cached

    response = await internal_client.rate_limit_salts(httpx_client)
    if response is None:
        return None
    current = response.get("current")
    if not current:
        return None

    previous = response.get("previous")
    # Bound the cache at ~2 entries (current + previous epoch).
    stale = [k for k in _internal_salt_cache if k < local_epoch - 1]
    for k in stale:
        del _internal_salt_cache[k]
    _internal_salt_cache[local_epoch] = (current, previous)
    return _internal_salt_cache[local_epoch]


async def _internal_hashed_ips(
    httpx_client: httpx.AsyncClient | None, rate_key: str
) -> tuple[str, str | None] | None:
    if httpx_client is None:
        return None
    salts = await _fetch_internal_salts(httpx_client)
    if salts is None:
        return None
    current_salt, previous_salt = salts
    hashed_current = hash_ip_with_salt(rate_key, current_salt)
    hashed_previous = (
        hash_ip_with_salt(rate_key, previous_salt) if previous_salt else None
    )
    return hashed_current, hashed_previous


def _request_httpx_client(
    raw_request: Request | None,
) -> httpx.AsyncClient | None:
    if raw_request is None:
        return None
    return getattr(raw_request.state, "httpx_client", None)


@dataclass
class RateLimitVerdict:
    allowed: bool
    fallback_model: str | None = None


async def check_rate_limit(
    raw_request: Request,
    model: str,
    is_premium_host: bool,
    rate_key: str,
    is_content_agent_request: bool = False,
    has_valid_premium_credential: bool = False,
) -> RateLimitVerdict:
    if not internal_settings.internal_api_enabled:
        return RateLimitVerdict(allowed=True)
    if "force-error-rate-limit-user" in raw_request.query_params:
        return RateLimitVerdict(allowed=False)

    is_premium_request = is_premium_host and has_valid_premium_credential

    if not is_premium_request:
        if is_premium_host or not model_settings.models.get(model, {}).get(
            "free", False
        ):
            return RateLimitVerdict(allowed=True)
        if is_content_agent_request:
            return RateLimitVerdict(allowed=True)

    if is_premium_request:
        maximum_requests = 0
        interval_in_seconds = 0
    else:
        config = model_settings.models.get(model, {})
        maximum_requests = config.get("rate_limit") or 0
        interval_in_seconds = config.get("rate_limit_interval_seconds") or 0
        if not maximum_requests or not interval_in_seconds:
            logger.warning("No rate limit information found for model %s", model)
            return RateLimitVerdict(allowed=False)

    httpx_client = _request_httpx_client(raw_request)
    hashed = await _internal_hashed_ips(httpx_client, rate_key)
    if hashed is None or httpx_client is None:
        logger.error(
            "aichat-internal /1/rate_limit unavailable, denying (model=%s)", model
        )
        return RateLimitVerdict(allowed=False)
    hashed_current, hashed_previous = hashed
    response = await internal_client.rate_limit_check(
        httpx_client,
        model=model,
        hashed_ip_current=hashed_current,
        hashed_ip_previous=hashed_previous,
        is_premium_host=False,
        is_content_agent_request=is_content_agent_request,
        is_free_model=True,
        maximum_requests=maximum_requests,
        interval_in_seconds=interval_in_seconds,
        force_error_rate_limit=False,
        is_premium_request=is_premium_request,
    )
    if response is None:
        logger.error(
            "aichat-internal /1/rate_limit unavailable, denying (model=%s)", model
        )
        return RateLimitVerdict(allowed=False)
    if not response.get("allowed", False):
        return RateLimitVerdict(allowed=False)

    return RateLimitVerdict(
        allowed=True, fallback_model=response.get("rate_limit_fallback_model")
    )


async def check_content_agent_rate_limits(
    rate_key: str,
    httpx_client: httpx.AsyncClient | None = None,
) -> bool:
    if not internal_settings.internal_api_enabled:
        return True
    hashed = await _internal_hashed_ips(httpx_client, rate_key)
    if hashed is None or httpx_client is None:
        logger.error("aichat-internal /1/rate_limit/content_agent unavailable, denying")
        return False
    hashed_current, hashed_previous = hashed
    verdict = await internal_client.rate_limit_content_agent(
        httpx_client,
        hashed_ip_current=hashed_current,
        hashed_ip_previous=hashed_previous,
    )
    if verdict is None:
        logger.error("aichat-internal /1/rate_limit/content_agent unavailable, denying")
        return False
    return bool(verdict.get("allowed", False))


async def check_automatic_mode_daily_limit(
    rate_key: str,
    httpx_client: httpx.AsyncClient | None = None,
) -> dict[str, int | bool]:
    if not rate_limiting_settings.rate_limiting_enabled:
        return {"count": 0, "exceeded": False, "limit": 0}
    if not internal_settings.internal_api_enabled:
        return {"count": 0, "exceeded": False, "limit": 0}
    limit = rate_limiting_settings.automatic_premium_model_daily_response_limit
    if limit <= 0:
        return {"count": 0, "exceeded": True, "limit": 0}

    hashed = await _internal_hashed_ips(httpx_client, rate_key)
    if hashed is None or httpx_client is None:
        logger.error(
            "aichat-internal /1/rate_limit/automatic_mode/peek unavailable, "
            "treating as exceeded"
        )
        return {"count": 0, "exceeded": True, "limit": 0}
    hashed_current, hashed_previous = hashed
    verdict = await internal_client.rate_limit_automatic_mode_peek(
        httpx_client,
        hashed_ip_current=hashed_current,
        hashed_ip_previous=hashed_previous,
    )
    if verdict is None:
        logger.error(
            "aichat-internal /1/rate_limit/automatic_mode/peek unavailable, "
            "treating as exceeded"
        )
        return {"count": 0, "exceeded": True, "limit": 0}
    return {
        "count": int(verdict.get("count", 0)),
        "exceeded": bool(verdict.get("exceeded", False)),
        "limit": int(verdict.get("limit", 0)),
    }


async def check_and_increment_automatic_mode_daily_count(
    rate_key: str,
    httpx_client: httpx.AsyncClient | None = None,
) -> bool:
    if not rate_limiting_settings.rate_limiting_enabled:
        return True
    if not internal_settings.internal_api_enabled:
        return True
    if rate_limiting_settings.automatic_premium_model_daily_response_limit <= 0:
        return False

    hashed = await _internal_hashed_ips(httpx_client, rate_key)
    if hashed is None or httpx_client is None:
        logger.error(
            "aichat-internal /1/rate_limit/automatic_mode unavailable, denying"
        )
        return False
    hashed_current, hashed_previous = hashed
    verdict = await internal_client.rate_limit_automatic_mode_check(
        httpx_client,
        hashed_ip_current=hashed_current,
        hashed_ip_previous=hashed_previous,
    )
    if verdict is None:
        logger.error(
            "aichat-internal /1/rate_limit/automatic_mode unavailable, denying"
        )
        return False
    return bool(verdict.get("allowed", False))


async def check_route_rate_limit(
    route: str,
    rate_key: str,
    daily_limit: int | None = None,
    httpx_client: httpx.AsyncClient | None = None,
    config_key: str | None = None,
) -> bool:
    if not rate_limiting_settings.rate_limiting_enabled:
        return True
    if not internal_settings.internal_api_enabled:
        return True

    hashed = await _internal_hashed_ips(httpx_client, rate_key)
    if hashed is None or httpx_client is None:
        logger.error(
            "aichat-internal /1/rate_limit/route unavailable, denying (route=%s)",
            route,
        )
        return False
    hashed_current, hashed_previous = hashed
    verdict = await internal_client.rate_limit_route(
        httpx_client,
        route=route,
        hashed_ip_current=hashed_current,
        hashed_ip_previous=hashed_previous,
        daily_limit=daily_limit,
        config_key=config_key,
    )
    if verdict is None:
        logger.error(
            "aichat-internal /1/rate_limit/route unavailable, denying (route=%s)",
            route,
        )
        return False
    return bool(verdict.get("allowed", False))


def rate_limit_route(
    daily_limit: int | None = None,
    route_path: str | None = None,
    error_message: str | None = None,
    config_key: str | None = None,
):

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if daily_limit is None and not config_key:
                return await func(*args, **kwargs)

            # Short-circuit before touching the request when rate
            # limiting is disabled (local dev / unit tests). Avoids
            # spurious calls into request.headers / get_real_ip when
            # the request is a MagicMock in unit tests.
            if not rate_limiting_settings.rate_limiting_enabled:
                return await func(*args, **kwargs)

            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if request is None and "request" in kwargs:
                request = kwargs["request"]
            if request is None:
                logger.error(
                    "rate_limit_route decorator requires Request parameter "
                    "in function signature"
                )
                return await func(*args, **kwargs)

            x_forwarded_for = request.headers.get("X-Forwarded-For", "")
            rate_key = get_real_ip(x_forwarded_for)
            # Defensive fail-open: unit tests often pass a
            # MagicMock(spec=Request), where ``headers.get`` returns a
            # Mock and ``rate_key`` is therefore not a string. In that
            # case we can't meaningfully hash an IP, so skip rate
            # limiting rather than crashing in hash_ip_with_salt.
            if not isinstance(rate_key, str):
                return await func(*args, **kwargs)
            route = route_path if route_path else request.url.path
            if not isinstance(route, str):
                return await func(*args, **kwargs)
            httpx_client = getattr(request.state, "httpx_client", None)

            if not await check_route_rate_limit(
                route,
                rate_key,
                daily_limit=daily_limit,
                httpx_client=httpx_client,
                config_key=config_key,
            ):
                logger.warning("Rate limit exceeded for route %s: %s", route, rate_key)
                detail = error_message or f"Daily rate limit exceeded for {route}"
                raise HTTPException(status_code=429, detail=detail)

            return await func(*args, **kwargs)

        return wrapper

    return decorator
