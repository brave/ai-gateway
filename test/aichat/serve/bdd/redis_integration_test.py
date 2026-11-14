# Integration scenarios for the Redis-backed NEAR attestation cache,
# run against a live Redis (the valkey service from docker-compose.test.yml).
# The upstream attestation and its verification are stubbed; everything past
# that boundary (connection pool, GET/SET/TTL on real Redis) is real.

import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import redis.asyncio as aioredis
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve.dependencies import discard_pool, get_pool
from aichat.serve.redis_settings import redis_settings
from aichat.serve.services import near as near_mod
from aichat.serve.services.near import (
    AttestationDetails,
    OHTTPAttestation,
    OHTTPConfig,
    verify_and_get_ohttp_config,
)

FEATURE = "features/redis_integration.feature"
scenarios(FEATURE)

ADDRESS = "https://addr/v1/"
ROUND_TRIP_KEY = "redis-integration:round-trip"


def _b64(hex_value: str) -> str:
    return base64.b64encode(bytes.fromhex(hex_value)).decode()


def _config(key_config: str = "deadbeef") -> OHTTPConfig:
    return OHTTPConfig(
        key_config=_b64(key_config),
        signing_key=_b64("aa" * 32),
        signature=_b64("bb" * 32),
        endpoint_url=f"{ADDRESS}chat/completions",
        upstream_model_name="upstream-x",
    )


def _attestation(model_name: str) -> AttestationDetails:
    return AttestationDetails(
        model_name=model_name,
        compose_actions=[],
        signing_keys=["aa" * 32],
        ohttp_attestation=OHTTPAttestation(
            key_config="deadbeef",
            signing_key="aa" * 32,
            signature="bb" * 32,
        ),
    )


def _setup_near(ctx, monkeypatch, model_name):
    monkeypatch.setattr(
        near_mod,
        "model_settings",
        SimpleNamespace(
            models={
                model_name: {
                    "address": ADDRESS,
                    "upstream_model": "upstream-x",
                }
            }
        ),
    )
    ctx["fetch_mock"] = AsyncMock(return_value=_attestation(model_name))
    monkeypatch.setattr(near_mod, "_fetch_attestation", ctx["fetch_mock"])
    monkeypatch.setattr(near_mod, "_verify_attestation", lambda *_args: ctx["verifies"])
    ctx["cache_key"] = near_mod.NEAR_ATTEST_RESULT_CACHE_KEY.format(
        model_name=model_name
    )


def _run_redis(pool, coro_factory):
    """Run one redis interaction on a fresh loop and drop the pool so the
    next step never reuses connections bound to a closed loop."""
    try:
        return asyncio.run(coro_factory(aioredis.Redis(connection_pool=pool)))
    finally:
        discard_pool()


def _round_trip(pool) -> bytes:
    async def _client(client):
        await client.set(ROUND_TRIP_KEY, "ping")
        return await client.get(ROUND_TRIP_KEY)

    return _run_redis(pool, _client)


@pytest.fixture(scope="session")
def _live_redis():
    """Skip the integration scenarios unless the configured Redis answers
    on a database explicitly designated for tests (REDIS_DB != 0), so a
    database shared with development can never be flushed."""

    async def _ping():
        pool = aioredis.ConnectionPool(
            host=redis_settings.redis_host,
            password=redis_settings.redis_password,
            db=redis_settings.redis_db,
        )
        try:
            return bool(await aioredis.Redis(connection_pool=pool).ping())
        finally:
            await pool.disconnect()

    if redis_settings.redis_db == 0:
        pytest.skip("set REDIS_DB to a non-zero test database to run these scenarios")
    try:
        reachable = asyncio.run(_ping())
    except Exception as exc:
        pytest.skip(f"no reachable redis ({exc.__class__.__name__})")
    else:
        if not reachable:
            pytest.skip("no reachable redis")


@pytest.fixture
def ctx():
    return {}


@given("a reachable redis")
def reachable_redis(_live_redis):
    pass


@given("the cache is empty")
def empty_cache():
    _run_redis(get_pool(), lambda client: client.flushdb())


@when("the pool is created")
def create_pool(ctx):
    discard_pool()
    ctx["pool"] = get_pool()


@when("the pool is requested again")
def request_pool_again(ctx):
    ctx["second_pool"] = get_pool()


@when("the pool is discarded")
def discard_current_pool(ctx):
    discard_pool()


@then("the same pool is returned")
def same_pool_assert(ctx):
    assert ctx["second_pool"] is ctx["pool"]


@then("a new pool is returned")
def new_pool_assert(ctx):
    assert ctx["second_pool"] is not ctx["pool"]


@then("a client from the pool round-trips a value")
def round_trip_assert(ctx):
    assert _round_trip(ctx["pool"]) == b"ping"


@then("a client from the new pool round-trips a value")
def new_pool_round_trip_assert(ctx):
    assert _round_trip(ctx["second_pool"]) == b"ping"


@given(parsers.parse('a fresh attestation for "{model_name}" that {verdict}'))
def fresh_attestation(ctx, model_name, verdict, monkeypatch):
    if verdict not in ("verifies cleanly", "fails verification"):
        raise AssertionError(f"unknown verdict: {verdict!r}")
    ctx["verifies"] = verdict == "verifies cleanly"
    _setup_near(ctx, monkeypatch, model_name)


@given(parsers.parse('the redis cache holds the ohttp config for "{model_name}"'))
def seeded_cache(ctx, model_name, monkeypatch):
    ctx["verifies"] = True
    _setup_near(ctx, monkeypatch, model_name)
    ctx["cached_config"] = _config()

    async def _client(client):
        await client.set(ctx["cache_key"], ctx["cached_config"].model_dump_json())

    _run_redis(get_pool(), _client)


@when(parsers.parse('the ohttp config is verified and fetched for "{model_name}"'))
def verify_and_fetch(ctx, model_name):
    _verify_and_fetch(ctx, model_name)


@when(
    parsers.parse('the ohttp config is verified and fetched again for "{model_name}"')
)
def verify_and_fetch_again(ctx, model_name):
    _verify_and_fetch(ctx, model_name)


def _verify_and_fetch(ctx, model_name):
    ctx["previous_result"] = ctx.get("result")
    try:
        ctx["result"] = asyncio.run(verify_and_get_ohttp_config(model_name))
    finally:
        discard_pool()


@then("the built config is returned")
def built_config_assert(ctx):
    result = ctx["result"]
    assert result is not None
    assert result == _config()


@then("it is stored in redis with a ttl")
def stored_with_ttl_assert(ctx):
    async def _client(client):
        stored = await client.get(ctx["cache_key"])
        return stored, await client.ttl(ctx["cache_key"])

    stored, ttl = _run_redis(get_pool(), _client)
    assert stored.decode() == ctx["result"].model_dump_json()
    assert 0 < ttl <= near_mod.NEAR_ATTEST_RESULT_TTL_SECONDS


@then("the same config is returned without a second fetch")
def cache_reuse_assert(ctx):
    assert ctx["result"] == ctx["previous_result"]
    assert ctx["fetch_mock"].await_count == 1


@then("the cached config is returned without a fetch")
def seeded_cache_hit_assert(ctx):
    assert ctx["result"] == ctx["cached_config"]
    ctx["fetch_mock"].assert_not_awaited()


@then("no ohttp config is returned")
def no_config_assert(ctx):
    assert ctx["result"] is None


@then("redis holds no cached config")
def nothing_cached_assert(ctx):
    async def _client(client):
        return await client.get(ctx["cache_key"])

    assert _run_redis(get_pool(), _client) is None
