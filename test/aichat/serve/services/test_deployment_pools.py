from unittest.mock import patch

from aichat.llm.metrics import DEPLOYMENT_POOL_ROUTE_TOTAL
from aichat.serve.services.deployment_pools import (
    POOL_LONG,
    POOL_SHORT,
    classify_deployment_pool,
    record_deployment_pool_route,
    short_admission_token_budget,
)

_POOLS = {
    "short_text": {"address": "http://short/v1", "max_model_tokens": 20480},
    "long_text": {"address": "http://long/v1", "max_model_tokens": 81920},
    "image": {"address": None, "enabled": False},
}


def test_short_admission_budget():
    assert short_admission_token_budget(_POOLS) == 20480 - 512


def test_short_admission_budget_explicit_admission_max_tokens():
    pools = {
        **_POOLS,
        "short_text": {
            **_POOLS["short_text"],
            "max_model_tokens": 20480,
            "admission_max_tokens": 12000,
        },
    }
    assert short_admission_token_budget(pools) == 12000


def test_classify_short_text():
    route = classify_deployment_pool(
        [{"role": "user", "content": "hello"}],
        max_tokens=1024,
        pools=_POOLS,
    )
    assert route.pool == POOL_SHORT
    assert route.tag == "pool:short"
    assert route.reason == "short"


def test_classify_long_when_over_budget():
    with patch(
        "aichat.serve.services.deployment_pools.calculate_message_tokens",
        return_value=25000,
    ):
        route = classify_deployment_pool(
            [{"role": "user", "content": "large conversation"}],
            max_tokens=8192,
            pools=_POOLS,
        )
    assert route.pool == POOL_LONG
    assert route.reason == "long"


def test_classify_interim_vision_to_long():
    route = classify_deployment_pool(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                ],
            }
        ],
        max_tokens=1024,
        pools=_POOLS,
    )
    assert route.pool == POOL_LONG
    assert route.reason == "interim_vision"


def test_record_shadow_metric():
    route = classify_deployment_pool(
        [{"role": "user", "content": "hi"}],
        max_tokens=100,
        pools=_POOLS,
    )
    before = DEPLOYMENT_POOL_ROUTE_TOTAL.labels(
        "qwen-14b-instruct", POOL_SHORT, "short", "shadow"
    )._value.get()
    record_deployment_pool_route("qwen-14b-instruct", route, routing_active=False)
    after = DEPLOYMENT_POOL_ROUTE_TOTAL.labels(
        "qwen-14b-instruct", POOL_SHORT, "short", "shadow"
    )._value.get()
    assert after == before + 1
