"""Tests for System One Triton request and response transforms."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from aichat.serve.services.system_one import generator

MODEL_KEY = "system_one_triton"


def test_build_triton_request_serializes_state_and_questions():
    req = generator.build_triton_request(
        state={"body": "hello"},
        questions={"q": {"type": "noul", "instructions": "test?"}},
        model_override=None,
        route_only=False,
    )
    assert req["outputs"] == [{"name": "result_json"}]
    state_inp = req["inputs"][0]
    assert state_inp["name"] == "state_json"
    assert json.loads(state_inp["data"][0][0]) == {"body": "hello"}


def test_build_triton_request_passes_model_override():
    req = generator.build_triton_request(
        state="x",
        questions={},
        model_override="multilingual",
        route_only=False,
    )
    names = [i["name"] for i in req["inputs"]]
    assert "model_override" in names
    override_inp = next(i for i in req["inputs"] if i["name"] == "model_override")
    assert override_inp["data"][0][0] == "multilingual"


def test_format_system_one_response_strips_routing_by_default():
    payload = {
        "answers": {"q": {"type": "noul", "noul": 0.9}},
        "usage": {"input_tokens": 10, "output_tokens": 0},
        "routing": {"model": "english"},
    }
    out = generator.format_system_one_response(
        payload,
        response_model=MODEL_KEY,
        include_routing=False,
        route_only=False,
    )
    assert out["model"] == MODEL_KEY
    assert "routing" not in out
    assert out["answers"]["q"]["noul"] == 0.9


def test_decode_result_json_from_triton_bytes_output():
    inner = {"answers": {}, "usage": {"input_tokens": 1, "output_tokens": 0}}
    triton = {
        "outputs": [
            {
                "name": "result_json",
                "data": [json.dumps(inner)],
            }
        ]
    }
    assert generator._decode_result_json(triton) == inner


@pytest.mark.asyncio
async def test_run_system_one_calls_passthrough(monkeypatch):
    monkeypatch.setattr(
        generator,
        "model_settings",
        type(
            "S",
            (),
            {
                "models": {
                    MODEL_KEY: {
                        "method": "post",
                        "endpoint": "v2/models/my-triton-model/infer",
                        "address": "https://triton.example/",
                    }
                }
            },
        )(),
    )
    router = MagicMock()
    router.allm_passthrough_route = AsyncMock(
        return_value={
            "outputs": [
                {
                    "name": "result_json",
                    "data": [
                        json.dumps(
                            {
                                "answers": {"q": {"type": "noul", "noul": 1.0}},
                                "usage": {"input_tokens": 5, "output_tokens": 0},
                            }
                        )
                    ],
                }
            ]
        }
    )
    monkeypatch.setattr(generator, "get_global_router", lambda: router)

    result = await generator.run_system_one(
        model_id=MODEL_KEY,
        state="hello",
        questions={"q": {"type": "noul", "instructions": "hi?"}},
        response_model=MODEL_KEY,
    )

    assert result["answers"]["q"]["noul"] == 1.0
    call_kwargs = router.allm_passthrough_route.call_args.kwargs
    assert call_kwargs["model"] == MODEL_KEY
    assert call_kwargs["method"] == "POST"
    assert call_kwargs["endpoint"] == "v2/models/my-triton-model/infer"
    assert call_kwargs["json"]["outputs"][0]["name"] == "result_json"
