import json
from typing import Any

from aichat.serve.backend.litellm import get_global_router
from aichat.serve.services.model_settings import model_settings


def _triton_response_to_dict(triton_response: Any) -> dict:
    if isinstance(triton_response, dict):
        return triton_response
    json_fn = getattr(triton_response, "json", None)
    if callable(json_fn):
        return json_fn()
    raise ValueError(f"Unexpected Triton response type: {type(triton_response)}")


def _allm_passthrough_kwargs(model_id: str, *, json: dict[str, Any]) -> dict[str, Any]:
    cfg = model_settings.models.get(model_id) or {}
    method = cfg.get("method", "post")
    if isinstance(method, str):
        method = method.upper()
    kwargs: dict[str, Any] = {
        "model": model_id,
        "method": method,
        "json": json,
    }
    if endpoint := cfg.get("endpoint"):
        kwargs["endpoint"] = endpoint
    if address := cfg.get("address"):
        kwargs["api_base"] = address
    if api_key := cfg.get("api_key"):
        kwargs["api_key"] = api_key
    return kwargs


def _bytes_tensor(name: str, payload: str) -> dict:
    return {
        "name": name,
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": [[payload]],
    }


def _serialize_json_field(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _decode_result_json(triton_data: dict) -> dict:
    outputs = triton_data.get("outputs") or []
    if not outputs:
        raise ValueError("No outputs in Triton response")
    out = next((o for o in outputs if o.get("name") == "result_json"), outputs[0])
    data = out.get("data")
    if data is None:
        raise ValueError("No data in result_json output")
    raw: Any = data[0]
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raw = str(raw)
    return json.loads(raw)


def format_system_one_response(
    backend_payload: dict,
    *,
    response_model: str,
    include_routing: bool,
    route_only: bool,
) -> dict:
    if route_only:
        routing = backend_payload.get("routing")
        if routing is None:
            raise ValueError("route_only response missing routing")
        return {"model": response_model, "routing": routing}

    answers = backend_payload.get("answers")
    if answers is None:
        raise ValueError("response missing answers")

    result: dict[str, Any] = {
        "model": response_model,
        "answers": answers,
        "usage": backend_payload.get("usage", {"input_tokens": 0, "output_tokens": 0}),
    }
    if include_routing and "routing" in backend_payload:
        result["routing"] = backend_payload["routing"]
    return result


def build_triton_request(
    *,
    state: Any,
    questions: Any,
    model_override: str | None,
    route_only: bool,
) -> dict:
    inputs = [
        _bytes_tensor("state_json", _serialize_json_field(state)),
        _bytes_tensor("questions_json", _serialize_json_field(questions)),
    ]
    if model_override:
        inputs.append(_bytes_tensor("model_override", model_override))
    if route_only:
        inputs.append(_bytes_tensor("route_only", "true"))
    return {
        "inputs": inputs,
        "outputs": [{"name": "result_json"}],
    }


async def run_system_one(
    *,
    model_id: str,
    state: Any,
    questions: Any,
    model_override: str | None = None,
    route_only: bool = False,
    response_model: str,
    include_routing: bool = False,
) -> dict:
    router = get_global_router()
    triton_request = build_triton_request(
        state=state,
        questions=questions,
        model_override=model_override,
        route_only=route_only,
    )
    triton_response = await router.allm_passthrough_route(
        **_allm_passthrough_kwargs(model_id, json=triton_request),
    )
    backend_payload = _decode_result_json(_triton_response_to_dict(triton_response))
    return format_system_one_response(
        backend_payload,
        response_model=response_model,
        include_routing=include_routing,
        route_only=route_only,
    )
