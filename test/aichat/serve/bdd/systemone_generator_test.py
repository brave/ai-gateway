# BDD: system one Triton response decoding (aichat/serve/services/system_one/generator.py).
# TODO: remove test_decode_result_json_from_triton_bytes_output \
#   test/aichat/serve/services/system_one_generator_test.py:70
# (dict passthrough branch of _triton_response_to_dict)

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve.services.system_one import generator as s1gen

FEATURE = "features/systemone_generator.feature"
scenarios(FEATURE)


@pytest.fixture
def ctx():
    return {"error": None}


@given("the system one generator harness")
def given_harness(ctx):
    pass


@given(parsers.parse("the Triton response is the dict {payload}"))
def given_dict_response(ctx, payload):
    import ast

    ctx["response"] = ast.literal_eval(payload)


@given("the Triton response object exposes a json method")
def given_json_method(ctx):
    class _Resp:
        def json(self):
            return {"outputs": [{"name": "result_json", "data": ['{"answers": []}']}]}

    ctx["response"] = _Resp()


@given("the Triton response is an object with a non-callable json attribute")
def given_noncallable_json(ctx):
    ctx["response"] = type("Bad", (), {"json": 123})()


@given("a Triton result dict without outputs")
def given_no_outputs(ctx):
    ctx["result_dict"] = {"outputs": []}


@given("a Triton result whose output has no data")
def given_no_data(ctx):
    ctx["result_dict"] = {"outputs": [{"name": "result_json"}]}


@when("the Triton response is converted to a dict")
def when_convert(ctx):
    try:
        ctx["converted"] = s1gen._triton_response_to_dict(ctx["response"])
    except ValueError as e:
        ctx["error"] = str(e)


@when("the result json is decoded")
def when_decode(ctx):
    try:
        ctx["decoded"] = s1gen._decode_result_json(ctx["result_dict"])
    except ValueError as e:
        ctx["error"] = str(e)


@then("the converted dict keeps the outputs")
def then_converted(ctx):
    assert "outputs" in ctx["converted"]
    assert ctx["error"] is None


_TOPIC_PATTERNS = {
    "missing outputs": "No outputs",
    "missing data": "No data",
    "the response type": "response type",
}


@then(parsers.parse("a ValueError about {topic} is raised"))
def then_value_error(ctx, topic):
    assert ctx["error"] is not None
    assert _TOPIC_PATTERNS[topic] in ctx["error"]
