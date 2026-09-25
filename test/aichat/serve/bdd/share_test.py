# BDD coverage for aichat/serve/share_api.py.
#
# TODO: remove test/aichat/serve/share_api_test.py#TestCreateShare#test_success test/aichat/serve/share_api_test.py:79
# TODO: remove test/aichat/serve/share_api_test.py#TestCreateShare#test_invalid_auth test/aichat/serve/share_api_test.py:94
# TODO: remove test/aichat/serve/share_api_test.py#TestCreateShare#test_missing_ciphertext test/aichat/serve/share_api_test.py:101
# TODO: remove test/aichat/serve/share_api_test.py#TestCreateShare#test_invalid_base64 test/aichat/serve/share_api_test.py:117
# TODO: remove test/aichat/serve/share_api_test.py#TestCreateShare#test_ciphertext_exceeds_size_limit test/aichat/serve/share_api_test.py:125
# TODO: remove test/aichat/serve/share_api_test.py#TestCreateShare#test_invalid_json test/aichat/serve/share_api_test.py:145
# TODO: remove test/aichat/serve/share_api_test.py#TestCreateShare#test_share_write_failure_after_sidecar_succeeds test/aichat/serve/share_api_test.py:169
# TODO: remove test/aichat/serve/share_api_test.py#TestGetShare#test_success test/aichat/serve/share_api_test.py:222
# TODO: remove test/aichat/serve/share_api_test.py#TestGetShare#test_not_found_no_such_key test/aichat/serve/share_api_test.py:240
# TODO: remove test/aichat/serve/share_api_test.py#TestGetShare#test_s3_read_failure test/aichat/serve/share_api_test.py:270
# TODO: remove test/aichat/serve/share_api_test.py#TestGetShare#test_cors_header_present_when_origin_configured test/aichat/serve/share_api_test.py:292
# TODO: remove test/aichat/serve/share_api_test.py#TestDeleteShare#test_success test/aichat/serve/share_api_test.py:382
# TODO: remove test/aichat/serve/share_api_test.py#TestDeleteShare#test_not_found_no_such_key test/aichat/serve/share_api_test.py:410
# TODO: remove test/aichat/serve/share_api_test.py#TestDeleteShare#test_sidecar_read_unexpected_exception test/aichat/serve/share_api_test.py:468
# TODO: remove test/aichat/serve/share_api_test.py#TestDeleteShare#test_missing_deletion_id test/aichat/serve/share_api_test.py:451
# NOTE: the sidecar AccessDenied scenario has no legacy unit test — it is new
# coverage for the 500 mapping of client errors on the sidecar read.
# TODO: remove test/aichat/serve/share_api_test.py#TestOptionsShare#test_returns_required_headers test/aichat/serve/share_api_test.py:521

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from botocore.exceptions import ClientError
from fastapi import Request
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve import share_api
from aichat.serve.external_service_settings import external_service_settings

FEATURE = "features/share.feature"
scenarios(FEATURE)


@pytest.fixture
def ctx():
    return {}


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": ""}}, "operation")


def _s3_body(text: str) -> dict:
    body = AsyncMock()
    body.read = AsyncMock(return_value=text.encode("utf-8"))
    return {"Body": body}


def _mock_s3_session(put_object=None, get_object=None, delete_object=None):
    mock_s3 = AsyncMock()
    mock_s3.put_object = (
        put_object if put_object is not None else AsyncMock(return_value={})
    )
    mock_s3.get_object = (
        get_object if get_object is not None else AsyncMock(return_value={})
    )
    mock_s3.delete_object = (
        delete_object if delete_object is not None else AsyncMock(return_value={})
    )
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_s3)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.create_client = MagicMock(return_value=mock_ctx)
    return session


def _recording_session(ctx):
    s3 = AsyncMock()

    async def put(**kwargs):
        ctx["s3_calls"].append(("put", kwargs))
        return {}

    async def get(**kwargs):
        ctx["s3_calls"].append(("get", kwargs))
        key = str(kwargs.get("Key", ""))
        # Deletion sidecars carry the share id, not ciphertext.
        if key.startswith("deletions/"):
            return _s3_body("share-abc-123")
        return _s3_body("ciphertext-body")

    async def delete(**kwargs):
        ctx["s3_calls"].append(("delete", kwargs))
        return {}

    s3.put_object = put
    s3.get_object = get
    s3.delete_object = delete
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=s3)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.create_client = MagicMock(return_value=mock_ctx)
    return session


def _mock_request(body):
    request = MagicMock(spec=Request)
    request.json = AsyncMock(return_value=body)
    return request


def _valid_ciphertext() -> str:
    return base64.b64encode(b"encrypted-conversation-data").decode("utf-8")


@given("the share harness")
def share_harness(ctx, monkeypatch):
    async def _always_allow(*args, **kwargs):
        return True

    monkeypatch.setattr(
        "aichat.serve.rate_limiting.check_route_rate_limit", _always_allow
    )
    ctx["s3_calls"] = []
    monkeypatch.setattr(share_api, "_S3_SESSION", _recording_session(ctx))
    return ctx


@when("a share is created with a valid ciphertext")
def create_share_valid(ctx):
    ctx["response"] = asyncio.run(
        share_api.create_share(
            _mock_request({"ciphertext": _valid_ciphertext()}),
            is_valid_x_brave_key=True,
        )
    )


@when("a share is created with a valid ciphertext and an invalid services key")
def create_share_bad_key(ctx):
    ctx["response"] = asyncio.run(
        share_api.create_share(
            _mock_request({"ciphertext": _valid_ciphertext()}),
            is_valid_x_brave_key=False,
        )
    )


@when("a share is created without a ciphertext")
def create_share_missing(ctx):
    ctx["response"] = asyncio.run(
        share_api.create_share(_mock_request({}), is_valid_x_brave_key=True)
    )


@when("a share is created with a non-base64 ciphertext")
def create_share_bad_b64(ctx):
    ctx["response"] = asyncio.run(
        share_api.create_share(
            _mock_request({"ciphertext": "!!!not-base64!!!"}),
            is_valid_x_brave_key=True,
        )
    )


@when("a share is created with a ciphertext over the size limit")
def create_share_oversized(ctx, monkeypatch):
    monkeypatch.setattr(external_service_settings, "share_max_ciphertext_bytes", 1)
    ctx["response"] = asyncio.run(
        share_api.create_share(
            _mock_request({"ciphertext": _valid_ciphertext()}),
            is_valid_x_brave_key=True,
        )
    )


@when("a share is created with an invalid json body")
def create_share_bad_json(ctx):
    request = MagicMock(spec=Request)
    request.json = AsyncMock(side_effect=ValueError("bad json"))
    ctx["response"] = asyncio.run(
        share_api.create_share(request, is_valid_x_brave_key=True)
    )


@given("the s3 put fails after the sidecar succeeds")
def s3_put_fails(ctx, monkeypatch):
    # The sidecar (deletions/{id}) is written first, then the share. Fail
    # only the share put so the orphaned-sidecar path is exercised, matching
    # the legacy test_share_write_failure_after_sidecar_succeeds.
    session = _recording_session(ctx)
    s3 = session.create_client.return_value.__aenter__.return_value
    inner_put = s3.put_object

    async def put(**kwargs):
        if str(kwargs.get("Key", "")).startswith("shares/"):
            ctx["s3_calls"].append(("put", kwargs))
            raise RuntimeError("s3 down")
        return await inner_put(**kwargs)

    s3.put_object = put
    monkeypatch.setattr(share_api, "_S3_SESSION", session)


@then("the deletion sidecar was written but the share was not")
def orphaned_sidecar_assert(ctx):
    puts = [k for op, k in ctx["s3_calls"] if op == "put"]
    # Exactly one share-put attempt (the one that failed) plus the sidecar
    # puts; everything that succeeded stayed under deletions/.
    share_puts = [k for k in puts if str(k.get("Key", "")).startswith("shares/")]
    assert len(share_puts) == 1
    sidecar_puts = [k for k in puts if str(k.get("Key", "")).startswith("deletions/")]
    assert len(sidecar_puts) == 1
    assert len(puts) == len(share_puts) + len(sidecar_puts)
    # KNOWN BEHAVIOR (not a spec): share-put failure currently leaves the
    # deletion sidecar in place. If prod adds sidecar cleanup on share-put
    # failure, update this scenario deliberately instead of guessing.


@when(parsers.parse('the share "{share_id}" is fetched'))
def get_share(ctx, share_id):
    ctx["response"] = asyncio.run(share_api.get_share(share_id))


def _failing_get(prefix: str, error: Exception):
    """get_object that raises only for the targeted key prefix.

    A handler that reads the wrong key family gets a success body instead of
    the injected failure, so the scenario fails loudly instead of silently.
    """

    async def _get(**kwargs):
        if str(kwargs.get("Key", "")).startswith(prefix):
            raise error
        return _s3_body("ciphertext-body")

    return _get


@given("the s3 get reports NoSuchKey")
def s3_get_no_such_key(ctx, monkeypatch):
    monkeypatch.setattr(
        share_api,
        "_S3_SESSION",
        _mock_s3_session(
            get_object=_failing_get("shares/", _client_error("NoSuchKey"))
        ),
    )


@given("the s3 get raises an unexpected error")
def s3_get_unexpected(ctx, monkeypatch):
    monkeypatch.setattr(
        share_api,
        "_S3_SESSION",
        _mock_s3_session(get_object=_failing_get("shares/", RuntimeError("boom"))),
    )


@given("the sidecar get reports NoSuchKey")
def sidecar_get_no_such_key(ctx, monkeypatch):
    monkeypatch.setattr(
        share_api,
        "_S3_SESSION",
        _mock_s3_session(
            get_object=_failing_get("deletions/", _client_error("NoSuchKey"))
        ),
    )


@given("the sidecar get reports AccessDenied")
def sidecar_get_denied(ctx, monkeypatch):
    monkeypatch.setattr(
        share_api,
        "_S3_SESSION",
        _mock_s3_session(
            get_object=_failing_get("deletions/", _client_error("AccessDenied"))
        ),
    )


@given("the sidecar get raises an unexpected error")
def sidecar_get_unexpected(ctx, monkeypatch):
    monkeypatch.setattr(
        share_api,
        "_S3_SESSION",
        _mock_s3_session(get_object=_failing_get("deletions/", RuntimeError("boom"))),
    )


@given("the share viewer origin is configured")
def viewer_origin(ctx, monkeypatch):
    monkeypatch.setattr(
        external_service_settings, "share_viewer_origin", "https://share.brave.com"
    )


@when(parsers.parse('the deletion "{deletion_id}" is posted'))
def delete_share(ctx, deletion_id):
    ctx["response"] = asyncio.run(
        share_api.delete_share(
            _mock_request({"deletion_id": deletion_id}), is_valid_x_brave_key=True
        )
    )


@when(
    parsers.parse('the deletion "{deletion_id}" is posted with an invalid services key')
)
def delete_share_bad_key(ctx, deletion_id):
    ctx["response"] = asyncio.run(
        share_api.delete_share(
            _mock_request({"deletion_id": deletion_id}), is_valid_x_brave_key=False
        )
    )


@when("the deletion is posted without a deletion id")
def delete_share_missing_id(ctx):
    ctx["response"] = asyncio.run(
        share_api.delete_share(_mock_request({}), is_valid_x_brave_key=True)
    )


@when(parsers.parse('the preflight for "{share_id}" is requested'))
def options_preflight(ctx, share_id):
    ctx["response"] = asyncio.run(share_api.options_share(share_id))


def _body_of(response):
    if hasattr(response, "body"):
        return json.loads(response.body)
    return response


@then("the share is stored at status 201 with distinct share and deletion ids")
def create_success_assert(ctx):
    resp = ctx["response"]
    assert resp.status_code == 201
    body = _body_of(resp)
    assert body["share_id"] != body["deletion_id"]


@then("the sidecar and share objects were written to the bucket")
def sidecar_objects_assert(ctx):
    puts = [call for call in ctx["s3_calls"] if call[0] == "put"]
    assert len(puts) == 2
    keys = [call[1]["Key"] for call in puts]
    assert keys[0].startswith("deletions/")
    assert keys[1].startswith("shares/")
    share_id = keys[1].split("/")[1]
    assert puts[0][1]["Body"] == share_id.encode("utf-8")


@then("the response is an invalid auth key error")
def invalid_auth_key_assert(ctx):
    resp = ctx["response"]
    assert resp.status_code == 401
    assert _body_of(resp)["error"]["type"] == "40101"


@then("the response is a bad request error")
def bad_request_assert(ctx):
    resp = ctx["response"]
    assert resp.status_code == 400
    assert _body_of(resp)["error"]["type"] == "40002"


@then("the response is an internal server error")
def internal_error_assert(ctx):
    resp = ctx["response"]
    assert resp.status_code == 500
    assert _body_of(resp)["error"]["type"] == "50001"


@then('the ciphertext "ciphertext-body" is returned')
def get_success_assert(ctx):
    resp = ctx["response"]
    assert resp.status_code == 200
    assert _body_of(resp) == {"ciphertext": "ciphertext-body"}
    # The get must target the share's own key, not just any non-deletion key.
    gets = [k.get("Key") for kind, k in ctx["s3_calls"] if kind == "get"]
    assert gets == ["shares/share-123"], gets


@then("the response is a share not found error")
def share_not_found_assert(ctx):
    resp = ctx["response"]
    assert resp.status_code == 404
    assert _body_of(resp)["error"]["type"] == "40401"


@then("the response carries the viewer cors header")
def viewer_cors_assert(ctx):
    assert ctx["response"].headers["Access-Control-Allow-Origin"] == (
        "https://share.brave.com"
    )


@then("the response is 200 and the share and sidecar objects are deleted")
def delete_success_assert(ctx):
    resp = ctx["response"]
    assert resp.status_code == 200
    puts = [call for call in ctx["s3_calls"] if call[0] == "put"]
    assert puts == []
    deletes = {call[1]["Key"] for call in ctx["s3_calls"] if call[0] == "delete"}
    assert deletes == {"shares/share-abc-123", "deletions/deletion-1"}


@then('the preflight lists "GET, OPTIONS" and the share content type')
def preflight_assert(ctx):
    resp = ctx["response"]
    assert resp.headers["Access-Control-Allow-Methods"] == "GET, OPTIONS"
    assert resp.headers["Access-Control-Allow-Headers"] == "Content-Type"
