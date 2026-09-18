import base64
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from fastapi import Request

from aichat.serve.share_api import (
    _cors_headers,
    create_share,
    delete_share,
    get_share,
    options_share,
)


@pytest.fixture(autouse=True)
def _bypass_rate_limit(monkeypatch):
    async def _always_allow(*args, **kwargs):
        return True

    monkeypatch.setattr(
        "aichat.serve.rate_limiting.check_route_rate_limit", _always_allow
    )


def _make_client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": ""}}, "operation")


def _s3_body_response(text: str) -> dict:
    mock_body = AsyncMock()
    mock_body.read = AsyncMock(return_value=text.encode("utf-8"))
    return {"Body": mock_body}


def _mock_s3_session(
    put_object=None,
    get_object=None,
    delete_object=None,
) -> MagicMock:
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

    mock_session = MagicMock()
    mock_session.create_client = MagicMock(return_value=mock_ctx)
    return mock_session


def _mock_request(body: dict) -> Request:
    request = MagicMock(spec=Request)
    request.json = AsyncMock(return_value=body)
    mock_headers = MagicMock()
    mock_headers.get = MagicMock(return_value="")
    request.headers = mock_headers
    return request


def _valid_ciphertext() -> str:
    return base64.b64encode(b"encrypted-conversation-data").decode("utf-8")


class TestCreateShare:
    @pytest.mark.asyncio
    async def test_success(self):
        request = _mock_request({"ciphertext": _valid_ciphertext()})
        with (
            patch("aichat.serve.share_api._S3_SESSION", _mock_s3_session()),
            patch("aichat.serve.share_api.SHARE_CREATED") as mock_metric,
        ):
            response = await create_share(request, is_valid_x_brave_key=True)

        assert response.status_code == 201
        body = json.loads(response.body)
        assert "share_id" in body
        assert "deletion_id" in body
        mock_metric.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_auth(self):
        request = _mock_request({"ciphertext": _valid_ciphertext()})
        response = await create_share(request, is_valid_x_brave_key=False)

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_ciphertext(self):
        request = _mock_request({})
        response = await create_share(request, is_valid_x_brave_key=True)

        assert response.status_code == 400
        assert b"ciphertext is required" in response.body

    @pytest.mark.asyncio
    async def test_empty_ciphertext(self):
        request = _mock_request({"ciphertext": ""})
        response = await create_share(request, is_valid_x_brave_key=True)

        assert response.status_code == 400
        assert b"ciphertext is required" in response.body

    @pytest.mark.asyncio
    async def test_invalid_base64(self):
        request = _mock_request({"ciphertext": "not-valid-base64!!!"})
        response = await create_share(request, is_valid_x_brave_key=True)

        assert response.status_code == 400
        assert b"not valid base64" in response.body

    @pytest.mark.asyncio
    async def test_ciphertext_exceeds_size_limit(self):
        # Generate a valid base64 string whose decoded size exceeds 1 MB
        oversized = base64.b64encode(b"x" * (1048576 + 1)).decode("utf-8")
        request = _mock_request({"ciphertext": oversized})
        response = await create_share(request, is_valid_x_brave_key=True)

        assert response.status_code == 400
        assert b"exceeds maximum size limit" in response.body

    @pytest.mark.asyncio
    async def test_ciphertext_exactly_at_size_limit(self):
        # Exactly 1 MB decoded is permitted (limit check is >, not >=)
        at_limit = base64.b64encode(b"x" * 1048576).decode("utf-8")
        request = _mock_request({"ciphertext": at_limit})
        with patch("aichat.serve.share_api._S3_SESSION", _mock_s3_session()):
            response = await create_share(request, is_valid_x_brave_key=True)

        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        request = _mock_request({})
        request.json = AsyncMock(side_effect=ValueError("bad json"))
        response = await create_share(request, is_valid_x_brave_key=True)

        assert response.status_code == 400
        assert b"Invalid JSON" in response.body

    @pytest.mark.asyncio
    async def test_sidecar_write_failure(self):
        request = _mock_request({"ciphertext": _valid_ciphertext()})
        mock_session = _mock_s3_session(
            put_object=AsyncMock(side_effect=Exception("S3 unavailable"))
        )

        with patch("aichat.serve.share_api._S3_SESSION", mock_session):
            response = await create_share(request, is_valid_x_brave_key=True)

        assert response.status_code == 500
        mock_s3 = mock_session.create_client.return_value.__aenter__.return_value
        assert mock_s3.put_object.call_count == 1
        mock_s3.delete_object.assert_not_called()

    @pytest.mark.asyncio
    async def test_share_write_failure_after_sidecar_succeeds(self):
        request = _mock_request({"ciphertext": _valid_ciphertext()})
        mock_session = _mock_s3_session(
            put_object=AsyncMock(side_effect=[{}, Exception("share write unavailable")])
        )

        with patch("aichat.serve.share_api._S3_SESSION", mock_session):
            response = await create_share(request, is_valid_x_brave_key=True)

        assert response.status_code == 500
        mock_s3 = mock_session.create_client.return_value.__aenter__.return_value
        assert mock_s3.put_object.call_count == 2
        mock_s3.delete_object.assert_not_called()

    @pytest.mark.asyncio
    async def test_share_id_and_deletion_id_are_distinct_uuids(self):
        request = _mock_request({"ciphertext": _valid_ciphertext()})
        with patch("aichat.serve.share_api._S3_SESSION", _mock_s3_session()):
            response = await create_share(request, is_valid_x_brave_key=True)

        body = json.loads(response.body)
        share_uuid = uuid.UUID(body["share_id"])  # raises ValueError if not a UUID
        deletion_uuid = uuid.UUID(body["deletion_id"])
        assert share_uuid != deletion_uuid

    @pytest.mark.asyncio
    async def test_s3_keys_and_bodies_correct(self):
        ct = _valid_ciphertext()
        request = _mock_request({"ciphertext": ct})
        mock_session = _mock_s3_session()

        with patch("aichat.serve.share_api._S3_SESSION", mock_session):
            response = await create_share(request, is_valid_x_brave_key=True)

        body = json.loads(response.body)
        share_id = body["share_id"]
        deletion_id = body["deletion_id"]
        mock_s3 = mock_session.create_client.return_value.__aenter__.return_value
        assert mock_s3.put_object.call_count == 2

        sidecar_call = mock_s3.put_object.call_args_list[0][1]
        assert sidecar_call["Key"] == f"deletions/{deletion_id}"
        assert sidecar_call["Body"] == share_id.encode("utf-8")
        assert sidecar_call["ChecksumAlgorithm"] == "CRC32"

        share_call = mock_s3.put_object.call_args_list[1][1]
        assert share_call["Key"] == f"shares/{share_id}"
        assert share_call["Body"] == ct.encode("utf-8")
        assert share_call["ChecksumAlgorithm"] == "CRC32"


class TestGetShare:
    @pytest.mark.asyncio
    async def test_success(self):
        ct = _valid_ciphertext()
        mock_session = _mock_s3_session(
            get_object=AsyncMock(return_value=_s3_body_response(ct))
        )

        with (
            patch("aichat.serve.share_api._S3_SESSION", mock_session),
            patch("aichat.serve.share_api.SHARE_ACCESSED") as mock_metric,
        ):
            response = await get_share("test-share-id")

        assert response.status_code == 200
        body = json.loads(response.body)
        assert body["ciphertext"] == ct
        mock_metric.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_not_found_no_such_key(self):
        mock_session = _mock_s3_session(
            get_object=AsyncMock(side_effect=_make_client_error("NoSuchKey"))
        )

        with (
            patch("aichat.serve.share_api._S3_SESSION", mock_session),
            patch("aichat.serve.share_api.SHARE_NOT_FOUND") as mock_metric,
        ):
            response = await get_share("missing-id")

        assert response.status_code == 404
        mock_metric.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_not_found_404_code(self):
        mock_session = _mock_s3_session(
            get_object=AsyncMock(side_effect=_make_client_error("404"))
        )

        with (
            patch("aichat.serve.share_api._S3_SESSION", mock_session),
            patch("aichat.serve.share_api.SHARE_NOT_FOUND") as mock_metric,
        ):
            response = await get_share("missing-id")

        assert response.status_code == 404
        mock_metric.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_s3_read_failure(self):
        mock_session = _mock_s3_session(
            get_object=AsyncMock(side_effect=_make_client_error("InternalError"))
        )

        with patch("aichat.serve.share_api._S3_SESSION", mock_session):
            response = await get_share("test-id")

        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_unexpected_exception(self):
        mock_session = _mock_s3_session(
            get_object=AsyncMock(side_effect=Exception("connection timeout"))
        )

        with patch("aichat.serve.share_api._S3_SESSION", mock_session):
            response = await get_share("test-id")

        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_cors_header_present_when_origin_configured(self):
        ct = _valid_ciphertext()
        mock_session = _mock_s3_session(
            get_object=AsyncMock(return_value=_s3_body_response(ct))
        )

        with (
            patch("aichat.serve.share_api._S3_SESSION", mock_session),
            patch("aichat.serve.share_api.external_service_settings") as mock_settings,
        ):
            mock_settings.share_viewer_origin = "https://brave.ai"
            mock_settings.share_s3_bucket = "test-bucket"
            response = await get_share("test-id")

        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "https://brave.ai"

    @pytest.mark.asyncio
    async def test_cors_header_absent_when_origin_not_configured(self):
        ct = _valid_ciphertext()
        mock_session = _mock_s3_session(
            get_object=AsyncMock(return_value=_s3_body_response(ct))
        )

        with (
            patch("aichat.serve.share_api._S3_SESSION", mock_session),
            patch("aichat.serve.share_api.external_service_settings") as mock_settings,
        ):
            mock_settings.share_viewer_origin = ""
            mock_settings.share_s3_bucket = "test-bucket"
            response = await get_share("test-id")

        assert "access-control-allow-origin" not in response.headers

    @pytest.mark.asyncio
    async def test_cors_header_present_on_not_found(self):
        # Without CORS headers the browser blocks the 404 outright, so the
        # viewer cannot tell a deleted share from an unreachable server.
        mock_session = _mock_s3_session(
            get_object=AsyncMock(side_effect=_make_client_error("NoSuchKey"))
        )

        with (
            patch("aichat.serve.share_api._S3_SESSION", mock_session),
            patch("aichat.serve.share_api.external_service_settings") as mock_settings,
        ):
            mock_settings.share_viewer_origin = "https://brave.ai"
            mock_settings.share_s3_bucket = "test-bucket"
            response = await get_share("missing-id")

        assert response.status_code == 404
        assert response.headers.get("access-control-allow-origin") == "https://brave.ai"

    @pytest.mark.asyncio
    async def test_cors_header_present_on_server_error(self):
        mock_session = _mock_s3_session(
            get_object=AsyncMock(side_effect=Exception("connection timeout"))
        )

        with (
            patch("aichat.serve.share_api._S3_SESSION", mock_session),
            patch("aichat.serve.share_api.external_service_settings") as mock_settings,
        ):
            mock_settings.share_viewer_origin = "https://brave.ai"
            mock_settings.share_s3_bucket = "test-bucket"
            response = await get_share("test-id")

        assert response.status_code == 500
        assert response.headers.get("access-control-allow-origin") == "https://brave.ai"

    @pytest.mark.asyncio
    async def test_cors_header_absent_on_not_found_when_origin_not_configured(self):
        mock_session = _mock_s3_session(
            get_object=AsyncMock(side_effect=_make_client_error("NoSuchKey"))
        )

        with (
            patch("aichat.serve.share_api._S3_SESSION", mock_session),
            patch("aichat.serve.share_api.external_service_settings") as mock_settings,
        ):
            mock_settings.share_viewer_origin = ""
            mock_settings.share_s3_bucket = "test-bucket"
            response = await get_share("missing-id")

        assert response.status_code == 404
        assert "access-control-allow-origin" not in response.headers


class TestDeleteShare:
    @pytest.mark.asyncio
    async def test_success(self):
        share_id = str(uuid.uuid4())
        deletion_id = str(uuid.uuid4())
        mock_session = _mock_s3_session(
            get_object=AsyncMock(return_value=_s3_body_response(share_id))
        )
        request = _mock_request({"deletion_id": deletion_id})

        with (
            patch("aichat.serve.share_api._S3_SESSION", mock_session),
            patch("aichat.serve.share_api.SHARE_DELETED") as mock_metric,
        ):
            response = await delete_share(request, is_valid_x_brave_key=True)

        assert response.status_code == 200
        mock_metric.inc.assert_called_once()

        mock_s3 = mock_session.create_client.return_value.__aenter__.return_value
        mock_s3.get_object.assert_called_once()
        assert mock_s3.get_object.call_args[1]["Key"] == f"deletions/{deletion_id}"

        assert mock_s3.delete_object.call_count == 2
        first_delete = mock_s3.delete_object.call_args_list[0][1]
        second_delete = mock_s3.delete_object.call_args_list[1][1]
        assert first_delete["Key"] == f"shares/{share_id}"
        assert second_delete["Key"] == f"deletions/{deletion_id}"

    @pytest.mark.asyncio
    async def test_not_found_no_such_key(self):
        mock_session = _mock_s3_session(
            get_object=AsyncMock(side_effect=_make_client_error("NoSuchKey"))
        )
        request = _mock_request({"deletion_id": "missing-deletion-id"})

        with (
            patch("aichat.serve.share_api._S3_SESSION", mock_session),
            patch("aichat.serve.share_api.SHARE_DELETE_NOT_FOUND") as mock_metric,
        ):
            response = await delete_share(request, is_valid_x_brave_key=True)

        assert response.status_code == 404
        mock_metric.inc.assert_called_once()
        mock_s3 = mock_session.create_client.return_value.__aenter__.return_value
        mock_s3.delete_object.assert_not_called()

    @pytest.mark.asyncio
    async def test_not_found_404_code(self):
        mock_session = _mock_s3_session(
            get_object=AsyncMock(side_effect=_make_client_error("404"))
        )
        request = _mock_request({"deletion_id": "missing-deletion-id"})

        with (
            patch("aichat.serve.share_api._S3_SESSION", mock_session),
            patch("aichat.serve.share_api.SHARE_DELETE_NOT_FOUND") as mock_metric,
        ):
            response = await delete_share(request, is_valid_x_brave_key=True)

        assert response.status_code == 404
        mock_metric.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_auth(self):
        request = _mock_request({"deletion_id": "some-id"})
        response = await delete_share(request, is_valid_x_brave_key=False)

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_deletion_id(self):
        request = _mock_request({})
        response = await delete_share(request, is_valid_x_brave_key=True)

        assert response.status_code == 400
        assert b"deletion_id is required" in response.body

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        request = _mock_request({})
        request.json = AsyncMock(side_effect=ValueError("bad json"))
        response = await delete_share(request, is_valid_x_brave_key=True)

        assert response.status_code == 400
        assert b"Invalid JSON" in response.body

    @pytest.mark.asyncio
    async def test_sidecar_read_unexpected_exception(self):
        mock_session = _mock_s3_session(
            get_object=AsyncMock(side_effect=Exception("connection timeout"))
        )
        request = _mock_request({"deletion_id": "some-id"})

        with patch("aichat.serve.share_api._S3_SESSION", mock_session):
            response = await delete_share(request, is_valid_x_brave_key=True)

        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_share_delete_failure_does_not_touch_sidecar(self):
        share_id = str(uuid.uuid4())
        deletion_id = str(uuid.uuid4())
        mock_session = _mock_s3_session(
            get_object=AsyncMock(return_value=_s3_body_response(share_id)),
            delete_object=AsyncMock(side_effect=Exception("delete unavailable")),
        )
        request = _mock_request({"deletion_id": deletion_id})

        with patch("aichat.serve.share_api._S3_SESSION", mock_session):
            response = await delete_share(request, is_valid_x_brave_key=True)

        assert response.status_code == 500
        mock_s3 = mock_session.create_client.return_value.__aenter__.return_value
        mock_s3.delete_object.assert_called_once()
        assert mock_s3.delete_object.call_args[1]["Key"] == f"shares/{share_id}"

    @pytest.mark.asyncio
    async def test_sidecar_cleanup_failure_after_share_deleted_still_succeeds(self):
        share_id = str(uuid.uuid4())
        deletion_id = str(uuid.uuid4())
        mock_session = _mock_s3_session(
            get_object=AsyncMock(return_value=_s3_body_response(share_id)),
            delete_object=AsyncMock(side_effect=[{}, Exception("cleanup unavailable")]),
        )
        request = _mock_request({"deletion_id": deletion_id})

        with (
            patch("aichat.serve.share_api._S3_SESSION", mock_session),
            patch("aichat.serve.share_api.SHARE_DELETED") as mock_metric,
        ):
            response = await delete_share(request, is_valid_x_brave_key=True)

        assert response.status_code == 200
        mock_metric.inc.assert_called_once()
        mock_s3 = mock_session.create_client.return_value.__aenter__.return_value
        assert mock_s3.delete_object.call_count == 2


class TestOptionsShare:
    @pytest.mark.asyncio
    async def test_returns_required_headers(self):
        with patch("aichat.serve.share_api.external_service_settings") as mock_settings:
            mock_settings.share_viewer_origin = ""
            response = await options_share("any-id")

        assert response.status_code == 200
        assert response.headers.get("access-control-allow-methods") == "GET, OPTIONS"
        assert response.headers.get("access-control-allow-headers") == "Content-Type"

    @pytest.mark.asyncio
    async def test_includes_cors_origin_when_configured(self):
        with patch("aichat.serve.share_api.external_service_settings") as mock_settings:
            mock_settings.share_viewer_origin = "https://brave.ai"
            response = await options_share("any-id")

        assert response.headers.get("access-control-allow-origin") == "https://brave.ai"

    @pytest.mark.asyncio
    async def test_excludes_cors_origin_when_not_configured(self):
        with patch("aichat.serve.share_api.external_service_settings") as mock_settings:
            mock_settings.share_viewer_origin = ""
            response = await options_share("any-id")

        assert "access-control-allow-origin" not in response.headers


class TestCorsHeaders:
    def test_returns_origin_header_when_configured(self):
        with patch("aichat.serve.share_api.external_service_settings") as mock_settings:
            mock_settings.share_viewer_origin = "https://brave.ai"
            headers = _cors_headers()
        assert headers == {"Access-Control-Allow-Origin": "https://brave.ai"}

    def test_returns_empty_dict_when_not_configured(self):
        with patch("aichat.serve.share_api.external_service_settings") as mock_settings:
            mock_settings.share_viewer_origin = ""
            headers = _cors_headers()
        assert headers == {}
