from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("pydantic_settings")
pytest.importorskip("redis")
pytest.importorskip("httpx")

import httpx

from aichat.serve.api_server import app
from aichat.serve.auth import check_x_brave_key


def _make_upstream_response(
    status_code=200, content=b"ohttp-response-bytes", content_type="message/ohttp-res"
):
    upstream = MagicMock()
    upstream.status_code = status_code
    upstream.headers = {"content-type": content_type}

    async def _aiter_bytes():
        yield content

    async def _aclose():
        pass

    upstream.aiter_bytes = _aiter_bytes
    upstream.aclose = _aclose
    return upstream


def _make_httpx_client(upstream_response):
    """Return a mock httpx.AsyncClient that yields upstream_response on send()."""
    client = MagicMock()
    client.build_request.return_value = MagicMock()
    client.send = AsyncMock(return_value=upstream_response)
    client.aclose = AsyncMock()
    return client


@pytest.fixture
def client():
    app.dependency_overrides[check_x_brave_key] = lambda: True
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


_OHTTP_CONFIG = {
    "key_config": "a2V5X2NvbmZpZw==",
    "signing_key": "c2lnbmluZ19rZXk=",
    "signature": "c2lnbmF0dXJl",
    "endpoint_url": "https://example.near.ai/v1/chat/completions",
}

_E2EE_MODEL = {
    "address": "https://example.near.ai/v1/",
    "e2ee_support": True,
}


class TestGetModelOhttpConfig:
    def test_returns_config_for_valid_model(self, client):
        with (
            patch("aichat.serve.ohttp.model_settings") as mock_ms,
            patch(
                "aichat.serve.ohttp.near.verify_and_get_ohttp_config",
                new_callable=AsyncMock,
            ) as mock_verify,
        ):
            mock_ms.models = {"near-model": _E2EE_MODEL}
            mock_verify.return_value = _OHTTP_CONFIG

            response = client.get("/v1/models/near-model/ohttp_config")

        assert response.status_code == 200
        assert response.json() == _OHTTP_CONFIG
        mock_verify.assert_awaited_once_with("near-model")

    def test_returns_404_for_unknown_model(self, client):
        with patch("aichat.serve.ohttp.model_settings") as mock_ms:
            mock_ms.models = {}

            response = client.get("/v1/models/unknown-model/ohttp_config")

        assert response.status_code == 404
        assert response.json()["error"]["type"] == "40401"

    def test_returns_404_for_model_without_e2ee(self, client):
        with patch("aichat.serve.ohttp.model_settings") as mock_ms:
            mock_ms.models = {"plain-model": {"address": "https://example.ai/v1/"}}

            response = client.get("/v1/models/plain-model/ohttp_config")

        assert response.status_code == 404
        assert response.json()["error"]["type"] == "40401"

    def test_returns_500_when_near_verify_fails(self, client):
        with (
            patch("aichat.serve.ohttp.model_settings") as mock_ms,
            patch(
                "aichat.serve.ohttp.near.verify_and_get_ohttp_config",
                new_callable=AsyncMock,
            ) as mock_verify,
        ):
            mock_ms.models = {"near-model": _E2EE_MODEL}
            mock_verify.return_value = None

            response = client.get("/v1/models/near-model/ohttp_config")

        assert response.status_code == 500
        assert response.json()["error"]["type"] == "50001"

    def test_returns_401_for_invalid_auth(self):
        app.dependency_overrides[check_x_brave_key] = lambda: False
        try:
            with TestClient(app) as test_client:
                response = test_client.get("/v1/models/near-model/ohttp_config")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 401
        assert response.json()["error"]["type"] == "40101"


class TestRelayModelOhttp:
    def _post(
        self,
        client,
        model_name="near-model",
        body=b"ohttp-request-bytes",
        content_type="message/ohttp-req",
    ):
        return client.post(
            f"/v1/models/{model_name}/relay",
            content=body,
            headers={"content-type": content_type},
        )

    def test_proxies_request_and_streams_response(self, client):
        upstream = _make_upstream_response(content=b"chunk1")
        mock_httpx_client = _make_httpx_client(upstream)

        with (
            patch("aichat.serve.ohttp.model_settings") as mock_ms,
            patch(
                "aichat.serve.ohttp.check_requests_common",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "aichat.serve.ohttp.near.verify_and_get_ohttp_config",
                new_callable=AsyncMock,
            ) as mock_verify,
            patch("aichat.serve.ohttp.external_service_settings") as mock_ext,
            patch(
                "aichat.serve.ohttp.httpx.AsyncClient", return_value=mock_httpx_client
            ),
        ):
            mock_ms.models = {"near-model": _E2EE_MODEL}
            mock_verify.return_value = _OHTTP_CONFIG
            mock_ext.near_api_key = "test-api-key"

            response = self._post(client)

        assert response.status_code == 200
        assert response.content == b"chunk1"
        assert response.headers["content-type"] == "message/ohttp-res"

        mock_verify.assert_awaited_once_with("near-model")
        mock_httpx_client.send.assert_awaited_once()
        mock_httpx_client.build_request.assert_called_once_with(
            "POST",
            "https://example.near.ai/ohttp",
            headers={
                "Authorization": "Bearer test-api-key",
                "content-type": "message/ohttp-req",
            },
            content=b"ohttp-request-bytes",
        )

    def test_returns_404_for_unknown_model(self, client):
        with (
            patch("aichat.serve.ohttp.model_settings") as mock_ms,
            patch(
                "aichat.serve.ohttp.check_requests_common",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            mock_ms.models = {}
            response = self._post(client, model_name="unknown-model")

        assert response.status_code == 404
        assert response.json()["error"]["type"] == "40401"

    def test_returns_404_for_model_without_e2ee(self, client):
        with (
            patch("aichat.serve.ohttp.model_settings") as mock_ms,
            patch(
                "aichat.serve.ohttp.check_requests_common",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            mock_ms.models = {"plain-model": {"address": "https://example.ai/v1/"}}
            response = self._post(client, model_name="plain-model")

        assert response.status_code == 404
        assert response.json()["error"]["type"] == "40401"

    def test_returns_500_when_near_verify_fails(self, client):
        with (
            patch("aichat.serve.ohttp.model_settings") as mock_ms,
            patch(
                "aichat.serve.ohttp.check_requests_common",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "aichat.serve.ohttp.near.verify_and_get_ohttp_config",
                new_callable=AsyncMock,
            ) as mock_verify,
        ):
            mock_ms.models = {"near-model": _E2EE_MODEL}
            mock_verify.return_value = None
            response = self._post(client)

        assert response.status_code == 500
        assert response.json()["error"]["type"] == "50001"

    def test_returns_500_when_near_api_key_missing(self, client):
        with (
            patch("aichat.serve.ohttp.model_settings") as mock_ms,
            patch(
                "aichat.serve.ohttp.check_requests_common",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "aichat.serve.ohttp.near.verify_and_get_ohttp_config",
                new_callable=AsyncMock,
            ) as mock_verify,
            patch("aichat.serve.ohttp.external_service_settings") as mock_ext,
        ):
            mock_ms.models = {"near-model": _E2EE_MODEL}
            mock_verify.return_value = _OHTTP_CONFIG
            mock_ext.near_api_key = None
            response = self._post(client)

        assert response.status_code == 500
        assert response.json()["error"]["type"] == "50001"

    def test_returns_502_on_upstream_http_error(self, client):
        mock_httpx_client = MagicMock()
        mock_httpx_client.build_request.return_value = MagicMock()
        mock_httpx_client.send = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with (
            patch("aichat.serve.ohttp.model_settings") as mock_ms,
            patch(
                "aichat.serve.ohttp.check_requests_common",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "aichat.serve.ohttp.near.verify_and_get_ohttp_config",
                new_callable=AsyncMock,
            ) as mock_verify,
            patch("aichat.serve.ohttp.external_service_settings") as mock_ext,
            patch(
                "aichat.serve.ohttp.httpx.AsyncClient", return_value=mock_httpx_client
            ),
        ):
            mock_ms.models = {"near-model": _E2EE_MODEL}
            mock_verify.return_value = _OHTTP_CONFIG
            mock_ext.near_api_key = "test-api-key"
            response = self._post(client)

        assert response.status_code == 502
        assert response.json()["error"]["type"] == "50201"
