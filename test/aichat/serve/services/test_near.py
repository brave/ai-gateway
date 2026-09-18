import base64
import json

import pytest
from pydantic import ValidationError

pytest.importorskip("pydantic_settings")
pytest.importorskip("redis")
pytest.importorskip("boto3")
pytest.importorskip("botocore")
pytest.importorskip("httpx")

from aichat.serve.services import near


class _FakeAsyncClient:
    last_headers: dict | None = None

    def __init__(self, data):
        self._data = data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def get(self, url, headers=None):
        _FakeAsyncClient.last_headers = headers
        return _DummyResponse(self._data)


class _DummyResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def _build_app_compose(image_map: dict[str, str]) -> str:
    lines: list[str] = ["services:"]
    for index, (repository, digest) in enumerate(image_map.items()):
        service_name = f"svc{index}"
        lines.append(f"  {service_name}:")
        lines.append(
            f"    image: ghcr.io/{repository}:stable@sha256:{digest}"  # tag + digest
        )
    docker_compose = "\n".join(lines)
    return json.dumps({"docker_compose_file": docker_compose})


@pytest.fixture
def reset_models(monkeypatch):
    original_models = near.model_settings.models
    monkeypatch.setattr(near.model_settings, "models", original_models.copy())
    return near.model_settings.models


def test_build_ohttp_config_returns_config(reset_models):
    model_name = "near-test"
    upstream_model = "upstream-test-model"
    base_url = "https://example.near.ai/v1/"
    reset_models[model_name] = {"address": base_url, "upstream_model": upstream_model}

    key_config = b"key_config_bytes"
    signing_key = b"signing_key_bytes"
    signature = b"signature_bytes"

    attestation = near.OHTTPAttestation(
        key_config=key_config.hex(),
        signing_key=signing_key.hex(),
        signature=signature.hex(),
    )

    config = near._build_ohttp_config(attestation, model_name)

    assert config is not None
    assert config.model_dump() == {
        "key_config": base64.b64encode(key_config).decode(),
        "signing_key": base64.b64encode(signing_key).decode(),
        "signature": base64.b64encode(signature).decode(),
        "endpoint_url": f"{base_url}chat/completions",
        "upstream_model_name": upstream_model,
    }


def test_ohttp_attestation_rejects_empty_fields():
    with pytest.raises(ValidationError):
        near.OHTTPAttestation(key_config="deadbeef", signing_key="", signature="")


def test_extract_target_digests_returns_all_targets():
    digests = {
        "nearaidev/cloud-api": "a" * 64,
        "nearaidev/cvm-ingress": "b" * 64,
        "nearaidev/dstack-vpc": "c" * 64,
        "nearaidev/dstack-vpc-client": "d" * 64,
    }
    app_compose = _build_app_compose(digests)

    result = near._extract_target_digests(app_compose)

    assert result == {key: value for key, value in digests.items()}


def test_extract_target_digests_requires_all_targets():
    partial = {
        "nearaidev/cloud-api": "a" * 64,
        "nearaidev/cvm-ingress": "b" * 64,
    }

    assert near._extract_target_digests(_build_app_compose(partial)) == {}


def test_extract_target_digests_handles_missing_app_compose():
    assert near._extract_target_digests(None) == {}


def _ohttp_attestation() -> dict[str, str]:
    return {"key_config": "aa", "signing_key": "bb", "signature": "cc"}


def _compose_manager_attestation(image: str) -> dict:
    return {
        "actions": [
            {"action": near.COMPOSE_MANAGER_STARTED_ACTION, "image": image},
        ]
    }


def _cloud_api_report(digests):
    return {
        "gateway_attestation": {
            "signing_address": "gw",
            "info": {"tcb_info": {"app_compose": _build_app_compose(digests)}},
        },
        "model_attestations": [
            {
                "signing_address": "md",
                "ohttp_attestation": _ohttp_attestation(),
                "info": {"tcb_info": {"app_compose": _build_app_compose(digests)}},
                "compose_manager_attestation": _compose_manager_attestation(
                    "repo@sha256:img"
                ),
            }
        ],
        "ohttp_attestation": _ohttp_attestation(),
    }


def _completions_report():
    return {
        "signing_address": "md",
        "ohttp_attestation": _ohttp_attestation(),
        "info": {"tcb_info": {"app_compose": _build_app_compose({})}},
        "compose_manager_attestation": _compose_manager_attestation("repo@sha256:img"),
    }


_ALL_DIGESTS = {
    "nearaidev/cloud-api": "a" * 64,
    "nearaidev/cvm-ingress": "b" * 64,
    "nearaidev/dstack-vpc": "c" * 64,
    "nearaidev/dstack-vpc-client": "d" * 64,
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url,report,has_gateway_compose,expected_signing_keys",
    [
        (
            "https://cloud-api.near.ai/v1/",
            _cloud_api_report(_ALL_DIGESTS),
            True,
            ["gw", "md"],
        ),
        (
            "https://example.near.ai/v1/",
            _completions_report(),
            False,
            ["md"],
        ),
    ],
)
async def test_fetch_attestation(
    monkeypatch,
    reset_models,
    base_url,
    report,
    has_gateway_compose,
    expected_signing_keys,
):
    model_name = "near-test"
    reset_models[model_name] = {
        "address": base_url,
        "upstream_model": "test-model",
    }

    monkeypatch.setattr(near.external_service_settings, "near_api_key", "test-key")
    monkeypatch.setattr(near.httpx, "AsyncClient", lambda: _FakeAsyncClient(report))

    details = await near._fetch_attestation(model_name)

    assert (details.gateway_app_compose is not None) == has_gateway_compose
    assert details.signing_keys == expected_signing_keys
    assert details.ohttp_attestation.signing_key == "bb"
    # The attestation endpoint is authenticated: the bearer token must be sent.
    assert _FakeAsyncClient.last_headers == {"Authorization": "Bearer test-key"}


class _FakeS3Client:
    def __init__(self):
        self.missing: set[str] = set()
        self.head_keys: list[str] = []

    def head_object(self, Bucket, Key):
        self.head_keys.append(Key)
        if Key in self.missing:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")


@pytest.fixture
def fake_s3(monkeypatch):
    """Installs a fake S3 client. Mutate `.missing` to simulate 404s."""
    client = _FakeS3Client()
    monkeypatch.setattr(near.boto3, "client", lambda *a, **k: client)
    return client


def _details(signing_keys: list[str]) -> near.AttestationDetails:
    return near.AttestationDetails(
        gateway_app_compose=None,
        compose_actions=[
            near.ComposeAction(
                action=near.COMPOSE_MANAGER_STARTED_ACTION,
                image="repo@sha256:cm",
            )
        ],
        signing_keys=signing_keys,
        ohttp_attestation=near.OHTTPAttestation(
            key_config="aa", signing_key="bb", signature="cc"
        ),
    )


def test_verify_attestation_fails_on_bad_signature(monkeypatch):
    monkeypatch.setattr(near, "_verify_ohttp_signature", lambda _: False)
    details = _details(["bb"])
    assert near._verify_attestation(details, "m") is False


def test_verify_attestation_fails_when_ohttp_key_not_in_signing_keys(monkeypatch):
    monkeypatch.setattr(near, "_verify_ohttp_signature", lambda _: True)
    details = _details(["other"])
    assert near._verify_attestation(details, "m") is False


def test_verify_attestation_completions_success(monkeypatch, fake_s3):
    monkeypatch.setattr(near, "_verify_ohttp_signature", lambda _: True)

    details = _details(["bb"])

    assert near._verify_attestation(details, "m") is True
    assert f"{near.PRE_VERIFIED_PREFIX}/cm.json" in fake_s3.head_keys
    assert f"{near.SIGNING_KEYS_PREFIX}/bb.json" in fake_s3.head_keys


def test_verify_attestation_fails_when_pre_verified_missing(monkeypatch, fake_s3):
    monkeypatch.setattr(near, "_verify_ohttp_signature", lambda _: True)
    fake_s3.missing.add(f"{near.PRE_VERIFIED_PREFIX}/cm.json")

    details = _details(["bb"])

    assert near._verify_attestation(details, "m") is False


def test_verify_attestation_fails_when_signing_key_unregistered(monkeypatch, fake_s3):
    monkeypatch.setattr(near, "_verify_ohttp_signature", lambda _: True)
    fake_s3.missing.add(f"{near.SIGNING_KEYS_PREFIX}/bb.json")

    details = _details(["bb"])

    assert near._verify_attestation(details, "m") is False
