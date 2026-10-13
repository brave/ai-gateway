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


_SERVING_SHA256 = "e" * 64


def _compose_manager_attestation(image: str) -> dict:
    return {
        "actions": [
            {"action": near.COMPOSE_MANAGER_STARTED_ACTION, "image": image},
            {"action": near.COMPOSE_UP_ACTION, "file_sha256": _SERVING_SHA256},
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
                "model_name": "org/model",
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
        "model_name": "org/model",
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
    assert details.model_name == "org/model"
    assert details.ohttp_attestation.signing_key == "bb"
    # The attestation endpoint is authenticated: the bearer token must be sent.
    assert _FakeAsyncClient.last_headers == {"Authorization": "Bearer test-key"}


class _FakeS3Client:
    def __init__(self):
        self.missing: set[str] = set()
        self.missing_prefixes: set[str] = set()
        self.head_keys: list[str] = []

    def head_object(self, Bucket, Key):
        self.head_keys.append(Key)
        if Key in self.missing or any(Key.startswith(p) for p in self.missing_prefixes):
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")


@pytest.fixture
def fake_s3(monkeypatch, reset_models):
    """Installs a fake S3 client. Mutate `.missing` to simulate 404s."""
    client = _FakeS3Client()
    monkeypatch.setattr(near.boto3, "client", lambda *a, **k: client)
    reset_models["m"] = {"attested_model": "org/model"}
    return client


def _up(sha256: str | None = _SERVING_SHA256, ts: str = "t1") -> near.ComposeAction:
    return near.ComposeAction(
        action=near.COMPOSE_UP_ACTION,
        commit="c",
        file="prod/m.yaml",
        file_sha256=sha256,
        services=["r1"],
        timestamp=ts,
    )


def _details(
    signing_keys: list[str], *ups: near.ComposeAction, model_name: str = "org/model"
) -> near.AttestationDetails:
    return near.AttestationDetails(
        model_name=model_name,
        gateway_app_compose=None,
        compose_actions=[
            near.ComposeAction(
                action=near.COMPOSE_MANAGER_STARTED_ACTION,
                image="repo@sha256:cm",
            ),
            *(ups or [_up()]),
        ],
        signing_keys=signing_keys,
        ohttp_attestation=near.OHTTPAttestation(
            key_config="aa", signing_key="bb", signature="cc"
        ),
    )


def _log_key(details: near.AttestationDetails, k: int) -> str:
    head = near._compose_log_chain(details.compose_actions)[k - 1]
    return f"{near.SERVING_LOG_PREFIX}/org/model/{head}.json"


def _compose_key(sha256: str) -> str:
    return f"{near.SERVING_COMPOSE_PREFIX}/org/model/{sha256}.json"


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
    assert _log_key(details, 1) in fake_s3.head_keys


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


@pytest.mark.parametrize("configured", [None, "org/other"])
def test_verify_attestation_fails_on_unexpected_model(
    monkeypatch, fake_s3, reset_models, configured
):
    # NEAR may route our upstream_model alias to another model.
    monkeypatch.setattr(near, "_verify_ohttp_signature", lambda _: True)
    reset_models["m"] = {"attested_model": configured} if configured else {}

    assert near._verify_attestation(_details(["bb"]), "m") is False


def test_verify_attestation_fails_without_verified_history(monkeypatch, fake_s3):
    monkeypatch.setattr(near, "_verify_ohttp_signature", lambda _: True)
    fake_s3.missing_prefixes.add(f"{near.SERVING_LOG_PREFIX}/")

    assert near._verify_attestation(_details(["bb"]), "m") is False


def test_verify_attestation_checks_ups_after_the_verified_point(monkeypatch, fake_s3):
    # The verifier saw only the first up. The second up, on another service,
    # must load an approved compose file.
    monkeypatch.setattr(near, "_verify_ohttp_signature", lambda _: True)
    details = _details(["bb"], _up(ts="t1"), _up("f" * 64, ts="t2"))
    fake_s3.missing.add(_log_key(details, 2))

    assert near._verify_attestation(details, "m") is True
    assert _compose_key("f" * 64) in fake_s3.head_keys

    fake_s3.missing.add(_compose_key("f" * 64))
    assert near._verify_attestation(details, "m") is False


@pytest.mark.parametrize("sha256", [None, "not-a-sha"])
def test_verify_attestation_fails_on_unusable_later_up(monkeypatch, fake_s3, sha256):
    monkeypatch.setattr(near, "_verify_ohttp_signature", lambda _: True)
    details = _details(["bb"], _up(ts="t1"), _up(sha256, ts="t2"))
    fake_s3.missing.add(_log_key(details, 2))

    assert near._verify_attestation(details, "m") is False


def test_verify_attestation_looks_back_a_bounded_number_of_actions(
    monkeypatch, fake_s3
):
    monkeypatch.setattr(near, "_verify_ohttp_signature", lambda _: True)
    n = near.SERVING_LOG_LOOKBACK + 1
    details = _details(["bb"], *[_up(ts=str(i)) for i in range(n)])
    for k in range(2, n + 1):
        fake_s3.missing.add(_log_key(details, k))

    assert near._verify_attestation(details, "m") is False
    assert _log_key(details, 1) not in fake_s3.head_keys


def test_compose_log_chain_matches_the_verifier():
    # Golden values from near-tee-verifier's compose_log_chain on the same
    # actions; the relay only works while the two agree.
    actions = [
        near.ComposeAction(
            action=near.COMPOSE_UP_ACTION,
            commit="c1",
            file="prod/a.yaml",
            file_sha256="a" * 64,
            services=["r1", "r2"],
            timestamp="2026-10-01T00:00:00Z",
        ),
        near.ComposeAction(action="docker_clean"),
        near.ComposeAction(
            action="compose_down",
            commit="c1",
            file="prod/a.yaml",
            timestamp="2026-10-02T00:00:00Z",
        ),
    ]
    assert near._compose_log_chain(actions) == [
        "6ac6eac7f896fa9bb0af31616e7532654d2ded2b02229a393950babffb42aa99",
        "c52c4f93d02081976630c4e96763ba82ff565fd416c111ef0da21e108e749f86",
    ]
