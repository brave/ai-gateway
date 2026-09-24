# BDD coverage for aichat/serve/services/near.py.

import asyncio
import json as json_mod
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from botocore.exceptions import ClientError
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve.services import near as near_mod
from aichat.serve.services.near import (
    AttestationDetails,
    ComposeAction,
    OHTTPAttestation,
    OHTTPConfig,
    _build_ohttp_config,
    _head_s3,
    _last_compose_manager_image,
    _verify_attestation,
    _verify_ohttp_signature,
    verify_and_get_ohttp_config,
)

FEATURE = "features/near.feature"
scenarios(FEATURE)


@pytest.fixture
def ctx():
    return {}


class FakeRedis:
    def __init__(self, connection_pool=None):
        self.store = {}
        self.expiries = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value
        self.expiries[key] = ex


class FakeS3:
    def __init__(self, ctx):
        self.ctx = ctx

    def head_object(self, Bucket=None, Key=None):
        self.ctx.setdefault("s3_heads", []).append(Key)
        behavior = self.ctx.get("s3_head_behavior", "exists")
        if behavior == "exists":
            return {}
        if behavior == "404":
            raise ClientError({"Error": {"Code": "404", "Message": ""}}, "HeadObject")
        raise ClientError(
            {"Error": {"Code": "AccessDenied", "Message": ""}}, "HeadObject"
        )


@given("the near harness")
def near_harness(ctx, monkeypatch):
    key = Ed25519PrivateKey.generate()
    ctx["signing_key_hex"] = (
        key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    )
    key_config_hex = "deadbeef"
    ctx["key_config_hex"] = key_config_hex
    ctx["signature_hex"] = key.sign(bytes.fromhex(key_config_hex)).hex()
    ctx["redis"] = FakeRedis()
    ctx["s3_head_behavior"] = "exists"
    monkeypatch.setattr(near_mod.boto3, "client", lambda *a, **k: FakeS3(ctx))
    monkeypatch.setattr(near_mod.aioredis, "Redis", lambda *a, **k: ctx["redis"])
    monkeypatch.setattr(near_mod, "get_pool", lambda: object())
    return ctx


@given(parsers.parse('a model "{model_name}" addressed at "{address}"'))
def near_model(ctx, model_name, address, monkeypatch):
    ms = SimpleNamespace(
        models={model_name: {"address": address, "upstream_model": "upstream-x"}}
    )
    monkeypatch.setattr(near_mod, "model_settings", ms)
    ctx["model_name"] = model_name


@given(parsers.parse("a key config hex of {key_hex}"))
def key_hex(ctx, key_hex):
    ctx["att_key_hex"] = key_hex


@when("the ohttp config is built from the attestation")
def build_config(ctx):
    attestation = OHTTPAttestation(
        key_config=ctx["att_key_hex"],
        signing_key="aa" * 32,
        signature="bb" * 32,
    )
    ctx["built"] = _build_ohttp_config(attestation, ctx.get("model_name", "near-model"))


@then(parsers.parse("the built config result is {built}"))
def built_config_assert(ctx, built):
    if built == "decodable":
        import base64

        assert ctx["built"] is not None
        assert ctx["built"].endpoint_url == "https://addr/v1/chat/completions"
        assert ctx["built"].upstream_model_name == "upstream-x"
        assert base64.b64decode(ctx["built"].key_config) == b"\xde\xad\xbe\xef"
    else:
        assert ctx["built"] is None


@given("a genuine ed25519 ohttp attestation")
def genuine_attestation(ctx):
    ctx["att"] = OHTTPAttestation(
        key_config=ctx["key_config_hex"],
        signing_key=ctx["signing_key_hex"],
        signature=ctx["signature_hex"],
    )


@when("the ohttp signature is verified")
def verify_signature(ctx):
    ctx["signature_ok"] = _verify_ohttp_signature(ctx["att"])


@then("the signature is valid")
def signature_valid(ctx):
    assert ctx["signature_ok"] is True


@when("the ohttp signature is verified after tampering")
def verify_signature_tampered(ctx):
    tampered = OHTTPAttestation(
        key_config=ctx["key_config_hex"],
        signing_key=ctx["signing_key_hex"],
        signature="ff" * 64,
    )
    ctx["signature_ok"] = _verify_ohttp_signature(tampered)


@then("the signature is invalid")
def signature_invalid(ctx):
    assert ctx["signature_ok"] is False


@when("the last compose manager image is extracted")
def extract_compose_image(ctx):
    ctx["compose_image"] = _last_compose_manager_image(ctx["compose_actions"])


@then('the bare digest "abc123" is returned')
def compose_digest_assert(ctx):
    assert ctx["compose_image"] == "abc123"


@then("no compose manager digest is returned")
def compose_digest_none_assert(ctx):
    assert ctx["compose_image"] is None


@given(parsers.parse("compose actions {compose_desc}"))
def compose_actions(ctx, compose_desc):
    if compose_desc == 'ending with "compose_manager_started"':
        ctx["compose_actions"] = [
            ComposeAction(action="other", image="x@sha256:0011"),
            ComposeAction(
                action="compose_manager_started",
                image="quay.io/nearaidev/cloud-api@sha256:abc123",
            ),
        ]
    elif compose_desc == "with a tagged but undigested image":
        ctx["compose_actions"] = [
            ComposeAction(
                action="compose_manager_started",
                image="quay.io/nearaidev/cloud-api:v1",
            ),
        ]
    else:
        raise ValueError(f"unknown compose actions fixture: {compose_desc!r}")


def _compose_app_compose(images):
    compose = {
        "services": {f"svc{i}": {"image": image} for i, image in enumerate(images)}
    }
    return json_mod.dumps({"docker_compose_file": json_mod.dumps(compose)})


TARGET_IMAGES = [
    "nearaidev/cloud-api",
    "nearaidev/cvm-ingress",
    "nearaidev/dstack-vpc",
    "nearaidev/dstack-vpc-client",
]


@given(parsers.parse("a gateway compose file {compose_desc}"))
def gateway_compose(ctx, compose_desc):
    if compose_desc == "with all four target images digested":
        images = [
            f"quay.io/{target}@sha256:digest{i}"
            for i, target in enumerate(TARGET_IMAGES)
        ]
        ctx["app_compose"] = _compose_app_compose(images)
        ctx["expected_digests"] = {
            target: f"digest{i}" for i, target in enumerate(TARGET_IMAGES)
        }
    elif compose_desc == "with only two target images":
        images = [
            "quay.io/nearaidev/cloud-api@sha256:digest0",
            "quay.io/nearaidev/cvm-ingress@sha256:digest1",
            "quay.io/other/unrelated@sha256:zzz",
        ]
        ctx["app_compose"] = _compose_app_compose(images)
    elif compose_desc == "without a registry prefix":
        images = [
            f"{target}@sha256:digest{i}" for i, target in enumerate(TARGET_IMAGES)
        ]
        ctx["app_compose"] = _compose_app_compose(images)
        ctx["expected_digests"] = {
            target: f"digest{i}" for i, target in enumerate(TARGET_IMAGES)
        }
    else:
        raise ValueError(f"unknown gateway compose fixture: {compose_desc!r}")


@when("the target digests are extracted")
def extract_target_digests(ctx):
    ctx["target_digests"] = near_mod._extract_target_digests(ctx["app_compose"])


@then("all four target digests are returned")
def target_digests_assert(ctx):
    assert ctx["target_digests"] == ctx["expected_digests"]


@then("no target digests are returned")
def no_target_digests_assert(ctx):
    assert ctx["target_digests"] == {}


@given(parsers.parse("an s3 head client that reports {presence}"))
def s3_head_presence(ctx, presence):
    ctx["s3_head_behavior"] = "exists" if presence == "the object" else "404"


@when(parsers.parse('a head object is performed for "{prefix}" "{value}"'))
def head_object(ctx, prefix, value):
    s3 = FakeS3(ctx)
    try:
        ctx["head"] = _head_s3(s3, prefix, value)
        ctx["head_error"] = None
    except Exception as e:
        ctx["head"] = None
        ctx["head_error"] = e


@then(parsers.parse("the head result is {result}"))
def head_result_assert(ctx, result):
    assert ctx["head"] == (result == "True")


@given("an s3 head client that raises AccessDenied")
def s3_head_denied(ctx):
    ctx["s3_head_behavior"] = "denied"


@then("the head raises ClientError")
def head_error_assert(ctx):
    assert isinstance(ctx["head_error"], ClientError)
    # The original AccessDenied is re-raised, not wrapped into another code.
    assert "AccessDenied" in str(ctx["head_error"])


def _attestation_for(ctx, *, tampered=False, registered=True, compose_state="missing"):
    signing_key_hex = ctx["signing_key_hex"]
    signature = ctx["signature_hex"] if not tampered else "ff" * 64
    actions = []
    gateway_app_compose = None
    if compose_state == "verified":
        actions = [
            ComposeAction(
                action="compose_manager_started",
                image="quay.io/nearaidev/cloud-api@sha256:abc123",
            )
        ]
    elif compose_state == "mismatched":
        # Only two of the four target images digested: _extract_target_digests
        # count mismatch yields no digests, so _verify_attestation fails on
        # the digest check BEFORE any S3 head object is performed.
        actions = [
            ComposeAction(
                action="compose_manager_started",
                image="quay.io/nearaidev/cloud-api@sha256:abc123",
            )
        ]
        gateway_app_compose = _compose_app_compose(
            [
                "quay.io/nearaidev/cloud-api@sha256:digest0",
                "quay.io/nearaidev/cvm-ingress@sha256:digest1",
                "quay.io/other/unrelated@sha256:zzz",
            ]
        )
    elif compose_state == "digested":
        actions = [
            ComposeAction(
                action="compose_manager_started",
                image="quay.io/nearaidev/cloud-api@sha256:abc123",
            )
        ]
        gateway_app_compose = _compose_app_compose(
            [
                f"quay.io/{target}@sha256:digest{i}"
                for i, target in enumerate(TARGET_IMAGES)
            ]
        )
    else:
        gateway_app_compose = None
    signing_keys = [signing_key_hex] if registered else ["de" * 32]
    return AttestationDetails(
        gateway_app_compose=gateway_app_compose,
        compose_actions=actions,
        signing_keys=signing_keys,
        ohttp_attestation=OHTTPAttestation(
            key_config=ctx["key_config_hex"],
            signing_key=signing_key_hex,
            signature=signature,
        ),
    )


@given(parsers.parse("a real ohttp attestation with signature {sig_state}"))
def attestation_sig(ctx, sig_state):
    ctx["att_sig_state"] = sig_state


@given(parsers.parse("signing keys {keys_state}"))
def attestation_keys(ctx, keys_state):
    ctx["att_keys_state"] = keys_state


@given(parsers.parse("a verified compose manager image {compose_state}"))
def attestation_compose(ctx, compose_state):
    ctx["att_compose_state"] = compose_state


@given(parsers.parse("the s3 heads return {s3_state}"))
def attestation_s3_state(ctx, s3_state):
    ctx["s3_head_behavior"] = "exists" if s3_state == "objects" else "404"


@when("the attestation is verified")
def verify_attestation(ctx):
    attestation = _attestation_for(
        ctx,
        tampered=ctx["att_sig_state"] == "tampered",
        registered=ctx["att_keys_state"] == "registered",
        compose_state=ctx["att_compose_state"],
    )
    ctx["verdict"] = _verify_attestation(
        attestation, ctx.get("model_name", "near-model")
    )


@then(parsers.parse("verification is {verdict}"))
def verdict_assert(ctx, verdict):
    assert ctx["verdict"] == (verdict == "successful")


@then(parsers.parse("the s3 heads checked {headed_keys}"))
def headed_keys_assert(ctx, headed_keys):
    """Each failing row must fail where intended, not for an accidental
    reason: assert exactly which S3 objects verification decided to head."""
    if headed_keys == "none":
        expected = []
    else:
        expected = []
        for fragment in headed_keys.split(" + "):
            if fragment == "pre-verified/abc123.json":
                expected.append(fragment)
            elif fragment == "signing-keys/{signing_key}.json":
                expected.append(f"signing-keys/{ctx['signing_key_hex']}.json")
            else:
                raise AssertionError(f"unknown headed-keys fragment: {fragment!r}")
    assert ctx.get("s3_heads", []) == expected


@when(parsers.parse('the ohttp config is verified and fetched for "{model_name}"'))
def verify_and_fetch(ctx, model_name, monkeypatch):
    fetch_mock = AsyncMock(return_value=ctx.get("fresh_att"))
    monkeypatch.setattr(near_mod, "_fetch_attestation", fetch_mock)
    ctx["fetch_mock"] = fetch_mock
    ctx["result"] = asyncio.run(verify_and_get_ohttp_config(model_name))


@given(parsers.parse('the redis cache holds the ohttp config for "{model_name}"'))
def redis_cached(ctx, model_name):
    cached = OHTTPConfig(
        key_config="cached-key",
        signing_key="cached-signing",
        signature="cached-sig",
        endpoint_url="https://addr/v1/chat/completions",
        upstream_model_name="upstream-x",
    )
    key = near_mod.NEAR_ATTEST_RESULT_CACHE_KEY.format(model_name=model_name)
    ctx["redis"].store[key] = cached.model_dump_json()
    ctx["cached_config"] = cached


@given(parsers.parse('a fresh attestation for "{model_name}" that {verify_state}'))
def fresh_attestation(ctx, model_name, verify_state):
    if verify_state not in ("fails verification", "verifies cleanly"):
        raise AssertionError(f"unknown verify state: {verify_state!r}")
    ctx["model_name"] = model_name
    ctx["fresh_att"] = _attestation_for(
        ctx,
        tampered=verify_state == "fails verification",
        registered=True,
        # compose_state="verified" means ONLY the compose-manager action is
        # present: gateway_app_compose stays None (verified in prod source,
        # near.py step 4 requires compose-manager even without app compose).
        compose_state="verified",
    )


@given(
    parsers.parse(
        'a fresh attestation for "{model_name}" with a fully digested compose'
    )
)
def fresh_attestation_digested(ctx, model_name):
    ctx["model_name"] = model_name
    ctx["fresh_att"] = _attestation_for(ctx, registered=True, compose_state="digested")


@then("the target digests were pre-verified in s3")
def digested_heads_assert(ctx):
    prefix = near_mod.PRE_VERIFIED_PREFIX
    heads = {str(k) for k in ctx.get("s3_heads", [])}
    expected = {f"{prefix}/digest{i}.json" for i in range(len(TARGET_IMAGES))}
    # The digested compose-manager head and the registered signing key are
    # also verified for this scenario.
    expected.add(f"{prefix}/abc123.json")
    expected.add(f"signing-keys/{ctx['signing_key_hex']}.json")
    assert heads == expected, (heads, expected)


@then("the cached config is returned without contacting the upstream")
def cached_config_assert(ctx):
    assert ctx["result"] == ctx["cached_config"]
    ctx["fetch_mock"].assert_not_awaited()


@then("the built config is returned and stored in redis")
def fresh_config_assert(ctx):
    result = ctx["result"]
    assert result is not None
    assert result.endpoint_url == "https://addr/v1/chat/completions"
    assert result.upstream_model_name == "upstream-x"
    key = near_mod.NEAR_ATTEST_RESULT_CACHE_KEY.format(
        model_name=ctx.get("model_name", "near-model")
    )
    assert ctx["redis"].store[key] == result.model_dump_json()
    # The cache entry must carry its TTL, not persist forever.
    assert ctx["redis"].expiries[key] == near_mod.NEAR_ATTEST_RESULT_TTL_SECONDS


@then("no ohttp config is returned")
def no_config_assert(ctx):
    assert ctx["result"] is None
    # A failed verification must not cache anything: the store stays empty.
    assert ctx["redis"].store == {}
