import base64
import json
import logging

import boto3
import httpx
import redis.asyncio as aioredis
import yaml
from botocore.exceptions import ClientError
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import BaseModel, Field

from aichat.serve.dependencies import get_pool
from aichat.serve.external_service_settings import external_service_settings
from aichat.serve.services.model_settings import model_settings

logger = logging.getLogger(__name__)


class OHTTPAttestation(BaseModel):
    key_config: str = Field(min_length=1)
    signing_key: str = Field(min_length=1)
    signature: str = Field(min_length=1)


class ComposeAction(BaseModel):
    action: str
    image: str | None = None


# Normalised result returned by _fetch_attestation regardless of endpoint format.
class AttestationDetails(BaseModel):
    gateway_app_compose: str | None = None
    compose_actions: list[ComposeAction]
    signing_keys: list[str]
    ohttp_attestation: OHTTPAttestation


class TCBInfo(BaseModel):
    app_compose: str


class AttestationInfo(BaseModel):
    tcb_info: TCBInfo


class ComposeManagerAttestation(BaseModel):
    actions: list[ComposeAction]


# Per-model attestation. Returned directly by the completions endpoint (old format,
# direct compute connection) and nested under model_attestations on the cloud-api endpoint.
class ModelAttestation(BaseModel):
    signing_address: str
    ohttp_attestation: OHTTPAttestation
    info: AttestationInfo
    compose_manager_attestation: ComposeManagerAttestation


# "CloudAPI" suffix (new format): nested response shape for the cloud-api attestation endpoint,
# used when connecting via load balancer.
class GatewayAttestationCloudAPI(BaseModel):
    signing_address: str
    info: AttestationInfo


class AttestationCloudAPI(BaseModel):
    gateway_attestation: GatewayAttestationCloudAPI
    model_attestations: list[ModelAttestation] = Field(min_length=1, max_length=1)
    ohttp_attestation: OHTTPAttestation


class OHTTPConfig(BaseModel):
    key_config: str
    signing_key: str
    signature: str
    endpoint_url: str
    upstream_model_name: str


PRE_VERIFIED_PREFIX = "pre-verified"
SIGNING_KEYS_PREFIX = "signing-keys"
COMPOSE_MANAGER_STARTED_ACTION = "compose_manager_started"
TARGET_IMAGES = (
    "nearaidev/cloud-api",
    "nearaidev/cvm-ingress",
    "nearaidev/dstack-vpc",
    "nearaidev/dstack-vpc-client",
)

NEAR_ATTEST_RESULT_CACHE_KEY = "NEAR_ATTEST_RESULT:{model_name}"
NEAR_ATTEST_RESULT_TTL_SECONDS = 3600


def _build_ohttp_config(
    attestation: OHTTPAttestation, model_name: str
) -> OHTTPConfig | None:
    model_config = model_settings.models[model_name]
    address = model_config["address"]
    try:
        return OHTTPConfig(
            key_config=base64.b64encode(bytes.fromhex(attestation.key_config)).decode(),
            signing_key=base64.b64encode(
                bytes.fromhex(attestation.signing_key)
            ).decode(),
            signature=base64.b64encode(bytes.fromhex(attestation.signature)).decode(),
            endpoint_url=f"{address}chat/completions",
            upstream_model_name=model_config["upstream_model"],
        )
    except Exception as e:
        logger.warning(f"OHTTP attestation hex decode failed: {e}")
        return None


def _head_s3(s3_client, prefix: str, value: str) -> bool:
    """
    Performs an S3 HEAD against `<prefix>/<value>.json`. Returns True if the
    object exists, False on 404, and re-raises any other ClientError.
    """
    key = f"{prefix}/{value}.json"
    try:
        s3_client.head_object(
            Bucket=external_service_settings.near_bucket_name, Key=key
        )
        return True
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == "404":
            return False
        raise


def _last_compose_manager_image(compose_actions: list[ComposeAction]) -> str | None:
    """
    Returns the bare sha256 hex of the image from the last
    `compose_manager_started` action, or None if not present.
    """
    image: str | None = None
    for action in compose_actions:
        if action.action == COMPOSE_MANAGER_STARTED_ACTION and action.image:
            image = action.image

    if image is None or "@sha256:" not in image:
        return None

    return image.split("@sha256:")[1]


def _verify_ohttp_signature(ohttp_attestation: OHTTPAttestation) -> bool:
    """
    Verifies the ed25519 signature over the OHTTP key_config using the
    OHTTP signing_key as the public key. Returns True on a valid signature.
    """
    try:
        public_key = Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(ohttp_attestation.signing_key)
        )
        public_key.verify(
            bytes.fromhex(ohttp_attestation.signature),
            bytes.fromhex(ohttp_attestation.key_config),
        )
        return True
    except (InvalidSignature, ValueError) as e:
        logger.warning(f"OHTTP signature verification failed: {e}")
        return False


def _verify_attestation(attestation: AttestationDetails, model_name: str) -> bool:
    """
    Performs the S3 HEAD checks required to consider the attestation verified.
    Returns True if all required objects are pre-verified / registered.
    """
    # Step 1: verify the ed25519 signature over the OHTTP key config.
    if not _verify_ohttp_signature(attestation.ohttp_attestation):
        logger.warning(f"Invalid OHTTP signature for {model_name}")
        return False

    # Step 2: ensure the OHTTP signing key is one of the verified signing keys.
    if attestation.ohttp_attestation.signing_key not in attestation.signing_keys:
        logger.warning(f"OHTTP signing key not among signing keys for {model_name}")
        return False

    s3_client = boto3.client(
        "s3", region_name=external_service_settings.near_bucket_region
    )

    # Step 3: confirm each gateway app_compose target image is pre-verified (cloud-api only).
    if attestation.gateway_app_compose is not None:
        target_digests = _extract_target_digests(attestation.gateway_app_compose)
        if not target_digests:
            logger.warning(f"No target digests for {model_name}")
            return False
        for digest in target_digests.values():
            if not _head_s3(s3_client, PRE_VERIFIED_PREFIX, digest):
                logger.warning(f"Target image not pre-verified for {model_name}")
                return False

    # Step 4: confirm the latest compose-manager image is pre-verified.
    compose_manager_image = _last_compose_manager_image(attestation.compose_actions)
    if compose_manager_image is None:
        logger.warning(f"No compose_manager_started image for {model_name}")
        return False
    if not _head_s3(s3_client, PRE_VERIFIED_PREFIX, compose_manager_image):
        logger.warning(f"compose-manager image not pre-verified for {model_name}")
        return False

    # Step 5: confirm every attested signing key is verified.
    for signing_key in attestation.signing_keys:
        if not _head_s3(s3_client, SIGNING_KEYS_PREFIX, signing_key):
            logger.warning(f"Signing key not registered for {model_name}")
            return False

    return True


def _extract_target_digests(app_compose: str | None) -> dict[str, str]:
    if not app_compose:
        return {}

    app_compose = json.loads(app_compose)

    docker_compose = app_compose.get("docker_compose_file")
    if not docker_compose:
        return {}

    compose_obj = yaml.safe_load(docker_compose) or {}
    services = compose_obj.get("services", {})
    digests: dict[str, str] = {}

    for service_config in services.values():
        image = service_config.get("image", "")
        if not image:
            continue

        reference = image.split("@")[0]
        last_colon = reference.rfind(":")
        last_slash = reference.rfind("/")
        if last_colon > last_slash:
            repository = reference[:last_colon]
        else:
            repository = reference

        matching_target = next(
            (
                target
                for target in TARGET_IMAGES
                if repository == target or repository.endswith(f"/{target}")
            ),
            None,
        )

        if not matching_target:
            continue

        digest: str | None = None
        if "@sha256:" in image:
            digest = image.split("@sha256:")[1]
        elif image.startswith("sha256:"):
            digest = image[len("sha256:") :]

        if digest:
            digests[matching_target] = digest

    if len(digests) != len(TARGET_IMAGES):
        logger.warning(
            f"Target digest count mismatch: expected {len(TARGET_IMAGES)}, got {len(digests)}"
        )
        return {}

    return digests


async def _fetch_attestation(model_name: str) -> AttestationDetails:
    """
    Fetch the full attestation report from the model's upstream.
    Raises on any failure.
    """
    model_config = model_settings.models[model_name]
    address = model_config["address"]
    upstream_model = model_config["upstream_model"]
    is_cloud_api = "cloud-api" in address
    report_url = f"{address}attestation/report?signing_algo=ed25519"
    if is_cloud_api:
        report_url += f"&model={upstream_model}"

    headers = {"Authorization": f"Bearer {external_service_settings.near_api_key}"}
    async with httpx.AsyncClient() as client:
        response = await client.get(report_url, headers=headers)
        response.raise_for_status()
        data = response.json()

    if is_cloud_api:
        parsed = AttestationCloudAPI.model_validate(data)
        model_attestation = parsed.model_attestations[0]
        gateway = parsed.gateway_attestation
        return AttestationDetails(
            gateway_app_compose=gateway.info.tcb_info.app_compose,
            compose_actions=model_attestation.compose_manager_attestation.actions,
            signing_keys=[gateway.signing_address, model_attestation.signing_address],
            ohttp_attestation=parsed.ohttp_attestation,
        )
    else:
        parsed = ModelAttestation.model_validate(data)
        return AttestationDetails(
            gateway_app_compose=None,
            compose_actions=parsed.compose_manager_attestation.actions,
            signing_keys=[parsed.signing_address],
            ohttp_attestation=parsed.ohttp_attestation,
        )


async def verify_and_get_ohttp_config(model_name: str) -> OHTTPConfig | None:
    """
    Verifies the NEAR TEE attestation for a model and returns the OHTTP config
    on success, or None on failure. Results are cached in Redis (1h TTL).
    """
    redis_client = aioredis.Redis(connection_pool=get_pool())
    cache_key = NEAR_ATTEST_RESULT_CACHE_KEY.format(model_name=model_name)

    cached = await redis_client.get(cache_key)
    if cached is not None:
        return OHTTPConfig.model_validate_json(cached)

    attestation = await _fetch_attestation(model_name)

    if not _verify_attestation(attestation, model_name):
        return None

    config = _build_ohttp_config(attestation.ohttp_attestation, model_name)
    if config is None:
        return None

    await redis_client.set(
        cache_key, config.model_dump_json(), ex=NEAR_ATTEST_RESULT_TTL_SECONDS
    )

    return config
