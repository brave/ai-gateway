import base64
import logging
import uuid

import fastapi
from aiobotocore.session import get_session
from botocore.exceptions import ClientError
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from aichat.protocol.open_ai_protocol import ErrorCode
from aichat.serve.auth import check_x_brave_key
from aichat.serve.common_api import create_error_response
from aichat.serve.external_service_settings import external_service_settings
from aichat.serve.rate_limiting import rate_limit_route
from aichat.serve.share_metrics import (
    SHARE_ACCESSED,
    SHARE_CREATED,
    SHARE_DELETE_NOT_FOUND,
    SHARE_DELETED,
    SHARE_NOT_FOUND,
)

logger = logging.getLogger(__name__)

v1_router = APIRouter()

_S3_SESSION = get_session()


def _cors_headers() -> dict:
    origin = external_service_settings.share_viewer_origin
    if not origin:
        return {}
    return {"Access-Control-Allow-Origin": origin}


def _with_cors(response: JSONResponse) -> JSONResponse:
    """Attach the viewer CORS headers to a response.

    Error responses need these just as much as successful ones: a cross-origin
    response without them is blocked by the browser before the page can read it,
    so the viewer sees an opaque network failure and cannot tell "this share was
    deleted" (404) from "the network is down".
    """
    response.headers.update(_cors_headers())
    return response


@v1_router.options("/share/{share_id}")
async def options_share(share_id: str):  # pylint: disable=unused-argument
    headers = {
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        **_cors_headers(),
    }
    return JSONResponse(content={}, headers=headers)


@v1_router.post("/share")
@rate_limit_route(
    config_key="share_create",
    route_path="/share",
    error_message="Daily rate limit exceeded for share endpoint",
)
async def create_share(
    request: Request,
    is_valid_x_brave_key: bool = fastapi.Depends(check_x_brave_key),
) -> JSONResponse:
    if not is_valid_x_brave_key:
        return create_error_response(ErrorCode.INVALID_AUTH_KEY, "Invalid services key")

    try:
        body = await request.json()
    except Exception as e:
        return create_error_response(ErrorCode.BAD_REQUEST_ERROR, f"Invalid JSON: {e}")

    ciphertext = body.get("ciphertext")
    if not ciphertext:
        return create_error_response(
            ErrorCode.BAD_REQUEST_ERROR, "ciphertext is required"
        )

    try:
        decoded = base64.b64decode(ciphertext, validate=True)
    except Exception:
        return create_error_response(
            ErrorCode.BAD_REQUEST_ERROR, "ciphertext is not valid base64"
        )

    if len(decoded) > external_service_settings.share_max_ciphertext_bytes:
        return create_error_response(
            ErrorCode.BAD_REQUEST_ERROR, "Ciphertext exceeds maximum size limit"
        )

    share_id = str(uuid.uuid4())
    deletion_id = str(uuid.uuid4())

    try:
        async with _S3_SESSION.create_client("s3") as s3:
            await s3.put_object(
                Bucket=external_service_settings.share_s3_bucket,
                Key=f"deletions/{deletion_id}",
                Body=share_id.encode("utf-8"),
                ContentType="text/plain",
                ChecksumAlgorithm="CRC32",
            )
            await s3.put_object(
                Bucket=external_service_settings.share_s3_bucket,
                Key=f"shares/{share_id}",
                Body=ciphertext.encode("utf-8"),
                ContentType="text/plain",
                ChecksumAlgorithm="CRC32",
            )
    except Exception:
        logger.exception("S3 write failed while creating share")
        return create_error_response(ErrorCode.INTERNAL_ERROR, "Failed to store share")

    SHARE_CREATED.inc()
    return JSONResponse(
        content={"share_id": share_id, "deletion_id": deletion_id}, status_code=201
    )


@v1_router.get("/share/{share_id}")
async def get_share(share_id: str) -> JSONResponse:
    try:
        async with _S3_SESSION.create_client("s3") as s3:
            obj = await s3.get_object(
                Bucket=external_service_settings.share_s3_bucket,
                Key=f"shares/{share_id}",
            )
            body = await obj["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            SHARE_NOT_FOUND.inc()
            return _with_cors(
                create_error_response(ErrorCode.MODEL_NOT_FOUND, "Share not found")
            )
        logger.exception("S3 read failed while retrieving share")
        return _with_cors(
            create_error_response(ErrorCode.INTERNAL_ERROR, "Failed to retrieve share")
        )
    except Exception:
        logger.exception("S3 read failed while retrieving share")
        return _with_cors(
            create_error_response(ErrorCode.INTERNAL_ERROR, "Failed to retrieve share")
        )

    SHARE_ACCESSED.inc()
    return _with_cors(JSONResponse(content={"ciphertext": body.decode("utf-8")}))


@v1_router.post("/share/delete")
@rate_limit_route(
    config_key="share_delete",
    route_path="/share/delete",
    error_message="Daily rate limit exceeded for share delete endpoint",
)
async def delete_share(
    request: Request,
    is_valid_x_brave_key: bool = fastapi.Depends(check_x_brave_key),
) -> JSONResponse:
    if not is_valid_x_brave_key:
        return create_error_response(ErrorCode.INVALID_AUTH_KEY, "Invalid services key")

    try:
        body = await request.json()
    except Exception as e:
        return create_error_response(ErrorCode.BAD_REQUEST_ERROR, f"Invalid JSON: {e}")

    deletion_id = body.get("deletion_id")
    if not deletion_id:
        return create_error_response(
            ErrorCode.BAD_REQUEST_ERROR, "deletion_id is required"
        )

    try:
        async with _S3_SESSION.create_client("s3") as s3:
            try:
                sidecar = await s3.get_object(
                    Bucket=external_service_settings.share_s3_bucket,
                    Key=f"deletions/{deletion_id}",
                )
                share_id = (await sidecar["Body"].read()).decode("utf-8")
            except ClientError as e:
                if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                    SHARE_DELETE_NOT_FOUND.inc()
                    return create_error_response(
                        ErrorCode.MODEL_NOT_FOUND, "Share not found"
                    )
                raise

            await s3.delete_object(
                Bucket=external_service_settings.share_s3_bucket,
                Key=f"shares/{share_id}",
            )

            try:
                await s3.delete_object(
                    Bucket=external_service_settings.share_s3_bucket,
                    Key=f"deletions/{deletion_id}",
                )
            except Exception:
                logger.exception(
                    "Failed to clean up deletion sidecar after deleting share"
                )
    except Exception:
        logger.exception("S3 delete failed while deleting share")
        return create_error_response(ErrorCode.INTERNAL_ERROR, "Failed to delete share")

    SHARE_DELETED.inc()
    return JSONResponse(content={}, status_code=200)
