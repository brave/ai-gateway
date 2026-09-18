from json import loads

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from aichat.serve.rate_limiting import rate_limit_route
from aichat.serve.services.search_settings import search_settings

router = APIRouter()


def get_cors_headers(request: Request):
    """
    Get the CORS headers for the request. If the origin
    is in the ALLOWED_CORS_ORIGINS list, return the CORS
    headers. Otherwise, return None.
    """
    origin = request.headers.get("Origin")
    allowed_origins = (
        loads(search_settings.search_api_allowed_cors_origins.strip().strip("'\""))
        if search_settings.search_api_allowed_cors_origins
        else []
    )
    if origin in allowed_origins:
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    return None


@router.options("/rhfetch/{path:path}")
async def options_rhfetch(request: Request):
    return JSONResponse(
        content={},
        headers=get_cors_headers(request),
    )


@router.get("/rhfetch/{path:path}")
@rate_limit_route(
    config_key="rhfetch",
    route_path="/rhfetch",
    error_message="Daily rate limit exceeded for rhfetch endpoint",
)
async def get_rhfetch(request: Request, path: str):
    url = f"{search_settings.brave_search_api_url}/res/v1/web/rich/fetch/{path}?{request.query_params}"
    try:
        httpx_client = request.state.httpx_client
        response = await httpx_client.get(
            url,
            headers={"x-subscription-token": search_settings.brave_search_rh_api_key},
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, detail="Failed to fetch rich results"
            )

        return JSONResponse(content=response.json(), headers=get_cors_headers(request))
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=500, detail="Failed to fetch rich results"
        ) from e
