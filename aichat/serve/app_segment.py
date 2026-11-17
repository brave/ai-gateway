from fastapi import Request

APP_DEFAULT = "default"
APP_BRAVEBOT = "bravebot"
APP_API_KEY = "api_key"


def resolve_app(brave_product: str | None, has_api_key: bool) -> str:
    if brave_product and brave_product.strip().lower() == APP_BRAVEBOT:
        return APP_BRAVEBOT
    if has_api_key:
        return APP_API_KEY
    return APP_DEFAULT


def get_app(request: Request | None) -> str:
    if request is None:
        return APP_DEFAULT
    return getattr(request.state, "app", APP_DEFAULT)


def app_dependency(request: Request) -> str:
    return get_app(request)
