import functools
import logging

from fastapi import APIRouter, Header
from pydantic import BaseModel

from aichat.serve.external_service_settings import external_service_settings
from aichat.serve.services.mcp.mcp_settings import mcp_settings
from aichat.serve.services.model_settings import model_settings

logger = logging.getLogger(__name__)
v1_router = APIRouter()


class LLMModelOptions(BaseModel):
    type: str
    name: str
    display_maker: str
    description: str
    access: str
    max_associated_content_length: int
    long_conversation_warning_character_limit: int


class LLMModel(BaseModel):
    key: str | None
    display_name: str
    capabilities: list[str]
    is_suggested_model: bool
    is_near_model: bool
    options: LLMModelOptions


def _capabilities_for_model(name: str, model: dict) -> list[str]:
    caps = [model["category"]]
    for cap in model.get("capabilities", []):
        if cap == "deep_research" and not mcp_settings.deep_research_enabled:
            continue
        caps.append(cap)
    return caps


def _model_visible_for_product(model: dict, product: str | None) -> bool:
    if product is None:
        return True

    products = model.get("brave_products")
    if not products:
        return False

    return product.lower() in {p.lower() for p in products}


def _has_browser_metadata(model: dict) -> bool:
    if model.get("category") not in ("chat", "summary"):
        return False
    return all(field in model for field in ("friendly_name", "maker", "description"))


def _is_listable_browser_model(model: dict) -> bool:
    model_type = model.get("type", "llm")
    if model_type == "llm":
        return True
    if model_type == "ensemble":
        return _has_browser_metadata(model)
    return False


@functools.lru_cache(maxsize=64)
def get_browser_models(lang: str, product: str | None = None) -> list[LLMModel]:
    models = []

    for name in model_settings.models:
        model = model_settings.models[name]

        if not _is_listable_browser_model(model):
            continue

        if model.get("category") not in ("chat", "summary"):
            continue

        if model.get("deprecated") is True:
            continue

        if name.startswith("near-") and not external_service_settings.near_api_key:
            continue

        if not _model_visible_for_product(model, product):
            continue

        descriptions = model["description"]
        description = descriptions[lang] if lang in descriptions else descriptions["en"]
        context_window = (
            model.get("conversation_token_limit")
            if model["free"]
            else model.get("conversation_token_limit_premium")
        )
        max_associated_content_length = context_window // 2 if context_window else 1
        long_conversation_warning_character_limit = (
            int(context_window * 0.8) if context_window else 1
        )
        modelOptions = LLMModelOptions(
            type="leo",
            name=name,
            display_maker=model["maker"],
            description=description,
            access="basic_and_premium" if model["free"] else "premium",
            max_associated_content_length=max_associated_content_length,
            long_conversation_warning_character_limit=long_conversation_warning_character_limit,
        )
        models.append(
            LLMModel(
                key=name,
                display_name=model["friendly_name"],
                capabilities=_capabilities_for_model(name, model),
                is_suggested_model=model.get("is_suggested_model", False),
                is_near_model=name.startswith("near-"),
                options=modelOptions,
            )
        )

    return models


def _parse_accept_language(accept_language: str | None) -> str:
    """Extract the highest-priority 2-letter language code from an Accept-Language header."""
    if not accept_language:
        return "en"

    langs = []
    for part in accept_language.split(","):
        part = part.strip()
        if ";q=" in part:
            lang_tag, q_str = part.rsplit(";q=", 1)
            try:
                q = float(q_str.strip())
            except ValueError:
                q = 1.0
        else:
            lang_tag = part
            q = 1.0
        lang_code = lang_tag.strip().split("-")[0].lower()
        if lang_code and lang_code != "*":
            langs.append((q, lang_code))

    if not langs:
        return "en"

    langs.sort(key=lambda x: x[0], reverse=True)
    return langs[0][1]


@v1_router.get("/models")
async def get_models(
    accept_language: str | None = Header(None),
    brave_product: str | None = Header(None, alias="Brave-Product"),
):
    lang = _parse_accept_language(accept_language)
    product = brave_product.strip().lower() if brave_product else None
    return get_browser_models(lang, product)
