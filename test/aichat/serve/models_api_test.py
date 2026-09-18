from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aichat.serve.api_server import app
from aichat.serve.models_api import (
    _capabilities_for_model,
    _has_browser_metadata,
    _is_listable_browser_model,
    _model_visible_for_product,
    _parse_accept_language,
    get_browser_models,
)


def _model(**overrides):
    """Build a minimal valid model config dict."""
    base = {
        "backend": "vllm",
        "friendly_name": "Test Model",
        "maker": "TestMaker",
        "free": True,
        "capabilities": ["files"],
        "category": "chat",
        "description": {"en": "A test model."},
    }
    base.update(overrides)
    return base


def _mock_settings(models_dict):
    mock = MagicMock()
    mock.models = models_dict
    return mock


def _mock_external_settings(near_api_key=None):
    mock = MagicMock()
    mock.near_api_key = near_api_key
    return mock


def _mock_mcp_settings(deep_research_enabled=False):
    mock = MagicMock()
    mock.deep_research_enabled = deep_research_enabled
    return mock


@pytest.fixture(autouse=True)
def clear_model_cache():
    get_browser_models.cache_clear()
    yield
    get_browser_models.cache_clear()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestGetModels:
    def test_returns_200(self, client):
        models = {"basic": _model()}
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get("/v1/models", headers={"Accept-Language": "en"})
        assert response.status_code == 200

    def test_response_shape(self, client):
        models = {"basic": _model(friendly_name="Basic Model", maker="Acme")}
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get("/v1/models", headers={"Accept-Language": "en"})
        data = response.json()
        assert len(data) == 1
        entry = data[0]
        assert entry["key"] == "basic"
        assert entry["display_name"] == "Basic Model"
        assert "capabilities" in entry
        assert "vision_support" not in entry
        assert "supports_tools" not in entry
        assert "audio_support" not in entry
        assert "video_support" not in entry
        assert entry["is_suggested_model"] is False
        assert entry["is_near_model"] is False
        assert entry["options"]["type"] == "leo"
        assert entry["options"]["name"] == "basic"
        assert entry["options"]["display_maker"] == "Acme"
        assert "category" not in entry["options"]
        assert "chat" in entry["capabilities"]

    def test_capability_flags(self, client):
        models = {
            "multimodal": _model(
                capabilities=["tools", "vision", "audio", "video", "files"]
            )
        }
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get("/v1/models", headers={"Accept-Language": "en"})
        caps = response.json()[0]["capabilities"]
        assert "tools" in caps
        assert "vision" in caps
        assert "audio" in caps
        assert "video" in caps

    def test_free_model_access(self, client):
        models = {"free-model": _model(free=True)}
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get("/v1/models", headers={"Accept-Language": "en"})
        assert response.json()[0]["options"]["access"] == "basic_and_premium"

    def test_premium_model_access(self, client):
        models = {"premium-model": _model(free=False)}
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get("/v1/models", headers={"Accept-Language": "en"})
        assert response.json()[0]["options"]["access"] == "premium"

    def test_key_is_model_name(self, client):
        models = {"my-model-name": _model()}
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get("/v1/models", headers={"Accept-Language": "en"})
        assert response.json()[0]["key"] == "my-model-name"

    def test_is_suggested_model(self, client):
        models = {"suggested": _model(is_suggested_model=True)}
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get("/v1/models", headers={"Accept-Language": "en"})
        assert response.json()[0]["is_suggested_model"] is True

    def test_summary_category_included(self, client):
        models = {"summariser": _model(category="summary")}
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get("/v1/models", headers={"Accept-Language": "en"})
        assert len(response.json()) == 1

    def test_excludes_non_chat_summary_category(self, client):
        models = {
            "a-chat": _model(category="chat"),
            "b-other": _model(category="other"),
            "c-empty": _model(category=""),
        }
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get("/v1/models", headers={"Accept-Language": "en"})
        keys = [e["key"] for e in response.json()]
        assert keys == ["a-chat"]

    def test_excludes_deprecated(self, client):
        models = {
            "active": _model(),
            "old": _model(deprecated=True),
        }
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get("/v1/models", headers={"Accept-Language": "en"})
        keys = [e["key"] for e in response.json()]
        assert keys == ["active"]

    def test_excludes_non_llm_type(self, client):
        models = {
            "llm-model": _model(),
            "other-type": _model(type="other"),
        }
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get("/v1/models", headers={"Accept-Language": "en"})
        keys = [e["key"] for e in response.json()]
        assert keys == ["llm-model"]

    def test_excludes_near_without_api_key(self, client):
        models = {
            "regular": _model(),
            "near-model": _model(),
        }
        with (
            patch("aichat.serve.models_api.model_settings", _mock_settings(models)),
            patch(
                "aichat.serve.models_api.external_service_settings",
                _mock_external_settings(near_api_key=None),
            ),
        ):
            response = client.get("/v1/models", headers={"Accept-Language": "en"})
        keys = [e["key"] for e in response.json()]
        assert "near-model" not in keys
        assert "regular" in keys

    def test_includes_near_with_api_key(self, client):
        models = {
            "regular": _model(),
            "near-model": _model(),
        }
        with (
            patch("aichat.serve.models_api.model_settings", _mock_settings(models)),
            patch(
                "aichat.serve.models_api.external_service_settings",
                _mock_external_settings(near_api_key="secret"),
            ),
        ):
            response = client.get("/v1/models", headers={"Accept-Language": "en"})
        keys = [e["key"] for e in response.json()]
        assert "near-model" in keys

    def test_near_model_flag(self, client):
        models = {"near-foo": _model()}
        with (
            patch("aichat.serve.models_api.model_settings", _mock_settings(models)),
            patch(
                "aichat.serve.models_api.external_service_settings",
                _mock_external_settings(near_api_key="secret"),
            ),
        ):
            response = client.get("/v1/models", headers={"Accept-Language": "en"})
        assert response.json()[0]["is_near_model"] is True

    def test_non_near_model_flag(self, client):
        models = {"regular": _model()}
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get("/v1/models", headers={"Accept-Language": "en"})
        assert response.json()[0]["is_near_model"] is False

    def test_language_selection(self, client):
        models = {"m": _model(description={"en": "English desc", "fr": "French desc"})}
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get("/v1/models", headers={"Accept-Language": "fr"})
        assert response.json()[0]["options"]["description"] == "French desc"

    def test_language_fallback_to_english(self, client):
        models = {"m": _model(description={"en": "English desc"})}
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get("/v1/models", headers={"Accept-Language": "de"})
        assert response.json()[0]["options"]["description"] == "English desc"

    def test_empty_models(self, client):
        with patch("aichat.serve.models_api.model_settings", _mock_settings({})):
            response = client.get("/v1/models", headers={"Accept-Language": "en"})
        assert response.status_code == 200
        assert response.json() == []

    def test_model_type_defaults_to_llm(self, client):
        """Models without a 'type' field are treated as llm and included."""
        model = {k: v for k, v in _model().items() if k != "type"}
        models = {"no-type": model}
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get("/v1/models", headers={"Accept-Language": "en"})
        assert len(response.json()) == 1

    def test_context_window_free_model(self, client):
        models = {"m": _model(free=True, conversation_token_limit=8000)}
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            entry = client.get("/v1/models", headers={"Accept-Language": "en"}).json()[
                0
            ]
        assert entry["options"]["max_associated_content_length"] == 4000
        assert entry["options"]["long_conversation_warning_character_limit"] == 6400

    def test_context_window_premium_model(self, client):
        models = {"m": _model(free=False, conversation_token_limit_premium=32000)}
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            entry = client.get("/v1/models", headers={"Accept-Language": "en"}).json()[
                0
            ]
        assert entry["options"]["max_associated_content_length"] == 16000
        assert entry["options"]["long_conversation_warning_character_limit"] == 25600

    def test_context_window_missing_falls_back_to_one(self, client):
        models = {"m": _model()}  # no token limit fields
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            entry = client.get("/v1/models", headers={"Accept-Language": "en"}).json()[
                0
            ]
        assert entry["options"]["max_associated_content_length"] == 1
        assert entry["options"]["long_conversation_warning_character_limit"] == 1

    def test_no_accept_language_header_defaults_to_english(self, client):
        models = {"m": _model(description={"en": "English desc", "fr": "French desc"})}
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get("/v1/models")
        assert response.json()[0]["options"]["description"] == "English desc"

    def test_accept_language_with_region_subtag(self, client):
        models = {"m": _model(description={"en": "English desc", "fr": "French desc"})}
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get("/v1/models", headers={"Accept-Language": "fr-CH"})
        assert response.json()[0]["options"]["description"] == "French desc"

    def test_accept_language_q_values(self, client):
        models = {"m": _model(description={"en": "English desc", "fr": "French desc"})}
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get(
                "/v1/models",
                headers={"Accept-Language": "en;q=0.5, fr;q=0.9"},
            )
        assert response.json()[0]["options"]["description"] == "French desc"

    def test_no_brave_product_header_includes_tagged_models(self, client):
        models = {
            "default": _model(),
            "bot-model": _model(brave_products=["brave-bot"]),
        }
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get("/v1/models", headers={"Accept-Language": "en"})
        keys = sorted(e["key"] for e in response.json())
        assert keys == ["bot-model", "default"]

    def test_brave_product_header_includes_matching_tagged_models(self, client):
        models = {
            "default": _model(),
            "bot-model": _model(brave_products=["brave-bot"]),
            "other-product-model": _model(brave_products=["other-product"]),
        }
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get(
                "/v1/models",
                headers={
                    "Accept-Language": "en",
                    "Brave-Product": "brave-bot",
                },
            )
        keys = sorted(e["key"] for e in response.json())
        assert keys == ["bot-model"]

    def test_brave_product_header_excludes_non_matching_tagged_models(self, client):
        models = {
            "default": _model(),
            "bot-model": _model(brave_products=["brave-bot"]),
            "other-product-model": _model(brave_products=["other-product"]),
        }
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get(
                "/v1/models",
                headers={
                    "Accept-Language": "en",
                    "Brave-Product": "other-product",
                },
            )
        keys = sorted(e["key"] for e in response.json())
        assert keys == ["other-product-model"]

    def test_brave_product_header_excludes_untagged_models(self, client):
        models = {
            "default": _model(),
            "bot-model": _model(brave_products=["brave-bot"]),
        }
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get(
                "/v1/models",
                headers={
                    "Accept-Language": "en",
                    "Brave-Product": "brave-bot",
                },
            )
        keys = [e["key"] for e in response.json()]
        assert keys == ["bot-model"]

    def test_brave_product_header_is_case_insensitive(self, client):
        models = {"bot-model": _model(brave_products=["brave-bot"])}
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get(
                "/v1/models",
                headers={
                    "Accept-Language": "en",
                    "Brave-Product": "Brave-Bot",
                },
            )
        assert [e["key"] for e in response.json()] == ["bot-model"]

    def test_includes_ensemble_with_browser_metadata(self, client):
        models = {
            "basic": _model(),
            "auto-ensemble": _model(
                type="ensemble",
                models=[{"model": "basic", "weight": 1}],
                friendly_name="Automatic",
            ),
        }
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get("/v1/models", headers={"Accept-Language": "en"})
        keys = sorted(e["key"] for e in response.json())
        assert keys == ["auto-ensemble", "basic"]

    def test_excludes_ensemble_without_browser_metadata(self, client):
        models = {
            "basic": _model(),
            "auto-ensemble": {
                "type": "ensemble",
                "backend": "litellm",
                "models": [{"model": "basic", "weight": 1}],
                "free": True,
            },
        }
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get("/v1/models", headers={"Accept-Language": "en"})
        keys = [e["key"] for e in response.json()]
        assert keys == ["basic"]

    def test_includes_tagged_ensemble_for_matching_product(self, client):
        models = {
            "auto-brave-bot": _model(
                type="ensemble",
                models=[{"model": "basic", "weight": 1}],
                friendly_name="Automatic",
                brave_products=["brave-bot"],
            )
        }
        with patch("aichat.serve.models_api.model_settings", _mock_settings(models)):
            response = client.get(
                "/v1/models",
                headers={
                    "Accept-Language": "en",
                    "Brave-Product": "brave-bot",
                },
            )
        assert [e["key"] for e in response.json()] == ["auto-brave-bot"]


class TestListableBrowserModel:
    def test_llm_is_listable(self):
        assert _is_listable_browser_model(_model()) is True

    def test_ensemble_with_metadata_is_listable(self):
        assert (
            _is_listable_browser_model(
                _model(
                    type="ensemble",
                    models=[{"model": "basic", "weight": 1}],
                )
            )
            is True
        )

    def test_ensemble_without_metadata_is_not_listable(self):
        assert (
            _is_listable_browser_model(
                {
                    "type": "ensemble",
                    "backend": "litellm",
                    "models": [{"model": "basic", "weight": 1}],
                }
            )
            is False
        )

    def test_other_types_are_not_listable(self):
        assert _is_listable_browser_model(_model(type="tts")) is False


class TestHasBrowserMetadata:
    def test_complete_metadata(self):
        assert _has_browser_metadata(_model()) is True

    def test_missing_friendly_name(self):
        model = _model()
        del model["friendly_name"]
        assert _has_browser_metadata(model) is False

    def test_invalid_category(self):
        assert _has_browser_metadata(_model(category="other")) is False


class TestModelVisibleForProduct:
    def test_no_product_always_visible(self):
        assert _model_visible_for_product(_model(), None) is True
        assert (
            _model_visible_for_product(_model(brave_products=["brave-bot"]), None)
            is True
        )

    def test_untagged_model_hidden_when_product_set(self):
        assert _model_visible_for_product(_model(), "brave-bot") is False

    def test_matching_product_visible(self):
        assert (
            _model_visible_for_product(
                _model(brave_products=["brave-bot"]), "brave-bot"
            )
            is True
        )

    def test_non_matching_product_hidden(self):
        assert (
            _model_visible_for_product(
                _model(brave_products=["brave-bot"]), "other-product"
            )
            is False
        )

    def test_product_match_is_case_insensitive(self):
        assert (
            _model_visible_for_product(
                _model(brave_products=["brave-bot"]), "Brave-Bot"
            )
            is True
        )


class TestParseAcceptLanguage:
    def test_none_returns_english(self):
        assert _parse_accept_language(None) == "en"

    def test_simple_language(self):
        assert _parse_accept_language("fr") == "fr"

    def test_language_with_region(self):
        assert _parse_accept_language("fr-CH") == "fr"

    def test_multiple_languages_first_wins(self):
        assert _parse_accept_language("de, en") == "de"

    def test_q_values_respected(self):
        assert _parse_accept_language("en;q=0.5, fr;q=0.9") == "fr"

    def test_wildcard_ignored(self):
        assert _parse_accept_language("*, en") == "en"

    def test_empty_string_returns_english(self):
        assert _parse_accept_language("") == "en"

    def test_language_codes_lowercased(self):
        assert _parse_accept_language("FR") == "fr"


class TestCapabilitiesForModel:
    def test_chat_category_always_included(self):
        caps = _capabilities_for_model("m", _model(category="chat"))
        assert caps[0] == "chat"

    def test_summary_category_always_included(self):
        caps = _capabilities_for_model("m", _model(category="summary"))
        assert caps[0] == "summary"

    def test_no_capabilities_when_list_empty(self):
        caps = _capabilities_for_model("m", _model(capabilities=[]))
        assert "tools" not in caps
        assert "vision" not in caps
        assert "audio" not in caps
        assert "video" not in caps
        assert "files" not in caps

    def test_all_media_capabilities(self):
        caps = _capabilities_for_model(
            "m",
            _model(capabilities=["tools", "vision", "audio", "video", "files"]),
        )
        assert "tools" in caps
        assert "vision" in caps
        assert "audio" in caps
        assert "video" in caps
        assert "files" in caps

    def test_content_agent_capability(self):
        caps = _capabilities_for_model("m", _model(capabilities=["content_agent"]))
        assert "content_agent" in caps

    def test_content_agent_not_included_when_absent(self):
        caps = _capabilities_for_model("m", _model(capabilities=[]))
        assert "content_agent" not in caps

    def test_deep_research_included_when_both_enabled(self):
        with patch(
            "aichat.serve.models_api.mcp_settings",
            _mock_mcp_settings(deep_research_enabled=True),
        ):
            caps = _capabilities_for_model("m", _model(capabilities=["deep_research"]))
        assert "deep_research" in caps

    def test_deep_research_excluded_when_server_disabled(self):
        with patch(
            "aichat.serve.models_api.mcp_settings",
            _mock_mcp_settings(deep_research_enabled=False),
        ):
            caps = _capabilities_for_model("m", _model(capabilities=["deep_research"]))
        assert "deep_research" not in caps

    def test_deep_research_excluded_when_not_in_capabilities(self):
        with patch(
            "aichat.serve.models_api.mcp_settings",
            _mock_mcp_settings(deep_research_enabled=True),
        ):
            caps = _capabilities_for_model("m", _model(capabilities=[]))
        assert "deep_research" not in caps
