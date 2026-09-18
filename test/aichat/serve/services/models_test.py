from dataclasses import asdict
from unittest.mock import patch

import pytest

from aichat.serve.services.models import (
    ModelConfig,
    get_model_config,
)


class TestModelConfig:
    @pytest.fixture
    def minimal_model_config(self):
        """Create a ModelConfig with minimal required fields"""
        return ModelConfig(
            model_id="test-model",
            upstream_model="test-upstream",
            backend="litellm",
            api_base="https://api.test.com",
            api_key="test-key",
            inference_profile=None,
            system_prompt_support=True,
            prompt_caching_support=None,
            prompt_caching_enabled=None,
            tool_support=True,
            image_support=False,
            audio_support=False,
            video_support=False,
            file_support=True,
            friendly_name="Test Model",
            maker="Test Maker",
            max_tokens=4096,
            max_tokens_premium=8192,
            conversation_token_limit=1000,
            conversation_token_limit_premium=2000,
            free=True,
            key="test-key-id",
            rate_limit=100,
            rate_limit_interval_seconds=60,
            max_pages=10,
            max_pages_premium=20,
            extra_body=None,
        )

    @pytest.fixture
    def full_model_config(self):
        """Create a ModelConfig with all fields populated"""
        return ModelConfig(
            model_id="premium-model",
            upstream_model="premium-upstream",
            backend="bedrock",
            api_base="https://bedrock.aws.com",
            api_key="bedrock-key",
            inference_profile="test-profile",
            system_prompt_support=True,
            prompt_caching_support=True,
            prompt_caching_enabled=True,
            tool_support=True,
            image_support=True,
            audio_support=True,
            video_support=True,
            file_support=True,
            friendly_name="Premium Model",
            maker="AWS",
            max_tokens=8192,
            max_tokens_premium=16384,
            conversation_token_limit=2000,
            conversation_token_limit_premium=4000,
            free=False,
            key="premium-key",
            rate_limit=50,
            rate_limit_interval_seconds=30,
            max_pages=20,
            max_pages_premium=40,
            extra_body={"temperature": 0.7, "top_p": 0.9},
        )

    @pytest.fixture
    def bedrock_model_config(self):
        """Create a ModelConfig for Bedrock backend"""
        return ModelConfig(
            model_id="bedrock-claude",
            upstream_model="anthropic.claude-3-sonnet",
            backend="bedrock",
            api_base=None,
            api_key=None,
            inference_profile="production-profile",
            system_prompt_support=True,
            prompt_caching_support=True,
            prompt_caching_enabled=True,
            tool_support=True,
            image_support=True,
            audio_support=False,
            video_support=False,
            file_support=True,
            friendly_name="Claude 3 Sonnet",
            maker="Anthropic",
            max_tokens=4096,
            max_tokens_premium=8192,
            conversation_token_limit=1500,
            conversation_token_limit_premium=3000,
            free=True,
            key="claude-key",
            rate_limit=20,
            rate_limit_interval_seconds=60,
            max_pages=5,
            max_pages_premium=15,
            extra_body={"max_tokens": 4096},
        )

    def test_model_config_initialization_minimal(self, minimal_model_config):
        """Test ModelConfig initialization with minimal fields"""
        config = minimal_model_config

        assert config.model_id == "test-model"
        assert config.upstream_model == "test-upstream"
        assert config.backend == "litellm"
        assert config.api_base == "https://api.test.com"
        assert config.api_key == "test-key"
        assert config.inference_profile is None
        assert config.system_prompt_support is True
        assert config.prompt_caching_support is None
        assert config.tool_support is True
        assert config.image_support is False
        assert config.free is True

    def test_model_config_initialization_full(self, full_model_config):
        """Test ModelConfig initialization with all fields"""
        config = full_model_config

        assert config.model_id == "premium-model"
        assert config.backend == "bedrock"
        assert config.inference_profile == "test-profile"
        assert config.system_prompt_support is True
        assert config.prompt_caching_support is True
        assert config.image_support is True
        assert config.audio_support is True
        assert config.video_support is True
        assert config.free is False
        assert config.extra_body == {"temperature": 0.7, "top_p": 0.9}

    def test_model_config_bedrock_specific(self, bedrock_model_config):
        """Test ModelConfig with Bedrock-specific fields"""
        config = bedrock_model_config

        assert config.backend == "bedrock"
        assert config.inference_profile == "production-profile"
        assert config.api_base is None
        assert config.api_key is None
        assert config.upstream_model == "anthropic.claude-3-sonnet"
        assert config.system_prompt_support is True
        assert config.prompt_caching_support is True

    def test_to_dict_method(self, minimal_model_config):
        """Test to_dict method converts ModelConfig to dictionary"""
        config = minimal_model_config
        result = config.to_dict()

        # Should be a dictionary
        assert isinstance(result, dict)

        # Should contain all fields
        assert result["model_id"] == "test-model"
        assert result["upstream_model"] == "test-upstream"
        assert result["backend"] == "litellm"
        assert result["api_base"] == "https://api.test.com"
        assert result["tool_support"] is True
        assert result["image_support"] is False
        assert result["free"] is True

        # Should be equivalent to asdict()
        expected = asdict(config)
        assert result == expected

    def test_to_dict_method_with_none_values(self, bedrock_model_config):
        """Test to_dict method handles None values correctly"""
        config = bedrock_model_config
        result = config.to_dict()

        assert result["api_base"] is None
        assert result["api_key"] is None
        assert result["inference_profile"] == "production-profile"

    def test_to_dict_method_with_complex_extra_body(self, full_model_config):
        """Test to_dict method with complex extra_body"""
        config = full_model_config
        result = config.to_dict()

        assert result["extra_body"] == {"temperature": 0.7, "top_p": 0.9}

    def test_get_method_existing_field(self, minimal_model_config):
        """Test get method returns existing field values"""
        config = minimal_model_config

        assert config.get("model_id") == "test-model"
        assert config.get("backend") == "litellm"
        assert config.get("tool_support") is True
        assert config.get("image_support") is False
        assert config.get("api_base") == "https://api.test.com"

    def test_get_method_nonexistent_field_default_none(self, minimal_model_config):
        """Test get method returns None for nonexistent fields"""
        config = minimal_model_config

        assert config.get("nonexistent_field") is None
        assert config.get("another_missing_field") is None

    def test_get_method_nonexistent_field_custom_default(self, minimal_model_config):
        """Test get method returns custom default for nonexistent fields"""
        config = minimal_model_config

        assert config.get("nonexistent_field", "default_value") == "default_value"
        assert config.get("missing_number", 42) == 42
        assert config.get("missing_bool", False) is False
        assert config.get("missing_list", []) == []

    def test_get_method_with_none_field_value(self, bedrock_model_config):
        """Test get method with fields that have None values"""
        config = bedrock_model_config

        # These fields are explicitly None, should return None not default
        assert config.get("api_base") is None
        assert config.get("api_key") is None
        assert config.get("api_base", "default") is None
        assert config.get("api_key", "default") is None

    def test_get_method_vs_direct_access(self, minimal_model_config):
        """Test get method returns same values as direct attribute access"""
        config = minimal_model_config

        assert config.get("model_id") == config.model_id
        assert config.get("backend") == config.backend
        assert config.get("tool_support") == config.tool_support
        assert config.get("max_tokens") == config.max_tokens

    def test_prompt_caching_support_in_to_dict(self, full_model_config):
        """Test that prompt_caching_support appears in to_dict output"""
        result = full_model_config.to_dict()
        assert "prompt_caching_support" in result
        assert result["prompt_caching_support"] is True

    def test_prompt_caching_support_get_method(self, bedrock_model_config):
        """Test accessing prompt_caching_support via get method"""
        assert bedrock_model_config.get("prompt_caching_support") is True
        assert bedrock_model_config.get("prompt_caching_support", False) is True

    def test_prompt_caching_support_get_method_with_default(self, minimal_model_config):
        """Test get method returns None for prompt_caching_support when explicitly None"""
        # prompt_caching_support is explicitly None in minimal_model_config
        assert minimal_model_config.get("prompt_caching_support") is None
        # get() with default should still return None since field exists and is None
        assert minimal_model_config.get("prompt_caching_support", True) is None


class TestGetModelConfig:
    @pytest.fixture
    def mock_model_settings(self):
        """Mock model_settings.models with test data"""
        return {
            "test-model-1": {
                "upstream_model": "test-upstream-1",
                "backend": "litellm",
                "address": "https://api.test1.com",
                "api_key": "key1",
                "system_prompt_support": True,
                "prompt_caching_support": False,
                "capabilities": ["tools", "files"],
                "friendly_name": "Test Model 1",
                "maker": "Test Corp",
                "max_tokens": 4096,
                "max_tokens_premium": 8192,
                "conversation_token_limit": 1000,
                "conversation_token_limit_premium": 2000,
                "free": True,
                "key": "test-key-1",
                "rate_limit": 100,
                "rate_limit_interval_seconds": 60,
                "max_pages": 10,
                "max_pages_premium": 20,
                "extra_body": {"param": "value"},
            },
            "bedrock-model": {
                "upstream_model": "anthropic.claude-3",
                "backend": "bedrock",
                "inference_profile": "prod-profile",
                "system_prompt_support": True,
                "prompt_caching_support": True,
                "capabilities": ["tools", "vision", "audio", "files"],
                "friendly_name": "Claude 3",
                "maker": "Anthropic",
                "max_tokens": 8192,
                "free": False,
                "key": "claude-key",
            },
            "minimal-model": {
                "upstream_model": "minimal-upstream",
                "backend": "test-backend",
            },
        }

    @patch("aichat.serve.services.models.model_settings")
    def test_get_model_config_full_model(
        self, mock_model_settings_obj, mock_model_settings
    ):
        """Test get_model_config with a fully configured model"""
        mock_model_settings_obj.models = mock_model_settings

        config = get_model_config("test-model-1")

        assert isinstance(config, ModelConfig)
        assert config.model_id == "test-model-1"
        assert config.upstream_model == "test-upstream-1"
        assert config.backend == "litellm"
        assert config.api_base == "https://api.test1.com"
        assert config.api_key == "key1"
        assert config.system_prompt_support is True
        assert config.prompt_caching_support is False
        assert config.tool_support is True
        assert config.image_support is False
        assert config.friendly_name == "Test Model 1"
        assert config.maker == "Test Corp"
        assert config.free is True
        assert config.extra_body == {"param": "value"}

    @patch("aichat.serve.services.models.model_settings")
    def test_get_model_config_bedrock_model(
        self, mock_model_settings_obj, mock_model_settings
    ):
        """Test get_model_config with a Bedrock model"""
        mock_model_settings_obj.models = mock_model_settings

        config = get_model_config("bedrock-model")

        assert config.model_id == "bedrock-model"
        assert config.upstream_model == "anthropic.claude-3"
        assert config.backend == "bedrock"
        assert config.inference_profile == "prod-profile"
        assert config.system_prompt_support is True
        assert config.prompt_caching_support is True
        assert config.api_base is None  # Not in settings, should be None
        assert config.api_key is None  # Not in settings, should be None
        assert config.tool_support is True
        assert config.image_support is True
        assert config.audio_support is True
        assert config.video_support is False
        assert config.free is False

    @patch("aichat.serve.services.models.model_settings")
    def test_get_model_config_bedrock_mantle_capabilities(
        self, mock_model_settings_obj, mock_model_settings
    ):
        """Model configs declare support via capabilities list."""
        mock_model_settings_obj.models = {
            **mock_model_settings,
            "mantle-tools-model": {
                "upstream_model": "openai.gpt-5.4",
                "backend": "bedrock_mantle",
                "capabilities": ["tools", "vision"],
            },
        }

        config = get_model_config("mantle-tools-model")

        assert config.tool_support is True
        assert config.image_support is True

    @patch("aichat.serve.services.models.model_settings")
    def test_get_model_config_minimal_model(
        self, mock_model_settings_obj, mock_model_settings
    ):
        """Test get_model_config with minimal model settings"""
        mock_model_settings_obj.models = mock_model_settings

        config = get_model_config("minimal-model")

        assert config.model_id == "minimal-model"
        assert config.upstream_model == "minimal-upstream"
        assert config.backend == "test-backend"
        # Test defaults are applied
        assert config.tool_support is False  # Default
        assert config.image_support is False  # Default
        assert config.audio_support is False  # Default
        assert config.video_support is False  # Default
        assert config.free is True  # Default
        assert config.api_base is None
        assert config.api_key is None
        assert config.inference_profile is None
        assert config.system_prompt_support is None
        assert config.prompt_caching_support is None
        assert config.tool_role_as_assistant is False
        assert config.friendly_name is None
        assert config.maker is None

    @patch("aichat.serve.services.models.model_settings")
    def test_get_model_config_caching(
        self, mock_model_settings_obj, mock_model_settings
    ):
        """Test that get_model_config uses caching"""
        mock_model_settings_obj.models = mock_model_settings

        # Clear the cache first
        get_model_config.cache_clear()

        # First call
        config1 = get_model_config("test-model-1")

        # Second call should return the same object due to caching
        config2 = get_model_config("test-model-1")

        # Should be the same object (reference equality due to caching)
        assert config1 is config2

        # Verify cache info
        cache_info = get_model_config.cache_info()
        assert cache_info.hits >= 1
        assert cache_info.misses >= 1

    @patch("aichat.serve.services.models.model_settings")
    def test_get_model_config_different_models_not_cached_together(
        self, mock_model_settings_obj, mock_model_settings
    ):
        """Test that different models don't share cache entries"""
        mock_model_settings_obj.models = mock_model_settings

        # Clear the cache first
        get_model_config.cache_clear()

        config1 = get_model_config("test-model-1")
        config2 = get_model_config("bedrock-model")

        # Should be different objects
        assert config1 is not config2
        assert config1.model_id != config2.model_id
        assert config1.backend != config2.backend

    @patch("aichat.serve.services.models.model_settings")
    def test_get_model_config_missing_model(
        self, mock_model_settings_obj, mock_model_settings
    ):
        """Test get_model_config with missing model ID"""
        mock_model_settings_obj.models = mock_model_settings

        with pytest.raises(KeyError):
            get_model_config("nonexistent-model")

    @patch("aichat.serve.services.models.model_settings")
    def test_get_model_config_handles_get_method_calls(
        self, mock_model_settings_obj, mock_model_settings
    ):
        """Test that get_model_config properly handles .get() calls on settings dict"""
        mock_model_settings_obj.models = mock_model_settings

        config = get_model_config("minimal-model")

        # Verify that missing fields get proper defaults
        assert config.rate_limit is None
        assert config.rate_limit_interval_seconds is None
        assert config.max_pages is None
        assert config.max_pages_premium is None
        assert config.extra_body is None
        assert config.max_tokens is None
        assert config.max_tokens_premium is None
        assert config.conversation_token_limit is None
        assert config.conversation_token_limit_premium is None

    def test_get_model_config_cache_clear(self):
        """Test that cache can be cleared"""
        # Clear the cache
        get_model_config.cache_clear()

        # Check cache info after clearing
        cache_info = get_model_config.cache_info()
        assert cache_info.hits == 0
        assert cache_info.misses == 0
        assert cache_info.currsize == 0

    @patch("aichat.serve.services.models.model_settings")
    def test_get_model_config_address_maps_to_api_base(
        self, mock_model_settings_obj, mock_model_settings
    ):
        """Test that 'address' field in settings maps to 'api_base' in ModelConfig"""
        mock_model_settings_obj.models = mock_model_settings

        config = get_model_config("test-model-1")

        # 'address' in settings should become 'api_base' in config
        assert config.api_base == "https://api.test1.com"
        assert mock_model_settings["test-model-1"]["address"] == "https://api.test1.com"


class TestPromptCachingSupport:
    """Test cases specifically for prompt_caching_support field"""

    @patch("aichat.serve.services.models.model_settings")
    def test_prompt_caching_support_true(self, mock_model_settings_obj):
        """Test model with prompt_caching_support=True"""
        mock_model_settings_obj.models = {
            "caching-model": {
                "upstream_model": "test-model",
                "backend": "bedrock",
                "prompt_caching_support": True,
            }
        }

        config = get_model_config("caching-model")
        assert config.prompt_caching_support is True

    @patch("aichat.serve.services.models.model_settings")
    def test_prompt_caching_support_false(self, mock_model_settings_obj):
        """Test model with prompt_caching_support=False"""
        mock_model_settings_obj.models = {
            "no-caching-model": {
                "upstream_model": "test-model",
                "backend": "litellm",
                "prompt_caching_support": False,
            }
        }

        config = get_model_config("no-caching-model")
        assert config.prompt_caching_support is False

    @patch("aichat.serve.services.models.model_settings")
    def test_prompt_caching_support_none_when_missing(self, mock_model_settings_obj):
        """Test model without prompt_caching_support field defaults to None"""
        mock_model_settings_obj.models = {
            "default-model": {
                "upstream_model": "test-model",
                "backend": "litellm",
                # prompt_caching_support omitted
            }
        }

        config = get_model_config("default-model")
        assert config.prompt_caching_support is None


class TestCapabilitiesList:
    @patch("aichat.serve.services.models.model_settings")
    def test_get_model_config_reads_capabilities_list(self, mock_settings):
        get_model_config.cache_clear()
        mock_settings.models = {
            "m": {
                "upstream_model": "x",
                "backend": "litellm",
                "capabilities": ["tools", "vision", "files", "content_agent"],
            }
        }
        config = get_model_config("m")
        assert config.tool_support is True
        assert config.image_support is True
        assert config.audio_support is False
        assert config.video_support is False
        assert config.file_support is True
        assert config.content_agent_support is True
        assert config.deep_research_support is False
