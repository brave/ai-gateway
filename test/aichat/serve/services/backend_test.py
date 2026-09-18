from unittest.mock import MagicMock, call, patch

import pytest

from aichat.serve.services.backend import (
    clear_backend_cache,
    get_backend,
    initialize_backends,
)
from aichat.serve.services.models import ModelConfig


@pytest.fixture
def mock_model_config_litellm():
    """Mock ModelConfig for LiteLLM backend"""
    return ModelConfig(
        model_id="test-litellm-model",
        upstream_model="test-upstream",
        backend="litellm",
        api_base="https://test-api.com",
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
        friendly_name="Test LiteLLM Model",
        maker="Test Maker",
        max_tokens=4096,
        max_tokens_premium=8192,
        conversation_token_limit=16384,
        conversation_token_limit_premium=32768,
        free=False,
        key="test-key",
        rate_limit=100,
        rate_limit_interval_seconds=60,
        max_pages=5,
        max_pages_premium=10,
        extra_body=None,
    )


@pytest.fixture
def mock_model_config_bedrock():
    """Mock ModelConfig for Bedrock backend"""
    return ModelConfig(
        model_id="test-bedrock-model",
        upstream_model="test-bedrock-upstream",
        backend="bedrock",
        api_base="https://bedrock-api.com",
        api_key=None,
        inference_profile="test-profile",
        system_prompt_support=True,
        prompt_caching_support=True,
        prompt_caching_enabled=True,
        tool_support=True,
        image_support=True,
        audio_support=False,
        video_support=False,
        file_support=True,
        friendly_name="Test Bedrock Model",
        maker="AWS",
        max_tokens=4096,
        max_tokens_premium=8192,
        conversation_token_limit=16384,
        conversation_token_limit_premium=32768,
        free=False,
        key="bedrock-key",
        rate_limit=50,
        rate_limit_interval_seconds=60,
        max_pages=3,
        max_pages_premium=8,
        extra_body=None,
    )


@pytest.fixture
def mock_model_config_vllm():
    """Mock ModelConfig for vLLM backend"""
    return ModelConfig(
        model_id="test-vllm-model",
        upstream_model="test-vllm-upstream",
        backend="vllm",
        api_base="https://vllm-api.com",
        api_key="vllm-key",
        inference_profile=None,
        system_prompt_support=True,
        prompt_caching_support=None,
        prompt_caching_enabled=None,
        tool_support=False,
        image_support=False,
        audio_support=False,
        video_support=False,
        file_support=True,
        friendly_name="Test vLLM Model",
        maker="vLLM",
        max_tokens=2048,
        max_tokens_premium=4096,
        conversation_token_limit=8192,
        conversation_token_limit_premium=16384,
        free=True,
        key="vllm-key",
        rate_limit=200,
        rate_limit_interval_seconds=60,
        max_pages=10,
        max_pages_premium=20,
        extra_body=None,
    )


@pytest.fixture
def mock_model_config_unsupported():
    """Mock ModelConfig for unsupported backend"""
    return ModelConfig(
        model_id="test-unsupported-model",
        upstream_model="test-unsupported-upstream",
        backend="unsupported",
        api_base="https://unsupported-api.com",
        api_key="unsupported-key",
        inference_profile=None,
        system_prompt_support=False,
        prompt_caching_support=None,
        prompt_caching_enabled=None,
        tool_support=False,
        image_support=False,
        audio_support=False,
        video_support=False,
        file_support=True,
        friendly_name="Unsupported Model",
        maker="Unknown",
        max_tokens=1024,
        max_tokens_premium=2048,
        conversation_token_limit=4096,
        conversation_token_limit_premium=8192,
        free=True,
        key="unsupported-key",
        rate_limit=10,
        rate_limit_interval_seconds=60,
        max_pages=1,
        max_pages_premium=2,
        extra_body=None,
    )


class TestInitializeBackends:
    @patch("aichat.serve.services.backend.get_global_router")
    @patch("aichat.serve.services.backend.logger")
    @patch("aichat.serve.services.backend.model_settings")
    @patch("aichat.serve.services.backend.get_backend")
    def test_initialize_backends_all_successful(
        self, mock_get_backend, mock_model_settings, mock_logger, mock_get_router
    ):
        """Test successful initialization of all backends"""
        # Setup mock models
        mock_models = {"model1": {}, "model2": {}, "model3": {}}
        mock_model_settings.models = mock_models
        mock_get_backend.side_effect = [MagicMock(), MagicMock(), MagicMock()]
        mock_get_router.return_value = MagicMock()

        initialize_backends()

        # Verify router was initialized first
        mock_get_router.assert_called_once()

        # Verify get_backend was called for each model
        assert mock_get_backend.call_count == 3
        mock_get_backend.assert_has_calls(
            [call("model1"), call("model2"), call("model3")]
        )

        # Verify logging
        mock_logger.info.assert_has_calls(
            [
                call("Initializing backends for all configured models..."),
                call("Successfully initialized 3/3 backends"),
            ]
        )
        assert not mock_logger.error.called

    @patch("aichat.serve.services.backend.get_global_router")
    @patch("aichat.serve.services.backend.logger")
    @patch("aichat.serve.services.backend.model_settings")
    @patch("aichat.serve.services.backend.get_backend")
    def test_initialize_backends_partial_failure(
        self, mock_get_backend, mock_model_settings, mock_logger, mock_get_router
    ):
        """Test initialization with some backend failures"""
        # Setup mock models
        mock_models = {"model1": {}, "model2": {}, "model3": {}}
        mock_model_settings.models = mock_models
        mock_get_router.return_value = MagicMock()

        # Second model fails
        def side_effect(model_id):
            if model_id == "model2":
                raise ValueError("Backend initialization failed")
            return MagicMock()

        mock_get_backend.side_effect = side_effect

        initialize_backends()

        # Verify get_backend was called for each model
        assert mock_get_backend.call_count == 3

        # Verify logging includes error for failed model
        mock_logger.info.assert_has_calls(
            [
                call("Initializing backends for all configured models..."),
                call("Successfully initialized 2/3 backends"),
            ]
        )
        mock_logger.error.assert_called_once_with(
            "Failed to initialize backend for model model2: Backend initialization failed"
        )

    @patch("aichat.serve.services.backend.get_global_router")
    @patch("aichat.serve.services.backend.logger")
    @patch("aichat.serve.services.backend.model_settings")
    @patch("aichat.serve.services.backend.get_backend")
    def test_initialize_backends_all_fail(
        self, mock_get_backend, mock_model_settings, mock_logger, mock_get_router
    ):
        """Test initialization when all backends fail"""
        # Setup mock models
        mock_models = {"model1": {}, "model2": {}}
        mock_model_settings.models = mock_models
        mock_get_router.return_value = MagicMock()
        mock_get_backend.side_effect = [
            Exception("First backend failed"),
            Exception("Second backend failed"),
        ]

        initialize_backends()

        # Verify get_backend was called for each model
        assert mock_get_backend.call_count == 2

        # Verify logging shows all failures
        mock_logger.info.assert_has_calls(
            [
                call("Initializing backends for all configured models..."),
                call("Successfully initialized 0/2 backends"),
            ]
        )
        assert mock_logger.error.call_count == 2

    @patch("aichat.serve.services.backend.get_global_router")
    @patch("aichat.serve.services.backend.logger")
    @patch("aichat.serve.services.backend.model_settings")
    @patch("aichat.serve.services.backend.get_backend")
    def test_initialize_backends_no_models(
        self, mock_get_backend, mock_model_settings, mock_logger, mock_get_router
    ):
        """Test initialization with no configured models"""
        # Setup empty models
        mock_model_settings.models = {}
        mock_get_router.return_value = MagicMock()

        initialize_backends()

        # Verify router is still initialized
        mock_get_router.assert_called_once()

        # Verify no backends were attempted
        assert not mock_get_backend.called

        # Verify logging
        mock_logger.info.assert_has_calls(
            [
                call("Initializing backends for all configured models..."),
                call("Successfully initialized 0/0 backends"),
            ]
        )
        assert not mock_logger.error.called


class TestGetBackend:
    @patch("aichat.serve.backend.litellm.get_global_router")
    @patch("aichat.serve.services.backend.get_model_config")
    @patch("aichat.serve.services.backend.LitellmBackend")
    def test_get_backend_litellm(
        self,
        mock_litellm_backend,
        mock_get_model_config,
        mock_get_router,
        mock_model_config_litellm,
    ):
        """Test getting backend for LiteLLM model"""
        mock_get_model_config.return_value = mock_model_config_litellm
        mock_backend_instance = MagicMock()
        mock_litellm_backend.return_value = mock_backend_instance
        mock_get_router.return_value = MagicMock()

        # Clear cache before test
        clear_backend_cache()

        result = get_backend("test-litellm-model")

        mock_get_model_config.assert_called_once_with("test-litellm-model")
        mock_litellm_backend.assert_called_once_with(mock_model_config_litellm)
        assert result == mock_backend_instance

    @patch("aichat.serve.backend.litellm.get_global_router")
    @patch("aichat.serve.services.backend.get_model_config")
    @patch("aichat.serve.services.backend.LitellmBackend")
    def test_get_backend_bedrock(
        self,
        mock_litellm_backend,
        mock_get_model_config,
        mock_get_router,
        mock_model_config_bedrock,
    ):
        """Test getting backend for Bedrock model"""
        mock_get_model_config.return_value = mock_model_config_bedrock
        mock_backend_instance = MagicMock()
        mock_litellm_backend.return_value = mock_backend_instance
        mock_get_router.return_value = MagicMock()

        # Clear cache before test
        clear_backend_cache()

        result = get_backend("test-bedrock-model")

        mock_get_model_config.assert_called_once_with("test-bedrock-model")
        mock_litellm_backend.assert_called_once_with(mock_model_config_bedrock)
        assert result == mock_backend_instance

    @patch("aichat.serve.backend.litellm.get_global_router")
    @patch("aichat.serve.services.backend.get_model_config")
    @patch("aichat.serve.services.backend.LitellmBackend")
    def test_get_backend_vllm(
        self,
        mock_litellm_backend,
        mock_get_model_config,
        mock_get_router,
        mock_model_config_vllm,
    ):
        """Test getting backend for vLLM model"""
        mock_get_model_config.return_value = mock_model_config_vllm
        mock_backend_instance = MagicMock()
        mock_litellm_backend.return_value = mock_backend_instance
        mock_get_router.return_value = MagicMock()

        # Clear cache before test
        clear_backend_cache()

        result = get_backend("test-vllm-model")

        mock_get_model_config.assert_called_once_with("test-vllm-model")
        mock_litellm_backend.assert_called_once_with(mock_model_config_vllm)
        assert result == mock_backend_instance

    @patch("aichat.serve.services.backend.get_model_config")
    def test_get_backend_unsupported(
        self, mock_get_model_config, mock_model_config_unsupported
    ):
        """Test getting backend for unsupported backend type"""
        mock_get_model_config.return_value = mock_model_config_unsupported

        # Clear cache before test
        clear_backend_cache()

        with pytest.raises(
            ValueError,
            match="Unsupported backend 'unsupported' for model test-unsupported-model",
        ):
            get_backend("test-unsupported-model")

        mock_get_model_config.assert_called_once_with("test-unsupported-model")

    @patch("aichat.serve.backend.litellm.get_global_router")
    @patch("aichat.serve.services.backend.get_model_config")
    @patch("aichat.serve.services.backend.LitellmBackend")
    def test_get_backend_caching(
        self,
        mock_litellm_backend,
        mock_get_model_config,
        mock_get_router,
        mock_model_config_litellm,
    ):
        """Test that backend instances are cached"""
        mock_get_model_config.return_value = mock_model_config_litellm
        mock_backend_instance = MagicMock()
        mock_litellm_backend.return_value = mock_backend_instance
        mock_get_router.return_value = MagicMock()

        # Clear cache before test
        clear_backend_cache()

        # First call
        result1 = get_backend("test-litellm-model")

        # Second call should return cached result
        result2 = get_backend("test-litellm-model")

        # Should be the same instance
        assert result1 == result2
        assert result1 is result2

        # Model config should only be called once due to caching
        mock_get_model_config.assert_called_once_with("test-litellm-model")
        mock_litellm_backend.assert_called_once_with(mock_model_config_litellm)

    @patch("aichat.serve.backend.litellm.get_global_router")
    @patch("aichat.serve.services.backend.get_model_config")
    @patch("aichat.serve.services.backend.LitellmBackend")
    def test_get_backend_different_models(
        self,
        mock_litellm_backend,
        mock_get_model_config,
        mock_get_router,
        mock_model_config_litellm,
        mock_model_config_bedrock,
    ):
        """Test getting backends for different models"""
        mock_get_router.return_value = MagicMock()

        def get_config_side_effect(model_id):
            if model_id == "test-litellm-model":
                return mock_model_config_litellm
            elif model_id == "test-bedrock-model":
                return mock_model_config_bedrock

        mock_get_model_config.side_effect = get_config_side_effect
        mock_litellm_backend.side_effect = [MagicMock(), MagicMock()]

        # Clear cache before test
        clear_backend_cache()

        # Get different backends
        result1 = get_backend("test-litellm-model")
        result2 = get_backend("test-bedrock-model")

        # Should be different instances
        assert result1 != result2
        assert result1 is not result2

        # Both models should be configured
        assert mock_get_model_config.call_count == 2
        assert mock_litellm_backend.call_count == 2


class TestClearBackendCache:
    @patch("aichat.serve.backend.litellm.get_global_router")
    @patch("aichat.serve.services.backend.get_model_config")
    @patch("aichat.serve.services.backend.LitellmBackend")
    def test_clear_backend_cache(
        self,
        mock_litellm_backend,
        mock_get_model_config,
        mock_get_router,
        mock_model_config_litellm,
    ):
        """Test that clearing cache forces new backend creation"""
        mock_get_model_config.return_value = mock_model_config_litellm
        mock_litellm_backend.side_effect = [MagicMock(), MagicMock()]
        mock_get_router.return_value = MagicMock()

        # Clear cache before test
        clear_backend_cache()

        # First call
        result1 = get_backend("test-litellm-model")

        # Clear cache
        clear_backend_cache()

        # Second call after cache clear should create new instance
        result2 = get_backend("test-litellm-model")

        # Should be different instances due to cache clear
        assert result1 != result2
        assert result1 is not result2

        # Model config should be called twice (once for each backend creation)
        assert mock_get_model_config.call_count == 2
        assert mock_litellm_backend.call_count == 2

    def test_clear_backend_cache_no_error_when_empty(self):
        """Test that clearing cache doesn't error when cache is empty"""
        # Should not raise any errors
        clear_backend_cache()
        clear_backend_cache()  # Clear again to ensure no error
