from unittest import mock
from unittest.mock import AsyncMock

import pytest

from aichat.serve.common_api import check_requests_common, resolve_model


@mock.patch(
    "aichat.serve.common_api.check_automatic_mode_daily_limit", new_callable=AsyncMock
)
@mock.patch("aichat.serve.common_api.rate_limiting_settings")
@mock.patch("aichat.serve.common_api.model_settings")
@pytest.mark.asyncio
async def test_check_requests_common_model_validation(
    mock_model_settings, mock_rate_limiting_settings, mock_auto_limit
):
    mock_model_settings.models = {
        "claude-3-haiku": {"type": "llm", "free": True},
        "claude-3-sonnet": {"type": "llm", "free": False},
    }
    mock_rate_limiting_settings.rate_limiting_enabled = False
    # Mock automatic mode check to return False (limit exceeded)
    mock_auto_limit.return_value = False

    # Create mock request
    mock_request = mock.Mock()
    mock_request.state = mock.Mock()
    mock_request.state.service_key_id = "test-key"

    # Test case 1: Valid model
    common = {
        "model": "claude-3-haiku",
        "x_forwarded_for": "host",
        "x_forwarded_host": "host",
        "is_premium_host": False,
        "is_valid_brave_services_key_v2": True,
        "has_valid_premium_credential": False,
        "is_automatic_model_request": False,
    }
    assert await check_requests_common(mock_request, common) is None

    # Test case 2: Invalid model
    common["model"] = "invalid"
    common["is_automatic_model_request"] = False
    response = await check_requests_common(mock_request, common)
    assert response is not None
    assert response.status_code == 404

    # Test case 3: Premium model without premium credential and automatic mode limit exceeded
    common["model"] = "claude-3-sonnet"
    common["is_automatic_model_request"] = False
    common["has_valid_premium_credential"] = False
    mock_auto_limit.return_value = False  # Limit exceeded
    response = await check_requests_common(mock_request, common)
    assert response is not None
    assert response.status_code == 403

    # Test case 4: Premium model with premium credential
    common["model"] = "claude-3-sonnet"
    common["is_automatic_model_request"] = False
    common["has_valid_premium_credential"] = True
    assert await check_requests_common(mock_request, common) is None


@mock.patch(
    "aichat.serve.common_api.check_automatic_mode_daily_limit", new_callable=AsyncMock
)
@mock.patch("aichat.serve.common_api.rate_limiting_settings")
@mock.patch("aichat.serve.common_api.model_settings")
@pytest.mark.asyncio
async def test_check_requests_common_undefined_capability(
    mock_model_settings, mock_rate_limiting_settings, mock_auto_limit
):
    """Test that undefined capability is handled correctly (defaults to non-content-agent)."""
    mock_model_settings.models = {"claude-3-haiku": {"type": "llm", "free": True}}
    mock_rate_limiting_settings.rate_limiting_enabled = False
    mock_auto_limit.return_value = False  # Not needed for free model

    # Mock request with no capability attribute
    mock_request = mock.Mock()
    mock_request.state = mock.Mock()
    # Deliberately don't set capability attribute

    common = {
        "model": "claude-3-haiku",
        "x_forwarded_for": "host",
        "x_forwarded_host": "host",
        "is_premium_host": False,
        "is_valid_brave_services_key_v2": True,
        "has_valid_premium_credential": False,
        "is_automatic_model_request": False,
    }

    # Should not raise an error and should work normally
    result = await check_requests_common(mock_request, common)
    assert result is None  # Should succeed


@mock.patch(
    "aichat.serve.common_api.check_automatic_mode_daily_limit", new_callable=AsyncMock
)
@mock.patch("aichat.serve.common_api.rate_limiting_settings")
@mock.patch("aichat.serve.common_api.model_settings")
@pytest.mark.asyncio
async def test_check_requests_common_none_capability(
    mock_model_settings, mock_rate_limiting_settings, mock_auto_limit
):
    """Test that None capability is handled correctly (defaults to non-content-agent)."""
    mock_model_settings.models = {"claude-3-haiku": {"type": "llm", "free": True}}
    mock_rate_limiting_settings.rate_limiting_enabled = False
    mock_auto_limit.return_value = False  # Not needed for free model

    # Mock request with capability explicitly set to None
    mock_request = mock.Mock()
    mock_request.state = mock.Mock()
    mock_request.state.capability = None

    common = {
        "model": "claude-3-haiku",
        "x_forwarded_for": "host",
        "x_forwarded_host": "host",
        "is_premium_host": False,
        "is_valid_brave_services_key_v2": True,
        "has_valid_premium_credential": False,
        "is_automatic_model_request": False,
    }

    # Should not raise an error and should work normally
    result = await check_requests_common(mock_request, common)
    assert result is None  # Should succeed


@mock.patch(
    "aichat.serve.common_api.check_automatic_mode_daily_limit", new_callable=AsyncMock
)
@mock.patch("aichat.serve.common_api.rate_limiting_settings")
@mock.patch("aichat.serve.common_api.model_settings")
@pytest.mark.asyncio
async def test_check_requests_common_automatic_with_content_agent_capability(
    mock_model_settings, mock_rate_limiting_settings, mock_auto_limit
):
    """'automatic' model with content_agent capability must not raise KeyError."""
    mock_model_settings.models = {"claude-3-sonnet": {"type": "llm", "free": False}}
    mock_rate_limiting_settings.rate_limiting_enabled = False
    mock_auto_limit.return_value = {"count": 0, "limit": 10, "exceeded": False}

    mock_request = mock.Mock()
    mock_request.state = mock.Mock()
    mock_request.state.capability = "content_agent"

    common = {
        "model": "automatic",
        "x_forwarded_for": "host",
        "x_forwarded_host": "host",
        "is_premium_host": False,
        "is_valid_brave_services_key_v2": True,
        "has_valid_premium_credential": False,
        "is_automatic_model_request": True,
    }

    # Must not raise KeyError; automatic model bypasses content_agent check
    result = await check_requests_common(mock_request, common)
    assert result is None


@mock.patch("aichat.serve.common_api.model_settings")
def test_resolve_model_passes_through_free_model(mock_model_settings):
    """Free model on non-premium host is returned as-is."""
    mock_model_settings.models = {
        "claude-3-haiku": {"type": "llm", "free": True},
        "claude-3-sonnet": {"type": "llm", "free": False},
    }
    request = mock.Mock()
    request.model = "claude-3-haiku"

    assert resolve_model(request, is_premium_host=False) == "claude-3-haiku"


@mock.patch("aichat.serve.common_api.model_settings")
def test_resolve_model_passes_through_premium_model_on_premium_host(
    mock_model_settings,
):
    """Premium model on premium host is returned as-is."""
    mock_model_settings.models = {
        "claude-3-haiku": {"type": "llm", "free": True},
        "claude-3-sonnet": {"type": "llm", "free": False},
    }
    request = mock.Mock()
    request.model = "claude-3-sonnet"

    assert resolve_model(request, is_premium_host=True) == "claude-3-sonnet"


@mock.patch("aichat.serve.common_api.PREMIUM_MODEL_DOWNGRADE_TOTAL")
@mock.patch("aichat.serve.common_api.model_settings")
def test_resolve_model_downgrades_premium_model_on_non_premium_host(
    mock_model_settings, mock_metric
):
    """Premium model on non-premium host is downgraded to 'automatic'."""
    mock_model_settings.models = {
        "claude-3-haiku": {"type": "llm", "free": True},
        "claude-3-sonnet": {"type": "llm", "free": False},
    }
    mock_metric.labels.return_value = mock.Mock()
    request = mock.Mock()
    request.model = "claude-3-sonnet"

    result = resolve_model(request, is_premium_host=False)

    assert result == "automatic"
    mock_metric.labels.assert_called_once_with(model="claude-3-sonnet")
    mock_metric.labels.return_value.inc.assert_called_once()


@mock.patch("aichat.serve.common_api.model_settings")
def test_resolve_model_preserves_automatic_on_non_premium_host(mock_model_settings):
    """Explicit 'automatic' requests on non-premium host are not double-downgraded."""
    mock_model_settings.models = {
        "claude-3-haiku": {"type": "llm", "free": True},
    }
    request = mock.Mock()
    request.model = "automatic"

    assert resolve_model(request, is_premium_host=False) == "automatic"


@mock.patch("aichat.serve.common_api.model_settings")
def test_resolve_model_preserves_automatic_variant_on_non_premium_host(
    mock_model_settings,
):
    """'automatic-*' variants on non-premium host are not downgraded."""
    mock_model_settings.models = {
        "claude-3-haiku": {"type": "llm", "free": True},
    }
    request = mock.Mock()
    request.model = "automatic-free-default"

    assert resolve_model(request, is_premium_host=False) == "automatic-free-default"
