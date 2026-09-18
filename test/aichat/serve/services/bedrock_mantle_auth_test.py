import os
from unittest.mock import patch

import pytest

from aichat.serve.services import bedrock_mantle_auth


@pytest.fixture(autouse=True)
def clear_token_cache():
    bedrock_mantle_auth.clear_bedrock_mantle_token_cache()
    yield
    bedrock_mantle_auth.clear_bedrock_mantle_token_cache()


def test_uses_env_key_when_set():
    with patch.dict(os.environ, {"BEDROCK_MANTLE_API_KEY": "static-key"}, clear=False):
        assert bedrock_mantle_auth.get_bedrock_mantle_bearer_token() == "static-key"


def test_uses_aws_bearer_token_bedrock_env():
    with patch.dict(
        os.environ,
        {"BEDROCK_MANTLE_API_KEY": "", "AWS_BEARER_TOKEN_BEDROCK": "bearer-key"},
        clear=False,
    ):
        # Empty BEDROCK_MANTLE_API_KEY is falsy; AWS_BEARER_TOKEN_BEDROCK is used
        os.environ.pop("BEDROCK_MANTLE_API_KEY", None)
        assert bedrock_mantle_auth.get_bedrock_mantle_bearer_token() == "bearer-key"


def test_caches_generated_token():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("BEDROCK_MANTLE_API_KEY", None)
        os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
        with patch(
            "aichat.serve.services.bedrock_mantle_auth.provide_token",
            return_value="token-1",
        ):
            assert bedrock_mantle_auth.get_bedrock_mantle_bearer_token() == "token-1"
            assert bedrock_mantle_auth.get_bedrock_mantle_bearer_token() == "token-1"
            with patch(
                "aichat.serve.services.bedrock_mantle_auth.provide_token",
                return_value="token-2",
            ) as mock_provide:
                assert (
                    bedrock_mantle_auth.get_bedrock_mantle_bearer_token() == "token-1"
                )
                mock_provide.assert_not_called()


def test_clear_cache_forces_refresh():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("BEDROCK_MANTLE_API_KEY", None)
        os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
        with patch(
            "aichat.serve.services.bedrock_mantle_auth.provide_token",
            side_effect=["token-a", "token-b"],
        ):
            assert bedrock_mantle_auth.get_bedrock_mantle_bearer_token() == "token-a"
            bedrock_mantle_auth.clear_bedrock_mantle_token_cache()
            assert bedrock_mantle_auth.get_bedrock_mantle_bearer_token() == "token-b"
