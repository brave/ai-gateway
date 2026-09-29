"""Tests for server startup settings checks."""

import os

os.environ.setdefault("ENV", "test")

from unittest.mock import patch

import pytest

from aichat.serve.server_settings import check_log_level


class TestCheckLogLevel:
    def test_allows_warning_in_production(self):
        with patch("aichat.serve.server_settings.server_settings") as settings:
            settings.env = "production"
            settings.log_level = "WARNING"
            check_log_level()

    def test_allows_error_in_production(self):
        with patch("aichat.serve.server_settings.server_settings") as settings:
            settings.env = "production"
            settings.log_level = "ERROR"
            check_log_level()

    def test_blocks_debug_in_production(self):
        with patch("aichat.serve.server_settings.server_settings") as settings:
            settings.env = "production"
            settings.log_level = "DEBUG"
            with pytest.raises(RuntimeError, match="LOG_LEVEL must be WARNING"):
                check_log_level()

    def test_blocks_info_in_production(self):
        with patch("aichat.serve.server_settings.server_settings") as settings:
            settings.env = "production"
            settings.log_level = "info"
            with pytest.raises(RuntimeError, match="LOG_LEVEL must be WARNING"):
                check_log_level()

    def test_allows_info_outside_production(self):
        with patch("aichat.serve.server_settings.server_settings") as settings:
            settings.env = "local"
            settings.log_level = "INFO"
            check_log_level()
