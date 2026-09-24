"""BDD: MCP integration shared-client lifecycle.

Dup TODOs (same behaviour covered by unit tests):
# TODO: remove test/aichat/serve/test_mcp_integration.py#test_ensure_registry_initialized test/aichat/serve/test_mcp_integration.py:64
# TODO: remove test/aichat/serve/test_mcp_integration.py#test_merge_tools_basic test/aichat/serve/test_mcp_integration.py:201
# TODO: remove test/aichat/serve/test_mcp_integration.py#test_merge_tools_with_include_only test/aichat/serve/test_mcp_integration.py:247
# TODO: remove test/aichat/serve/test_mcp_integration.py#test_merge_tools_duplicate_names test/aichat/serve/test_mcp_integration.py:345
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

FEATURE = "features/mcp_integration.feature"
scenarios(FEATURE)

from aichat.serve import mcp_integration


@pytest.fixture
def ctx():
    return {}


@pytest.fixture(autouse=True)
def cleanup_shared_client():
    yield
    mcp_integration.set_shared_mcp_client(None)


@given("MCP integration state is reset")
def given_reset():
    mcp_integration.set_shared_mcp_client(None)


@given("a shared MCP client is set")
def given_shared_set(ctx):
    shared = SimpleNamespace(name="shared")
    mcp_integration.set_shared_mcp_client(shared)
    ctx["shared"] = shared


@then("the shared MCP client is returned")
def then_shared_returned(ctx):
    assert mcp_integration.get_shared_mcp_client() is ctx["shared"]


@when("the shared client is cleared")
def when_cleared():
    mcp_integration.set_shared_mcp_client(None)


@then("no shared MCP client remains")
def then_no_shared():
    assert mcp_integration.get_shared_mcp_client() is None


@given("no shared MCP client")
def given_no_shared():
    mcp_integration.set_shared_mcp_client(None)


@when("a client is obtained for the request")
def when_client_for_request(ctx):
    ctx["client"] = mcp_integration._client_for_request(MagicMock())


@then("the shared client is reused")
def then_reused(ctx):
    assert ctx["client"] is ctx["shared"]


@then("a new MCP client is created")
def then_new_client(ctx):
    from aichat.serve.services.mcp.client import MCPClient

    assert isinstance(ctx["client"], MCPClient)
    assert ctx["client"] is not ctx.get("shared")


def _make_fake_client(ctx):
    """Fake MCPClient whose behaviour is driven by ctx flags."""

    class FakeMCPClient:
        def __init__(self, registry=None):
            self.registry = registry
            self.get_tools = AsyncMock(return_value=[])
            if ctx.get("warmup_raises"):
                self.get_tools.side_effect = RuntimeError("warmup boom")
            self.close_all_stdio_transports = AsyncMock()
            self.get_tools_with_guidance = AsyncMock(
                return_value=([], ctx.get("guidance", {}))
            )
            if ctx.get("guidance_raises"):
                self.get_tools_with_guidance.side_effect = RuntimeError("guidance boom")

    return FakeMCPClient


def _patch_mcp(monkeypatch, ctx, *, enabled, stdio):
    monkeypatch.setattr(
        mcp_integration,
        "mcp_settings",
        SimpleNamespace(mcp_enabled=enabled, deep_research_enabled=False),
    )
    monkeypatch.setattr(mcp_integration, "has_stdio_mcp_servers", lambda: stdio)
    fake_cls = _make_fake_client(ctx)
    monkeypatch.setattr(mcp_integration, "MCPClient", fake_cls)
    return fake_cls


@given(parsers.parse("mcp enabled is {enabled} and stdio servers are {stdio}"))
def given_warmup_config(ctx, monkeypatch, enabled, stdio):
    _patch_mcp(monkeypatch, ctx, enabled=enabled == "true", stdio=stdio == "yes")


@when("the shared MCP client warmup runs")
def when_warmup(ctx):
    ctx["warmup_result"] = asyncio.run(mcp_integration.warmup_shared_mcp_client())


@then(parsers.parse("the warmup result is {outcome}"))
def then_warmup(ctx, outcome):
    if outcome == "none":
        assert ctx["warmup_result"] is None
    else:
        assert ctx["warmup_result"] is not None


@given("mcp enabled is true and stdio servers exist")
def given_warmup_ready(ctx, monkeypatch):
    _patch_mcp(monkeypatch, ctx, enabled=True, stdio=True)


@then("a client is warmed up and stored as shared")
def then_warmed(ctx):
    client = ctx["warmup_result"]
    assert client is not None
    client.get_tools.assert_awaited_once()


@given("mcp enabled is true and stdio servers exist but startup fails")
def given_warmup_fails(ctx, monkeypatch):
    ctx["warmup_raises"] = True
    _patch_mcp(monkeypatch, ctx, enabled=True, stdio=True)


@given(parsers.parse("a shared client that is {state}"))
def given_shutdown_state(ctx, state):
    if state == "healthy":
        ctx["client"] = SimpleNamespace(close_all_stdio_transports=AsyncMock())
        mcp_integration.set_shared_mcp_client(ctx["client"])
    elif state == "failing":
        client = SimpleNamespace(
            close_all_stdio_transports=AsyncMock(side_effect=RuntimeError("boom"))
        )
        ctx["client"] = client
        mcp_integration.set_shared_mcp_client(client)


@when("the shared MCP client shuts down")
def when_shutdown(ctx):
    ctx["client"] = mcp_integration.get_shared_mcp_client()
    asyncio.run(mcp_integration.shutdown_shared_mcp_client(ctx["client"]))


@then(parsers.parse("the close behaviour is {outcome}"))
def then_close(ctx, outcome):
    if outcome == "skipped":
        assert ctx["client"] is None
    else:
        ctx["client"].close_all_stdio_transports.assert_awaited_once()
    assert mcp_integration.get_shared_mcp_client() is None


@given("a registry with handlers and a guidance client")
def given_guidance_client(ctx, monkeypatch):
    ctx["guidance"] = {"brave_web_search": "use it"}
    fake_cls = _patch_mcp(monkeypatch, ctx, enabled=True, stdio=False)
    mcp_integration.set_shared_mcp_client(fake_cls())


@when("tool guidance is fetched")
def when_guidance_fetch(ctx):
    ctx["guidance_result"] = asyncio.run(mcp_integration.get_tool_guidance())


@then("the guidance dictionary is returned")
def then_guidance_result(ctx):
    assert ctx["guidance_result"] == ctx["guidance"]


@given("a guidance client that raises")
def given_guidance_raises(ctx, monkeypatch):
    ctx["guidance_raises"] = True
    fake_cls = _patch_mcp(monkeypatch, ctx, enabled=True, stdio=False)
    mcp_integration.set_shared_mcp_client(fake_cls())


@then("the guidance is empty")
def then_guidance_empty(ctx):
    assert ctx["guidance_result"] == {}
