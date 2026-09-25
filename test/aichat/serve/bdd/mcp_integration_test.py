"""BDD: MCP integration shared-client lifecycle."""

import asyncio
import logging
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

    client = ctx["client"]
    assert isinstance(client, MCPClient)
    assert client is not ctx.get("shared")
    # The fallback client is real: MCPClient.__init__ only stores the
    # registry (no I/O, no subprocess spawn — spawning happens in
    # get_tools), so constructing it against a bare MagicMock registry is
    # safe. Close it so no transports leak.
    asyncio.run(client.close_all_stdio_transports())


def _make_fake_client(ctx):
    """Fake MCPClient whose behaviour is driven by ctx flags."""

    class FakeMCPClient:
        constructed = 0

        def __init__(self, registry=None):
            FakeMCPClient.constructed += 1
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
        SimpleNamespace(
            mcp_enabled=enabled,
            deep_research_enabled=False,
            # Any other attr read by warmup/guidance must AttributeError
            # loudly, so only the fields prod reads today are provided.
            mcp_servers=[],
        ),
    )
    monkeypatch.setattr(mcp_integration, "has_stdio_mcp_servers", lambda: stdio)
    fake_cls = _make_fake_client(ctx)
    monkeypatch.setattr(mcp_integration, "MCPClient", fake_cls)
    ctx["mcp_client_cls"] = fake_cls
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
        if ctx.get("warmup_raises"):
            # Failure path: one construction attempt whose get_tools raised.
            assert ctx["mcp_client_cls"].constructed == 1
        else:
            # Skip path: no client is constructed at all.
            assert ctx["mcp_client_cls"].constructed == 0
    else:
        assert ctx["warmup_result"] is not None
        assert ctx["mcp_client_cls"].constructed == 1


@given("mcp enabled is true and stdio servers exist")
def given_warmup_ready(ctx, monkeypatch):
    _patch_mcp(monkeypatch, ctx, enabled=True, stdio=True)


@then("a client is warmed up and returned unshared")
def then_warmed(ctx):
    client = ctx["warmup_result"]
    assert client is not None
    client.get_tools.assert_awaited_once()
    # warmup_shared_mcp_client returns the client but does NOT store it;
    # storing is the lifespan's job, so the shared slot stays empty here.
    assert mcp_integration.get_shared_mcp_client() is None


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
def when_shutdown(ctx, caplog):
    ctx["client"] = mcp_integration.get_shared_mcp_client()
    # Capture at WARNING before the shutdown runs so the failure record is
    # observable regardless of the logger's ambient effective level.
    with caplog.at_level(logging.WARNING, logger="aichat.serve.mcp_integration"):
        asyncio.run(mcp_integration.shutdown_shared_mcp_client(ctx["client"]))


@then(parsers.parse("the close behaviour is {outcome}"))
def then_close(ctx, outcome, caplog):
    if outcome == "skipped":
        assert ctx["client"] is None
    else:
        ctx["client"].close_all_stdio_transports.assert_awaited_once()
    if outcome == "error logged safely":
        assert any(
            "Error closing MCP stdio transports" in r.message for r in caplog.records
        )
    # Pinned on purpose: shutdown_shared_mcp_client clears the shared slot
    # in its finally block, so even a failing close leaves no shared client.
    assert mcp_integration.get_shared_mcp_client() is None


@given("a shared client that provides guidance")
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
    # The empty dict comes from the failure branch of the real fetch, not
    # from a skipped call: the fake client's fetch was attempted.
    client = mcp_integration.get_shared_mcp_client()
    client.get_tools_with_guidance.assert_awaited_once()
