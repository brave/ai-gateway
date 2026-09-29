"""PDF trust-boundary security scenarios.

Committed scenarios cover the B1 data-URL detection fix. The B2 (api-key
content loss / file_id smuggling) and B3 (title-path forwarding) scenarios
are kept red and uncommitted in tmp/red_uncommitted/ until their fixes land.
See the comment header in features/pdf_security.feature.
"""

import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from aichat.serve.services import pdf as pdf_service

scenarios("features/pdf_security.feature")

PDF = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
B64 = base64.b64encode(PDF).decode()


@pytest.fixture
def ctx():
    return {}


@given("a pdf payload")
def pdf_payload(ctx):
    ctx["b64"] = B64
    return ctx


@when(
    parsers.parse('a file part with data URL "data:application/pdf{suffix}" is checked')
)
def check_data_url(ctx, suffix):
    if suffix.endswith("base64,"):
        url = f"data:application/pdf{suffix}{ctx['b64']}"
    else:
        url = suffix
    ctx["url"] = url
    ctx["is_pdf"] = pdf_service._is_pdf_file_data(url)
    return ctx


@when('a file part with data URL "" is checked')
def check_empty_data_url(ctx):
    ctx["url"] = ""
    ctx["is_pdf"] = pdf_service._is_pdf_file_data("")
    return ctx


@then("it is classified as a pdf")
def classified_as_pdf(ctx):
    assert ctx["is_pdf"], f"PDF went undetected: {ctx['url'][:64]!r}"


@then("it is not classified as a pdf")
def not_classified_as_pdf(ctx):
    assert not ctx["is_pdf"]


@when(
    parsers.parse(
        'a user message carries a pdf with data URL "data:application/pdf{suffix}"'
    )
)
def user_message_with_pdf(ctx, suffix):
    url = f"data:application/pdf{suffix}{ctx['b64']}"
    ctx["message"] = {
        "role": "user",
        "content": [{"type": "file", "file": {"filename": "a.pdf", "file_data": url}}],
    }
    analyzer = SimpleNamespace(calls=[])

    async def fake_analyze(args):
        analyzer.calls.append(args)
        return {"passthrough": None, "total_pages": 1, "estimated_tokens": 1}

    ctx["analyzer"] = analyzer
    patcher = patch.object(pdf_service, "call_pdf_analyze", fake_analyze)
    patcher.start()
    return ctx


@when("pdf limits are enforced")
def enforce_pdf_limits(ctx):
    asyncio.run(pdf_service.process_messages_for_pdf_limits([ctx["message"]], 100000))


@then("the sandbox was invoked")
def sandbox_invoked(ctx):
    assert ctx["analyzer"].calls, "PDF skipped sandbox processing"
