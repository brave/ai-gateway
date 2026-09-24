"""BDD: Deep research citation formatting helpers."""

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

FEATURE = "features/citations.feature"
scenarios(FEATURE)

from aichat.serve.services.mcp.handlers.deep_research.citations import (
    build_citation_map,
    encode_url_for_markdown,
    format_answer_with_citations,
    format_citations_plaintext,
)


@pytest.fixture
def ctx():
    return {}


@then("a url with spaces and parens encodes percent tokens")
def then_encode():
    assert (
        encode_url_for_markdown("https://x.example/a b (c)")
        == "https://x.example/a%20b%20%28c%29"
    )


CITATION_FIXTURES = {
    "numbered and linked": [{"number": 1, "url": "https://ex one.example/a (b)"}],
    "missing number": [{"url": "https://x.example"}],
    "missing url": [{"number": 2}],
    "hostnameless url falls back": [{"number": 3, "url": "weird url no scheme"}],
    "none": [],
}


@given(parsers.parse("citations {citations}"))
def given_citations(ctx, citations):
    ctx["citations"] = CITATION_FIXTURES[citations]
    ctx["citations_name"] = citations


@when("the citation map is built")
def when_build_map(ctx):
    ctx["cmap"] = build_citation_map(ctx["citations"])


@then(parsers.parse("the map has {count:d} entries"))
def then_map_count(ctx, count):
    assert len(ctx["cmap"]) == count


@then("each entry carries url encoded url and hostname")
def then_map_entries(ctx):
    if len(ctx["cmap"]) == 0:
        return
    if ctx["citations_name"] == "hostnameless url falls back":
        entry = ctx["cmap"][3]
        assert entry["hostname"] == "weird url no scheme"
        assert entry["url"] == "weird url no scheme"
    else:
        entry = ctx["cmap"][1]
        assert entry["url"] == "https://ex one.example/a (b)"
        assert entry["encoded_url"] == "https://ex%20one.example/a%20%28b%29"
        assert entry["hostname"] == "ex one.example"


@given(parsers.parse('an answer "The sky is blue[1] indeed"'))
def given_answer(ctx):
    ctx["answer"] = "The sky is blue[1] indeed"
    ctx["citations"] = CITATION_FIXTURES["numbered and linked"]


@when("the answer is formatted with citations")
def when_format_answer(ctx):
    ctx["out"] = format_answer_with_citations(ctx["answer"], ctx["citations"])


@then("the answer starts with a separator")
def then_separator(ctx):
    assert ctx["out"].startswith("---\n\n")


@then("citation markers are preceded by spaces")
def then_spaced(ctx):
    assert ctx["out"].endswith("The sky is blue [1] indeed")


@given(parsers.parse('an answer "Final verdict" with {citations}'))
def given_answer_with(ctx, citations):
    ctx["answer"] = "Final verdict"
    ctx["citations"] = CITATION_FIXTURES[citations]


@when("plaintext citations are appended")
def when_plaintext(ctx):
    ctx["out"] = format_citations_plaintext(ctx["answer"], ctx["citations"])


@then(parsers.parse("the plaintext result {expectation}"))
def then_plaintext(ctx, expectation):
    if expectation == "lists sources":
        assert ctx["out"].startswith("Final verdict\n\nSources:\n")
        assert "[1] https://ex one.example/a (b)" in ctx["out"]
    else:
        assert ctx["out"] == "Final verdict"
