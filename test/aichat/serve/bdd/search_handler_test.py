"""BDD: SearchServerHandler.

Dup TODOs (same behaviour covered by unit tests):
# TODO: remove test/aichat/serve/services/mcp/test_search_handler.py#test_format_web_sources_content_basic test/aichat/serve/services/mcp/test_search_handler.py:55
# TODO: remove test/aichat/serve/services/mcp/test_search_handler.py#test_format_web_sources_content_with_page_content test/aichat/serve/services/mcp/test_search_handler.py:83
# TODO: remove test/aichat/serve/services/mcp/test_search_handler.py#test_format_web_sources_content_with_extra_snippets test/aichat/serve/services/mcp/test_search_handler.py:105
# TODO: remove test/aichat/serve/services/mcp/test_search_handler.py#test_format_web_sources_content_empty_sources test/aichat/serve/services/mcp/test_search_handler.py:157
"""

import json

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

FEATURE = "features/search_handler.feature"
scenarios(FEATURE)

from aichat.prompts.search_results import SearchResults
from aichat.protocol.open_ai_protocol import TextContentPart
from aichat.serve.services.mcp.handlers.search import SearchServerHandler


def _source(url="https://e.example", title="Example"):
    return {"title": title, "url": url, "snippet": "a snippet"}


@pytest.fixture
def ctx():
    return {}


@given("a search server handler")
def given_handler(ctx):
    ctx["handler"] = SearchServerHandler()


@when(parsers.parse("the handler validates a result of kind {kind}"))
def when_validate(ctx, kind):
    h = ctx["handler"]
    if kind == "dict_content":
        result = {"content": [{"type": "text", "text": "x"}]}
    elif kind == "dict_no_content":
        result = {"other": 1}
    else:  # non_dict
        result = "text"
    ctx["valid"] = h.validate_result(result)


@then("the result is valid")
def then_valid(ctx):
    assert ctx["valid"] is True


@then("the result is invalid")
def then_invalid(ctx):
    assert ctx["valid"] is False


@given("a search data row with a provider url and weather data")
def given_weather_row(ctx):
    ctx["search_data"] = {
        "provider": {"name": "Test", "url": "https://w.example"},
        "weather": {"temp": 10},
    }


@when("rich weather data is extracted")
def when_extract_weather(ctx):
    ctx["ok"] = SearchServerHandler._extract_rich_weather_data(ctx["search_data"])


@then("the extraction succeeds")
def then_extract_ok(ctx):
    assert ctx["ok"] is True


@then("the extraction fails")
def then_extract_fail(ctx):
    assert ctx["ok"] is False


@then(
    'the search data has title "Weather JSON Data from Test" '
    'url "https://w.example" and page content containing the weather'
)
def then_weather_fields(ctx):
    d = ctx["search_data"]
    assert d["title"] == "Weather JSON Data from Test"
    assert d["url"] == "https://w.example"
    assert "weather data in JSON format" in d["page_content"]
    assert "temp" in d["page_content"]


@given(parsers.parse("a weather payload variant {variant}"))
def given_weather_variant(ctx, variant):
    d = {}
    if variant == "no_provider":
        # Weather data present so the missing provider key is the only
        # reason the extraction fails.
        d["weather"] = {"temp": "20"}
    elif variant == "blank_url":
        # Weather data present so the blank provider url is the only
        # reason the extraction fails.
        d["provider"] = {"name": "N", "url": "   "}
        d["weather"] = {"temp": "20"}
    elif variant == "missing_weather":
        d["provider"] = {"name": "N", "url": "https://w.example"}
    elif variant == "provider_not_dict":
        d["provider"] = "nope"
        d["weather"] = {"temp": "20"}
    ctx["search_data"] = d


@given("a currency payload with provider url conversion and name")
def given_currency_row(ctx):
    ctx["search_data"] = {
        "provider": {"name": "X", "url": "https://c.example"},
        "currency": {
            "conversion": {"query": "100 USD to EUR", "info": "92.3"},
            "timeseries": [1, 2],
        },
    }


@when("rich currency data is extracted")
def when_extract_currency(ctx):
    ctx["ok"] = SearchServerHandler._extract_rich_currency_data(ctx["search_data"])


@then("the currency payload gains title url and page content")
def then_currency_fields(ctx):
    d = ctx["search_data"]
    assert d["title"] == "Currency Conversion JSON Data from X"
    assert d["url"] == "https://c.example"
    assert "Conversion Query: 100 USD to EUR" in d["page_content"]
    assert "Conversion Result: 92.3" in d["page_content"]
    assert "Currency Data Timeseries: [1, 2]" in d["page_content"]


@given(parsers.parse("a currency payload variant {variant}"))
def given_currency_variant(ctx, variant):
    d = {}
    if variant == "blank_url":
        # Currency data present so the blank provider url is the only
        # reason the extraction fails.
        d["provider"] = {"name": "N", "url": "   "}
        d["currency"] = {
            "conversion": {"query": "1 USD to EUR", "info": "1"},
            "timeseries": 1,
        }
    elif variant == "provider_missing":
        # provider key absent entirely; currency is otherwise usable so the
        # failure is attributable to the missing provider.
        d["currency"] = {
            "conversion": {"query": "1 USD to EUR", "info": "1"},
            "timeseries": 1,
        }
    elif variant == "currency_missing":
        d["provider"] = {"name": "N", "url": "https://c.example"}
    elif variant == "conversion_missing":
        # provider and currency are usable; the conversion key is what is
        # missing, exercising the conversion-dict check.
        d["provider"] = {"name": "N", "url": "https://c.example"}
        d["currency"] = {"timeseries": 1}
    ctx["search_data"] = d


def _result(texts, **extra):
    result = {"content": [{"type": "text", "text": t} for t in texts]}
    result.update(extra)
    return result


@given("a raw search result with json list of two results")
def given_json_list(ctx):
    ctx["raw"] = _result(
        [
            json.dumps(
                [
                    _source(url="https://a.example", title="A"),
                    _source(url="https://b.example", title="B"),
                ]
            )
        ]
    )


@given("a raw search result wrapping results list")
def given_wrapper(ctx):
    ctx["raw"] = _result([json.dumps({"results": [_source(title="A")]})])


@given("a raw search result with scalar results value")
def given_scalar_results(ctx):
    ctx["raw"] = _result([json.dumps({"results": _source(title="A")})])


@given("a raw search result with two jsonl lines and one broken line")
def given_jsonl(ctx):
    ctx["raw"] = _result(
        [
            json.dumps(_source(url="https://a.example", title="A"))
            + "\n"
            + "not json at all"
            + "\n"
            + json.dumps(_source(url="https://b.example", title="B")),
        ]
    )


@given("a raw search result with blank and empty texts")
def given_blank_texts(ctx):
    ctx["raw"] = _result(["   ", ""])


@given("a raw search result containing a rich weather entry")
def given_rich_weather(ctx):
    ctx["raw"] = _result(
        [
            json.dumps(
                {
                    "type": "rich",
                    "subtype": "weather",
                    "provider": {
                        "name": "Weather Provider",
                        "url": "https://w.example",
                    },
                    "weather": {"temp": 21},
                }
            )
        ]
    )


@given("a raw search result containing an unusable rich weather entry")
def given_rich_weather_bad(ctx):
    ctx["raw"] = _result([json.dumps({"type": "rich", "subtype": "weather"})])


@given("a raw search result containing a rich currency entry")
def given_rich_currency(ctx):
    ctx["raw"] = _result(
        [
            json.dumps(
                {
                    "type": "rich",
                    "subtype": "currency",
                    "provider": {"name": "X", "url": "https://c.example"},
                    "currency": {
                        "conversion": {"query": "100 USD to EUR", "info": "92.3"},
                        "timeseries": [1, 2],
                    },
                }
            )
        ]
    )


@given("a raw search result containing a rich entry with unknown subtype")
def given_rich_unknown(ctx):
    ctx["raw"] = _result(
        [json.dumps({"type": "rich", "subtype": "sports", "url": "https://x.example"})]
    )


@given("a raw search result with meta_url favicon and extra snippets")
def given_meta_favicon(ctx):
    ctx["raw"] = _result(
        [
            json.dumps(
                {
                    **_source(title="Meta"),
                    "meta_url": {"favicon": "https://m.example/f.ico"},
                    "extra_snippets": ["s1", "s2"],
                }
            )
        ]
    )


@given("a raw search result with scalar extra snippets and a description")
def given_bad_extra(ctx):
    ctx["raw"] = _result(
        [
            json.dumps(
                {
                    "title": "T",
                    "url": "https://t.example",
                    "description": "the description",
                    "extra_snippets": "oops",
                }
            )
        ]
    )


@given("a raw search result with four results")
def given_four(ctx):
    sources = []
    for i in range(4):
        sources.append(_source(url=f"https://{i}.example", title=f"T{i}"))
    ctx["raw"] = _result([json.dumps(sources)])


@given("a raw search result with top level queries")
def given_top_queries(ctx):
    ctx["raw"] = _result([json.dumps([_source(title="A")])], queries=["q1", "q2"])
    ctx["expected_queries"] = ["q1", "q2"]


@given("a raw search result with structured content queries")
def given_structured_queries(ctx):
    ctx["raw"] = _result(
        [json.dumps([_source(title="A")])],
        structuredContent={"queries": ["s1"]},
    )
    ctx["expected_queries"] = ["s1"]


@when(parsers.parse("the result is formatted for tool {tool_name}"))
def when_format(ctx, tool_name):
    ctx["formatted"] = ctx["handler"].format_result(tool_name, ctx["raw"])


@then("zero sources are produced")
def then_zero_sources(ctx):
    assert ctx["formatted"]["sources"] == []


@then("two sources are produced")
def then_two_sources(ctx):
    assert len(ctx["formatted"]["sources"]) == 2


@then("one source is produced")
def then_one_source(ctx):
    assert len(ctx["formatted"]["sources"]) == 1


@then("the content contains search_result markup")
def then_markup(ctx):
    assert "<search_result citation_number=1>" in ctx["formatted"]["content"]
    assert "</search_result>" in ctx["formatted"]["content"]


@then("the search results text is joined")
def then_joined(ctx):
    assert "search_result" in ctx["formatted"]["search_results"]


@then("no queries are attached")
def then_no_queries(ctx):
    assert "queries" not in ctx["formatted"]


@then("the queries are attached")
def then_queries(ctx):
    assert ctx["formatted"]["queries"] == ctx["expected_queries"]


@then("the fallback summary is returned")
def then_fallback_summary(ctx):
    assert ctx["formatted"]["content"] == "Found 0 search results"
    assert ctx["formatted"]["search_results"] == "Found 0 search results"


@then("rich results are attached")
def then_rich(ctx):
    assert len(ctx["formatted"]["rich_results"]) == 1


@then("rich results are still recorded")
def then_rich_recorded(ctx):
    # Unusable entries are recorded but never become a source — the distinct
    # fact vs then_rich is the empty sources list, so assert both.
    assert ctx["formatted"]["sources"] == []
    then_rich(ctx)


@then("the weather source has the provider title and url")
def then_weather_source(ctx):
    src = ctx["formatted"]["sources"][0]
    assert src["title"] == "Weather JSON Data from Weather Provider"
    assert src["url"] == "https://w.example"
    assert "weather data" in src["page_content"]


@then("the source carries meta_url favicon extra snippets and snippet")
def then_meta_fields(ctx):
    src = ctx["formatted"]["sources"][0]
    assert src["favicon"] == "https://m.example/f.ico"
    assert src["extra_snippets"] == ["s1", "s2"]
    # The raw snippet is embedded in the formatted content after the url.
    assert "a snippet" in ctx["formatted"]["search_results"]


@then("the source carries description as snippet and no extra snippets")
def then_description_fields(ctx):
    src = ctx["formatted"]["sources"][0]
    assert src["extra_snippets"] is None
    assert "the description" in ctx["formatted"]["search_results"]


@then("the summary names three titles and mentions 1 more")
def then_summary(ctx):
    content = ctx["formatted"]["content"]
    assert "Found 4 search results: T0, T1, T2" in content
    assert "and 1 more" in content


@given(parsers.parse("a tool start for {tool} with query {query}"))
def given_tool_start(ctx, tool, query):
    ctx["tool_start"] = (tool, {} if query == "none" else {"query": query})


@when("the tool start message is requested")
def when_start_message(ctx):
    tool, args = ctx["tool_start"]
    ctx["start_message"] = ctx["handler"].get_tool_start_message(tool, args)


@then(parsers.parse('the start message is "{message}"'))
def then_start_message(ctx, message):
    assert ctx["start_message"] == message


@when("tool guidance is requested")
def when_guidance(ctx):
    ctx["guidance"] = ctx["handler"].get_tool_guidance()


@then("guidance covers brave_web_search and brave_news_search")
def then_guidance(ctx):
    assert "brave_web_search" in ctx["guidance"]
    assert "brave_news_search" in ctx["guidance"]


@then("there are no augmented tools")
def then_no_augmented(ctx):
    assert ctx["handler"].get_augmented_tools() == []


@given("a formatted result with two sources and queries")
def given_fmt_sources(ctx):
    ctx["formatted"] = {
        "sources": [
            _source("https://a.example", "A"),
            _source("https://b.example", "B"),
        ],
        "content": "Found 2 search results",
        "queries": ["q1", "q2"],
    }


@when("the tool message content is requested")
def when_tmc_sources(ctx):
    ctx["tmc"] = ctx["handler"].get_tool_message_content(ctx["formatted"], None)


@then("the content is a web sources part list")
def then_web_sources_list(ctx):
    assert isinstance(ctx["tmc"], list) and len(ctx["tmc"]) == 1
    part = ctx["tmc"][0]
    assert part.type == "brave-chat.webSources"
    assert part.query == ["q1", "q2"]
    assert len(part.sources) == 2


@given("a formatted result with only rich results")
def given_fmt_rich_only(ctx):
    ctx["formatted"] = {
        "sources": [],
        "rich_results": [{"url": "https://w.example"}],
        "content": "Found rich data",
    }


@then("the content is a text part summary")
def then_text_summary(ctx):
    assert isinstance(ctx["tmc"], list) and len(ctx["tmc"]) == 1
    part = ctx["tmc"][0]
    assert isinstance(part, TextContentPart)
    assert part.text == "Found rich data"


@given('a formatted result with plain content "search finished"')
def given_fmt_plain(ctx):
    ctx["formatted"] = {"sources": [], "rich_results": [], "content": "search finished"}


@then(parsers.parse('the tool message content is "{content}"'))
def then_tmc_value(ctx, content):
    assert ctx["tmc"] == content


@given("a formatted result with no sources and blank content")
def given_fmt_blank(ctx):
    ctx["formatted"] = {"sources": [], "rich_results": [], "content": "  "}


@given("a formatted result with two sources and a query list")
def given_fmt_out_sources(ctx):
    ctx["formatted"] = {
        "sources": [
            _source("https://a.example", "A"),
            _source("https://b.example", "B"),
        ],
        "queries": ["only q"],
    }


@when("the output content parts are built")
def when_output_parts(ctx):
    ctx["parts"] = ctx["handler"].get_output_content_parts(ctx["formatted"], None)


@then("one web sources output part is produced")
def then_one_output_part(ctx):
    assert len(ctx["parts"]) == 1
    assert ctx["parts"][0]["type"] == "brave-chat.webSources"


@then("the output part carries the query list")
def then_output_query(ctx):
    assert ctx["parts"][0]["query"] == ["only q"]


@given("a formatted result with no sources")
def given_fmt_no_sources(ctx):
    ctx["formatted"] = {"sources": [], "rich_results": []}


@then("the output parts are empty")
def then_output_empty(ctx):
    assert ctx["parts"] == []


# --- format_web_sources_content ---

WEB_SOURCE_PAYLOADS = {
    "basic": [{"title": "Example", "url": "https://e.example", "snippet": "snip"}],
    "with_page_content": [
        {
            "title": "Example",
            "url": "https://e.example",
            "snippet": "s",
            "page_content": "Full page content here",
        }
    ],
    "with_extra_snippets": [
        {
            "title": "Example",
            "url": "https://e.example",
            "snippet": "s",
            "extra_snippets": ["extra one", "extra two"],
        }
    ],
    "empty": [],
}

EXPECTATION_CHECKS = {
    "has one citation block": lambda t: "<search_result citation_number=1>" in t
    and SearchResults.CITATIONS in t,
    "embeds full page content": lambda t: "Full page content:" in t
    and "Full page content here" in t,
    "embeds additional snippets": lambda t: "Additional snippets:" in t
    and "- extra one" in t,
    "reports zero results": lambda t: t == "Found 0 search results",
}


@given(parsers.parse("web sources payload kind {kind}"))
def given_ws_payload(ctx, kind):
    ctx["ws_kind"] = kind


@when("web sources content is formatted")
def when_ws_format(ctx):
    ctx["ws_text"] = ctx["handler"].format_web_sources_content(
        WEB_SOURCE_PAYLOADS[ctx["ws_kind"]]
    )


@then(parsers.parse("the formatted web sources text {expectation}"))
def then_ws_text(ctx, expectation):
    assert EXPECTATION_CHECKS[expectation](ctx["ws_text"])


@then(parsers.parse("the server name is {name}"))
def then_server_name(ctx, name):
    assert ctx["handler"].server_name == name
