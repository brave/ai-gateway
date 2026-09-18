import pytest

from aichat.services.security_utils import sanitize_untrusted_content

WRAPPER_TAGS = [
    "page",
    "excerpt",
    "transcript",
    "results",
    "user_memory",
    "tool_output",
    "tabs",
]


@pytest.mark.parametrize("tag", WRAPPER_TAGS)
def test_replaces_opening_and_closing_wrapper_tags(tag):
    assert sanitize_untrusted_content(f"before <{tag}> after") == (
        "before <fake_tag> after"
    )
    assert sanitize_untrusted_content(f"before </{tag}> after") == (
        "before <fake_tag> after"
    )


def test_is_case_insensitive():
    assert sanitize_untrusted_content("<PAGE>") == "<fake_tag>"
    assert sanitize_untrusted_content("</Page>") == "<fake_tag>"


@pytest.mark.parametrize(
    "text",
    ["< page>", "</ page>", "</  page  >", "< /page>", "<  /  page  >"],
)
def test_ignores_whitespace_around_the_tag_name(text):
    assert sanitize_untrusted_content(text) == "<fake_tag>"


@pytest.mark.parametrize(
    "text",
    ['<page id="1">', "<page foo=bar baz>", '</page  data-x="y">'],
)
def test_replaces_tags_carrying_attributes(text):
    # A page closing its own wrapper as `</page foo>` would otherwise leave
    # everything after it reading as prompt rather than as data.
    assert sanitize_untrusted_content(text) == "<fake_tag>"


@pytest.mark.parametrize(
    "text",
    ["<pages>", "<tool_outputs>", "<div>", "<b>bold</b>", "<pagelike>"],
)
def test_leaves_other_tags_alone(text):
    assert sanitize_untrusted_content(text) == text


def test_leaves_plain_comparisons_alone():
    text = "1 < 2 and 3 > 2, page > excerpt"
    assert sanitize_untrusted_content(text) == text


def test_replaces_every_occurrence():
    assert sanitize_untrusted_content("<page>a</page><excerpt>b</excerpt>") == (
        "<fake_tag>a<fake_tag><fake_tag>b<fake_tag>"
    )
